"""Counterfactual action labeler for the cost-aware value controller.

Leakage boundary
----------------
`label_state` is a **training-only** path and may read the full `source`
sentence: it needs the oracle translation to decide how many committed words
are correct. No online feature function may read `source`, the reference, the
oracle continuation, or any future source word; `features.FeatureBuilder.build`
sees only the observed source prefix, the committed target prefix, history,
runtime costs, and optional confidence values.

Label construction mirrors `week/week_run.py::label_step` while sharing the
pool across actions:

1. Sample one nested pool of ``max(branches) x max(horizon)`` token rows with a
   single seed, so B=2 is exactly the first two rows of B=4. Unlike
   ``week_run.py::label_step`` this path deliberately does **not** call
   ``shuffle_rows``: the brief mandates first-b nested candidates, and a shuffle
   would make the branch subsets depend on the maximum pool size.
2. For every distinct horizon, take the first ``max(branches)`` rows for that
   horizon, truncate each row to the horizon, decode it, and translate the
   resulting requests in **one** batch with the committed prefix forced.
3. For every branch budget at that horizon, run RALCP over the first ``b``
   translated outputs and compare against the oracle words.

Translation is therefore done once per unique horizon, never once per
``(b, h)`` pair. `pred_s` is the one shared candidate-generation time; `mt_s`
is the measured translation time for the horizon. Branch variants at the same
horizon share that timing, which is the cost-table approximation: the table is
keyed by ``(branches, horizon)`` and cannot separate b variants that were
produced by a single max-b batch.

`committed` inputs are never mutated; every committed prefix in an outcome is a
fresh tuple.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .action import READ, Action
from .config import ControllerConfig
from .runtime import ControlledRuntime, run_pfba, vote_statistics

__all__ = ["ActionOutcome", "CounterfactualLabeler"]


class _NullTimer:
    """Fallback when `run_pfba` is unavailable in a stripped checkout."""

    def run(self, name, function):
        return function(), 0.0


@dataclass(frozen=True)
class ActionOutcome:
    """Everything one labelled action produced at one decision point.

    `new_words` are the words RALCP selected and committed. `matched` is the
    common-prefix length against the oracle translation and `incorrect` counts
    the selected words after the first mismatch. `emitted` is ``len(new_words)``
    and `no_emit` is true when RALCP selected nothing. `candidates` are the
    **translated** strings RALCP voted over, matching ``StepResult.candidates``.

    READ outcomes are produced by `read_outcome` and carry zero compute, no
    words, and the unchanged committed prefix.
    """

    action: Action
    new_words: tuple[str, ...]
    matched: int
    incorrect: int
    emitted: int
    no_emit: bool
    next_committed: tuple[str, ...]
    pred_s: float
    mt_s: float
    vote_top1: float
    vote_margin: float
    vote_entropy: float
    candidates: tuple[str, ...] = ()
    # Pipeline marginal wall time for this action as priced by the cost model at
    # its decision state. ``None`` means no cost model was supplied; the
    # training clock then falls back to the serial ``pred_s + mt_s`` compute.
    marginal_wall_s: float | None = None


def _default_timer(timer):
    if timer is not None:
        return timer
    if run_pfba is not None and hasattr(run_pfba, "NullTimer"):
        return run_pfba.NullTimer()
    return _NullTimer()


class CounterfactualLabeler:
    """Builds counterfactual `(state, action)` outcomes for training.

    `predictor` must expose one of:

    * ``sample_token_rows(observed, max_b, max_h, seed) -> list[list[int]]``
    * ``model`` and ``tok`` (a Hugging Face causal LM plus tokenizer), or
    * ``sample(observed, max_b, max_h, seed) -> list[str]`` plus ``tok``.

    The last two paths need a tokenizer as well; without one the labeler raises
    `RuntimeError` rather than returning an empty pool that would silently mark
    every write as a no-emit.

    `candidate_mode` controls how candidates are drawn:

    * ``"per_candidate"`` (default) calls ``predictor.sample(observed, 1,
      horizon, seed + i)`` once per candidate, exactly like
      `ControlledRuntime.predict`. The labeled action's return distribution is
      then the distribution the runtime actually executes.
    * ``"batched"`` is the legacy ablation: one ``num_return_sequences=max_b``
      call (or one ``sample_token_rows``) for the whole pool. It is cheaper but
      can draw a different distribution than the runtime.

    When `per_candidate` is requested but the predictor has no ``sample``
    method, the labeler falls back to the batched paths and records the actual
    mode it used.
    """

    CANDIDATE_MODES = ("per_candidate", "batched")

    def __init__(
        self,
        predictor,
        translator,
        cfg: ControllerConfig,
        timer=None,
        candidate_mode: str = "per_candidate",
    ) -> None:
        mode = str(candidate_mode)
        if mode not in self.CANDIDATE_MODES:
            raise ValueError(
                f"candidate_mode must be one of {self.CANDIDATE_MODES}, got "
                f"{mode!r}"
            )
        if mode == "per_candidate" and not callable(
            getattr(predictor, "sample", None)
        ):
            # No per-candidate sampler: use the batched path, but warn instead
            # of silently downgrading and claiming parity the predictor cannot
            # give. The recorded `candidate_mode` reflects what actually ran.
            import warnings

            warnings.warn(
                "candidate_mode='per_candidate' was requested but the "
                "predictor has no `sample` method; downgrading to the legacy "
                "'batched' path, which draws from a different sampling stream",
                RuntimeWarning,
                stacklevel=2,
            )
            mode = "batched"
        self.candidate_mode = mode
        self.predictor = predictor
        self.translator = translator
        self.cfg = cfg if cfg is not None else ControllerConfig()
        self.timer = _default_timer(timer)
        # Reuse the controlled runtime's exact RALCP gate, request-text helper,
        # and translation call so labels and rollouts cannot drift apart.
        self.runtime = ControlledRuntime(
            predictor, translator, self.cfg, timer=self.timer
        )

    # ------------------------------------------------------------------
    # Candidate sampling
    # ------------------------------------------------------------------
    def sample_candidates(
        self, observed: str, max_b: int, max_h: int, seed: int
    ) -> list[list[int]]:
        """Sample one nested pool of token rows, ``max_b`` by ``max_h``.

        Precedence is the predictor's ``sample_token_rows``, then an explicit
        ``model`` + ``tok`` pair via ``model.generate`` (seeded exactly like
        ``week/week_run.py``), then ``sample`` tokenized through ``tok``. The
        returned rows are token id lists; B=2 is the first two rows of B=4.
        """
        max_b = int(max_b)
        max_h = int(max_h)
        if max_b <= 0 or max_h <= 0:
            raise ValueError("sample_candidates needs positive max_b and max_h")

        sampler = getattr(self.predictor, "sample_token_rows", None)
        if callable(sampler):
            rows = sampler(observed, max_b, max_h, seed) or []
            return [self._normalize_row(row) for row in rows]

        model = getattr(self.predictor, "model", None)
        tok = getattr(self.predictor, "tok", None)
        if model is not None and tok is not None:
            return self._generate_token_rows(observed, max_b, max_h, seed)

        sample = getattr(self.predictor, "sample", None)
        if callable(sample) and tok is not None:
            continuations = sample(observed, max_b, max_h, seed) or []
            return [self._encode_text(continuation) for continuation in continuations]

        raise RuntimeError(
            "CounterfactualLabeler cannot sample candidates: the predictor must "
            "provide sample_token_rows(observed, max_b, max_h, seed), a "
            "model+tok pair, or sample(...)+tok. No tokenizer is available, so "
            "returning an empty pool would silently label every write as a "
            "no-emit."
        )

    def sample_text_candidates(
        self, observed: str, count: int, horizon: int, seed: int
    ) -> tuple[list[str], float] | None:
        """Sample `count` continuations one seed at a time, like the runtime.

        Candidate ``i`` uses ``predictor.sample(observed, 1, horizon,
        seed + i)``, which is the exact call `ControlledRuntime.predict` makes.
        Returns ``(continuations, pred_s)`` where `pred_s` is the summed stage
        time, or ``None`` when the predictor has no ``sample`` method.
        """
        sample = getattr(self.predictor, "sample", None)
        if not callable(sample):
            return None
        continuations: list[str] = []
        seconds = 0.0
        for index in range(int(count)):
            batch, elapsed = self.runtime._timed(
                "pred",
                lambda sample_seed=int(seed) + index: sample(
                    observed, 1, int(horizon), sample_seed
                ),
            )
            continuations.extend(str(text) for text in (batch or []))
            seconds += float(elapsed)
        return continuations, seconds

    def _generate_token_rows(
        self, observed: str, max_b: int, max_h: int, seed: int
    ) -> list[list[int]]:
        """``model.generate`` path, seeded exactly like ``week_run.py``."""
        import torch

        tok = self.predictor.tok
        model = self.predictor.model
        device = getattr(self.predictor, "device", "cpu")
        inputs = tok(
            observed, return_tensors="pt", add_special_tokens=False
        ).to(device)
        torch_device = torch.device(device)
        devices = [torch_device.index or 0] if torch_device.type == "cuda" else []
        with torch.inference_mode(), torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(int(seed))
            if devices:
                with torch.cuda.device(devices[0]):
                    torch.cuda.manual_seed(int(seed))
            output = model.generate(
                **inputs,
                max_new_tokens=int(max_h),
                do_sample=True,
                temperature=1.0,
                top_p=0.9,
                top_k=0,
                num_return_sequences=int(max_b),
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.pad_token_id,
            )
        width = inputs["input_ids"].shape[1]
        return [self._normalize_row(row) for row in output[:, width:]]

    @staticmethod
    def _normalize_row(row) -> list[int]:
        if hasattr(row, "tolist"):
            row = row.tolist()
        return [int(token) for token in row]

    def _encode_text(self, text: str) -> list[int]:
        """Tokenize one continuation string through ``predictor.tok``."""
        tok = getattr(self.predictor, "tok", None)
        if tok is None:
            raise RuntimeError(
                "CounterfactualLabeler needs predictor.tok to tokenize sampled "
                "continuations"
            )
        return self._extract_ids(tok(text, add_special_tokens=False))

    @staticmethod
    def _extract_ids(encoded) -> list[int]:
        ids = getattr(encoded, "input_ids", None)
        if ids is None and isinstance(encoded, dict):
            ids = encoded["input_ids"]
        if ids is None:
            ids = encoded
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if ids and isinstance(ids[0], (list, tuple)):
            ids = ids[0]
        return [int(token) for token in ids]

    # ------------------------------------------------------------------
    # Oracle
    # ------------------------------------------------------------------
    def oracle_words(
        self, source: str, committed: Sequence[str]
    ) -> tuple[str, ...]:
        """Translate the full `source` with the committed prefix forced.

        Training-only: this is the label side of the leakage boundary. The
        translation is tokenized by ``text.split()`` so a trailing space cannot
        create a phantom word.

        The oracle is timed under its own ``"oracle"`` stage. It must never be
        billed to the ``"mt"`` bucket that action labels read: action ``mt_s``
        comes from the specific horizon translation call, and keeping the
        oracle on a distinct stage makes that separation explicit in
        `timer.seconds` as well.
        """
        prefix = " ".join(tuple(committed))
        outputs, _seconds = self.runtime._timed(
            "oracle", lambda: self._translate_raw([source], prefix)
        )
        text = outputs[0] if outputs else ""
        return tuple(text.split())

    # ------------------------------------------------------------------
    # Labeling
    # ------------------------------------------------------------------
    def read_outcome(self, committed: Sequence[str]) -> ActionOutcome:
        """READ: no prediction, no translation, no emitted words."""
        return ActionOutcome(
            action=READ,
            new_words=(),
            matched=0,
            incorrect=0,
            emitted=0,
            no_emit=True,
            next_committed=tuple(committed),
            pred_s=0.0,
            mt_s=0.0,
            vote_top1=0.0,
            vote_margin=0.0,
            vote_entropy=0.0,
            candidates=(),
        )

    def label_state(
        self,
        observed: str,
        committed: Sequence[str],
        source: str,
        actions: Sequence[Action],
        seed: int,
    ) -> list[ActionOutcome]:
        """Label every action at one state, returning outcomes in input order.

        Only write actions sample and translate. A READ in `actions` is filled
        in with `read_outcome`, which performs no compute. `committed` is never
        mutated.
        """
        committed_tuple = tuple(committed)
        action_list = tuple(actions)
        write_actions = [action for action in action_list if not action.is_read]

        outcomes: dict[Action, ActionOutcome] = {}
        if write_actions:
            # One translation per unique horizon, using that horizon's maximum
            # branch budget. Smaller-b actions reuse the prefix of the output
            # pool, so they never trigger a second translate call.
            by_horizon: dict[int, list[Action]] = {}
            for action in write_actions:
                by_horizon.setdefault(int(action.horizon), []).append(action)

            prefix = " ".join(committed_tuple)
            per_candidate = self.candidate_mode == "per_candidate"
            oracle: tuple[str, ...] | None = None
            batched_rows: list[list[int]] = []
            batched_pred_s = 0.0
            if not per_candidate:
                max_b = max(int(action.branches) for action in write_actions)
                max_h = max(int(action.horizon) for action in write_actions)
                batched_rows, batched_pred_s = self.runtime._timed(
                    "pred",
                    lambda: self.sample_candidates(observed, max_b, max_h, seed),
                )
                if len(batched_rows) < max_b:
                    raise RuntimeError(
                        f"sample_candidates returned {len(batched_rows)} rows "
                        f"but the largest action needs {max_b}; a (b, h) label "
                        "must be voted over exactly b candidates, not a "
                        "shortened pool"
                    )

            for horizon in sorted(by_horizon):
                group = by_horizon[horizon]
                need = max(int(action.branches) for action in group)
                if per_candidate:
                    sampled = self.sample_text_candidates(
                        observed, need, horizon, seed
                    )
                    if sampled is None:  # pragma: no cover - constructor guards
                        raise RuntimeError(
                            "per_candidate mode needs predictor.sample"
                        )
                    continuations, pred_s = sampled
                    if len(continuations) < need:
                        raise RuntimeError(
                            f"predictor.sample returned {len(continuations)} "
                            f"continuations but the largest action at horizon "
                            f"{horizon} needs {need}; a (b, h) label must be "
                            "voted over exactly b candidates, not a shortened "
                            "pool"
                        )
                    requests = [
                        self.runtime._request_text(observed, text)
                        for text in continuations[:need]
                    ]
                else:
                    pred_s = batched_pred_s
                    requests = [
                        self.runtime._request_text(
                            observed, self._decode_tokens(row, horizon)
                        )
                        for row in batched_rows[:need]
                    ]

                if oracle is None:
                    # Timed after the first candidate pool is drawn, before the
                    # first candidate translation, so the stage order stays
                    # sample -> oracle -> translate.
                    oracle = self.oracle_words(source, committed_tuple)
                outputs, mt_s = self.runtime.translate(requests, prefix)
                outputs = list(outputs)
                if len(outputs) < need:
                    raise RuntimeError(
                        f"translator returned {len(outputs)} outputs for "
                        f"{len(requests)} requests at horizon {horizon}; a "
                        f"(b, h) label needs {need} translated candidates, not "
                        "a shortened pool"
                    )

                for action in group:
                    pool = tuple(outputs[: int(action.branches)])
                    words = self._selected_words(pool)
                    matched = self._common_prefix(words, oracle)
                    emitted = len(words)
                    top1, margin, entropy = vote_statistics(
                        pool, self.cfg.ralcp_threshold
                    )
                    outcomes[action] = ActionOutcome(
                        action=action,
                        new_words=words,
                        matched=matched,
                        incorrect=emitted - matched,
                        emitted=emitted,
                        no_emit=emitted == 0,
                        next_committed=committed_tuple + words,
                        # pred_s is this horizon's generation measurement;
                        # mt_s is the max-b translation time for this horizon,
                        # shared by every b variant (cost-table approximation).
                        pred_s=float(pred_s),
                        mt_s=float(mt_s),
                        vote_top1=float(top1),
                        vote_margin=float(margin),
                        vote_entropy=float(entropy),
                        candidates=pool,
                    )

        return [
            self.read_outcome(committed_tuple) if action.is_read else outcomes[action]
            for action in action_list
        ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _translate_raw(
        self, requests: Sequence[str], prefix: str
    ) -> list[str]:
        """Translate without a stage timer; callers time the stage they own."""
        sources = list(requests)
        if run_pfba is not None and hasattr(run_pfba, "call_translate_batch"):
            return list(run_pfba.call_translate_batch(self.translator, sources, prefix))
        return list(self.translator.translate_batch(sources, prefix))

    def _decode_tokens(self, token_ids: Sequence[int], horizon: int) -> str:
        """Truncate one token row to `horizon` tokens and decode it."""
        tok = getattr(self.predictor, "tok", None)
        if tok is None:
            raise RuntimeError(
                "CounterfactualLabeler needs predictor.tok to decode candidate "
                "token rows"
            )
        eos_id = getattr(tok, "eos_token_id", None)
        kept: list[int] = []
        for token in token_ids:
            token = int(token)
            if eos_id is not None and token == int(eos_id):
                break
            kept.append(token)
            if len(kept) >= horizon:
                break
        text = tok.decode(
            kept,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return text.split("\n")[0].strip()

    def _selected_words(self, pool: Sequence[str]) -> tuple[str, ...]:
        selected = self.runtime.ralcp(pool)
        return tuple(selected.split()) if selected else ()

    @staticmethod
    def _common_prefix(words: Sequence[str], oracle: Sequence[str]) -> int:
        matched = 0
        for left, right in zip(words, oracle):
            if left != right:
                break
            matched += 1
        return matched
