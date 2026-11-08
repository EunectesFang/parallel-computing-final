"""Streaming policy state machines: TAF, RALCP, RALCP+TAF, Wait-K.

All policies share three hard invariants:
1. the committed target prefix is forced into every MT decode;
2. committed tokens are never rewritten or deleted (``extend_only``);
3. a final flush finishes the hypothesis after the source is exhausted.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from .config import StreamingConfig
from .instrumentation import Stopwatch, summarize_timings
from .majority import exact_majority, extend_only
from .ralcp import author_ralcp


@dataclass
class StepTrace:
    step: int
    read_pos: int
    read_text: str
    n_predictions: int
    n_candidates: int
    pi: float | None
    support: int
    wrote: bool
    committed_before: int
    committed_after: int
    delay: int
    timings_ms: dict[str, float] = field(default_factory=dict)
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "read_pos": self.read_pos,
            "read_text": self.read_text,
            "n_predictions": self.n_predictions,
            "n_candidates": self.n_candidates,
            "pi": self.pi,
            "support": self.support,
            "wrote": self.wrote,
            "committed_before": self.committed_before,
            "committed_after": self.committed_after,
            "delay": self.delay,
            "timings_ms": self.timings_ms,
            "detail": self.detail,
        }


@dataclass
class PolicyResult:
    committed_ids: list[int]
    delays: list[int]
    traces: list[StepTrace]
    text: str
    timing: dict[str, float]
    metadata: dict[str, Any]

    def to_dict(self, include_tokens: bool = True) -> dict:
        out = {
            "text": self.text,
            "n_target_tokens": len(self.committed_ids),
            "delays": list(self.delays),
            "timing": self.timing,
            "metadata": self.metadata,
            "traces": [t.to_dict() for t in self.traces],
        }
        if include_tokens:
            out["committed_ids"] = list(self.committed_ids)
        return out


def render_units(units: Sequence[str], unit: str) -> str:
    return " ".join(units) if unit == "word" else "".join(units)


def _join_prediction(prefix: str, prediction: str, unit: str) -> str:
    if not prediction:
        return prefix
    return f"{prefix} {prediction}" if unit == "word" else f"{prefix}{prediction}"


def _author_ralcp_continuation(mt, continuations, threshold: float, unit: str):
    """Vote on decoded word (or character) prefixes, exactly as the authors.

    ``continuations`` are token-id continuations generated with the committed
    prefix forced.  They are decoded to text, voted on, and the winning text
    is encoded back to target ids for appending.
    """
    texts = [mt.decode(ids) for ids in continuations]
    selection = author_ralcp(texts, threshold, char=(unit == "char"))
    text = (
        "".join(selection.prefix)
        if unit == "char"
        else " ".join(selection.prefix)
    )
    ids = list(mt.encode(text)) if text else []
    return selection, text, ids


class _BasePolicy:
    def __init__(self, mt, cfg: StreamingConfig) -> None:
        self.mt = mt
        self.cfg = cfg

    def _decode_text(self, ids: Sequence[int]) -> str:
        return self.mt.decode(ids)

    def _finish(
        self,
        committed: list[int],
        delays: list[int],
        last_candidates: list[tuple[int, ...]],
        source_units: list[str],
        full_source_text: str,
    ) -> tuple[int, float]:
        """Final flush.  Always attempts to finish, never fabricates tokens."""
        started = time.perf_counter()
        if not source_units:
            return 0, 0.0

        proposal: tuple[int, ...] | None = None
        if last_candidates:
            # Longest candidate is the best completion available.  All
            # candidates already contain the committed prefix.
            proposal = max(last_candidates, key=len)
        else:
            conts = self.mt.translate(
                full_source_text,
                committed,
                num_sequences=1,
                num_beams=1,
                max_new_tokens=self.cfg.max_new_tokens,
            )
            if conts:
                proposal = tuple(committed) + tuple(conts[0])

        added = 0
        if proposal is not None:
            proposal = extend_only(committed, proposal)
            new = list(proposal[len(committed):])
            if new:
                committed.extend(new)
                delays.extend([len(source_units)] * len(new))
                added = len(new)
        return added, (time.perf_counter() - started) * 1000.0

    def _result(
        self,
        committed: list[int],
        delays: list[int],
        traces: list[StepTrace],
        flush_ms: float,
        flush_added: int,
        wall_ms: float,
        metadata: dict[str, Any],
    ) -> PolicyResult:
        timing = summarize_timings(traces, {"flush_ms": flush_ms})
        timing["wall_ms"] = round(wall_ms, 6)
        timing["flush_added_tokens"] = flush_added
        metadata = dict(metadata)
        metadata.update(
            {
                "flush_added_tokens": flush_added,
                "n_steps": len(traces),
                "n_target_tokens": len(committed),
            }
        )
        return PolicyResult(
            committed_ids=list(committed),
            delays=list(delays),
            traces=traces,
            text=self._decode_text(committed),
            timing=timing,
            metadata=metadata,
        )


class TAFPolicy(_BasePolicy):
    """Paper Eq. 4-7 exact-majority interpretation (``--policy taf``).

    The authors' released agent is ``ralcp_taf`` and is implemented by
    :class:`AuthorRALCPTAFPolicy`; this class is retained as the paper-formula
    comparison point.
    """

    def __init__(self, llm, mt, cfg: StreamingConfig) -> None:
        super().__init__(mt, cfg)
        self.llm = llm

    def run(self, source_units: Sequence[str]) -> PolicyResult:
        cfg = self.cfg
        units = list(source_units)
        total_source = len(units)
        committed: list[int] = []
        delays: list[int] = []
        traces: list[StepTrace] = []
        last_hypotheses: list[tuple[int, ...]] = []
        last_candidates: list[tuple[int, ...]] = []
        wall_start = time.perf_counter()

        if total_source == 0:
            return self._result(committed, delays, traces, 0.0, 0, 0.0, self._metadata())

        read_pos = min(max(1, cfg.initial_read), total_source)
        step = 0
        while read_pos <= total_source and step < cfg.max_steps:
            step += 1
            partial = render_units(units[:read_pos], cfg.src_unit)
            step_sw = Stopwatch(cfg.cuda_sync)
            step_start = time.perf_counter()

            with step_sw.stage("llm_sample"):
                predictions = self._sample(partial)

            hypotheses: list[tuple[int, ...]] = []
            with step_sw.stage("mt_decode"):
                for pred in predictions:
                    src_text = _join_prediction(partial, pred, cfg.src_unit)
                    conts = self.mt.translate(
                        src_text,
                        committed,
                        num_sequences=1,
                        num_beams=1,
                        max_new_tokens=cfg.max_new_tokens,
                    )
                    if not conts:
                        continue
                    full = tuple(committed) + tuple(conts[0])
                    hypotheses.append(extend_only(committed, full))

            with step_sw.stage("aggregate"):
                majority = exact_majority(hypotheses, committed)
                proposal = majority.hypothesis
                pi: float | None = majority.pi
                support = majority.count
                wrote = majority.pi + 1e-12 >= cfg.tau and len(proposal) > len(committed)

            total_ms = (time.perf_counter() - step_start) * 1000.0
            timings = step_sw.as_dict()
            timings["total"] = round(total_ms, 6)

            before = len(committed)
            added = 0
            if wrote:
                proposal = extend_only(committed, proposal)
                new = list(proposal[len(committed):])
                if new:
                    committed.extend(new)
                    delays.extend([read_pos] * len(new))
                    added = len(new)

            traces.append(
                StepTrace(
                    step=step,
                    read_pos=read_pos,
                    read_text=partial,
                    n_predictions=len(predictions),
                    n_candidates=len(hypotheses),
                    pi=pi,
                    support=support,
                    wrote=wrote,
                    committed_before=before,
                    committed_after=len(committed),
                    delay=read_pos,
                    timings_ms=timings,
                    detail={"added_tokens": added, "tau": cfg.tau},
                )
            )
            last_hypotheses = hypotheses
            last_candidates = hypotheses

            if read_pos >= total_source:
                break
            read_pos = min(read_pos + 1, total_source)

        # Final flush uses the last step's candidates (or a fresh full decode).
        flush_added = 0
        flush_ms = 0.0
        if last_hypotheses:
            majority = exact_majority(last_hypotheses, committed)
            flush_proposal = majority.hypothesis
            if len(flush_proposal) > len(committed):
                flush_proposal = extend_only(committed, flush_proposal)
                new = list(flush_proposal[len(committed):])
                started = time.perf_counter()
                committed.extend(new)
                delays.extend([total_source] * len(new))
                flush_added += len(new)
                flush_ms += (time.perf_counter() - started) * 1000.0
        if flush_added == 0:
            added, ms = self._finish(
                committed, delays, last_candidates, units, render_units(units, cfg.src_unit)
            )
            flush_added += added
            flush_ms += ms

        wall_ms = (time.perf_counter() - wall_start) * 1000.0
        return self._result(
            committed, delays, traces, flush_ms, flush_added, wall_ms, self._metadata()
        )

    def _sample(self, partial: str) -> list[str]:
        # Kept separate so tests can wrap/monkeypatch sampling if needed.
        return self.llm.sample(
            partial,
            self.cfg.n,
            self.cfg.horizon,
            self.cfg.top_k,
            seed=self.cfg.seed,
        )

    def _metadata(self) -> dict[str, Any]:
        return {
            "policy": self.cfg.policy,
            "n": self.cfg.n,
            "horizon": self.cfg.horizon,
            "top_k": self.cfg.top_k,
            "tau": self.cfg.tau,
            "beam_width": self.cfg.beam_width,
            "effective_beam_width": self.cfg.effective_beam_width,
            "candidate_budget": self.cfg.candidate_budget,
            "ralcp_fraction": self.cfg.ralcp_fraction,
            "ralcp_content": self.cfg.ralcp_content,
            "forced_committed_prefix": True,
            "final_flush": True,
        }


class AuthorRALCPTAFPolicy(_BasePolicy):
    """Fidelity port of the authors' ``agents/ralcp_taf.py`` protocol.

    At every read position (including the full source) the LLM samples ``n``
    source continuations.  Each predicted source is translated with the
    committed target prefix forced and a per-call beam width, so the candidate
    pool is ``n * beam``.  The emitted content is the authors' word-prefix
    RALCP vote (no exact-majority gate).  After the source stream ends, a
    fresh full-source translation is generated with the committed prefix and
    finishes the hypothesis.
    """

    def __init__(self, llm, mt, cfg: StreamingConfig) -> None:
        super().__init__(mt, cfg)
        self.llm = llm

    def _sample(self, partial: str) -> list[str]:
        return self.llm.sample(
            partial,
            self.cfg.n,
            self.cfg.horizon,
            self.cfg.top_k,
            seed=self.cfg.seed,
        )

    def run(self, source_units: Sequence[str]) -> PolicyResult:
        cfg = self.cfg
        units = list(source_units)
        total_source = len(units)
        committed: list[int] = []
        delays: list[int] = []
        traces: list[StepTrace] = []
        wall_start = time.perf_counter()

        if total_source == 0:
            return self._result(
                committed, delays, traces, 0.0, 0, 0.0, self._metadata()
            )

        read_pos = min(max(1, cfg.initial_read), total_source)
        step = 0
        while read_pos <= total_source and step < cfg.max_steps:
            step += 1
            partial = render_units(units[:read_pos], cfg.src_unit)
            step_sw = Stopwatch(cfg.cuda_sync)
            step_start = time.perf_counter()

            predictions: list[str] = []
            candidates: list[list[int]] = []
            selection = None
            selected_text = ""
            new_ids: list[int] = []

            if read_pos >= cfg.min_start:
                with step_sw.stage("llm_sample"):
                    predictions = self._sample(partial)
                with step_sw.stage("mt_decode"):
                    beam = cfg.effective_beam_width
                    for pred in predictions:
                        src_text = _join_prediction(partial, pred, cfg.src_unit)
                        candidates.extend(
                            self.mt.translate(
                                src_text,
                                committed,
                                num_sequences=beam,
                                num_beams=beam,
                                max_new_tokens=cfg.max_new_tokens,
                            )
                        )
                with step_sw.stage("aggregate"):
                    selection, selected_text, new_ids = _author_ralcp_continuation(
                        self.mt, candidates, cfg.ralcp_fraction, cfg.tgt_unit
                    )

            total_ms = (time.perf_counter() - step_start) * 1000.0
            timings = step_sw.as_dict()
            timings["total"] = round(total_ms, 6)

            before = len(committed)
            if new_ids:
                committed.extend(new_ids)
                delays.extend([read_pos] * len(new_ids))

            traces.append(
                StepTrace(
                    step=step,
                    read_pos=read_pos,
                    read_text=partial,
                    n_predictions=len(predictions),
                    n_candidates=len(candidates),
                    pi=selection.fraction if selection is not None else None,
                    support=selection.support if selection is not None else 0,
                    wrote=bool(new_ids),
                    committed_before=before,
                    committed_after=len(committed),
                    delay=read_pos,
                    timings_ms=timings,
                    detail={
                        "added_tokens": len(new_ids),
                        "selected": selected_text,
                        "agree_thres": cfg.ralcp_fraction,
                        "min_start": cfg.min_start,
                    },
                )
            )

            if read_pos >= total_source:
                break
            read_pos = min(read_pos + 1, total_source)

        # Fresh full-source EOS decode (authors' states.source_finished path).
        flush_ms = 0.0
        flush_added = 0
        step += 1
        full_source = render_units(units, cfg.src_unit)
        started = time.perf_counter()
        # The authors' generate(...)[0] uses the configured MT beam size but
        # returns a single best sequence.
        eos_beams = cfg.effective_beam_width
        conts = self.mt.translate(
            full_source,
            committed,
            num_sequences=1,
            num_beams=eos_beams,
            max_new_tokens=cfg.max_new_tokens,
        )
        new_ids = list(conts[0]) if conts else []
        before = len(committed)
        if new_ids:
            committed.extend(new_ids)
            delays.extend([total_source] * len(new_ids))
            flush_added = len(new_ids)
        flush_ms = (time.perf_counter() - started) * 1000.0
        traces.append(
            StepTrace(
                step=step,
                read_pos=total_source,
                read_text=full_source,
                n_predictions=0,
                n_candidates=1 if conts else 0,
                pi=None,
                support=0,
                wrote=bool(new_ids),
                committed_before=before,
                committed_after=len(committed),
                delay=total_source,
                timings_ms={"mt_decode": round(flush_ms, 6),
                            "total": round(flush_ms, 6)},
                detail={
                    "event": "eos_fresh_full_source",
                    "added_tokens": len(new_ids),
                    "num_beams": eos_beams,
                },
            )
        )

        wall_ms = (time.perf_counter() - wall_start) * 1000.0
        return self._result(
            committed, delays, traces, flush_ms, flush_added, wall_ms, self._metadata()
        )

    def _metadata(self) -> dict[str, Any]:
        return {
            "policy": "ralcp_taf",
            "n": self.cfg.n,
            "horizon": self.cfg.horizon,
            "top_k": self.cfg.top_k,
            "agree_thres": self.cfg.ralcp_fraction,
            "min_start": self.cfg.min_start,
            "beam_width": self.cfg.beam_width,
            "effective_beam_width": self.cfg.effective_beam_width,
            "candidate_budget": self.cfg.candidate_budget,
            "word_prefix_voting": True,
            "exact_majority_gate": False,
            "forced_committed_prefix": True,
            "fresh_full_source_eos": True,
            "final_flush": True,
        }


class RALCPPolicy(_BasePolicy):
    """RALCP baseline: beam candidates aggregated by relaxed LCP, no LLM."""

    def run(self, source_units: Sequence[str]) -> PolicyResult:
        cfg = self.cfg
        units = list(source_units)
        total_source = len(units)
        committed: list[int] = []
        delays: list[int] = []
        traces: list[StepTrace] = []
        wall_start = time.perf_counter()

        if total_source == 0:
            return self._result(committed, delays, traces, 0.0, 0, 0.0, self._metadata())

        read_pos = min(max(1, cfg.initial_read), total_source)
        step = 0
        while read_pos <= total_source and step < cfg.max_steps:
            step += 1
            partial = render_units(units[:read_pos], cfg.src_unit)
            step_sw = Stopwatch(cfg.cuda_sync)
            step_start = time.perf_counter()

            selection = None
            selected_text = ""
            new_ids: list[int] = []
            candidate_count = 0

            if read_pos >= cfg.min_start:
                with step_sw.stage("mt_decode"):
                    conts = self.mt.translate(
                        partial,
                        committed,
                        num_sequences=cfg.beam_width,
                        num_beams=cfg.beam_width,
                        max_new_tokens=cfg.max_new_tokens,
                    )
                    candidate_count = len(conts)
                with step_sw.stage("aggregate"):
                    selection, selected_text, new_ids = _author_ralcp_continuation(
                        self.mt, conts, cfg.ralcp_fraction, cfg.tgt_unit
                    )

            total_ms = (time.perf_counter() - step_start) * 1000.0
            timings = step_sw.as_dict()
            timings["total"] = round(total_ms, 6)

            before = len(committed)
            if new_ids:
                committed.extend(new_ids)
                delays.extend([read_pos] * len(new_ids))

            traces.append(
                StepTrace(
                    step=step,
                    read_pos=read_pos,
                    read_text=partial,
                    n_predictions=0,
                    n_candidates=candidate_count,
                    pi=selection.fraction if selection is not None else None,
                    support=selection.support if selection is not None else 0,
                    wrote=bool(new_ids),
                    committed_before=before,
                    committed_after=len(committed),
                    delay=read_pos,
                    timings_ms=timings,
                    detail={
                        "added_tokens": len(new_ids),
                        "selected": selected_text,
                        "fraction": cfg.ralcp_fraction,
                    },
                )
            )

            if read_pos >= total_source:
                break
            read_pos = min(read_pos + 1, total_source)

        # Fresh full-source EOS decode, as in agents/ralcp.py.
        flush_ms = 0.0
        flush_added = 0
        step += 1
        full_source = render_units(units, cfg.src_unit)
        started = time.perf_counter()
        conts = self.mt.translate(
            full_source,
            committed,
            num_sequences=1,
            num_beams=cfg.beam_width,
            max_new_tokens=cfg.max_new_tokens,
        )
        new_ids = list(conts[0]) if conts else []
        before = len(committed)
        if new_ids:
            committed.extend(new_ids)
            delays.extend([total_source] * len(new_ids))
            flush_added = len(new_ids)
        flush_ms = (time.perf_counter() - started) * 1000.0
        traces.append(
            StepTrace(
                step=step,
                read_pos=total_source,
                read_text=full_source,
                n_predictions=0,
                n_candidates=1 if conts else 0,
                pi=None,
                support=0,
                wrote=bool(new_ids),
                committed_before=before,
                committed_after=len(committed),
                delay=total_source,
                timings_ms={"mt_decode": round(flush_ms, 6),
                            "total": round(flush_ms, 6)},
                detail={"event": "eos_fresh_full_source",
                        "added_tokens": len(new_ids),
                        "num_beams": cfg.beam_width},
            )
        )

        wall_ms = (time.perf_counter() - wall_start) * 1000.0
        return self._result(
            committed, delays, traces, flush_ms, flush_added, wall_ms, self._metadata()
        )

    def _metadata(self) -> dict[str, Any]:
        return {
            "policy": "ralcp",
            "beam_width": self.cfg.beam_width,
            "ralcp_fraction": self.cfg.ralcp_fraction,
            "forced_committed_prefix": True,
            "final_flush": True,
        }


class WaitKPolicy(_BasePolicy):
    """Wait-K-Stride-N baseline (paper Sec. 4.2; N=1 for De-En)."""

    def run(self, source_units: Sequence[str]) -> PolicyResult:
        cfg = self.cfg
        units = list(source_units)
        total_source = len(units)
        committed: list[int] = []
        delays: list[int] = []
        traces: list[StepTrace] = []
        last_candidates: list[tuple[int, ...]] = []
        wall_start = time.perf_counter()

        if total_source == 0:
            return self._result(committed, delays, traces, 0.0, 0, 0.0, self._metadata())

        read_pos = min(max(1, cfg.wait_k), total_source)
        step = 0
        while read_pos <= total_source and step < cfg.max_steps:
            step += 1
            partial = render_units(units[:read_pos], cfg.src_unit)
            step_sw = Stopwatch(cfg.cuda_sync)
            step_start = time.perf_counter()

            with step_sw.stage("llm_sample"):
                pass

            with step_sw.stage("mt_decode"):
                conts = self.mt.translate(
                    partial,
                    committed,
                    num_sequences=1,
                    num_beams=1,
                    max_new_tokens=cfg.max_new_tokens,
                )
                full = (
                    extend_only(committed, tuple(committed) + tuple(conts[0]))
                    if conts
                    else tuple(committed)
                )
                candidates = [full]

            with step_sw.stage("aggregate"):
                new = list(full[len(committed):])
                new_token = new[:1]

            total_ms = (time.perf_counter() - step_start) * 1000.0
            timings = step_sw.as_dict()
            timings["total"] = round(total_ms, 6)

            before = len(committed)
            wrote = bool(new_token)
            if new_token:
                committed.extend(new_token)
                delays.extend([read_pos] * len(new_token))

            traces.append(
                StepTrace(
                    step=step,
                    read_pos=read_pos,
                    read_text=partial,
                    n_predictions=0,
                    n_candidates=1,
                    pi=None,
                    support=0,
                    wrote=wrote,
                    committed_before=before,
                    committed_after=len(committed),
                    delay=read_pos,
                    timings_ms=timings,
                    detail={"added_tokens": len(new_token), "wait_k": cfg.wait_k,
                            "stride": cfg.wait_stride},
                )
            )
            last_candidates = candidates

            if read_pos >= total_source:
                break
            read_pos = min(read_pos + cfg.wait_stride, total_source)

        flush_added, flush_ms = self._finish(
            committed, delays, last_candidates, units, render_units(units, cfg.src_unit)
        )
        wall_ms = (time.perf_counter() - wall_start) * 1000.0
        return self._result(
            committed, delays, traces, flush_ms, flush_added, wall_ms, self._metadata()
        )

    def _metadata(self) -> dict[str, Any]:
        return {
            "policy": "waitk",
            "wait_k": self.cfg.wait_k,
            "wait_stride": self.cfg.wait_stride,
            "forced_committed_prefix": True,
            "final_flush": True,
        }


def build_policy(llm, mt, cfg: StreamingConfig):
    if cfg.policy in ("ralcp_taf", "ralcp+taf"):
        if llm is None:
            raise ValueError(f"policy {cfg.policy!r} requires an LLM backend")
        return AuthorRALCPTAFPolicy(llm, mt, cfg)
    if cfg.policy == "taf":
        if llm is None:
            raise ValueError(f"policy {cfg.policy!r} requires an LLM backend")
        return TAFPolicy(llm, mt, cfg)
    if cfg.policy == "ralcp":
        return RALCPPolicy(mt, cfg)
    if cfg.policy == "waitk":
        return WaitKPolicy(mt, cfg)
    raise ValueError(f"unknown policy {cfg.policy!r}")
