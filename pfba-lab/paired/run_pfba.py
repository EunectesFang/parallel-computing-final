#!/usr/bin/env python3
"""Source-streaming TAF / PFBA for WMT de->en (standalone, CPU-testable).

Reveal source words from min_start=3.  Each non-EOS step: Qwen base samples B
German continuations of <=h tokens from the observed German prefix only; the
observed source plus B futures are translated (Marian/Tower) in one batch with
the committed target prefix explicitly forced; NEW target words are aggregated
by the exact upstream agents/utils.py::ralcp gate and only the selected suffix
is committed with its read position.  At source EOS the full TRUE source is
translated once with the forced prefix and flushed.  Earlier steps never see
it.  taf is the matched batch baseline, serial a diagnostic, pfba adds exact
sentence-scoped request reuse while preserving duplicate votes.  Timing is
GPU-synchronized; fixed word arrival yields source-position delays.  LAAL is
the SimulEval length-adaptive definition (first full-source emission cutoff,
no clamp).  BLEU is sacreBLEU; COMET optional.
"""

from __future__ import annotations

import argparse, gc, hashlib, json, time, traceback, urllib.request
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

PREDICTORS = {"small": "Qwen/Qwen2.5-0.5B", "large": "Qwen/Qwen2.5-7B"}
TRANSLATORS = {"small": "Helsinki-NLP/opus-mt-de-en", "large": "Unbabel/TowerInstruct-7B-v0.2"}
UPSTREAM_COMMIT = "99e373a887e0556c4d50e9ea08497889037876f8"
DATA_URL = "https://raw.githubusercontent.com/owaski/TAF/{commit}/data/de_en/{name}"
LOCAL_DATA = Path(__file__).resolve().parent.parent / "taf-upstream" / "data" / "de_en"
SPLITS, MODES = ("dev", "test"), ("taf", "serial", "pfba", "pfba-kv", "pfba-latest")
DEFAULT_COMET = "Unbabel/wmt22-comet-da"
MT_TEMPLATE = """<|im_start|> user
Translate the following text from German into English.
German: {src}
English:<|im_end|>
<|im_start|> assistant
{prefix}"""


def derive_step_seed(base_seed: int, split: str, sentence_id: str, read_pos: int) -> int:
    """Deterministic per-sentence/word seed, identical across modes."""
    payload = f"{int(base_seed)}|{split}|{sentence_id}|{int(read_pos)}"
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big") & ((1 << 63) - 1)


def upstream_ralcp(candidates: Sequence[str], threshold: float = 0.6) -> str:
    """Exact port of upstream agents/utils.py::ralcp for word candidates.

    Support = identical prefixes / ORIGINAL candidate count, including ended
    candidates.  Ties keep the first-appearing prefix.  Duplicates count.
    """
    if not candidates:
        return ""
    words, translation = [c.split(" ") for c in candidates], ""
    for i in range(max(len(w) for w in words)):
        counts = Counter(" ".join(w[:i + 1]) for w in words)
        best = max(counts, key=counts.get)
        if counts[best] < len(candidates) * threshold:
            break
        translation = best
    return translation


def completed_word_text(raw: str, ended: bool) -> str:
    """Finished words only. Completion is judged before strip().

    A trailing space means the last word is finished. strip() would delete
    that boundary, so it cannot be used to make the decision. EOS finishes
    the last word even when no space follows it.
    """
    if ended or raw != raw.rstrip():
        return raw.strip()
    head, separator, _tail = raw.rpartition(" ")
    if not separator:
        return ""
    return head.strip()


def _word_count(text: str) -> int:
    return 0 if text == "" else len(text.split(" "))


@dataclass(frozen=True)
class EarlyStopDecision:
    action: str
    texts: tuple[str, ...]
    resolved_words: int
    selected_words: int


def vote_early_stop_decision(rows: Sequence[tuple[str, bool]], threshold: float,
                             k: int) -> EarlyStopDecision:
    """Decide whether a greedy batch may stop, must continue, or should finish.

    rows are (completed words, ended). A word position is resolved only when
    every row that is still generating has completed it. Ended rows keep their
    short strings and still vote. k is a checkpoint: once that many words still
    agree, the caller finishes the original token budget instead of truncating.
    """
    parsed = [( [] if text == "" else text.split(" "), ended) for text, ended in rows]
    max_depth = max((len(words) for words, _ended in parsed), default=0)
    resolved = 0
    for index in range(max_depth):
        if any((not ended) and len(words) <= index for words, ended in parsed):
            break
        resolved = index + 1
    texts = tuple(" ".join(words[:resolved]) for words, _ended in parsed)
    if resolved == 0 or k <= 0:
        return EarlyStopDecision("continue", texts, resolved, 0)
    selected = upstream_ralcp(texts, threshold)
    selected_words = _word_count(selected)
    if selected_words >= k:
        return EarlyStopDecision("finish", texts, resolved, selected_words)
    if selected_words < resolved:
        return EarlyStopDecision("stop", texts, resolved, selected_words)
    return EarlyStopDecision("continue", texts, resolved, selected_words)


def call_translate_batch(translator, sources, prefix, **kwargs):
    """Pass final/early-stop arguments only when this translator accepts them."""
    import inspect
    try:
        parameters = inspect.signature(translator.translate_batch).parameters
    except (TypeError, ValueError):
        parameters = {}
    if any(item.kind == inspect.Parameter.VAR_KEYWORD for item in parameters.values()):
        accepted = kwargs
    else:
        accepted = {key: value for key, value in kwargs.items() if key in parameters}
    return translator.translate_batch(sources, prefix, **accepted)


def arrived_words(elapsed_s: float, gap_s: float, source_len: int) -> int:
    if source_len <= 0:
        return 0
    return source_len if gap_s <= 0 else min(source_len, int(elapsed_s / gap_s + 1e-9) + 1)


def laal_score(delays: Sequence[int], source_len: int, reference_len: int,
               prediction_len: int) -> float | None:
    """SimulEval-style length-adaptive average lagging (unclamped).

    Counts tokens up to and including the first full-source emission; adaptive
    target length is max(reference_len, prediction_len).
    """
    used = []
    for delay in delays:
        used.append(float(delay))
        if delay >= source_len:
            break
    target_len = max(int(reference_len), int(prediction_len))
    if not used or target_len <= 0:
        return None
    slope = float(source_len) / target_len
    return sum(d - i * slope for i, d in enumerate(used)) / len(used)


class StageTimer:
    """Real wall time with device synchronization around each stage."""

    def __init__(self, sync=None):
        self.sync = sync or (lambda: None)
        self.seconds: dict[str, float] = {}

    def run(self, name, fn):
        self.sync(); start = time.perf_counter(); out = fn(); self.sync()
        duration = max(0.0, time.perf_counter() - start)
        self.seconds[name] = self.seconds.get(name, 0.0) + duration
        return out, duration


class NullTimer(StageTimer):
    def __init__(self):
        super().__init__()

    def run(self, name, fn):
        return fn(), 0.0


class MTCache:
    """Exact sentence-scoped MT request reuse; duplicate votes preserved."""

    def __init__(self):
        self.cache: dict[tuple[str, str], str] = {}
        self.hits = self.misses = self.full_reuse_hits = 0

    def translate(self, translator, sources, prefix, timer, final=False):
        keys = [(source, prefix) for source in sources]
        if final and any(key in self.cache for key in keys):
            self.full_reuse_hits += 1
        missing, seen, seconds = [], set(), 0.0
        for key in keys:
            if key in self.cache or key in seen:
                self.hits += 1
            else:
                seen.add(key); missing.append(key)
        if missing:
            self.misses += len(missing)
            results, seconds = timer.run(
                "translate", lambda: call_translate_batch(
                    translator, [k[0] for k in missing], prefix, final=final))
            for key, value in zip(missing, results):
                self.cache[key] = value
        return [self.cache[key] for key in keys], seconds


@dataclass(frozen=True)
class StreamConfig:
    branches: int = 10; horizon: int = 10; min_start: int = 3
    ralcp_threshold: float = 0.6; seed: int = 13; arrival_gap: float = 0.3
    max_len_a: float = 1.5; max_len_b: int = 20; split: str = "test"
    predictor: str = "small"; translator: str = "small"
    local_files_only: bool = False; trust_remote_code: bool = False

@dataclass(frozen=True)
class Step:
    read_pos: int; observed: str; n_candidates: int; selected: tuple[str, ...]
    committed_after: int; delay: int; predict_s: float; translate_s: float
    candidate_sha256: str | None = None
    reused_prompt_tokens: int = 0

@dataclass(frozen=True)
class StreamTrace:
    sentence_id: str; split: str; mode: str; predictor: str; translator: str
    source: str; reference: str; hypothesis: str; committed: tuple[str, ...]
    delays: tuple[int, ...]; steps: tuple[Step, ...]; laal: float | None
    predict_s: float; translate_s: float; inference_s: float
    reuse_hits: int = 0; reuse_misses: int = 0; full_reuse_hits: int = 0
    first_output_s: float | None = None
    completion_s: float = 0.0


def run_streaming(sentence_id, split, source, reference, predictor, translator,
                  mode, cfg, timer=None):
    """Run one sentence under one policy mode and return a StreamTrace."""
    timer = timer or NullTimer()
    words, n = source.split(), len(source.split())
    committed, delays, steps = [], [], []
    cache = MTCache() if mode in ("pfba", "pfba-kv") else None
    if mode == "pfba-kv":
        from kv_prefix import PrefixKVPredictor

        predictor = PrefixKVPredictor(predictor.model, predictor.tok)
    predict_s = translate_s = virtual_s = 0.0
    first_output_s = None

    def translate(sources, prefix, final=False):
        nonlocal translate_s
        if mode == "serial":
            outs, seconds = [], 0.0
            for text in sources:
                result, elapsed = timer.run(
                    "translate", lambda text=text: call_translate_batch(
                        translator, [text], prefix, final=final))
                outs.append(result[0]); seconds += elapsed
        elif cache is not None:
            outs, seconds = cache.translate(translator, list(sources), prefix, timer, final)
        else:
            result, seconds = timer.run(
                "translate", lambda: call_translate_batch(
                    translator, list(sources), prefix, final=final))
            outs = list(result)
        translate_s += seconds
        return outs, seconds

    if n == 0:
        return StreamTrace(sentence_id, split, mode, cfg.predictor, cfg.translator,
                           source, reference, "", (), (), (), None, 0.0, 0.0, 0.0)
    read_pos = min(cfg.min_start, n)
    while True:
        prefix = " ".join(committed)
        if read_pos >= n:  # EOS: translate the TRUE full source with the forced prefix
            virtual_s = max(virtual_s, (n - 1) * cfg.arrival_gap)
            outs, seconds = translate([source], prefix, final=True)
            virtual_s += seconds
            delay = arrived_words(virtual_s, cfg.arrival_gap, n)
            new = tuple(outs[0].split())
            if new and first_output_s is None:
                first_output_s = virtual_s
            committed.extend(new); delays.extend([delay] * len(new))
            steps.append(Step(n, source, 1, new, len(committed), delay, 0.0, seconds))
            break
        observed = " ".join(words[:read_pos])
        virtual_s = max(virtual_s, (read_pos - 1) * cfg.arrival_gap)
        seed = derive_step_seed(cfg.seed, split, sentence_id, read_pos)
        continuations, predict_elapsed = timer.run(
            "predict", lambda: predictor.sample(observed, cfg.branches, cfg.horizon, seed))
        predict_s += predict_elapsed
        futures = [observed if not c.strip() else f"{observed} {c.strip()}"
                   for c in (continuations or [])]
        requests = futures or [observed]
        outs, translate_elapsed = translate(requests, prefix)
        virtual_s += predict_elapsed + translate_elapsed
        selected, aggregate_s = timer.run(
            "aggregate", lambda: upstream_ralcp(list(outs), cfg.ralcp_threshold))
        virtual_s += aggregate_s
        delay = arrived_words(virtual_s, cfg.arrival_gap, n)
        new = tuple(selected.split()) if selected else ()
        if new and first_output_s is None:
            first_output_s = virtual_s
        committed.extend(new); delays.extend([read_pos] * len(new))
        steps.append(Step(read_pos, observed, len(requests), new, len(committed),
                          delay, predict_elapsed, translate_elapsed,
                          hashlib.sha256(json.dumps(requests).encode()).hexdigest(),
                          getattr(predictor, "last_common_prefix", 0)))
        read_pos = max(read_pos + 1, delay) if mode == "pfba-latest" else read_pos + 1
    return StreamTrace(
        sentence_id, split, mode, cfg.predictor, cfg.translator, source, reference,
        " ".join(committed), tuple(committed), tuple(delays), tuple(steps),
        laal_score(delays, n, len(reference.split()), len(committed)),
        predict_s, translate_s, predict_s + translate_s + timer.seconds.get("aggregate", 0.0),
        cache.hits if cache else 0, cache.misses if cache else 0,
        cache.full_reuse_hits if cache else 0, first_output_s, virtual_s)


# Model adapters.  Imports stay local so the CPU tests need no torch.
def _torch():
    import torch
    return torch


def _dtype(torch, requested, device):
    if requested != "auto":
        return getattr(torch, requested)
    if device.startswith("cuda"):
        with torch.cuda.device(device):
            supported = torch.cuda.is_bf16_supported(including_emulation=False)
        return torch.bfloat16 if supported else torch.float16
    return torch.float32


def _sync(device):
    torch = _torch()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _prep(model_cls, model_id, device, dtype, local_only, trust, left=False):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        model_id, local_files_only=local_only, trust_remote_code=trust)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if left:
        tok.padding_side = "left"
    torch = _torch()
    model = model_cls.from_pretrained(
        model_id, torch_dtype=_dtype(torch, dtype, device), low_cpu_mem_usage=True,
        local_files_only=local_only, trust_remote_code=trust)
    model.to(device); model.eval()
    return tok, model


def _decode(tok, rows):
    return [tok.decode(row, skip_special_tokens=True,
                       clean_up_tokenization_spaces=False).strip() for row in rows]


def _generated_content(token_ids, eos_id, pad_id):
    content = []
    for token in token_ids:
        if eos_id is not None and token == eos_id:
            return content, True
        if pad_id is not None and token == pad_id:
            break
        content.append(int(token))
    return content, False


def _vote_stop(tokenizer, prompt_len, eos_id, pad_id, threshold, k):
    """Stopping criterion that uses the same generate() loop as full decoding.

    Returning early does not change the greedy tokens already produced. Once k
    agreed words are visible, generation continues to the original budget.
    """
    from transformers import StoppingCriteria, StoppingCriteriaList
    torch = _torch()
    state = {"continued_full": False, "stopped_early": False, "texts": None}

    class _VoteStop(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            batch = input_ids.shape[0]
            keep_going = torch.zeros(batch, dtype=torch.bool, device=input_ids.device)
            if state["continued_full"] or input_ids.shape[1] < prompt_len:
                return keep_going
            rows = []
            for row in input_ids:
                content, ended = _generated_content(row[prompt_len:].tolist(), eos_id, pad_id)
                raw = tokenizer.decode(content, skip_special_tokens=True,
                                       clean_up_tokenization_spaces=False)
                rows.append((completed_word_text(raw, ended), ended))
            decision = vote_early_stop_decision(rows, threshold, k)
            if decision.action == "finish":
                state["continued_full"] = True
                return keep_going
            if decision.action == "stop":
                state["stopped_early"] = True
                state["texts"] = list(decision.texts)
                return torch.ones(batch, dtype=torch.bool, device=input_ids.device)
            return keep_going

    return StoppingCriteriaList([_VoteStop()]), state


def _record_decode(translator, output, prompt_len, state):
    new_rows = output[:, prompt_len:]
    translator.last_decode_stats = {
        "new_tokens": int(new_rows.shape[1]),
        "continued_full": bool(state and state["continued_full"]),
        "stopped_early": bool(state and state["stopped_early"]),
    }
    if state and state["stopped_early"]:
        return list(state["texts"])
    return _decode(translator.tok, new_rows)


class QwenPredictor:
    def __init__(self, model_id, device, dtype, local_only, trust):
        self.model_id, self.device, self.dtype = model_id, device, dtype
        self.local_only, self.trust = local_only, trust

    def load(self):
        from transformers import AutoModelForCausalLM
        self.tok, self.model = _prep(
            AutoModelForCausalLM, self.model_id, self.device, self.dtype,
            self.local_only, self.trust)

    def sample(self, prefix, branches, horizon, seed):
        batch_size = getattr(self, "prediction_batch_size", 0)
        if 0 < batch_size < branches:
            continuations = []
            for offset in range(0, branches, batch_size):
                continuations.extend(self.sample(prefix, min(batch_size, branches - offset),
                                                 horizon, seed + offset))
            return continuations
        torch = _torch()
        inputs = self.tok(prefix, return_tensors="pt", add_special_tokens=False).to(self.device)
        device = torch.device(self.device)
        devices = [device.index or 0] if device.type == "cuda" else []
        with torch.inference_mode(), torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(int(seed))
            if devices:
                with torch.cuda.device(devices[0]):
                    torch.cuda.manual_seed(int(seed))
            output = self.model.generate(
                **inputs, max_new_tokens=horizon, do_sample=True, temperature=1.0,
                top_p=0.9, top_k=0, num_return_sequences=branches,
                eos_token_id=self.tok.eos_token_id, pad_token_id=self.tok.pad_token_id)
        return [t.split("\n")[0].strip()
                for t in _decode(self.tok, output[:, inputs["input_ids"].shape[1]:])]


class _Translator:
    def __init__(self, model_id, device, dtype, max_len_a, max_len_b, local_only, trust):
        self.model_id, self.device, self.dtype = model_id, device, dtype
        self.max_len_a, self.max_len_b = max_len_a, max_len_b
        self.local_only, self.trust = local_only, trust


class MarianTranslator(_Translator):
    def load(self):
        from transformers import AutoModelForSeq2SeqLM
        self.tok, self.model = _prep(
            AutoModelForSeq2SeqLM, self.model_id, self.device, self.dtype,
            self.local_only, self.trust)
        config = self.model.config
        self.start = int(config.decoder_start_token_id or config.pad_token_id)

    def translate_batch(self, sources, prefix, final=False, early_stop_k=0, threshold=0.6):
        torch = _torch()
        batch = self.tok(list(sources), return_tensors="pt", padding=True,
                         truncation=True, max_length=512).to(self.device)
        prefix_ids = self.tok(text_target=prefix, add_special_tokens=False)["input_ids"] if prefix else []
        decoder = torch.tensor([[self.start] + prefix_ids] * len(sources), device=self.device)
        max_new = max(1, 96 - len(prefix_ids))
        criteria, state = (None, None)
        if not final and early_stop_k > 0:
            criteria, state = _vote_stop(
                self.tok, decoder.shape[1], self.tok.eos_token_id, self.tok.pad_token_id,
                threshold, early_stop_k)
        with torch.inference_mode():
            kwargs = {} if criteria is None else {"stopping_criteria": criteria}
            output = self.model.generate(**batch, decoder_input_ids=decoder,
                                         max_new_tokens=max_new, num_beams=1, do_sample=False,
                                         **kwargs)
        return _record_decode(self, output, decoder.shape[1], state)


class TowerTranslator(_Translator):
    def load(self):
        from transformers import AutoModelForCausalLM
        self.tok, self.model = _prep(
            AutoModelForCausalLM, self.model_id, self.device, self.dtype,
            self.local_only, self.trust, left=True)

    def translate_batch(self, sources, prefix, final=False, early_stop_k=0, threshold=0.6):
        torch = _torch()
        prompts = [MT_TEMPLATE.format(src=source, prefix=prefix) for source in sources]
        batch = self.tok(prompts, padding=True, return_tensors="pt",
                         truncation=True, max_length=2048).to(self.device)
        prefix_ids = self.tok(prefix, add_special_tokens=False)["input_ids"] if prefix else []
        max_new = max(1, 96 - len(prefix_ids))
        prompt_len = batch["input_ids"].shape[1]
        criteria, state = (None, None)
        if not final and early_stop_k > 0:
            criteria, state = _vote_stop(
                self.tok, prompt_len, self.tok.eos_token_id, self.tok.pad_token_id,
                threshold, early_stop_k)
        with torch.inference_mode():
            kwargs = {} if criteria is None else {"stopping_criteria": criteria}
            output = self.model.generate(
                **batch, max_new_tokens=max_new, num_beams=1, do_sample=False,
                pad_token_id=self.tok.pad_token_id, eos_token_id=self.tok.eos_token_id,
                **kwargs)
        return _record_decode(self, output, prompt_len, state)


def load_translator(name, model_id, device, cfg):
    kwargs = dict(model_id=model_id, device=device, dtype="auto", max_len_a=cfg.max_len_a,
                  max_len_b=cfg.max_len_b, local_only=cfg.local_files_only,
                  trust=cfg.trust_remote_code)
    translator = MarianTranslator(**kwargs) if name == "small" else TowerTranslator(**kwargs)
    translator.load()
    return translator


@dataclass(frozen=True)
class Sample:
    sentence_id: str; source: str; reference: str


def read_data_file(split, lang):
    path = LOCAL_DATA / f"{split}.{lang}"
    if path.is_file():
        text = path.read_text(encoding="utf-8")
    else:
        url = DATA_URL.format(commit=UPSTREAM_COMMIT, name=f"{split}.{lang}")
        with urllib.request.urlopen(url, timeout=60) as response:
            text = response.read().decode("utf-8")
    return [line.rstrip("\n") for line in text.splitlines()]


def load_split(split, limit=0):
    sources, references = read_data_file(split, "de"), read_data_file(split, "en")
    count = min(len(sources), len(references))
    if limit > 0:
        count = min(count, limit)
    return [Sample(f"{split}:{i:04d}", sources[i], references[i]) for i in range(count)]


def corpus_bleu(hypotheses, references):
    try:
        from sacrebleu.metrics import BLEU
    except Exception as exc:
        return {"status": "unavailable", "reason": str(exc)}
    metric = BLEU()
    result = metric.corpus_score(list(hypotheses), [list(references)])
    return {"status": "ok", "score": float(result.score), "signature": str(metric.get_signature())}


_COMET_PATHS = {}


def comet_score(hypotheses, sources, references, model_name, device):
    try:
        from comet import download_model, load_from_checkpoint
        path = _COMET_PATHS.get(model_name)
        if path is None:
            path = _COMET_PATHS[model_name] = download_model(model_name)
        model = load_from_checkpoint(path)
        data = [{"src": s, "mt": h, "ref": r}
                for s, h, r in zip(sources, hypotheses, references)]
        result = model.predict(data, batch_size=8,
                               gpus=1 if str(device).startswith("cuda") else 0)
        return {"status": "ok", "score": float(result.system_score), "model": model_name,
                "sentence_scores": [float(score) for score in result.scores]}
    except Exception as exc:
        return {"status": "error", "reason": f"{type(exc).__name__}: {exc}"}


def corpus_laal(traces):
    values = [t.laal for t in traces if t.laal is not None]
    return sum(values) / len(values) if values else None


def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Source-streaming TAF/PFBA: Qwen base futures x Marian/Tower.")
    p.add_argument("--predictor", nargs="+", choices=sorted(PREDICTORS), default=["small"])
    p.add_argument("--translator", nargs="+", choices=sorted(TRANSLATORS), default=["small"])
    p.add_argument("--modes", nargs="+", choices=MODES, default=["taf", "pfba"])
    p.add_argument("--limit", type=int, default=0); p.add_argument("--branches", type=int, default=10)
    p.add_argument("--horizon", type=int, default=10); p.add_argument("--split", choices=SPLITS, default="test")
    p.add_argument("--output-dir", default="outputs/paired"); p.add_argument("--skip-comet", action="store_true")
    p.add_argument("--ralcp-threshold", type=float, default=0.6); p.add_argument("--min-start", type=int, default=3)
    p.add_argument("--seed", type=int, default=13); p.add_argument("--arrival-gap", type=float, default=0.3)
    p.add_argument("--max-len-a", type=float, default=1.5); p.add_argument("--max-len-b", type=int, default=20)
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--predictor-device", default=None); p.add_argument("--translator-device", default=None)
    p.add_argument("--local-files-only", action="store_true"); p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


def resolve_config(args):
    if not 0.0 <= float(args.ralcp_threshold) <= 1.0:
        raise ValueError("--ralcp-threshold must be in [0, 1]")
    if int(args.branches) <= 0 or int(args.horizon) <= 0 or int(args.min_start) < 1:
        raise ValueError("--branches/--horizon must be positive; --min-start >= 1")
    if int(args.limit) < 0 or float(args.arrival_gap) < 0:
        raise ValueError("--limit and --arrival-gap must be non-negative")
    return StreamConfig(
        branches=int(args.branches), horizon=int(args.horizon),
        min_start=int(args.min_start), ralcp_threshold=float(args.ralcp_threshold),
        seed=int(args.seed), arrival_gap=float(args.arrival_gap),
        max_len_a=float(args.max_len_a), max_len_b=int(args.max_len_b), split=args.split,
        predictor=args.predictor[0], translator=args.translator[0],
        local_files_only=bool(args.local_files_only),
        trust_remote_code=bool(args.trust_remote_code))


def resolve_devices(torch, args, predictor_name, translator_name):
    if args.device == "cpu":
        return "cpu", "cpu"
    if not torch.cuda.is_available():
        if args.device == "cuda":
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        return "cpu", "cpu"
    predictor_device, translator_device = args.predictor_device, args.translator_device
    if (predictor_device is None and translator_device is None
            and torch.cuda.device_count() >= 2
            and "large" in (predictor_name, translator_name)):
        predictor_device, translator_device = "cuda:0", "cuda:1"
    predictor_device = predictor_device or translator_device or "cuda:0"
    return predictor_device, translator_device or predictor_device


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _summary_markdown(cells, cfg):
    lines = [
        "# Source-streaming TAF/PFBA", "",
        f"- split: {cfg.split}; B={cfg.branches}; h={cfg.horizon}; "
        f"RALCP={cfg.ralcp_threshold}; min_start={cfg.min_start}; seed={cfg.seed}",
        "- LAAL: SimulEval length-adaptive, first full-source-emission cutoff, unclamped.",
        "", "| predictor | translator | mode | n | BLEU | COMET | LAAL | predict s | translate s | reuse hits |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for cell in cells:
        bleu = cell["bleu"].get("score") if cell["bleu"].get("status") == "ok" else None
        comet = cell["comet"].get("score") if cell["comet"].get("status") == "ok" else None
        lines.append(
            f"| {cell['predictor']} | {cell['translator']} | {cell['mode']} | {cell['n']} "
            f"| {bleu if bleu is not None else cell['bleu'].get('status')} "
            f"| {comet if comet is not None else cell['comet'].get('status')} "
            f"| {cell['laal'] if cell['laal'] is not None else 'n/a'} "
            f"| {cell['predict_s']:.3f} | {cell['translate_s']:.3f} | {cell['reuse_hits']} |")
    return "\n".join(lines) + "\n"


def run(args):
    torch, cfg = _torch(), resolve_config(args)
    samples = load_split(cfg.split, args.limit)
    if not samples:
        raise RuntimeError("no sentences loaded")
    output_dir = Path(args.output_dir) / time.strftime(f"{cfg.split}_%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[pfba] output: {output_dir}; sentences={len(samples)}")
    cells = []
    for translator_name in args.translator:
        for predictor_name in args.predictor:
            pdev, tdev = resolve_devices(torch, args, predictor_name, translator_name)
            print(f"[pfba] {predictor_name}@{pdev} x {translator_name}@{tdev}")
            predictor = QwenPredictor(PREDICTORS[predictor_name], pdev, "auto",
                                      cfg.local_files_only, cfg.trust_remote_code)
            predictor.load()
            translator = load_translator(translator_name, TRANSLATORS[translator_name], tdev, cfg)
            def sync():
                _sync(pdev)
                if tdev != pdev:
                    _sync(tdev)
            warmup = samples[0]
            run_streaming(warmup.sentence_id, cfg.split, warmup.source, warmup.reference,
                          predictor, translator, "taf", cfg, StageTimer(sync))
            for mode in args.modes:
                traces = [run_streaming(s.sentence_id, cfg.split, s.source, s.reference,
                                        predictor, translator, mode, cfg, StageTimer(sync))
                          for s in samples]
                _write_jsonl(output_dir / f"sentences_{predictor_name}_{translator_name}_{mode}.jsonl",
                             [asdict(t) for t in traces])
                hypotheses = [t.hypothesis for t in traces]
                references = [t.reference for t in traces]
                bleu = corpus_bleu(hypotheses, references)
                comet = ({"status": "skipped"} if args.skip_comet else
                         comet_score(hypotheses, [t.source for t in traces], references,
                                     DEFAULT_COMET,
                                     "cpu" if "large" in (predictor_name, translator_name) else tdev))
                completion = sorted(t.completion_s for t in traces)
                cell = {"predictor": predictor_name, "translator": translator_name,
                        "mode": mode, "split": cfg.split, "n": len(traces), "bleu": bleu,
                        "comet": comet, "laal": corpus_laal(traces),
                        "mean_completion_s": sum(t.completion_s for t in traces) / len(traces),
                        "p95_completion_s": completion[(95 * len(completion) - 1) // 100],
                        "predictor_device": pdev, "translator_device": tdev,
                        "predictor_dtype": str(predictor.model.dtype),
                        "translator_dtype": str(translator.model.dtype),
                        **{k: sum(getattr(t, k) for t in traces) for k in (
                            "predict_s", "translate_s", "inference_s", "reuse_hits",
                            "reuse_misses", "full_reuse_hits")}}
                cells.append(cell)
                print(f"[pfba] {predictor_name}/{translator_name}/{mode}: "
                      f"BLEU={bleu.get('score', bleu.get('status'))} "
                      f"LAAL={cell['laal']} infer={cell['inference_s']:.3f}s")
            del predictor, translator; gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    payload = {"config": cfg.__dict__, "cells": cells, "predictors": PREDICTORS,
               "translators": TRANSLATORS, "upstream_commit": UPSTREAM_COMMIT}
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "summary.md").write_text(_summary_markdown(cells, cfg), encoding="utf-8")
    print(f"[pfba] wrote {output_dir / 'summary.json'}")
    return output_dir


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        cfg = resolve_config(args)
    except ValueError as exc:
        parser.error(str(exc))
        return 2
    if args.dry_run:
        print(json.dumps({"config": cfg.__dict__,
                          "predictors": {n: PREDICTORS[n] for n in args.predictor},
                          "translators": {n: TRANSLATORS[n] for n in args.translator},
                          "modes": args.modes}, ensure_ascii=False, indent=2))
        return 0
    try:
        run(args)
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
