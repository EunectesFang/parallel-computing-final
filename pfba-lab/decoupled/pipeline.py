"""Two independent inference stages on a real-time, causally paced source replay."""
from __future__ import annotations

import hashlib
import json
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Condition
from time import perf_counter, sleep

# The Kaggle package may place the shared adapter beside this file.
if not Path(__file__).with_name("run_pfba.py").is_file():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paired"))

from run_pfba import arrived_words, derive_step_seed, laal_score, upstream_ralcp

DEFAULT_MODES = ("taf", "overlap-fifo", "overlap-latest")
BOUNDED_SKIPS = {f"bounded-{cap}": cap for cap in (1, 2, 3)}
NAMED_CADENCES = {f"bounded-1-every{cadence}": cadence for cadence in (2, 3)}
BOUNDED_SKIPS.update({name: 1 for name in NAMED_CADENCES})
SCHEDULERS = DEFAULT_MODES + tuple(BOUNDED_SKIPS)
MODES = SCHEDULERS + tuple(f"{mode}-anchor" for mode in SCHEDULERS)


@dataclass(frozen=True)
class Config:
    branches: int = 4
    horizon: int = 6
    min_start: int = 3
    arrival_gap: float = 0.3
    threshold: float = 0.6
    seed: int = 13
    ready_limit: int = 1
    anchor_policy: str = "vote"
    anchor_threshold: float | None = None
    bounded_threshold: float | None = None
    bounded_initial_steps: int = 2
    bounded_initial_words: int = 1
    bounded_skip_every: int = 1

    def __post_init__(self):
        if min(self.branches, self.horizon, self.min_start, self.ready_limit) < 1:
            raise ValueError("branches, horizon, min_start and ready_limit must be positive")
        if self.arrival_gap <= 0 or not 0 < self.threshold <= 1:
            raise ValueError("arrival_gap must be positive and threshold must be in (0, 1]")
        if self.anchor_policy not in ("vote", "veto"):
            raise ValueError("anchor_policy must be vote or veto")
        if self.anchor_threshold is not None and not 0 < self.anchor_threshold <= 1:
            raise ValueError("anchor_threshold must be in (0, 1]")
        if self.bounded_threshold is not None and not 0 < self.bounded_threshold <= 1:
            raise ValueError("bounded_threshold must be in (0, 1]")
        if min(self.bounded_initial_steps, self.bounded_initial_words) < 0:
            raise ValueError("bounded initial steps and words must be nonnegative")
        if self.bounded_skip_every < 1:
            raise ValueError("bounded_skip_every must be positive")


@dataclass(frozen=True)
class Span:
    read_pos: int
    start_s: float
    end_s: float
    completed_mt_steps_at_submit: int | None = None
    skipped_before: int = 0


@dataclass(frozen=True)
class Step:
    read_pos: int
    arrived_at_commit: int
    new_words: tuple[str, ...]
    translation: Span
    candidate_sha256: str
    translation_requests: int


@dataclass(frozen=True)
class Trace:
    hypothesis: str
    delays: tuple[int, ...]
    steps: tuple[Step, ...]
    predictions: tuple[Span, ...]
    laal: float | None
    first_output_s: float | None
    completion_s: float
    drain_s: float
    overlap_s: float
    peak_ready: int
    discarded_predictions: int


def measure_overlap(predictions, translations):
    overlap = 0.0
    i = j = 0
    while i < len(predictions) and j < len(translations):
        p, t = predictions[i], translations[j]
        overlap += max(0.0, min(p.end_s, t.end_s) - max(p.start_s, t.start_s))
        if p.end_s <= t.end_s:
            i += 1
        else:
            j += 1
    return overlap


def run_sentence(
    source, reference, predictor, translator, cfg, mode, *,
    sentence_id="0", split="dev", sync_predict=lambda: None, sync_translate=lambda: None,
):
    """Replay one sentence with at most one in-flight prediction and bounded ready work.

    Prediction depends on source alone. Translation and commitment run in a single
    consumer so every MT request observes the committed target prefix.
    """
    if mode not in MODES:
        raise ValueError(f"unsupported mode: {mode}")
    anchored = mode.endswith("-anchor")
    scheduler = mode.removesuffix("-anchor")
    skip_cap = BOUNDED_SKIPS.get(scheduler)
    skip_every = NAMED_CADENCES.get(scheduler, cfg.bounded_skip_every)
    threshold = cfg.threshold if skip_cap is None else cfg.bounded_threshold or cfg.threshold
    words = source.split()
    n = len(words)
    if n == 0:
        return Trace("", (), (), (), None, None, 0.0, 0.0, 0.0, 0, 0)

    origin = perf_counter()
    eos_s = (n - 1) * cfg.arrival_gap
    latest = scheduler == "overlap-latest"
    committed, delays, predictions, steps = [], [], [], []
    first_output_s = None

    def elapsed():
        return perf_counter() - origin

    def predict(read_pos, completed_at_submit=None, skipped_before=0):
        observed = " ".join(words[:read_pos])
        seed = derive_step_seed(cfg.seed, split, sentence_id, read_pos)
        start = elapsed()
        continuations = predictor.sample(observed, cfg.branches, cfg.horizon, seed)
        sync_predict()
        predictions.append(Span(read_pos, start, elapsed(), completed_at_submit, skipped_before))
        return [f"{observed} {value.strip()}" if value.strip() else observed
                for value in (continuations or ())] or [observed]

    def translate(read_pos, requests, final=False):
        nonlocal first_output_s
        start = elapsed()
        anchor_index = None
        batch = requests
        if anchored and not final:
            observed = " ".join(words[:read_pos])
            if observed in requests:
                anchor_index = requests.index(observed)
            else:
                anchor_index = len(requests)
                batch = [*requests, observed]
        outputs = translator.translate_batch(batch, " ".join(committed))
        if final:
            selected = outputs[0]
        elif anchor_index is None:
            selected = upstream_ralcp(outputs, threshold)
        else:
            votes = outputs[:len(requests)]
            anchor = outputs[anchor_index]
            anchor_threshold = cfg.anchor_threshold or threshold
            if cfg.anchor_policy == "vote":
                selected = upstream_ralcp([*votes, anchor], anchor_threshold)
            else:
                consensus = upstream_ralcp(votes, anchor_threshold).split()
                shared = []
                for candidate_word, anchor_word in zip(consensus, anchor.split()):
                    if candidate_word != anchor_word:
                        break
                    shared.append(candidate_word)
                selected = " ".join(shared)
        sync_translate()
        end = elapsed()
        new_words = tuple(selected.split())
        committed.extend(new_words)
        delays.extend([read_pos] * len(new_words))
        if new_words and first_output_s is None:
            first_output_s = end
        steps.append(Step(
            read_pos, arrived_words(end, cfg.arrival_gap, n),
            new_words, Span(read_pos, start, end),
            hashlib.sha256(json.dumps(requests).encode()).hexdigest(), len(batch),
        ))

    def wait_for(deadline):
        remaining = deadline - elapsed()
        if remaining > 0:
            sleep(remaining)

    peak_ready = discarded = 0
    if scheduler == "taf":
        for read_pos in range(cfg.min_start, n):
            wait_for((read_pos - 1) * cfg.arrival_gap)
            translate(read_pos, predict(read_pos))
        wait_for(eos_s)
        translate(n, [source], final=True)
    else:
        wake = Condition()
        ready = deque()
        stopped = finished = False
        error = None
        last_dispatched = cfg.min_start - 1
        completed_steps = completed_words = 0

        def produce():
            nonlocal peak_ready, discarded, finished, error
            read_pos = cfg.min_start
            last_skip_completed = None
            try:
                while read_pos < n:
                    with wake:
                        while not stopped:
                            now = elapsed()
                            if latest and now >= eos_s:
                                return
                            delay = (read_pos - 1) * cfg.arrival_gap - now
                            in_range = skip_cap is None or read_pos <= last_dispatched + skip_cap + 1
                            if delay <= 0 and in_range and (latest or len(ready) < cfg.ready_limit):
                                break
                            wake.wait(timeout=delay if delay > 0 else None)
                        if stopped:
                            return
                        previous_read_pos = read_pos
                        if latest:
                            read_pos = max(read_pos, arrived_words(elapsed(), cfg.arrival_gap, n))
                            if read_pos >= n:
                                return
                        elif (skip_cap is not None and last_dispatched >= cfg.min_start
                              and completed_steps >= cfg.bounded_initial_steps
                              and completed_words >= cfg.bounded_initial_words
                              and (skip_every == 1 or last_skip_completed is None
                                   or completed_steps - last_skip_completed >= skip_every)):
                            newest = arrived_words(elapsed(), cfg.arrival_gap, n)
                            read_pos = max(read_pos, min(newest, n - 1, last_dispatched + skip_cap + 1))
                        skipped_before = read_pos - previous_read_pos
                        completed_at_submit = completed_steps
                        if skip_cap is not None and skipped_before:
                            last_skip_completed = completed_steps
                    requests = predict(read_pos, completed_at_submit, skipped_before)
                    with wake:
                        if stopped or (latest and elapsed() >= eos_s):
                            discarded += 1
                            return
                        if len(ready) == cfg.ready_limit:
                            ready.popleft()
                            discarded += 1
                        ready.append((read_pos, requests))
                        peak_ready = max(peak_ready, len(ready))
                        wake.notify_all()
                    read_pos += 1
            except Exception as exc:
                with wake:
                    error = exc
            finally:
                with wake:
                    finished = True
                    wake.notify_all()

        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="pfba-predict") as pool:
            future = pool.submit(produce)
            try:
                while True:
                    with wake:
                        while not ready and not finished:
                            remaining = eos_s - elapsed() if latest else None
                            if remaining is not None and remaining <= 0:
                                break
                            wake.wait(timeout=remaining)
                        if error is not None:
                            raise error
                        if latest and elapsed() >= eos_s:
                            discarded += len(ready)
                            ready.clear()
                            stopped = True
                            wake.notify_all()
                            break
                        if not ready:
                            break
                        if latest:
                            read_pos, requests = ready.pop()
                            discarded += len(ready)
                            ready.clear()
                        else:
                            read_pos, requests = ready.popleft()
                        last_dispatched = read_pos
                        wake.notify_all()
                    translate(read_pos, requests)
                    with wake:
                        completed_steps += 1
                        completed_words = len(committed)
                        wake.notify_all()
                wait_for(eos_s)
                translate(n, [source], final=True)
            finally:
                with wake:
                    stopped = True
                    wake.notify_all()
            future.result()
            if error is not None:
                raise error

    completion_s = steps[-1].translation.end_s
    drain_s = max(completion_s, predictions[-1].end_s) if predictions else completion_s
    return Trace(
        " ".join(committed), tuple(delays), tuple(steps), tuple(predictions),
        laal_score(delays, n, len(reference.split()), len(committed)),
        first_output_s, completion_s, drain_s,
        measure_overlap(predictions, [step.translation for step in steps]),
        peak_ready, discarded,
    )
