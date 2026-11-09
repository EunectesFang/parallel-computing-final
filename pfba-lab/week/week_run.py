#!/usr/bin/env python3
"""Second one-split run. The first holdout (hash bucket 0) is left untouched.

Train sentences are dev ids whose hash bucket is 2..9. Bucket 1 is the new
holdout. Bucket 0 was already reported and is neither trained nor evaluated.
Labeled steps still expand one (10, 10) sample. Steps in between do not run
the fixed (4, 6) teacher: the committed prefix stays where the reward argmax
left it, so states start from an empty prefix. The read reward is minus the
number of correct words the best write would have emitted. A binary head
decides read versus write, and a second head picks (B, H) only when writing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
import traceback
from pathlib import Path

import sys

_HERE = Path(__file__).resolve().parent
for _extra in (_HERE, _HERE.parent / "paired"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import run_pfba as pfba

LAMBDA = 1.0
BETA = 0.5
GAMMA = 0.25
ETA = 1.0
SEED = 13
MIN_START = 3
THRESHOLD = 0.6
SURFACE_FEATURES = (
    "observed_len", "observed_chars", "observed_punct", "observed_punct_ratio",
    "observed_ends_punct", "committed_words", "committed_chars", "lag_words",
    "last_word_len", "last_word_capital",
)
CONFIDENCE_FEATURES = ("mt_next_max_prob", "mt_next_entropy", "mt_prefix_logprob")
FEATURE_NAMES = SURFACE_FEATURES + CONFIDENCE_FEATURES
_PUNCT = set(".,;:!?\"'()[]{}") | set("，。！？；：、")


# Bucket 0 is the reported fold. Bucket 1 is the new holdout.
HOLDOUT_BUCKET = 1
EXCLUDED_BUCKETS = frozenset({0, 1})
READ_THRESHOLD = 0.5


def hash_bucket(sentence_id: str) -> int:
    digest = hashlib.sha256(f"13|{sentence_id}".encode()).digest()[:8]
    return int.from_bytes(digest, "big") % 10


def is_holdout(sentence_id: str) -> bool:
    return hash_bucket(sentence_id) == HOLDOUT_BUCKET


def is_train(sentence_id: str) -> bool:
    return hash_bucket(sentence_id) not in EXCLUDED_BUCKETS


def action_index(branches: int, horizon: int) -> int:
    """Index in the 101-vector. 0 is read. Writes are horizon-major, then branch."""
    if branches <= 0 or horizon <= 0:
        return 0
    return 1 + (horizon - 1) * 10 + (branches - 1)


def decode_action(index: int) -> tuple[int, int]:
    if index <= 0:
        return 0, 0
    index -= 1
    return index % 10 + 1, index // 10 + 1


def decision_positions(n_words: int, min_start: int = MIN_START) -> list[int]:
    if n_words <= min_start:
        return []
    return list(range(min_start, n_words))


def labeled_positions(positions: list[int], stride: int) -> list[int]:
    if stride < 1:
        raise ValueError("stride must be positive")
    return [pos for index, pos in enumerate(positions) if index % stride == 0]


def truncate_token_ids(token_ids: list[int], horizon: int, eos_id: int | None) -> list[int]:
    kept = []
    for token in token_ids:
        if eos_id is not None and token == eos_id:
            break
        kept.append(int(token))
        if len(kept) >= horizon:
            break
    return kept


def continuation_text(tokenizer, token_ids: list[int], horizon: int) -> str:
    kept = truncate_token_ids(token_ids, horizon, tokenizer.eos_token_id)
    text = tokenizer.decode(kept, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return text.split("\n")[0].strip()


def shuffle_rows(rows: list, seed: int) -> list:
    order = list(range(len(rows)))
    random.Random(int(seed)).shuffle(order)
    return [rows[index] for index in order]


def request_text(observed: str, continuation: str) -> str:
    continuation = continuation.strip()
    return observed if not continuation else f"{observed} {continuation}"


def surface_features(observed: str, committed: list[str]) -> dict[str, float]:
    words = observed.split()
    punct = sum(1 for ch in observed if ch in _PUNCT)
    last = words[-1] if words else ""
    values = {
        "observed_len": float(len(words)),
        "observed_chars": float(len(observed)),
        "observed_punct": float(punct),
        "observed_punct_ratio": float(punct / max(1, len(observed))),
        "observed_ends_punct": float(bool(observed) and observed[-1] in _PUNCT),
        "committed_words": float(len(committed)),
        "committed_chars": float(len(" ".join(committed))),
        "lag_words": float(len(words) - len(committed)),
        "last_word_len": float(len(last)),
        "last_word_capital": float(bool(last[:1].isupper())),
    }
    return {name: values[name] for name in SURFACE_FEATURES}


def step_features(observed: str, committed: list[str], confidence: dict[str, float]) -> dict[str, float]:
    values = surface_features(observed, committed)
    values.update(confidence)
    return {name: float(values[name]) for name in FEATURE_NAMES}


def softmax_confidence(logits: list[float]) -> tuple[float, float]:
    """Max probability and entropy of one next-token distribution."""
    if not logits:
        return 0.0, 0.0
    peak = max(logits)
    weights = [math.exp(value - peak) for value in logits]
    total = sum(weights)
    max_prob = max(weights) / total
    entropy = 0.0
    for weight in weights:
        probability = weight / total
        entropy -= probability * math.log(max(probability, 1e-12))
    return float(max_prob), float(entropy)


def mean_or_zero(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def translation_confidence(translator, observed: str, committed: list[str]) -> dict[str, float]:
    """How sure the translator is, before this step chooses an action.

    One forward pass on the heard German and the English already said.
    The next-token max probability and entropy describe the next English word.
    The prefix score is the mean log probability of the words already said.
    Nothing said yet gives that score as 0.
    """
    import torch
    prefix = " ".join(committed)
    prefix_logprobs: list[float] = []
    with torch.inference_mode():
        if hasattr(translator, "start"):
            batch = translator.tok(
                observed, return_tensors="pt", truncation=True, max_length=512).to(translator.device)
            prefix_ids = translator.tok(
                text_target=prefix, add_special_tokens=False)["input_ids"] if prefix else []
            decoder_ids = torch.tensor([[translator.start, *prefix_ids]], device=translator.device)
            logits = translator.model(**batch, decoder_input_ids=decoder_ids).logits[0].float()
            next_logits = logits[-1]
            if prefix_ids:
                token_log = torch.log_softmax(logits[:-1], dim=-1)
                for index, token_id in enumerate(prefix_ids):
                    prefix_logprobs.append(float(token_log[index, int(token_id)]))
        else:
            full = pfba.MT_TEMPLATE.format(src=observed, prefix=prefix)
            bare = pfba.MT_TEMPLATE.format(src=observed, prefix="")
            full_ids = translator.tok(full, truncation=True, max_length=2048)["input_ids"]
            bare_ids = translator.tok(bare, truncation=True, max_length=2048)["input_ids"]
            batch = torch.tensor([full_ids], device=translator.device)
            logits = translator.model(input_ids=batch).logits[0].float()
            next_logits = logits[-1]
            if len(full_ids) > len(bare_ids) and full_ids[:len(bare_ids)] == bare_ids:
                token_log = torch.log_softmax(logits, dim=-1)
                for offset, token_id in enumerate(full_ids[len(bare_ids):]):
                    position = len(bare_ids) + offset - 1
                    if position >= 0:
                        prefix_logprobs.append(float(token_log[position, int(token_id)]))
        next_values = next_logits.detach().float().cpu().tolist()
    max_prob, entropy = softmax_confidence(next_values)
    return {
        "mt_next_max_prob": max_prob,
        "mt_next_entropy": entropy,
        "mt_prefix_logprob": mean_or_zero(prefix_logprobs),
    }


def action_cost(pred_s: float, mt_s: list[float], branches: int, horizon: int) -> float:
    return float(pred_s) * (branches * horizon) / 100.0 + float(mt_s[horizon - 1]) * (branches / 10.0)


def state_rewards(votes: list[dict], pred_s: float, mt_s: list[float], median_46: float) -> list[float]:
    """Read costs the correct words the best write would have emitted.

    The best write is the first action that attains the highest quality.
    Its matched count is m. Read reward is -m, so m = 0 makes waiting free
    and does not add a bonus. A write reward is quality minus the normalized
    time cost. Quality stays matched - λ * incorrect.
    """
    qualities = [float(item["matched"]) - LAMBDA * float(item["incorrect"]) for item in votes]
    scale = median_46 if median_46 > 0 else 1.0
    if not qualities:
        return [0.0]
    best = max(range(len(qualities)), key=lambda i: (qualities[i], -i))
    rewards = [-float(votes[best]["matched"])]
    for item, quality in zip(votes, qualities):
        cost = action_cost(pred_s, mt_s, int(item["b"]), int(item["h"]))
        rewards.append(quality - BETA * cost / scale)
    return rewards


def reward_winner(rewards: list[float]) -> int:
    return max(range(len(rewards)), key=lambda i: (rewards[i], -i))


def median(values: list[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        return 1.0
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return float(0.5 * (ordered[mid - 1] + ordered[mid]))


def sample_token_rows(predictor, prefix: str, branches: int, horizon: int, seed: int) -> list[list[int]]:
    torch = pfba._torch()
    inputs = predictor.tok(prefix, return_tensors="pt", add_special_tokens=False).to(predictor.device)
    device = torch.device(predictor.device)
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.inference_mode(), torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(int(seed))
        if devices:
            with torch.cuda.device(devices[0]):
                torch.cuda.manual_seed(int(seed))
        output = predictor.model.generate(
            **inputs, max_new_tokens=horizon, do_sample=True, temperature=1.0,
            top_p=0.9, top_k=0, num_return_sequences=branches,
            eos_token_id=predictor.tok.eos_token_id, pad_token_id=predictor.tok.pad_token_id)
    width = inputs["input_ids"].shape[1]
    return [row.tolist() for row in output[:, width:]]


def _sync(predictor, translator):
    pfba._sync(predictor.device)
    if translator.device != predictor.device:
        pfba._sync(translator.device)


def timed(predictor, translator, fn):
    _sync(predictor, translator)
    start = time.perf_counter()
    value = fn()
    _sync(predictor, translator)
    return value, max(0.0, time.perf_counter() - start)


def advance_fixed(predictor, translator, observed: str, committed: list[str], seed: int):
    continuations, pred_s = timed(
        predictor, translator, lambda: predictor.sample(observed, 4, 6, seed))
    requests = [request_text(observed, text) for text in continuations] or [observed]
    outputs, mt_s = timed(
        predictor, translator, lambda: translator.translate_batch(requests, " ".join(committed)))
    selected = pfba.upstream_ralcp(list(outputs), THRESHOLD)
    return tuple(selected.split()) if selected else (), pred_s, mt_s


def label_step(predictor, translator, observed: str, committed: list[str], source: str, seed: int) -> dict:
    rows, pred_s = timed(
        predictor, translator, lambda: sample_token_rows(predictor, observed, 10, 10, seed))
    rows = shuffle_rows(rows, seed)
    oracle, oracle_s = timed(
        predictor, translator,
        lambda: translator.translate_batch([source], " ".join(committed)))
    oracle_words = tuple((oracle[0] if oracle else "").split())
    votes = []
    mt_s = []
    for horizon in range(1, 11):
        requests = [request_text(observed, continuation_text(predictor.tok, row, horizon)) for row in rows]
        outputs, seconds = timed(
            predictor, translator,
            lambda requests=requests: translator.translate_batch(requests, " ".join(committed)))
        mt_s.append(seconds)
        for branches in range(1, 11):
            selected = pfba.upstream_ralcp(list(outputs[:branches]), THRESHOLD)
            words = tuple(selected.split()) if selected else ()
            matched = 0
            for left, right in zip(words, oracle_words):
                if left != right:
                    break
                matched += 1
            votes.append({
                "b": branches, "h": horizon, "matched": matched,
                "incorrect": len(words) - matched, "words": words,
            })
    return {
        "votes": votes, "pred_s": pred_s, "mt_s": mt_s, "oracle_s": oracle_s,
        "features": step_features(observed, committed, translation_confidence(translator, observed, committed)),
    }


def load_done(path: Path) -> set[str]:
    done = set()
    if not path.is_file():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            done.add(json.loads(line)["sentence_id"])
    return done


def read_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def new_policy():
    import torch

    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Sequential(
                torch.nn.Linear(len(FEATURE_NAMES), 32), torch.nn.Tanh(),
                torch.nn.Linear(32, 16), torch.nn.Tanh())
            self.branch = torch.nn.Linear(16, 10)
            self.horizon = torch.nn.Linear(16, 10)
            self.interact = torch.nn.Linear(16, 100)
            self.read = torch.nn.Linear(16, 1)

        def forward(self, features):
            hidden = self.backbone(features)
            # Horizon is the row, branch is the column: flat (h - 1) * 10 + (b - 1).
            grid = self.horizon(hidden).unsqueeze(2) + self.branch(hidden).unsqueeze(1)
            grid = (grid + self.interact(hidden).reshape(-1, 10, 10)).reshape(-1, 100)
            return self.read(hidden).reshape(-1), grid

    return Policy()


def train_policy(rows: list[dict], output: Path) -> dict:
    import torch
    costs = []
    for row in rows:
        for step in row["steps"]:
            costs.append(action_cost(step["pred_s"], step["mt_s"], 4, 6))
    scale = median(costs)
    xs, read_y, write_y, early = [], [], [], []
    for row in rows:
        for step in row["steps"]:
            rewards = state_rewards(step["votes"], step["pred_s"], step["mt_s"], scale)
            winner = reward_winner(rewards)
            xs.append([step["features"][name] for name in FEATURE_NAMES])
            read_y.append(1.0 if winner == 0 else 0.0)
            write_y.append(-1 if winner == 0 else winner - 1)
            early.append(step["features"]["observed_len"] <= 6)
    x = torch.tensor(xs, dtype=torch.float32)
    read_target = torch.tensor(read_y, dtype=torch.float32)
    write_target = torch.tensor(write_y, dtype=torch.long)
    early_flag = torch.tensor(early, dtype=torch.bool)
    mean = x.mean(dim=0)
    std = x.std(dim=0).clamp_min(1e-6)
    x = (x - mean) / std

    torch.manual_seed(SEED)
    model = new_policy()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    read_loss_fn = torch.nn.BCEWithLogitsLoss()
    write_loss_fn = torch.nn.CrossEntropyLoss()
    model.train()
    for _ in range(8):
        for start in range(0, len(x), 256):
            batch_x = x[start:start + 256]
            batch_read = read_target[start:start + 256]
            batch_write = write_target[start:start + 256]
            optimizer.zero_grad()
            read_logit, grid = model(batch_x)
            loss = read_loss_fn(read_logit, batch_read)
            mask = batch_write >= 0
            if bool(mask.any()):
                loss = loss + write_loss_fn(grid[mask], batch_write[mask])
            loss.backward()
            optimizer.step()
    payload = {
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "mean": mean.tolist(), "std": std.tolist(), "median_46": scale,
        "lambda": LAMBDA, "beta": BETA, "read_threshold": READ_THRESHOLD,
        "reward": "read=-matched_of_best_quality",
    }
    torch.save(payload, output)
    model.eval()
    policy = {"model": model, "mean": mean, "std": std}
    with torch.inference_mode():
        read_logit, _grid = model(x)
        predicted_read = torch.sigmoid(read_logit) >= READ_THRESHOLD
    diagnostic = {
        "n": int(read_target.shape[0]),
        "label_read_rate": float(read_target.mean()),
        "predicted_read_rate": float(predicted_read.float().mean()),
        "early_label_read_rate": float(read_target[early_flag].mean()) if bool(early_flag.any()) else None,
        "early_predicted_read_rate": float(predicted_read[early_flag].float().mean()) if bool(early_flag.any()) else None,
    }
    return {"model": model, "mean": mean, "std": std, "diagnostic": diagnostic, "policy": policy}


def load_policy(path: Path):
    import torch
    payload = torch.load(path, map_location="cpu")
    model = new_policy()
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return {
        "model": model,
        "mean": torch.tensor(payload["mean"], dtype=torch.float32),
        "std": torch.tensor(payload["std"], dtype=torch.float32),
    }


def choose_action(policy: dict, features: dict[str, float]) -> tuple[int, int]:
    import torch
    values = torch.tensor([[features[name] for name in FEATURE_NAMES]], dtype=torch.float32)
    values = (values - policy["mean"]) / policy["std"]
    with torch.inference_mode():
        read_logit, grid = policy["model"](values)
        if float(torch.sigmoid(read_logit)) >= READ_THRESHOLD:
            return 0, 0
        return decode_action(int(torch.argmax(grid, dim=1)) + 1)


def rollout(sentence, predictor, translator, policy, early_stop_k: int) -> dict:
    words = sentence.source.split()
    n = len(words)
    committed: list[str] = []
    delays: list[int] = []
    virtual_s = 0.0
    histogram: dict[str, int] = {}
    read_pos = min(MIN_START, n)
    while read_pos < n:
        observed = " ".join(words[:read_pos])
        virtual_s = max(virtual_s, (read_pos - 1) * 0.3)
        if policy is None:
            branches, horizon = 4, 6
        else:
            confidence, probe_s = timed(
                predictor, translator,
                lambda: translation_confidence(translator, observed, committed))
            virtual_s += probe_s
            branches, horizon = choose_action(policy, step_features(observed, committed, confidence))
        label = "read" if branches == 0 or horizon == 0 else f"b{branches}h{horizon}"
        histogram[label] = histogram.get(label, 0) + 1
        if branches and horizon:
            seed = pfba.derive_step_seed(SEED, "dev", sentence.sentence_id, read_pos)
            continuations, pred_s = timed(
                predictor, translator,
                lambda: predictor.sample(observed, branches, horizon, seed))
            requests = [request_text(observed, text) for text in continuations] or [observed]
            outputs, mt_s = timed(
                predictor, translator,
                lambda: translator.translate_batch(
                    requests, " ".join(committed), final=False,
                    early_stop_k=early_stop_k, threshold=THRESHOLD))
            selected = pfba.upstream_ralcp(list(outputs), THRESHOLD)
            words_out = tuple(selected.split()) if selected else ()
            virtual_s += pred_s + mt_s
            committed.extend(words_out)
            delays.extend([read_pos] * len(words_out))
        read_pos += 1
    outputs, mt_s = timed(
        predictor, translator,
        lambda: translator.translate_batch([sentence.source], " ".join(committed), final=True))
    virtual_s = max(virtual_s, (n - 1) * 0.3) + mt_s
    tail = tuple((outputs[0] if outputs else "").split())
    committed.extend(tail)
    delays.extend([n] * len(tail))
    return {
        "sentence_id": sentence.sentence_id,
        "hypothesis": " ".join(committed),
        "reference": sentence.reference,
        "source": sentence.source,
        "delays": delays,
        "completion_s": virtual_s,
        "histogram": histogram,
        "laal": pfba.laal_score(delays, n, len(sentence.reference.split()), len(committed)),
    }


def summarize(traces: list[dict]) -> dict:
    bleu = pfba.corpus_bleu([item["hypothesis"] for item in traces],
                            [item["reference"] for item in traces])
    laals = [item["laal"] for item in traces if item["laal"] is not None]
    reads = actions = 0
    for item in traces:
        for label, count in (item.get("histogram") or {}).items():
            actions += count
            reads += count if label == "read" else 0
    return {
        "n": len(traces),
        "bleu": bleu.get("score"),
        "laal": sum(laals) / len(laals) if laals else None,
        "mean_completion_s": sum(item["completion_s"] for item in traces) / max(1, len(traces)),
        "read_fraction": reads / actions if actions else None,
    }


def label_split(sentences, predictor, translator, path: Path, stride: int, deadline: float) -> int:
    done = load_done(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a", encoding="utf-8")
    started = time.perf_counter()
    fresh = 0
    scale_costs: list[float] = []
    train = [sentence for sentence in sentences if is_train(sentence.sentence_id)]
    try:
        for sentence in train:
            if sentence.sentence_id in done:
                continue
            if time.perf_counter() > deadline:
                print("[week] labeling stopped for the session deadline", flush=True)
                break
            words = sentence.source.split()
            positions = decision_positions(len(words))
            keep = set(labeled_positions(positions, stride))
            committed: list[str] = []
            steps = []
            for read_pos in positions:
                observed = " ".join(words[:read_pos])
                if read_pos not in keep:
                    continue
                seed = pfba.derive_step_seed(SEED, "dev", sentence.sentence_id, read_pos)
                step = label_step(predictor, translator, observed, committed, sentence.source, seed)
                scale_costs.append(action_cost(step["pred_s"], step["mt_s"], 4, 6))
                rewards = state_rewards(step["votes"], step["pred_s"], step["mt_s"], median(scale_costs))
                winner = reward_winner(rewards)
                new_words = () if winner == 0 else tuple(step["votes"][winner - 1]["words"])
                for vote in step["votes"]:
                    vote.pop("words", None)
                committed.extend(new_words)
                step["read_pos"] = read_pos
                steps.append(step)
            handle.write(json.dumps({
                "sentence_id": sentence.sentence_id, "stride": stride, "steps": steps,
            }, ensure_ascii=False) + "\n")
            handle.flush()
            fresh += 1
            if fresh == 200:
                elapsed = time.perf_counter() - started
                projected = elapsed / 200 * len(train)
                print(f"[week] 200 sentences in {elapsed:.0f}s; projected {projected / 3600:.2f}h", flush=True)
                if projected > 8 * 3600 and stride == 4:
                    stride = 8
                    print("[week] switching the remaining sentences to every 8th step", flush=True)
            if fresh % 25 == 0:
                print(f"[week] labeled {fresh} new sentences", flush=True)
    finally:
        handle.close()
    return fresh


def run(args) -> None:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    sentences = pfba.load_split("dev", 0)
    holdout = [sentence for sentence in sentences if is_holdout(sentence.sentence_id)]
    train = [sentence for sentence in sentences if is_train(sentence.sentence_id)]
    print(f"[week] train={len(train)} holdout={len(holdout)} excluded_bucket0", flush=True)
    import torch
    predictor_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    translator_device = "cuda:1" if torch.cuda.device_count() >= 2 else predictor_device
    cfg = pfba.StreamConfig(local_files_only=args.local_files_only)
    predictor = pfba.QwenPredictor(pfba.PREDICTORS["small"], predictor_device, "auto",
                                   args.local_files_only, False)
    predictor.load()
    translator = pfba.load_translator("small", pfba.TRANSLATORS["small"], translator_device, cfg)
    deadline = time.perf_counter() + args.max_hours * 3600
    label_split(sentences, predictor, translator, output / "labels.jsonl", args.stride, deadline)
    rows = [row for row in read_rows(output / "labels.jsonl")
            if is_train(row["sentence_id"]) and row.get("steps")]
    if not rows:
        raise RuntimeError("no training rows were written")
    trained = train_policy(rows, output / "policy.pt")
    policy = trained["policy"]
    print("[week] train diagnostic " + json.dumps(trained["diagnostic"], ensure_ascii=False), flush=True)
    small_holdout = holdout
    learned = [rollout(sentence, predictor, translator, policy, 0) for sentence in small_holdout]
    fixed = [rollout(sentence, predictor, translator, None, 0) for sentence in small_holdout]
    summary = {
        "train_sentences": len(train),
        "holdout_sentences": len(holdout),
        "labeled_sentences": len(rows),
        "small": {"learned": summarize(learned), "fixed_4_6": summarize(fixed)},
        "weights": {
            "lambda": LAMBDA, "beta": BETA, "read_threshold": READ_THRESHOLD,
            "gamma_not_used": GAMMA, "eta_not_used": ETA,
        },
        "reward": "read=-matched_of_best_quality; write=quality-beta*normalized_cost",
        "split": {"holdout_bucket": HOLDOUT_BUCKET, "excluded_buckets": sorted(EXCLUDED_BUCKETS)},
        "trajectory": "unlabeled steps wait; labeled steps commit the reward argmax",
        "train_diagnostic": trained["diagnostic"],
        "early_stop_on_small": False,
        "cross_step_kv": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[week] small holdout " + json.dumps(summary["small"], ensure_ascii=False), flush=True)
    del predictor, translator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    remaining = deadline - time.perf_counter()
    if args.large_limit and remaining > 2.0 * 3600:
        print("[week] starting the 7B holdout slice", flush=True)
        try:
            predictor = pfba.QwenPredictor(pfba.PREDICTORS["large"], predictor_device, "float16",
                                           args.local_files_only, False)
            predictor.load()
            try:
                from cpu_embedding import offload_input_embedding
                offload_input_embedding(predictor.model)
            except Exception as exc:
                print(f"[week] embedding offload skipped: {exc}", flush=True)
            translator = pfba.load_translator("large", pfba.TRANSLATORS["large"], translator_device, cfg)
            chosen = holdout[:args.large_limit]
            summary["large"] = {
                "n": len(chosen),
                "early_stop_k": 4,
                "learned": summarize([rollout(sentence, predictor, translator, policy, 4) for sentence in chosen]),
                "fixed_4_6": summarize([rollout(sentence, predictor, translator, None, 4) for sentence in chosen]),
            }
        except Exception:
            traceback.print_exc()
            summary["large"] = {"status": "failed"}
    else:
        summary["large"] = {"status": "skipped", "remaining_s": max(0.0, remaining)}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[week] large slice " + json.dumps(summary.get("large"), ensure_ascii=False), flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="/kaggle/working/week")
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--max-hours", type=float, default=10.5)
    parser.add_argument("--large-limit", type=int, default=64)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        run(args)
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
