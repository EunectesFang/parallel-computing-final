"""Deterministic CPU backends for invariant tests and smoke runs.

These are *not* MT or LLM substitutes for scored experiments.  They model the
interfaces exactly so policy semantics (majority, RALCP, forced prefix, flush)
can be tested without downloads or a GPU.
"""
from __future__ import annotations

from typing import Callable, Iterable, Sequence


def make_vocab(words: Iterable[str]) -> dict[str, int]:
    return {word: idx for idx, word in enumerate(sorted(set(words)))}


def _lcp_len(a: Sequence[int], b: Sequence[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


class MockLLM:
    """Table-driven continuation sampler.

    ``table`` maps a whitespace-normalized partial source to a list of
    continuations.  The list is padded/truncated to the requested n so tests
    can specify exact agreement rates.
    """

    def __init__(
        self,
        table: dict[str, list[str]] | None = None,
        default: Callable[[str, int], list[str]] | None = None,
    ) -> None:
        self.table = table or {}
        self.default = default
        self.calls: list[dict] = []

    def sample(
        self,
        partial_source: str,
        n: int,
        length: int,
        top_k: int,
        *,
        seed: int | None = None,
    ) -> list[str]:
        key = " ".join(partial_source.split())
        self.calls.append({"partial": key, "n": n, "length": length, "top_k": top_k})
        if key in self.table:
            values = list(self.table[key])
        elif self.default is not None:
            values = list(self.default(key, n))
        else:
            values = [""] * n
        if not values:
            values = [""]
        while len(values) < n:
            values.append(values[-1])
        values = values[:n]
        # Continuation horizon is measured in MT/source tokens; mock strings
        # are word sequences, so truncate by words.
        return [" ".join(v.split()[:length]) for v in values]


class MockMT:
    """Deterministic prefix-forced MT backend.

    Target units are vocabulary ids.  The base translation is
    ``mapping[source_word]`` for every word in the (partial source +
    prediction) string.  Forced-prefix behavior: if the base translation
    agrees with the committed prefix for m units, only ``base[m:]`` is
    returned; otherwise the whole base translation is appended after the
    committed prefix, never rewriting it.
    """

    word_level = True

    def __init__(
        self,
        mapping: dict[str, str] | None = None,
        vocab: dict[str, int] | None = None,
        variant_fn: Callable[[str, tuple[int, ...], int], list[list[int]]] | None = None,
    ) -> None:
        self.mapping = mapping or {}
        if vocab is None:
            words = set(self.mapping.keys()) | set(self.mapping.values())
            vocab = make_vocab(words)
        self.vocab = vocab
        self.id_to_token = {idx: word for word, idx in vocab.items()}
        self.variant_fn = variant_fn
        self.calls: list[dict] = []

    def _target_ids(self, source_text: str) -> list[int]:
        ids: list[int] = []
        for word in source_text.split():
            mapped = self.mapping.get(word, word)
            if mapped in self.vocab:
                ids.append(self.vocab[mapped])
        return ids

    def translate(
        self,
        source_text: str,
        committed_ids: Sequence[int] = (),
        num_sequences: int = 1,
        num_beams: int = 1,
        max_new_tokens: int = 64,
    ) -> list[list[int]]:
        count = max(1, num_sequences, num_beams)
        committed = [int(x) for x in committed_ids]
        self.calls.append(
            {
                "source": source_text,
                "committed_ids": list(committed),
                "num_sequences": num_sequences,
                "num_beams": num_beams,
            }
        )

        if self.variant_fn is not None:
            continuations = [list(c) for c in self.variant_fn(source_text, tuple(committed), count)]
        else:
            base = self._target_ids(source_text)
            match = _lcp_len(committed, base)
            continuations = [list(base[match:])]

        if not continuations:
            continuations = [[]]
        while len(continuations) < count:
            continuations.append(list(continuations[-1]))
        continuations = continuations[:count]
        continuations = [c[:max_new_tokens] for c in continuations]
        return continuations

    def decode(self, ids: Sequence[int]) -> str:
        return " ".join(self.id_to_token[i] for i in ids if i in self.id_to_token)

    def encode(self, text: str) -> list[int]:
        """Encode already-translated target text (no source mapping)."""
        return [self.vocab[word] for word in text.split() if word in self.vocab]

    def token_strings(self, ids: Sequence[int]) -> list[str]:
        return [self.id_to_token[i] for i in ids if i in self.id_to_token]

    def sync(self) -> None:
        return None
