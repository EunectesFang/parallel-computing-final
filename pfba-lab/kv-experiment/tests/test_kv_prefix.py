"""CPU-only tests for shared-prefix KV reuse.

Run from ``pfba-lab/kv-experiment``:

    D:/MINICONDA/envs/py312/python.exe -m pytest tests -q

No downloads and no GPU: a tiny randomly initialized Qwen2 model and a
greedy character tokenizer are built in-process.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kv_prefix import (  # noqa: E402
    PrefixKVPredictor,
    clone_cache,
    common_token_prefix,
    new_dynamic_cache,
    repeat_cache,
)


# Explicit specials first; the greedy matcher ignores names in angle brackets.
VOCAB = [
    "<unk>",      # 0
    "<eos>",      # 1
    "a", "b", "c", "bc", "d", "e", "f", "g",  # 2..9
    "h", "i", "l", "m", "n", "r", "s", "t",  # 10..17
    "x", "y", "z", "G", "E",                  # 18..22
    " ", ":", "\n",                           # 23..25
]
EOS = VOCAB.index("<eos>")


class StubTokenizer:
    """Longest-match tokenizer with a total decode, no external files."""

    def __init__(self, vocab):
        self.vocab = list(vocab)
        self._ids = {token: i for i, token in enumerate(vocab)}
        self._matches = sorted(
            ((i, token) for i, token in enumerate(vocab) if not token.startswith("<")),
            key=lambda pair: -len(pair[1]),
        )
        self.eos_token_id = self._ids["<eos>"]
        self.pad_token_id = self.eos_token_id

    def encode(self, text, add_special_tokens=False):
        ids, pos = [], 0
        while pos < len(text):
            for token_id, token in self._matches:
                if text.startswith(token, pos):
                    ids.append(token_id)
                    pos += len(token)
                    break
            else:  # pragma: no cover - test inputs are all in vocab
                raise ValueError(f"cannot tokenize {text[pos:]!r}")
        return ids

    def decode(self, ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        pieces = []
        for token_id in ids:
            token = self.vocab[token_id] if 0 <= token_id < len(self.vocab) else f"<{token_id}>"
            if skip_special_tokens and token.startswith("<"):
                continue
            pieces.append(token)
        return "".join(pieces)


def tiny_model(tokenizer):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    config = Qwen2Config(
        vocab_size=len(tokenizer.vocab),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        tie_word_embeddings=False,
        bos_token_id=EOS,
        eos_token_id=EOS,
        pad_token_id=EOS,
    )
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(config)
    model.config._attn_implementation = "eager"
    # Double precision makes prefix-chunked matmuls agree to far below the
    # float32 logits that generate() samples from, so parity tests are stable.
    return model.double().eval()


# Module-level so both pytest and unittest see the fixtures (pytest does not
# call unittest's setUpModule for plain TestCase classes).
TOKENIZER = StubTokenizer(VOCAB)
MODEL = tiny_model(TOKENIZER)


def encode(text):
    return TOKENIZER.encode(text, add_special_tokens=False)


def reference_ids(prompt, branches, horizon, seed):
    """The ordinary ``generate(num_return_sequences=...)`` path."""
    input_ids = torch.tensor(encode(prompt), dtype=torch.long).unsqueeze(0)
    with torch.inference_mode(), torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(int(seed))
        output = MODEL.generate(
            input_ids=input_ids,
            max_new_tokens=horizon,
            do_sample=True,
            temperature=1.0,
            top_p=0.9,
            top_k=0,
            num_return_sequences=branches,
            eos_token_id=TOKENIZER.eos_token_id,
            pad_token_id=TOKENIZER.pad_token_id,
        )
    return output[:, input_ids.shape[1]:].tolist()


def full_cache(input_ids):
    cache = new_dynamic_cache(MODEL)
    with torch.inference_mode():
        MODEL(
            input_ids=torch.tensor(input_ids, dtype=torch.long).unsqueeze(0),
            past_key_values=cache,
            use_cache=True,
        )
    return cache


def cache_layers(cache):
    """(keys, values) per layer, for both 4.x and 5.x DynamicCache."""
    if hasattr(cache, "layers"):
        return [(layer.keys, layer.values) for layer in cache.layers]
    return list(zip(cache.key_cache, cache.value_cache))


def cache_values(cache):
    return [(keys.clone(), values.clone()) for keys, values in cache_layers(cache)]


def assert_cache_close(test, cache, expected):
    got_layers = cache_layers(cache)
    exp_layers = cache_layers(expected)
    test.assertEqual(len(got_layers), len(exp_layers))
    for (got_k, got_v), (exp_k, exp_v) in zip(got_layers, exp_layers):
        torch.testing.assert_close(got_k, exp_k, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(got_v, exp_v, atol=1e-5, rtol=1e-4)


class CommonTokenPrefixTests(unittest.TestCase):
    def test_counts_exact_token_overlap(self):
        self.assertEqual(common_token_prefix([3, 4, 5, 6], [3, 4, 9, 9]), 2)

    def test_disjoint_and_empty(self):
        self.assertEqual(common_token_prefix([1, 2], [3, 4]), 0)
        self.assertEqual(common_token_prefix([], [1, 2]), 0)
        self.assertEqual(common_token_prefix([1, 2], []), 0)

    def test_full_overlap(self):
        self.assertEqual(common_token_prefix([1, 2, 3], [1, 2, 3]), 3)


class CacheCopyTests(unittest.TestCase):
    def test_clone_is_independent(self):
        source = full_cache(encode("German: ab\nEnglish:"))
        cloned = clone_cache(source)
        cloned_keys = cache_layers(cloned)[0][0]
        source_keys = cache_layers(source)[0][0]
        torch.testing.assert_close(cloned_keys, source_keys)
        with torch.no_grad():
            cloned_keys[0, 0, 0, 0] += 1.0
        self.assertNotEqual(
            cloned_keys[0, 0, 0, 0].item(), source_keys[0, 0, 0, 0].item()
        )

    def test_repeat_duplicates_rows_without_aliasing(self):
        source = full_cache(encode("German: ab\nEnglish:"))
        repeated = repeat_cache(source, 3)
        repeated_keys = cache_layers(repeated)[0][0]
        source_keys = cache_layers(source)[0][0]
        self.assertEqual(repeated_keys.shape[0], 3)
        for row in range(3):
            torch.testing.assert_close(repeated_keys[row], source_keys[0])
        with torch.no_grad():
            repeated_keys[0, 0, 0, 0] += 1.0
        self.assertNotEqual(
            repeated_keys[0, 0, 0, 0].item(), repeated_keys[1, 0, 0, 0].item()
        )
        self.assertNotEqual(
            repeated_keys[0, 0, 0, 0].item(), source_keys[0, 0, 0, 0].item()
        )


class SuffixPrefillTests(unittest.TestCase):
    def test_cached_suffix_logits_match_full_forward(self):
        ids = encode("German: ab c\nEnglish:")
        split = 4
        cache = new_dynamic_cache(MODEL)
        with torch.inference_mode():
            first = MODEL(
                input_ids=torch.tensor(ids[:split]).unsqueeze(0),
                past_key_values=cache,
                use_cache=True,
            )
            second = MODEL(
                input_ids=torch.tensor(ids[split:]).unsqueeze(0),
                past_key_values=first.past_key_values,
                use_cache=True,
            )
            full = MODEL(
                input_ids=torch.tensor(ids).unsqueeze(0),
                use_cache=False,
            )
        torch.testing.assert_close(
            second.logits, full.logits[:, split:], atol=1e-5, rtol=1e-4
        )

    def test_predictor_prefix_cache_matches_full_recompute(self):
        prompt = "German: ab c\nEnglish:"
        predictor = PrefixKVPredictor(MODEL, TOKENIZER)
        predictor.sample(prompt, branches=2, horizon=2, seed=3)

        expected = full_cache(encode(prompt))
        self.assertEqual(predictor.cache_length, len(encode(prompt)))
        assert_cache_close(self, predictor._cache, expected)


class GenerationParityTests(unittest.TestCase):
    def test_cached_path_matches_ordinary_generate_exactly(self):
        prompt = "German: ab c\nEnglish:"
        predictor = PrefixKVPredictor(MODEL, TOKENIZER)
        predictor.sample(prompt, branches=2, horizon=2, seed=11)  # warm the cache
        predictor.sample(prompt, branches=3, horizon=4, seed=11)

        self.assertEqual(
            predictor.last_generated_ids,
            reference_ids(prompt, branches=3, horizon=4, seed=11),
        )

    def test_fresh_and_reused_paths_agree(self):
        old_prompt = "German: ab\nEnglish:"
        new_prompt = "German: ab cd\nEnglish:"

        reused = PrefixKVPredictor(MODEL, TOKENIZER)
        reused.sample(old_prompt, branches=2, horizon=2, seed=5)
        reused.sample(new_prompt, branches=2, horizon=3, seed=17)

        fresh = PrefixKVPredictor(MODEL, TOKENIZER)
        fresh.sample(new_prompt, branches=2, horizon=3, seed=17)

        self.assertEqual(reused.last_generated_ids, fresh.last_generated_ids)
        self.assertEqual(reused.last_generated_ids, reference_ids(new_prompt, 2, 3, 17))

    def test_same_seed_is_deterministic(self):
        prompt = "German: ab c\nEnglish:"
        predictor = PrefixKVPredictor(MODEL, TOKENIZER)
        first = predictor.sample(prompt, branches=2, horizon=3, seed=23)
        second = predictor.sample(prompt, branches=2, horizon=3, seed=23)
        self.assertEqual(first, second)
        self.assertEqual(predictor.last_generated_ids, reference_ids(prompt, 2, 3, 23))


class PrefixGrowthTests(unittest.TestCase):
    def test_growing_source_reuses_the_exact_token_prefix(self):
        old_prompt = "German: ab c\nEnglish:"
        new_prompt = "German: ab cd\nEnglish:"
        predictor = PrefixKVPredictor(MODEL, TOKENIZER)

        predictor.sample(old_prompt, branches=2, horizon=2, seed=7)
        predictor.sample(new_prompt, branches=2, horizon=2, seed=7)

        # The observed source grew, but the trailing "\nEnglish:" part was
        # rewritten, so reuse stops exactly at the end of the old source.
        old_source = "German: ab c"
        self.assertEqual(predictor.last_common_prefix, len(encode(old_source)))
        self.assertLess(predictor.last_common_prefix, len(encode(old_prompt)))
        self.assertEqual(predictor.cache_length, len(encode(new_prompt)))

        expected = full_cache(encode(new_prompt))
        assert_cache_close(self, predictor._cache, expected)

    def test_token_boundary_resegmentation_falls_back(self):
        # "ab" tokenizes as [a, b], "abc" as [a, bc]: the textual prefix
        # is not a token prefix, so only the exact token overlap may be reused.
        old_prompt = "German: ab\nEnglish:"
        new_prompt = "German: abc\nEnglish:"
        old_ids, new_ids = encode(old_prompt), encode(new_prompt)
        self.assertLess(common_token_prefix(old_ids, new_ids), len(old_ids))

        predictor = PrefixKVPredictor(MODEL, TOKENIZER)
        predictor.sample(old_prompt, branches=2, horizon=2, seed=9)
        predictor.sample(new_prompt, branches=2, horizon=2, seed=9)

        self.assertEqual(
            predictor.last_common_prefix, common_token_prefix(old_ids, new_ids)
        )
        self.assertEqual(predictor.last_generated_ids, reference_ids(new_prompt, 2, 2, 9))

        expected = full_cache(new_ids)
        assert_cache_close(self, predictor._cache, expected)

    def test_unrelated_prompt_recomputes_from_scratch(self):
        predictor = PrefixKVPredictor(MODEL, TOKENIZER)
        predictor.sample("German: ab\nEnglish:", branches=2, horizon=2, seed=3)
        predictor.sample("x y z", branches=2, horizon=2, seed=3)
        self.assertEqual(predictor.last_common_prefix, 0)
        self.assertEqual(predictor.last_generated_ids, reference_ids("x y z", 2, 2, 3))


class CacheIndependenceTests(unittest.TestCase):
    def test_branch_generation_never_mutates_the_prefix_cache(self):
        prompt = "German: ab c\nEnglish:"
        predictor = PrefixKVPredictor(MODEL, TOKENIZER)
        predictor.sample(prompt, branches=1, horizon=1, seed=2)  # establish cache

        before = cache_values(predictor._cache)
        predictor.sample(prompt, branches=4, horizon=3, seed=2)
        after = cache_values(predictor._cache)

        self.assertEqual(predictor.cache_length, len(encode(prompt)))
        self.assertEqual(cache_layers(predictor._cache)[0][0].shape[0], 1)
        for (before_k, before_v), (after_k, after_v) in zip(before, after):
            torch.testing.assert_close(after_k, before_k)
            torch.testing.assert_close(after_v, before_v)

    def test_shorter_prompt_crops_without_stale_tokens(self):
        predictor = PrefixKVPredictor(MODEL, TOKENIZER)
        predictor.sample("German: ab cd\nEnglish:", branches=2, horizon=2, seed=1)
        predictor.sample("German: ab\nEnglish:", branches=2, horizon=2, seed=1)
        self.assertEqual(predictor.cache_length, len(encode("German: ab\nEnglish:")))
        self.assertEqual(
            predictor.last_generated_ids,
            reference_ids("German: ab\nEnglish:", 2, 2, 1),
        )


if __name__ == "__main__":
    unittest.main()
