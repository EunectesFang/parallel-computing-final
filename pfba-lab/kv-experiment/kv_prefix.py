"""Exact shared-prefix KV reuse for Qwen2.5 branch sampling.

The streaming predictor sees a prompt grow from step to step.  If the new
prompt's token IDs begin with the previous prompt's token IDs, the cached KV
for those positions is causally identical, so it can be reused.  Any mismatch
-- including a BPE merge at the boundary that changes the tokenization of a
textually shared suffix -- ends the reusable prefix; the remaining suffix is
prefilled from scratch.

For ``B`` sampled continuations the cached prefix is copied, repeated along
the batch dimension, and passed to ``generate`` together with the full prompt
``input_ids`` and a full-length attention mask.  ``generate`` (via the model's
``prepare_inputs_for_generation`` / generation prefill) slices the input down
to the uncovered suffix, so positions and the causal mask line up with the
ordinary ``generate`` path.  At most ``prompt_len - 1`` cache tokens are kept
so that at least one prompt token is always left to prefill.  The cache is
always copied before repeating, so speculative branch generation can never
mutate the prefix retained for the next streaming step.

Integration (model/tokenizer already loaded by the caller)::

    predictor = PrefixKVPredictor(model, tokenizer)
    continuations = predictor.sample(prompt, branches=B, horizon=H, seed=seed)

``sample`` keeps the signature of ``paired/run_pfba.py::QwenPredictor.sample``.
"""

from __future__ import annotations

import copy
from typing import Sequence

import torch
from transformers import DynamicCache


def common_token_prefix(left: Sequence[int], right: Sequence[int]) -> int:
    """Length of the exact token-ID prefix shared by two 1-D sequences."""
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


def new_dynamic_cache(model) -> DynamicCache:
    """Create a DynamicCache, tolerating pre-config-argument transformers."""
    try:
        return DynamicCache(config=model.config)
    except (TypeError, AttributeError):
        # Older 4.x DynamicCache has no config argument/layer inference;
        # Qwen2.5 has no sliding layers, so a plain cache is equivalent.
        return DynamicCache()


def clone_cache(cache: DynamicCache) -> DynamicCache:
    """Deep-copy a dynamic cache (tensor storage is not shared)."""
    return copy.deepcopy(cache)


def repeat_cache(cache: DynamicCache, repeats: int) -> DynamicCache:
    """Tile each batch row into independent storage in a single copy."""
    if hasattr(cache, "layers"):
        repeated = copy.copy(cache)
        repeated.layers = [copy.copy(layer) for layer in cache.layers]
    else:
        repeated = clone_cache(cache)
    repeated.batch_repeat_interleave(int(repeats))
    return repeated


class PrefixKVPredictor:
    """Qwen2.5 sampler that reuses KV for exact prompt-token prefixes."""

    def __init__(self, model, tokenizer):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self._cache: DynamicCache | None = None
        self._prefix_ids: list[int] | None = None
        self.last_common_prefix = 0
        self.last_prefilled = 0
        self.last_generated_ids: list[list[int]] = []
        eos = getattr(tokenizer, "eos_token_id", None)
        pad = getattr(tokenizer, "pad_token_id", None)
        self.eos_token_id = eos
        self.pad_token_id = eos if pad is None else pad

    @property
    def cache_length(self) -> int:
        """Number of prompt tokens currently held in the retained cache."""
        return 0 if self._cache is None else self._cache.get_seq_length()

    def _encode(self, prompt: str) -> list[int]:
        ids = list(self.tokenizer.encode(prompt, add_special_tokens=False))
        if not ids:
            raise ValueError("prompt tokenized to an empty sequence")
        return ids

    def _update_prefix(self, ids: list[int]) -> DynamicCache:
        """Return a cache covering ``ids``, reusing the exact token overlap."""
        reused = 0
        if self._cache is not None and self._prefix_ids is not None:
            reused = common_token_prefix(self._prefix_ids, ids)

        if reused:
            cache = clone_cache(self._cache)
            cache.crop(reused)
        else:
            cache = new_dynamic_cache(self.model)

        suffix = ids[reused:]
        if suffix:
            suffix_ids = torch.tensor(
                suffix, dtype=torch.long, device=self.device
            ).unsqueeze(0)
            with torch.inference_mode():
                output = self.model(
                    input_ids=suffix_ids,
                    past_key_values=cache,
                    use_cache=True,
                )
            cache = output.past_key_values

        self._cache = cache
        self._prefix_ids = list(ids)
        self.last_common_prefix = reused
        self.last_prefilled = len(suffix)
        return cache

    def sample(self, prefix: str, branches: int, horizon: int, seed: int) -> list[str]:
        """Sample ``branches`` continuations of at most ``horizon`` tokens."""
        ids = self._encode(prefix)
        cache = self._update_prefix(ids)

        branches = int(branches)
        if branches <= 0:
            self.last_generated_ids = []
            return []

        prompt_ids = torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)
        prompt_len = prompt_ids.shape[1]

        # Generation must be given at least one uncovered prompt token, so keep
        # at most prompt_len - 1 cached tokens.  The prefix cache retained for
        # the next streaming step is untouched.
        branch_cache = repeat_cache(cache, branches)
        if branch_cache.get_seq_length() >= prompt_len:
            branch_cache.crop(prompt_len - 1)

        input_ids = prompt_ids.repeat_interleave(branches, dim=0)
        attention_mask = torch.ones_like(input_ids)

        device = self.device
        devices = [device.index or 0] if device.type == "cuda" else []
        with torch.inference_mode(), torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(int(seed))
            if devices:
                with torch.cuda.device(devices[0]):
                    torch.cuda.manual_seed(int(seed))
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=branch_cache,
                max_new_tokens=int(horizon),
                do_sample=True,
                temperature=1.0,
                top_p=0.9,
                top_k=0,
                eos_token_id=self.eos_token_id,
                pad_token_id=self.pad_token_id,
            )

        generated = output[:, prompt_len:]
        self.last_generated_ids = generated.tolist()
        return [
            self.tokenizer.decode(
                row,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).split("\n")[0].strip()
            for row in self.last_generated_ids
        ]
