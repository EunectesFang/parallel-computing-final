"""Hugging Face backends for LLaMA-style predictors and NMT decoders.

Exact paper artifacts are LLaMA-2-7B base plus the authors' Transformer-Big
checkpoint.  The latter is not available, so ``--mt-model`` is expected to
point at a substitution unless the author checkpoint is supplied locally.
No model is downloaded at import time.
"""
from __future__ import annotations

from typing import Sequence


def _torch():
    import torch  # type: ignore

    return torch


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    torch = _torch()
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_dtype(dtype: str, device: str):
    torch = _torch()
    if dtype == "auto":
        return torch.float16 if device.startswith("cuda") else torch.float32
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype]


class HFLLM:
    """Causal-LM next-continuation sampler (paper: LLaMA-2-7B base)."""

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        dtype: str = "auto",
        load_in_4bit: bool = False,
        prompt_template: str = "{source}",
        max_context: int = 2048,
    ) -> None:
        torch = _torch()
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        self.model_name = model_name
        self.prompt_template = prompt_template
        self.max_context = max_context
        self.device = _resolve_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs = {}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig  # type: ignore

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            kwargs["device_map"] = "auto"
        else:
            kwargs["torch_dtype"] = _resolve_dtype(dtype, self.device)
            if self.device == "cuda":
                kwargs["device_map"] = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        if not load_in_4bit and "device_map" not in kwargs:
            self.model.to(self.device)
        self.model.eval()
        self.torch = torch

    def sample(
        self,
        partial_source: str,
        n: int,
        length: int,
        top_k: int,
        *,
        seed: int | None = None,
    ) -> list[str]:
        prompt = self.prompt_template.format(source=partial_source)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_context,
        )
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        if seed is not None:
            self.torch.manual_seed(seed)
        with self.torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=length,
                do_sample=True,
                top_k=top_k,
                num_return_sequences=n,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        prompt_len = inputs["input_ids"].shape[1]
        generated = outputs[:, prompt_len:]
        return [
            self.tokenizer.decode(row, skip_special_tokens=True).strip()
            for row in generated
        ]

    def sync(self) -> None:
        if self.torch.cuda.is_available():
            self.torch.cuda.synchronize()


class HFMT:
    """Prefix-forced MT decoder.

    ``mt_arch='seq2seq'`` (default) uses ``decoder_input_ids`` so the committed
    target prefix is forced exactly at the token level.  ``mt_arch='causal'``
    is a documented substitution for instruction-tuned decoder-only MT models
    (e.g. TowerInstruct); it cannot force tokens through a decoder API.
    """

    word_level = False

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        dtype: str = "auto",
        mt_arch: str = "seq2seq",
        max_context: int = 1024,
        prompt_template: str = "Translate German to English:\n{source}\nEnglish:",
    ) -> None:
        torch = _torch()
        from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore

        self.model_name = model_name
        self.mt_arch = mt_arch
        self.max_context = max_context
        self.prompt_template = prompt_template
        self.device = _resolve_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs = {"torch_dtype": _resolve_dtype(dtype, self.device)}
        if self.device == "cuda":
            kwargs["device_map"] = "auto"
        if mt_arch == "seq2seq":
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **kwargs)
        elif mt_arch == "causal":
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        else:
            raise ValueError("mt_arch must be 'seq2seq' or 'causal'")
        if "device_map" not in kwargs:
            self.model.to(self.device)
        self.model.eval()
        self.torch = torch
        self.decoder_start_token_id = (
            getattr(self.model.config, "decoder_start_token_id", None)
            or self.tokenizer.pad_token_id
            or self.tokenizer.bos_token_id
        )

    def _clean(self, ids: Sequence[int]) -> list[int]:
        eos = self.tokenizer.eos_token_id
        pad = self.tokenizer.pad_token_id
        out = [int(x) for x in ids if int(x) not in (eos, pad)]
        return out

    def translate(
        self,
        source_text: str,
        committed_ids: Sequence[int] = (),
        num_sequences: int = 1,
        num_beams: int = 1,
        max_new_tokens: int = 64,
    ) -> list[list[int]]:
        torch = self.torch
        count = max(1, num_sequences)
        beams = max(count, num_beams)
        committed = [int(x) for x in committed_ids]
        device = next(self.model.parameters()).device

        if self.mt_arch == "seq2seq":
            inputs = self.tokenizer(
                source_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_context,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            prefix = [self.decoder_start_token_id] + committed
            decoder_input_ids = torch.tensor([prefix], device=device)
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    decoder_input_ids=decoder_input_ids,
                    max_new_tokens=max_new_tokens,
                    num_beams=beams,
                    num_return_sequences=count,
                    early_stopping=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            results = []
            for row in outputs.tolist():
                # Depending on the transformers version, generate returns either
                # only the continuation or the decoder prefix plus continuation.
                if len(row) >= len(prefix) and row[: len(prefix)] == prefix:
                    row = row[len(prefix):]
                results.append(self._clean(row))
            return results

        # Causal MT substitution: prompt-only, no hard token forcing.
        prompt = self.prompt_template.format(source=source_text)
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=self.max_context
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=beams,
                num_return_sequences=count,
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        prompt_len = inputs["input_ids"].shape[1]
        results = []
        for row in outputs.tolist():
            text = self.tokenizer.decode(row[prompt_len:], skip_special_tokens=True).strip()
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            results.append(self._clean(ids))
        return results

    def decode(self, ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(ids), skip_special_tokens=True).strip()

    def encode(self, text: str) -> list[int]:
        return self._clean(self.tokenizer.encode(text, add_special_tokens=False))

    def token_strings(self, ids: Sequence[int]) -> list[str]:
        return list(self.tokenizer.convert_ids_to_tokens([int(x) for x in ids]))

    def sync(self) -> None:
        if self.torch.cuda.is_available():
            self.torch.cuda.synchronize()
