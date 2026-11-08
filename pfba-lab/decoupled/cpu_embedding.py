"""Keep an untied input embedding on CPU while preserving GPU execution identity."""
from __future__ import annotations

import torch
from torch import nn


class CpuInputEmbedding(nn.Module):
    def __init__(self, embedding: nn.Embedding, execution_device):
        super().__init__()
        # GenerationMixin derives model.device/dtype from the first parameter.
        # This zero-element parameter preserves the original execution identity.
        self.register_parameter("_execution_device", nn.Parameter(
            torch.empty(0, device=execution_device, dtype=embedding.weight.dtype),
            requires_grad=False,
        ))
        self.embedding = embedding.to(device="cpu")

    @property
    def weight(self):
        return self.embedding.weight

    def forward(self, input_ids):
        # Lookup preserves the original stored dtype and exact weight values.
        return self.embedding(input_ids.to(device="cpu")).to(device=input_ids.device)


def offload_input_embedding(model):
    embedding = model.get_input_embeddings()
    if isinstance(embedding, CpuInputEmbedding):
        raise ValueError("input embedding is already offloaded")
    if not isinstance(embedding, nn.Embedding):
        raise ValueError("CPU lookup requires a standard torch.nn.Embedding")
    output = model.get_output_embeddings()
    if output is not None and output.weight is embedding.weight:
        raise ValueError("tied input/output embedding weights cannot be offloaded this way")
    execution_device = embedding.weight.device
    weight_bytes = embedding.weight.numel() * embedding.weight.element_size()
    wrapper = CpuInputEmbedding(embedding, execution_device)
    wrapper.train(model.training)
    model.set_input_embeddings(wrapper)
    if execution_device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "weight_bytes": weight_bytes, "weight_dtype": str(wrapper.weight.dtype),
        "weight_device": "cpu", "execution_device": str(execution_device),
    }
