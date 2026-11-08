import copy

import pytest
import torch

from cpu_embedding import offload_input_embedding


def test_lookup_and_model_device_dtype_stay_identical_on_cpu():
    from torch import nn

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(32, 12, dtype=torch.float16)
            self.output = nn.Linear(12, 32, bias=False, dtype=torch.float16)

        def get_input_embeddings(self):
            return self.embed

        def set_input_embeddings(self, value):
            self.embed = value

        def get_output_embeddings(self):
            return self.output

    model = Model()
    ids = torch.tensor([[1, 2, 3], [4, 2, 1]])
    expected = model.embed(ids).detach().clone()
    original_dtype = next(model.parameters()).dtype
    info = offload_input_embedding(model)
    assert info["weight_bytes"] == 32 * 12 * 2
    assert next(model.parameters()).dtype == original_dtype
    assert model.embed.weight.device.type == "cpu"
    assert torch.equal(expected, model.embed(ids))
    with pytest.raises(ValueError, match="already"):
        offload_input_embedding(model)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_random_qwen_sampling_preserves_full_batch_and_first_parameter(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from transformers import Qwen2Config, Qwen2ForCausalLM

    config = Qwen2Config(vocab_size=48, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=1, num_attention_heads=4,
                         num_key_value_heads=2, tie_word_embeddings=False,
                         eos_token_id=None, pad_token_id=0, bos_token_id=1)
    torch.manual_seed(10)
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = Qwen2ForCausalLM(config).to(device=device, dtype=dtype).eval()
    inputs = torch.tensor([[1, 3, 4, 5]], device=device)
    with torch.inference_mode():
        expected_embeddings = model.get_input_embeddings()(inputs).clone()
        torch.manual_seed(123)
        expected = model.generate(inputs, max_new_tokens=5, do_sample=True,
                                  num_return_sequences=4, top_p=0.9, top_k=0)
    expected_device, expected_dtype = model.device, model.dtype
    offload_input_embedding(model)
    assert model.device == expected_device
    assert model.dtype == expected_dtype
    with torch.inference_mode():
        assert torch.equal(expected_embeddings, model.get_input_embeddings()(inputs))
        torch.manual_seed(123)
        actual = model.generate(inputs, max_new_tokens=5, do_sample=True,
                                num_return_sequences=4, top_p=0.9, top_k=0)
    assert torch.equal(expected, actual)


def test_tied_output_weight_is_rejected_before_model_mutation():
    from torch import nn

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(8, 4)
            self.output = nn.Linear(4, 8, bias=False)
            self.output.weight = self.embedding.weight

        def get_input_embeddings(self):
            return self.embedding

        def get_output_embeddings(self):
            return self.output

    with pytest.raises(ValueError, match="tied"):
        offload_input_embedding(Model())
