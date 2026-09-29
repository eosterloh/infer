"""The FFN residual can wait for the next layer's norm without changing the value."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from engine.agent_api import load_engine
from engine.config import ModelConfig
from engine.layers.block import decoder_block
from engine.synth import random_engine_weights, write_config, write_hf_folder

_LLAMA = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "vocab_size": 64,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 3,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "rms_norm_eps": 1e-5,
    "rope_theta": 10000.0,
    "max_position_embeddings": 128,
    "tie_word_embeddings": False,
    "torch_dtype": "float32",
    "hidden_act": "silu",
}

_GPT2 = {
    "architectures": ["GPT2LMHeadModel"],
    "model_type": "gpt2",
    "vocab_size": 64,
    "n_embd": 32,
    "n_head": 4,
    "n_layer": 3,
    "n_inner": 64,
    "n_positions": 64,
    "layer_norm_epsilon": 1e-5,
    "torch_dtype": "float32",
    "tie_word_embeddings": True,
}


def _logits(folder: Path, raw: dict) -> torch.Tensor:
    write_config(folder, raw)
    cfg = ModelConfig.from_pretrained(folder)
    write_hf_folder(folder, cfg, random_engine_weights(cfg, seed=4))
    model = load_engine(folder, device="cpu", dtype="float32").model
    ids = torch.randint(0, 64, (1, 6))
    return model, model.forward(ids)


@pytest.mark.parametrize("name,raw", [("llama", _LLAMA), ("gpt2", _GPT2)])
def test_carrying_the_residual_matches_adding_it_immediately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, raw: dict
) -> None:
    """Deferring the add into the next norm is the same arithmetic."""
    model, fused = _logits(tmp_path / name, raw)

    def unfused(*args, **kwargs):
        defer = kwargs.get("defer_ffn_add", False)
        kwargs["incoming_delta"] = None
        kwargs["defer_ffn_add"] = False
        out = decoder_block(*args, **kwargs)
        return (out, None) if defer else out

    monkeypatch.setattr("engine.model.decoder_block", unfused)
    ids = torch.randint(0, 64, (1, 6))
    # Same ids as the fused run: _logits drew them after seeding inside load.
    # Recompute both from one prompt so the comparison is the residual, not the tokens.
    prompt = torch.tensor([[1, 2, 3, 4, 5, 6]])
    monkeypatch.undo()
    fused = model.forward(prompt)
    monkeypatch.setattr("engine.model.decoder_block", unfused)
    plain = model.forward(prompt)
    torch.testing.assert_close(fused, plain, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="layer-norm kernel runs on the GPU")
def test_cuda_fused_add_layer_norm_matches_the_two_steps() -> None:
    from engine.kernels import fused_add_layer_norm, load_extension

    assert load_extension() is not None
    torch.manual_seed(12)
    delta = torch.randn(1, 4, 768, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(delta)
    weight = torch.randn(768, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(768, device="cuda", dtype=torch.bfloat16)
    got, got_residual = fused_add_layer_norm(
        delta.clone(), residual.clone(), weight, bias, 1e-5
    )
    want_residual = residual + delta
    want = F.layer_norm(want_residual.float(), (768,), weight.float(), bias.float(), 1e-5)
    torch.testing.assert_close(got_residual.float(), want_residual.float(), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(got.float(), want, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="stacked GEMV runs on the GPU")
def test_gemv_stack_matches_separate_projections() -> None:
    from engine.kernels import gemv, gemv_stack, load_extension

    assert load_extension() is not None
    torch.manual_seed(3)
    k, ns = 128, (64, 32, 32)
    x = torch.randn(k, device="cuda", dtype=torch.bfloat16)
    weights = [torch.randn(n, k, device="cuda", dtype=torch.bfloat16) for n in ns]
    biases = [torch.randn(n, device="cuda", dtype=torch.bfloat16) for n in ns]
    got = gemv_stack(x, weights, biases)
    assert got is not None
    for part, weight, bias in zip(got, weights, biases):
        want = gemv(x, weight, bias)
        torch.testing.assert_close(part, want, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the cache bug is on the CUDA allocator")
def test_stacked_gemv_does_not_keep_a_freed_matrix() -> None:
    """A new weight tensor must not inherit the previous one's concatenated copy."""
    from engine.kernels import gemv, gemv_stack, load_extension

    assert load_extension() is not None
    torch.manual_seed(4)
    k = 64
    x = torch.randn(k, device="cuda", dtype=torch.bfloat16)

    def pair() -> list[torch.Tensor]:
        return [
            torch.randn(32, k, device="cuda", dtype=torch.bfloat16),
            torch.randn(16, k, device="cuda", dtype=torch.bfloat16),
        ]

    first = pair()
    gemv_stack(x, first, [None, None])
    del first
    torch.cuda.empty_cache()
    second = pair()
    got = gemv_stack(x, second, [None, None])
    assert got is not None
    for part, weight in zip(got, second):
        torch.testing.assert_close(part, gemv(x, weight), atol=0, rtol=0)


def test_fused_add_layer_norm_matches_the_two_steps() -> None:
    from engine.kernels import fused_add_layer_norm

    torch.manual_seed(9)
    delta = torch.randn(2, 5, 32)
    residual = torch.randn(2, 5, 32)
    weight = torch.randn(32)
    bias = torch.randn(32)
    got, got_residual = fused_add_layer_norm(
        delta.clone(), residual.clone(), weight, bias, 1e-5
    )
    want_residual = residual + delta
    want = F.layer_norm(want_residual, (32,), weight, bias, 1e-5)
    torch.testing.assert_close(got_residual, want_residual, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)
