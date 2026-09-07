"""Unit tests for new recipe math: GeGLU, granite scalars, MoE shared gate, residuals."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch

from engine.config import ModelConfig
from engine.layers.mlp import mlp
from engine.model import DecoderModel
from engine.synth import random_engine_weights, write_config


_BASE = {
    "vocab_size": 64,
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "rms_norm_eps": 1e-5,
    "rope_theta": 10000.0,
    "max_position_embeddings": 64,
    "tie_word_embeddings": False,
    "torch_dtype": "float32",
    "hidden_act": "silu",
}


def test_geglu_gated_vs_ungated() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 3, 8)
    w_gate = torch.randn(16, 8)
    w_up = torch.randn(16, 8)
    w_down = torch.randn(8, 16)
    gated = mlp(x, w_gate, w_up, w_down, act="gelu_pytorch_tanh")
    ungated = mlp(x, w_up, w_up, w_down, act="gelu_pytorch_tanh")
    assert gated.shape == ungated.shape == x.shape
    assert not torch.allclose(gated, ungated)


def test_granite_scalars_change_outputs(tmp_path: Path) -> None:
    raw = {
        **_BASE,
        "architectures": ["GraniteForCausalLM"],
        "model_type": "granite",
        "attention_multiplier": 0.125,
        "residual_multiplier": 1.0,
        "embedding_multiplier": 1.0,
        "logits_scaling": 1.0,
    }
    cfg = ModelConfig.from_pretrained(write_config(tmp_path / "granite", raw))
    torch.manual_seed(1)
    weights = random_engine_weights(cfg, seed=1)
    ids = torch.randint(0, cfg.vocab_size, (1, 4))
    base = DecoderModel(cfg, weights).forward(ids)
    scaled = DecoderModel(
        replace(cfg, residual_multiplier=1.5, embedding_multiplier=1.2, logits_scaling=2.0),
        weights,
    ).forward(ids)
    assert not torch.allclose(base, scaled)
    attn_scaled = DecoderModel(replace(cfg, attention_multiplier=0.5), weights).forward(ids)
    assert not torch.allclose(base, attn_scaled)


def test_qwen2_moe_shared_expert_gate(tmp_path: Path) -> None:
    raw = {
        **_BASE,
        "architectures": ["Qwen2MoeForCausalLM"],
        "model_type": "qwen2_moe",
        "num_hidden_layers": 1,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 32,
        "shared_expert_intermediate_size": 64,
        "qkv_bias": True,
    }
    cfg = ModelConfig.from_pretrained(write_config(tmp_path / "qwen2moe", raw))
    torch.manual_seed(2)
    weights = random_engine_weights(cfg, seed=2)
    ids = torch.randint(0, cfg.vocab_size, (1, 3))
    with_gate = DecoderModel(cfg, weights).forward(ids)
    no_gate = dict(weights)
    no_gate.pop("layers.0.moe.shared_gate.weight")
    without = DecoderModel(cfg, no_gate).forward(ids)
    assert not torch.allclose(with_gate, without)


def test_olmo2_post_norm_differs_from_sequential(tmp_path: Path) -> None:
    raw = {
        **_BASE,
        "architectures": ["Olmo2ForCausalLM"],
        "model_type": "olmo2",
        "qk_norm": True,
        "num_hidden_layers": 1,
    }
    cfg = ModelConfig.from_pretrained(write_config(tmp_path / "olmo2", raw))
    torch.manual_seed(3)
    weights = random_engine_weights(cfg, seed=3)
    h = cfg.hidden_size
    weights["layers.0.input_norm.weight"] = torch.ones(h)
    ids = torch.randint(0, cfg.vocab_size, (1, 4))
    post = DecoderModel(cfg, weights).forward(ids)
    sequential = DecoderModel(replace(cfg, residual_kind="sequential"), weights).forward(ids)
    assert not torch.allclose(post, sequential)


def test_gemma2_expected_shapes_include_extra_norms(tmp_path: Path) -> None:
    raw = {
        **_BASE,
        "architectures": ["Gemma2ForCausalLM"],
        "model_type": "gemma2",
        "hidden_activation": "gelu_pytorch_tanh",
        "query_pre_attn_scalar": 8,
        "layer_types": ["sliding_attention", "full_attention"],
        "sliding_window": 16,
    }
    cfg = ModelConfig.from_pretrained(write_config(tmp_path / "gemma2", raw))
    shapes = cfg.expected_shapes()
    for layer in range(cfg.num_hidden_layers):
        p = f"layers.{layer}"
        assert f"{p}.input_norm.weight" in shapes
        assert f"{p}.post_attn_norm.weight" in shapes
        assert f"{p}.pre_ff_norm.weight" in shapes
        assert f"{p}.post_ff_norm.weight" in shapes


def test_seed_oss_optional_o_bias(tmp_path: Path) -> None:
    raw = {
        **_BASE,
        "architectures": ["SeedOssForCausalLM"],
        "model_type": "seed_oss",
        "attention_bias": True,
        "attention_out_bias": False,
    }
    cfg = ModelConfig.from_pretrained(write_config(tmp_path / "seed", raw))
    assert cfg.recipe_id == "llama"
    shapes = cfg.expected_shapes()
    assert "layers.0.attn.q.bias" in shapes
    assert "layers.0.attn.o.bias" not in shapes


def test_gemma3_mixed_sliding_changes_outputs(tmp_path: Path) -> None:
    raw = {
        **_BASE,
        "architectures": ["Gemma3ForCausalLM"],
        "model_type": "gemma3",
        "hidden_activation": "gelu_pytorch_tanh",
        "query_pre_attn_scalar": 8,
        "qk_norm": True,
        "sliding_window": 4,
        "layer_types": ["sliding_attention", "full_attention"],
        "num_hidden_layers": 2,
        "rope_theta": 1_000_000.0,
        "rope_parameters": {
            "full_attention": {"rope_type": "default", "rope_theta": 1_000_000.0},
            "sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0},
        },
    }
    cfg = ModelConfig.from_pretrained(write_config(tmp_path / "gemma3mix", raw))
    torch.manual_seed(4)
    weights = random_engine_weights(cfg, seed=4)
    ids = torch.randint(0, cfg.vocab_size, (1, 6))
    mixed = DecoderModel(cfg, weights).forward(ids)
    full = DecoderModel(replace(cfg, layer_types=("full_attention", "full_attention")), weights).forward(ids)
    assert not torch.allclose(mixed, full)


def test_olmo3_sliding_post_norm_changes_outputs(tmp_path: Path) -> None:
    raw = {
        **_BASE,
        "architectures": ["Olmo3ForCausalLM"],
        "model_type": "olmo3",
        "layer_types": ["sliding_attention", "full_attention"],
        "sliding_window": 4,
        "num_hidden_layers": 2,
        "qk_norm": True,
    }
    cfg = ModelConfig.from_pretrained(write_config(tmp_path / "olmo3u", raw))
    torch.manual_seed(5)
    weights = random_engine_weights(cfg, seed=5)
    h = cfg.hidden_size
    weights["layers.0.input_norm.weight"] = torch.ones(h)
    weights["layers.1.input_norm.weight"] = torch.ones(h)
    ids = torch.randint(0, cfg.vocab_size, (1, 6))
    post = DecoderModel(cfg, weights).forward(ids)
    sequential = DecoderModel(replace(cfg, residual_kind="sequential"), weights).forward(ids)
    assert not torch.allclose(post, sequential)
    no_slide = DecoderModel(replace(cfg, sliding_window=None, layer_types=("full_attention", "full_attention")), weights).forward(ids)
    assert not torch.allclose(post, no_slide)


def test_phi_fc_ungated_and_partial_rope(tmp_path: Path) -> None:
    raw = {
        **_BASE,
        "architectures": ["PhiForCausalLM"],
        "model_type": "phi",
        "hidden_act": "gelu_new",
        "num_hidden_layers": 1,
        "tie_word_embeddings": False,
    }
    cfg = ModelConfig.from_pretrained(write_config(tmp_path / "phiu", raw))
    torch.manual_seed(6)
    weights = random_engine_weights(cfg, seed=6)
    ids = torch.randint(0, cfg.vocab_size, (1, 4))
    base = DecoderModel(cfg, weights).forward(ids)
    full_rope = DecoderModel(replace(cfg, partial_rotary_factor=1.0), weights).forward(ids)
    assert not torch.allclose(base, full_rope)


def test_granite_swa_sinks_change_outputs(tmp_path: Path) -> None:
    raw = {
        **_BASE,
        "architectures": ["GraniteSWAForCausalLM"],
        "model_type": "granite_swa",
        "attention_multiplier": 0.25,
        "num_hidden_layers": 1,
        "layer_types": ["full_attention"],
        "tie_word_embeddings": False,
    }
    cfg = ModelConfig.from_pretrained(write_config(tmp_path / "gswa", raw))
    torch.manual_seed(7)
    weights = random_engine_weights(cfg, seed=7)
    ids = torch.randint(0, cfg.vocab_size, (1, 4))
    with_sinks = DecoderModel(cfg, weights).forward(ids)
    no_sinks = dict(weights)
    no_sinks.pop("layers.0.attn.sinks")
    without = DecoderModel(cfg, no_sinks).forward(ids)
    assert not torch.allclose(with_sinks, without)


def test_stablelm_partial_rope_changes_outputs(tmp_path: Path) -> None:
    raw = {
        **_BASE,
        "architectures": ["StableLmForCausalLM"],
        "model_type": "stablelm",
        "use_parallel_residual": False,
        "use_qkv_bias": False,
        "num_hidden_layers": 1,
        "tie_word_embeddings": False,
    }
    cfg = ModelConfig.from_pretrained(write_config(tmp_path / "slm", raw))
    torch.manual_seed(8)
    weights = random_engine_weights(cfg, seed=8)
    ids = torch.randint(0, cfg.vocab_size, (1, 4))
    partial = DecoderModel(cfg, weights).forward(ids)
    full = DecoderModel(replace(cfg, partial_rotary_factor=1.0), weights).forward(ids)
    assert not torch.allclose(partial, full)
