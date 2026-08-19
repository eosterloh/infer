"""Tests: MoE router + expert path (synthetic weights)."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from engine.config import ModelConfig
from engine.layers.moe import moe, route_topk
from engine.schedule import FfnKind, MixerKind


def _moe_cfg(tmp_path: Path) -> ModelConfig:
    raw = {
        "architectures": ["NemotronHForCausalLM"],
        "model_type": "nemotron_h",
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 8,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "layer_norm_epsilon": 1e-5,
        "rope_theta": 10000.0,
        "max_position_embeddings": 64,
        "tie_word_embeddings": False,
        "torch_dtype": "float32",
        "hybrid_override_pattern": "E",
        "n_routed_experts": 4,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 8,
        "moe_shared_expert_intermediate_size": 16,
        "routed_scaling_factor": 2.5,
        "mlp_hidden_act": "relu2",
        "n_group": 1,
        "topk_group": 1,
        "norm_topk_prob": True,
    }
    (tmp_path / "config.json").write_text(json.dumps(raw))
    return ModelConfig.from_pretrained(tmp_path)


def test_moe_forward_shape_and_finite(tmp_path: Path) -> None:
    torch.manual_seed(0)
    cfg = _moe_cfg(tmp_path)
    assert cfg.layers[0].mixer == MixerKind.NONE
    assert cfg.layers[0].ffn == FfnKind.MOE

    h = cfg.hidden_size
    n_e = cfg.n_routed_experts
    mi = cfg.moe_intermediate_size
    si = cfg.moe_shared_expert_intermediate_size
    assert n_e and mi and si

    w: dict[str, torch.Tensor] = {
        "layers.0.input_norm.weight": torch.ones(h),
        "layers.0.moe.gate.weight": torch.randn(n_e, h),
        "layers.0.moe.gate.e_score_correction_bias": torch.zeros(n_e),
        "layers.0.moe.shared.up.weight": torch.randn(si, h),
        "layers.0.moe.shared.down.weight": torch.randn(h, si),
    }
    for e in range(n_e):
        w[f"layers.0.moe.experts.{e}.up.weight"] = torch.randn(mi, h)
        w[f"layers.0.moe.experts.{e}.down.weight"] = torch.randn(h, mi)

    x = torch.randn(2, 3, h)
    y = moe(x, w, 0, cfg)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()

    idx, weights = route_topk(
        x.reshape(-1, h),
        w["layers.0.moe.gate.weight"],
        w["layers.0.moe.gate.e_score_correction_bias"],
        top_k=2,
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        routed_scaling_factor=2.5,
    )
    assert idx.shape == (6, 2)
    assert weights.shape == (6, 2)
