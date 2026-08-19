#!/usr/bin/env python3
"""Unit smoke for Nemotron-H MoE router + expert path (synthetic weights)."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.config import ModelConfig
from engine.layers.moe import moe, route_topk
from engine.schedule import FfnKind, LayerSpec, MixerKind


def _tiny_moe_config() -> ModelConfig:
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
    with tempfile.TemporaryDirectory() as td:
        path = Path(td)
        (path / "config.json").write_text(json.dumps(raw))
        return ModelConfig.from_pretrained(path)


def main() -> int:
    torch.manual_seed(0)
    cfg = _tiny_moe_config()
    assert cfg.layers == (LayerSpec(0, MixerKind.NONE, FfnKind.MOE),)

    h = cfg.hidden_size
    n_e = cfg.n_routed_experts
    mi = cfg.moe_intermediate_size
    si = cfg.moe_shared_expert_intermediate_size
    assert n_e is not None and mi is not None and si is not None

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

    flat = x.reshape(-1, h)
    idx, weights = route_topk(
        flat,
        w["layers.0.moe.gate.weight"],
        w["layers.0.moe.gate.e_score_correction_bias"],
        top_k=cfg.num_experts_per_tok or 2,
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        routed_scaling_factor=float(cfg.routed_scaling_factor or 1.0),
    )
    assert idx.shape == (flat.shape[0], cfg.num_experts_per_tok)
    assert weights.shape == idx.shape
    print(f"moe_out_abs_mean={y.abs().mean().item():.4f}")
    print(f"route_idx_sample={idx[0].tolist()}")
    print("moe_smoke=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
