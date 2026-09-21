"""Tests: MoE router + expert path (synthetic weights)."""

from __future__ import annotations

import json

import pytest
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


def test_fused_dispatch_gate_follows_rows_per_expert() -> None:
    """Decode always fuses; a wide prefill must not re-read every expert."""
    from engine.layers.moe import fused_worth_it

    assert fused_worth_it(1, 8, 128) is True
    assert fused_worth_it(32, 8, 128) is True  # 2 rows per expert
    assert fused_worth_it(33, 8, 128) is False
    assert fused_worth_it(512, 8, 128) is False
    assert fused_worth_it(8, 2, 8) is True
    assert fused_worth_it(64, 2, 8) is False
    # A router with no experts to spread over cannot be improved on.
    assert fused_worth_it(512, 8, 0) is True


def test_both_dispatch_paths_agree(tmp_path: Path, monkeypatch) -> None:
    """The gate is a speed switch, so it must not move the numbers."""
    from engine.layers.moe import moe
    from engine.quantize import stack_moe_experts

    torch.manual_seed(5)
    cfg = _moe_cfg(tmp_path)
    h = cfg.hidden_size
    n_e, mi, si = cfg.n_routed_experts, cfg.moe_intermediate_size, cfg.moe_shared_expert_intermediate_size
    w: dict[str, torch.Tensor] = {
        "layers.0.moe.gate.weight": torch.randn(n_e, h),
        "layers.0.moe.gate.e_score_correction_bias": torch.zeros(n_e),
        "layers.0.moe.shared.up.weight": torch.randn(si, h),
        "layers.0.moe.shared.down.weight": torch.randn(h, si),
    }
    for e in range(n_e):
        w[f"layers.0.moe.experts.{e}.up.weight"] = torch.randn(mi, h)
        w[f"layers.0.moe.experts.{e}.down.weight"] = torch.randn(h, mi)
    stack_moe_experts(w)

    x = torch.randn(1, 24, h)
    monkeypatch.setenv("INFER_MOE_FUSED_ROWS_PER_EXPERT", "1000")
    fused = moe(x, w, 0, cfg)
    monkeypatch.setenv("INFER_MOE_FUSED_ROWS_PER_EXPERT", "0")
    looped = moe(x, w, 0, cfg)
    torch.testing.assert_close(fused, looped, atol=2e-5, rtol=2e-5)


_STACK_BASE = {
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
_STACK_RECIPES = {
    "gpt_oss": {
        "architectures": ["GptOssForCausalLM"],
        "model_type": "gpt_oss",
        "num_local_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 32,
    },
    "qwen3_moe": {
        "architectures": ["Qwen3MoeForCausalLM"],
        "model_type": "qwen3_moe",
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 32,
        "decoder_sparse_step": 1,
    },
    "mixtral": {
        "architectures": ["MixtralForCausalLM"],
        "model_type": "mixtral",
        "num_local_experts": 4,
        "num_experts_per_tok": 2,
    },
}


@pytest.mark.parametrize("recipe", sorted(_STACK_RECIPES))
def test_stacking_experts_does_not_change_the_logits(
    tmp_path: Path, recipe: str, monkeypatch
) -> None:
    """Stacking is a layout change, so it has to be invisible in the output.

    GPT-OSS ships per-expert biases as [E, N] next to [E, N, K] weights, and the
    stack is the only place the layer looks for them once one exists — so a bias
    the adopt step skips is a bias dropped from the math, which showed up here as
    an 11.7 logit difference before the fix.
    """
    from engine.agent_api import load_engine
    from engine.synth import random_engine_weights, write_config, write_hf_folder

    folder = write_config(tmp_path / recipe, {**_STACK_BASE, **_STACK_RECIPES[recipe]})
    cfg = ModelConfig.from_pretrained(folder)
    write_hf_folder(folder, cfg, random_engine_weights(cfg))
    ids = torch.arange(4).reshape(1, 4) % cfg.vocab_size

    out = {}
    for flag in ("0", "1"):
        monkeypatch.setenv("INFER_MOE_STACK", flag)
        engine = load_engine(folder, device="cpu", dtype="float32")
        out[flag] = engine.model.forward(ids).detach().clone()
    torch.testing.assert_close(out["1"], out["0"], atol=1e-4, rtol=1e-4)

    monkeypatch.setenv("INFER_MOE_STACK", "1")
    stacked = load_engine(folder, device="cpu", dtype="float32")
    stack = stacked.model.weights.get("_expert_stacks")
    assert stack, "stacking was requested and did not happen"
    fields = set(next(iter(stack.values())))
    loose = {
        name
        for name in stacked.model.weights
        if ".experts." in name and name.endswith(".bias")
    }
    assert not loose or {f for f in fields if f.endswith("_bias")}, (
        f"{recipe} has per-expert biases {loose} that no stack field covers"
    )
