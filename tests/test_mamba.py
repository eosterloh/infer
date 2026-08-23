"""Tests: Mamba-2 mixer + RuntimeState prefill/decode."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from engine.cache import RuntimeState
from engine.config import ModelConfig
from engine.layers.mamba2 import mamba2
from engine.layers.norm import rms_norm
from engine.schedule import MixerKind


def _mamba_cfg(tmp_path: Path) -> ModelConfig:
    raw = {
        "architectures": ["NemotronHForCausalLM"],
        "model_type": "nemotron_h",
        "vocab_size": 32,
        "hidden_size": 32,
        "intermediate_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 16,
        "layer_norm_epsilon": 1e-5,
        "rope_theta": 10000.0,
        "max_position_embeddings": 64,
        "tie_word_embeddings": False,
        "torch_dtype": "float32",
        "hybrid_override_pattern": "M",
        "mamba_num_heads": 4,
        "mamba_head_dim": 8,
        "ssm_state_size": 8,
        "n_groups": 2,
        "conv_kernel": 4,
        "mamba_hidden_act": "silu",
        "chunk_size": 16,
        "time_step_limit": [0.0, float("inf")],
    }
    (tmp_path / "config.json").write_text(json.dumps(raw))
    return ModelConfig.from_pretrained(tmp_path)


def _mamba_weights(cfg: ModelConfig) -> dict[str, torch.Tensor]:
    h = cfg.hidden_size
    inter = cfg.mamba_intermediate
    conv_dim = cfg.mamba_conv_dim
    n_heads = cfg.mamba_num_heads
    assert n_heads is not None and cfg.conv_kernel is not None
    proj = inter + conv_dim + n_heads
    k = cfg.conv_kernel
    return {
        "layers.0.input_norm.weight": torch.ones(h),
        "layers.0.mamba.in_proj.weight": torch.randn(proj, h) * 0.02,
        "layers.0.mamba.out_proj.weight": torch.randn(h, inter) * 0.02,
        "layers.0.mamba.conv1d.weight": torch.randn(conv_dim, 1, k) * 0.02,
        "layers.0.mamba.conv1d.bias": torch.zeros(conv_dim),
        "layers.0.mamba.A_log": torch.log(
            torch.arange(1, n_heads + 1, dtype=torch.float32)
        ),
        "layers.0.mamba.D": torch.ones(n_heads),
        "layers.0.mamba.dt_bias": torch.ones(n_heads),
        "layers.0.mamba.norm.weight": torch.ones(inter),
    }


def test_mamba_prefill_decode(tmp_path: Path) -> None:
    torch.manual_seed(0)
    cfg = _mamba_cfg(tmp_path)
    assert cfg.layers[0].mixer == MixerKind.MAMBA2
    w = _mamba_weights(cfg)
    h = cfg.hidden_size

    x = torch.randn(1, 5, h)
    xn = rms_norm(x, w["layers.0.input_norm.weight"], cfg.rms_norm_eps)
    y0 = mamba2(xn, w, 0, cfg, cache=None)
    assert y0.shape == x.shape
    assert torch.isfinite(y0).all()

    cache = RuntimeState(cfg, batch_size=1, device="cpu", dtype=torch.float32)
    y_pre = mamba2(xn, w, 0, cfg, cache=cache)
    assert cache._mamba_has_state
    cache._token_len = 5

    x1 = torch.randn(1, 1, h)
    x1n = rms_norm(x1, w["layers.0.input_norm.weight"], cfg.rms_norm_eps)
    y1 = mamba2(x1n, w, 0, cfg, cache=cache)
    assert y1.shape == (1, 1, h)
    assert torch.isfinite(y1).all()
    assert torch.isfinite(y_pre).all()

    # Prefill+decode last token must match a full recompute of the concat sequence.
    x_cat = torch.cat([xn, x1n], dim=1)
    y_full = mamba2(x_cat, w, 0, cfg, cache=None)
    torch.testing.assert_close(y_pre, y0, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(y1[0, 0], y_full[0, -1], atol=2e-4, rtol=2e-4)


def test_mamba_sequential_matches_hf_ssd(tmp_path: Path) -> None:
    torch.manual_seed(1)
    cfg = _mamba_cfg(tmp_path)
    w = _mamba_weights(cfg)
    h = cfg.hidden_size
    x = torch.randn(2, 7, h)
    xn = rms_norm(x, w["layers.0.input_norm.weight"], cfg.rms_norm_eps)
    y_seq = mamba2(xn, w, 0, cfg, cache=None)

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from mamba_ssd_ref import ssd_scan

    # Replay the mixer up to the scan, then SSD.
    p = "layers.0"
    n_heads = cfg.mamba_num_heads
    head_dim = cfg.mamba_head_dim
    n_groups = cfg.n_groups
    state_size = cfg.ssm_state_size
    inter = cfg.mamba_intermediate
    conv_dim = cfg.mamba_conv_dim
    kernel = cfg.conv_kernel
    assert n_heads and head_dim and n_groups and state_size and kernel
    projected = torch.nn.functional.linear(xn, w[f"{p}.mamba.in_proj.weight"])
    gate, bc, dt = projected.split([inter, conv_dim, n_heads], dim=-1)
    from engine.layers.mamba2 import _depthwise_conv1d, _rms_norm_gated

    bc_out = torch.nn.functional.silu(
        _depthwise_conv1d(bc, w[f"{p}.mamba.conv1d.weight"], w[f"{p}.mamba.conv1d.bias"], kernel)
    )
    x_ssm, B, C = bc_out.split(
        [inter, n_groups * state_size, n_groups * state_size], dim=-1
    )
    A = -torch.exp(w[f"{p}.mamba.A_log"].float())
    dt = torch.nn.functional.softplus(dt.float() + w[f"{p}.mamba.dt_bias"].float())
    b, s, _ = xn.shape
    x_h = x_ssm.reshape(b, s, n_heads, head_dim).float()
    B_g = B.reshape(b, s, n_groups, state_size).float()
    C_g = C.reshape(b, s, n_groups, state_size).float()
    reps = n_heads // n_groups
    B_h = B_g.repeat_interleave(reps, dim=2)
    C_h = C_g.repeat_interleave(reps, dim=2)
    y_ssd = ssd_scan(
        x_h, dt, A, B_h, C_h, w[f"{p}.mamba.D"].float(), chunk_size=int(cfg.raw["chunk_size"])
    )
    scan_out = _rms_norm_gated(
        y_ssd.to(xn.dtype),
        w[f"{p}.mamba.norm.weight"],
        gate,
        eps=cfg.rms_norm_eps,
        group_size=inter // n_groups,
    )
    y_ref = torch.nn.functional.linear(scan_out, w[f"{p}.mamba.out_proj.weight"])
    torch.testing.assert_close(y_seq, y_ref, atol=5e-4, rtol=5e-4)


def test_mamba_two_layer_single_token_prefill(tmp_path: Path) -> None:
    """A 1-token prompt must prefill EVERY Mamba layer, not decode later ones."""
    raw = {
        "architectures": ["NemotronHForCausalLM"],
        "model_type": "nemotron_h",
        "vocab_size": 32,
        "hidden_size": 32,
        "intermediate_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 16,
        "layer_norm_epsilon": 1e-5,
        "rope_theta": 10000.0,
        "max_position_embeddings": 64,
        "tie_word_embeddings": False,
        "torch_dtype": "float32",
        "hybrid_override_pattern": "MM",
        "mamba_num_heads": 4,
        "mamba_head_dim": 8,
        "ssm_state_size": 8,
        "n_groups": 2,
        "conv_kernel": 4,
        "mamba_hidden_act": "silu",
        "chunk_size": 16,
        "time_step_limit": [0.0, float("inf")],
    }
    (tmp_path / "config.json").write_text(json.dumps(raw))
    cfg = ModelConfig.from_pretrained(tmp_path)
    torch.manual_seed(2)
    w = {}
    for layer in (0, 1):
        base = _mamba_weights(cfg)
        for k, v in base.items():
            w[k.replace("layers.0.", f"layers.{layer}.")] = v.clone()
    x = torch.randn(1, 1, cfg.hidden_size)
    cache = RuntimeState(cfg, batch_size=1, device="cpu", dtype=torch.float32)
    y_cache = mamba2(
        rms_norm(x, w["layers.0.input_norm.weight"], cfg.rms_norm_eps),
        w,
        0,
        cfg,
        cache=cache,
    )
    y1_cache = mamba2(
        rms_norm(y_cache, w["layers.1.input_norm.weight"], cfg.rms_norm_eps),
        w,
        1,
        cfg,
        cache=cache,
    )
    y0 = mamba2(
        rms_norm(x, w["layers.0.input_norm.weight"], cfg.rms_norm_eps),
        w,
        0,
        cfg,
        cache=None,
    )
    y1 = mamba2(
        rms_norm(y0, w["layers.1.input_norm.weight"], cfg.rms_norm_eps),
        w,
        1,
        cfg,
        cache=None,
    )
    torch.testing.assert_close(y1_cache, y1, atol=1e-5, rtol=1e-5)
    assert cache.mamba_ready(0) and cache.mamba_ready(1)
