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
