"""Gated DeltaNet mixer: prefill vs decode state matches a full pass."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from engine.cache import RuntimeState
from engine.config import ModelConfig
from engine.layers.gdn import gated_delta_net
from engine.layers.norm import gemma_rms_norm
from engine.schedule import MixerKind
from engine.synth import random_engine_weights


def _gdn_cfg(tmp_path: Path) -> ModelConfig:
    raw = {
        "architectures": ["Qwen3_5ForCausalLM"],
        "model_type": "qwen3_5",
        "vocab_size": 32,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "max_position_embeddings": 64,
        "tie_word_embeddings": False,
        "torch_dtype": "float32",
        "layer_types": ["linear_attention"],
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 8,
        "linear_value_head_dim": 8,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "attn_output_gate": True,
        "partial_rotary_factor": 0.5,
    }
    (tmp_path / "config.json").write_text(json.dumps(raw))
    return ModelConfig.from_pretrained(tmp_path)


def test_gdn_prefill_decode(tmp_path: Path) -> None:
    torch.manual_seed(0)
    cfg = _gdn_cfg(tmp_path)
    assert cfg.layers[0].mixer == MixerKind.GATED_DELTANET
    w = random_engine_weights(cfg)
    h = cfg.hidden_size
    x = torch.randn(1, 5, h)
    xn = gemma_rms_norm(x, w["layers.0.input_norm.weight"], cfg.rms_norm_eps)

    y0 = gated_delta_net(xn, w, 0, cfg, cache=None)
    assert y0.shape == x.shape
    assert torch.isfinite(y0).all()

    cache = RuntimeState(cfg, batch_size=1, device="cpu", dtype=torch.float32)
    y_pre = gated_delta_net(xn[:, :4], w, 0, cfg, cache=cache)
    y_step = gated_delta_net(xn[:, 4:], w, 0, cfg, cache=cache)
    y_cached = torch.cat([y_pre, y_step], dim=1)
    assert torch.allclose(y0, y_cached, atol=1e-4, rtol=1e-4)
