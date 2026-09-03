"""Gated DeltaNet mixer: prefill vs decode state matches a full pass."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

from engine.cache import RuntimeState
from engine.config import ModelConfig
from engine.layers.gdn import (
    _gated_delta_chunk,
    _gated_delta_recurrent,
    gated_delta_net,
)
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

    chunked = RuntimeState(cfg, batch_size=1, device="cpu", dtype=torch.float32)
    y_a = gated_delta_net(xn[:, :3], w, 0, cfg, cache=chunked)
    y_b = gated_delta_net(xn[:, 3:], w, 0, cfg, cache=chunked)
    assert torch.allclose(y0, torch.cat((y_a, y_b), dim=1), atol=1e-4, rtol=1e-4)


def test_gdn_matches_transformers_reference(tmp_path: Path) -> None:
    torch.manual_seed(11)
    cfg = _gdn_cfg(tmp_path)
    w = random_engine_weights(cfg)
    hf_cfg = Qwen3_5TextConfig(**cfg.raw)
    hf = Qwen3_5GatedDeltaNet(hf_cfg, layer_idx=0).eval()
    prefix = "layers.0.gdn."
    state = {
        name: w[f"{prefix}{name}"].clone()
        for name in (
            "A_log",
            "dt_bias",
            "conv1d.weight",
            "norm.weight",
            "out_proj.weight",
            "in_proj_qkv.weight",
            "in_proj_z.weight",
            "in_proj_b.weight",
            "in_proj_a.weight",
        )
    }
    hf.load_state_dict(state)
    x = torch.randn(1, 6, cfg.hidden_size)
    with torch.inference_mode():
        expected = hf(x, cache_params=None, attention_mask=None)
        actual = gated_delta_net(x, w, 0, cfg, cache=None)
    assert torch.allclose(actual, expected, atol=2e-5, rtol=2e-5)


def test_gdn_chunk_matches_recurrent() -> None:
    torch.manual_seed(13)
    shape = (1, 65, 6, 8)
    q, k, v = (torch.randn(shape) * 0.1 for _ in range(3))
    g = -torch.rand(1, 65, 6) * 0.1
    beta = torch.sigmoid(torch.randn(1, 65, 6))
    expected, expected_state = _gated_delta_recurrent(q, k, v, g, beta, None)
    actual, actual_state = _gated_delta_chunk(q, k, v, g, beta, None)
    assert torch.allclose(actual, expected, atol=2e-5, rtol=2e-5)
    assert torch.allclose(actual_state, expected_state, atol=2e-5, rtol=2e-5)
