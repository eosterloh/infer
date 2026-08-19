"""Tests: dense DecoderModel matches legacy transformer_block loop."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from engine.config import ModelConfig
from engine.layers import build_inv_freq, build_rope_cos_sin, rms_norm, transformer_block
from engine.model import DecoderModel, LlamaModel


def test_decoder_matches_transformer_block(tmp_path: Path) -> None:
    raw = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
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
    (tmp_path / "config.json").write_text(json.dumps(raw))
    cfg = ModelConfig.from_pretrained(tmp_path)
    assert LlamaModel is DecoderModel

    torch.manual_seed(0)
    h, i, v = cfg.hidden_size, cfg.intermediate_size, cfg.vocab_size
    nq, nkv, dh = cfg.nq, cfg.nkv, cfg.head_dim
    w: dict[str, torch.Tensor] = {
        "embed.weight": torch.randn(v, h),
        "final_norm.weight": torch.ones(h),
        "lm_head.weight": torch.randn(v, h),
    }
    for layer in range(cfg.num_hidden_layers):
        p = f"layers.{layer}"
        w.update(
            {
                f"{p}.attn.q.weight": torch.randn(nq * dh, h),
                f"{p}.attn.k.weight": torch.randn(nkv * dh, h),
                f"{p}.attn.v.weight": torch.randn(nkv * dh, h),
                f"{p}.attn.o.weight": torch.randn(h, nq * dh),
                f"{p}.mlp.gate.weight": torch.randn(i, h),
                f"{p}.mlp.up.weight": torch.randn(i, h),
                f"{p}.mlp.down.weight": torch.randn(h, i),
                f"{p}.input_norm.weight": torch.ones(h),
                f"{p}.post_attn_norm.weight": torch.ones(h),
            }
        )

    model = DecoderModel(cfg, w)
    ids = torch.randint(0, v, (1, 5))
    logits_new = model.forward(ids)

    inv = build_inv_freq(cfg, device=ids.device)
    cos, sin = build_rope_cos_sin(inv, torch.arange(5)[None, :], dtype=w["embed.weight"].dtype)
    x = w["embed.weight"][ids]
    for li in range(cfg.num_hidden_layers):
        x = transformer_block(x, w, li, cos, sin, cfg)
    x = rms_norm(x, w["final_norm.weight"], cfg.rms_norm_eps)
    logits_old = F.linear(x, w["lm_head.weight"])
    assert torch.allclose(logits_new, logits_old)
