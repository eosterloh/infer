"""Capability discovery — what this checkpoint / config can do.

Agents inspect this before calling generate, so MTP / Mamba / MoE / quant
light up from the model directory rather than hard-coded switches.
"""

from __future__ import annotations

from typing import Any

from engine.config import ModelConfig
from engine.schedule import FfnKind, MixerKind, build_schedule


def describe_capabilities(config: ModelConfig) -> dict[str, Any]:
    layers = config.layers or build_schedule(config)
    has_attn = any(s.mixer == MixerKind.ATTENTION for s in layers)
    has_mamba = any(s.mixer == MixerKind.MAMBA2 for s in layers)
    has_moe = any(s.ffn == FfnKind.MOE for s in layers)
    has_dense_mlp = any(s.ffn == FfnKind.DENSE_MLP for s in layers)

    # MTP is Super-only; present when config advertises MTP layers / heads.
    raw = config.raw or {}
    mtp_layers = raw.get("num_nextn_predict_layers") or raw.get("mtp_num_layers")
    has_mtp = bool(mtp_layers) and int(mtp_layers) > 0

    quant = "bf16"
    if "nvfp4" in (config.torch_dtype or "").lower() or raw.get("quantization_config"):
        qcfg = raw.get("quantization_config") or {}
        quant = str(qcfg.get("quant_method") or config.torch_dtype)

    return {
        "model_type": config.model_type,
        "architectures": list(config.architectures),
        "num_layers": config.num_hidden_layers,
        "hidden_size": config.hidden_size,
        "vocab_size": config.vocab_size,
        "max_position_embeddings": config.max_position_embeddings,
        "dtype": config.torch_dtype,
        "quantization": quant,
        "features": {
            "attention": has_attn,
            "mamba2": has_mamba,
            "moe": has_moe,
            "dense_mlp": has_dense_mlp,
            "rope": (config.model_type or "").lower()
            not in {"nemotron_h", "nemotronh"},
            "mtp": has_mtp,
            "kv_cache": has_attn,
            "ssm_cache": has_mamba,
        },
        "moe": {
            "n_routed_experts": config.n_routed_experts,
            "num_experts_per_tok": config.num_experts_per_tok,
            "moe_intermediate_size": config.moe_intermediate_size,
        }
        if has_moe
        else None,
        "mamba": {
            "mamba_num_heads": config.mamba_num_heads,
            "mamba_head_dim": config.mamba_head_dim,
            "ssm_state_size": config.ssm_state_size,
            "n_groups": config.n_groups,
            "conv_kernel": config.conv_kernel,
        }
        if has_mamba
        else None,
        "mtp": {"num_layers": int(mtp_layers)} if has_mtp else None,
        "hybrid_override_pattern": config.hybrid_override_pattern,
        "schedule_summary": _schedule_summary(layers),
    }


def _schedule_summary(layers) -> dict[str, int]:
    out: dict[str, int] = {}
    for s in layers:
        key = f"{s.mixer.value}+{s.ffn.value}"
        out[key] = out.get(key, 0) + 1
    return out
