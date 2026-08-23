"""HuggingFace / GGUF tensor names → engine names.

Plug-and-play: no caller registers a map. detect.py already picked a recipe;
this file only translates strings.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from engine.config import ModelConfig


def _layer_rest(name: str, prefixes: tuple[str, ...]) -> tuple[str, str, str] | None:
    """Return (prefix, layer_index, rest) if name matches prefix.layers.N.rest."""
    for prefix in prefixes:
        m = re.match(rf"^{re.escape(prefix)}layers\.(\d+)\.(.+)$", name)
        if m:
            return prefix, m.group(1), m.group(2)
    return None


_LLAMA_LAYER = {
    "self_attn.q_proj.weight": "attn.q.weight",
    "self_attn.k_proj.weight": "attn.k.weight",
    "self_attn.v_proj.weight": "attn.v.weight",
    "self_attn.o_proj.weight": "attn.o.weight",
    "self_attn.q_proj.bias": "attn.q.bias",
    "self_attn.k_proj.bias": "attn.k.bias",
    "self_attn.v_proj.bias": "attn.v.bias",
    "self_attn.o_proj.bias": "attn.o.bias",
    "self_attn.q_norm.weight": "attn.q_norm.weight",
    "self_attn.k_norm.weight": "attn.k_norm.weight",
    "self_attn.sinks": "attn.sinks",
    "mlp.gate_proj.weight": "mlp.gate.weight",
    "mlp.up_proj.weight": "mlp.up.weight",
    "mlp.down_proj.weight": "mlp.down.weight",
    "mlp.gate_proj.bias": "mlp.gate.bias",
    "mlp.up_proj.bias": "mlp.up.bias",
    "mlp.down_proj.bias": "mlp.down.bias",
    "mlp.gate_up_proj.weight": "mlp.gate_up.weight",
    "self_attn.qkv_proj.weight": "attn.qkv.weight",
    "input_layernorm.weight": "input_norm.weight",
    "post_attention_layernorm.weight": "post_attn_norm.weight",
    "pre_feedforward_layernorm.weight": "pre_ff_norm.weight",
    "post_feedforward_layernorm.weight": "post_ff_norm.weight",
    "input_layernorm.bias": "input_norm.bias",
    "post_attention_layernorm.bias": "post_attn_norm.bias",
}


def _map_llama_family(hf_name: str) -> str | None:
    if hf_name in {
        "model.embed_tokens.weight",
        "model.language_model.embed_tokens.weight",
        "language_model.model.embed_tokens.weight",
    }:
        return "embed.weight"
    if hf_name in {
        "model.norm.weight",
        "model.language_model.norm.weight",
        "language_model.model.norm.weight",
    }:
        return "final_norm.weight"
    if hf_name in {
        "lm_head.weight",
        "language_model.lm_head.weight",
        "model.lm_head.weight",
    }:
        return "lm_head.weight"

    hit = _layer_rest(
        hf_name,
        (
            "model.language_model.",
            "language_model.model.",
            "model.",
        ),
    )
    if not hit:
        return None
    _, i, rest = hit

    if rest in _LLAMA_LAYER:
        return f"layers.{i}.{_LLAMA_LAYER[rest]}"

    # Mixtral classic experts
    m = re.match(
        r"^block_sparse_moe\.experts\.(\d+)\.(w1|w2|w3)\.weight$", rest
    )
    if m:
        e, which = m.group(1), m.group(2)
        engine = {"w1": "gate", "w3": "up", "w2": "down"}[which]
        return f"layers.{i}.moe.experts.{e}.{engine}.weight"
    if rest == "block_sparse_moe.gate.weight":
        return f"layers.{i}.moe.gate.weight"

    # Llama 4 packed MoE
    if rest == "feed_forward.router.weight":
        return f"layers.{i}.moe.gate.weight"
    if rest == "feed_forward.experts.gate_up_proj":
        return f"layers.{i}.moe.experts.gate_up.weight"
    if rest == "feed_forward.experts.down_proj":
        return f"layers.{i}.moe.experts.down.weight"
    if rest == "feed_forward.shared_expert.gate_proj.weight":
        return f"layers.{i}.moe.shared.gate.weight"
    if rest == "feed_forward.shared_expert.up_proj.weight":
        return f"layers.{i}.moe.shared.up.weight"
    if rest == "feed_forward.shared_expert.down_proj.weight":
        return f"layers.{i}.moe.shared.down.weight"
    if rest == "feed_forward.gate_proj.weight":
        return f"layers.{i}.mlp.gate.weight"
    if rest == "feed_forward.up_proj.weight":
        return f"layers.{i}.mlp.up.weight"
    if rest == "feed_forward.down_proj.weight":
        return f"layers.{i}.mlp.down.weight"

    # GPT-OSS
    if rest == "mlp.router.weight":
        return f"layers.{i}.moe.gate.weight"
    if rest == "mlp.router.bias":
        return f"layers.{i}.moe.gate.bias"
    if rest == "mlp.experts.gate_up_proj":
        return f"layers.{i}.moe.experts.gate_up.weight"
    if rest == "mlp.experts.gate_up_proj_bias":
        return f"layers.{i}.moe.experts.gate_up.bias"
    if rest == "mlp.experts.down_proj":
        return f"layers.{i}.moe.experts.down.weight"
    if rest == "mlp.experts.down_proj_bias":
        return f"layers.{i}.moe.experts.down.bias"

    # DeepSeek MLA + MoE
    ds = {
        "self_attn.q_proj.weight": "attn.q.weight",
        "self_attn.q_a_proj.weight": "attn.q_a.weight",
        "self_attn.q_a_layernorm.weight": "attn.q_a_norm.weight",
        "self_attn.q_b_proj.weight": "attn.q_b.weight",
        "self_attn.kv_a_proj_with_mqa.weight": "attn.kv_a.weight",
        "self_attn.kv_a_layernorm.weight": "attn.kv_a_norm.weight",
        "self_attn.kv_b_proj.weight": "attn.kv_b.weight",
        "self_attn.o_proj.weight": "attn.o.weight",
        "mlp.shared_experts.gate_proj.weight": "moe.shared.gate.weight",
        "mlp.shared_experts.up_proj.weight": "moe.shared.up.weight",
        "mlp.shared_experts.down_proj.weight": "moe.shared.down.weight",
        "mlp.gate.weight": "moe.gate.weight",
        "mlp.gate.e_score_correction_bias": "moe.gate.e_score_correction_bias",
    }
    if rest in ds:
        return f"layers.{i}.{ds[rest]}"
    em = re.match(
        r"^mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$", rest
    )
    if em:
        e, which = em.group(1), em.group(2)
        short = {"gate_proj": "gate", "up_proj": "up", "down_proj": "down"}[which]
        return f"layers.{i}.moe.experts.{e}.{short}.weight"

    return None


def _map_nemotron_h_hf_name(hf_name: str) -> str | None:
    if hf_name == "backbone.embeddings.weight":
        return "embed.weight"
    if hf_name == "backbone.norm_f.weight":
        return "final_norm.weight"
    if hf_name == "lm_head.weight":
        return "lm_head.weight"

    m = re.match(r"^backbone\.layers\.(\d+)\.(.+)$", hf_name)
    if m:
        return _nemotron_layer(m.group(1), m.group(2))

    m = re.match(r"^backbone\.mtp_layers\.(\d+)\.(.+)$", hf_name)
    if m:
        mapped = _nemotron_layer(m.group(1), m.group(2))
        if mapped is None:
            return f"mtp.layers.{m.group(1)}.{m.group(2)}"
        return "mtp." + mapped
    m = re.match(r"^mtp\.layers\.(\d+)\.(.+)$", hf_name)
    if m:
        mapped = _nemotron_layer(m.group(1), m.group(2))
        if mapped is None:
            return hf_name
        return "mtp." + mapped
    if hf_name.startswith("mtp."):
        return hf_name
    return None


def _nemotron_layer(i: str, rest: str) -> str | None:
    if rest == "norm.weight":
        return f"layers.{i}.input_norm.weight"
    attn = {
        "mixer.q_proj.weight": f"layers.{i}.attn.q.weight",
        "mixer.k_proj.weight": f"layers.{i}.attn.k.weight",
        "mixer.v_proj.weight": f"layers.{i}.attn.v.weight",
        "mixer.o_proj.weight": f"layers.{i}.attn.o.weight",
    }
    if rest in attn:
        return attn[rest]
    mlp = {
        "mixer.up_proj.weight": f"layers.{i}.mlp.up.weight",
        "mixer.down_proj.weight": f"layers.{i}.mlp.down.weight",
    }
    if rest in mlp:
        return mlp[rest]
    if rest == "mixer.gate.weight":
        return f"layers.{i}.moe.gate.weight"
    if rest == "mixer.gate.e_score_correction_bias":
        return f"layers.{i}.moe.gate.e_score_correction_bias"
    if rest == "mixer.shared_experts.up_proj.weight":
        return f"layers.{i}.moe.shared.up.weight"
    if rest == "mixer.shared_experts.down_proj.weight":
        return f"layers.{i}.moe.shared.down.weight"
    if rest == "mixer.latent_down.weight":
        return f"layers.{i}.moe.latent_down.weight"
    if rest == "mixer.latent_up.weight":
        return f"layers.{i}.moe.latent_up.weight"
    em = re.match(r"^mixer\.experts\.(\d+)\.(up_proj|down_proj)\.weight$", rest)
    if em:
        e, which = em.group(1), em.group(2)
        short = "up" if which == "up_proj" else "down"
        return f"layers.{i}.moe.experts.{e}.{short}.weight"
    mamba = {
        "mixer.in_proj.weight": f"layers.{i}.mamba.in_proj.weight",
        "mixer.out_proj.weight": f"layers.{i}.mamba.out_proj.weight",
        "mixer.conv1d.weight": f"layers.{i}.mamba.conv1d.weight",
        "mixer.conv1d.bias": f"layers.{i}.mamba.conv1d.bias",
        "mixer.A_log": f"layers.{i}.mamba.A_log",
        "mixer.D": f"layers.{i}.mamba.D",
        "mixer.dt_bias": f"layers.{i}.mamba.dt_bias",
        "mixer.norm.weight": f"layers.{i}.mamba.norm.weight",
    }
    return mamba.get(rest)


def _map_gpt2(hf_name: str) -> str | None:
    if hf_name in {"transformer.wte.weight", "wte.weight"}:
        return "embed.weight"
    if hf_name in {"transformer.wpe.weight", "wpe.weight"}:
        return "pos_embed.weight"
    if hf_name in {"transformer.ln_f.weight", "ln_f.weight"}:
        return "final_norm.weight"
    if hf_name in {"transformer.ln_f.bias", "ln_f.bias"}:
        return "final_norm.bias"
    if hf_name == "lm_head.weight":
        return "lm_head.weight"
    m = re.match(r"^transformer\.h\.(\d+)\.(.+)$", hf_name)
    if not m:
        return None
    i, rest = m.group(1), m.group(2)
    table = {
        "ln_1.weight": "input_norm.weight",
        "ln_1.bias": "input_norm.bias",
        "ln_2.weight": "post_attn_norm.weight",
        "ln_2.bias": "post_attn_norm.bias",
        "attn.c_attn.weight": "attn.c_attn.weight",
        "attn.c_attn.bias": "attn.c_attn.bias",
        "attn.c_proj.weight": "attn.c_proj.weight",
        "attn.c_proj.bias": "attn.c_proj.bias",
        "mlp.c_fc.weight": "mlp.c_fc.weight",
        "mlp.c_fc.bias": "mlp.c_fc.bias",
        "mlp.c_proj.weight": "mlp.c_proj.weight",
        "mlp.c_proj.bias": "mlp.c_proj.bias",
    }
    mapped = table.get(rest)
    return f"layers.{i}.{mapped}" if mapped else None


def _map_gpt_neox(hf_name: str) -> str | None:
    if hf_name in {"gpt_neox.embed_in.weight", "embed_in.weight"}:
        return "embed.weight"
    if hf_name in {"gpt_neox.final_layer_norm.weight", "final_layer_norm.weight"}:
        return "final_norm.weight"
    if hf_name in {"gpt_neox.final_layer_norm.bias", "final_layer_norm.bias"}:
        return "final_norm.bias"
    if hf_name in {"embed_out.weight", "lm_head.weight"}:
        return "lm_head.weight"
    m = re.match(r"^gpt_neox\.layers\.(\d+)\.(.+)$", hf_name)
    if not m:
        return None
    i, rest = m.group(1), m.group(2)
    table = {
        "input_layernorm.weight": "input_norm.weight",
        "input_layernorm.bias": "input_norm.bias",
        "post_attention_layernorm.weight": "post_attn_norm.weight",
        "post_attention_layernorm.bias": "post_attn_norm.bias",
        "attention.query_key_value.weight": "attn.qkv.weight",
        "attention.query_key_value.bias": "attn.qkv.bias",
        "attention.dense.weight": "attn.o.weight",
        "attention.dense.bias": "attn.o.bias",
        "mlp.dense_h_to_4h.weight": "mlp.up.weight",
        "mlp.dense_h_to_4h.bias": "mlp.up.bias",
        "mlp.dense_4h_to_h.weight": "mlp.down.weight",
        "mlp.dense_4h_to_h.bias": "mlp.down.bias",
    }
    mapped = table.get(rest)
    return f"layers.{i}.{mapped}" if mapped else None


_GGUF_SKIP = {
    "rope_freqs.weight",
    "rope_factors_long.weight",
    "rope_factors_short.weight",
}


def map_gguf_name(gguf_name: str) -> str | None:
    """llama.cpp GGUF tensor names → engine names (dense Llama-like)."""
    if gguf_name in _GGUF_SKIP:
        return None
    if gguf_name in {"token_embd.weight", "token_embd"}:
        return "embed.weight"
    if gguf_name in {"output_norm.weight", "output_norm"}:
        return "final_norm.weight"
    if gguf_name in {"output.weight", "output"}:
        return "lm_head.weight"
    m = re.match(r"^blk\.(\d+)\.(.+)$", gguf_name)
    if not m:
        return None
    i, rest = m.group(1), m.group(2)
    table = {
        "attn_q.weight": "attn.q.weight",
        "attn_k.weight": "attn.k.weight",
        "attn_v.weight": "attn.v.weight",
        "attn_output.weight": "attn.o.weight",
        "ffn_gate.weight": "mlp.gate.weight",
        "ffn_up.weight": "mlp.up.weight",
        "ffn_down.weight": "mlp.down.weight",
        "attn_norm.weight": "input_norm.weight",
        "ffn_norm.weight": "post_attn_norm.weight",
        "attn_q.bias": "attn.q.bias",
        "attn_k.bias": "attn.k.bias",
        "attn_v.bias": "attn.v.bias",
    }
    mapped = table.get(rest)
    return f"layers.{i}.{mapped}" if mapped else None


def map_hf_name(hf_name: str, config: ModelConfig | None = None) -> str | None:
    """Dispatch HF→engine rename from auto-detected recipe."""
    recipe = getattr(config, "recipe_id", "") if config else ""
    if not recipe and config is not None:
        from engine.detect import detect_recipe_id

        recipe = detect_recipe_id(config.raw or {"model_type": config.model_type})

    if hf_name.startswith(("blk.", "token_embd", "output_norm", "output.")):
        return map_gguf_name(hf_name)

    if recipe == "gpt2":
        return _map_gpt2(hf_name)
    if recipe == "gpt_neox":
        return _map_gpt_neox(hf_name)
    if recipe == "nemotron_h" or (
        hf_name.startswith("backbone.") and recipe != "llama"
    ):
        return _map_nemotron_h_hf_name(hf_name)
    return _map_llama_family(hf_name)


# Skip scale/zero tensors; quant.py consumes them next to the weight.
_QUANT_SUFFIXES = (
    ".weight_scale",
    ".weight_scale_inv",
    ".input_scale",
    ".output_scale",
    ".weight_zero_point",
    ".qzeros",
    ".scales",
    ".g_idx",
)


def is_quant_aux(hf_name: str) -> bool:
    return any(hf_name.endswith(s) for s in _QUANT_SUFFIXES) or hf_name.endswith(
        ".weight_scale_2"
    )
