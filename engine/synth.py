"""Tiny drop-in folders for recipe tests and local experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from engine.config import ModelConfig
from engine.maps import map_hf_name


def write_config(folder: Path, raw: dict[str, Any]) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.json").write_text(json.dumps(raw, indent=2))
    return folder


def random_engine_weights(cfg: ModelConfig, seed: int = 0) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    state: dict[str, torch.Tensor] = {}
    for name, shape in cfg.expected_shapes().items():
        if name.endswith((".weight",)) and "norm" in name:
            state[name] = torch.ones(shape)
        elif name.endswith("bias") and "norm" in name:
            state[name] = torch.zeros(shape)
        else:
            state[name] = torch.randn(shape)
    if cfg.tie_word_embeddings:
        state["lm_head.weight"] = state["embed.weight"]
    return state


def _hf_candidates(cfg: ModelConfig) -> list[str]:
    names = [
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
        "model.language_model.embed_tokens.weight",
        "language_model.lm_head.weight",
        "backbone.embeddings.weight",
        "backbone.norm_f.weight",
        "transformer.wte.weight",
        "transformer.wpe.weight",
        "transformer.ln_f.weight",
        "transformer.ln_f.bias",
        "gpt_neox.embed_in.weight",
        "gpt_neox.final_layer_norm.weight",
        "gpt_neox.final_layer_norm.bias",
        "embed_out.weight",
    ]
    n = cfg.num_hidden_layers
    n_e = cfg.n_routed_experts or 0
    n_mtp = cfg.num_nextn_predict_layers or 0
    for i in range(n):
        names.extend(
            [
                f"model.layers.{i}.self_attn.q_proj.weight",
                f"model.layers.{i}.self_attn.k_proj.weight",
                f"model.layers.{i}.self_attn.v_proj.weight",
                f"model.layers.{i}.self_attn.o_proj.weight",
                f"model.layers.{i}.self_attn.q_proj.bias",
                f"model.layers.{i}.self_attn.k_proj.bias",
                f"model.layers.{i}.self_attn.v_proj.bias",
                f"model.layers.{i}.self_attn.o_proj.bias",
                f"model.layers.{i}.self_attn.q_norm.weight",
                f"model.layers.{i}.self_attn.k_norm.weight",
                f"model.layers.{i}.self_attn.sinks",
                f"model.layers.{i}.self_attn.qkv_proj.weight",
                f"model.layers.{i}.linear_attn.in_proj_qkv.weight",
                f"model.layers.{i}.linear_attn.in_proj_z.weight",
                f"model.layers.{i}.linear_attn.in_proj_b.weight",
                f"model.layers.{i}.linear_attn.in_proj_a.weight",
                f"model.layers.{i}.linear_attn.conv1d.weight",
                f"model.layers.{i}.linear_attn.A_log",
                f"model.layers.{i}.linear_attn.dt_bias",
                f"model.layers.{i}.linear_attn.norm.weight",
                f"model.layers.{i}.linear_attn.out_proj.weight",
                f"model.language_model.layers.{i}.linear_attn.in_proj_qkv.weight",
                f"model.layers.{i}.mlp.gate_proj.weight",
                f"model.layers.{i}.mlp.up_proj.weight",
                f"model.layers.{i}.mlp.down_proj.weight",
                f"model.layers.{i}.mlp.gate_up_proj.weight",
                f"model.layers.{i}.input_layernorm.weight",
                f"model.layers.{i}.post_attention_layernorm.weight",
                f"model.layers.{i}.input_layernorm.bias",
                f"model.layers.{i}.post_attention_layernorm.bias",
                f"model.layers.{i}.block_sparse_moe.gate.weight",
                f"model.layers.{i}.mlp.router.weight",
                f"model.layers.{i}.mlp.router.bias",
                f"model.layers.{i}.mlp.experts.gate_up_proj",
                f"model.layers.{i}.mlp.experts.gate_up_proj_bias",
                f"model.layers.{i}.mlp.experts.down_proj",
                f"model.layers.{i}.mlp.experts.down_proj_bias",
                f"model.layers.{i}.feed_forward.router.weight",
                f"model.layers.{i}.feed_forward.experts.gate_up_proj",
                f"model.layers.{i}.feed_forward.experts.down_proj",
                f"model.layers.{i}.feed_forward.shared_expert.gate_proj.weight",
                f"model.layers.{i}.feed_forward.shared_expert.up_proj.weight",
                f"model.layers.{i}.feed_forward.shared_expert.down_proj.weight",
                f"model.layers.{i}.feed_forward.gate_proj.weight",
                f"model.layers.{i}.feed_forward.up_proj.weight",
                f"model.layers.{i}.feed_forward.down_proj.weight",
                f"model.layers.{i}.self_attn.q_a_proj.weight",
                f"model.layers.{i}.self_attn.q_a_layernorm.weight",
                f"model.layers.{i}.self_attn.q_b_proj.weight",
                f"model.layers.{i}.self_attn.kv_a_proj_with_mqa.weight",
                f"model.layers.{i}.self_attn.kv_a_layernorm.weight",
                f"model.layers.{i}.self_attn.kv_b_proj.weight",
                f"model.layers.{i}.mlp.gate.weight",
                f"model.layers.{i}.mlp.gate.e_score_correction_bias",
                f"model.layers.{i}.mlp.shared_experts.gate_proj.weight",
                f"model.layers.{i}.mlp.shared_experts.up_proj.weight",
                f"model.layers.{i}.mlp.shared_experts.down_proj.weight",
                f"transformer.h.{i}.ln_1.weight",
                f"transformer.h.{i}.ln_1.bias",
                f"transformer.h.{i}.ln_2.weight",
                f"transformer.h.{i}.ln_2.bias",
                f"transformer.h.{i}.attn.c_attn.weight",
                f"transformer.h.{i}.attn.c_attn.bias",
                f"transformer.h.{i}.attn.c_proj.weight",
                f"transformer.h.{i}.attn.c_proj.bias",
                f"transformer.h.{i}.mlp.c_fc.weight",
                f"transformer.h.{i}.mlp.c_fc.bias",
                f"transformer.h.{i}.mlp.c_proj.weight",
                f"transformer.h.{i}.mlp.c_proj.bias",
                f"gpt_neox.layers.{i}.input_layernorm.weight",
                f"gpt_neox.layers.{i}.input_layernorm.bias",
                f"gpt_neox.layers.{i}.post_attention_layernorm.weight",
                f"gpt_neox.layers.{i}.post_attention_layernorm.bias",
                f"gpt_neox.layers.{i}.attention.query_key_value.weight",
                f"gpt_neox.layers.{i}.attention.query_key_value.bias",
                f"gpt_neox.layers.{i}.attention.dense.weight",
                f"gpt_neox.layers.{i}.attention.dense.bias",
                f"gpt_neox.layers.{i}.mlp.dense_h_to_4h.weight",
                f"gpt_neox.layers.{i}.mlp.dense_h_to_4h.bias",
                f"gpt_neox.layers.{i}.mlp.dense_4h_to_h.weight",
                f"gpt_neox.layers.{i}.mlp.dense_4h_to_h.bias",
                f"backbone.layers.{i}.norm.weight",
                f"backbone.layers.{i}.mixer.q_proj.weight",
                f"backbone.layers.{i}.mixer.k_proj.weight",
                f"backbone.layers.{i}.mixer.v_proj.weight",
                f"backbone.layers.{i}.mixer.o_proj.weight",
                f"backbone.layers.{i}.mixer.up_proj.weight",
                f"backbone.layers.{i}.mixer.down_proj.weight",
                f"backbone.layers.{i}.mixer.gate.weight",
                f"backbone.layers.{i}.mixer.gate.e_score_correction_bias",
                f"backbone.layers.{i}.mixer.shared_experts.up_proj.weight",
                f"backbone.layers.{i}.mixer.shared_experts.down_proj.weight",
                f"backbone.layers.{i}.mixer.latent_down.weight",
                f"backbone.layers.{i}.mixer.latent_up.weight",
                f"backbone.layers.{i}.mixer.in_proj.weight",
                f"backbone.layers.{i}.mixer.out_proj.weight",
                f"backbone.layers.{i}.mixer.conv1d.weight",
                f"backbone.layers.{i}.mixer.conv1d.bias",
                f"backbone.layers.{i}.mixer.A_log",
                f"backbone.layers.{i}.mixer.D",
                f"backbone.layers.{i}.mixer.dt_bias",
                f"backbone.layers.{i}.mixer.norm.weight",
            ]
        )
        for e in range(n_e):
            names.extend(
                [
                    f"model.layers.{i}.block_sparse_moe.experts.{e}.w1.weight",
                    f"model.layers.{i}.block_sparse_moe.experts.{e}.w2.weight",
                    f"model.layers.{i}.block_sparse_moe.experts.{e}.w3.weight",
                    f"model.layers.{i}.mlp.experts.{e}.gate_proj.weight",
                    f"model.layers.{i}.mlp.experts.{e}.up_proj.weight",
                    f"model.layers.{i}.mlp.experts.{e}.down_proj.weight",
                    f"backbone.layers.{i}.mixer.experts.{e}.up_proj.weight",
                    f"backbone.layers.{i}.mixer.experts.{e}.down_proj.weight",
                ]
            )
    for j in range(n_mtp):
        names.extend(
            [
                f"mtp.layers.{j}.enorm.weight",
                f"mtp.layers.{j}.hnorm.weight",
                f"mtp.layers.{j}.eh_proj.weight",
            ]
        )
    return names


def invert_engine_to_hf(cfg: ModelConfig) -> dict[str, str]:
    inv: dict[str, str] = {}
    for hf in _hf_candidates(cfg):
        eng = map_hf_name(hf, cfg)
        if eng and eng not in inv:
            inv[eng] = hf
    return inv


def write_hf_folder(folder: Path, cfg: ModelConfig, engine_state: dict[str, torch.Tensor]) -> None:
    inv = invert_engine_to_hf(cfg)
    hf_state: dict[str, torch.Tensor] = {}
    missing = []
    for name, tensor in engine_state.items():
        if name == "lm_head.weight" and cfg.tie_word_embeddings:
            continue
        hf = inv.get(name)
        if hf is None:
            missing.append(name)
            continue
        t = tensor
        if cfg.recipe_id == "gpt2" and name.endswith(
            (".attn.c_attn.weight", ".attn.c_proj.weight", ".mlp.c_fc.weight", ".mlp.c_proj.weight")
        ):
            t = tensor.t().contiguous()
        hf_state[hf] = t.contiguous()
    if missing:
        raise KeyError(f"no HF name for engine tensors: {missing[:12]}")
    save_file(hf_state, str(folder / "model.safetensors"))
