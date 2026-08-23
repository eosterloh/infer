"""NVFP4 dequant + GGUF drop-in."""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import save_file

from engine.config import ModelConfig
from engine.gguf import find_gguf, load_gguf_state, write_gguf_f32
from engine.model import DecoderModel
from engine.quant import dequant_nvfp4, pack_nvfp4
from engine.weights import load_weights
from engine.synth import random_engine_weights, write_config

_LLAMA = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "vocab_size": 32,
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_hidden_layers": 1,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 4,
    "rms_norm_eps": 1e-5,
    "rope_theta": 10000.0,
    "max_position_embeddings": 32,
    "tie_word_embeddings": True,
    "torch_dtype": "float32",
    "hidden_act": "silu",
}


def test_nvfp4_roundtrip_pack() -> None:
    torch.manual_seed(0)
    w = torch.randn(8, 16)
    packed, scales = pack_nvfp4(w, group_size=16)
    rec = dequant_nvfp4(packed, scales, tuple(w.shape), group_size=16)
    assert rec.shape == w.shape
    # E2M1 is coarse; just require finite and correlated.
    assert torch.isfinite(rec).all()
    assert torch.corrcoef(torch.stack([w.reshape(-1), rec.reshape(-1)]))[0, 1] > 0.9


def test_nvfp4_folder_loads(tmp_path: Path) -> None:
    raw = {**_LLAMA, "quantization_config": {"quant_method": "nvfp4"}}
    folder = write_config(tmp_path / "nvfp4", raw)
    cfg = ModelConfig.from_pretrained(folder)
    engine_w = random_engine_weights(cfg)
    hf: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": engine_w["embed.weight"],
        "model.norm.weight": engine_w["final_norm.weight"],
        "model.layers.0.self_attn.q_proj.weight": engine_w["layers.0.attn.q.weight"],
        "model.layers.0.self_attn.k_proj.weight": engine_w["layers.0.attn.k.weight"],
        "model.layers.0.self_attn.v_proj.weight": engine_w["layers.0.attn.v.weight"],
        "model.layers.0.self_attn.o_proj.weight": engine_w["layers.0.attn.o.weight"],
        "model.layers.0.mlp.gate_proj.weight": engine_w["layers.0.mlp.gate.weight"],
        "model.layers.0.mlp.up_proj.weight": engine_w["layers.0.mlp.up.weight"],
        "model.layers.0.mlp.down_proj.weight": engine_w["layers.0.mlp.down.weight"],
        "model.layers.0.input_layernorm.weight": engine_w["layers.0.input_norm.weight"],
        "model.layers.0.post_attention_layernorm.weight": engine_w["layers.0.post_attn_norm.weight"],
    }
    packed, scales = pack_nvfp4(hf["model.layers.0.mlp.down_proj.weight"])
    hf["model.layers.0.mlp.down_proj.weight"] = packed
    hf["model.layers.0.mlp.down_proj.weight_scale"] = scales
    save_file(hf, str(folder / "model.safetensors"))
    loaded = load_weights(folder, cfg, device="cpu", dtype="float32")
    assert loaded["layers.0.mlp.down.weight"].dtype == torch.float32
    assert torch.isfinite(loaded["layers.0.mlp.down.weight"]).all()
    logits = DecoderModel(cfg, loaded).forward(torch.randint(0, 32, (1, 3)))
    assert torch.isfinite(logits).all()


def test_gguf_dropin(tmp_path: Path) -> None:
    folder = tmp_path / "ggufmod"
    folder.mkdir()
    h, i, v, nq, nkv, dh = 16, 32, 32, 4, 2, 4
    tensors = {
        "token_embd.weight": torch.randn(v, h),
        "output_norm.weight": torch.ones(h),
        "blk.0.attn_q.weight": torch.randn(nq * dh, h),
        "blk.0.attn_k.weight": torch.randn(nkv * dh, h),
        "blk.0.attn_v.weight": torch.randn(nkv * dh, h),
        "blk.0.attn_output.weight": torch.randn(h, nq * dh),
        "blk.0.ffn_gate.weight": torch.randn(i, h),
        "blk.0.ffn_up.weight": torch.randn(i, h),
        "blk.0.ffn_down.weight": torch.randn(h, i),
        "blk.0.attn_norm.weight": torch.ones(h),
        "blk.0.ffn_norm.weight": torch.ones(h),
    }
    path = folder / "model.gguf"
    write_gguf_f32(
        path,
        tensors,
        {
            "general.architecture": "llama",
            "llama.embedding_length": h,
            "llama.feed_forward_length": i,
            "llama.block_count": 1,
            "llama.attention.head_count": nq,
            "llama.attention.head_count_kv": nkv,
            "llama.context_length": 32,
            "llama.attention.layer_norm_rms_epsilon": 1e-5,
            "llama.rope.freq_base": 10000.0,
            "llama.vocab_size": v,
        },
    )
    assert find_gguf(folder) == path
    cfg = ModelConfig.from_pretrained(folder)
    assert cfg.recipe_id == "llama"
    loaded = load_weights(folder, cfg, device="cpu", dtype="float32")
    logits = DecoderModel(cfg, loaded).forward(torch.randint(0, v, (1, 2)))
    assert logits.shape == (1, 2, v)
    meta, state = load_gguf_state(path)
    assert "embed.weight" in state
    assert meta["general.architecture"] == "llama"
