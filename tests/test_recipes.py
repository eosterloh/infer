"""Drop-in recipes: config.json + weights → logits / generate."""

from __future__ import annotations

from pathlib import Path

import torch

from engine.agent_api import inspect_capabilities, load_engine
from engine.config import ModelConfig
from engine.detect import detect_recipe_id
from engine.model import DecoderModel
from engine.weights import load_weights, validate_name_map
from engine.synth import random_engine_weights, write_config, write_hf_folder

_H = 32
_BASE = {
    "vocab_size": 64,
    "hidden_size": _H,
    "intermediate_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "rms_norm_eps": 1e-5,
    "rope_theta": 10000.0,
    "max_position_embeddings": 64,
    "tie_word_embeddings": True,
    "torch_dtype": "float32",
    "hidden_act": "silu",
}


def _run_folder(tmp_path: Path, raw: dict, recipe: str) -> None:
    folder = write_config(tmp_path / recipe, raw)
    cfg = ModelConfig.from_pretrained(folder)
    assert cfg.recipe_id == recipe
    assert detect_recipe_id(raw) == recipe
    caps = inspect_capabilities(folder)
    assert caps.can_run is True
    assert caps.recipe_id == recipe

    engine_w = random_engine_weights(cfg)
    model = DecoderModel(cfg, engine_w)
    ids = torch.randint(0, cfg.vocab_size, (1, 4))
    logits = model.forward(ids)
    assert logits.shape == (1, 4, cfg.vocab_size)
    assert torch.isfinite(logits).all()

    cache = model.make_cache(batch_size=1, device=ids.device, dtype=engine_w["embed.weight"].dtype)
    pre = model.forward(ids, cache=cache)
    step = model.forward(ids[:, -1:], cache=cache)
    assert pre.shape[-1] == cfg.vocab_size
    assert step.shape == (1, 1, cfg.vocab_size)

    write_hf_folder(folder, cfg, engine_w)
    inv_ok = validate_name_map(cfg, __import__("safetensors.torch", fromlist=["load_file"]).load_file(str(folder / "model.safetensors")).keys())
    assert inv_ok
    loaded = load_weights(folder, cfg, device="cpu", dtype="float32")
    model2 = DecoderModel(cfg, loaded)
    logits2 = model2.forward(ids)
    assert logits2.shape == logits.shape
    assert torch.isfinite(logits2).all()

    eng = load_engine(folder, device="cpu", dtype="float32")
    text = eng.generate("hi", max_new_tokens=2, apply_chat_template=False)
    assert isinstance(text, str)


def test_mistral_dropin(tmp_path: Path) -> None:
    raw = {
        **_BASE,
        "architectures": ["MistralForCausalLM"],
        "model_type": "mistral",
        "sliding_window": 16,
    }
    _run_folder(tmp_path, raw, "mistral")


def test_qwen2_dropin(tmp_path: Path) -> None:
    raw = {
        **_BASE,
        "architectures": ["Qwen2ForCausalLM"],
        "model_type": "qwen2",
        "tie_word_embeddings": False,
    }
    _run_folder(tmp_path, raw, "qwen2")


def test_qwen3_dropin(tmp_path: Path) -> None:
    raw = {
        **_BASE,
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "qk_norm": True,
    }
    _run_folder(tmp_path, raw, "qwen3")


def test_yi_dropin(tmp_path: Path) -> None:
    raw = {**_BASE, "architectures": ["YiForCausalLM"], "model_type": "yi"}
    _run_folder(tmp_path, raw, "yi")


def test_gemma_dropin(tmp_path: Path) -> None:
    raw = {**_BASE, "architectures": ["GemmaForCausalLM"], "model_type": "gemma"}
    _run_folder(tmp_path, raw, "gemma")


def test_phi3_dropin(tmp_path: Path) -> None:
    raw = {
        **_BASE,
        "architectures": ["Phi3ForCausalLM"],
        "model_type": "phi3",
        "attention_bias": False,
        "tie_word_embeddings": False,
    }
    _run_folder(tmp_path, raw, "phi3")


def test_mixtral_dropin(tmp_path: Path) -> None:
    raw = {
        **_BASE,
        "architectures": ["MixtralForCausalLM"],
        "model_type": "mixtral",
        "num_local_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 32,
        "tie_word_embeddings": False,
    }
    _run_folder(tmp_path, raw, "mixtral")


def test_llama4_dropin(tmp_path: Path) -> None:
    raw = {
        **_BASE,
        "architectures": ["Llama4ForCausalLM"],
        "model_type": "llama4",
        "num_local_experts": 4,
        "num_experts_per_tok": 1,
        "moe_intermediate_size": 32,
        "interleave_moe_layer_step": 1,
        "tie_word_embeddings": False,
    }
    _run_folder(tmp_path, raw, "llama4")


def test_gpt2_dropin(tmp_path: Path) -> None:
    raw = {
        "architectures": ["GPT2LMHeadModel"],
        "model_type": "gpt2",
        "vocab_size": 64,
        "n_embd": 32,
        "n_head": 4,
        "n_layer": 2,
        "n_inner": 64,
        "n_positions": 64,
        "layer_norm_epsilon": 1e-5,
        "torch_dtype": "float32",
        "tie_word_embeddings": True,
    }
    _run_folder(tmp_path, raw, "gpt2")


def test_gpt_neox_dropin(tmp_path: Path) -> None:
    raw = {
        **_BASE,
        "architectures": ["GPTNeoXForCausalLM"],
        "model_type": "gpt_neox",
        "num_key_value_heads": 4,
        "tie_word_embeddings": False,
        "hidden_act": "gelu",
    }
    _run_folder(tmp_path, raw, "gpt_neox")


def test_gpt_oss_dropin(tmp_path: Path) -> None:
    raw = {
        **_BASE,
        "architectures": ["GptOssForCausalLM"],
        "model_type": "gpt_oss",
        "num_local_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 32,
        "tie_word_embeddings": False,
    }
    _run_folder(tmp_path, raw, "gpt_oss")


def test_deepseek_v3_dropin(tmp_path: Path) -> None:
    raw = {
        **_BASE,
        "architectures": ["DeepseekV3ForCausalLM"],
        "model_type": "deepseek_v3",
        "q_lora_rank": 16,
        "kv_lora_rank": 16,
        "qk_nope_head_dim": 8,
        "qk_rope_head_dim": 8,
        "v_head_dim": 8,
        "first_k_dense_replace": 1,
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "n_shared_experts": 1,
        "moe_intermediate_size": 32,
        "tie_word_embeddings": False,
        "num_hidden_layers": 2,
    }
    _run_folder(tmp_path, raw, "deepseek_v3")


def test_super_latent_mtp_dropin(tmp_path: Path) -> None:
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
        "max_position_embeddings": 32,
        "tie_word_embeddings": False,
        "torch_dtype": "float32",
        "hybrid_override_pattern": "E",
        "n_routed_experts": 4,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 8,
        "moe_shared_expert_intermediate_size": 16,
        "moe_latent_size": 8,
        "routed_scaling_factor": 1.0,
        "mlp_hidden_act": "relu2",
        "n_group": 1,
        "topk_group": 1,
        "num_nextn_predict_layers": 1,
    }
    _run_folder(tmp_path, raw, "nemotron_h")


def test_qwen3_5_dropin(tmp_path: Path) -> None:
    raw = {
        **_BASE,
        "architectures": ["Qwen3_5ForCausalLM"],
        "model_type": "qwen3_5",
        "num_hidden_layers": 2,
        "layer_types": ["linear_attention", "full_attention"],
        "attn_output_gate": True,
        "partial_rotary_factor": 0.5,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 8,
        "linear_value_head_dim": 8,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "tie_word_embeddings": False,
    }
    _run_folder(tmp_path, raw, "qwen3_5")


def test_qwen3_5_nested_text_config(tmp_path: Path) -> None:
    raw = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "language_model_only": False,
        "vision_config": {"hidden_size": 16, "depth": 1},
        "text_config": {
            **_BASE,
            "model_type": "qwen3_5_text",
            "num_hidden_layers": 2,
            "layer_types": ["linear_attention", "full_attention"],
            "attn_output_gate": True,
            "partial_rotary_factor": 0.5,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "mtp_num_hidden_layers": 1,
            "tie_word_embeddings": False,
            "dtype": "float32",
            "rope_parameters": {
                "rope_theta": 1000000.0,
                "partial_rotary_factor": 0.5,
                "rope_type": "default",
            },
        },
    }
    folder = write_config(tmp_path / "qwen38nested", raw)
    cfg = ModelConfig.from_pretrained(folder)
    assert cfg.recipe_id == "qwen3_5"
    assert cfg.attn_output_gate is True
    assert cfg.layers[0].mixer.value == "gated_deltanet"
    assert cfg.layers[1].mixer.value == "attention"
    assert cfg.rope_theta == 1000000.0
    caps = inspect_capabilities(folder)
    assert caps.can_run is True
    assert "vision" in caps.missing
    assert "mtp_decode" in caps.missing
    from engine.maps import is_ignored_hf_name

    assert is_ignored_hf_name("model.visual.blocks.0.attn.proj.bias", cfg)
    assert is_ignored_hf_name("mtp.fc.weight", cfg)
    assert is_ignored_hf_name("mtp.layers.0.mlp.down_proj.weight", cfg)


def test_llama4_nested_text_config(tmp_path: Path) -> None:
    raw = {
        "architectures": ["Llama4ForConditionalGeneration"],
        "model_type": "llama4",
        "text_config": {
            **_BASE,
            "model_type": "llama4_text",
            "num_local_experts": 4,
            "num_experts_per_tok": 1,
            "moe_intermediate_size": 32,
            "interleave_moe_layer_step": 2,
            "tie_word_embeddings": False,
        },
    }
    folder = write_config(tmp_path / "llama4nested", raw)
    cfg = ModelConfig.from_pretrained(folder)
    assert cfg.recipe_id == "llama4"
    assert cfg.n_routed_experts == 4
