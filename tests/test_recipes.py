"""Drop-in recipes: config.json + weights → logits / generate."""

from __future__ import annotations

from pathlib import Path

import torch

from engine.agent_api import inspect_capabilities, load_engine
from engine.config import ModelConfig
from engine.detect import detect_recipe_id, detect_missing
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


def _decode_matches_prefill(model, ids: torch.Tensor, *, atol: float, rtol: float) -> None:
    """A decode step must land on the logits the full pass gives for that row.

    Decode takes different code than prefill — fused attention over the cache, a
    GEMV instead of a GEMM, a grouped dispatch instead of a loop — so this is the
    assertion that catches a fast path whose math drifted from the reference.
    """
    full = model.forward(ids)
    warm = model.make_cache(batch_size=1, device=ids.device, dtype=model.dtype)
    model.forward(ids[:, :-1], cache=warm)
    decoded = model.forward(ids[:, -1:], cache=warm)
    torch.testing.assert_close(
        decoded[0, -1].float(), full[0, -1].float(), atol=atol, rtol=rtol
    )


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

    long_ids = torch.cat([ids, ids[:, :1]], dim=1)
    _decode_matches_prefill(model, long_ids, atol=2e-4, rtol=2e-4)

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

    # Only load_engine stacks MoE experts, so the grouped dispatch the fused
    # kernels exist for is unreachable from the model built above.
    _decode_matches_prefill(eng.model, long_ids, atol=2e-4, rtol=2e-4)

    if torch.cuda.is_available():
        # The same assertion where it counts: real kernels, BF16, on the GPU.
        # Every recipe in this file reaches it, which is the only CUDA coverage
        # the integration tests have.
        gpu = load_engine(folder, device="cuda", dtype="bfloat16")
        _decode_matches_prefill(
            gpu.model, long_ids.to("cuda"), atol=6e-2, rtol=6e-2
        )
        del gpu
        torch.cuda.empty_cache()


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
        "num_nextn_predict_layers": 1,
    }
    _run_folder(tmp_path, raw, "deepseek_v3")
    assert "mtp_decode" not in detect_missing(raw, "deepseek")


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
    assert caps.missing == ()
    assert any("native MTP" in note for note in caps.notes)
    assert any("image/video tower" in note for note in caps.notes)
    from engine.maps import is_ignored_hf_name

    assert is_ignored_hf_name("model.visual.blocks.0.attn.proj.bias", cfg)
    assert is_ignored_hf_name("mtp.fc.weight", cfg)
    assert is_ignored_hf_name("mtp.layers.0.mlp.down_proj.weight", cfg)


def test_qwen3_5_defaults_and_nested_text_precedence(tmp_path: Path) -> None:
    raw = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "hidden_size": 999,  # wrapper metadata must not replace text_config
        "text_config": {
            **_BASE,
            "model_type": "qwen3_5_text",
            "num_hidden_layers": 1,
            "layer_types": ["linear_attention"],
            "tie_word_embeddings": False,
        },
    }
    cfg = ModelConfig.from_pretrained(write_config(tmp_path / "defaults", raw))
    assert cfg.hidden_size == _BASE["hidden_size"]
    assert cfg.partial_rotary_factor == 0.25
    assert cfg.linear_num_key_heads == 16
    assert cfg.linear_num_value_heads == 32
    assert cfg.linear_key_head_dim == 128
    assert cfg.linear_value_head_dim == 128
    assert cfg.linear_conv_kernel_dim == 4


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


def test_granite_dropin(tmp_path: Path) -> None:
    raw = {
        **_BASE,
        "architectures": ["GraniteForCausalLM"],
        "model_type": "granite",
        "embedding_multiplier": 1.0,
        "residual_multiplier": 1.0,
        "attention_multiplier": 0.125,
        "logits_scaling": 1.0,
        "tie_word_embeddings": False,
    }
    _run_folder(tmp_path, raw, "granite")


def test_olmo_dropin(tmp_path: Path) -> None:
    raw = {
        **_BASE,
        "architectures": ["OlmoForCausalLM"],
        "model_type": "olmo",
        "clip_qkv": 8.0,
        "tie_word_embeddings": False,
    }
    _run_folder(tmp_path, raw, "olmo")
    cfg = ModelConfig.from_pretrained(write_config(tmp_path / "olmoshapes", raw))
    shapes = cfg.expected_shapes()
    assert not any("norm" in k for k in shapes)


def test_olmo2_dropin(tmp_path: Path) -> None:
    raw = {
        **_BASE,
        "architectures": ["Olmo2ForCausalLM"],
        "model_type": "olmo2",
        "qk_norm": True,
        "tie_word_embeddings": False,
    }
    _run_folder(tmp_path, raw, "olmo2")
    cfg = ModelConfig.from_pretrained(write_config(tmp_path / "olmo2shapes", raw))
    shapes = cfg.expected_shapes()
    assert "layers.0.input_norm.weight" not in shapes
    assert "layers.0.post_attn_norm.weight" in shapes
    assert "layers.0.post_ff_norm.weight" in shapes
    assert shapes["layers.0.attn.q_norm.weight"] == (cfg.nq * cfg.head_dim,)


def test_smollm3_dropin(tmp_path: Path) -> None:
    raw = {
        **_BASE,
        "architectures": ["SmolLM3ForCausalLM"],
        "model_type": "smollm3",
        "no_rope_layers": [1, 0],
        "tie_word_embeddings": True,
    }
    _run_folder(tmp_path, raw, "smollm3")


def test_starcoder2_dropin(tmp_path: Path) -> None:
    raw = {
        **_BASE,
        "architectures": ["Starcoder2ForCausalLM"],
        "model_type": "starcoder2",
        "hidden_act": "gelu_pytorch_tanh",
        "use_bias": True,
        "norm_epsilon": 1e-5,
        "tie_word_embeddings": False,
    }
    _run_folder(tmp_path, raw, "starcoder2")


def test_nemotron_dropin(tmp_path: Path) -> None:
    raw = {
        **_BASE,
        "architectures": ["NemotronForCausalLM"],
        "model_type": "nemotron",
        "hidden_act": "relu2",
        "norm_eps": 1e-5,
        "tie_word_embeddings": False,
    }
    _run_folder(tmp_path, raw, "nemotron")
    cfg = ModelConfig.from_pretrained(write_config(tmp_path / "nemotronshapes", raw))
    shapes = cfg.expected_shapes()
    assert "layers.0.mlp.gate.weight" not in shapes
    assert "layers.0.mlp.up.weight" in shapes
    assert cfg.partial_rotary_factor == 0.5
    assert cfg.uses_swiglu is False


def test_gemma2_dropin(tmp_path: Path) -> None:
    raw = {
        **_BASE,
        "architectures": ["Gemma2ForCausalLM"],
        "model_type": "gemma2",
        "hidden_activation": "gelu_pytorch_tanh",
        "query_pre_attn_scalar": 8,
        "sliding_window": 16,
        "attn_logit_softcapping": 50.0,
        "final_logit_softcapping": 30.0,
        "layer_types": ["sliding_attention", "full_attention"],
    }
    _run_folder(tmp_path, raw, "gemma2")
    cfg = ModelConfig.from_pretrained(write_config(tmp_path / "gemma2shapes", raw))
    shapes = cfg.expected_shapes()
    assert "layers.0.pre_ff_norm.weight" in shapes
    assert "layers.0.post_ff_norm.weight" in shapes
    assert "layers.0.post_attn_norm.weight" in shapes
    assert "layers.0.input_norm.weight" in shapes


def test_gemma3_dropin(tmp_path: Path) -> None:
    raw = {
        **_BASE,
        "architectures": ["Gemma3ForCausalLM"],
        "model_type": "gemma3",
        "hidden_activation": "gelu_pytorch_tanh",
        "query_pre_attn_scalar": 8,
        "qk_norm": True,
        "layer_types": ["full_attention", "full_attention"],
    }
    _run_folder(tmp_path, raw, "gemma3")


def test_gemma3_mixed_sliding_full_dropin(tmp_path: Path) -> None:
    raw = {
        **_BASE,
        "architectures": ["Gemma3ForCausalLM"],
        "model_type": "gemma3",
        "hidden_activation": "gelu_pytorch_tanh",
        "query_pre_attn_scalar": 8,
        "qk_norm": True,
        "sliding_window": 4,
        "layer_types": ["sliding_attention", "full_attention"],
        "rope_theta": 1_000_000.0,
        "rope_parameters": {
            "full_attention": {"rope_type": "default", "rope_theta": 1_000_000.0},
            "sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0},
        },
    }
    _run_folder(tmp_path, raw, "gemma3")
    cfg = ModelConfig.from_pretrained(write_config(tmp_path / "gemma3mixedshapes", raw))
    assert cfg.layer_types == ("sliding_attention", "full_attention")
    assert cfg.qk_norm is True
    assert cfg.residual_kind == "gemma2"


def test_gemma3_nested_text_config(tmp_path: Path) -> None:
    raw = {
        "architectures": ["Gemma3ForConditionalGeneration"],
        "model_type": "gemma3",
        "text_config": {
            **_BASE,
            "model_type": "gemma3_text",
            "hidden_activation": "gelu_pytorch_tanh",
            "query_pre_attn_scalar": 8,
            "layer_types": ["full_attention", "full_attention"],
        },
    }
    cfg = ModelConfig.from_pretrained(write_config(tmp_path / "gemma3nested", raw))
    assert cfg.recipe_id == "gemma3"
    assert cfg.qk_norm is True
    assert cfg.hidden_size == _H


def test_qwen3_moe_dropin(tmp_path: Path) -> None:
    raw = {
        **_BASE,
        "architectures": ["Qwen3MoeForCausalLM"],
        "model_type": "qwen3_moe",
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 32,
        "decoder_sparse_step": 1,
        "mlp_only_layers": [],
        "tie_word_embeddings": False,
    }
    _run_folder(tmp_path, raw, "qwen3_moe")


def test_qwen2_moe_dropin(tmp_path: Path) -> None:
    raw = {
        **_BASE,
        "architectures": ["Qwen2MoeForCausalLM"],
        "model_type": "qwen2_moe",
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 32,
        "shared_expert_intermediate_size": 64,
        "qkv_bias": True,
        "tie_word_embeddings": False,
    }
    _run_folder(tmp_path, raw, "qwen2_moe")
    cfg = ModelConfig.from_pretrained(write_config(tmp_path / "qwen2moeshapes", raw))
    shapes = cfg.expected_shapes()
    assert "layers.0.moe.shared.gate.weight" in shapes
    assert "layers.0.moe.shared_gate.weight" in shapes
    assert shapes["layers.0.moe.shared_gate.weight"] == (1, cfg.hidden_size)


def test_cohere_dropin(tmp_path: Path) -> None:
    raw = {
        **_BASE,
        "architectures": ["CohereForCausalLM"],
        "model_type": "cohere",
        "logit_scale": 0.0625,
        "tie_word_embeddings": True,
    }
    _run_folder(tmp_path, raw, "cohere")
    cfg = ModelConfig.from_pretrained(write_config(tmp_path / "cohereshapes", raw))
    assert "layers.0.post_attn_norm.weight" not in cfg.expected_shapes()
    assert "layers.0.input_norm.weight" in cfg.expected_shapes()


def test_glm_dropin(tmp_path: Path) -> None:
    raw = {
        **_BASE,
        "architectures": ["GlmForCausalLM"],
        "model_type": "glm",
        "attention_bias": True,
        "tie_word_embeddings": False,
    }
    _run_folder(tmp_path, raw, "glm")
    cfg = ModelConfig.from_pretrained(write_config(tmp_path / "glmshapes", raw))
    assert "layers.0.mlp.gate_up.weight" in cfg.expected_shapes()
    assert cfg.partial_rotary_factor == 0.5


def test_glm4_dropin(tmp_path: Path) -> None:
    raw = {
        **_BASE,
        "architectures": ["Glm4ForCausalLM"],
        "model_type": "glm4",
        "attention_bias": True,
        "tie_word_embeddings": False,
    }
    _run_folder(tmp_path, raw, "glm")
    cfg = ModelConfig.from_pretrained(write_config(tmp_path / "glm4shapes", raw))
    assert cfg.recipe_id == "glm"
    assert cfg.residual_kind == "gemma2"
    shapes = cfg.expected_shapes()
    assert "layers.0.pre_ff_norm.weight" in shapes
    assert "layers.0.post_ff_norm.weight" in shapes


def test_helium_alias_dropin(tmp_path: Path) -> None:
    raw = {
        **_BASE,
        "architectures": ["HeliumForCausalLM"],
        "model_type": "helium",
        "tie_word_embeddings": False,
    }
    _run_folder(tmp_path, raw, "llama")


def test_seed_oss_alias_dropin(tmp_path: Path) -> None:
    raw = {
        **_BASE,
        "architectures": ["SeedOssForCausalLM"],
        "model_type": "seed_oss",
        "attention_bias": True,
        "attention_out_bias": False,
        "tie_word_embeddings": False,
    }
    _run_folder(tmp_path, raw, "llama")


def test_ministral_alias_dropin(tmp_path: Path) -> None:
    raw = {
        **_BASE,
        "architectures": ["MinistralForCausalLM"],
        "model_type": "ministral",
        "sliding_window": 16,
    }
    _run_folder(tmp_path, raw, "mistral")


def test_phi_dropin(tmp_path: Path) -> None:
    raw = {
        **_BASE,
        "architectures": ["PhiForCausalLM"],
        "model_type": "phi",
        "hidden_act": "gelu_new",
        "tie_word_embeddings": False,
    }
    _run_folder(tmp_path, raw, "phi")
    cfg = ModelConfig.from_pretrained(write_config(tmp_path / "phishapes", raw))
    assert cfg.residual_kind == "parallel"
    assert cfg.partial_rotary_factor == 0.5
    shapes = cfg.expected_shapes()
    assert "layers.0.mlp.gate.weight" not in shapes
    assert "layers.0.mlp.up.weight" in shapes
    assert "layers.0.post_attn_norm.weight" not in shapes
    assert "layers.0.attn.o.weight" in shapes


def test_olmo3_dropin(tmp_path: Path) -> None:
    raw = {
        **_BASE,
        "architectures": ["Olmo3ForCausalLM"],
        "model_type": "olmo3",
        "layer_types": ["sliding_attention", "full_attention"],
        "sliding_window": 4,
        "qk_norm": True,
        "tie_word_embeddings": False,
    }
    _run_folder(tmp_path, raw, "olmo3")
    cfg = ModelConfig.from_pretrained(write_config(tmp_path / "olmo3shapes", raw))
    assert cfg.residual_kind == "post_norm"
    assert cfg.qk_norm is True
    shapes = cfg.expected_shapes()
    assert "layers.0.input_norm.weight" not in shapes
    assert "layers.0.post_attn_norm.weight" in shapes
    assert "layers.0.post_ff_norm.weight" in shapes
    assert shapes["layers.0.attn.q_norm.weight"] == (cfg.num_attention_heads * cfg.head_dim,)


def test_stablelm_dropin(tmp_path: Path) -> None:
    raw = {
        **_BASE,
        "architectures": ["StableLmForCausalLM"],
        "model_type": "stablelm",
        "use_parallel_residual": True,
        "use_qkv_bias": True,
        "qk_layernorm": False,
        "tie_word_embeddings": False,
    }
    _run_folder(tmp_path, raw, "stablelm")
    cfg = ModelConfig.from_pretrained(write_config(tmp_path / "stablelmshapes", raw))
    assert cfg.residual_kind == "parallel"
    assert cfg.partial_rotary_factor == 0.25
    assert cfg.attention_bias is True
    assert "layers.0.post_attn_norm.weight" not in cfg.expected_shapes()


def test_olmoe_dropin(tmp_path: Path) -> None:
    raw = {
        **_BASE,
        "architectures": ["OlmoeForCausalLM"],
        "model_type": "olmoe",
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "norm_topk_prob": False,
        "tie_word_embeddings": False,
    }
    _run_folder(tmp_path, raw, "olmoe")
    cfg = ModelConfig.from_pretrained(write_config(tmp_path / "olmoeshapes", raw))
    shapes = cfg.expected_shapes()
    assert "layers.0.moe.experts.gate_up.weight" in shapes
    assert cfg.qk_norm is True


def test_granite_swa_dropin(tmp_path: Path) -> None:
    raw = {
        **_BASE,
        "architectures": ["GraniteSWAForCausalLM"],
        "model_type": "granite_swa",
        "attention_multiplier": 0.25,
        "residual_multiplier": 1.0,
        "embedding_multiplier": 1.0,
        "logits_scaling": 1.0,
        "sliding_window": 4,
        "layer_types": ["sliding_attention", "full_attention"],
        "layer_rope_theta": [10000.0, 0.0],
        "tie_word_embeddings": False,
    }
    _run_folder(tmp_path, raw, "granite_swa")
    cfg = ModelConfig.from_pretrained(write_config(tmp_path / "graniteswashapes", raw))
    assert cfg.no_rope_layers == (1, 0)
    assert "layers.0.attn.sinks" in cfg.expected_shapes()


def test_granitemoe_dropin(tmp_path: Path) -> None:
    raw = {
        **_BASE,
        "architectures": ["GraniteMoeForCausalLM"],
        "model_type": "granitemoe",
        "num_local_experts": 4,
        "num_experts_per_tok": 2,
        "attention_multiplier": 1.0,
        "residual_multiplier": 1.0,
        "embedding_multiplier": 1.0,
        "logits_scaling": 1.0,
        "tie_word_embeddings": False,
    }
    _run_folder(tmp_path, raw, "granitemoe")


def test_granitemoe_swa_dropin(tmp_path: Path) -> None:
    raw = {
        **_BASE,
        "architectures": ["GraniteMoeSWAForCausalLM"],
        "model_type": "granitemoe_swa",
        "num_local_experts": 4,
        "num_experts_per_tok": 2,
        "attention_multiplier": 0.25,
        "residual_multiplier": 1.0,
        "embedding_multiplier": 1.0,
        "logits_scaling": 1.0,
        "sliding_window": 4,
        "layer_types": ["sliding_attention", "full_attention"],
        "tie_word_embeddings": False,
    }
    _run_folder(tmp_path, raw, "granitemoe_swa")
    cfg = ModelConfig.from_pretrained(write_config(tmp_path / "gmswashapes", raw))
    assert "layers.0.attn.sinks" in cfg.expected_shapes()
    assert cfg.attention_kind == "gqa_sinks"


def test_granitemoeshared_dropin(tmp_path: Path) -> None:
    raw = {
        **_BASE,
        "architectures": ["GraniteMoeSharedForCausalLM"],
        "model_type": "granitemoeshared",
        "num_local_experts": 4,
        "num_experts_per_tok": 2,
        "shared_intermediate_size": 64,
        "attention_multiplier": 1.0,
        "residual_multiplier": 1.0,
        "embedding_multiplier": 1.0,
        "logits_scaling": 1.0,
        "tie_word_embeddings": False,
    }
    _run_folder(tmp_path, raw, "granitemoeshared")
    cfg = ModelConfig.from_pretrained(write_config(tmp_path / "granitemoesharedshapes", raw))
    shapes = cfg.expected_shapes()
    assert "layers.0.moe.shared.gate_up.weight" in shapes
    assert shapes["layers.0.moe.shared.gate_up.weight"][0] == 128


def test_cohere2_alias_dropin(tmp_path: Path) -> None:
    raw = {
        **_BASE,
        "architectures": ["Cohere2ForCausalLM"],
        "model_type": "cohere2",
        "logit_scale": 0.0625,
        "sliding_window": 4,
        "layer_types": ["sliding_attention", "full_attention"],
        "tie_word_embeddings": True,
    }
    _run_folder(tmp_path, raw, "cohere")
    cfg = ModelConfig.from_pretrained(write_config(tmp_path / "cohere2shapes", raw))
    assert cfg.no_rope_layers == (1, 0)
    assert cfg.residual_kind == "parallel"
    assert cfg.rope_interleaved is True


def test_exaone4_dropin(tmp_path: Path) -> None:
    raw = {
        **_BASE,
        "architectures": ["Exaone4ForCausalLM"],
        "model_type": "exaone4",
        "qk_norm": True,
        "sliding_window": 4,
        "layer_types": ["sliding_attention", "full_attention"],
        "tie_word_embeddings": False,
    }
    _run_folder(tmp_path, raw, "exaone4")
    cfg = ModelConfig.from_pretrained(write_config(tmp_path / "exaone4shapes", raw))
    assert cfg.residual_kind == "post_norm"
    assert cfg.no_rope_layers == (1, 0)
    assert cfg.qk_norm is True


def test_exaone_moe_dropin(tmp_path: Path) -> None:
    raw = {
        **_BASE,
        "architectures": ["ExaoneMoeForCausalLM"],
        "model_type": "exaone_moe",
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "num_shared_experts": 1,
        "moe_intermediate_size": 16,
        "first_k_dense_replace": 1,
        "n_group": 1,
        "topk_group": 1,
        "routed_scaling_factor": 1.0,
        "sliding_window": 4,
        "layer_types": ["sliding_attention", "full_attention"],
        "tie_word_embeddings": False,
    }
    _run_folder(tmp_path, raw, "exaone_moe")
    cfg = ModelConfig.from_pretrained(write_config(tmp_path / "exaonemoeshapes", raw))
    assert cfg.qk_norm is True
    assert cfg.no_rope_layers == (1, 0)
    assert "layers.1.moe.experts.gate_up.weight" in cfg.expected_shapes()
    assert "layers.0.mlp.gate.weight" in cfg.expected_shapes()


def test_arcee_dropin(tmp_path: Path) -> None:
    raw = {
        **_BASE,
        "architectures": ["ArceeForCausalLM"],
        "model_type": "arcee",
        "hidden_act": "relu2",
        "tie_word_embeddings": False,
    }
    _run_folder(tmp_path, raw, "arcee")
    cfg = ModelConfig.from_pretrained(write_config(tmp_path / "arceeshapes", raw))
    assert cfg.uses_swiglu is False
    assert "layers.0.mlp.gate.weight" not in cfg.expected_shapes()
    assert "layers.0.mlp.up.weight" in cfg.expected_shapes()


def test_mistral3_nested_text_config(tmp_path: Path) -> None:
    raw = {
        "architectures": ["Mistral3ForConditionalGeneration"],
        "model_type": "mistral3",
        "text_config": {
            **_BASE,
            "model_type": "mistral",
            "sliding_window": 16,
        },
        "vision_config": {"hidden_size": 16},
    }
    cfg = ModelConfig.from_pretrained(write_config(tmp_path / "mistral3nested", raw))
    assert cfg.recipe_id == "mistral"
    missing = detect_missing(raw, "mistral3")
    assert "vision" not in missing


def test_gptj_dropin(tmp_path: Path) -> None:
    raw = {
        **_BASE,
        "architectures": ["GPTJForCausalLM"],
        "model_type": "gptj",
        "rotary_dim": 4,
        "activation_function": "gelu_new",
        "n_inner": 64,
        "tie_word_embeddings": False,
        "num_key_value_heads": 4,
    }
    _run_folder(tmp_path, raw, "gptj")


def test_gpt_neo_dropin(tmp_path: Path) -> None:
    raw = {
        **_BASE,
        "architectures": ["GPTNeoForCausalLM"],
        "model_type": "gpt_neo",
        "attention_types": [[["global", "local"], 1]],
        "window_size": 4,
        "activation_function": "gelu_new",
        "num_key_value_heads": 4,
        "tie_word_embeddings": True,
    }
    _run_folder(tmp_path, raw, "gpt_neo")


def test_opt_dropin(tmp_path: Path) -> None:
    raw = {
        **_BASE,
        "architectures": ["OPTForCausalLM"],
        "model_type": "opt",
        "ffn_dim": 64,
        "enable_bias": True,
        "activation_function": "relu",
        "do_layer_norm_before": True,
        "num_key_value_heads": 4,
        "tie_word_embeddings": True,
    }
    _run_folder(tmp_path, raw, "opt")


def test_bloom_dropin(tmp_path: Path) -> None:
    raw = {
        **_BASE,
        "architectures": ["BloomForCausalLM"],
        "model_type": "bloom",
        "n_head": 4,
        "num_key_value_heads": 4,
        "tie_word_embeddings": True,
    }
    _run_folder(tmp_path, raw, "bloom")


def test_falcon_dropin(tmp_path: Path) -> None:
    raw = {
        **_BASE,
        "architectures": ["FalconForCausalLM"],
        "model_type": "falcon",
        "multi_query": True,
        "parallel_attn": True,
        "alibi": False,
        "bias": False,
        "num_key_value_heads": 1,
        "tie_word_embeddings": True,
    }
    _run_folder(tmp_path, raw, "falcon")


def test_mpt_dropin(tmp_path: Path) -> None:
    raw = {
        **_BASE,
        "architectures": ["MptForCausalLM"],
        "model_type": "mpt",
        "n_heads": 4,
        "d_model": 32,
        "expansion_ratio": 2,
        "max_seq_len": 64,
        "tie_word_embeddings": True,
    }
    _run_folder(tmp_path, raw, "mpt")


def test_gpt_bigcode_dropin(tmp_path: Path) -> None:
    raw = {
        "architectures": ["GPTBigCodeForCausalLM"],
        "model_type": "gpt_bigcode",
        "vocab_size": 64,
        "n_embd": 32,
        "n_head": 4,
        "n_layer": 2,
        "n_inner": 64,
        "n_positions": 64,
        "multi_query": True,
        "layer_norm_epsilon": 1e-5,
        "torch_dtype": "float32",
        "tie_word_embeddings": True,
    }
    _run_folder(tmp_path, raw, "gpt_bigcode")


def test_phi4_dropin(tmp_path: Path) -> None:
    raw = {
        **_BASE,
        "architectures": ["Phi3ForCausalLM"],
        "model_type": "phi4",
        "attention_bias": False,
        "tie_word_embeddings": False,
    }
    _run_folder(tmp_path, raw, "phi4")
    assert detect_recipe_id(raw) != "phi"


def test_bitnet_dropin(tmp_path: Path) -> None:
    raw = {
        **_BASE,
        "architectures": ["BitNetForCausalLM"],
        "model_type": "bitnet",
        "attention_bias": False,
        "tie_word_embeddings": False,
    }
    _run_folder(tmp_path, raw, "bitnet")


def test_deepseek_v2_dropin(tmp_path: Path) -> None:
    raw = {
        **_BASE,
        "architectures": ["DeepseekV2ForCausalLM"],
        "model_type": "deepseek_v2",
        "kv_lora_rank": 8,
        "qk_nope_head_dim": 8,
        "qk_rope_head_dim": 4,
        "v_head_dim": 8,
        "q_lora_rank": None,
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "n_shared_experts": 1,
        "moe_intermediate_size": 16,
        "first_k_dense_replace": 1,
        "n_group": 1,
        "topk_group": 1,
        "tie_word_embeddings": False,
    }
    _run_folder(tmp_path, raw, "deepseek_v2")


def test_glm4_moe_dropin(tmp_path: Path) -> None:
    raw = {
        **_BASE,
        "architectures": ["Glm4MoeForCausalLM"],
        "model_type": "glm4_moe",
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "n_shared_experts": 1,
        "moe_intermediate_size": 16,
        "first_k_dense_replace": 1,
        "n_group": 1,
        "topk_group": 1,
        "e_score_correction_bias": True,
        "routed_scaling_factor": 1.0,
        "tie_word_embeddings": False,
    }
    _run_folder(tmp_path, raw, "glm4_moe")


def test_phimoe_dropin(tmp_path: Path) -> None:
    raw = {
        **_BASE,
        "architectures": ["PhimoeForCausalLM"],
        "model_type": "phimoe",
        "num_local_experts": 4,
        "num_experts_per_tok": 2,
        "router_jitter_noise": 0.01,
        "tie_word_embeddings": False,
    }
    _run_folder(tmp_path, raw, "phimoe")


def test_flex_olmo_dropin(tmp_path: Path) -> None:
    raw = {
        **_BASE,
        "architectures": ["FlexOlmoForCausalLM"],
        "model_type": "flex_olmo",
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "norm_topk_prob": False,
        "tie_word_embeddings": False,
    }
    _run_folder(tmp_path, raw, "flex_olmo")


def test_hunyuan_v1_moe_dropin(tmp_path: Path) -> None:
    raw = {
        **_BASE,
        "architectures": ["HunYuanMoEV1ForCausalLM"],
        "model_type": "hunyuan_v1_moe",
        "num_experts": 4,
        "moe_topk": 2,
        "tie_word_embeddings": False,
    }
    _run_folder(tmp_path, raw, "hunyuan_v1_moe")


def test_ernie4_5_moe_dropin(tmp_path: Path) -> None:
    raw = {
        **_BASE,
        "architectures": ["Ernie4_5_MoeForCausalLM"],
        "model_type": "ernie4_5_moe",
        "moe_num_experts": 4,
        "moe_k": 2,
        "moe_num_shared_experts": 1,
        "moe_layer_start_index": 0,
        "moe_intermediate_size": 16,
        "tie_word_embeddings": False,
    }
    _run_folder(tmp_path, raw, "ernie4_5_moe")


def test_dbrx_dropin(tmp_path: Path) -> None:
    raw = {
        **_BASE,
        "architectures": ["DbrxForCausalLM"],
        "model_type": "dbrx",
        "d_model": 32,
        "n_heads": 4,
        "n_layers": 2,
        "max_seq_len": 64,
        "attn_config": {"kv_n_heads": 2, "clip_qkv": 8.0},
        "ffn_config": {
            "ffn_hidden_size": 32,
            "moe_num_experts": 4,
            "moe_top_k": 2,
            "moe_normalize_expert_weights": 1.0,
        },
        "tie_word_embeddings": False,
    }
    _run_folder(tmp_path, raw, "dbrx")


def test_cohere2_moe_dropin(tmp_path: Path) -> None:
    raw = {
        **_BASE,
        "architectures": ["Cohere2MoeForCausalLM"],
        "model_type": "cohere2_moe",
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "num_shared_experts": 1,
        "shared_expert_combination_strategy": "average",
        "expert_selection_fn": "softmax",
        "first_k_dense_replace": 1,
        "mlp_layer_types": ["dense", "sparse"],
        "layer_types": ["sliding_attention", "full_attention"],
        "sliding_window": 4,
        "sliding_window_pattern": 2,
        "rms_norm_eps": 1e-5,
        "logit_scale": 0.0625,
        "tie_word_embeddings": False,
    }
    _run_folder(tmp_path, raw, "cohere2_moe")


def test_diffllama_dropin(tmp_path: Path) -> None:
    raw = {
        **_BASE,
        "architectures": ["DiffLlamaForCausalLM"],
        "model_type": "diffllama",
        "num_key_value_heads": 2,
        "tie_word_embeddings": False,
    }
    _run_folder(tmp_path, raw, "diffllama")


def test_jamba_dropin(tmp_path: Path) -> None:
    raw = {
        **_BASE,
        "architectures": ["JambaForCausalLM"],
        "model_type": "jamba",
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "attn_layer_period": 2,
        "attn_layer_offset": 0,
        "expert_layer_period": 2,
        "expert_layer_offset": 1,
        "mamba_d_state": 8,
        "mamba_d_conv": 4,
        "mamba_expand": 2,
        "mamba_dt_rank": 2,
        "tie_word_embeddings": False,
    }
    _run_folder(tmp_path, raw, "jamba")


def test_olmo_hybrid_dropin(tmp_path: Path) -> None:
    raw = {
        **_BASE,
        "architectures": ["OlmoHybridForCausalLM"],
        "model_type": "olmo_hybrid",
        "layer_types": ["linear_attention", "full_attention"],
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "linear_key_head_dim": 8,
        "linear_value_head_dim": 8,
        "linear_conv_kernel_dim": 4,
        "linear_allow_neg_eigval": True,
        "qk_norm": True,
        "tie_word_embeddings": False,
    }
    _run_folder(tmp_path, raw, "olmo_hybrid")


def test_qwen3_next_dropin(tmp_path: Path) -> None:
    raw = {
        **_BASE,
        "architectures": ["Qwen3NextForCausalLM"],
        "model_type": "qwen3_next",
        "layer_types": ["linear_attention", "full_attention"],
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 16,
        "shared_expert_intermediate_size": 16,
        "decoder_sparse_step": 1,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "linear_key_head_dim": 8,
        "linear_value_head_dim": 8,
        "linear_conv_kernel_dim": 4,
        "tie_word_embeddings": False,
    }
    _run_folder(tmp_path, raw, "qwen3_next")


def test_qwen3_5_moe_dropin(tmp_path: Path) -> None:
    raw = {
        **_BASE,
        "architectures": ["Qwen3_5MoeForCausalLM"],
        "model_type": "qwen3_5_moe",
        "layer_types": ["linear_attention", "full_attention"],
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 16,
        "shared_expert_intermediate_size": 16,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "linear_key_head_dim": 8,
        "linear_value_head_dim": 8,
        "linear_conv_kernel_dim": 4,
        "tie_word_embeddings": False,
    }
    _run_folder(tmp_path, raw, "qwen3_5_moe")


def test_qwen2_vl_text_alias(tmp_path: Path) -> None:
    raw = {
        **_BASE,
        "architectures": ["Qwen2VLForConditionalGeneration"],
        "model_type": "qwen2_vl",
        "text_config": {**_BASE, "model_type": "qwen2"},
        "vision_config": {"hidden_size": 16},
    }
    folder = write_config(tmp_path / "qwen2vl", raw)
    cfg = ModelConfig.from_pretrained(folder)
    assert cfg.recipe_id == "qwen2"
    assert "vision" not in detect_missing(raw, "qwen2vl")
