"""Transformers logit/argmax parity for new decoder recipes."""

from __future__ import annotations

from dataclasses import fields
from typing import Any

import pytest
import torch

from engine.config import ModelConfig
from engine.maps import is_ignored_hf_name, map_hf_name
from engine.model import DecoderModel
from engine.synth import write_config


_TINY = dict(
    vocab_size=64,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    max_position_embeddings=64,
    tie_word_embeddings=False,
)


def _cfg_kwargs(cls: type, kwargs: dict[str, Any]) -> dict[str, Any]:
    allowed: set[str] = set()
    for klass in cls.__mro__:
        allowed.update(getattr(klass, "__annotations__", {}))
        try:
            allowed.update(f.name for f in fields(klass))
        except TypeError:
            pass
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    if "torch_dtype" in allowed:
        filtered["torch_dtype"] = torch.float32
    return filtered


def _map_hf_state(reference, config: ModelConfig) -> dict[str, torch.Tensor]:
    weights: dict[str, torch.Tensor] = {}
    for name, tensor in reference.state_dict().items():
        if "rotary" in name or "inv_freq" in name:
            continue
        if is_ignored_hf_name(name, config):
            continue
        mapped = map_hf_name(name, config)
        if mapped is None:
            raise AssertionError(f"unmapped HF tensor {name}")
        weights[mapped] = tensor.detach().float()
    if "lm_head.weight" not in weights and "embed.weight" in weights:
        weights["lm_head.weight"] = weights["embed.weight"]
    missing = set(config.expected_shapes()) - set(weights)
    missing.discard("lm_head.weight")
    assert not missing, f"missing engine tensors: {sorted(missing)[:12]}"
    return weights


def _assert_logits_match(config: ModelConfig, weights: dict[str, torch.Tensor], reference) -> None:
    model = DecoderModel(config, weights)
    input_ids = torch.randint(0, config.vocab_size, (1, 7))
    with torch.inference_mode():
        expected = reference(input_ids, use_cache=False).logits.float()
        actual = model.forward(input_ids).float()
    assert torch.allclose(actual, expected, atol=3e-5, rtol=3e-5)
    assert torch.equal(actual.argmax(dim=-1), expected.argmax(dim=-1))


def _sanitize_hf_config(hf_config: Any, engine_raw: dict[str, Any]) -> None:
    vocab = int(getattr(hf_config, "vocab_size", engine_raw.get("vocab_size", 0)) or 0)
    for key in ("pad_token_id", "bos_token_id", "eos_token_id"):
        val = getattr(hf_config, key, None)
        if isinstance(val, int) and vocab and val >= vocab:
            setattr(hf_config, key, None)
        elif isinstance(val, list):
            filtered = [t for t in val if isinstance(t, int) and t < vocab]
            setattr(hf_config, key, filtered or None)
    theta = engine_raw.get("rope_theta")
    rp = getattr(hf_config, "rope_parameters", None)
    engine_rp = engine_raw.get("rope_parameters") or engine_raw.get("rope_scaling")
    if isinstance(rp, dict):
        if isinstance(engine_rp, dict):
            if engine_rp.get("rope_theta") is not None:
                rp["rope_theta"] = float(engine_rp["rope_theta"])
            for key, nested in engine_rp.items():
                if isinstance(nested, dict) and nested.get("rope_theta") is not None:
                    target = rp.get(key)
                    if isinstance(target, dict):
                        target["rope_theta"] = float(nested["rope_theta"])
        elif theta is not None:
            if "rope_theta" in rp:
                rp["rope_theta"] = float(theta)
            full = rp.get("full_attention")
            if isinstance(full, dict) and "rope_theta" in full:
                full["rope_theta"] = float(theta)
    hf_config._attn_implementation = "eager"


def _run_parity(
    tmp_path,
    *,
    engine_raw: dict[str, Any],
    hf_cls: type,
    hf_model_cls: type,
    hf_kwargs: dict[str, Any] | None = None,
) -> None:
    folder = write_config(tmp_path, engine_raw)
    config = ModelConfig.from_pretrained(folder)
    kwargs = _cfg_kwargs(hf_cls, hf_kwargs or engine_raw)
    hf_config = hf_cls(**kwargs)
    _sanitize_hf_config(hf_config, engine_raw)
    reference = hf_model_cls(hf_config).eval().float()
    weights = _map_hf_state(reference, config)
    _assert_logits_match(config, weights, reference)


def test_granite_logits_match_transformers(tmp_path) -> None:
    try:
        from transformers.models.granite.configuration_granite import GraniteConfig
        from transformers.models.granite.modeling_granite import GraniteForCausalLM
    except ImportError:
        pytest.skip("GraniteForCausalLM not importable")
    raw = {
        **_TINY,
        "architectures": ["GraniteForCausalLM"],
        "model_type": "granite",
        "hidden_act": "silu",
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "attention_multiplier": 0.3535533905932738,
        "residual_multiplier": 1.0,
        "embedding_multiplier": 1.0,
        "logits_scaling": 1.0,
        "torch_dtype": "float32",
    }
    _run_parity(
        tmp_path,
        engine_raw=raw,
        hf_cls=GraniteConfig,
        hf_model_cls=GraniteForCausalLM,
    )


def test_olmo_logits_match_transformers(tmp_path) -> None:
    try:
        from transformers.models.olmo.configuration_olmo import OlmoConfig
        from transformers.models.olmo.modeling_olmo import OlmoForCausalLM
    except ImportError:
        pytest.skip("OlmoForCausalLM not importable")
    raw = {
        **_TINY,
        "architectures": ["OlmoForCausalLM"],
        "model_type": "olmo",
        "hidden_act": "silu",
        "clip_qkv": 8.0,
        "rope_theta": 10000.0,
        "torch_dtype": "float32",
    }
    _run_parity(tmp_path, engine_raw=raw, hf_cls=OlmoConfig, hf_model_cls=OlmoForCausalLM)


def test_olmo2_logits_match_transformers(tmp_path) -> None:
    try:
        from transformers.models.olmo2.configuration_olmo2 import Olmo2Config
        from transformers.models.olmo2.modeling_olmo2 import Olmo2ForCausalLM
    except ImportError:
        pytest.skip("Olmo2ForCausalLM not importable")
    raw = {
        **_TINY,
        "architectures": ["Olmo2ForCausalLM"],
        "model_type": "olmo2",
        "hidden_act": "silu",
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "torch_dtype": "float32",
    }
    _run_parity(tmp_path, engine_raw=raw, hf_cls=Olmo2Config, hf_model_cls=Olmo2ForCausalLM)


def test_gemma2_logits_match_transformers(tmp_path) -> None:
    try:
        from transformers.models.gemma2.configuration_gemma2 import Gemma2Config
        from transformers.models.gemma2.modeling_gemma2 import Gemma2ForCausalLM
    except ImportError:
        pytest.skip("Gemma2ForCausalLM not importable")
    raw = {
        **_TINY,
        "architectures": ["Gemma2ForCausalLM"],
        "model_type": "gemma2",
        "head_dim": 8,
        "hidden_activation": "gelu_pytorch_tanh",
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "query_pre_attn_scalar": 8,
        "sliding_window": 16,
        "attn_logit_softcapping": 50.0,
        "final_logit_softcapping": 30.0,
        "layer_types": ["sliding_attention", "full_attention"],
        "tie_word_embeddings": True,
        "torch_dtype": "float32",
    }
    _run_parity(tmp_path, engine_raw=raw, hf_cls=Gemma2Config, hf_model_cls=Gemma2ForCausalLM)


def test_qwen3_moe_logits_match_transformers(tmp_path) -> None:
    try:
        from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM
    except ImportError:
        pytest.skip("Qwen3MoeForCausalLM not importable")
    raw = {
        **_TINY,
        "architectures": ["Qwen3MoeForCausalLM"],
        "model_type": "qwen3_moe",
        "hidden_act": "silu",
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 32,
        "decoder_sparse_step": 1,
        "mlp_only_layers": [],
        "norm_topk_prob": False,
        "attention_bias": False,
        "torch_dtype": "float32",
    }
    _run_parity(tmp_path, engine_raw=raw, hf_cls=Qwen3MoeConfig, hf_model_cls=Qwen3MoeForCausalLM)


def test_starcoder2_logits_match_transformers(tmp_path) -> None:
    try:
        from transformers.models.starcoder2.configuration_starcoder2 import Starcoder2Config
        from transformers.models.starcoder2.modeling_starcoder2 import Starcoder2ForCausalLM
    except ImportError:
        pytest.skip("Starcoder2ForCausalLM not importable")
    raw = {
        **_TINY,
        "architectures": ["Starcoder2ForCausalLM"],
        "model_type": "starcoder2",
        "hidden_act": "gelu_pytorch_tanh",
        "use_bias": True,
        "norm_epsilon": 1e-5,
        "rope_theta": 10000.0,
        "torch_dtype": "float32",
    }
    _run_parity(
        tmp_path, engine_raw=raw, hf_cls=Starcoder2Config, hf_model_cls=Starcoder2ForCausalLM
    )


def test_nemotron_logits_match_transformers(tmp_path) -> None:
    try:
        from transformers.models.nemotron.configuration_nemotron import NemotronConfig
        from transformers.models.nemotron.modeling_nemotron import NemotronForCausalLM
    except ImportError:
        pytest.skip("NemotronForCausalLM not importable")
    raw = {
        **_TINY,
        "architectures": ["NemotronForCausalLM"],
        "model_type": "nemotron",
        "hidden_act": "relu2",
        "head_dim": 8,
        "norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "attention_bias": False,
        "mlp_bias": False,
        "torch_dtype": "float32",
    }
    _run_parity(
        tmp_path, engine_raw=raw, hf_cls=NemotronConfig, hf_model_cls=NemotronForCausalLM
    )


def test_cohere_logits_match_transformers(tmp_path) -> None:
    try:
        from transformers.models.cohere.configuration_cohere import CohereConfig
        from transformers.models.cohere.modeling_cohere import CohereForCausalLM
    except ImportError:
        pytest.skip("CohereForCausalLM not importable")
    raw = {
        **_TINY,
        "architectures": ["CohereForCausalLM"],
        "model_type": "cohere",
        "hidden_act": "silu",
        "layer_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "logit_scale": 0.0625,
        "tie_word_embeddings": True,
        "use_qk_norm": False,
        "torch_dtype": "float32",
    }
    _run_parity(tmp_path, engine_raw=raw, hf_cls=CohereConfig, hf_model_cls=CohereForCausalLM)


def test_glm_logits_match_transformers(tmp_path) -> None:
    try:
        from transformers.models.glm.configuration_glm import GlmConfig
        from transformers.models.glm.modeling_glm import GlmForCausalLM
    except ImportError:
        pytest.skip("GlmForCausalLM not importable")
    raw = {
        **_TINY,
        "architectures": ["GlmForCausalLM"],
        "model_type": "glm",
        "hidden_act": "silu",
        "head_dim": 8,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "attention_bias": True,
        "torch_dtype": "float32",
    }
    _run_parity(tmp_path, engine_raw=raw, hf_cls=GlmConfig, hf_model_cls=GlmForCausalLM)


def test_smollm3_logits_match_transformers(tmp_path) -> None:
    try:
        from transformers.models.smollm3.configuration_smollm3 import SmolLM3Config
        from transformers.models.smollm3.modeling_smollm3 import SmolLM3ForCausalLM
    except ImportError:
        pytest.skip("SmolLM3ForCausalLM not importable")
    raw = {
        **_TINY,
        "architectures": ["SmolLM3ForCausalLM"],
        "model_type": "smollm3",
        "hidden_act": "silu",
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "no_rope_layers": [1, 0],
        "tie_word_embeddings": True,
        "torch_dtype": "float32",
    }
    _run_parity(
        tmp_path, engine_raw=raw, hf_cls=SmolLM3Config, hf_model_cls=SmolLM3ForCausalLM
    )


def test_seed_oss_logits_match_transformers(tmp_path) -> None:
    try:
        from transformers.models.seed_oss.configuration_seed_oss import SeedOssConfig
        from transformers.models.seed_oss.modeling_seed_oss import SeedOssForCausalLM
    except ImportError:
        pytest.skip("SeedOssForCausalLM not importable")
    raw = {
        **_TINY,
        "architectures": ["SeedOssForCausalLM"],
        "model_type": "seed_oss",
        "hidden_act": "silu",
        "head_dim": 8,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "attention_bias": True,
        "attention_out_bias": False,
        "mlp_bias": False,
        "torch_dtype": "float32",
    }
    _run_parity(
        tmp_path, engine_raw=raw, hf_cls=SeedOssConfig, hf_model_cls=SeedOssForCausalLM
    )


def test_helium_logits_match_transformers(tmp_path) -> None:
    try:
        from transformers.models.helium.configuration_helium import HeliumConfig
        from transformers.models.helium.modeling_helium import HeliumForCausalLM
    except ImportError:
        pytest.skip("HeliumForCausalLM not importable")
    raw = {
        **_TINY,
        "architectures": ["HeliumForCausalLM"],
        "model_type": "helium",
        "hidden_act": "silu",
        "head_dim": 8,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "attention_bias": False,
        "mlp_bias": False,
        "pad_token_id": None,
        "torch_dtype": "float32",
    }
    _run_parity(tmp_path, engine_raw=raw, hf_cls=HeliumConfig, hf_model_cls=HeliumForCausalLM)


def test_gemma3_mixed_logits_match_transformers(tmp_path) -> None:
    try:
        from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig
        from transformers.models.gemma3.modeling_gemma3 import Gemma3ForCausalLM
    except ImportError:
        pytest.skip("Gemma3ForCausalLM not importable")
    raw = {
        **_TINY,
        "architectures": ["Gemma3ForCausalLM"],
        "model_type": "gemma3",
        "head_dim": 8,
        "hidden_activation": "gelu_pytorch_tanh",
        "rms_norm_eps": 1e-6,
        "rope_theta": 1_000_000.0,
        "query_pre_attn_scalar": 8,
        "sliding_window": 4,
        "qk_norm": True,
        "layer_types": ["sliding_attention", "full_attention"],
        "rope_parameters": {
            "full_attention": {"rope_type": "default", "rope_theta": 1_000_000.0},
            "sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0},
        },
        "tie_word_embeddings": True,
        "pad_token_id": None,
        "torch_dtype": "float32",
    }
    _run_parity(
        tmp_path,
        engine_raw=raw,
        hf_cls=Gemma3TextConfig,
        hf_model_cls=Gemma3ForCausalLM,
    )


def test_qwen2_moe_logits_match_transformers(tmp_path) -> None:
    try:
        from transformers.models.qwen2_moe.configuration_qwen2_moe import Qwen2MoeConfig
        from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeForCausalLM
    except ImportError:
        pytest.skip("Qwen2MoeForCausalLM not importable")
    raw = {
        **_TINY,
        "architectures": ["Qwen2MoeForCausalLM"],
        "model_type": "qwen2_moe",
        "hidden_act": "silu",
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 32,
        "shared_expert_intermediate_size": 64,
        "decoder_sparse_step": 1,
        "mlp_only_layers": [],
        "norm_topk_prob": False,
        "qkv_bias": True,
        "tie_word_embeddings": False,
        "torch_dtype": "float32",
    }
    _run_parity(
        tmp_path, engine_raw=raw, hf_cls=Qwen2MoeConfig, hf_model_cls=Qwen2MoeForCausalLM
    )


def test_glm4_logits_match_transformers(tmp_path) -> None:
    try:
        from transformers.models.glm4.configuration_glm4 import Glm4Config
        from transformers.models.glm4.modeling_glm4 import Glm4ForCausalLM
    except ImportError:
        pytest.skip("Glm4ForCausalLM not importable")
    raw = {
        **_TINY,
        "architectures": ["Glm4ForCausalLM"],
        "model_type": "glm4",
        "hidden_act": "silu",
        "head_dim": 8,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "attention_bias": True,
        "torch_dtype": "float32",
    }
    _run_parity(tmp_path, engine_raw=raw, hf_cls=Glm4Config, hf_model_cls=Glm4ForCausalLM)


def test_olmo3_logits_match_transformers(tmp_path) -> None:
    try:
        from transformers.models.olmo3.configuration_olmo3 import Olmo3Config
        from transformers.models.olmo3.modeling_olmo3 import Olmo3ForCausalLM
    except ImportError:
        pytest.skip("Olmo3ForCausalLM not importable")
    raw = {
        **_TINY,
        "architectures": ["Olmo3ForCausalLM"],
        "model_type": "olmo3",
        "hidden_act": "silu",
        "rms_norm_eps": 1e-5,
        "rope_theta": 500000.0,
        "layer_types": ["sliding_attention", "full_attention"],
        "sliding_window": 4,
        "rope_parameters": {
            "full_attention": {"rope_type": "default", "rope_theta": 500000.0},
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
        },
        "pad_token_id": None,
        "torch_dtype": "float32",
    }
    _run_parity(tmp_path, engine_raw=raw, hf_cls=Olmo3Config, hf_model_cls=Olmo3ForCausalLM)


def test_stablelm_logits_match_transformers(tmp_path) -> None:
    try:
        from transformers.models.stablelm.configuration_stablelm import StableLmConfig
        from transformers.models.stablelm.modeling_stablelm import StableLmForCausalLM
    except ImportError:
        pytest.skip("StableLmForCausalLM not importable")
    raw = {
        **_TINY,
        "architectures": ["StableLmForCausalLM"],
        "model_type": "stablelm",
        "hidden_act": "silu",
        "layer_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "use_parallel_residual": True,
        "use_qkv_bias": True,
        "qk_layernorm": False,
        "partial_rotary_factor": 0.25,
        "bos_token_id": None,
        "eos_token_id": None,
        "torch_dtype": "float32",
    }
    _run_parity(
        tmp_path, engine_raw=raw, hf_cls=StableLmConfig, hf_model_cls=StableLmForCausalLM
    )


def test_olmoe_logits_match_transformers(tmp_path) -> None:
    try:
        from transformers.models.olmoe.configuration_olmoe import OlmoeConfig
        from transformers.models.olmoe.modeling_olmoe import OlmoeForCausalLM
    except ImportError:
        pytest.skip("OlmoeForCausalLM not importable")
    raw = {
        **_TINY,
        "architectures": ["OlmoeForCausalLM"],
        "model_type": "olmoe",
        "hidden_act": "silu",
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "norm_topk_prob": False,
        "pad_token_id": None,
        "torch_dtype": "float32",
    }
    _run_parity(tmp_path, engine_raw=raw, hf_cls=OlmoeConfig, hf_model_cls=OlmoeForCausalLM)


def test_granite_swa_logits_match_transformers(tmp_path) -> None:
    try:
        from transformers.models.granite_swa.configuration_granite_swa import GraniteSWAConfig
        from transformers.models.granite_swa.modeling_granite_swa import GraniteSWAForCausalLM
    except ImportError:
        pytest.skip("GraniteSWAForCausalLM not importable")
    raw = {
        **_TINY,
        "architectures": ["GraniteSWAForCausalLM"],
        "model_type": "granite_swa",
        "hidden_act": "silu",
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "attention_multiplier": 0.25,
        "residual_multiplier": 1.0,
        "embedding_multiplier": 1.0,
        "logits_scaling": 1.0,
        "sliding_window": 4,
        "layer_types": ["sliding_attention", "full_attention"],
        "layer_rope_theta": [10000.0, 0.0],
        "tie_word_embeddings": False,
        "torch_dtype": "float32",
    }
    _run_parity(
        tmp_path,
        engine_raw=raw,
        hf_cls=GraniteSWAConfig,
        hf_model_cls=GraniteSWAForCausalLM,
    )


def test_granitemoe_logits_match_transformers(tmp_path) -> None:
    try:
        from transformers.models.granitemoe.configuration_granitemoe import GraniteMoeConfig
        from transformers.models.granitemoe.modeling_granitemoe import GraniteMoeForCausalLM
    except ImportError:
        pytest.skip("GraniteMoeForCausalLM not importable")
    raw = {
        **_TINY,
        "architectures": ["GraniteMoeForCausalLM"],
        "model_type": "granitemoe",
        "hidden_act": "silu",
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "num_local_experts": 4,
        "num_experts_per_tok": 2,
        "attention_multiplier": 1.0,
        "residual_multiplier": 1.0,
        "embedding_multiplier": 1.0,
        "logits_scaling": 1.0,
        "torch_dtype": "float32",
    }
    _run_parity(
        tmp_path,
        engine_raw=raw,
        hf_cls=GraniteMoeConfig,
        hf_model_cls=GraniteMoeForCausalLM,
    )


def test_granitemoeshared_logits_match_transformers(tmp_path) -> None:
    try:
        from transformers.models.granitemoeshared.configuration_granitemoeshared import (
            GraniteMoeSharedConfig,
        )
        from transformers.models.granitemoeshared.modeling_granitemoeshared import (
            GraniteMoeSharedForCausalLM,
        )
    except ImportError:
        pytest.skip("GraniteMoeSharedForCausalLM not importable")
    raw = {
        **_TINY,
        "architectures": ["GraniteMoeSharedForCausalLM"],
        "model_type": "granitemoeshared",
        "hidden_act": "silu",
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "num_local_experts": 4,
        "num_experts_per_tok": 2,
        "shared_intermediate_size": 64,
        "attention_multiplier": 1.0,
        "residual_multiplier": 1.0,
        "embedding_multiplier": 1.0,
        "logits_scaling": 1.0,
        "torch_dtype": "float32",
    }
    _run_parity(
        tmp_path,
        engine_raw=raw,
        hf_cls=GraniteMoeSharedConfig,
        hf_model_cls=GraniteMoeSharedForCausalLM,
    )


def test_cohere2_logits_match_transformers(tmp_path) -> None:
    try:
        from transformers.models.cohere2.configuration_cohere2 import Cohere2Config
        from transformers.models.cohere2.modeling_cohere2 import Cohere2ForCausalLM
    except ImportError:
        pytest.skip("Cohere2ForCausalLM not importable")
    raw = {
        **_TINY,
        "architectures": ["Cohere2ForCausalLM"],
        "model_type": "cohere2",
        "hidden_act": "silu",
        "layer_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "logit_scale": 0.0625,
        "sliding_window": 4,
        "layer_types": ["sliding_attention", "full_attention"],
        "tie_word_embeddings": True,
        "pad_token_id": None,
        "torch_dtype": "float32",
    }
    _run_parity(
        tmp_path, engine_raw=raw, hf_cls=Cohere2Config, hf_model_cls=Cohere2ForCausalLM
    )


def test_exaone4_logits_match_transformers(tmp_path) -> None:
    try:
        from transformers.models.exaone4.configuration_exaone4 import Exaone4Config
        from transformers.models.exaone4.modeling_exaone4 import Exaone4ForCausalLM
    except ImportError:
        pytest.skip("Exaone4ForCausalLM not importable")
    raw = {
        **_TINY,
        "architectures": ["Exaone4ForCausalLM"],
        "model_type": "exaone4",
        "hidden_act": "silu",
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "sliding_window": 4,
        "layer_types": ["sliding_attention", "full_attention"],
        "torch_dtype": "float32",
    }
    _run_parity(
        tmp_path, engine_raw=raw, hf_cls=Exaone4Config, hf_model_cls=Exaone4ForCausalLM
    )


def test_phi_logits_match_transformers(tmp_path) -> None:
    try:
        from transformers.models.phi.configuration_phi import PhiConfig
        from transformers.models.phi.modeling_phi import PhiForCausalLM
    except ImportError:
        pytest.skip("PhiForCausalLM not importable")
    raw = {
        **_TINY,
        "architectures": ["PhiForCausalLM"],
        "model_type": "phi",
        "hidden_act": "gelu_new",
        "layer_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "partial_rotary_factor": 0.5,
        "qk_layernorm": False,
        "bos_token_id": None,
        "eos_token_id": None,
        "torch_dtype": "float32",
    }
    _run_parity(tmp_path, engine_raw=raw, hf_cls=PhiConfig, hf_model_cls=PhiForCausalLM)


def test_arcee_logits_match_transformers(tmp_path) -> None:
    try:
        from transformers.models.arcee.configuration_arcee import ArceeConfig
        from transformers.models.arcee.modeling_arcee import ArceeForCausalLM
    except ImportError:
        pytest.skip("ArceeForCausalLM not importable")
    raw = {
        **_TINY,
        "architectures": ["ArceeForCausalLM"],
        "model_type": "arcee",
        "hidden_act": "relu2",
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "torch_dtype": "float32",
    }
    _run_parity(tmp_path, engine_raw=raw, hf_cls=ArceeConfig, hf_model_cls=ArceeForCausalLM)
