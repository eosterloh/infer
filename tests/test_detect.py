"""Tests: folder-in auto-detect (no model registration)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.config import ModelConfig
from engine.detect import (
    KNOWN_RECIPES,
    UnsupportedRecipeError,
    can_run,
    detect_missing,
    detect_recipe_id,
)
from engine.agent_api import inspect_capabilities


def test_detect_llama(llama_config_dir: Path) -> None:
    cfg = ModelConfig.from_pretrained(llama_config_dir)
    assert cfg.recipe_id == "llama"
    caps = inspect_capabilities(llama_config_dir)
    assert caps.can_run is True
    assert caps.recipe_id == "llama"
    assert caps.missing == ()


def test_detect_nano(nano_dir: Path) -> None:
    cfg = ModelConfig.from_pretrained(nano_dir)
    assert cfg.recipe_id == "nemotron_h"
    caps = inspect_capabilities(nano_dir)
    assert caps.can_run is True
    assert caps.recipe_id == "nemotron_h"
    assert "nvfp4_runtime" not in caps.missing


def test_unknown_recipe_fails(tmp_path: Path) -> None:
    raw = {
        "architectures": ["BertForMaskedLM"],
        "model_type": "bert",
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "rms_norm_eps": 1e-5,
        "max_position_embeddings": 32,
        "tie_word_embeddings": True,
        "torch_dtype": "float32",
    }
    (tmp_path / "config.json").write_text(json.dumps(raw))
    with pytest.raises(UnsupportedRecipeError) as ei:
        ModelConfig.from_pretrained(tmp_path)
    msg = str(ei.value)
    assert "bert" in msg
    for name in KNOWN_RECIPES:
        assert name in msg


def test_nvfp4_folder_dequants_on_load(tmp_path: Path) -> None:
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
        "torch_dtype": "bfloat16",
        "hybrid_override_pattern": "M",
        "mamba_num_heads": 2,
        "mamba_head_dim": 8,
        "ssm_state_size": 4,
        "n_groups": 1,
        "conv_kernel": 4,
        "quantization_config": {"quant_method": "nvfp4"},
    }
    folder = tmp_path / "NVIDIA-Nemotron-3-Super-NVFP4"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps(raw))
    missing = detect_missing(raw, folder.name)
    assert "nvfp4_runtime" not in missing
    assert can_run("nemotron_h", missing) is True
    caps = inspect_capabilities(folder)
    assert caps.can_run is True
    assert caps.nvfp4 is True


def test_detect_recipe_id_from_raw() -> None:
    assert detect_recipe_id({"model_type": "llama"}) == "llama"
    assert detect_recipe_id({"model_type": "qwen3"}) == "qwen3"
    assert detect_recipe_id({"model_type": "qwen3_5"}) == "qwen3_5"
    assert detect_recipe_id({"model_type": "qwen3_next"}) == "qwen3_next"
    assert detect_recipe_id({"model_type": "qwen3_5_moe"}) == "qwen3_5_moe"
    assert detect_recipe_id({"model_type": "qwen3_next"}) != "qwen3_5"
    assert detect_recipe_id({"model_type": "qwen3_5_moe"}) != "qwen3_5"
    assert (
        detect_recipe_id(
            {
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "model_type": "qwen3_5",
            }
        )
        == "qwen3_5"
    )
    assert detect_recipe_id({"architectures": ["LlamaForCausalLM"]}) == "llama"
    assert (
        detect_recipe_id(
            {"model_type": "nemotron_h", "hybrid_override_pattern": "ME*"}
        )
        == "nemotron_h"
    )
    assert detect_recipe_id({"model_type": "helium"}) == "llama"
    assert detect_recipe_id({"architectures": ["HeliumForCausalLM"]}) == "llama"
    assert detect_recipe_id({"model_type": "ernie4_5"}) == "llama"
    assert detect_recipe_id({"model_type": "hunyuan_v1_dense"}) == "llama"
    assert detect_recipe_id({"model_type": "seed_oss"}) == "llama"
    assert detect_recipe_id({"model_type": "cwm"}) == "llama"
    assert detect_recipe_id({"model_type": "ministral"}) == "mistral"
    assert detect_recipe_id({"architectures": ["MinistralForCausalLM"]}) == "mistral"
    assert detect_recipe_id({"model_type": "ministral3"}) == "mistral"
    assert detect_recipe_id({"model_type": "granite"}) == "granite"
    assert detect_recipe_id({"model_type": "olmo"}) == "olmo"
    assert detect_recipe_id({"model_type": "olmo2"}) == "olmo2"
    assert detect_recipe_id({"model_type": "olmo3"}) == "olmo3"
    assert detect_recipe_id({"model_type": "olmoe"}) == "olmoe"
    assert detect_recipe_id({"model_type": "olmo3"}) != "olmo"
    assert detect_recipe_id({"model_type": "olmoe"}) != "olmo"
    assert detect_recipe_id({"model_type": "phi3"}) == "phi3"
    assert detect_recipe_id({"architectures": ["Phi3ForCausalLM"]}) == "phi3"
    assert detect_recipe_id({"model_type": "phi"}) == "phi"
    assert detect_recipe_id({"architectures": ["PhiForCausalLM"]}) == "phi"
    assert detect_recipe_id({"model_type": "stablelm"}) == "stablelm"
    assert detect_recipe_id({"model_type": "granite_swa"}) == "granite_swa"
    assert detect_recipe_id({"model_type": "granitemoe"}) == "granitemoe"
    assert detect_recipe_id({"model_type": "granitemoe_swa"}) == "granitemoe_swa"
    assert detect_recipe_id({"model_type": "granitemoe_swa"}) != "granitemoe"
    assert detect_recipe_id({"model_type": "granitemoe_swa"}) != "granite_swa"
    assert detect_recipe_id({"model_type": "granitemoeshared"}) == "granitemoeshared"
    assert detect_recipe_id({"model_type": "granitemoe"}) != "granite"
    assert detect_recipe_id({"model_type": "cohere2"}) == "cohere"
    assert detect_recipe_id({"architectures": ["Cohere2ForCausalLM"]}) == "cohere"
    assert detect_recipe_id({"model_type": "exaone4"}) == "exaone4"
    assert detect_recipe_id({"model_type": "exaone_moe"}) == "exaone_moe"
    assert detect_recipe_id({"model_type": "exaone_moe"}) != "exaone4"
    assert detect_recipe_id({"model_type": "arcee"}) == "arcee"
    assert detect_recipe_id({"model_type": "mistral3"}) == "mistral"
    assert detect_recipe_id({"architectures": ["Mistral3ForConditionalGeneration"]}) == "mistral"
    assert detect_recipe_id({"model_type": "smollm3"}) == "smollm3"
    assert detect_recipe_id({"model_type": "starcoder2"}) == "starcoder2"
    assert detect_recipe_id({"model_type": "nemotron"}) == "nemotron"
    assert detect_recipe_id({"model_type": "gemma2"}) == "gemma2"
    assert detect_recipe_id({"model_type": "gemma2_text"}) == "gemma2"
    assert detect_recipe_id({"model_type": "gemma3"}) == "gemma3"
    assert detect_recipe_id({"architectures": ["Gemma3ForCausalLM"]}) == "gemma3"
    assert detect_recipe_id({"model_type": "gemma"}) == "gemma"
    assert detect_recipe_id({"model_type": "qwen3_moe"}) == "qwen3_moe"
    assert detect_recipe_id({"architectures": ["Qwen3MoeForCausalLM"]}) == "qwen3_moe"
    assert detect_recipe_id({"model_type": "qwen2_moe"}) == "qwen2_moe"
    assert detect_recipe_id({"model_type": "cohere"}) == "cohere"
    assert detect_recipe_id({"model_type": "glm"}) == "glm"
    assert detect_recipe_id({"model_type": "glm4"}) == "glm"
    assert detect_recipe_id({"model_type": "phimoe"}) == "phimoe"
    assert detect_recipe_id({"model_type": "diffllama"}) == "diffllama"
    with pytest.raises(UnsupportedRecipeError):
        detect_recipe_id({"model_type": "granitemoehybrid"})
    assert detect_recipe_id({"model_type": "olmo_hybrid"}) == "olmo_hybrid"
    assert detect_recipe_id({"model_type": "olmo_hybrid"}) != "olmo"
    assert detect_recipe_id({"architectures": ["Phi4ForCausalLM"], "model_type": "phi4"}) == "phi4"
    assert detect_recipe_id({"model_type": "phi4"}) != "phi"
    assert detect_recipe_id({"model_type": "qwen3_moe"}) != "mixtral"
    assert detect_recipe_id({"model_type": "gemma2"}) != "gemma"
    assert detect_recipe_id({"model_type": "nemotron"}) != "nemotron_h"
    assert detect_recipe_id({"model_type": "ernie4_5_moe"}) == "ernie4_5_moe"
    assert detect_recipe_id({"architectures": ["Ernie4_5MoeForCausalLM"], "model_type": "ernie4_5"}) == "ernie4_5_moe"
    assert detect_recipe_id({"model_type": "deepseek_v2"}) == "deepseek_v2"
    assert detect_recipe_id({"model_type": "deepseek_v2"}) != "deepseek_v3"
    assert detect_recipe_id({"model_type": "gptj"}) == "gptj"
    assert detect_recipe_id({"model_type": "gpt_neo"}) == "gpt_neo"
    assert detect_recipe_id({"model_type": "opt"}) == "opt"
    assert detect_recipe_id({"model_type": "bloom"}) == "bloom"
    assert detect_recipe_id({"model_type": "falcon"}) == "falcon"
    assert detect_recipe_id({"model_type": "mpt"}) == "mpt"
    assert detect_recipe_id({"model_type": "gpt_bigcode"}) == "gpt_bigcode"
    assert detect_recipe_id({"model_type": "bitnet"}) == "bitnet"
    assert detect_recipe_id({"model_type": "glm4_moe"}) == "glm4_moe"
    assert detect_recipe_id({"model_type": "glm4_moe"}) != "glm"
    assert detect_recipe_id({"model_type": "flex_olmo"}) == "flex_olmo"
    assert detect_recipe_id({"model_type": "hunyuan_v1_moe"}) == "hunyuan_v1_moe"
    assert detect_recipe_id({"model_type": "qwen2_vl"}) == "qwen2"
    assert detect_recipe_id({"model_type": "internvl", "text_config": {"model_type": "qwen2"}}) == "qwen2"
    with pytest.raises(UnsupportedRecipeError, match="PLE"):
        detect_recipe_id({"model_type": "gemma4"})
    assert detect_recipe_id({"model_type": "jamba"}) == "jamba"
    assert detect_recipe_id({"model_type": "dbrx"}) == "dbrx"
    with pytest.raises(UnsupportedRecipeError, match="minimax"):
        detect_recipe_id({"model_type": "minimax"})
    with pytest.raises(UnsupportedRecipeError, match="glm_moe_dsa"):
        detect_recipe_id({"model_type": "glm_moe_dsa"})


def test_generation_config_eos_overrides_config_json(tmp_path: Path) -> None:
    raw = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size": 128,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "max_position_embeddings": 128,
        "tie_word_embeddings": True,
        "torch_dtype": "float32",
        "eos_token_id": 2,
        "bos_token_id": 1,
    }
    (tmp_path / "config.json").write_text(json.dumps(raw))
    (tmp_path / "generation_config.json").write_text(
        json.dumps({"eos_token_id": [2, 11], "bos_token_id": 1})
    )
    cfg = ModelConfig.from_pretrained(tmp_path)
    assert cfg.eos_token_id == [2, 11]
