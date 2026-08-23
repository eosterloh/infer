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
    assert detect_recipe_id({"architectures": ["LlamaForCausalLM"]}) == "llama"
    assert (
        detect_recipe_id(
            {"model_type": "nemotron_h", "hybrid_override_pattern": "ME*"}
        )
        == "nemotron_h"
    )


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
