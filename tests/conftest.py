"""Shared fixtures for infer tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NANO_CFG = ROOT / "testdata" / "nemotron3-nano-30b-a3b"


@pytest.fixture
def repo_root() -> Path:
    return ROOT


@pytest.fixture
def nano_dir() -> Path:
    if not (NANO_CFG / "config.json").is_file():
        pytest.skip("nano testdata config missing")
    return NANO_CFG


@pytest.fixture
def llama_config_dir(tmp_path: Path) -> Path:
    raw = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size": 128,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "max_position_embeddings": 128,
        "tie_word_embeddings": True,
        "torch_dtype": "float32",
        "hidden_act": "silu",
    }
    (tmp_path / "config.json").write_text(json.dumps(raw))
    return tmp_path
