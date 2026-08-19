"""Tests: agent plug-and-play capabilities + inspect CLI path."""

from __future__ import annotations

import json
from pathlib import Path

from engine.agent_api import Capabilities, inspect_capabilities


def test_inspect_nano_capabilities(nano_dir: Path) -> None:
    caps = inspect_capabilities(nano_dir)
    assert isinstance(caps, Capabilities)
    d = caps.to_dict()
    assert d["model_type"] == "nemotron_h"
    assert d["mamba2"] is True
    assert d["moe"] is True
    assert d["attention"] is True
    assert d["dense_mlp"] is False
    assert d["rope"] is False  # Nemotron-H has no RoPE
    assert d["mtp"] is False
    assert d["nvfp4"] is False
    assert d["can_run"] is True
    assert d["recipe_id"] == "nemotron_h"
    assert not d["missing"]
    assert d["num_layers"] == 52
    assert d["hybrid_pattern"] and "M" in d["hybrid_pattern"] and "E" in d["hybrid_pattern"]


def test_inspect_llama_capabilities(llama_config_dir: Path) -> None:
    caps = inspect_capabilities(llama_config_dir)
    d = caps.to_dict()
    assert d["model_type"] == "llama"
    assert d["dense_mlp"] is True
    assert d["attention"] is True
    assert d["moe"] is False
    assert d["mamba2"] is False
    assert d["rope"] is True
    assert d["mtp"] is False
    assert d["can_run"] is True
    assert d["recipe_id"] == "llama"


def test_capabilities_json_serializable(nano_dir: Path) -> None:
    caps = inspect_capabilities(nano_dir)
    # Agents often JSON-dump this
    payload = json.dumps(caps.to_dict())
    assert "nemotron_h" in payload
