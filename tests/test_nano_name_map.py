"""Tests: Nano HF → engine name map against weight index (no tensor load)."""

from __future__ import annotations

from pathlib import Path

from engine.config import ModelConfig
from engine.weights import load_weight_index, validate_name_map


def test_nano_name_map_covers_index(nano_dir: Path) -> None:
    cfg = ModelConfig.from_pretrained(nano_dir)
    weight_map = load_weight_index(nano_dir)
    mapped = validate_name_map(cfg, weight_map.keys())
    expected = cfg.expected_shapes()
    assert len(weight_map) == 6243
    assert len(mapped) == 6243
    assert len(expected) == 6243
    assert set(mapped) == set(expected)
