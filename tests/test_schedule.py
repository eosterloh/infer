"""Tests: config → schedule (Llama dense + Nemotron-H hybrid)."""

from __future__ import annotations

from pathlib import Path

from engine.config import ModelConfig
from engine.schedule import FfnKind, MixerKind, build_schedule, hybrid_pattern_schedule


def test_llama_schedule_all_dense_attn(llama_config_dir: Path) -> None:
    cfg = ModelConfig.from_pretrained(llama_config_dir)
    assert len(cfg.layers) == 4
    assert all(
        s.mixer == MixerKind.ATTENTION and s.ffn == FfnKind.DENSE_MLP for s in cfg.layers
    )
    assert "attention+dense_mlp" in cfg.summary()


def test_hybrid_pattern_chars() -> None:
    sched = hybrid_pattern_schedule("ME*-")
    assert sched[0].mixer == MixerKind.MAMBA2 and sched[0].ffn == FfnKind.NONE
    assert sched[1].mixer == MixerKind.NONE and sched[1].ffn == FfnKind.MOE
    assert sched[2].mixer == MixerKind.ATTENTION and sched[2].ffn == FfnKind.NONE
    assert sched[3].mixer == MixerKind.NONE and sched[3].ffn == FfnKind.DENSE_MLP


def test_nano_schedule_counts(nano_dir: Path) -> None:
    cfg = ModelConfig.from_pretrained(nano_dir)
    assert cfg.model_type == "nemotron_h"
    assert len(cfg.layers) == 52
    assert cfg.hybrid_override_pattern is not None
    assert len(cfg.hybrid_override_pattern) == 52
    mamba = sum(1 for s in cfg.layers if s.mixer == MixerKind.MAMBA2)
    attn = sum(1 for s in cfg.layers if s.mixer == MixerKind.ATTENTION)
    moe = sum(1 for s in cfg.layers if s.ffn == FfnKind.MOE)
    assert (mamba, attn, moe) == (23, 6, 23)
    assert build_schedule(cfg) == cfg.layers
