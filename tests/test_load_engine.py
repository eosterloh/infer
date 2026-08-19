"""Tests: end-to-end load_engine on tiny public Llama fixture (optional)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from engine.agent_api import load_engine

ROOT = Path(__file__).resolve().parents[1]
TINY = ROOT / "testdata" / "tiny-random-llama"
SPARK_LLAMA = Path.home() / "models" / "Llama-3.2-1B-Instruct"


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.mark.integration
def test_load_engine_tiny_or_skip() -> None:
    if not (TINY / "config.json").is_file():
        pytest.skip("testdata/tiny-random-llama not present — download fixture to enable")
    eng = load_engine(TINY, device=_device(), dtype="float32")
    info = eng.info()
    assert info["capabilities"]["model_type"] == "llama"
    assert eng.n_params > 0
    # Tiny random weights — just check generate returns something
    out = eng.generate("hi", max_new_tokens=4)
    assert isinstance(out, str)


@pytest.mark.integration
@pytest.mark.spark
def test_load_engine_llama_1b_spark() -> None:
    if not (SPARK_LLAMA / "config.json").is_file():
        pytest.skip("~/models/Llama-3.2-1B-Instruct not present")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for 1B integration test")
    eng = load_engine(SPARK_LLAMA, device="cuda")
    caps = eng.info()["capabilities"]
    assert caps["dense_mlp"] and caps["rope"] and not caps["moe"]
    out = eng.generate("The capital of France is", max_new_tokens=8)
    assert len(out) > 0
