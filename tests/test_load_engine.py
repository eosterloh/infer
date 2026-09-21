"""Tests: end-to-end load_engine on tiny public Llama fixture (optional)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from engine.agent_api import load_engine
from engine.config import ModelConfig

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


def _folder_without_tie_flag(root: Path, head: bool) -> Path:
    """A Gemma-2-shaped config that never mentions tie_word_embeddings."""
    from safetensors.torch import save_file

    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Gemma2ForCausalLM"],
                "model_type": "gemma2",
                "vocab_size": 16,
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 4,
            }
        )
    )
    state = {"model.embed_tokens.weight": torch.zeros(16, 8)}
    if head:
        state["lm_head.weight"] = torch.zeros(16, 8)
    save_file(state, str(root / "model.safetensors"))
    return root


def test_a_missing_tie_flag_is_read_off_the_checkpoint(tmp_path: Path) -> None:
    """HF defaults this to true and Gemma-2 leaves it out; the weights decide.

    Reading the absent flag as false asked Gemma-2 for an lm_head it does not
    ship, and the load failed on a checkpoint transformers reads fine.
    """
    tied = ModelConfig.from_pretrained(_folder_without_tie_flag(tmp_path / "t", False))
    untied = ModelConfig.from_pretrained(_folder_without_tie_flag(tmp_path / "u", True))
    assert tied.tie_word_embeddings
    assert not untied.tie_word_embeddings


def test_an_explicit_tie_flag_still_wins(tmp_path: Path) -> None:
    folder = _folder_without_tie_flag(tmp_path / "x", False)
    raw = json.loads((folder / "config.json").read_text())
    raw["tie_word_embeddings"] = False
    (folder / "config.json").write_text(json.dumps(raw))
    assert not ModelConfig.from_pretrained(folder).tie_word_embeddings
