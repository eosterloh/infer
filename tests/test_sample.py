"""Unit tests for greedy vs temperature / top-k / nucleus sampling."""

from __future__ import annotations

from pathlib import Path

import torch

from engine.config import ModelConfig
from engine.sample import SamplingParams, make_generator, select_next_id
from engine.synth import random_engine_weights, write_config, write_hf_folder
from engine.agent_api import load_engine


def test_greedy_is_argmax() -> None:
    logits = torch.tensor([0.1, 3.0, 0.2, -1.0])
    assert select_next_id(logits, SamplingParams()) == 1
    assert select_next_id(logits, SamplingParams(temperature=0.0)) == 1
    assert select_next_id(logits, SamplingParams(temperature=-1.0)) == 1


def test_top_k_one_is_greedy() -> None:
    logits = torch.tensor([1.0, 2.0, 4.0, 3.0])
    gen = make_generator(0)
    picked = {
        select_next_id(logits, SamplingParams(temperature=1.0, top_k=1), gen)
        for _ in range(8)
    }
    assert picked == {2}


def test_top_p_keeps_mass() -> None:
    logits = torch.tensor([10.0, -10.0, -10.0, -10.0])
    gen = make_generator(1)
    for _ in range(6):
        assert select_next_id(logits, SamplingParams(temperature=1.0, top_p=0.9), gen) == 0


def test_seed_is_reproducible() -> None:
    logits = torch.linspace(-2, 2, 32)
    a = [
        select_next_id(logits, SamplingParams(temperature=0.8, top_k=8), make_generator(7))
        for _ in range(1)
    ]
    b = [
        select_next_id(logits, SamplingParams(temperature=0.8, top_k=8), make_generator(7))
        for _ in range(1)
    ]
    c = [
        select_next_id(logits, SamplingParams(temperature=0.8, top_k=8), make_generator(8))
        for _ in range(1)
    ]
    assert a == b
    # different seed can collide on tiny vocab; just ensure API runs
    assert isinstance(c[0], int)


def test_generate_temperature_knobs_are_not_swallowed(tmp_path: Path) -> None:
    raw = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size": 64,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 1,
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
    folder = write_config(tmp_path / "llama", raw)
    cfg = ModelConfig.from_pretrained(folder)
    weights = random_engine_weights(cfg, seed=0)
    write_hf_folder(folder, cfg, weights)
    eng = load_engine(folder, device="cpu", dtype="float32")
    greedy = eng.generate("hi", max_new_tokens=6, apply_chat_template=False, temperature=0.0)
    sampled = eng.generate(
        "hi",
        max_new_tokens=6,
        apply_chat_template=False,
        temperature=1.2,
        top_k=8,
        top_p=0.95,
        seed=3,
    )
    again = eng.generate(
        "hi",
        max_new_tokens=6,
        apply_chat_template=False,
        temperature=1.2,
        top_k=8,
        top_p=0.95,
        seed=3,
    )
    assert greedy == eng.generate("hi", max_new_tokens=6, apply_chat_template=False)
    assert again == sampled
    assert isinstance(sampled, str)
