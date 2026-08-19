#!/usr/bin/env python3
"""Sanity-check layer schedule + DecoderModel import for dense Llama configs."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.config import ModelConfig
from engine.model import DecoderModel, LlamaModel
from engine.schedule import FfnKind, MixerKind, build_schedule


def main() -> int:
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
    with tempfile.TemporaryDirectory() as td:
        path = Path(td)
        (path / "config.json").write_text(json.dumps(raw))
        cfg = ModelConfig.from_pretrained(path)

    assert len(cfg.layers) == 4
    assert all(
        s.mixer == MixerKind.ATTENTION and s.ffn == FfnKind.DENSE_MLP for s in cfg.layers
    )
    assert LlamaModel is DecoderModel
    sched = build_schedule(cfg)
    assert sched == cfg.layers
    print(f"config: {cfg.summary()}")
    print("schedule_smoke=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
