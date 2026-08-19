#!/usr/bin/env python3
"""Phase N0: parse Nano config + validate HF→engine name map against index.

Does not load weight tensors (safe on machines without room for 30B).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.config import ModelConfig
from engine.schedule import FfnKind, MixerKind
from engine.weights import load_weight_index, validate_name_map


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        type=Path,
        default=ROOT / "testdata" / "nemotron3-nano-30b-a3b",
        help="Dir with config.json + model.safetensors.index.json",
    )
    args = p.parse_args()
    model_dir = args.model.expanduser().resolve()

    config = ModelConfig.from_pretrained(model_dir)
    print(f"config: {config.summary()}")
    if config.hybrid_override_pattern:
        print(f"pattern: {config.hybrid_override_pattern}")

    counts = {
        "mamba": sum(1 for s in config.layers if s.mixer == MixerKind.MAMBA2),
        "attn": sum(1 for s in config.layers if s.mixer == MixerKind.ATTENTION),
        "moe": sum(1 for s in config.layers if s.ffn == FfnKind.MOE),
        "dense_mlp": sum(1 for s in config.layers if s.ffn == FfnKind.DENSE_MLP),
    }
    print(f"layer_counts: {counts}")

    weight_map = load_weight_index(model_dir)
    mapped = validate_name_map(config, weight_map.keys())
    expected = config.expected_shapes()
    print(f"hf_tensors={len(weight_map)}")
    print(f"engine_tensors={len(mapped)}")
    print(f"expected_shapes={len(expected)}")
    print("name_map_ok=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
