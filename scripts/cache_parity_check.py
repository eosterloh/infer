#!/usr/bin/env python3
"""Verify KV-cache greedy matches no-cache greedy (and optionally HF)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.config import ModelConfig
from engine.generate import generate_greedy
from engine.model import LlamaModel
from engine.tokenizer import Tokenizer
from engine.weights import load_weights


def collect(
    model: LlamaModel,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    use_cache: bool,
) -> str:
    return "".join(
        generate_greedy(
            model,
            tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            use_cache=use_cache,
        )
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        type=Path,
        default=Path.home() / "models" / "Llama-3.2-1B-Instruct",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--max-new-tokens", type=int, default=32)
    args = p.parse_args()

    model_dir = args.model.expanduser().resolve()
    config = ModelConfig.from_pretrained(model_dir)
    weights = load_weights(model_dir, config, device=args.device)
    model = LlamaModel(config, weights)
    tokenizer = Tokenizer.from_pretrained(model_dir)

    print(f"model_dir={model_dir}")
    print(f"prompt={args.prompt!r}")

    text_cache = collect(
        model, tokenizer, args.prompt, args.max_new_tokens, use_cache=True
    )
    text_nocache = collect(
        model, tokenizer, args.prompt, args.max_new_tokens, use_cache=False
    )
    print(f"cache_text={text_cache!r}")
    print(f"nocache_text={text_nocache!r}")

    if text_cache != text_nocache:
        print("FAIL: cache vs no-cache mismatch")
        return 1

    print("cache_parity_ok=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
