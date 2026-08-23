"""CLI: load model (Llama or Nemotron-H) + optional greedy generate / inspect."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from engine.agent_api import inspect_capabilities, load_engine
from engine.generate import generate_greedy


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="engine",
        description="DIY plug-and-play decoder — config-driven hybrid + agent API",
    )
    p.add_argument(
        "--model",
        required=True,
        type=Path,
        help="Path to HF model directory (config.json + weights)",
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="torch device (default: cuda if available else cpu)",
    )
    p.add_argument(
        "--dtype",
        default=None,
        choices=["bfloat16", "float16", "float32"],
        help="Override config torch_dtype (default: use config.json)",
    )
    p.add_argument(
        "--inspect",
        action="store_true",
        help="Print capabilities from config.json only (no weight load)",
    )
    p.add_argument(
        "--skip-tokenizer-smoke",
        action="store_true",
        help="Skip encode/decode roundtrip check",
    )
    p.add_argument(
        "--prompt",
        default=None,
        help="If set, run greedy generate on this prompt",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Max new tokens when --prompt is set",
    )
    p.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable cache (full recompute each step)",
    )
    p.add_argument(
        "--raw-prompt",
        action="store_true",
        help="Do not wrap --prompt in the folder's chat template",
    )
    p.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Leave the reasoning/think prefix open (Nano default template)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    model_dir: Path = args.model.expanduser().resolve()

    if not model_dir.is_dir():
        print(f"error: model dir not found: {model_dir}", file=sys.stderr)
        return 1

    print(f"model_dir={model_dir}")

    if args.inspect:
        caps = inspect_capabilities(model_dir)
        print(json.dumps(caps.to_dict(), indent=2))
        print("inspect_ok=true")
        return 0

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print(
            "error: --device cuda but torch.cuda.is_available() is False",
            file=sys.stderr,
        )
        return 1

    print(f"loading weights → device={args.device} …")
    eng = load_engine(model_dir, device=args.device, dtype=args.dtype)
    if not args.skip_tokenizer_smoke:
        eng.tokenizer.smoke_test()

    info = eng.info()
    print(f"config: {info['config']}")
    print(f"params={eng.n_params:,} ({eng.n_params / 1e9:.3f}B unique storage)")
    print(f"dtype={info['dtype']} device={info['device']}")
    print(f"capabilities={json.dumps(info['capabilities'], sort_keys=True)}")
    print("tokenizer_smoke=ok")
    print("load_ok=true")

    if args.prompt is None:
        return 0

    print(f"prompt={args.prompt!r}")
    use_cache = not args.no_cache
    apply_tmpl = False if args.raw_prompt else None
    print(f"use_cache={use_cache}")
    print(f"apply_chat_template={not args.raw_prompt}")
    print("generate:", end="", flush=True)
    pieces: list[str] = []
    for piece in generate_greedy(
        eng.model,
        eng.tokenizer,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        use_cache=use_cache,
        apply_chat_template=apply_tmpl,
        enable_thinking=args.enable_thinking,
    ):
        pieces.append(piece)
        print(piece, end="", flush=True)
    print()
    print(f"completion={''.join(pieces)!r}")
    print("generate_ok=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
