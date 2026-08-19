#!/usr/bin/env python3
"""Download Meta Llama-3.2-1B-Instruct into a local directory.

Requires Hugging Face access to the gated repo and `huggingface-cli login`
(or HF_TOKEN in the environment).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--repo",
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="HF repo id",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path.home() / "models" / "Llama-3.2-1B-Instruct",
        help="Local directory for the snapshot",
    )
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=args.repo,
        local_dir=str(args.out),
        local_dir_use_symlinks=False,
    )
    print(f"downloaded → {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
