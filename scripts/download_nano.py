#!/usr/bin/env python3
"""Download Nemotron 3 Nano BF16 into a local directory (Spark)."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--repo",
        default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        help="HF repo id",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path.home() / "models" / "NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        help="Local directory for the snapshot",
    )
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=args.repo,
        local_dir=str(args.out),
    )
    print(f"downloaded → {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
