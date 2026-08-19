#!/usr/bin/env python3
"""Smoke: agent inspect_capabilities on Llama + Nano configs."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.agent_api import inspect_capabilities


def main() -> int:
    nano = ROOT / "testdata" / "nemotron3-nano-30b-a3b"
    caps = inspect_capabilities(nano)
    d = caps.to_dict()
    assert d["mamba2"] and d["moe"] and d["attention"]
    assert d["rope"] is False
    assert d["mtp"] is False
    assert d["can_run"] is True
    assert d["recipe_id"] == "nemotron_h"
    print("nano_caps", {k: d[k] for k in ("recipe_id", "can_run", "missing", "mamba2", "moe", "nvfp4")})
    print(f"pattern={d['hybrid_pattern']}")
    print("agent_caps_smoke=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
