"""Opt-in real Qwen3.8-27B acceptance on DGX Spark."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from engine.agent_api import load_engine


@pytest.mark.spark
@pytest.mark.integration
def test_qwen38_real_text_mtp_and_vision() -> None:
    model_dir = os.environ.get("SPARK_QWEN38_DIR")
    if not model_dir:
        pytest.skip("set SPARK_QWEN38_DIR to run the real Qwen3.8 acceptance")
    if not torch.cuda.is_available():
        pytest.skip("real Qwen3.8 acceptance requires CUDA")
    path = Path(model_dir)
    if not (path / "model.safetensors.index.json").is_file():
        pytest.skip(f"Qwen3.8 checkpoint not found under {path}")

    import numpy as np
    from PIL import Image

    engine = load_engine(path, device="cuda")
    assert engine.config.recipe_id == "qwen3_5"
    assert engine.n_params == 27_781_427_952
    assert engine.mtp is not None
    assert engine.processor is not None
    assert engine.vision_weights is not None

    prompt = "Reply with exactly: OK"
    baseline = engine.generate(
        prompt, max_new_tokens=4, enable_thinking=False
    )
    speculative = engine.generate(
        prompt,
        max_new_tokens=4,
        enable_thinking=False,
        num_speculative_tokens=2,
    )
    assert speculative == baseline
    assert engine.last_mtp_stats is not None

    image = Image.fromarray(np.zeros((64, 96, 3), dtype=np.uint8))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": "What is the dominant color? Answer briefly.",
                },
            ],
        }
    ]
    vision_baseline = engine.generate_messages(messages, max_new_tokens=4)
    vision_speculative = engine.generate_messages(
        messages, max_new_tokens=4, num_speculative_tokens=2
    )
    assert vision_baseline
    assert vision_speculative == vision_baseline
