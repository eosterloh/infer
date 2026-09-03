"""Qwen3.5 vision tower parity against Transformers on a tiny config."""

from __future__ import annotations

import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

from engine.vision import qwen35_vision_forward


def test_qwen35_vision_tiny_parity() -> None:
    torch.manual_seed(4)
    cfg = Qwen3_5VisionConfig(
        depth=2,
        hidden_size=16,
        intermediate_size=32,
        num_heads=4,
        in_channels=3,
        patch_size=2,
        temporal_patch_size=1,
        spatial_merge_size=2,
        out_hidden_size=32,
        num_position_embeddings=16,
        hidden_act="gelu_pytorch_tanh",
    )
    cfg._attn_implementation = "eager"
    hf = Qwen3_5VisionModel(cfg).eval()
    weights = {
        f"model.visual.{name}": tensor.detach()
        for name, tensor in hf.state_dict().items()
    }
    grid = torch.tensor([[1, 4, 4]], dtype=torch.long)
    # The processor flattens each C×T×P×P patch.
    pixels = torch.randn(16, 3 * 1 * 2 * 2)

    with torch.inference_mode():
        expected = hf(pixels, grid_thw=grid, return_dict=True).pooler_output
        actual = qwen35_vision_forward(pixels, grid, weights, cfg.to_dict())

    assert actual.shape == (4, 32)
    assert torch.allclose(actual, expected, atol=2e-5, rtol=2e-5)
