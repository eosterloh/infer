"""Qwen3.5 vision tower parity against Transformers on a tiny config."""

from __future__ import annotations

import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import (
    Qwen3_5TextConfig,
    Qwen3_5VisionConfig,
)
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5TextRotaryEmbedding,
    Qwen3_5VisionModel,
    Qwen3_5VisionRotaryEmbedding,
)

from engine.layers.rope import build_mrope_cos_sin
from engine.vision import (
    _vision_rope,
    qwen35_rope_index,
    qwen35_vision_forward,
    validate_qwen35_vision_weights,
)


def test_qwen35_vision_rope_bfloat16_is_exact() -> None:
    position_ids = torch.arange(40).reshape(20, 2)
    reference = Qwen3_5VisionRotaryEmbedding(36).to(dtype=torch.bfloat16)
    frequencies = reference(position_ids)
    embedding = torch.cat((frequencies, frequencies), dim=-1)
    actual_cos, actual_sin = _vision_rope(
        position_ids, 72, torch.device("cpu"), torch.bfloat16
    )
    assert torch.equal(actual_cos, embedding.cos())
    assert torch.equal(actual_sin, embedding.sin())


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
    validate_qwen35_vision_weights(weights, cfg.to_dict())
    grid = torch.tensor([[1, 4, 4]], dtype=torch.long)
    # The processor flattens each C×T×P×P patch.
    pixels = torch.randn(16, 3 * 1 * 2 * 2)

    with torch.inference_mode():
        expected = hf(pixels, grid_thw=grid, return_dict=True).pooler_output
        actual = qwen35_vision_forward(pixels, grid, weights, cfg.to_dict())

    assert actual.shape == (4, 32)
    assert torch.allclose(actual, expected, atol=2e-5, rtol=2e-5)


def test_qwen35_mrope_parity() -> None:
    cfg = Qwen3_5TextConfig(
        vocab_size=32,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        layer_types=["full_attention"],
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 1_000_000.0,
            "partial_rotary_factor": 1.0,
            "mrope_interleaved": True,
            "mrope_section": [11, 11, 10],
        },
    )
    hf = Qwen3_5TextRotaryEmbedding(cfg)
    position_ids = torch.tensor(
        [
            [[0, 1, 2, 3, 4]],
            [[0, 1, 3, 5, 7]],
            [[0, 1, 4, 7, 10]],
        ]
    )
    x = torch.zeros(1, 5, 256)
    expected_cos, expected_sin = hf(x, position_ids)
    actual_cos, actual_sin = build_mrope_cos_sin(
        hf.inv_freq,
        position_ids,
        torch.float32,
        cfg.rope_parameters["mrope_section"],
    )
    assert torch.equal(actual_cos, expected_cos)
    assert torch.equal(actual_sin, expected_sin)


def test_qwen35_rope_index_image_segment() -> None:
    input_ids = torch.tensor([[10, 11, 99, 99, 99, 99, 12]])
    token_types = torch.tensor([[0, 0, 1, 1, 1, 1, 0]])
    positions, delta = qwen35_rope_index(
        input_ids,
        token_types,
        spatial_merge_size=2,
        image_grid_thw=torch.tensor([[1, 4, 4]]),
    )
    expected = torch.tensor(
        [
            [[0, 1, 2, 2, 2, 2, 4]],
            [[0, 1, 2, 2, 3, 3, 4]],
            [[0, 1, 2, 3, 2, 3, 4]],
        ]
    )
    assert torch.equal(positions, expected)
    assert torch.equal(delta, torch.tensor([[-2]]))
