"""Qwen3.5 / Qwen3.8 vision encoder.

The processor supplies flattened temporal-spatial patches plus ``grid_thw``.
This module mirrors the checkpoint's Conv3D patch embed, learned/interpolated
2D positions, vision RoPE, non-causal transformer blocks, and 2x2 merger.
"""

from __future__ import annotations

import itertools
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F
from transformers.vision_utils import (
    get_vision_attention_seqlens,
    get_vision_interpolation_indices_and_weights,
    get_vision_position_ids,
)

from engine.layers.rope import rotate_half


def _linear(
    x: torch.Tensor, weights: dict[str, torch.Tensor], key: str
) -> torch.Tensor:
    return F.linear(x, weights[f"{key}.weight"], weights.get(f"{key}.bias"))


def _layer_norm(
    x: torch.Tensor, weights: dict[str, torch.Tensor], key: str
) -> torch.Tensor:
    return F.layer_norm(
        x,
        (x.shape[-1],),
        weights[f"{key}.weight"],
        weights.get(f"{key}.bias"),
        1e-6,
    )


def _vision_rope(
    position_ids: torch.Tensor, head_dim: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    dim = head_dim // 2
    inv = 1.0 / (
        10000.0
        ** (
            torch.arange(0, dim, 2, device=device, dtype=torch.float32)
            / float(dim)
        )
    )
    freqs = (position_ids.to(device=device).unsqueeze(-1).float() * inv).flatten(1)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def _vision_attention(
    x: torch.Tensor,
    weights: dict[str, torch.Tensor],
    prefix: str,
    *,
    num_heads: int,
    cos: torch.Tensor,
    sin: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    seq, hidden = x.shape
    head_dim = hidden // num_heads
    q, k, v = (
        _linear(x, weights, f"{prefix}.qkv")
        .reshape(seq, 3, num_heads, head_dim)
        .permute(1, 0, 2, 3)
        .unbind(0)
    )
    cos_h = cos[:, None, :].float()
    sin_h = sin[:, None, :].float()
    q_dtype, k_dtype = q.dtype, k.dtype
    q = ((q.float() * cos_h) + (rotate_half(q.float()) * sin_h)).to(q_dtype)
    k = ((k.float() * cos_h) + (rotate_half(k.float()) * sin_h)).to(k_dtype)

    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    q_parts = torch.split(q, lengths, dim=0)
    k_parts = torch.split(k, lengths, dim=0)
    v_parts = torch.split(v, lengths, dim=0)
    outs: list[torch.Tensor] = []
    scale = head_dim**-0.5
    for q_i, k_i, v_i in zip(q_parts, k_parts, v_parts):
        scores = torch.einsum("thd,shd->hts", q_i.float(), k_i.float()) * scale
        probs = torch.softmax(scores, dim=-1).to(v_i.dtype)
        outs.append(torch.einsum("hts,shd->thd", probs, v_i))
    out = torch.cat(outs, dim=0).reshape(seq, hidden)
    return _linear(out, weights, f"{prefix}.proj")


def qwen35_vision_forward(
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
    weights: dict[str, torch.Tensor],
    config: dict[str, Any],
) -> torch.Tensor:
    """Return one language-width embedding per merged image/video patch."""
    hidden = int(config["hidden_size"])
    in_channels = int(config.get("in_channels", 3))
    temporal = int(config.get("temporal_patch_size", 2))
    patch = int(config.get("patch_size", 16))
    spatial_merge = int(config.get("spatial_merge_size", 2))
    num_heads = int(config["num_heads"])
    depth = int(config["depth"])
    num_pos = int(config.get("num_position_embeddings", 2304))
    device = pixel_values.device
    dtype = weights["model.visual.patch_embed.proj.weight"].dtype

    patches = pixel_values.reshape(-1, in_channels, temporal, patch, patch)
    x = F.conv3d(
        patches.to(dtype),
        weights["model.visual.patch_embed.proj.weight"],
        weights.get("model.visual.patch_embed.proj.bias"),
        stride=(temporal, patch, patch),
    ).reshape(-1, hidden)

    helper_config = SimpleNamespace(**config)
    helper_config._attn_implementation = "eager"
    grid_thw = grid_thw.to(device=device, dtype=torch.long)
    interp_i, interp_w = get_vision_interpolation_indices_and_weights(
        grid_thw,
        num_grid_per_side=int(num_pos**0.5),
        mode="bilinear",
        align_corners=True,
        spatial_merge_size=spatial_merge,
    )
    pos_table = weights["model.visual.pos_embed.weight"]
    pos = (pos_table[interp_i] * interp_w[:, :, None].to(pos_table.dtype)).sum(1)
    x = x + pos.to(dtype=x.dtype)

    position_ids = get_vision_position_ids(grid_thw, spatial_merge)
    cu_seqlens, _ = get_vision_attention_seqlens(grid_thw, helper_config)
    cos, sin = _vision_rope(position_ids, hidden // num_heads, device)
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    cu_seqlens = cu_seqlens.to(device)

    for i in range(depth):
        p = f"model.visual.blocks.{i}"
        h = _layer_norm(x, weights, f"{p}.norm1")
        x = x + _vision_attention(
            h,
            weights,
            f"{p}.attn",
            num_heads=num_heads,
            cos=cos,
            sin=sin,
            cu_seqlens=cu_seqlens,
        )
        h = _layer_norm(x, weights, f"{p}.norm2")
        h = _linear(h, weights, f"{p}.mlp.linear_fc1")
        h = F.gelu(h, approximate="tanh")
        x = x + _linear(h, weights, f"{p}.mlp.linear_fc2")

    p = "model.visual.merger"
    x = _layer_norm(x, weights, f"{p}.norm")
    x = x.reshape(-1, hidden * spatial_merge * spatial_merge)
    x = F.gelu(_linear(x, weights, f"{p}.linear_fc1"))
    return _linear(x, weights, f"{p}.linear_fc2")


def qwen35_rope_index(
    input_ids: torch.Tensor,
    mm_token_type_ids: torch.Tensor,
    *,
    spatial_merge_size: int,
    image_grid_thw: torch.Tensor | None = None,
    video_grid_thw: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build Qwen text/image/video 3D positions and per-batch decode deltas."""
    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(
            video_grid_thw, video_grid_thw[:, 0], dim=0
        ).clone()
        video_grid_thw[:, 0] = 1

    position_ids = torch.zeros(
        3,
        input_ids.shape[0],
        input_ids.shape[1],
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    deltas: list[torch.Tensor] = []
    grid_iters = {
        1: iter(image_grid_thw) if image_grid_thw is not None else None,
        2: iter(video_grid_thw) if video_grid_thw is not None else None,
    }
    for batch_idx, ids in enumerate(input_ids):
        types = mm_token_type_ids[batch_idx]
        mask = attention_mask[batch_idx].bool() if attention_mask is not None else None
        if mask is not None:
            ids = ids[mask]
            types = types[mask]

        groups: list[tuple[int, int, int]] = []
        for kind, members in itertools.groupby(
            enumerate(types.tolist()), lambda pair: pair[1]
        ):
            members_l = list(members)
            groups.append((int(kind), members_l[0][0], members_l[-1][0] + 1))

        current = 0
        pieces: list[torch.Tensor] = []
        for modality, start, end in groups:
            if modality == 0:
                length = end - start
                pieces.append(
                    torch.arange(length, device=input_ids.device)[None]
                    .expand(3, -1)
                    .add(current)
                )
                current += length
                continue

            iterator = grid_iters.get(modality)
            if iterator is None:
                raise ValueError(f"missing grid_thw for modality type {modality}")
            grid = next(iterator)
            t = int(grid[0].item())
            h = int(grid[1].item()) // spatial_merge_size
            w = int(grid[2].item()) // spatial_merge_size
            temporal = torch.arange(t, device=input_ids.device)
            height = torch.arange(h, device=input_ids.device) + current
            width = torch.arange(w, device=input_ids.device) + current
            tt, hh, ww = torch.meshgrid(
                temporal, height, width, indexing="ij"
            )
            vision_pos = torch.stack((tt, hh, ww), dim=0).reshape(3, -1)
            vision_pos[0] += current
            pieces.append(vision_pos)
            current += max(h, w)

        merged = torch.cat(pieces, dim=1)
        if mask is not None:
            position_ids[:, batch_idx, mask] = merged
        else:
            position_ids[:, batch_idx] = merged
        deltas.append(merged.max() + 1 - len(ids))
    return position_ids, torch.stack(deltas).unsqueeze(1)


def qwen35_multimodal_embeddings(
    input_ids: torch.Tensor,
    text_embeddings: torch.Tensor,
    vision_weights: dict[str, torch.Tensor],
    raw_config: dict[str, Any],
    *,
    pixel_values: torch.Tensor | None = None,
    pixel_values_videos: torch.Tensor | None = None,
    image_grid_thw: torch.Tensor | None = None,
    video_grid_thw: torch.Tensor | None = None,
    mm_token_type_ids: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Insert vision embeddings at placeholders and return MRoPE positions."""
    vcfg = raw_config.get("vision_config")
    if not isinstance(vcfg, dict):
        raise ValueError("Qwen multimodal checkpoint has no vision_config")
    if mm_token_type_ids is None:
        raise ValueError("mm_token_type_ids is required for Qwen multimodal RoPE")

    x = text_embeddings.clone()
    image_token_id = int(raw_config["image_token_id"])
    video_token_id = int(raw_config["video_token_id"])

    def insert(
        pixels: torch.Tensor | None,
        grid: torch.Tensor | None,
        token_id: int,
        label: str,
    ) -> None:
        nonlocal x
        if pixels is None:
            return
        if grid is None:
            raise ValueError(f"{label}_grid_thw is required with {label} pixels")
        features = qwen35_vision_forward(
            pixels.to(device=x.device), grid.to(device=x.device), vision_weights, vcfg
        ).to(device=x.device, dtype=x.dtype)
        mask = (input_ids == token_id).unsqueeze(-1)
        expected = int(mask.sum().item()) * x.shape[-1]
        if expected != features.numel():
            raise ValueError(
                f"{label} features and placeholders differ: "
                f"tokens={int(mask.sum())}, features={features.shape[0]}"
            )
        x = x.masked_scatter(mask, features)

    insert(pixel_values, image_grid_thw, image_token_id, "image")
    insert(pixel_values_videos, video_grid_thw, video_token_id, "video")
    positions, delta = qwen35_rope_index(
        input_ids,
        mm_token_type_ids,
        spatial_merge_size=int(vcfg.get("spatial_merge_size", 2)),
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        attention_mask=attention_mask,
    )
    return x, positions, delta
