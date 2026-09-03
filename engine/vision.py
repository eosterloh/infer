"""Qwen3.5 / Qwen3.8 vision encoder.

The processor supplies flattened temporal-spatial patches plus ``grid_thw``.
This module mirrors the checkpoint's Conv3D patch embed, learned/interpolated
2D positions, vision RoPE, non-causal transformer blocks, and 2x2 merger.
"""

from __future__ import annotations

import itertools
from typing import Any

import torch
import torch.nn.functional as F

from engine.layers.rope import rotate_half


def qwen35_vision_expected_shapes(
    config: dict[str, Any],
) -> dict[str, tuple[int, ...]]:
    """Return the exact HF vision tensor contract for this config."""
    h = int(config["hidden_size"])
    inter = int(config["intermediate_size"])
    out = int(config["out_hidden_size"])
    c = int(config.get("in_channels", 3))
    t = int(config.get("temporal_patch_size", 2))
    p = int(config.get("patch_size", 16))
    m = int(config.get("spatial_merge_size", 2))
    npos = int(config.get("num_position_embeddings", 2304))
    expected: dict[str, tuple[int, ...]] = {
        "model.visual.patch_embed.proj.weight": (h, c, t, p, p),
        "model.visual.patch_embed.proj.bias": (h,),
        "model.visual.pos_embed.weight": (npos, h),
        "model.visual.merger.norm.weight": (h,),
        "model.visual.merger.norm.bias": (h,),
        "model.visual.merger.linear_fc1.weight": (h * m * m, h * m * m),
        "model.visual.merger.linear_fc1.bias": (h * m * m,),
        "model.visual.merger.linear_fc2.weight": (out, h * m * m),
        "model.visual.merger.linear_fc2.bias": (out,),
    }
    for i in range(int(config["depth"])):
        prefix = f"model.visual.blocks.{i}"
        expected.update(
            {
                f"{prefix}.norm1.weight": (h,),
                f"{prefix}.norm1.bias": (h,),
                f"{prefix}.norm2.weight": (h,),
                f"{prefix}.norm2.bias": (h,),
                f"{prefix}.attn.qkv.weight": (3 * h, h),
                f"{prefix}.attn.qkv.bias": (3 * h,),
                f"{prefix}.attn.proj.weight": (h, h),
                f"{prefix}.attn.proj.bias": (h,),
                f"{prefix}.mlp.linear_fc1.weight": (inter, h),
                f"{prefix}.mlp.linear_fc1.bias": (inter,),
                f"{prefix}.mlp.linear_fc2.weight": (h, inter),
                f"{prefix}.mlp.linear_fc2.bias": (h,),
            }
        )
    return expected


def validate_qwen35_vision_weights(
    weights: dict[str, torch.Tensor], config: dict[str, Any]
) -> None:
    """Require the complete vision tower and exact checkpoint shapes."""
    expected = qwen35_vision_expected_shapes(config)
    missing = sorted(set(expected) - set(weights))
    extra = sorted(set(weights) - set(expected))
    if missing or extra:
        raise KeyError(
            f"vision weight mismatch: missing={missing[:5]} extra={extra[:5]}"
        )
    for name, shape in expected.items():
        if tuple(weights[name].shape) != shape:
            raise ValueError(
                f"vision {name}: got {tuple(weights[name].shape)}, expected {shape}"
            )


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
    position_ids: torch.Tensor,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    dim = head_dim // 2
    inv = 1.0 / (
        10000.0
        ** (
            torch.arange(0, dim, 2, device=device, dtype=torch.float32)
            / float(dim)
        )
    ).to(dtype=dtype)
    freqs = (position_ids.to(device=device).unsqueeze(-1) * inv).flatten(1)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def _vision_position_ids(
    grid_thw: torch.Tensor, spatial_merge_size: int
) -> torch.Tensor:
    positions: list[torch.Tensor] = []
    device = grid_thw.device
    m = spatial_merge_size
    for t, h, w in grid_thw.tolist():
        hp, wp = torch.meshgrid(
            torch.arange(h, device=device),
            torch.arange(w, device=device),
            indexing="ij",
        )
        shape = (h // m, m, w // m, m)
        hp = hp.reshape(shape).transpose(1, 2).flatten()
        wp = wp.reshape(shape).transpose(1, 2).flatten()
        positions.append(torch.stack((hp, wp), dim=-1).repeat(t, 1))
    return torch.cat(positions, dim=0)


def _interpolate_vision_positions(
    table: torch.Tensor,
    grid_thw: torch.Tensor,
    *,
    spatial_merge_size: int,
    align_corners: bool = True,
) -> torch.Tensor:
    side = int(table.shape[0] ** 0.5)
    index_parts: list[list[torch.Tensor]] = [[] for _ in range(4)]
    weight_parts: list[list[torch.Tensor]] = [[] for _ in range(4)]
    m = spatial_merge_size
    for t, h, w in grid_thw.tolist():
        if not align_corners:
            raise ValueError("Qwen3.5 vision positions require align_corners=True")
        h_grid = torch.linspace(0, side - 1, h, device=table.device)
        w_grid = torch.linspace(0, side - 1, w, device=table.device)
        h_floor, w_floor = h_grid.int(), w_grid.int()
        h_ceil = (h_floor + 1).clamp(max=side - 1)
        w_ceil = (w_floor + 1).clamp(max=side - 1)
        h_frac, w_frac = h_grid - h_floor, w_grid - w_floor
        h_floor_offset, h_ceil_offset = h_floor * side, h_ceil * side
        corners = [
            (h_floor_offset[:, None] + w_floor[None]).flatten(),
            (h_floor_offset[:, None] + w_ceil[None]).flatten(),
            (h_ceil_offset[:, None] + w_floor[None]).flatten(),
            (h_ceil_offset[:, None] + w_ceil[None]).flatten(),
        ]
        weights = [
            ((1 - h_frac)[:, None] * (1 - w_frac)[None]).flatten(),
            ((1 - h_frac)[:, None] * w_frac[None]).flatten(),
            (h_frac[:, None] * (1 - w_frac)[None]).flatten(),
            (h_frac[:, None] * w_frac[None]).flatten(),
        ]
        h_idx = torch.arange(h, device=table.device).view(h // m, m)
        w_idx = torch.arange(w, device=table.device).view(w // m, m)
        reorder = (
            (h_idx[:, :, None, None] * w + w_idx[None, None])
            .transpose(1, 2)
            .flatten()
            .repeat(t)
        )
        for i in range(4):
            index_parts[i].append(corners[i][reorder])
            weight_parts[i].append(weights[i][reorder])
    indices = torch.stack([torch.cat(parts) for parts in index_parts])
    weights = torch.stack([torch.cat(parts) for parts in weight_parts])
    return (table[indices] * weights[:, :, None]).sum(0)


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
        q_h = q_i.transpose(0, 1)
        k_h = k_i.transpose(0, 1)
        v_h = v_i.transpose(0, 1)
        if x.is_cuda:
            attended = F.scaled_dot_product_attention(
                q_h[None],
                k_h[None],
                v_h[None],
                dropout_p=0.0,
                is_causal=False,
                scale=scale,
            )[0]
        else:
            scores = torch.matmul(q_h, k_h.transpose(-2, -1)) * scale
            probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(
                q_i.dtype
            )
            attended = torch.matmul(probs, v_h)
        outs.append(attended.transpose(0, 1))
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

    grid_thw = grid_thw.to(device=device, dtype=torch.long)
    if int(num_pos**0.5) ** 2 != num_pos:
        raise ValueError("vision num_position_embeddings must be a square grid")
    pos = _interpolate_vision_positions(
        weights["model.visual.pos_embed.weight"],
        grid_thw,
        spatial_merge_size=spatial_merge,
    )
    x = x + pos.to(dtype=x.dtype)

    position_ids = _vision_position_ids(grid_thw, spatial_merge)
    frame_lengths = torch.repeat_interleave(
        grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
    )
    cu_seqlens = F.pad(
        frame_lengths.cumsum(dim=0, dtype=torch.int32), (1, 0), value=0
    )
    cos, sin = _vision_rope(
        position_ids, hidden // num_heads, device, dtype=x.dtype
    )
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
