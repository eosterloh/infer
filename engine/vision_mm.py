"""Vision towers for Gemma3, Llama4, Mistral3, Qwen2-VL, and Qwen2.5-VL."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from engine.detect import detect_vision_family
from engine.layers.norm import gemma_rms_norm, rms_norm
from engine.layers.rope import rotate_half
from engine.vision import (
    _linear,
    _layer_norm,
    _vision_attention,
    _vision_position_ids,
    _vision_rope,
    qwen35_rope_index,
)


def _act(name: str, x: torch.Tensor) -> torch.Tensor:
    key = (name or "gelu").lower()
    if key in {"silu", "swish"}:
        return F.silu(x)
    if key in {"quick_gelu"}:
        return x * torch.sigmoid(1.702 * x)
    if key in {"gelu_pytorch_tanh", "gelu_tanh"}:
        return F.gelu(x, approximate="tanh")
    if key in {"gelu_new"}:
        return F.gelu(x)
    return F.gelu(x)


def _image_token_id(raw: dict[str, Any]) -> int:
    for key in ("image_token_id", "image_token_index"):
        if raw.get(key) is not None:
            return int(raw[key])
    text = raw.get("text_config")
    if isinstance(text, dict):
        for key in ("image_token_id", "image_token_index"):
            if text.get(key) is not None:
                return int(text[key])
    raise KeyError("image_token_id")


def _video_token_id(raw: dict[str, Any]) -> int | None:
    for key in ("video_token_id", "video_token_index"):
        if raw.get(key) is not None:
            return int(raw[key])
    return None


def _mm_token_types(
    input_ids: torch.Tensor,
    raw: dict[str, Any],
    given: torch.Tensor | None,
) -> torch.Tensor:
    if given is not None:
        return given
    types = torch.zeros_like(input_ids)
    image_id = _image_token_id(raw)
    types = types.masked_fill(input_ids == image_id, 1)
    video_id = _video_token_id(raw)
    if video_id is not None:
        types = types.masked_fill(input_ids == video_id, 2)
    return types


def _scatter_features(
    embeds: torch.Tensor,
    input_ids: torch.Tensor,
    token_id: int,
    features: torch.Tensor,
    label: str,
) -> torch.Tensor:
    mask = (input_ids == token_id).unsqueeze(-1)
    expected = int(mask.sum().item()) * embeds.shape[-1]
    if expected != features.numel():
        raise ValueError(
            f"{label} features and placeholders differ: "
            f"tokens={int(mask.sum())}, features={features.shape[0]}"
        )
    return embeds.masked_scatter(mask, features.to(device=embeds.device, dtype=embeds.dtype))


def qwen2vl_vision_expected_shapes(config: dict[str, Any]) -> dict[str, tuple[int, ...]]:
    embed = int(config.get("embed_dim") or config["hidden_size"])
    out = int(config.get("hidden_size") or embed)
    inter = int(embed * float(config.get("mlp_ratio", 4)))
    c = int(config.get("in_channels", 3))
    t = int(config.get("temporal_patch_size", 2))
    p = int(config.get("patch_size", 14))
    merge = int(config.get("spatial_merge_size", 2))
    merged = embed * merge * merge
    expected: dict[str, tuple[int, ...]] = {
        "visual.patch_embed.proj.weight": (embed, c, t, p, p),
        "visual.merger.ln_q.weight": (embed,),
        "visual.merger.ln_q.bias": (embed,),
        "visual.merger.mlp.0.weight": (merged, merged),
        "visual.merger.mlp.0.bias": (merged,),
        "visual.merger.mlp.2.weight": (out, merged),
        "visual.merger.mlp.2.bias": (out,),
    }
    for i in range(int(config["depth"])):
        prefix = f"visual.blocks.{i}"
        expected.update(
            {
                f"{prefix}.norm1.weight": (embed,),
                f"{prefix}.norm1.bias": (embed,),
                f"{prefix}.norm2.weight": (embed,),
                f"{prefix}.norm2.bias": (embed,),
                f"{prefix}.attn.qkv.weight": (3 * embed, embed),
                f"{prefix}.attn.qkv.bias": (3 * embed,),
                f"{prefix}.attn.proj.weight": (embed, embed),
                f"{prefix}.attn.proj.bias": (embed,),
                f"{prefix}.mlp.fc1.weight": (inter, embed),
                f"{prefix}.mlp.fc1.bias": (inter,),
                f"{prefix}.mlp.fc2.weight": (embed, inter),
                f"{prefix}.mlp.fc2.bias": (embed,),
            }
        )
    return expected


def qwen25vl_vision_expected_shapes(config: dict[str, Any]) -> dict[str, tuple[int, ...]]:
    h = int(config["hidden_size"])
    inter = int(config["intermediate_size"])
    out = int(config.get("out_hidden_size") or h)
    c = int(config.get("in_channels", 3))
    t = int(config.get("temporal_patch_size", 2))
    p = int(config.get("patch_size", 14))
    merge = int(config.get("spatial_merge_size", 2))
    merged = h * merge * merge
    expected: dict[str, tuple[int, ...]] = {
        "visual.patch_embed.proj.weight": (h, c, t, p, p),
        "visual.merger.ln_q.weight": (h,),
        "visual.merger.mlp.0.weight": (merged, merged),
        "visual.merger.mlp.0.bias": (merged,),
        "visual.merger.mlp.2.weight": (out, merged),
        "visual.merger.mlp.2.bias": (out,),
    }
    for i in range(int(config["depth"])):
        prefix = f"visual.blocks.{i}"
        expected.update(
            {
                f"{prefix}.norm1.weight": (h,),
                f"{prefix}.norm2.weight": (h,),
                f"{prefix}.attn.qkv.weight": (3 * h, h),
                f"{prefix}.attn.qkv.bias": (3 * h,),
                f"{prefix}.attn.proj.weight": (h, h),
                f"{prefix}.attn.proj.bias": (h,),
                f"{prefix}.mlp.gate_proj.weight": (inter, h),
                f"{prefix}.mlp.gate_proj.bias": (inter,),
                f"{prefix}.mlp.up_proj.weight": (inter, h),
                f"{prefix}.mlp.up_proj.bias": (inter,),
                f"{prefix}.mlp.down_proj.weight": (h, inter),
                f"{prefix}.mlp.down_proj.bias": (h,),
            }
        )
    return expected


def _canonical_visual(weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for name, tensor in weights.items():
        key = name
        if key.startswith("model.visual."):
            key = key[len("model.") :]
        out[key] = tensor
    return out


def qwen2vl_vision_forward(
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
    weights: dict[str, torch.Tensor],
    config: dict[str, Any],
) -> torch.Tensor:
    weights = _canonical_visual(weights)
    embed = int(config.get("embed_dim") or config["hidden_size"])
    in_channels = int(config.get("in_channels", 3))
    temporal = int(config.get("temporal_patch_size", 2))
    patch = int(config.get("patch_size", 14))
    spatial_merge = int(config.get("spatial_merge_size", 2))
    num_heads = int(config["num_heads"])
    depth = int(config["depth"])
    act = str(config.get("hidden_act", "quick_gelu"))
    proj = weights["visual.patch_embed.proj.weight"]
    patches = pixel_values.reshape(-1, in_channels, temporal, patch, patch)
    hidden = F.conv3d(
        patches.to(proj.dtype),
        proj,
        weights.get("visual.patch_embed.proj.bias"),
        stride=(temporal, patch, patch),
    ).reshape(-1, embed)

    grid_thw = grid_thw.to(device=hidden.device, dtype=torch.long)
    position_ids = _vision_position_ids(grid_thw, spatial_merge)
    frame_lengths = torch.repeat_interleave(
        grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
    )
    cu_seqlens = F.pad(
        frame_lengths.cumsum(dim=0, dtype=torch.int32), (1, 0), value=0
    ).to(hidden.device)
    cos, sin = _vision_rope(
        position_ids, embed // num_heads, hidden.device, dtype=hidden.dtype
    )
    for i in range(depth):
        p = f"visual.blocks.{i}"
        residual = hidden
        hidden = _layer_norm(hidden, weights, f"{p}.norm1")
        hidden = residual + _vision_attention(
            hidden,
            weights,
            f"{p}.attn",
            num_heads=num_heads,
            cos=cos,
            sin=sin,
            cu_seqlens=cu_seqlens,
        )
        residual = hidden
        hidden = _layer_norm(hidden, weights, f"{p}.norm2")
        hidden = _linear(hidden, weights, f"{p}.mlp.fc1")
        hidden = _act(act, hidden)
        hidden = residual + _linear(hidden, weights, f"{p}.mlp.fc2")

    hidden = _layer_norm(hidden, weights, "visual.merger.ln_q")
    hidden = hidden.reshape(-1, embed * spatial_merge * spatial_merge)
    hidden = F.gelu(_linear(hidden, weights, "visual.merger.mlp.0"))
    return _linear(hidden, weights, "visual.merger.mlp.2")


def qwen25vl_vision_forward(
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
    weights: dict[str, torch.Tensor],
    config: dict[str, Any],
) -> torch.Tensor:
    from transformers.vision_utils import (
        get_vision_position_ids,
        get_vision_window_index,
    )

    weights = _canonical_visual(weights)
    hidden_size = int(config["hidden_size"])
    in_channels = int(config.get("in_channels", 3))
    temporal = int(config.get("temporal_patch_size", 2))
    patch = int(config.get("patch_size", 14))
    spatial_merge = int(config.get("spatial_merge_size", 2))
    num_heads = int(config["num_heads"])
    depth = int(config["depth"])
    merge_unit = spatial_merge * spatial_merge
    fullatt = set(int(i) for i in (config.get("fullatt_block_indexes") or ()))
    proj = weights["visual.patch_embed.proj.weight"]
    patches = pixel_values.reshape(-1, in_channels, temporal, patch, patch)
    hidden = F.conv3d(
        patches.to(proj.dtype),
        proj,
        weights.get("visual.patch_embed.proj.bias"),
        stride=(temporal, patch, patch),
    ).reshape(-1, hidden_size)

    grid_thw = grid_thw.to(device=hidden.device, dtype=torch.long)
    position_ids = get_vision_position_ids(grid_thw, spatial_merge)
    frame_lengths = torch.repeat_interleave(
        grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
    )
    cu_seqlens = F.pad(
        frame_lengths.cumsum(dim=0, dtype=torch.int32), (1, 0), value=0
    )
    window_index, cu_window_seqlens = get_vision_window_index(
        grid_thw,
        spatial_merge_size=spatial_merge,
        window_size=int(config.get("window_size", 112)),
        patch_size=patch,
    )
    window_index = window_index.to(device=hidden.device)
    seq_len = hidden.shape[0]
    hidden = hidden.reshape(seq_len // merge_unit, merge_unit, -1)[window_index]
    hidden = hidden.reshape(seq_len, -1)
    rope_ids = position_ids.reshape(seq_len // merge_unit, merge_unit, -1)[window_index]
    rope_ids = rope_ids.reshape(seq_len, -1)
    cos, sin = _vision_rope(
        rope_ids, hidden_size // num_heads, hidden.device, dtype=hidden.dtype
    )
    cu_seqlens = cu_seqlens.to(device=hidden.device)
    cu_window_seqlens = cu_window_seqlens.to(device=hidden.device)
    for i in range(depth):
        p = f"visual.blocks.{i}"
        seqlens = cu_seqlens if i in fullatt else cu_window_seqlens
        residual = hidden
        hidden = rms_norm(hidden, weights[f"{p}.norm1.weight"], 1e-6)
        hidden = residual + _vision_attention(
            hidden,
            weights,
            f"{p}.attn",
            num_heads=num_heads,
            cos=cos,
            sin=sin,
            cu_seqlens=seqlens,
        )
        residual = hidden
        hidden = rms_norm(hidden, weights[f"{p}.norm2.weight"], 1e-6)
        gate = _act("silu", _linear(hidden, weights, f"{p}.mlp.gate_proj"))
        up = _linear(hidden, weights, f"{p}.mlp.up_proj")
        hidden = residual + _linear(gate * up, weights, f"{p}.mlp.down_proj")

    hidden = rms_norm(hidden, weights["visual.merger.ln_q.weight"], 1e-6)
    hidden = hidden.reshape(-1, hidden_size * spatial_merge * spatial_merge)
    hidden = F.gelu(_linear(hidden, weights, "visual.merger.mlp.0"))
    hidden = _linear(hidden, weights, "visual.merger.mlp.2")
    return hidden[torch.argsort(window_index)]


def gemma3_vision_expected_shapes(
    vision_config: dict[str, Any], text_hidden: int, *, depth: int | None = None
) -> dict[str, tuple[int, ...]]:
    h = int(vision_config["hidden_size"])
    inter = int(vision_config["intermediate_size"])
    image_size = int(vision_config["image_size"])
    patch = int(vision_config["patch_size"])
    npos = (image_size // patch) ** 2
    layers = int(depth if depth is not None else vision_config["num_hidden_layers"])
    expected: dict[str, tuple[int, ...]] = {
        "vision_tower.embeddings.patch_embedding.weight": (h, 3, patch, patch),
        "vision_tower.embeddings.patch_embedding.bias": (h,),
        "vision_tower.embeddings.position_embedding.weight": (npos, h),
        "vision_tower.post_layernorm.weight": (h,),
        "vision_tower.post_layernorm.bias": (h,),
        "multi_modal_projector.mm_input_projection_weight": (h, text_hidden),
        "multi_modal_projector.mm_soft_emb_norm.weight": (h,),
    }
    for i in range(layers):
        p = f"vision_tower.encoder.layers.{i}"
        expected.update(
            {
                f"{p}.layer_norm1.weight": (h,),
                f"{p}.layer_norm1.bias": (h,),
                f"{p}.layer_norm2.weight": (h,),
                f"{p}.layer_norm2.bias": (h,),
                f"{p}.self_attn.q_proj.weight": (h, h),
                f"{p}.self_attn.q_proj.bias": (h,),
                f"{p}.self_attn.k_proj.weight": (h, h),
                f"{p}.self_attn.k_proj.bias": (h,),
                f"{p}.self_attn.v_proj.weight": (h, h),
                f"{p}.self_attn.v_proj.bias": (h,),
                f"{p}.self_attn.out_proj.weight": (h, h),
                f"{p}.self_attn.out_proj.bias": (h,),
                f"{p}.mlp.fc1.weight": (inter, h),
                f"{p}.mlp.fc1.bias": (inter,),
                f"{p}.mlp.fc2.weight": (h, inter),
                f"{p}.mlp.fc2.bias": (h,),
            }
        )
    return expected


def _canonical_gemma3(weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for name, tensor in weights.items():
        key = name
        if key.startswith("model."):
            key = key[len("model.") :]
        if key.startswith("vision_tower.vision_model."):
            key = "vision_tower." + key[len("vision_tower.vision_model.") :]
        out[key] = tensor
    return out


def gemma3_vision_forward(
    pixel_values: torch.Tensor,
    weights: dict[str, torch.Tensor],
    vision_config: dict[str, Any],
    raw_config: dict[str, Any],
) -> torch.Tensor:
    weights = _canonical_gemma3(weights)
    hidden_size = int(vision_config["hidden_size"])
    patch = int(vision_config["patch_size"])
    image_size = int(vision_config["image_size"])
    num_heads = int(vision_config["num_attention_heads"])
    depth = int(vision_config["num_hidden_layers"])
    eps = float(vision_config.get("layer_norm_eps", 1e-6))
    act = str(vision_config.get("hidden_act", "gelu_pytorch_tanh"))
    conv = weights["vision_tower.embeddings.patch_embedding.weight"]
    hidden = F.conv2d(
        pixel_values.to(conv.dtype),
        conv,
        weights.get("vision_tower.embeddings.patch_embedding.bias"),
        stride=patch,
    )
    batch, _, height, width = hidden.shape
    hidden = hidden.flatten(2).transpose(1, 2)
    pos = weights["vision_tower.embeddings.position_embedding.weight"]
    if hidden.shape[1] != pos.shape[0]:
        side = int(pos.shape[0] ** 0.5)
        table = pos.reshape(1, side, side, hidden_size).permute(0, 3, 1, 2)
        table = F.interpolate(
            table, size=(height, width), mode="bicubic", align_corners=False
        )
        pos = table.permute(0, 2, 3, 1).reshape(1, -1, hidden_size)
        hidden = hidden + pos.to(hidden.dtype)
    else:
        hidden = hidden + pos.to(hidden.dtype)

    head_dim = hidden_size // num_heads
    scale = head_dim**-0.5
    for i in range(depth):
        p = f"vision_tower.encoder.layers.{i}"
        residual = hidden
        h = F.layer_norm(
            hidden,
            (hidden_size,),
            weights[f"{p}.layer_norm1.weight"],
            weights.get(f"{p}.layer_norm1.bias"),
            eps,
        )
        q = _linear(h, weights, f"{p}.self_attn.q_proj")
        k = _linear(h, weights, f"{p}.self_attn.k_proj")
        v = _linear(h, weights, f"{p}.self_attn.v_proj")
        q = q.view(batch, -1, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch, -1, num_heads, head_dim).transpose(1, 2)
        v = v.view(batch, -1, num_heads, head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        attn = attn.transpose(1, 2).reshape(batch, -1, hidden_size)
        hidden = residual + _linear(attn, weights, f"{p}.self_attn.out_proj")
        residual = hidden
        h = F.layer_norm(
            hidden,
            (hidden_size,),
            weights[f"{p}.layer_norm2.weight"],
            weights.get(f"{p}.layer_norm2.bias"),
            eps,
        )
        h = _act(act, _linear(h, weights, f"{p}.mlp.fc1"))
        hidden = residual + _linear(h, weights, f"{p}.mlp.fc2")

    hidden = F.layer_norm(
        hidden,
        (hidden_size,),
        weights["vision_tower.post_layernorm.weight"],
        weights.get("vision_tower.post_layernorm.bias"),
        eps,
    )

    tokens = int(raw_config.get("mm_tokens_per_image", 256))
    patches_per_image = image_size // patch
    tokens_per_side = int(tokens**0.5)
    kernel = patches_per_image // tokens_per_side
    pooled = F.avg_pool2d(
        hidden.transpose(1, 2).reshape(batch, hidden_size, patches_per_image, patches_per_image),
        kernel_size=kernel,
        stride=kernel,
    )
    pooled = pooled.flatten(2).transpose(1, 2)
    pooled = gemma_rms_norm(
        pooled,
        weights["multi_modal_projector.mm_soft_emb_norm.weight"],
        float(vision_config.get("layer_norm_eps", 1e-6)),
    )
    projected = torch.matmul(
        pooled, weights["multi_modal_projector.mm_input_projection_weight"]
    )
    return projected.reshape(-1, projected.shape[-1])


def llama4_vision_expected_shapes(
    vision_config: dict[str, Any], text_hidden: int, *, depth: int | None = None
) -> dict[str, tuple[int, ...]]:
    h = int(vision_config["hidden_size"])
    inter = int(vision_config["intermediate_size"])
    image_size = int(vision_config["image_size"])
    patch = int(vision_config["patch_size"])
    channels = int(vision_config.get("num_channels", 3))
    npos = (image_size // patch) ** 2 + 1
    layers = int(depth if depth is not None else vision_config["num_hidden_layers"])
    proj_in = int(vision_config.get("projector_input_dim", inter))
    proj_out = int(vision_config.get("projector_output_dim", h))
    vision_out = int(vision_config.get("vision_output_dim", proj_out))
    expected: dict[str, tuple[int, ...]] = {
        "vision_model.patch_embedding.linear.weight": (h, channels * patch * patch),
        "vision_model.class_embedding": (h,),
        "vision_model.positional_embedding_vlm": (npos, h),
        "vision_model.layernorm_pre.weight": (h,),
        "vision_model.layernorm_pre.bias": (h,),
        "vision_model.layernorm_post.weight": (h,),
        "vision_model.layernorm_post.bias": (h,),
        "vision_model.vision_adapter.mlp.fc1.weight": (proj_in, inter),
        "vision_model.vision_adapter.mlp.fc2.weight": (proj_out, proj_out),
        "multi_modal_projector.linear_1.weight": (text_hidden, vision_out),
    }
    for i in range(layers):
        p = f"vision_model.model.layers.{i}"
        expected.update(
            {
                f"{p}.input_layernorm.weight": (h,),
                f"{p}.input_layernorm.bias": (h,),
                f"{p}.post_attention_layernorm.weight": (h,),
                f"{p}.post_attention_layernorm.bias": (h,),
                f"{p}.self_attn.q_proj.weight": (h, h),
                f"{p}.self_attn.q_proj.bias": (h,),
                f"{p}.self_attn.k_proj.weight": (h, h),
                f"{p}.self_attn.k_proj.bias": (h,),
                f"{p}.self_attn.v_proj.weight": (h, h),
                f"{p}.self_attn.v_proj.bias": (h,),
                f"{p}.self_attn.o_proj.weight": (h, h),
                f"{p}.self_attn.o_proj.bias": (h,),
                f"{p}.mlp.fc1.weight": (inter, h),
                f"{p}.mlp.fc1.bias": (inter,),
                f"{p}.mlp.fc2.weight": (h, inter),
                f"{p}.mlp.fc2.bias": (h,),
            }
        )
    return expected


def _llama4_freqs_ci(config: dict[str, Any], device: torch.device) -> torch.Tensor:
    idx = int(config["image_size"]) // int(config["patch_size"])
    img_idx = torch.arange(idx**2, dtype=torch.int32, device=device).reshape(idx**2, 1)
    img_idx = torch.cat([img_idx, img_idx[:1]], dim=0)
    img_idx = img_idx.clone()
    img_idx[-1, -1] = -2
    frequencies_x = img_idx % idx
    frequencies_y = img_idx // idx
    heads = int(config["num_attention_heads"])
    freq_dim = int(config["hidden_size"]) // heads // 2
    theta = float((config.get("rope_parameters") or {}).get("rope_theta", 10000.0))
    rope_freq = 1.0 / (
        theta ** (torch.arange(0, freq_dim, 2, device=device)[: (freq_dim // 2)].float() / freq_dim)
    )
    freqs_x = ((frequencies_x + 1)[..., None] * rope_freq[None, None, :]).repeat_interleave(
        2, dim=-1
    )
    freqs_y = ((frequencies_y + 1)[..., None] * rope_freq[None, None, :]).repeat_interleave(
        2, dim=-1
    )
    freqs = torch.cat([freqs_x, freqs_y], dim=-1).float().contiguous()[..., ::2]
    freqs = freqs.masked_fill(img_idx.reshape(-1, 1, 1) < 0, 0)
    return torch.view_as_complex(torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1))


def _llama4_apply_rope(
    query: torch.Tensor, key: torch.Tensor, freqs_ci: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    query_c = torch.view_as_complex(query.float().reshape(*query.shape[:-1], -1, 2))
    key_c = torch.view_as_complex(key.float().reshape(*key.shape[:-1], -1, 2))
    shape = [d if i == 1 or i == query_c.ndim - 1 else 1 for i, d in enumerate(query_c.shape)]
    freqs = freqs_ci.view(*shape).to(query_c.device)
    query_out = torch.view_as_real(query_c * freqs).flatten(3)
    key_out = torch.view_as_real(key_c * freqs).flatten(3)
    return query_out.type_as(query), key_out.type_as(key)


def _pixel_shuffle(tensor: torch.Tensor, ratio: float) -> torch.Tensor:
    batch, num_patches, channels = tensor.shape
    patch_size = int(math.sqrt(num_patches))
    tensor = tensor.view(batch, patch_size, patch_size, -1)
    height, width = tensor.shape[1], tensor.shape[2]
    channels = tensor.shape[-1]
    tensor = tensor.view(batch, height, int(width * ratio), int(channels / ratio))
    tensor = tensor.permute(0, 2, 1, 3).contiguous()
    tensor = tensor.view(
        batch,
        int(height * ratio),
        int(width * ratio),
        int(channels / (ratio**2)),
    )
    tensor = tensor.permute(0, 2, 1, 3).contiguous()
    return tensor.view(batch, -1, tensor.shape[-1])


def llama4_vision_forward(
    pixel_values: torch.Tensor,
    weights: dict[str, torch.Tensor],
    vision_config: dict[str, Any],
) -> torch.Tensor:
    h = int(vision_config["hidden_size"])
    patch = int(vision_config["patch_size"])
    channels = int(vision_config.get("num_channels", 3))
    depth = int(vision_config["num_hidden_layers"])
    heads = int(vision_config["num_attention_heads"])
    eps = float(vision_config.get("norm_eps", 1e-5))
    ratio = float(vision_config.get("pixel_shuffle_ratio", 0.5))
    unfolded = F.unfold(pixel_values, kernel_size=patch, stride=patch)
    unfolded = unfolded.permute(0, 2, 1)
    hidden = F.linear(unfolded, weights["vision_model.patch_embedding.linear.weight"])
    batch = hidden.shape[0]
    cls = weights["vision_model.class_embedding"].to(hidden.dtype)
    hidden = torch.cat([hidden, cls.expand(batch, 1, -1)], dim=1)
    hidden = hidden + weights["vision_model.positional_embedding_vlm"].to(hidden.dtype)
    hidden = F.layer_norm(
        hidden,
        (h,),
        weights["vision_model.layernorm_pre.weight"],
        weights.get("vision_model.layernorm_pre.bias"),
        eps,
    )
    freqs = _llama4_freqs_ci(vision_config, hidden.device)
    head_dim = h // heads
    scale = head_dim**-0.5
    for i in range(depth):
        p = f"vision_model.model.layers.{i}"
        residual = hidden
        h_norm = F.layer_norm(
            hidden,
            (h,),
            weights[f"{p}.input_layernorm.weight"],
            weights.get(f"{p}.input_layernorm.bias"),
            eps,
        )
        q = _linear(h_norm, weights, f"{p}.self_attn.q_proj").view(batch, -1, heads, head_dim)
        k = _linear(h_norm, weights, f"{p}.self_attn.k_proj").view(batch, -1, heads, head_dim)
        v = _linear(h_norm, weights, f"{p}.self_attn.v_proj").view(batch, -1, heads, head_dim)
        q, k = _llama4_apply_rope(q, k, freqs)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False, scale=scale)
        attn = attn.transpose(1, 2).reshape(batch, -1, h)
        hidden = residual + _linear(attn, weights, f"{p}.self_attn.o_proj")
        residual = hidden
        h_norm = F.layer_norm(
            hidden,
            (h,),
            weights[f"{p}.post_attention_layernorm.weight"],
            weights.get(f"{p}.post_attention_layernorm.bias"),
            eps,
        )
        h_norm = F.gelu(_linear(h_norm, weights, f"{p}.mlp.fc1"))
        hidden = residual + _linear(h_norm, weights, f"{p}.mlp.fc2")

    hidden = F.layer_norm(
        hidden,
        (h,),
        weights["vision_model.layernorm_post.weight"],
        weights.get("vision_model.layernorm_post.bias"),
        eps,
    )
    hidden = hidden[:, :-1, :]
    hidden = _pixel_shuffle(hidden, ratio)
    hidden = F.linear(hidden, weights["vision_model.vision_adapter.mlp.fc1.weight"])
    hidden = F.gelu(hidden)
    hidden = F.gelu(F.linear(hidden, weights["vision_model.vision_adapter.mlp.fc2.weight"]))
    hidden = F.linear(hidden, weights["multi_modal_projector.linear_1.weight"])
    return hidden.reshape(-1, hidden.shape[-1])


def mistral3_vision_expected_shapes(
    vision_config: dict[str, Any],
    text_hidden: int,
    *,
    spatial_merge_size: int = 2,
    projector_bias: bool = False,
    depth: int | None = None,
) -> dict[str, tuple[int, ...]]:
    h = int(vision_config["hidden_size"])
    inter = int(vision_config["intermediate_size"])
    patch = int(vision_config["patch_size"])
    layers = int(depth if depth is not None else vision_config["num_hidden_layers"])
    merge = int(spatial_merge_size)
    expected: dict[str, tuple[int, ...]] = {
        "vision_tower.patch_conv.weight": (h, 3, patch, patch),
        "vision_tower.ln_pre.weight": (h,),
        "multi_modal_projector.norm.weight": (h,),
        "multi_modal_projector.patch_merger.merging_layer.weight": (h, h * merge * merge),
        "multi_modal_projector.linear_1.weight": (text_hidden, h),
        "multi_modal_projector.linear_2.weight": (text_hidden, text_hidden),
    }
    if projector_bias:
        expected["multi_modal_projector.linear_1.bias"] = (text_hidden,)
        expected["multi_modal_projector.linear_2.bias"] = (text_hidden,)
    for i in range(layers):
        p = f"vision_tower.transformer.layers.{i}"
        expected.update(
            {
                f"{p}.attention_norm.weight": (h,),
                f"{p}.ffn_norm.weight": (h,),
                f"{p}.attention.q_proj.weight": (h, h),
                f"{p}.attention.k_proj.weight": (h, h),
                f"{p}.attention.v_proj.weight": (h, h),
                f"{p}.attention.o_proj.weight": (h, h),
                f"{p}.feed_forward.gate_proj.weight": (inter, h),
                f"{p}.feed_forward.up_proj.weight": (inter, h),
                f"{p}.feed_forward.down_proj.weight": (h, inter),
            }
        )
    return expected


def _canonical_mistral3(weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for name, tensor in weights.items():
        key = name
        if key.startswith("model."):
            key = key[len("model.") :]
        out[key] = tensor
    return out


def _pixtral_inv_freq(config: dict[str, Any], device: torch.device) -> torch.Tensor:
    image_size = int(config["image_size"])
    patch = int(config["patch_size"])
    max_side = image_size // patch
    dim = int(config["hidden_size"]) // int(config["num_attention_heads"])
    theta = float((config.get("rope_parameters") or {}).get("rope_theta", 10000.0))
    freqs = 1.0 / (
        theta
        ** (
            torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim
        )
    )
    height = torch.arange(max_side, device=device)
    width = torch.arange(max_side, device=device)
    freqs_h = torch.outer(height, freqs[::2]).float()
    freqs_w = torch.outer(width, freqs[1::2]).float()
    inv = torch.cat(
        [
            freqs_h[:, None, :].repeat(1, max_side, 1),
            freqs_w[None, :, :].repeat(max_side, 1, 1),
        ],
        dim=-1,
    ).reshape(-1, dim // 2)
    return torch.cat((inv, inv), dim=-1)


def _pixtral_position_ids(
    heights_widths: list[tuple[int, int]], max_width: int, device: torch.device
) -> torch.Tensor:
    positions: list[torch.Tensor] = []
    for height, width in heights_widths:
        mesh = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing="ij",
        )
        h_grid, v_grid = torch.stack(mesh, dim=-1).reshape(-1, 2).chunk(2, -1)
        positions.append((h_grid * max_width + v_grid)[:, 0])
    return torch.cat(positions)


def mistral3_vision_forward(
    pixel_values: torch.Tensor,
    weights: dict[str, torch.Tensor],
    vision_config: dict[str, Any],
    raw_config: dict[str, Any],
    image_sizes: torch.Tensor | None = None,
) -> torch.Tensor:
    weights = _canonical_mistral3(weights)
    h = int(vision_config["hidden_size"])
    patch = int(vision_config["patch_size"])
    depth = int(vision_config["num_hidden_layers"])
    heads = int(vision_config["num_attention_heads"])
    image_size = int(vision_config["image_size"])
    merge = int(raw_config.get("spatial_merge_size", 2))
    act = str(vision_config.get("hidden_act", "silu"))
    conv = weights["vision_tower.patch_conv.weight"]
    patches = F.conv2d(pixel_values.to(conv.dtype), conv, bias=None, stride=patch)
    if image_sizes is None:
        image_sizes = torch.tensor(
            [[pixel_values.shape[-2], pixel_values.shape[-1]]] * pixel_values.shape[0],
            device=pixel_values.device,
        )
    cropped: list[torch.Tensor] = []
    hw: list[tuple[int, int]] = []
    for embed, size in zip(patches, image_sizes.tolist()):
        height, width = int(size[0]) // patch, int(size[1]) // patch
        cropped.append(embed[..., :height, :width])
        hw.append((height, width))
    hidden = torch.cat([piece.flatten(1).T for piece in cropped], dim=0).unsqueeze(0)
    hidden = rms_norm(hidden, weights["vision_tower.ln_pre.weight"], 1e-5)
    max_width = image_size // patch
    pos = _pixtral_position_ids(hw, max_width, hidden.device)
    inv = _pixtral_inv_freq(vision_config, hidden.device)
    emb = inv[pos]
    cos, sin = emb.cos().to(hidden.dtype), emb.sin().to(hidden.dtype)
    lengths = [h_i * w_i for h_i, w_i in hw]
    seq = hidden.shape[1]
    block = torch.full((seq, seq), torch.finfo(hidden.dtype).min, device=hidden.device)
    start = 0
    for length in lengths:
        block[start : start + length, start : start + length] = 0
        start += length
    mask = block[None, None]
    head_dim = h // heads
    scale = head_dim**-0.5
    batch = hidden.shape[0]
    for i in range(depth):
        p = f"vision_tower.transformer.layers.{i}"
        residual = hidden
        h_norm = rms_norm(hidden, weights[f"{p}.attention_norm.weight"], 1e-5)
        q = _linear(h_norm, weights, f"{p}.attention.q_proj")
        k = _linear(h_norm, weights, f"{p}.attention.k_proj")
        v = _linear(h_norm, weights, f"{p}.attention.v_proj")
        q = q.view(batch, seq, heads, head_dim).transpose(1, 2)
        k = k.view(batch, seq, heads, head_dim).transpose(1, 2)
        v = v.view(batch, seq, heads, head_dim).transpose(1, 2)
        cos_h = cos.unsqueeze(0)
        sin_h = sin.unsqueeze(0)
        q = (q * cos_h) + (rotate_half(q) * sin_h)
        k = (k * cos_h) + (rotate_half(k) * sin_h)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale + mask
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        attn = torch.matmul(probs, v).transpose(1, 2).reshape(batch, seq, h)
        hidden = residual + _linear(attn, weights, f"{p}.attention.o_proj")
        residual = hidden
        h_norm = rms_norm(hidden, weights[f"{p}.ffn_norm.weight"], 1e-5)
        gate = _act(act, _linear(h_norm, weights, f"{p}.feed_forward.gate_proj"))
        up = _linear(h_norm, weights, f"{p}.feed_forward.up_proj")
        hidden = residual + _linear(gate * up, weights, f"{p}.feed_forward.down_proj")

    features = hidden[0]
    features = rms_norm(
        features,
        weights["multi_modal_projector.norm.weight"],
        float((raw_config.get("text_config") or {}).get("rms_norm_eps", 1e-5)),
    )
    merged: list[torch.Tensor] = []
    offset = 0
    for height, width in hw:
        tokens = features[offset : offset + height * width]
        offset += height * width
        grid = tokens.view(height, width, h).permute(2, 0, 1).unsqueeze(0)
        unfolded = F.unfold(grid, kernel_size=merge, stride=merge)
        unfolded = unfolded.view(h * merge * merge, -1).t()
        merged.append(unfolded)
    features = torch.cat(merged, dim=0)
    features = F.linear(
        features, weights["multi_modal_projector.patch_merger.merging_layer.weight"]
    )
    features = _linear(features, weights, "multi_modal_projector.linear_1")
    features = _act(str(raw_config.get("projector_hidden_act", "gelu")), features)
    return _linear(features, weights, "multi_modal_projector.linear_2")


def validate_vision_weights(
    family: str,
    weights: dict[str, torch.Tensor],
    raw_config: dict[str, Any],
) -> None:
    vcfg = raw_config.get("vision_config")
    if not isinstance(vcfg, dict):
        raise ValueError("vision_config is required")
    text_hidden = int(
        (raw_config.get("text_config") or {}).get("hidden_size")
        or raw_config.get("hidden_size")
        or vcfg.get("out_hidden_size")
        or 0
    )
    if family == "qwen3_5":
        from engine.vision import validate_qwen35_vision_weights

        validate_qwen35_vision_weights(weights, vcfg)
        return
    if family == "qwen2_vl":
        expected = qwen2vl_vision_expected_shapes(vcfg)
        canon = _canonical_visual(weights)
    elif family == "qwen2_5_vl":
        expected = qwen25vl_vision_expected_shapes(vcfg)
        canon = _canonical_visual(weights)
    elif family == "gemma3":
        expected = gemma3_vision_expected_shapes(vcfg, text_hidden)
        canon = _canonical_gemma3(weights)
    elif family == "llama4":
        expected = llama4_vision_expected_shapes(vcfg, text_hidden)
        canon = weights
    elif family == "mistral3":
        expected = mistral3_vision_expected_shapes(
            vcfg,
            text_hidden,
            spatial_merge_size=int(raw_config.get("spatial_merge_size", 2)),
            projector_bias=bool(raw_config.get("multimodal_projector_bias", False)),
        )
        canon = _canonical_mistral3(weights)
    else:
        raise ValueError(f"unknown vision family {family}")
    missing = sorted(set(expected) - set(canon))
    if missing:
        raise KeyError(f"{family} vision missing tensors: {missing[:8]}")
    for name, shape in expected.items():
        if tuple(canon[name].shape) != shape:
            raise ValueError(
                f"{family} vision {name}: got {tuple(canon[name].shape)}, expected {shape}"
            )


def vision_prefixes(family: str) -> tuple[str, ...]:
    if family in {"qwen3_5"}:
        return ("model.visual.",)
    if family in {"qwen2_vl", "qwen2_5_vl"}:
        return ("visual.", "model.visual.")
    if family == "gemma3":
        return ("model.vision_tower.", "vision_tower.", "model.multi_modal_projector.")
    if family == "llama4":
        return ("vision_model.", "multi_modal_projector.")
    if family == "mistral3":
        return ("model.vision_tower.", "model.multi_modal_projector.", "vision_tower.")
    raise ValueError(family)


def multimodal_embeddings(
    raw_config: dict[str, Any],
    input_ids: torch.Tensor,
    text_embeddings: torch.Tensor,
    vision_weights: dict[str, torch.Tensor],
    *,
    pixel_values: torch.Tensor | None = None,
    pixel_values_videos: torch.Tensor | None = None,
    image_grid_thw: torch.Tensor | None = None,
    video_grid_thw: torch.Tensor | None = None,
    mm_token_type_ids: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    image_sizes: torch.Tensor | None = None,
    recipe_id: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Insert vision tokens and return (embeds, position_ids, rope_delta)."""
    family = detect_vision_family(raw_config, recipe_id)
    if family is None:
        raise ValueError("no implemented vision family for this config")
    vcfg = raw_config.get("vision_config")
    if not isinstance(vcfg, dict):
        raise ValueError("vision_config is required")

    if family == "qwen3_5":
        if mm_token_type_ids is None:
            mm_token_type_ids = _mm_token_types(input_ids, raw_config, None)
        return qwen35_multimodal_embeddings(
            input_ids,
            text_embeddings,
            vision_weights,
            raw_config,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            attention_mask=attention_mask,
        )

    x = text_embeddings.clone()
    if family in {"qwen2_vl", "qwen2_5_vl"}:
        forward = qwen2vl_vision_forward if family == "qwen2_vl" else qwen25vl_vision_forward
        image_id = _image_token_id(raw_config)
        video_id = _video_token_id(raw_config)
        if pixel_values is not None:
            if image_grid_thw is None:
                raise ValueError("image_grid_thw is required with image pixels")
            features = forward(pixel_values, image_grid_thw, vision_weights, vcfg)
            x = _scatter_features(x, input_ids, image_id, features, "image")
        if pixel_values_videos is not None and video_id is not None:
            if video_grid_thw is None:
                raise ValueError("video_grid_thw is required with video pixels")
            features = forward(
                pixel_values_videos, video_grid_thw, vision_weights, vcfg
            )
            x = _scatter_features(x, input_ids, video_id, features, "video")
        types = _mm_token_types(input_ids, raw_config, mm_token_type_ids)
        positions, delta = qwen35_rope_index(
            input_ids,
            types,
            spatial_merge_size=int(vcfg.get("spatial_merge_size", 2)),
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
        )
        return x, positions, delta

    if pixel_values is None:
        return x, None, None
    image_id = _image_token_id(raw_config)
    if family == "gemma3":
        features = gemma3_vision_forward(pixel_values, vision_weights, vcfg, raw_config)
    elif family == "llama4":
        features = llama4_vision_forward(pixel_values, vision_weights, vcfg)
    else:
        features = mistral3_vision_forward(
            pixel_values, vision_weights, vcfg, raw_config, image_sizes=image_sizes
        )
    x = _scatter_features(x, input_ids, image_id, features, "image")
    return x, None, None
