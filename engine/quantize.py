"""Pack a loaded state dict's projection weights in place.

Decode reads every weight once per token, so this is the largest single lever
on tokens/second: 4-bit weights move a quarter of the bytes. Only the big
matrix multiplies are packed — norms, biases, routers, embeddings and anything
whose inner dimension the kernel cannot vectorize stay in the compute dtype.
"""

from __future__ import annotations

import os
import re

import torch

from engine.memory import check_floor
from engine.qweight import QuantWeight, concat_quant_weights, quantize

# Projections the engine runs through `dense()`. Everything else is either tiny
# or is not a matrix multiply against the activation.
_PROJECTION_SUFFIXES = (
    ".attn.q.weight",
    ".attn.k.weight",
    ".attn.v.weight",
    ".attn.o.weight",
    ".attn.qkv.weight",
    ".attn.c_attn.weight",
    ".attn.c_proj.weight",
    ".attn.q_a.weight",
    ".attn.q_b.weight",
    ".attn.kv_a.weight",
    ".attn.kv_b.weight",
    ".mlp.gate.weight",
    ".mlp.up.weight",
    ".mlp.down.weight",
    ".mlp.gate_up.weight",
    ".mlp.c_fc.weight",
    ".mlp.c_proj.weight",
)

# Per-expert tensors only. The stacked [E, N, K] variants and the packed
# `gate_up` forms get reshaped or chunked by the MoE layer, so they stay dense
# until the grouped-GEMM path handles them.
_MOE_EXPERT = re.compile(r"\.moe\.(experts\.\d+|shared)\.(gate|up|down)\.weight$")

# Below this the packing overhead (a scale per group) stops paying for itself
# and the rounding error is proportionally larger.
_MIN_NUMEL = 1 << 18
MIN_QUANT_NUMEL = _MIN_NUMEL


def quantize_head() -> bool:
    """Whether the output head is packed with everything else. It is.

    The head is the one projection whose rounding reaches the logits with no
    later layer to absorb it, which makes holding it dense the obvious first
    thing to try when a quantized run drifts. Measured on the GB10 at NVFP4, it
    does not pay: decode gave up 17-27% (Llama-3.2-1B 173 to 133 tok/s, gemma-2
    97 to 71) because at these vocabularies the head is worth several layers of
    traffic, and the output did not reliably improve — Llama held 12 of 16
    greedy tokens dense against 16 of 16 packed. Only Gemma-2 improved, and only
    its logit gap (0.94 to 0.63). So NVFP4's drift is spread across the weights
    rather than concentrated here. INFER_QUANT_HEAD=0 keeps it dense; the knob
    stays because it is the right experiment to be able to repeat per model.
    """
    return os.environ.get("INFER_QUANT_HEAD", "1") == "1"


def is_quantizable_name(name: str) -> bool:
    if name == "lm_head.weight":
        return quantize_head()
    if any(name.endswith(suffix) for suffix in _PROJECTION_SUFFIXES):
        return True
    return bool(_MOE_EXPERT.search(name))


def eligible_for_quant(name: str, tensor: torch.Tensor, min_numel: int = _MIN_NUMEL) -> bool:
    """Whether this tensor is one of the big matrix multiplies worth packing."""
    return (
        isinstance(tensor, torch.Tensor)
        and not name.startswith("_")
        and tensor.dim() == 2
        and tensor.numel() >= min_numel
        and tensor.shape[1] % 64 == 0
        and is_quantizable_name(name)
        and tensor.dtype in (torch.bfloat16, torch.float16, torch.float32)
    )


def _group_for(kind: str, in_features: int, requested: int) -> int | None:
    """Largest usable group at or below ``requested``, or None if unusable."""
    if kind == "fp8":
        return in_features
    if kind == "nvfp4":
        return 16 if in_features % 16 == 0 else None
    for group in (requested, 128, 64, 32):
        if group and in_features % group == 0 and group % 32 == 0:
            return group
    return None


# Public alias: the loader needs the same eligibility rules while streaming.
group_for_quant = _group_for


def quantize_state_dict(
    weights: dict[str, torch.Tensor],
    *,
    kind: str,
    group_size: int = 128,
    skip: tuple[str, ...] = (),
    min_numel: int = _MIN_NUMEL,
) -> dict[str, object]:
    """Return a new dict with eligible weights replaced by `QuantWeight`.

    Consumes ``weights``: each source tensor is released as soon as it is
    packed, so peak memory stays near the dense model rather than holding both
    copies of every layer at once. Tensors that share storage (a tied LM head)
    are packed once per distinct storage.
    """
    out: dict[str, object] = {}
    by_storage: dict[tuple[int, int], QuantWeight] = {}
    for name in list(weights):
        tensor = weights[name]
        if name.startswith("_"):
            out[name] = _quantize_stacks(weights, name, kind=kind, group_size=group_size)
            continue
        if (
            name in skip
            or not isinstance(tensor, torch.Tensor)
            or tensor.dim() != 2
            or tensor.numel() < min_numel
            or not is_quantizable_name(name)
            or tensor.dtype not in (torch.bfloat16, torch.float16, torch.float32)
        ):
            out[name] = tensor
            continue
        in_features = int(tensor.shape[1])
        group = _group_for(kind, in_features, group_size)
        if group is None or in_features % 64 != 0:
            out[name] = tensor
            continue
        key = (tensor.data_ptr(), tensor.numel())
        packed = by_storage.get(key)
        if packed is None:
            dense = tensor.contiguous()
            packed = quantize(dense, kind, group_size=group)
            by_storage[key] = packed
            del dense
        out[name] = packed
        # Release the dense copy now; a 30B checkpoint will not hold both.
        del tensor
        del weights[name]
    return out


def _quantize_stacks(
    weights: dict[str, object], name: str, *, kind: str, group_size: int
) -> object:
    """Pack stacked ``[E, N, K]`` expert blocks as one ``[E * N, K]`` weight.

    A routed row reads weight row ``expert * N + n``, so flattening the expert
    axis into the row axis lets the quantized grouped GEMV use exactly the same
    packing and scale layout as a dense projection.
    """
    from engine.layers.moe import EXPERT_STACK_KEY

    value = weights[name]
    if name != EXPERT_STACK_KEY or not isinstance(value, dict):
        return value
    for prefix, fields in value.items():
        for field in list(fields):
            block = fields[field]
            if not isinstance(block, torch.Tensor) or block.dim() != 3:
                continue
            experts, n_cols, k_dim = block.shape
            group = _group_for(kind, int(k_dim), group_size)
            if group is None or k_dim % 64 != 0:
                continue
            flat = block.reshape(experts * n_cols, k_dim)
            packed = quantize(flat, kind, group_size=group)
            packed.experts = experts
            packed.expert_cols = int(n_cols)
            fields[field] = packed
            del block, flat
    return value


_EXPERT_KEY = re.compile(r"^(layers\.\d+\.moe)\.experts\.(\d+)\.(gate|up|down)\.(weight|bias)$")

# Recipes whose checkpoints already ship one [E, ...] block per projection.
_PACKED_KEY = re.compile(r"^(layers\.\d+\.moe)\.experts\.(gate_up|down)\.(weight|bias)$")


def adopt_packed_experts(
    weights: dict[str, object], *, transposed: bool = False
) -> dict[str, float]:
    """Register checkpoint-provided ``[E, N, K]`` expert blocks as stacks.

    Qwen3-MoE, GPT-OSS and Llama4 already store experts as one tensor per
    projection, which the Python dispatch loop indexed one expert at a time.
    Handing the same tensors to the grouped GEMV costs nothing at load and
    removes the loop. GPT-OSS and Llama4 store ``token @ weight``, so those get
    transposed once here rather than on every token.
    """
    from engine.layers.moe import EXPERT_STACK_KEY

    stacks: dict[str, dict[str, object]] = weights.get(EXPERT_STACK_KEY) or {}
    adopted = 0
    for name in list(weights):
        m = _PACKED_KEY.match(name)
        if not m:
            continue
        value = weights[name]
        prefix, field, kindpart = m.group(1), m.group(2), m.group(3)
        is_bias = kindpart == "bias"
        # A per-expert bias is [E, N] where the weight is [E, N, K]. Both have to
        # be adopted: once a stack exists the layer reads biases only out of it,
        # so a bias left behind here is a bias silently dropped from the math.
        want_dim = 2 if is_bias else 3
        if not isinstance(value, torch.Tensor) or value.dim() != want_dim:
            continue
        block = value.transpose(1, 2).contiguous() if transposed and not is_bias else value
        key = f"{field}_bias" if is_bias else field
        stacks.setdefault(prefix, {})[key] = block
        adopted += 1
        # The block owns the data now; the layer reads experts out of the stack
        # when the kernel declines, so keeping a second entry would only pin the
        # dense copy after packing.
        del weights[name]
    if not stacks:
        return {"adopted": 0}
    weights[EXPERT_STACK_KEY] = stacks
    return {"adopted": adopted, "layers": len(stacks)}


def stack_moe_experts(weights: dict[str, object]) -> dict[str, float]:
    """Concatenate per-expert tensors into ``[E, N, K]`` blocks for the kernel.

    The checkpoints store one tensor per expert, which forces the dispatch loop
    to touch every expert from Python. One contiguous block per projection lets
    the grouped GEMV read ``w[expert]`` on the device instead, and the original
    tensors are dropped so resident memory does not grow.
    """
    names: dict[str, dict[str, dict[int, str]]] = {}
    for name, value in weights.items():
        m = _EXPERT_KEY.match(name)
        if not m or not isinstance(value, (torch.Tensor, QuantWeight)):
            continue
        prefix, index, proj, kindpart = m.group(1), int(m.group(2)), m.group(3), m.group(4)
        field = proj if kindpart == "weight" else f"{proj}_bias"
        names.setdefault(prefix, {}).setdefault(field, {})[index] = name

    stacks: dict[str, dict[str, torch.Tensor]] = {}
    stacked_bytes = 0
    # One layer at a time, freeing each source as soon as it is copied: holding
    # every layer's block and its per-expert originals at once would double the
    # resident MoE weights, which is tens of gigabytes on a 30B checkpoint.
    for prefix, fields in names.items():
        built: dict[str, torch.Tensor] = {}
        for field, by_index in fields.items():
            n_experts = max(by_index) + 1
            if len(by_index) != n_experts:
                built = {}
                break
            shapes = {tuple(weights[by_index[i]].shape) for i in range(n_experts)}
            if len(shapes) != 1:
                built = {}
                break
            if isinstance(weights[by_index[0]], QuantWeight):
                # Already packed by a streaming load: join the bytes directly.
                parts = [weights[by_index[i]] for i in range(n_experts)]
                built[field] = concat_quant_weights(parts, free=True)
                del parts
                for i in range(n_experts):
                    del weights[by_index[i]]
                continue
            first = weights[by_index[0]]
            block = torch.empty(
                (n_experts, *first.shape), dtype=first.dtype, device=first.device
            )
            del first
            for i in range(n_experts):
                block[i].copy_(weights[by_index[i]])
                del weights[by_index[i]]
            built[field] = block
        if not built or "up" not in built or "down" not in built:
            continue
        stacks[prefix] = built
        check_floor(f"stacking experts for {prefix}")
        stacked_bytes += sum(
            t.stored_bytes() if isinstance(t, QuantWeight) else t.numel() * t.element_size()
            for t in built.values()
        )

    if not stacks:
        return {"layers": 0, "bytes": 0}

    from engine.layers.moe import EXPERT_STACK_KEY

    # Anything left behind (a partial layer that failed the shape check) stays
    # dense for the Python dispatch path.
    weights[EXPERT_STACK_KEY] = stacks
    return {"layers": len(stacks), "bytes": stacked_bytes}


def quantization_report(weights: dict[str, object]) -> dict[str, float]:
    """Bytes actually resident vs the dense BF16 equivalent."""
    packed_bytes = 0
    dense_bytes = 0
    packed_params = 0
    total_params = 0
    seen: set[int] = set()
    values = list(weights.values())
    # Stacked MoE experts live one dict deep; on a sparse model they are most of
    # the checkpoint, so a report that skipped them would be meaningless.
    for value in list(values):
        if isinstance(value, dict):
            values.extend(inner for block in value.values() for inner in block.values())
    for value in values:
        if isinstance(value, QuantWeight):
            if value.qweight.data_ptr() in seen:
                continue
            seen.add(value.qweight.data_ptr())
            packed_bytes += value.stored_bytes()
            dense_bytes += value.numel() * 2
            packed_params += value.numel()
            total_params += value.numel()
        elif isinstance(value, torch.Tensor):
            if value.data_ptr() in seen:
                continue
            seen.add(value.data_ptr())
            packed_bytes += value.numel() * value.element_size()
            dense_bytes += value.numel() * value.element_size()
            total_params += value.numel()
    return {
        "resident_gb": packed_bytes / 1e9,
        "dense_gb": dense_bytes / 1e9,
        "compression": (dense_bytes / packed_bytes) if packed_bytes else 1.0,
        "packed_fraction": (packed_params / total_params) if total_params else 0.0,
    }
