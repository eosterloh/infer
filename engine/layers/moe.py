"""Mixture-of-Experts FFN (Nemotron-H style: sigmoid top-k + shared expert)."""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from engine.config import ModelConfig
from engine.kernels import act_and_mul, act_mul, moe_combine, moe_gemv, qmoe_gemv
from engine.layers.linear import dense

# Set by the loader when it stacks the per-expert tensors: "layers.N.moe" →
# {"up": [E, N, K], "down": ..., optional "gate"/"*_bias"}.
EXPERT_STACK_KEY = "_expert_stacks"


def expert_stack(weights: dict, p: str) -> dict[str, torch.Tensor] | None:
    stacks = weights.get(EXPERT_STACK_KEY)
    if not stacks:
        return None
    return stacks.get(f"{p}.moe")


def _relu2(x: torch.Tensor) -> torch.Tensor:
    return torch.square(F.relu(x))


def expert_mlp(
    x: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    act: str,
) -> torch.Tensor:
    """NemotronHMLP: down(act(up(x))). No SwiGLU gate."""
    h = dense(x, w_up)
    if act in {"relu2", "relu_squared", "squared_relu"}:
        h = _relu2(h)
    elif act == "silu":
        h = F.silu(h)
    else:
        raise ValueError(f"unsupported moe/mlp act {act!r}")
    return dense(h, w_down)


def route_topk(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    *,
    top_k: int,
    n_group: int,
    topk_group: int,
    norm_topk_prob: bool,
    routed_scaling_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match NemotronHTopkRouter: sigmoid scores + group mask + top-k.

    x: [N, H] flat tokens.
    Returns topk_indices [N, K], topk_weights [N, K].
    """
    n_routed = gate_weight.shape[0]
    # Router matmul kept in fp32 like the reference.
    logits = F.linear(x.float(), gate_weight.float())
    scores = logits.sigmoid()
    scores_for_choice = scores + e_score_correction_bias.float().unsqueeze(0)

    group_scores = (
        scores_for_choice.view(-1, n_group, n_routed // n_group)
        .topk(2, dim=-1)[0]
        .sum(dim=-1)
    )
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(-1, n_group, n_routed // n_group)
        .reshape(-1, n_routed)
    )
    scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
    topk_indices = torch.topk(scores_for_choice, k=top_k, dim=-1, sorted=False)[1]
    topk_weights = scores.gather(1, topk_indices)
    if norm_topk_prob:
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
    topk_weights = topk_weights * routed_scaling_factor
    return topk_indices, topk_weights


def expert_swiglu(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
) -> torch.Tensor:
    return dense(act_mul(dense(x, w_gate), dense(x, w_up), "silu"), w_down)


def softmax_topk(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    top_k: int,
    gate_bias: torch.Tensor | None = None,
    *,
    norm_topk_prob: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = F.linear(x.float(), gate_weight.float(), gate_bias.float() if gate_bias is not None else None)
    weights = torch.softmax(logits, dim=-1)
    topk_w, topk_i = torch.topk(weights, k=top_k, dim=-1)
    if norm_topk_prob:
        topk_w = topk_w / (topk_w.sum(dim=-1, keepdim=True) + 1e-20)
    return topk_i, topk_w


def _phimoe_sparsemixer(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    top_k: int,
    jitter: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PhiMoE eval-mode SparseMixer (no Gumbel)."""
    scores = F.linear(x.float(), gate_weight.float())
    indices = []
    weights = []
    remaining = scores
    for _ in range(top_k):
        with torch.no_grad():
            thresh, max_ind = remaining.max(dim=-1, keepdim=True)
            factor = remaining.abs().clamp(min=thresh)
            mask = ((thresh - remaining) / factor) > (2 * jitter)
        masked = remaining.masked_fill(mask, float("-inf"))
        selected = max_ind
        probs = torch.softmax(masked, dim=-1)
        w = probs.gather(1, selected)
        indices.append(selected)
        weights.append(w)
        remaining = remaining.scatter(-1, selected, float("-inf"))
    return torch.cat(indices, dim=-1), torch.cat(weights, dim=-1)


def _dispatch_experts(
    flat: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    n_routed: int,
    run_expert,
) -> torch.Tensor:
    """Reference dispatch: one pass per expert over the tokens it received.

    ``torch.where`` has to know how many tokens matched, so every expert costs a
    device synchronization. Stacked expert weights take `fused_dispatch`
    instead; this stays the correctness reference and the CPU path.
    """
    routed = torch.zeros_like(flat, dtype=topk_weights.dtype)
    expert_mask = F.one_hot(topk_indices, num_classes=n_routed).permute(2, 0, 1)
    for expert_idx in range(n_routed):
        mask = expert_mask[expert_idx]
        token_indices, weight_indices = torch.where(mask)
        if token_indices.numel() == 0:
            continue
        expert_out = run_expert(expert_idx, flat[token_indices])
        expert_out = expert_out * topk_weights[token_indices, weight_indices].unsqueeze(-1)
        routed.index_add_(0, token_indices, expert_out.to(routed.dtype))
    return routed


def _routing_rows(
    tokens: int, topk_indices: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten [T, topk] routing into per-output-row expert / token indices."""
    topk = topk_indices.shape[-1]
    row_expert = topk_indices.reshape(-1).to(torch.int32)
    row_input = torch.arange(
        tokens, device=topk_indices.device, dtype=torch.int32
    ).repeat_interleave(topk)
    return row_expert, row_input


def _expert_from_stack(stack: dict, idx: int) -> dict:
    """One expert's weights as views into the stacked block."""
    out = {}
    for field in ("gate", "up", "down"):
        block = stack.get(field)
        if block is None:
            continue
        out[field] = (
            block.expert_view(idx) if hasattr(block, "expert_view") else block[idx]
        )
    return out


def _stack_gemv(
    x: torch.Tensor,
    w,
    row_expert: torch.Tensor,
    row_input: torch.Tensor | None,
    bias: torch.Tensor | None,
) -> torch.Tensor | None:
    """Routed GEMV against one expert block, packed or dense."""
    if getattr(w, "expert_cols", 0):
        out = qmoe_gemv(x, w, row_expert, row_input)
        if out is None:
            return None
        if bias is not None:
            out = out + bias.index_select(0, row_expert.to(torch.long))
        return out
    return moe_gemv(x, w, row_expert, row_input, bias)


def fused_worth_it(tokens: int, topk: int, n_routed: int) -> bool:
    """Whether the grouped GEMV beats a per-expert GEMM for this many tokens.

    The grouped kernel gives every routed row its own warp, and that warp reads
    the whole expert row it needs. Two rows on one expert therefore read that
    expert twice. Decode has one row per expert at most, so it wins outright —
    no host round trip, no synchronization per expert, and it is the only form
    a CUDA graph can capture. A wide prefill is the opposite case: dozens of
    rows land on each expert, and reading the weights once into a cuBLAS tile
    beats reading them dozens of times.
    """
    if tokens <= 1:
        return True
    try:
        limit = float(os.environ.get("INFER_MOE_FUSED_ROWS_PER_EXPERT", "2"))
    except ValueError:
        limit = 2.0
    if n_routed <= 0:
        return True
    return (tokens * max(topk, 1)) <= limit * n_routed


def fused_dispatch(
    flat: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    stack: dict[str, torch.Tensor],
    act: str,
) -> torch.Tensor | None:
    """One launch per projection over the routed rows, or None if unsupported.

    ``stack`` holds the experts as contiguous ``[E, N, K]`` tensors. Every token
    in the batch is handled in the same launches, so cost tracks the tokens
    actually routed instead of the expert count.
    """
    down = stack.get("down")
    gate_up = stack.get("gate_up")
    up = stack.get("up")
    if down is None or (up is None and gate_up is None):
        return None
    gate = stack.get("gate")
    tokens = flat.shape[0]
    topk = int(topk_indices.shape[-1])
    row_expert, row_input = _routing_rows(tokens, topk_indices)
    flat_c = flat.contiguous()

    if gate_up is not None:
        # One block holds gate rows then up rows, so a single GEMV feeds the
        # fused activation directly.
        fused_h = _stack_gemv(flat_c, gate_up, row_expert, row_input, stack.get("gate_up_bias"))
        if fused_h is None:
            return None
        h = act_and_mul(fused_h, "silu" if act in {"silu", "swiglu"} else act)
        out = _stack_gemv(h.contiguous(), down, row_expert, None, stack.get("down_bias"))
        if out is None:
            return None
        return moe_combine(out, topk_weights, topk)

    h = _stack_gemv(flat_c, up, row_expert, row_input, stack.get("up_bias"))
    if h is None:
        return None
    if gate is not None:
        g = _stack_gemv(flat_c, gate, row_expert, row_input, stack.get("gate_bias"))
        if g is None:
            return None
        h = act_mul(g, h, "silu" if act in {"silu", "swiglu"} else act)
    elif act in {"relu2", "relu_squared", "squared_relu"}:
        h = _relu2(h)
    elif act == "silu":
        h = F.silu(h)
    else:
        return None

    out = _stack_gemv(h.contiguous(), down, row_expert, None, stack.get("down_bias"))
    if out is None:
        return None
    return moe_combine(out, topk_weights, topk)


def moe(
    x: torch.Tensor,
    weights: dict[str, torch.Tensor],
    layer: int,
    config: ModelConfig,
) -> torch.Tensor:
    """MoE mixer — Nemotron / Latent / Mixtral / Llama4 / GPT-OSS / DeepSeek."""
    kind = config.moe_kind
    p = f"layers.{layer}"
    n_routed = config.n_routed_experts
    top_k = config.num_experts_per_tok
    if n_routed is None or top_k is None:
        raise ValueError("MoE config fields missing (n_routed_experts / num_experts_per_tok)")

    orig_shape = x.shape
    flat = x.reshape(-1, x.shape[-1])
    residuals = x

    if kind in {"nemotron", "latent"}:
        return _moe_nemotron(x, weights, layer, config, latent=(kind == "latent"))

    gate_w = weights[f"{p}.moe.gate.weight"]
    gate_b = weights.get(f"{p}.moe.gate.bias")
    norm_topk = bool(getattr(config, "norm_topk_prob", True))
    logits_then_softmax = kind == "gpt_oss" or config.recipe_id in {
        "granitemoe",
        "granitemoe_swa",
        "granitemoeshared",
        "cohere2_moe",
    }
    if config.recipe_id == "phimoe":
        topk_i, topk_w = _phimoe_sparsemixer(
            flat, gate_w, top_k, jitter=float((config.raw or {}).get("router_jitter_noise") or 0.01)
        )
    elif kind == "deepseek" and (
        config.recipe_id in {"glm4_moe", "exaone_moe"}
        or (
            config.recipe_id == "deepseek_v3"
            and f"{p}.moe.gate.e_score_correction_bias" in weights
        )
    ):
        n_group = int((config.raw or {}).get("n_group", 1) or 1)
        topk_group = int((config.raw or {}).get("topk_group", 1) or 1)
        scale = float(config.routed_scaling_factor or 1.0)
        bias = weights.get(f"{p}.moe.gate.e_score_correction_bias")
        if bias is None:
            bias = torch.zeros(n_routed, device=flat.device, dtype=torch.float32)
        topk_i, topk_w = route_topk(
            flat,
            gate_w,
            bias.reshape(-1),
            top_k=top_k,
            n_group=n_group,
            topk_group=topk_group,
            norm_topk_prob=norm_topk,
            routed_scaling_factor=scale,
        )
    elif config.recipe_id == "ernie4_5_moe":
        logits = F.linear(flat.float(), gate_w.float())
        probs = torch.softmax(logits, dim=-1)
        choice = probs
        bias = weights.get(f"{p}.moe.gate.e_score_correction_bias")
        if bias is not None:
            choice = choice + bias.reshape(-1).float()
        topk_w, topk_i = torch.topk(choice, k=top_k, dim=-1)
        topk_w = probs.gather(1, topk_i)
        min_norm = float((config.raw or {}).get("moe_norm_min") or 1e-12)
        topk_w = topk_w / torch.clamp(topk_w.sum(dim=-1, keepdim=True), min=min_norm)
    elif config.recipe_id == "dbrx":
        logits = F.linear(flat.float(), gate_w.float())
        scores = torch.softmax(logits, dim=-1)
        topk_w, topk_i = torch.topk(scores, k=top_k, dim=-1)
        p_norm = (config.raw or {}).get("moe_normalize_expert_weights", 1.0)
        if p_norm is not None:
            topk_w = topk_w / torch.norm(topk_w, p=float(p_norm), dim=-1, keepdim=True)
    elif logits_then_softmax:
        logits = F.linear(flat.float(), gate_w.float(), gate_b.float() if gate_b is not None else None)
        topk_v, topk_i = torch.topk(logits, k=top_k, dim=-1)
        sel = str((config.raw or {}).get("expert_selection_fn") or "softmax")
        if config.recipe_id == "cohere2_moe" and sel == "sigmoid":
            topk_w = torch.sigmoid(topk_v)
            if norm_topk:
                topk_w = topk_w / (topk_w.sum(dim=-1, keepdim=True) + 1e-20)
        else:
            topk_w = torch.softmax(topk_v, dim=-1)
    else:
        topk_i, topk_w = softmax_topk(flat, gate_w, top_k, gate_b, norm_topk_prob=norm_topk)

    stack = expert_stack(weights, p)
    packed = f"{p}.moe.experts.gate_up.weight" in weights or (
        stack is not None and "gate_up" in stack
    )
    if stack is not None and fused_worth_it(flat.shape[0], top_k, n_routed):
        fused = fused_dispatch(
            flat, topk_i, topk_w, stack, config.mlp_hidden_act or "silu"
        )
        if fused is not None:
            routed = fused
            return _add_shared(
                routed, x, residuals, orig_shape, weights, p, config
            )
    if kind == "dbrx" or f"{p}.moe.experts.w1.weight" in weights:
        routed = _dbrx_experts(flat, topk_i, topk_w, weights, p, n_routed)
    elif kind in {"gpt_oss", "llama4"} or packed:
        routed = _packed_experts(
            flat, topk_i, topk_w, weights, p, n_routed, kind, stack=stack
        )
    else:

        def run(idx: int, tok: torch.Tensor) -> torch.Tensor:
            # Stacking removes the per-expert entries, so when the fused path
            # declines the loop reads slices of the block instead.
            expert = (
                _expert_from_stack(stack, idx)
                if stack is not None
                else {
                    field: weights[f"{p}.moe.experts.{idx}.{field}.weight"]
                    for field in ("gate", "up", "down")
                    if f"{p}.moe.experts.{idx}.{field}.weight" in weights
                }
            )
            if "gate" in expert:
                return expert_swiglu(tok, expert["gate"], expert["up"], expert["down"])
            return expert_mlp(
                tok, expert["up"], expert["down"], config.mlp_hidden_act or "silu"
            )

        routed = _dispatch_experts(flat, topk_i, topk_w, n_routed, run)

    return _add_shared(routed, x, residuals, orig_shape, weights, p, config)


def _add_shared(
    routed: torch.Tensor,
    x: torch.Tensor,
    residuals: torch.Tensor,
    orig_shape: torch.Size,
    weights: dict[str, torch.Tensor],
    p: str,
    config: ModelConfig,
) -> torch.Tensor:
    """Add the always-on shared expert (and its gate) to the routed output."""
    routed = routed.view(*orig_shape).to(dtype=x.dtype)
    if f"{p}.moe.shared.up.weight" in weights or f"{p}.moe.shared.gate_up.weight" in weights:
        if f"{p}.moe.shared.gate_up.weight" in weights:
            gate, up = weights[f"{p}.moe.shared.gate_up.weight"].chunk(2, dim=0)
            shared = expert_swiglu(
                residuals,
                gate,
                up,
                weights[f"{p}.moe.shared.down.weight"],
            )
        elif f"{p}.moe.shared.gate.weight" in weights:
            shared = expert_swiglu(
                residuals,
                weights[f"{p}.moe.shared.gate.weight"],
                weights[f"{p}.moe.shared.up.weight"],
                weights[f"{p}.moe.shared.down.weight"],
            )
        else:
            shared = expert_mlp(
                residuals,
                weights[f"{p}.moe.shared.up.weight"],
                weights[f"{p}.moe.shared.down.weight"],
                config.mlp_hidden_act or "silu",
            )
        gate_w = weights.get(f"{p}.moe.shared_gate.weight")
        if gate_w is not None:
            shared = shared * torch.sigmoid(F.linear(residuals, gate_w))
        combo = str((config.raw or {}).get("shared_expert_combination_strategy") or "sum")
        if config.recipe_id == "cohere2_moe" and combo == "average":
            return (routed + shared) / 2
        return routed + shared
    return routed


def _dbrx_expert_split(
    fused: torch.Tensor, n_routed: int, hidden: int
) -> torch.Tensor:
    """One DBRX projection → [E, inter, hidden] blocks.

    DBRX keeps every expert of a projection in a single tensor. The chunk an
    expert owns is stored either [inter, hidden] or [hidden, inter], and the
    hidden size is the only thing that separates them, so the row count cannot
    stand in for the intermediate size.
    """
    chunks = fused.reshape(n_routed, -1)
    inter = chunks.shape[-1] // hidden
    stored_rows = fused.shape[0] // n_routed if fused.dim() == 2 else inter
    if stored_rows == hidden and inter != hidden:
        return chunks.reshape(n_routed, hidden, inter).transpose(1, 2)
    return chunks.reshape(n_routed, inter, hidden)


def _dbrx_experts(
    flat: torch.Tensor,
    topk_i: torch.Tensor,
    topk_w: torch.Tensor,
    weights: dict[str, torch.Tensor],
    p: str,
    n_routed: int,
) -> torch.Tensor:
    hidden = flat.shape[-1]
    w1 = _dbrx_expert_split(weights[f"{p}.moe.experts.w1.weight"], n_routed, hidden)
    v1 = _dbrx_expert_split(weights[f"{p}.moe.experts.v1.weight"], n_routed, hidden)
    w2 = _dbrx_expert_split(weights[f"{p}.moe.experts.w2.weight"], n_routed, hidden)

    def run(idx: int, tok: torch.Tensor) -> torch.Tensor:
        gate = F.silu(F.linear(tok, w1[idx]))
        up = F.linear(tok, v1[idx])
        return (gate * up) @ w2[idx]

    return _dispatch_experts(flat, topk_i, topk_w, n_routed, run)


def _packed_experts(
    flat: torch.Tensor,
    topk_i: torch.Tensor,
    topk_w: torch.Tensor,
    weights: dict[str, torch.Tensor],
    p: str,
    n_routed: int,
    kind: str,
    stack: dict | None = None,
) -> torch.Tensor:
    # llama4/gpt_oss store [E, H, 2I] / [E, I, H] (token @ weight).
    # Qwen2/3-MoE store Linear layouts [E, 2I, H] / [E, H, I]. Adoption into a
    # stack normalizes both to the Linear layout.
    transposed = kind in {"gpt_oss", "llama4"} and stack is None
    if stack is not None:
        gate_up, down = stack["gate_up"], stack["down"]
        gu_bias, dn_bias = stack.get("gate_up_bias"), stack.get("down_bias")
    else:
        gate_up = weights[f"{p}.moe.experts.gate_up.weight"]
        down = weights[f"{p}.moe.experts.down.weight"]
        gu_bias = weights.get(f"{p}.moe.experts.gate_up.bias")
        dn_bias = weights.get(f"{p}.moe.experts.down.bias")

    def slice_expert(block, idx: int):
        return block.expert_view(idx) if hasattr(block, "expert_view") else block[idx]

    def run(idx: int, tok: torch.Tensor) -> torch.Tensor:
        w_gu = slice_expert(gate_up, idx)
        fused = tok @ w_gu if transposed else dense(tok, w_gu)
        if gu_bias is not None:
            fused = fused + gu_bias[idx]
        gate, up = fused.chunk(2, dim=-1)
        h = F.silu(gate) * up
        w_dn = slice_expert(down, idx)
        out = h @ w_dn if transposed else dense(h, w_dn)
        if dn_bias is not None:
            out = out + dn_bias[idx]
        return out

    return _dispatch_experts(flat, topk_i, topk_w, n_routed, run)


def _moe_nemotron(
    x: torch.Tensor,
    weights: dict[str, torch.Tensor],
    layer: int,
    config: ModelConfig,
    *,
    latent: bool,
) -> torch.Tensor:
    if config.n_routed_experts is None or config.num_experts_per_tok is None:
        raise ValueError("MoE config fields missing (n_routed_experts / num_experts_per_tok)")

    p = f"layers.{layer}"
    act = config.mlp_hidden_act or "relu2"
    n_routed = config.n_routed_experts
    top_k = config.num_experts_per_tok
    n_group = int(config.raw.get("n_group", 1)) if config.raw else 1
    topk_group = int(config.raw.get("topk_group", 1)) if config.raw else 1
    norm_topk = bool(config.raw.get("norm_topk_prob", True)) if config.raw else True
    scale = float(config.routed_scaling_factor or 1.0)

    orig_shape = x.shape
    flat = x.reshape(-1, x.shape[-1])
    residuals = x

    topk_indices, topk_weights = route_topk(
        flat,
        weights[f"{p}.moe.gate.weight"],
        weights[f"{p}.moe.gate.e_score_correction_bias"],
        top_k=top_k,
        n_group=n_group,
        topk_group=topk_group,
        norm_topk_prob=norm_topk,
        routed_scaling_factor=scale,
    )

    expert_in = flat
    if latent:
        expert_in = dense(flat, weights[f"{p}.moe.latent_down.weight"])

    stack = expert_stack(weights, p)

    def run(idx: int, tok: torch.Tensor) -> torch.Tensor:
        expert = (
            _expert_from_stack(stack, idx)
            if stack is not None
            else {
                "up": weights[f"{p}.moe.experts.{idx}.up.weight"],
                "down": weights[f"{p}.moe.experts.{idx}.down.weight"],
            }
        )
        return expert_mlp(tok, expert["up"], expert["down"], act)

    routed = None
    if stack is not None and fused_worth_it(expert_in.shape[0], top_k, n_routed):
        routed = fused_dispatch(expert_in, topk_indices, topk_weights, stack, act)
    if routed is None:
        routed = _dispatch_experts(expert_in, topk_indices, topk_weights, n_routed, run)
    # Back to the activation dtype before the latent projection, not after: the
    # combine multiplies by fp32 router weights, so `routed` comes back fp32 and
    # latent_up is a BF16 weight. Casting afterwards left this path throwing on
    # any GPU checkpoint, which an fp32 CPU run cannot show.
    routed = routed.to(dtype=x.dtype)
    if latent:
        routed = dense(routed, weights[f"{p}.moe.latent_up.weight"])

    routed = routed.view(*orig_shape)
    shared = expert_mlp(
        residuals,
        weights[f"{p}.moe.shared.up.weight"],
        weights[f"{p}.moe.shared.down.weight"],
        act,
    )
    return routed + shared
