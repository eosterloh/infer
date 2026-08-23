"""Mixture-of-Experts FFN (Nemotron-H style: sigmoid top-k + shared expert)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from engine.config import ModelConfig


def _relu2(x: torch.Tensor) -> torch.Tensor:
    return torch.square(F.relu(x))


def expert_mlp(
    x: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    act: str,
) -> torch.Tensor:
    """NemotronHMLP: down(act(up(x))). No SwiGLU gate."""
    h = F.linear(x, w_up)
    if act in {"relu2", "relu_squared", "squared_relu"}:
        h = _relu2(h)
    elif act == "silu":
        h = F.silu(h)
    else:
        raise ValueError(f"unsupported moe/mlp act {act!r}")
    return F.linear(h, w_down)


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
    scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)
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
    return F.linear(F.silu(F.linear(x, w_gate)) * F.linear(x, w_up), w_down)


def softmax_topk(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    top_k: int,
    gate_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = F.linear(x.float(), gate_weight.float(), gate_bias.float() if gate_bias is not None else None)
    weights = torch.softmax(logits, dim=-1)
    topk_w, topk_i = torch.topk(weights, k=top_k, dim=-1)
    topk_w = topk_w / (topk_w.sum(dim=-1, keepdim=True) + 1e-20)
    return topk_i, topk_w


def _dispatch_experts(
    flat: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    n_routed: int,
    run_expert,
) -> torch.Tensor:
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
    if kind == "gpt_oss":
        logits = F.linear(flat.float(), gate_w.float(), gate_b.float() if gate_b is not None else None)
        topk_v, topk_i = torch.topk(logits, k=top_k, dim=-1)
        topk_w = torch.softmax(topk_v, dim=-1)
    else:
        topk_i, topk_w = softmax_topk(flat, gate_w, top_k, gate_b)

    if kind in {"gpt_oss", "llama4"}:
        routed = _packed_experts(flat, topk_i, topk_w, weights, p, n_routed, kind)
    else:
        act_gate = True

        def run(idx: int, tok: torch.Tensor) -> torch.Tensor:
            if f"{p}.moe.experts.{idx}.gate.weight" in weights:
                return expert_swiglu(
                    tok,
                    weights[f"{p}.moe.experts.{idx}.gate.weight"],
                    weights[f"{p}.moe.experts.{idx}.up.weight"],
                    weights[f"{p}.moe.experts.{idx}.down.weight"],
                )
            return expert_mlp(
                tok,
                weights[f"{p}.moe.experts.{idx}.up.weight"],
                weights[f"{p}.moe.experts.{idx}.down.weight"],
                config.mlp_hidden_act or "silu",
            )

        routed = _dispatch_experts(flat, topk_i, topk_w, n_routed, run)

    routed = routed.view(*orig_shape).to(dtype=x.dtype)
    if f"{p}.moe.shared.up.weight" in weights:
        if f"{p}.moe.shared.gate.weight" in weights:
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
        return routed + shared
    return routed


def _packed_experts(
    flat: torch.Tensor,
    topk_i: torch.Tensor,
    topk_w: torch.Tensor,
    weights: dict[str, torch.Tensor],
    p: str,
    n_routed: int,
    kind: str,
) -> torch.Tensor:
    gate_up = weights[f"{p}.moe.experts.gate_up.weight"]
    down = weights[f"{p}.moe.experts.down.weight"]
    gu_bias = weights.get(f"{p}.moe.experts.gate_up.bias")
    dn_bias = weights.get(f"{p}.moe.experts.down.bias")

    def run(idx: int, tok: torch.Tensor) -> torch.Tensor:
        # llama4/gpt_oss: gate_up [E, H, 2I] (transposed vs Linear)
        w_gu = gate_up[idx]
        if w_gu.shape[0] == tok.shape[-1]:
            fused = tok @ w_gu
        else:
            fused = F.linear(tok, w_gu)
        if gu_bias is not None:
            fused = fused + gu_bias[idx]
        gate, up = fused.chunk(2, dim=-1)
        h = F.silu(gate) * up
        w_dn = down[idx]
        if w_dn.shape[0] == h.shape[-1]:
            out = h @ w_dn
        else:
            out = F.linear(h, w_dn)
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
        expert_in = F.linear(flat, weights[f"{p}.moe.latent_down.weight"])

    def run(idx: int, tok: torch.Tensor) -> torch.Tensor:
        return expert_mlp(
            tok,
            weights[f"{p}.moe.experts.{idx}.up.weight"],
            weights[f"{p}.moe.experts.{idx}.down.weight"],
            act,
        )

    routed = _dispatch_experts(expert_in, topk_indices, topk_weights, n_routed, run)
    if latent:
        routed = F.linear(routed, weights[f"{p}.moe.latent_up.weight"])

    routed = routed.view(*orig_shape).to(dtype=x.dtype)
    shared = expert_mlp(
        residuals,
        weights[f"{p}.moe.shared.up.weight"],
        weights[f"{p}.moe.shared.down.weight"],
        act,
    )
    return routed + shared
