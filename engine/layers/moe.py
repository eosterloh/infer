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


def moe(
    x: torch.Tensor,
    weights: dict[str, torch.Tensor],
    layer: int,
    config: ModelConfig,
) -> torch.Tensor:
    """Nemotron-H MoE mixer: routed experts + always-on shared expert."""
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

    routed = torch.zeros_like(flat, dtype=topk_weights.dtype)
    expert_mask = F.one_hot(topk_indices, num_classes=n_routed).permute(2, 0, 1)

    for expert_idx in range(n_routed):
        mask = expert_mask[expert_idx]
        token_indices, weight_indices = torch.where(mask)
        w_up = weights[f"{p}.moe.experts.{expert_idx}.up.weight"]
        w_down = weights[f"{p}.moe.experts.{expert_idx}.down.weight"]
        if token_indices.numel() > 0:
            expert_in = flat[token_indices]
            expert_out = expert_mlp(expert_in, w_up, w_down, act)
            expert_out = expert_out * topk_weights[token_indices, weight_indices].unsqueeze(
                -1
            )
            routed.index_add_(0, token_indices, expert_out.to(routed.dtype))
        else:
            # Keep unused experts in the autograd graph for training parity;
            # inference no-op still touches the weights.
            dummy = expert_mlp(
                torch.zeros(1, flat.shape[-1], device=flat.device, dtype=flat.dtype),
                w_up,
                w_down,
                act,
            )
            routed = routed + dummy.to(routed.dtype) * 0

    routed = routed.view(*orig_shape).to(dtype=x.dtype)
    shared = expert_mlp(
        residuals,
        weights[f"{p}.moe.shared.up.weight"],
        weights[f"{p}.moe.shared.down.weight"],
        act,
    )
    return routed + shared
