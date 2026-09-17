"""Token sampling: greedy, temperature, top-k, nucleus (top-p)."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SamplingParams:
    """Decode knobs. temperature <= 0 is greedy argmax (top-k / top-p ignored)."""

    temperature: float = 0.0
    top_k: int | None = None
    top_p: float | None = None
    seed: int | None = None

    @property
    def greedy(self) -> bool:
        return self.temperature is None or float(self.temperature) <= 0.0


def make_generator(seed: int | None, device: torch.device | str | None = None) -> torch.Generator | None:
    """CPU generator; torch.multinomial accepts it for CPU and CUDA logits."""
    del device
    if seed is None:
        return None
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    return gen


def _apply_top_k(logits: torch.Tensor, top_k: int | None) -> torch.Tensor:
    if top_k is None or int(top_k) <= 0:
        return logits
    k = min(int(top_k), logits.shape[-1])
    cutoff = torch.topk(logits, k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < cutoff, float("-inf"))


def _apply_top_p(logits: torch.Tensor, top_p: float | None) -> torch.Tensor:
    if top_p is None or float(top_p) >= 1.0:
        return logits
    p = float(top_p)
    if p <= 0.0:
        keep = torch.argmax(logits, dim=-1, keepdim=True)
        masked = torch.full_like(logits, float("-inf"))
        return masked.scatter(-1, keep, logits.gather(-1, keep))
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    cum = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
    remove = cum > p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    return torch.empty_like(logits).scatter(-1, sorted_idx, sorted_logits)


def select_next_id(
    logits: torch.Tensor,
    params: SamplingParams | None = None,
    generator: torch.Generator | None = None,
) -> int:
    """Pick one token from a 1-D logit vector."""
    if logits.dim() != 1:
        raise ValueError(f"expected 1-D logits, got {tuple(logits.shape)}")
    params = params or SamplingParams()
    row = logits.float()
    if params.greedy:
        return int(torch.argmax(row).item())
    scaled = row / float(params.temperature)
    filtered = _apply_top_p(_apply_top_k(scaled, params.top_k), params.top_p)
    if not torch.isfinite(filtered).any():
        return int(torch.argmax(row).item())
    probs = torch.softmax(filtered, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0)
    if float(probs.sum()) <= 0.0:
        return int(torch.argmax(row).item())
    pick = torch.multinomial(probs, 1, generator=generator)
    return int(pick.item())
