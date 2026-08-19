"""Greedy generation — KV / hybrid RuntimeState; optional full recompute."""

from __future__ import annotations

from collections.abc import Iterator

import torch

from engine.model import DecoderModel
from engine.tokenizer import Tokenizer


def _eos_id_set(eos_token_id: int | list[int] | None) -> set[int]:
    if eos_token_id is None:
        return set()
    if isinstance(eos_token_id, int):
        return {eos_token_id}
    return set(int(x) for x in eos_token_id)


@torch.inference_mode()
def generate_greedy(
    model: DecoderModel,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 32,
    add_special_tokens: bool = True,
    use_cache: bool = True,
) -> Iterator[str]:
    """Yield decoded text pieces (greedy / argmax)."""
    ids = tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
    if not ids:
        raise ValueError("prompt encoded to empty token list")

    device = model.device
    tokens = torch.tensor([ids], dtype=torch.long, device=device)
    eos_ids = _eos_id_set(model.config.eos_token_id)

    if not use_cache:
        for _ in range(max_new_tokens):
            logits = model.forward(tokens, cache=None)
            next_id = int(torch.argmax(logits[0, -1, :]).item())
            tokens = torch.cat(
                [tokens, torch.tensor([[next_id]], dtype=torch.long, device=device)],
                dim=1,
            )
            yield tokenizer.decode([next_id], skip_special_tokens=False)
            if next_id in eos_ids:
                break
        return

    cache = model.make_cache(batch_size=1, device=device, dtype=model.dtype)

    logits = model.forward(tokens, cache=cache)
    next_id = int(torch.argmax(logits[0, -1, :]).item())
    yield tokenizer.decode([next_id], skip_special_tokens=False)
    if next_id in eos_ids:
        return

    for _ in range(max_new_tokens - 1):
        step = torch.tensor([[next_id]], dtype=torch.long, device=device)
        logits = model.forward(step, cache=cache)
        next_id = int(torch.argmax(logits[0, -1, :]).item())
        yield tokenizer.decode([next_id], skip_special_tokens=False)
        if next_id in eos_ids:
            break


def generate(
    model: DecoderModel,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 32,
    use_cache: bool = True,
    **_knobs,
) -> Iterator[str]:
    yield from generate_greedy(
        model,
        tokenizer,
        prompt,
        max_new_tokens=max_new_tokens,
        use_cache=use_cache,
    )
