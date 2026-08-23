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
        return {int(eos_token_id)}
    return {int(x) for x in eos_token_id}


def _delta_piece(tokenizer: Tokenizer, gen_ids: list[int], prev: str) -> tuple[str, str]:
    """Decode the whole generated span so BPE fragments reassemble correctly."""
    text = tokenizer.decode(gen_ids, skip_special_tokens=False)
    if text.startswith(prev):
        return text, text[len(prev) :]
    return text, text if not prev else text


@torch.inference_mode()
def generate_greedy(
    model: DecoderModel,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 32,
    add_special_tokens: bool = True,
    use_cache: bool = True,
    apply_chat_template: bool | None = None,
    enable_thinking: bool = False,
) -> Iterator[str]:
    """Yield decoded text pieces (greedy / argmax).

    If the folder shipped a chat template and `prompt` is a raw user string,
    wrap it automatically. Pass apply_chat_template=False to send raw tokens.
    """
    text, templated = tokenizer.format_for_generate(
        prompt,
        apply_chat_template=apply_chat_template,
        enable_thinking=enable_thinking,
    )
    add_bos = add_special_tokens and not templated
    ids = tokenizer.encode(text, add_special_tokens=add_bos)
    if not ids:
        raise ValueError("prompt encoded to empty token list")

    device = model.device
    tokens = torch.tensor([ids], dtype=torch.long, device=device)
    eos_ids = _eos_id_set(model.config.eos_token_id)
    gen_ids: list[int] = []
    prev = ""

    def _emit(next_id: int) -> str:
        nonlocal prev
        gen_ids.append(next_id)
        prev, piece = _delta_piece(tokenizer, gen_ids, prev)
        return piece

    if not use_cache:
        for _ in range(max_new_tokens):
            logits = model.forward(tokens, cache=None)
            next_id = int(torch.argmax(logits[0, -1, :]).item())
            tokens = torch.cat(
                [tokens, torch.tensor([[next_id]], dtype=torch.long, device=device)],
                dim=1,
            )
            if next_id in eos_ids:
                break
            yield _emit(next_id)
        return

    cache = model.make_cache(batch_size=1, device=device, dtype=model.dtype)

    logits = model.forward(tokens, cache=cache)
    next_id = int(torch.argmax(logits[0, -1, :]).item())
    if next_id in eos_ids:
        return
    yield _emit(next_id)

    for _ in range(max_new_tokens - 1):
        step = torch.tensor([[next_id]], dtype=torch.long, device=device)
        logits = model.forward(step, cache=cache)
        next_id = int(torch.argmax(logits[0, -1, :]).item())
        if next_id in eos_ids:
            break
        yield _emit(next_id)


def generate(
    model: DecoderModel,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 32,
    use_cache: bool = True,
    apply_chat_template: bool | None = None,
    enable_thinking: bool = False,
    **_knobs,
) -> Iterator[str]:
    yield from generate_greedy(
        model,
        tokenizer,
        prompt,
        max_new_tokens=max_new_tokens,
        use_cache=use_cache,
        apply_chat_template=apply_chat_template,
        enable_thinking=enable_thinking,
    )
