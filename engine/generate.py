"""Greedy generation — KV / hybrid RuntimeState; optional full recompute."""

from __future__ import annotations

from collections.abc import Iterator

import torch

from engine.model import DecoderModel
from engine.mtp import Qwen35MTP
from engine.tokenizer import Tokenizer
from engine.vision import qwen35_multimodal_embeddings


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


@torch.inference_mode()
def generate_multimodal_greedy(
    model: DecoderModel,
    tokenizer: Tokenizer,
    vision_weights: dict[str, torch.Tensor],
    processor_inputs: dict[str, torch.Tensor],
    *,
    max_new_tokens: int = 32,
) -> Iterator[str]:
    """Greedy Qwen image/video generation from official processor tensors."""
    required = {"input_ids", "mm_token_type_ids"}
    missing = required - set(processor_inputs)
    if missing:
        raise ValueError(f"processor inputs missing {sorted(missing)}")

    device = model.device
    batch = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in processor_inputs.items()
    }
    input_ids = batch["input_ids"]
    if input_ids.shape[0] != 1:
        raise ValueError("multimodal greedy currently supports batch_size=1")

    embeds = model.weights["embed.weight"][input_ids]
    embeds, position_ids, rope_delta = qwen35_multimodal_embeddings(
        input_ids,
        embeds,
        vision_weights,
        model.config.raw,
        pixel_values=batch.get("pixel_values"),
        pixel_values_videos=batch.get("pixel_values_videos"),
        image_grid_thw=batch.get("image_grid_thw"),
        video_grid_thw=batch.get("video_grid_thw"),
        mm_token_type_ids=batch["mm_token_type_ids"],
        attention_mask=batch.get("attention_mask"),
    )
    cache = model.make_cache(batch_size=1, device=device, dtype=model.dtype)
    logits = model.forward(
        cache=cache, inputs_embeds=embeds, position_ids=position_ids
    )
    assert isinstance(logits, torch.Tensor)
    next_id = int(torch.argmax(logits[0, -1]).item())
    eos_ids = _eos_id_set(model.config.eos_token_id)
    generated: list[int] = []
    previous = ""

    for i in range(max_new_tokens):
        if next_id in eos_ids:
            break
        previous, piece = _delta_piece(tokenizer, generated + [next_id], previous)
        generated.append(next_id)
        yield piece
        if i + 1 == max_new_tokens:
            break

        step = torch.tensor([[next_id]], dtype=torch.long, device=device)
        position = torch.arange(
            cache.seq_len(), cache.seq_len() + 1, device=device, dtype=torch.long
        ).view(1, 1, 1)
        position = position.expand(3, 1, 1) + rope_delta.to(device).view(1, 1, 1)
        logits = model.forward(step, cache=cache, position_ids=position)
        assert isinstance(logits, torch.Tensor)
        next_id = int(torch.argmax(logits[0, -1]).item())


@torch.inference_mode()
def generate_mtp_greedy(
    model: DecoderModel,
    mtp: Qwen35MTP,
    tokenizer: Tokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 32,
    num_speculative_tokens: int = 3,
    apply_chat_template: bool | None = None,
    enable_thinking: bool = False,
    stats: dict[str, int] | None = None,
    processor_inputs: dict[str, torch.Tensor] | None = None,
    vision_weights: dict[str, torch.Tensor] | None = None,
) -> Iterator[str]:
    """Lossless greedy native-MTP decode with batched target verification."""
    if num_speculative_tokens < 1:
        raise ValueError("num_speculative_tokens must be >= 1")
    if stats is not None:
        stats.update(rounds=0, drafted=0, accepted=0)
    if max_new_tokens <= 0:
        return

    device = model.device
    rope_delta: torch.Tensor | None = None
    prime_embeddings: torch.Tensor | None = None
    if processor_inputs is None:
        text, templated = tokenizer.format_for_generate(
            prompt,
            apply_chat_template=apply_chat_template,
            enable_thinking=enable_thinking,
        )
        ids = tokenizer.encode(text, add_special_tokens=not templated)
        if not ids:
            return
        tokens = torch.tensor([ids], dtype=torch.long, device=device)
        prefill_embeddings = None
        prefill_positions = torch.arange(
            tokens.shape[1], device=device, dtype=torch.long
        )[None]
        prefill_mask = None
    else:
        if vision_weights is None:
            raise ValueError("vision_weights required with multimodal processor inputs")
        batch = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in processor_inputs.items()
        }
        tokens = batch["input_ids"]
        if tokens.shape[0] != 1:
            raise ValueError("multimodal MTP currently supports batch_size=1")
        prefill_embeddings = model.weights["embed.weight"][tokens]
        prefill_embeddings, prefill_positions, rope_delta = (
            qwen35_multimodal_embeddings(
                tokens,
                prefill_embeddings,
                vision_weights,
                model.config.raw,
                pixel_values=batch.get("pixel_values"),
                pixel_values_videos=batch.get("pixel_values_videos"),
                image_grid_thw=batch.get("image_grid_thw"),
                video_grid_thw=batch.get("video_grid_thw"),
                mm_token_type_ids=batch["mm_token_type_ids"],
                attention_mask=batch.get("attention_mask"),
            )
        )
        prime_embeddings = prefill_embeddings[:, 1:]
        prefill_mask = batch.get("attention_mask")

    target_cache = model.make_cache(batch_size=1, device=device, dtype=model.dtype)
    if not hasattr(target_cache, "snapshot"):
        raise TypeError("MTP requires a hybrid target cache with snapshot/restore")
    if prefill_embeddings is None:
        logits, target_hidden = model.forward(
            tokens,
            cache=target_cache,
            position_ids=prefill_positions,
            attention_mask=prefill_mask,
            return_hidden=True,
        )
    else:
        logits, target_hidden = model.forward(
            cache=target_cache,
            inputs_embeds=prefill_embeddings,
            position_ids=prefill_positions,
            attention_mask=prefill_mask,
            return_hidden=True,
        )
    assert isinstance(logits, torch.Tensor)
    seed = int(torch.argmax(logits[0, -1]).item())

    mtp_cache = mtp.make_cache()
    if tokens.shape[1] > 1:
        prime_positions = prefill_positions[..., 1:]
        mtp.forward(
            tokens[:, 1:],
            target_hidden[:, :-1],
            cache=mtp_cache,
            position_ids=prime_positions,
            input_embeddings=prime_embeddings,
        )
    previous_hidden = target_hidden[:, -1:]
    eos_ids = _eos_id_set(model.config.eos_token_id)
    generated: list[int] = []
    decoded = ""

    def emit(token_id: int) -> str:
        nonlocal decoded
        generated.append(token_id)
        decoded, piece = _delta_piece(tokenizer, generated, decoded)
        return piece

    def positions(start: int, length: int) -> torch.Tensor:
        pos = torch.arange(
            start, start + length, device=device, dtype=torch.long
        )[None]
        if rope_delta is None:
            return pos
        return pos[None].expand(3, 1, -1) + rope_delta.to(device).view(1, 1, 1)

    while len(generated) < max_new_tokens:
        if seed in eos_ids:
            return

        remaining = max_new_tokens - len(generated)
        if remaining == 1:
            yield emit(seed)
            return
        k = min(num_speculative_tokens, remaining - 1)
        mtp_base = mtp_cache.seq_len()
        drafts: list[int] = []
        draft_hidden = previous_hidden
        draft_input = seed
        for j in range(k):
            position = positions(target_cache.seq_len() + j, 1)
            draft_logits, draft_hidden = mtp.forward(
                torch.tensor([[draft_input]], device=device, dtype=torch.long),
                draft_hidden,
                cache=mtp_cache,
                position_ids=position,
            )
            draft_input = int(torch.argmax(draft_logits[0, -1]).item())
            drafts.append(draft_input)

        snapshot = target_cache.snapshot()  # type: ignore[union-attr]
        target_start = target_cache.seq_len()
        verify_ids = torch.tensor(
            [[seed, *drafts]], device=device, dtype=torch.long
        )
        verify_logits, verify_hidden = model.forward(
            verify_ids,
            cache=target_cache,
            position_ids=positions(target_start, k + 1),
            return_hidden=True,
        )
        assert isinstance(verify_logits, torch.Tensor)
        target_next = torch.argmax(verify_logits[0], dim=-1).tolist()
        accepted = 0
        while accepted < k and drafts[accepted] == int(target_next[accepted]):
            accepted += 1
        if stats is not None:
            stats["rounds"] += 1
            stats["drafted"] += k
            stats["accepted"] += accepted

        round_tokens = [seed, *drafts[:accepted]]
        if accepted == k:
            bonus = int(target_next[k])
            # Keep the MTP KV stream aligned through the final accepted draft.
            mtp.forward(
                torch.tensor([[drafts[-1]]], device=device, dtype=torch.long),
                draft_hidden,
                cache=mtp_cache,
                position_ids=positions(target_cache.seq_len() - 1, 1),
            )
            previous_hidden = verify_hidden[:, -1:]
            seed = bonus
        else:
            replacement = int(target_next[accepted])
            target_cache.restore(snapshot)  # type: ignore[union-attr]
            replay_ids = torch.tensor(
                [[seed, *drafts[:accepted]]], device=device, dtype=torch.long
            )
            _, replay_hidden = model.forward(
                replay_ids,
                cache=target_cache,
                position_ids=positions(target_start, 1 + accepted),
                return_hidden=True,
            )
            mtp_cache.truncate(mtp_base + 1 + accepted)
            previous_hidden = replay_hidden[:, -1:]
            seed = replacement

        for token_id in round_tokens:
            if token_id in eos_ids:
                return
            yield emit(token_id)
            if len(generated) == max_new_tokens:
                return
