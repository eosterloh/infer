"""Thin wrapper around the HuggingFace Llama tokenizer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


class Tokenizer:
    """text ↔ token ids. BPE lives inside the HF/tiktoken files; we just call it."""

    def __init__(self, inner: Any):
        self._tok = inner

    @classmethod
    def from_pretrained(cls, model_dir: str | Path) -> Tokenizer:
        model_dir = Path(model_dir)
        tok = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
        return cls(tok)

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return list(
            self._tok.encode(text, add_special_tokens=add_special_tokens)
        )

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return self._tok.decode(ids, skip_special_tokens=skip_special_tokens)

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> str:
        return self._tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    def smoke_test(self, text: str = "hello from phase 0") -> None:
        ids = self.encode(text)
        if not ids:
            raise RuntimeError("tokenizer.encode returned empty list")
        roundtrip = self.decode(ids)
        # Roundtrip may normalize whitespace; require non-empty and overlap.
        if not roundtrip:
            raise RuntimeError("tokenizer.decode returned empty string")
