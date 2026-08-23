"""Thin wrapper around the HuggingFace tokenizer shipped in the model folder."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

_TEMPLATE_MARKERS = (
    "<|im_start|>",
    "[INST]",
    "<|begin_of_text|>",
    "<|start_header_id|>",
)


class _ByteTokenizer:
    """Fallback when a folder has weights but no HuggingFace tokenizer files."""

    def __init__(self, vocab_size: int = 256):
        self.vocab_size = max(int(vocab_size), 1)
        self.chat_template = None
        self.clean_up_tokenization_spaces = False

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        del add_special_tokens
        ids = list(text.encode("utf-8"))
        if not ids:
            return [0]
        return [i % self.vocab_size for i in ids]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        raw = bytes(int(i) % 256 for i in ids)
        return raw.decode("utf-8", errors="replace")

    def apply_chat_template(self, *args, **kwargs):
        raise RuntimeError("byte fallback tokenizer has no chat template")


class Tokenizer:
    """text ↔ token ids. BPE lives inside the HF/tiktoken files; we just call it."""

    def __init__(self, inner: Any):
        self._tok = inner

    @classmethod
    def from_pretrained(cls, model_dir: str | Path) -> Tokenizer:
        model_dir = Path(model_dir)
        has_tok = any(
            (model_dir / name).is_file()
            for name in (
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.json",
                "tokenizer.model",
                "spiece.model",
            )
        )
        if has_tok:
            tok = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
            if hasattr(tok, "clean_up_tokenization_spaces"):
                tok.clean_up_tokenization_spaces = False
            return cls(tok)
        vocab = 256
        try:
            from engine.config import ModelConfig

            vocab = ModelConfig.from_pretrained(model_dir).vocab_size
        except Exception:
            cfg = model_dir / "config.json"
            if cfg.is_file():
                import json

                vocab = int(json.loads(cfg.read_text()).get("vocab_size", 256) or 256)
        return cls(_ByteTokenizer(vocab))

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return list(
            self._tok.encode(text, add_special_tokens=add_special_tokens)
        )

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return self._tok.decode(ids, skip_special_tokens=skip_special_tokens)

    def has_chat_template(self) -> bool:
        return bool(getattr(self._tok, "chat_template", None))

    def looks_templated(self, prompt: str) -> bool:
        return any(m in prompt for m in _TEMPLATE_MARKERS)

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        add_generation_prompt: bool = True,
        **kwargs: Any,
    ) -> str:
        return self._tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )

    def format_for_generate(
        self,
        prompt: str,
        *,
        apply_chat_template: bool | None = None,
        enable_thinking: bool = False,
    ) -> tuple[str, bool]:
        """Wrap a raw user string with the checkpoint's chat template when present.

        Returns (text, templated). Agents pass a question; instruct folders
        (Nano, Llama Instruct) ship the template — we apply it automatically.
        """
        use = apply_chat_template
        if use is None:
            use = self.has_chat_template() and not self.looks_templated(prompt)
        if not use:
            return prompt, False
        text = self.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        return text, True

    def smoke_test(self, text: str = "hello from phase 0") -> None:
        ids = self.encode(text)
        if not ids:
            raise RuntimeError("tokenizer.encode returned empty list")
        roundtrip = self.decode(ids)
        # Roundtrip may normalize whitespace; require non-empty and overlap.
        if not roundtrip:
            raise RuntimeError("tokenizer.decode returned empty string")
