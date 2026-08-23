"""Tests: folder chat template is applied automatically for agents."""

from __future__ import annotations

from engine.tokenizer import Tokenizer


class _StubInner:
    chat_template = "present"

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **kwargs,
    ) -> str:
        think = kwargs.get("enable_thinking", True)
        return f"TMPL|{messages[0]['content']}|think={think}"

    def encode(self, text, add_special_tokens=True):
        return [1, 2, 3]

    def decode(self, ids, skip_special_tokens=True):
        return "ok"


def test_format_for_generate_auto_wraps_raw_prompt() -> None:
    tok = Tokenizer(_StubInner())
    text, templated = tok.format_for_generate("Hello", enable_thinking=False)
    assert templated is True
    assert text == "TMPL|Hello|think=False"


def test_format_for_generate_skips_already_wrapped() -> None:
    tok = Tokenizer(_StubInner())
    raw = "<|im_start|>user\nHello"
    text, templated = tok.format_for_generate(raw)
    assert templated is False
    assert text == raw


def test_format_for_generate_raw_opt_out() -> None:
    tok = Tokenizer(_StubInner())
    text, templated = tok.format_for_generate("Hello", apply_chat_template=False)
    assert templated is False
    assert text == "Hello"
