"""Structured output: schema compiler, byte matcher, budget closing, and constrained decode.

The decode tests run random-weight models. A random model has no idea what
JSON is, so every reply that parses is the constraint's doing, not the model's.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest
import torch

from engine.agent_api import load_engine
from engine.config import ModelConfig
from engine.structured import (
    SchemaError,
    StructuredOutputError,
    as_grammar,
    closing,
    initial_states,
    matches,
    step,
    token_byte_table,
)
from engine.synth import random_engine_weights, write_config, write_hf_folder
from engine.tokenizer import Tokenizer

TOOL_CALL = {
    "anyOf": [
        {
            "type": "object",
            "properties": {"tool": {"const": "run_eval"}, "args": {"type": "object", "properties": {}}},
            "required": ["tool", "args"],
        },
        {
            "type": "object",
            "properties": {
                "tool": {"const": "write_hypothesis"},
                "args": {
                    "type": "object",
                    "properties": {
                        "claim": {"type": "string", "maxLength": 40},
                        "why": {"type": "string"},
                        "falsify": {"type": "string"},
                    },
                    "required": ["claim", "why", "falsify"],
                },
            },
            "required": ["tool", "args"],
        },
    ]
}

PACK = {
    "type": "object",
    "properties": {
        "hypothesis": {"type": "string", "maxLength": 60},
        "trainer": {"enum": ["lab", "dummy"]},
        "config": {
            "type": "object",
            "properties": {
                "lr": {"type": "number"},
                "steps": {"type": "integer"},
                "note": {"type": "string"},
            },
            "required": ["lr", "steps"],
            "additionalProperties": True,
        },
        "sources": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 3},
        "parent": {"type": ["string", "null"]},
    },
    "required": ["hypothesis", "trainer", "config", "sources"],
}


@pytest.mark.parametrize(
    "schema, text, ok",
    [
        (TOOL_CALL, '{"tool": "run_eval", "args": {}}', True),
        (TOOL_CALL, '{\n  "tool": "write_hypothesis",\n  "args": {"claim": "a", "why": "b", "falsify": "c"}\n}', True),
        (TOOL_CALL, '{"tool": "write_hypothesis", "args": {"claim": "a", "why": "b"}}', False),
        (TOOL_CALL, '{"tool": "halt", "args": {}}', False),
        (TOOL_CALL, '{"args": {}, "tool": "run_eval"}', False),
        (TOOL_CALL, '{"tool": "run_eval", "args": {"extra": 1}}', False),
        (TOOL_CALL, '```json\n{"tool": "run_eval", "args": {}}', False),
        (TOOL_CALL, '{"tool": "run_eval", "args": {}}\n', False),
        (PACK, '{"hypothesis": "h", "trainer": "lab", "config": {"lr": 3e-4, "steps": 64}, "sources": ["a"]}', True),
        (PACK, '{"hypothesis": "h", "trainer": "lab", "config": {"lr": "0.01", "steps": 64}, "sources": ["a"]}', False),
        (PACK, '{"hypothesis": "h", "trainer": "lab", "config": {"lr": 0.01, "steps": 6.5}, "sources": ["a"]}', False),
        (PACK, '{"hypothesis": "h", "trainer": "lab", "config": {"lr": 1, "steps": 2, "batch": 8}, "sources": ["a"]}', True),
        (PACK, '{"hypothesis": "h", "trainer": "lab", "config": {"lr": 1, "steps": 2, "lr": 8}, "sources": ["a"]}', False),
        (PACK, '{"hypothesis": "h", "trainer": "tinytrain", "config": {"lr": 1, "steps": 2}, "sources": ["a"]}', False),
        (PACK, '{"hypothesis": "h", "trainer": "lab", "config": {"lr": 1, "steps": 2}, "sources": []}', False),
        (PACK, '{"hypothesis": "h", "trainer": "lab", "config": {"lr": 1, "steps": 2}, "sources": ["a"], "parent": null}', True),
        (True, '{"a": [1, -2.5e-3, true, null, "caf\u00e9 \\u00e9\\n"], "b": {}}', True),
        (True, "[1,]", False),
        (True, "01", False),
        (True, '"line\nbreak"', False),
        ({"type": "string", "maxLength": 3}, '"abc"', True),
        ({"type": "string", "maxLength": 3}, '"abcd"', False),
        ({"type": "string", "maxLength": 1}, '"\u00e9"', True),
        ({"type": "string", "minLength": 2}, '"a"', False),
        ({"type": "integer"}, "-12", True),
        ({"type": "integer"}, "1.0", False),
        ({"$defs": {"n": {"type": "number"}}, "type": "array", "items": {"$ref": "#/$defs/n"}}, "[1, 2.5]", True),
    ],
)
def test_matcher_accepts_exactly_the_schema(schema, text, ok) -> None:
    assert matches(schema, text) is ok


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "integer", "minimum": 0},
        {"type": "string", "pattern": "^a"},
        {"allOf": [{"type": "string"}]},
        {"type": "object", "properties": {"a": {}}, "required": ["b"]},
        {"type": "object", "properties": {"a": {}}, "anyOf": [{"required": ["a"]}]},
        {"$defs": {"t": {"type": "array", "items": {"$ref": "#/$defs/t"}}}, "$ref": "#/$defs/t"},
        {"type": "tuple"},
        False,
    ],
)
def test_unsupported_schema_is_refused_not_ignored(schema) -> None:
    with pytest.raises(SchemaError):
        as_grammar(schema)


def _prefix_closes(schema, text: str) -> None:
    grammar = as_grammar(schema)
    data = text.encode()
    states = initial_states(grammar)
    for i, b in enumerate(data):
        states = step(states, bytes((b,)))
        assert states, f"{text[: i + 1]!r} rejected"
        for stack in states:
            full = data[: i + 1] + closing(stack)
            assert matches(schema, full), full
            json.loads(full.decode())


def test_every_prefix_has_a_valid_shortest_ending() -> None:
    _prefix_closes(TOOL_CALL, '{"tool": "write_hypothesis", "args": {"claim": "x\\u00e9y", "why": "b", "falsify": "c"}}')
    _prefix_closes(PACK, '{"hypothesis": "h", "trainer": "lab", "config": {"lr": -1.5e+3, "steps": 64, "x": [1]}, "sources": ["a", "b"], "parent": null}')
    _prefix_closes(True, '{"a": [1, {"b": null}], "c": "d\u00e9"}')


def test_random_walks_always_close() -> None:
    rng = random.Random(0)
    for schema in (TOOL_CALL, PACK, True):
        for _ in range(20):
            states = initial_states(as_grammar(schema))
            data = b""
            for _ in range(rng.randint(1, 80)):
                legal = [b for b in range(256) if step(states, bytes((b,)))]
                if not legal:
                    break
                b = rng.choice(legal)
                data += bytes((b,))
                states = step(states, bytes((b,)))
            end = min((closing(s) for s in states), key=len)
            assert matches(schema, data + end), data + end
            json.loads((data + end).decode())


def test_string_bytes_are_strict_utf8() -> None:
    """A string prefix is legal exactly when some UTF-8 completion of it exists."""
    import itertools

    start = step(initial_states(as_grammar({"type": "string"})), b'"')
    edges = [bytes((b,)) for b in (0x80, 0x8F, 0x90, 0x9F, 0xA0, 0xBF)]

    def utf8_prefix(body: bytes) -> bool:
        for n in range(4 - len(body) + 1):
            for tail in itertools.product(edges, repeat=n):
                try:
                    (body + b"".join(tail)).decode()
                except UnicodeDecodeError:
                    continue
                return True
        return False

    bodies = [bytes((a,)) for a in range(0x80, 0x100)]
    bodies += [bytes((a, b)) for a in range(0xC0, 0x100) for b in range(0x70, 0x100)]
    bodies += [bytes((a, b, c)) for a in (0xE0, 0xED, 0xF0, 0xF4) for b in range(0x80, 0xC0) for c in (0x7F, 0x80, 0xBF, 0xC0)]
    for body in bodies:
        states = step(start, body)
        assert bool(states) is utf8_prefix(body), body
        for stack in states:
            assert json.loads((b'"' + body + closing(stack)).decode()) is not None


# --------------------------------------------------------------- decoding


def _llama_folder(tmp_path: Path, vocab_size: int, eos: int, name: str = "llama") -> Path:
    raw = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size": vocab_size,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "max_position_embeddings": 1024,
        "tie_word_embeddings": True,
        "torch_dtype": "float32",
        "hidden_act": "silu",
        "eos_token_id": eos,
    }
    folder = write_config(tmp_path / name, raw)
    cfg = ModelConfig.from_pretrained(folder)
    write_hf_folder(folder, cfg, random_engine_weights(cfg, seed=0))
    return folder


@pytest.fixture
def byte_engine(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("INFER_MEM_FLOOR_GB", "0")
    return load_engine(_llama_folder(tmp_path, 300, 299), device="cpu", dtype="float32")


@pytest.mark.parametrize("schema", [TOOL_CALL, PACK, True, {"type": "array", "items": {"type": "boolean"}, "minItems": 2}])
def test_random_model_replies_match_the_schema(byte_engine, schema) -> None:
    free = byte_engine.generate("hi", max_new_tokens=24, apply_chat_template=False)
    with pytest.raises(json.JSONDecodeError):
        json.loads(free)
    for kwargs in ({}, {"temperature": 1.0, "seed": 1}, {"temperature": 0.7, "top_k": 5, "top_p": 0.9, "seed": 2}):
        text = byte_engine.generate("hi", max_new_tokens=400, apply_chat_template=False, json_schema=schema, **kwargs)
        assert matches(schema, text), text
        json.loads(text)


def test_reply_closes_inside_a_tight_budget(byte_engine) -> None:
    shortest = len(as_grammar(PACK).shortest)
    for budget in (shortest, shortest + 3, shortest + 40):
        text = byte_engine.generate("hi", max_new_tokens=budget, apply_chat_template=False, json_schema=PACK)
        assert len(text.encode()) <= budget
        assert matches(PACK, text), (budget, text)


def test_generate_json_and_stream(byte_engine) -> None:
    value = byte_engine.generate_json("hi", {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}, apply_chat_template=False)
    assert isinstance(value["ok"], bool)
    pieces = list(byte_engine.stream("hi", max_new_tokens=200, apply_chat_template=False, json_schema=TOOL_CALL))
    assert matches(TOOL_CALL, "".join(pieces))
    assert byte_engine.generate("hi", max_new_tokens=200, apply_chat_template=False, json_schema=TOOL_CALL, use_cache=False) == "".join(pieces)
    with pytest.raises(StructuredOutputError) as err:
        byte_engine.generate_json("hi", PACK, max_new_tokens=5, apply_chat_template=False)
    assert err.value.text


def test_schema_refuses_modes_it_cannot_govern(byte_engine) -> None:
    with pytest.raises(ValueError, match="enable_thinking"):
        byte_engine.generate("hi", json_schema=True, enable_thinking=True)
    with pytest.raises(ValueError, match="MTP"):
        byte_engine.generate("hi", json_schema=True, num_speculative_tokens=2)


# ------------------------------------------------------------ tokenizers

_CORPUS = [
    '{"tool": "write_pack", "args": {"pack": {"hypothesis": "raise lr", "config": {"lr": 0.003, "steps": 64}}}}',
    '{"tool": "run_eval", "args": {}}',
    "The model writes JSON with keys, values, numbers like 12.5 and words like caf\u00e9.",
] * 20


def _byte_level_tokenizer(folder: Path) -> None:
    from tokenizers import Tokenizer as Raw, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    raw = Raw(models.BPE())
    raw.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    raw.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=420, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), special_tokens=["<|eos|>"]
    )
    raw.train_from_iterator(_CORPUS, trainer)
    PreTrainedTokenizerFast(tokenizer_object=raw, eos_token="<|eos|>").save_pretrained(folder)


def _sentencepiece_tokenizer(folder: Path) -> None:
    from tokenizers import Tokenizer as Raw, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2}
    for b in range(256):
        vocab[f"<0x{b:02X}>"] = len(vocab)
    for ch in "\u2581" + "".join(chr(c) for c in range(33, 127)):
        vocab.setdefault(ch, len(vocab))
    merges = [("\u2581", '"'), ('"', ":"), ("t", "o"), ("to", "o"), ("too", "l"), ('"', ","), ("\u2581", "{")]
    for a, b in merges:
        vocab.setdefault(a + b, len(vocab))
    raw = Raw(models.BPE(vocab=vocab, merges=merges, unk_token="<unk>", byte_fallback=True, fuse_unk=True))
    raw.pre_tokenizer = pre_tokenizers.Metaspace(replacement="\u2581", prepend_scheme="first")
    raw.decoder = decoders.Sequence(
        [decoders.Replace("\u2581", " "), decoders.ByteFallback(), decoders.Fuse(), decoders.Strip(" ", 1, 0)]
    )
    PreTrainedTokenizerFast(
        tokenizer_object=raw, bos_token="<s>", eos_token="</s>", unk_token="<unk>"
    ).save_pretrained(folder)


@pytest.mark.parametrize("build", [_byte_level_tokenizer, _sentencepiece_tokenizer])
def test_token_bytes_agree_with_decode(tmp_path: Path, build) -> None:
    build(tmp_path)
    tok = Tokenizer.from_pretrained(tmp_path)
    inner = tok._tok
    table = token_byte_table(tok, len(inner))
    special = set(inner.all_special_ids)
    assert all(table[i] is None for i in special)
    ascii_ids = [i for i, t in enumerate(table) if t and all(32 <= b < 127 for b in t)]
    rng = random.Random(0)
    for _ in range(200):
        ids = [rng.choice(ascii_ids) for _ in range(rng.randint(1, 12))]
        want = b"".join(table[i] for i in ids).decode()
        got = inner.decode(ids, skip_special_tokens=False)
        # SentencePiece decoders strip the one leading space they add to a sequence.
        assert got in (want, want[1:] if want.startswith(" ") else want), (ids, want, got)
    multi = inner.encode("caf\u00e9", add_special_tokens=False)
    assert b"".join(table[i] for i in multi).decode().strip() == "caf\u00e9"


@pytest.mark.parametrize("build", [_byte_level_tokenizer, _sentencepiece_tokenizer])
def test_constrained_decode_with_real_tokenizers(tmp_path: Path, monkeypatch, build) -> None:
    monkeypatch.setenv("INFER_MEM_FLOOR_GB", "0")
    tok_dir = tmp_path / "tok"
    build(tok_dir)
    inner = Tokenizer.from_pretrained(tok_dir)._tok
    folder = _llama_folder(tmp_path, len(inner) + 7, inner.eos_token_id)
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        if (tok_dir / name).is_file():
            (folder / name).write_bytes((tok_dir / name).read_bytes())
    eng = load_engine(folder, device="cpu", dtype="float32")
    for schema in (TOOL_CALL, PACK):
        for seed in (None, 3):
            kwargs = {"temperature": 1.0, "seed": seed} if seed is not None else {}
            text = eng.generate("hi", max_new_tokens=300, apply_chat_template=False, json_schema=schema, **kwargs)
            assert matches(schema, text.encode()), text
            json.loads(text)


def test_constraint_never_picks_padding_rows(byte_engine) -> None:
    """Logit rows past the tokenizer's vocabulary are never legal tokens."""
    from engine.structured import JsonConstraint, TokenVocab

    vocab = TokenVocab([bytes((i,)) for i in range(256)], eos_ids=[])
    constraint = JsonConstraint(as_grammar(True), vocab)
    logits = torch.full((300,), -10.0)
    logits[280] = 50.0
    assert constraint.pick(logits, remaining=10) < 256
