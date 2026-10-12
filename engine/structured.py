"""Structured output: decode only tokens that keep the reply valid JSON for a schema.

The model still chooses every token. This module only takes away the ones that
would break the format, so a small agent that has to answer with one JSON
object per turn (the lab harness's experimenter and its roles) parses on the
first try instead of burning a retry, a nudge, or a harness fallback.

How it fits together:
    * ``compile_schema`` turns a JSON Schema (the subset in
      ``SUPPORTED_KEYWORDS``) into a small tree of nodes.
    * A pushdown matcher walks the reply one *byte* at a time. Tokens are
      bytes — byte-level BPE directly, SentencePiece through its ``<0xNN>``
      fallback — so a token is legal when the matcher survives every byte of
      it. ``anyOf`` is handled by carrying several stacks at once.
    * ``JsonConstraint.pick`` asks for the model's own choice first and only
      walks further down the ranking when that token is illegal, with a
      first-byte mask over the vocabulary to skip what cannot even start.
    * Every open string, array and object knows its shortest legal ending, and
      a token is only taken if the tokens left can still write that ending, so
      a reply that runs into ``max_new_tokens`` still closes.

What the schema means here, where JSON Schema leaves it open:
    * object keys come out in ``properties`` order; optional ones may be
      skipped, required ones may not;
    * ``additionalProperties`` defaults to false when ``properties`` is given
      (an open object when it is not);
    * whitespace between tokens is allowed but capped at ``MAX_WS_RUN`` bytes
      per gap, and nothing may follow the closing brace.
"""

from __future__ import annotations

import json
import re
import weakref
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import torch

from engine.sample import SamplingParams, filtered_logits

MAX_WS_RUN = 20
MAX_NUMBER_BYTES = 24
MAX_KEY_CHARS = 64
MAX_DEPTH = 48

SUPPORTED_KEYWORDS = frozenset(
    {
        "type",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "minItems",
        "maxItems",
        "minLength",
        "maxLength",
        "enum",
        "const",
        "anyOf",
        "oneOf",
        "$ref",
    }
)
_ANNOTATIONS = frozenset(
    {"title", "description", "default", "examples", "$schema", "$id", "$comment", "$defs", "definitions"}
)


class SchemaError(ValueError):
    """The schema asks for something the matcher cannot enforce."""


class StructuredOutputError(RuntimeError):
    """The reply did not come out as JSON for the schema; ``text`` is what was produced."""

    def __init__(self, message: str, text: str = "") -> None:
        super().__init__(message)
        self.text = text


def _dump(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


# --------------------------------------------------------------------- nodes
# Every node carries ``shortest``: the fewest bytes that make a legal value of
# it. The budget rule and the closing logic are both built on that.


class Node:
    __slots__ = ("shortest", "_starts")

    def starts(self) -> tuple:
        if self._starts is None:
            self._starts = _start_frames(self)
        return self._starts


class StringNode(Node):
    __slots__ = ("min_len", "max_len")

    def __init__(self, min_len: int = 0, max_len: int | None = None) -> None:
        self.min_len = min_len
        self.max_len = max_len
        self.shortest = b'"' + b"a" * min_len + b'"'
        self._starts = None


class NumberNode(Node):
    __slots__ = ("integer",)

    def __init__(self, integer: bool) -> None:
        self.integer = integer
        self.shortest = b"0"
        self._starts = None


class LiteralNode(Node):
    """``const`` / ``enum`` / ``true`` / ``false`` / ``null``: exact JSON text."""

    __slots__ = ("options",)

    def __init__(self, options: tuple[bytes, ...]) -> None:
        self.options = options
        self.shortest = min(options, key=len)
        self._starts = None


class ArrayNode(Node):
    __slots__ = ("item", "min_items", "max_items")

    def __init__(self, item: Node, min_items: int = 0, max_items: int | None = None) -> None:
        self.item = item
        self.min_items = min_items
        self.max_items = max_items
        self.shortest = b"[" + b",".join([item.shortest] * min_items) + b"]"
        self._starts = None


class UnionNode(Node):
    __slots__ = ("options",)

    def __init__(self, options: tuple[Node, ...]) -> None:
        self.options = options
        self.shortest = min((o.shortest for o in options), key=len)
        self._starts = None


class AnyNode(Node):
    """Any JSON value — what ``{}`` / ``true`` / an open ``additionalProperties`` mean."""

    __slots__ = ()

    def __init__(self) -> None:
        self.shortest = b"0"
        self._starts = None


class ObjectNode(Node):
    __slots__ = ("names", "keys", "values", "required", "extra", "next_required", "tail", "free_key", "_key_frames")

    def __init__(self, props: list[tuple[str, Node]], required: Iterable[str], extra: Node | None) -> None:
        want = set(required)
        self.names = tuple(name for name, _ in props)
        self.keys = tuple(_dump(name) for name in self.names)
        self.values = tuple(node for _, node in props)
        self.required = frozenset(i for i, name in enumerate(self.names) if name in want)
        self.extra = extra
        n = len(self.names)
        next_required: list[int | None] = [None] * (n + 1)
        upcoming = None
        for i in range(n - 1, -1, -1):
            if i in self.required:
                upcoming = i
            next_required[i] = upcoming
        self.next_required = tuple(next_required)
        # tail[pos]: what closes the object right after a value, once every
        # property before ``pos`` is settled — the required ones still owed, then '}'.
        self.tail = tuple(
            b"".join(b"," + self.keys[i] + b":" + self.values[i].shortest for i in range(pos, n) if i in self.required)
            + b"}"
            for pos in range(n + 1)
        )
        self.shortest = b"{" + self.tail[0][1:] if self.required else b"{}"
        free = b'""'
        while extra is not None and free in self.keys:
            free = free[:-1] + b'_"'
        self.free_key = free
        self._key_frames: dict[int, tuple] = {}
        self._starts = None

    def key_frames(self, pos: int) -> tuple:
        """Frames that may start a key at ``pos``: declared names up to the next required one, or a free key."""
        cached = self._key_frames.get(pos)
        if cached is None:
            upto = self.next_required[pos]
            end = len(self.names) if upto is None else upto + 1
            options = tuple(
                (self.keys[i], i, 1 + len(self.values[i].shortest) + len(self.tail[i + 1])) for i in range(pos, end)
            )
            frames: list = [_Lit(options)] if options else []
            if upto is None and self.extra is not None:
                frames.append(_Str(0, MAX_KEY_CHARS, key=True, forbidden=frozenset(self.names)))
            cached = self._key_frames[pos] = tuple(frames)
        return cached


ANY = AnyNode()
_FREE_OBJECT = ObjectNode([], (), ANY)
_FREE_ARRAY = ArrayNode(ANY)


# ------------------------------------------------------------------ compiler


def compile_schema(schema: Any) -> Node:
    """JSON Schema (dict or bool) → node tree. Raises SchemaError on anything not enforced."""
    return _Compiler(schema).node(schema)


@lru_cache(maxsize=256)
def _compile_text(schema_json: str) -> Node:
    return compile_schema(json.loads(schema_json))


def as_grammar(schema: Any) -> Node:
    """Compiled node for ``schema``, cached on its JSON text (property order is significant)."""
    if isinstance(schema, Node):
        return schema
    try:
        text = json.dumps(schema)
    except TypeError as e:
        raise SchemaError(f"schema is not JSON-serializable: {e}") from e
    return _compile_text(text)


def _union(nodes: list[Node]) -> Node:
    flat: list[Node] = []
    for node in nodes:
        flat.extend(node.options if isinstance(node, UnionNode) else (node,))
    if any(node is ANY for node in flat):
        return ANY
    return flat[0] if len(flat) == 1 else UnionNode(tuple(flat))


class _Compiler:
    def __init__(self, root: Any) -> None:
        self.root = root
        self.resolving: list[str] = []

    def node(self, s: Any) -> Node:
        if s is True:
            return ANY
        if s is False:
            raise SchemaError("schema `false` admits no value")
        if not isinstance(s, dict):
            raise SchemaError(f"schema must be an object or a boolean, got {type(s).__name__}")
        unknown = set(s) - SUPPORTED_KEYWORDS - _ANNOTATIONS
        if unknown:
            raise SchemaError(
                f"unsupported JSON Schema keyword(s) {sorted(unknown)}; supported: {sorted(SUPPORTED_KEYWORDS)}"
            )
        if "$ref" in s:
            self._alone(s, "$ref")
            return self.ref(s["$ref"])
        if "const" in s:
            return LiteralNode((_dump(s["const"]),))
        if "enum" in s:
            values = s["enum"]
            if not isinstance(values, list) or not values:
                raise SchemaError("enum must be a non-empty list")
            return LiteralNode(tuple(dict.fromkeys(_dump(v) for v in values)))
        for key in ("anyOf", "oneOf"):
            if key in s:
                self._alone(s, key)
                branches = s[key]
                if not isinstance(branches, list) or not branches:
                    raise SchemaError(f"{key} must be a non-empty list")
                return _union([self.node(b) for b in branches])
        kind = s.get("type")
        if isinstance(kind, list):
            if not kind:
                raise SchemaError("type list is empty")
            return _union([self.typed(s, k) for k in kind])
        if kind is None:
            kind = _implied_type(s)
            if kind is None:
                return ANY
        return self.typed(s, kind)

    @staticmethod
    def _alone(s: dict, key: str) -> None:
        rest = set(s) - _ANNOTATIONS - {key}
        if rest:
            raise SchemaError(f"{key} next to {sorted(rest)} is not supported; move those into each branch")

    def typed(self, s: dict, kind: str) -> Node:
        if kind == "string":
            lo = int(s.get("minLength", 0))
            hi = s.get("maxLength")
            hi = None if hi is None else int(hi)
            if lo < 0 or (hi is not None and hi < lo):
                raise SchemaError(f"bad string length bounds {lo}..{hi}")
            return StringNode(lo, hi)
        if kind in ("integer", "number"):
            return NumberNode(kind == "integer")
        if kind == "boolean":
            return LiteralNode((b"true", b"false"))
        if kind == "null":
            return LiteralNode((b"null",))
        if kind == "object":
            props = s.get("properties") or {}
            if not isinstance(props, dict):
                raise SchemaError("properties must be an object")
            required = s.get("required") or []
            missing = [name for name in required if name not in props]
            if missing:
                raise SchemaError(f"required {missing} not listed in properties")
            if "additionalProperties" in s:
                extra_spec = s["additionalProperties"]
                extra = None if extra_spec is False else self.node(extra_spec)
            else:
                extra = None if "properties" in s else ANY
            return ObjectNode([(name, self.node(sub)) for name, sub in props.items()], required, extra)
        if kind == "array":
            lo = int(s.get("minItems", 0))
            hi = s.get("maxItems")
            hi = None if hi is None else int(hi)
            if lo < 0 or (hi is not None and hi < lo):
                raise SchemaError(f"bad array length bounds {lo}..{hi}")
            return ArrayNode(self.node(s.get("items", True)), lo, hi)
        raise SchemaError(f"unknown type {kind!r}")

    def ref(self, pointer: Any) -> Node:
        if not isinstance(pointer, str) or not pointer.startswith("#"):
            raise SchemaError(f"only local $ref ('#/...') is supported, got {pointer!r}")
        if pointer in self.resolving:
            raise SchemaError(f"recursive $ref {pointer} is not supported")
        target: Any = self.root
        for part in [p for p in pointer[1:].split("/") if p]:
            part = part.replace("~1", "/").replace("~0", "~")
            if not isinstance(target, dict) or part not in target:
                raise SchemaError(f"$ref {pointer} does not resolve")
            target = target[part]
        self.resolving.append(pointer)
        try:
            return self.node(target)
        finally:
            self.resolving.pop()


def _implied_type(s: dict) -> str | None:
    if {"properties", "required", "additionalProperties"} & set(s):
        return "object"
    if {"items", "minItems", "maxItems"} & set(s):
        return "array"
    if {"minLength", "maxLength"} & set(s):
        return "string"
    return None


# ------------------------------------------------------------------- frames
# A matcher state is a stack of frames, innermost last. ``feed(b)`` returns what
# the top frame does with one byte:
#   (STAY, frame)           consumed it, still open
#   (PUSH, frame, children) a value starts here; each child frame gets the byte
#   (DONE, result)          consumed it and closed (result: which key, for keys)
#   (REFEED, result)        closed without it; the parent gets the byte (numbers)
# Frames are immutable, so branching a state for a candidate token costs nothing.

STAY, PUSH, DONE, REFEED = range(4)
EXTRA = -1  # result of a key frame that matched an additionalProperties name
_WS = frozenset(b" \t\n\r")
_ESCAPES = frozenset(b'"\\/bfnrt')
_HEX = frozenset(b"0123456789abcdefABCDEF")
# Lead bytes whose first continuation byte is narrower than 80..BF: these ranges
# rule out overlong forms, UTF-16 surrogates, and code points past U+10FFFF.
_UTF8_SECOND = {0xE0: (0xA0, 0xBF), 0xED: (0x80, 0x9F), 0xF0: (0x90, 0xBF), 0xF4: (0x80, 0x8F)}


def _decode_key(raw: bytes) -> str | None:
    try:
        return json.loads('"' + raw.decode("utf-8", "replace") + '"')
    except ValueError:
        return None


@dataclass(frozen=True, slots=True)
class _Root:
    node: Node
    phase: int = 0  # 0 before the value, 1 inside it, 2 finished
    ws: int = 0

    def feed(self, b: int) -> list:
        if self.phase != 0:
            return []
        if b in _WS:
            return [(STAY, _Root(self.node, 0, self.ws + 1))] if self.ws < MAX_WS_RUN else []
        return [(PUSH, _Root(self.node, 1), self.node.starts())]

    def child_done(self, result: Any) -> _Root:
        return _Root(self.node, 2)

    def close(self) -> bytes:
        return self.node.shortest if self.phase == 0 else b""

    def close_result(self) -> Any:
        return None

    def finishable(self) -> bool:
        return False


S_OPEN, S_BODY, S_ESC, S_HEX, S_UTF8 = range(5)


@dataclass(frozen=True, slots=True)
class _Str:
    min_len: int = 0
    max_len: int | None = None
    phase: int = S_OPEN
    length: int = 0  # characters, not bytes
    pending: int = 0  # hex digits or UTF-8 continuation bytes still owed
    key: bool = False  # an additionalProperties name rather than a value
    forbidden: frozenset = frozenset()  # names that key may not take (the declared ones)
    raw: bytes = b""  # the body so far, kept only while there is a name to refuse
    low: int = 0x80  # range of the next UTF-8 continuation byte
    high: int = 0xBF

    def _to(
        self, b: int, phase: int, length: int, pending: int = 0, low: int = 0x80, high: int = 0xBF
    ) -> _Str:
        raw = self.raw + bytes((b,)) if self.forbidden else b""
        return _Str(
            self.min_len, self.max_len, phase, length, pending, self.key, self.forbidden, raw, low, high
        )

    def feed(self, b: int) -> list:
        phase = self.phase
        if phase == S_BODY:
            room = self.max_len is None or self.length < self.max_len
            if b == 0x22:
                if self.length < self.min_len:
                    return []
                if self.forbidden:
                    name = _decode_key(self.raw)
                    if name is None or name in self.forbidden:
                        return []
                return [(DONE, EXTRA if self.key else None)]
            if not room or b < 0x20:
                return []
            if b == 0x5C:
                return [(STAY, self._to(b, S_ESC, self.length + 1))]
            if b < 0x80:
                return [(STAY, self._to(b, S_BODY, self.length + 1))]
            if 0xC2 <= b <= 0xF4:
                owed = 1 if b <= 0xDF else 2 if b <= 0xEF else 3
                low, high = _UTF8_SECOND.get(b, (0x80, 0xBF))
                return [(STAY, self._to(b, S_UTF8, self.length + 1, owed, low, high))]
            return []
        if phase == S_OPEN:
            if b != 0x22:
                return []
            return [(STAY, _Str(self.min_len, self.max_len, S_BODY, 0, 0, self.key, self.forbidden))]
        if phase == S_ESC:
            if b in _ESCAPES:
                return [(STAY, self._to(b, S_BODY, self.length))]
            if b == 0x75:
                return [(STAY, self._to(b, S_HEX, self.length, 4))]
            return []
        if phase == S_HEX:
            if b not in _HEX:
                return []
            left = self.pending - 1
            return [(STAY, self._to(b, S_HEX if left else S_BODY, self.length, left))]
        if self.low <= b <= self.high:
            left = self.pending - 1
            return [(STAY, self._to(b, S_UTF8 if left else S_BODY, self.length, left))]
        return []

    def child_done(self, result: Any) -> _Str:
        raise AssertionError("strings have no children")

    def close(self) -> bytes:
        if self.phase == S_OPEN:
            return b'"' + b"a" * self.min_len + b'"'
        owed = {
            S_ESC: b"n",
            S_HEX: b"0" * self.pending,
            S_UTF8: bytes((self.low,)) + b"\x80" * (self.pending - 1),
        }.get(self.phase, b"")
        body = owed + b"a" * max(0, self.min_len - self.length)
        if self.forbidden:
            while _decode_key(self.raw + body) in self.forbidden:
                body += b"_"
        return body + b'"'

    def close_result(self) -> Any:
        return EXTRA if self.key else None

    def finishable(self) -> bool:
        return False


N_START, N_MINUS, N_ZERO, N_INT, N_DOT, N_FRAC, N_E, N_ESIGN, N_EXP = range(9)
_NUMBER_ENDS = frozenset({N_ZERO, N_INT, N_FRAC, N_EXP})


@dataclass(frozen=True, slots=True)
class _Num:
    integer: bool
    phase: int = N_START
    n: int = 0

    def feed(self, b: int) -> list:
        phase = self.phase
        digit = 0x30 <= b <= 0x39
        nxt = None
        if self.n < MAX_NUMBER_BYTES:
            exp = b in (0x65, 0x45) and not self.integer
            dot = b == 0x2E and not self.integer
            if phase == N_START:
                nxt = N_MINUS if b == 0x2D else N_ZERO if b == 0x30 else N_INT if digit else None
            elif phase == N_MINUS:
                nxt = N_ZERO if b == 0x30 else N_INT if digit else None
            elif phase == N_ZERO:
                nxt = N_DOT if dot else N_E if exp else None
            elif phase == N_INT:
                nxt = N_INT if digit else N_DOT if dot else N_E if exp else None
            elif phase == N_DOT:
                nxt = N_FRAC if digit else None
            elif phase == N_FRAC:
                nxt = N_FRAC if digit else N_E if exp else None
            elif phase == N_E:
                nxt = N_ESIGN if b in (0x2B, 0x2D) else N_EXP if digit else None
            else:  # N_ESIGN, N_EXP
                nxt = N_EXP if digit else None
        if nxt is not None:
            return [(STAY, _Num(self.integer, nxt, self.n + 1))]
        return [(REFEED, None)] if phase in _NUMBER_ENDS else []

    def child_done(self, result: Any) -> _Num:
        raise AssertionError("numbers have no children")

    def close(self) -> bytes:
        return b"" if self.phase in _NUMBER_ENDS else b"0"

    def close_result(self) -> Any:
        return None

    def finishable(self) -> bool:
        return self.phase in _NUMBER_ENDS


@dataclass(frozen=True, slots=True)
class _Lit:
    # (remaining bytes, result, extra close cost): the cost lets a key choose the
    # ending that is cheapest including the value and the keys still owed after it.
    options: tuple

    def feed(self, b: int) -> list:
        finished: list = []
        rest = []
        for suffix, result, cost in self.options:
            if suffix[0] == b:
                if len(suffix) == 1:
                    finished.append(result)
                else:
                    rest.append((suffix[1:], result, cost))
        out: list = [(DONE, r) for r in dict.fromkeys(finished)]
        if rest:
            out.append((STAY, _Lit(tuple(rest))))
        return out

    def child_done(self, result: Any) -> _Lit:
        raise AssertionError("literals have no children")

    def _best(self) -> tuple:
        return min(self.options, key=lambda o: len(o[0]) + o[2])

    def close(self) -> bytes:
        return self._best()[0]

    def close_result(self) -> Any:
        return self._best()[1]

    def finishable(self) -> bool:
        return False


O_OPEN, O_FIRST, O_NEXT, O_KEY, O_COLON, O_VALUE, O_AWAIT, O_AFTER = range(8)
_O_GAPS = frozenset({O_FIRST, O_NEXT, O_COLON, O_VALUE, O_AFTER})


@dataclass(frozen=True, slots=True)
class _Obj:
    node: ObjectNode
    phase: int = O_OPEN
    pos: int = 0  # every declared property before this one is settled
    current: int | None = None  # property index (or EXTRA) whose value is being written
    ws: int = 0

    def feed(self, b: int) -> list:
        node, phase = self.node, self.phase
        if phase == O_OPEN:
            return [(STAY, _Obj(node, O_FIRST))] if b == 0x7B else []
        if b in _WS and phase in _O_GAPS:
            if self.ws >= MAX_WS_RUN:
                return []
            return [(STAY, _Obj(node, phase, self.pos, self.current, self.ws + 1))]
        if phase == O_FIRST or phase == O_NEXT:
            if b == 0x22:
                keys = node.key_frames(self.pos)
                return [(PUSH, _Obj(node, O_KEY, self.pos), keys)] if keys else []
            if b == 0x7D and phase == O_FIRST and node.next_required[self.pos] is None:
                return [(DONE, None)]
            return []
        if phase == O_COLON:
            return [(STAY, _Obj(node, O_VALUE, self.pos, self.current))] if b == 0x3A else []
        if phase == O_VALUE:
            return [(PUSH, _Obj(node, O_AWAIT, self.pos, self.current), self._value().starts())]
        if phase == O_AFTER:
            if b == 0x2C:
                more = self.pos < len(node.names) or (
                    node.next_required[self.pos] is None and node.extra is not None
                )
                return [(STAY, _Obj(node, O_NEXT, self.pos))] if more else []
            if b == 0x7D and node.next_required[self.pos] is None:
                return [(DONE, None)]
        return []

    def _value(self) -> Node:
        return self.node.extra if self.current == EXTRA else self.node.values[self.current]

    def _pos_after_value(self) -> int:
        return self.pos if self.current == EXTRA else self.current + 1

    def child_done(self, result: Any) -> _Obj:
        if self.phase == O_KEY:
            return _Obj(self.node, O_COLON, self.pos, result)
        return _Obj(self.node, O_AFTER, self._pos_after_value())

    def close(self) -> bytes:
        node, phase = self.node, self.phase
        if phase == O_OPEN:
            return node.shortest
        if phase == O_AFTER:
            return node.tail[self.pos]
        if phase == O_COLON:
            return b":" + self._value().shortest + node.tail[self._pos_after_value()]
        if phase == O_VALUE:
            return self._value().shortest + node.tail[self._pos_after_value()]
        owed = node.next_required[self.pos] is not None
        if phase == O_FIRST:
            return node.tail[self.pos][1:] if owed else b"}"
        if phase == O_NEXT:
            if owed:
                return node.tail[self.pos][1:]
            endings = [
                node.keys[i] + b":" + node.values[i].shortest + node.tail[i + 1] for i in range(self.pos, len(node.names))
            ]
            if node.extra is not None:
                endings.append(node.free_key + b":" + node.extra.shortest + node.tail[self.pos])
            return min(endings, key=len)
        return b""  # O_KEY / O_AWAIT: the child above closes first

    def close_result(self) -> Any:
        return None

    def finishable(self) -> bool:
        return False


A_OPEN, A_FIRST, A_NEXT, A_AWAIT, A_AFTER = range(5)


@dataclass(frozen=True, slots=True)
class _Arr:
    node: ArrayNode
    phase: int = A_OPEN
    count: int = 0
    ws: int = 0

    def feed(self, b: int) -> list:
        node, phase = self.node, self.phase
        if phase == A_OPEN:
            return [(STAY, _Arr(node, A_FIRST))] if b == 0x5B else []
        if b in _WS and phase != A_AWAIT:
            return [(STAY, _Arr(node, phase, self.count, self.ws + 1))] if self.ws < MAX_WS_RUN else []
        if phase == A_FIRST:
            if b == 0x5D:
                return [(DONE, None)] if node.min_items == 0 else []
            if node.max_items == 0:
                return []
            return [(PUSH, _Arr(node, A_AWAIT, 0), node.item.starts())]
        if phase == A_NEXT:
            return [(PUSH, _Arr(node, A_AWAIT, self.count), node.item.starts())]
        if phase == A_AFTER:
            if b == 0x2C and (node.max_items is None or self.count < node.max_items):
                return [(STAY, _Arr(node, A_NEXT, self.count))]
            if b == 0x5D and self.count >= node.min_items:
                return [(DONE, None)]
        return []

    def child_done(self, result: Any) -> _Arr:
        return _Arr(self.node, A_AFTER, self.count + 1)

    def close(self) -> bytes:
        node, phase = self.node, self.phase
        item = node.item.shortest
        if phase == A_OPEN:
            return node.shortest
        if phase == A_FIRST:
            return node.shortest[1:]
        if phase == A_NEXT:
            return item + (b"," + item) * max(0, node.min_items - self.count - 1) + b"]"
        if phase == A_AFTER:
            return (b"," + item) * max(0, node.min_items - self.count) + b"]"
        return b""

    def close_result(self) -> Any:
        return None

    def finishable(self) -> bool:
        return False


def _start_frames(node: Node) -> tuple:
    if isinstance(node, UnionNode):
        return tuple(frame for option in node.options for frame in option.starts())
    if node is ANY:
        return (
            _Obj(_FREE_OBJECT),
            _Arr(_FREE_ARRAY),
            _Str(),
            _Num(False),
            _Lit(((b"true", None, 0), (b"false", None, 0), (b"null", None, 0))),
        )
    if isinstance(node, StringNode):
        return (_Str(node.min_len, node.max_len),)
    if isinstance(node, NumberNode):
        return (_Num(node.integer),)
    if isinstance(node, LiteralNode):
        return (_Lit(tuple((option, None, 0) for option in node.options)),)
    if isinstance(node, ObjectNode):
        return (_Obj(node),)
    if isinstance(node, ArrayNode):
        return (_Arr(node),)
    raise TypeError(f"unknown node {type(node).__name__}")


# ------------------------------------------------------------------ matcher

Stack = tuple
States = tuple


def _advance(stack: Stack, b: int, out: list) -> None:
    for outcome in stack[-1].feed(b):
        tag = outcome[0]
        if tag == STAY:
            out.append(stack[:-1] + (outcome[1],))
        elif tag == PUSH:
            if len(stack) >= MAX_DEPTH:
                continue
            base = stack[:-1] + (outcome[1],)
            for child in outcome[2]:
                _advance(base + (child,), b, out)
        else:
            settled = stack[:-2] + (stack[-2].child_done(outcome[1]),)
            if tag == DONE:
                out.append(settled)
            else:
                _advance(settled, b, out)


def step(states: States, data: bytes) -> States:
    """Feed bytes to every live stack; an empty result means the bytes are illegal."""
    for b in data:
        out: list = []
        for stack in states:
            _advance(stack, b, out)
        if not out:
            return ()
        states = tuple(dict.fromkeys(out))
    return states


def accepting(stack: Stack) -> bool:
    while len(stack) > 1 and stack[-1].finishable():
        stack = stack[:-2] + (stack[-2].child_done(None),)
    return len(stack) == 1 and stack[0].phase == 2


def closing(stack: Stack) -> bytes:
    """The shortest bytes that finish this stack into a complete value."""
    parts = []
    while True:
        top = stack[-1]
        parts.append(top.close())
        if len(stack) == 1:
            return b"".join(parts)
        stack = stack[:-2] + (stack[-2].child_done(top.close_result()),)


def initial_states(grammar: Node) -> States:
    return ((_Root(grammar),),)


def matches(schema: Any, text: str | bytes) -> bool:
    """Whether ``text`` is a reply this module could have produced for ``schema``."""
    data = text.encode("utf-8") if isinstance(text, str) else text
    states = step(initial_states(as_grammar(schema)), data)
    return any(accepting(s) for s in states)


# ------------------------------------------------------------- vocabulary


def _byte_decoder() -> dict[str, int]:
    """Inverse of GPT-2's bytes_to_unicode: the alphabet byte-level BPE vocabularies are written in."""
    printable = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    chars = printable[:]
    extra = 0
    for b in range(256):
        if b not in printable:
            printable.append(b)
            chars.append(256 + extra)
            extra += 1
    return {chr(c): b for b, c in zip(printable, chars)}


_BYTE_DECODER = _byte_decoder()
_BYTE_PIECE = re.compile(r"<0x([0-9A-Fa-f]{2})>")


def _decoder_kind(inner: Any) -> str:
    backend = getattr(inner, "backend_tokenizer", None)
    if backend is None:
        return "other"
    try:
        spec = json.loads(backend.decoder.__getstate__())
    except Exception:
        try:
            spec = json.loads(backend.to_str()).get("decoder") or {}
        except Exception:
            return "other"
    kinds: set[str] = set()
    todo = [spec]
    while todo:
        item = todo.pop()
        if isinstance(item, dict):
            if isinstance(item.get("type"), str):
                kinds.add(item["type"])
            todo.extend(item.values())
        elif isinstance(item, list):
            todo.extend(item)
    if "ByteLevel" in kinds:
        return "byte_level"
    if kinds & {"ByteFallback", "Metaspace", "Replace"}:
        return "sentencepiece"
    return "other"


def token_byte_table(tokenizer: Any, vocab_size: int) -> list[bytes | None]:
    """The bytes each token id writes, or None for ids that may never appear in a reply.

    Special and added tokens are None: a reply is plain JSON text, and the end of
    it is signalled with EOS, which the constraint handles on its own.
    """
    from engine.tokenizer import _ByteTokenizer

    inner = getattr(tokenizer, "_tok", tokenizer)
    if isinstance(inner, _ByteTokenizer):
        return [bytes((i % 256,)) for i in range(vocab_size)]
    n = min(vocab_size, len(inner))
    pieces = inner.convert_ids_to_tokens(list(range(n)))
    banned = set(getattr(inner, "all_special_ids", None) or ())
    banned |= set((getattr(inner, "added_tokens_decoder", None) or {}).keys())
    kind = _decoder_kind(inner)
    table: list[bytes | None] = [None] * vocab_size
    for i, piece in enumerate(pieces):
        if i in banned or not piece:
            continue
        if kind == "byte_level":
            try:
                table[i] = bytes(_BYTE_DECODER[c] for c in piece)
            except KeyError:
                continue
        elif kind == "sentencepiece":
            byte = _BYTE_PIECE.fullmatch(piece)
            table[i] = bytes((int(byte.group(1), 16),)) if byte else piece.replace("\u2581", " ").encode("utf-8")
        else:
            text = inner.decode([i], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            table[i] = text.encode("utf-8")
    return [t or None for t in table]


MAX_PIECE = 16  # longest token considered when spelling an ending; longer only lowers the count


class TokenVocab:
    """Token id → bytes, plus each token's first byte as a device tensor for masking."""

    def __init__(self, token_bytes: list[bytes | None], eos_ids: Iterable[int]) -> None:
        self.eos_ids = frozenset(int(i) for i in eos_ids if 0 <= int(i) < len(token_bytes))
        self.token_bytes = [None if i in self.eos_ids else t for i, t in enumerate(token_bytes)]
        self._first = [t[0] if t else 256 for t in self.token_bytes]
        self._first_on: dict[str, torch.Tensor] = {}
        self._blank = [bool(t) and all(b in _WS for b in t) for t in self.token_bytes]
        self._blank_on: dict[str, torch.Tensor] = {}
        self._piece_ids: dict[bytes, int] = {}
        for i, t in enumerate(self.token_bytes):
            if t and len(t) <= MAX_PIECE:
                self._piece_ids.setdefault(t, i)
        self._spelled: dict[bytes, tuple[int, int | None]] = {}

    def __len__(self) -> int:
        return len(self.token_bytes)

    def spell(self, data: bytes) -> tuple[int, int | None]:
        """Fewest tokens that write exactly ``data``, and the first of them.

        The count is ``len(data) + 1`` or more when some byte has no token.
        """
        hit = self._spelled.get(data)
        if hit is not None:
            return hit
        n = len(data)
        cost = [0] * (n + 1)
        first: list[int | None] = [None] * (n + 1)
        for i in range(n - 1, -1, -1):
            best, pick = 2 * n + 2, None
            for j in range(i + 1, min(n, i + MAX_PIECE) + 1):
                token_id = self._piece_ids.get(data[i:j])
                if token_id is not None and cost[j] + 1 < best:
                    best, pick = cost[j] + 1, token_id
            cost[i], first[i] = best, pick
        if len(self._spelled) > 4096:
            self._spelled.clear()
        self._spelled[data] = (cost[0], first[0])
        return cost[0], first[0]

    def first_byte(self, device: torch.device) -> torch.Tensor:
        key = str(device)
        if key not in self._first_on:
            self._first_on[key] = torch.tensor(self._first, dtype=torch.long, device=device)
        return self._first_on[key]

    def blank(self, device: torch.device) -> torch.Tensor:
        """Mask of tokens that are nothing but JSON whitespace."""
        key = str(device)
        if key not in self._blank_on:
            self._blank_on[key] = torch.tensor(self._blank, dtype=torch.bool, device=device)
        return self._blank_on[key]


_VOCABS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def vocab_for(tokenizer: Any, vocab_size: int, eos_ids: Iterable[int]) -> TokenVocab:
    """Build (once per tokenizer and vocab size) the byte table the constraint needs."""
    eos = tuple(sorted({int(i) for i in eos_ids}))
    per_tok = _VOCABS.setdefault(tokenizer, {})
    key = (int(vocab_size), eos)
    if key not in per_tok:
        per_tok[key] = TokenVocab(token_byte_table(tokenizer, vocab_size), eos)
    return per_tok[key]


# --------------------------------------------------------------- decoding

_CANDIDATES = 64
_SAMPLE_TRIES = 32


class JsonConstraint:
    """One reply's worth of matcher state, and the rule for choosing each token."""

    def __init__(self, grammar: Node, vocab: TokenVocab) -> None:
        self.grammar = grammar
        self.vocab = vocab
        self.states: States = initial_states(grammar)
        self._first_masks: dict[States, torch.Tensor] = {}
        self._mask_tensors: dict[tuple, torch.Tensor] = {}

    @property
    def accepting(self) -> bool:
        return any(accepting(s) for s in self.states)

    @property
    def done(self) -> bool:
        """The value is complete and nothing more may follow it."""
        return all(len(s) == 1 and s[0].phase == 2 for s in self.states)

    def closing(self, states: States | None = None) -> bytes:
        return min((closing(s) for s in (states or self.states)), key=len)

    def _fits(self, states: States, room: int) -> bool:
        """Whether ``room`` tokens can still write the shortest ending of ``states``."""
        ending = self.closing(states)
        # One token per byte always works, so the token count only matters when that does not.
        return len(ending) <= room or self.vocab.spell(ending)[0] <= room

    def advance(self, token_id: int) -> None:
        if token_id in self.vocab.eos_ids:
            if not self.accepting:
                raise StructuredOutputError("EOS before the JSON value was complete")
            return
        data = self.vocab.token_bytes[token_id] if 0 <= token_id < len(self.vocab) else None
        nxt = step(self.states, data) if data else ()
        if not nxt:
            raise StructuredOutputError(f"token {token_id} ({data!r}) breaks the schema")
        self.states = nxt

    def _legal(self, token_id: int, room: int, can_end: bool) -> bool:
        if token_id in self.vocab.eos_ids:
            return can_end
        data = self.vocab.token_bytes[token_id] if 0 <= token_id < len(self.vocab) else None
        if not data:
            return False
        nxt = step(self.states, data)
        return bool(nxt) and self._fits(nxt, room)

    def _first_byte_mask(self, device: torch.device) -> torch.Tensor:
        """Vocabulary mask of tokens whose first byte can come next (a superset of the legal ones)."""
        mask = self._first_masks.get(self.states)
        if mask is None:
            allowed = tuple(bool(step(self.states, bytes((b,)))) for b in range(256)) + (False,)
            table = self._mask_tensors.get(allowed)
            if table is None:
                table = self._mask_tensors[allowed] = torch.tensor(allowed, device=device)
            mask = table[self.vocab.first_byte(device)]
            if len(self._first_masks) > 256:
                self._first_masks.clear()
            self._first_masks[self.states] = mask
        return mask

    def pick(
        self,
        logits: torch.Tensor,
        params: SamplingParams | None = None,
        generator: torch.Generator | None = None,
        *,
        remaining: int,
    ) -> int:
        """Choose the next token from a 1-D logit row, taking only legal ones.

        ``remaining`` counts the tokens left including this one. A token is legal
        when the matcher survives its bytes and the tokens left after it can
        still spell the shortest ending. If the budget is already shorter than
        that, every token has to bring the end closer.
        """
        params = params or SamplingParams()
        row = logits.float()
        vocab_n = min(row.shape[-1], len(self.vocab))
        row = row[:vocab_n]
        can_end = self.accepting
        ending = self.closing()
        room = remaining - 1
        tight = len(ending) > room
        if tight:
            room = max(room, self.vocab.spell(ending)[0] - 1)
        tried: set[int] = set()

        def ok(token_id: int) -> bool:
            if token_id in tried:
                return False
            tried.add(token_id)
            return self._legal(token_id, room, can_end)

        wants_end = False
        if params.greedy:
            first = int(torch.argmax(row).item())
            if ok(first):
                return first
            wants_end = first in self.vocab.eos_ids
        else:
            choice = self._sample(row, params, generator, ok, can_end)
            if choice is not None:
                return choice
        return self._scan(row, ok, can_end, ending if tight else None, skip_blank=wants_end)

    def _allowed(self, device: torch.device, can_end: bool, n: int) -> torch.Tensor:
        mask = self._first_byte_mask(device)[:n]
        if can_end and self.vocab.eos_ids:
            mask = mask.clone()
            for eos in self.vocab.eos_ids:
                if eos < n:
                    mask[eos] = True
        return mask

    def _scan(
        self, row: torch.Tensor, ok, can_end: bool, tight_ending: bytes | None, *, skip_blank: bool = False
    ) -> int:
        """Walk the ranking, best first, until a token is legal.

        Short on budget (``tight_ending`` set), only the top candidates get a
        look before the reply takes the cheapest spelling of its ending.
        ``skip_blank`` is for a model that asked to stop early: whitespace would
        only pad the reply, so its best real continuation goes first.
        """
        masked = row.masked_fill(~self._allowed(row.device, can_end, row.shape[0]), float("-inf"))
        if skip_blank:
            masked = masked.masked_fill(self.vocab.blank(row.device)[: masked.shape[0]], float("-inf"))
        k = min(_CANDIDATES, masked.shape[0])
        values, ids = torch.topk(masked, k)
        for value, token_id in zip(values.tolist(), ids.tolist()):
            if value == float("-inf"):
                break
            if ok(token_id):
                return token_id
        if tight_ending is not None:
            if not tight_ending and can_end and self.vocab.eos_ids:
                return min(self.vocab.eos_ids)
            first = self.vocab.spell(tight_ending)[1]
            if first is not None:
                return first
        order = torch.argsort(masked, descending=True).tolist()
        finite = torch.isfinite(masked).tolist()
        for token_id in order:
            if not finite[token_id]:
                break
            if ok(token_id):
                return token_id
        raise StructuredOutputError("no token in the vocabulary can continue this reply")

    def _sample(self, row: torch.Tensor, params: SamplingParams, generator, ok, can_end: bool) -> int | None:
        """Draw from the sampling distribution, discarding illegal draws.

        Rejecting a draw and drawing again from what is left samples exactly the
        filtered distribution restricted to legal tokens. If top-k / top-p left no
        legal token at all, the caller falls back to the best legal one.
        """
        probs = torch.softmax(filtered_logits(row, params), dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0)
        masked_once = False
        for _ in range(_SAMPLE_TRIES):
            if float(probs.sum()) <= 0.0:
                return None
            token_id = int(torch.multinomial(probs, 1, generator=generator).item())
            if ok(token_id):
                return token_id
            if token_id in self.vocab.eos_ids:
                probs = probs.masked_fill(self.vocab.blank(row.device)[: probs.shape[0]], 0.0)
            if not masked_once:
                probs = probs * self._allowed(row.device, can_end, probs.shape[0]).to(probs.dtype)
                masked_once = True
            probs[token_id] = 0.0
        return None


def json_constraint(tokenizer: Any, schema: Any, *, vocab_size: int, eos_ids: Iterable[int]) -> JsonConstraint:
    """A fresh constraint for one reply. Compiled schemas and byte tables are cached."""
    return JsonConstraint(as_grammar(schema), vocab_for(tokenizer, vocab_size, eos_ids))


__all__ = [
    "JsonConstraint",
    "SchemaError",
    "StructuredOutputError",
    "SUPPORTED_KEYWORDS",
    "TokenVocab",
    "as_grammar",
    "compile_schema",
    "json_constraint",
    "matches",
    "token_byte_table",
    "vocab_for",
]
