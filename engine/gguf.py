"""Minimal GGUF reader — F32/F16/BF16/Q8_0/Q4_0 → engine float tensors.

A folder with a .gguf (and optional config.json) is a valid drop-in, same as
safetensors. Quantized types dequant on load.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import torch

from engine.maps import map_gguf_name

GGUF_MAGIC = b"GGUF"
GGUF_F32 = 0
GGUF_F16 = 1
GGUF_Q4_0 = 2
GGUF_Q8_0 = 8
GGUF_BF16 = 30

_GGUF_TYPE_SIZE = {
    0: 4,  # UINT8
    1: 2,  # INT8  (we treat as 1-byte later via type id)
}

# GGUF metadata value types
_U8, _I8, _U16, _I16, _U32, _I32, _F32, _BOOL, _STR, _ARRAY, _U64, _I64, _F64 = range(13)


@dataclass
class GgufTensor:
    name: str
    dims: tuple[int, ...]
    ggml_type: int
    offset: int


def find_gguf(model_dir: str | Path) -> Path | None:
    path = Path(model_dir)
    if path.is_file() and path.suffix.lower() == ".gguf":
        return path
    if path.is_dir():
        files = sorted(path.glob("*.gguf"))
        return files[0] if files else None
    return None


def _read_str(f: BinaryIO) -> str:
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n).decode("utf-8", errors="replace")


def _read_value(f: BinaryIO, typ: int) -> Any:
    if typ == _U8:
        return struct.unpack("<B", f.read(1))[0]
    if typ == _I8:
        return struct.unpack("<b", f.read(1))[0]
    if typ == _U16:
        return struct.unpack("<H", f.read(2))[0]
    if typ == _I16:
        return struct.unpack("<h", f.read(2))[0]
    if typ == _U32:
        return struct.unpack("<I", f.read(4))[0]
    if typ == _I32:
        return struct.unpack("<i", f.read(4))[0]
    if typ == _F32:
        return struct.unpack("<f", f.read(4))[0]
    if typ == _BOOL:
        return bool(struct.unpack("<B", f.read(1))[0])
    if typ == _STR:
        return _read_str(f)
    if typ == _U64:
        return struct.unpack("<Q", f.read(8))[0]
    if typ == _I64:
        return struct.unpack("<q", f.read(8))[0]
    if typ == _F64:
        return struct.unpack("<d", f.read(8))[0]
    if typ == _ARRAY:
        (elem,) = struct.unpack("<I", f.read(4))
        (count,) = struct.unpack("<Q", f.read(8))
        return [_read_value(f, elem) for _ in range(count)]
    raise ValueError(f"unsupported GGUF value type {typ}")


def read_gguf_header(path: Path) -> tuple[dict[str, Any], list[GgufTensor], int]:
    with path.open("rb") as f:
        magic = f.read(4)
        if magic != GGUF_MAGIC:
            raise ValueError(f"{path} is not GGUF (magic={magic!r})")
        (version,) = struct.unpack("<I", f.read(4))
        if version < 2:
            raise ValueError(f"GGUF version {version} not supported")
        n_tensors, n_kv = struct.unpack("<QQ", f.read(16))
        meta: dict[str, Any] = {}
        for _ in range(n_kv):
            key = _read_str(f)
            (typ,) = struct.unpack("<I", f.read(4))
            meta[key] = _read_value(f, typ)
        tensors: list[GgufTensor] = []
        for _ in range(n_tensors):
            name = _read_str(f)
            (n_dims,) = struct.unpack("<I", f.read(4))
            dims = tuple(struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims))
            (ggml_type,) = struct.unpack("<I", f.read(4))
            (offset,) = struct.unpack("<Q", f.read(8))
            tensors.append(GgufTensor(name, dims, ggml_type, offset))
        data_start = f.tell()
        alignment = int(meta.get("general.alignment", 32))
        data_start = (data_start + alignment - 1) // alignment * alignment
        return meta, tensors, data_start


def gguf_meta_to_raw(meta: dict[str, Any]) -> dict[str, Any]:
    """Build a config.json-like dict from GGUF general.* / llama.* keys."""
    arch = str(meta.get("general.architecture") or "llama")
    p = f"{arch}."

    def g(*keys: str, default: Any = None) -> Any:
        for k in keys:
            if k in meta:
                return meta[k]
            if p + k in meta:
                return meta[p + k]
            if f"{arch}.{k}" in meta:
                return meta[f"{arch}.{k}"]
        return default

    n_head = int(g("attention.head_count", default=8))
    n_embd = int(g("embedding_length", default=n_head * 64))
    n_kv = int(g("attention.head_count_kv", default=n_head))
    head_dim = n_embd // n_head if n_head else 64
    return {
        "model_type": arch,
        "architectures": [f"{arch.capitalize()}ForCausalLM"],
        "vocab_size": int(g("vocab_size", default=meta.get("tokenizer.ggml.tokens") and len(meta["tokenizer.ggml.tokens"]) or 32000)),
        "hidden_size": n_embd,
        "intermediate_size": int(g("feed_forward_length", default=4 * n_embd)),
        "num_hidden_layers": int(g("block_count", default=1)),
        "num_attention_heads": n_head,
        "num_key_value_heads": n_kv,
        "head_dim": int(g("attention.key_length", default=head_dim)),
        "rms_norm_eps": float(g("attention.layer_norm_rms_epsilon", default=1e-5)),
        "rope_theta": float(g("rope.freq_base", default=10000.0)),
        "max_position_embeddings": int(g("context_length", default=2048)),
        "tie_word_embeddings": "output.weight" not in {
            # filled later; default tied if no output.weight tensor
        },
        "torch_dtype": "float32",
        "hidden_act": "silu",
    }


def _block_size(ggml_type: int, n_elem: int) -> int:
    if ggml_type == GGUF_F32:
        return n_elem * 4
    if ggml_type in {GGUF_F16, GGUF_BF16}:
        return n_elem * 2
    if ggml_type == GGUF_Q8_0:
        # QK=32, block = 2 (fp16 scale) + 32 int8
        n_blocks = (n_elem + 31) // 32
        return n_blocks * 34
    if ggml_type == GGUF_Q4_0:
        # QK=32, block = 2 (fp16 scale) + 16 bytes qs
        n_blocks = (n_elem + 31) // 32
        return n_blocks * 18
    raise ValueError(f"unsupported GGUF ggml type {ggml_type}")


def _dequant_q8_0(blob: bytes, n_elem: int) -> torch.Tensor:
    n_blocks = (n_elem + 31) // 32
    out = torch.empty(n_blocks * 32, dtype=torch.float32)
    off = 0
    for b in range(n_blocks):
        scale = struct.unpack_from("<e", blob, off)[0]
        off += 2
        qs = torch.tensor(list(blob[off : off + 32]), dtype=torch.int8).float()
        off += 32
        out[b * 32 : (b + 1) * 32] = qs * float(scale)
    return out[:n_elem]


def _dequant_q4_0(blob: bytes, n_elem: int) -> torch.Tensor:
    n_blocks = (n_elem + 31) // 32
    out = torch.empty(n_blocks * 32, dtype=torch.float32)
    off = 0
    for b in range(n_blocks):
        scale = struct.unpack_from("<e", blob, off)[0]
        off += 2
        qs = blob[off : off + 16]
        off += 16
        vals = []
        for byte in qs:
            vals.append((byte & 0x0F) - 8)
            vals.append((byte >> 4) - 8)
        out[b * 32 : (b + 1) * 32] = torch.tensor(vals, dtype=torch.float32) * float(scale)
    return out[:n_elem]


def load_gguf_state(path: Path) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Return (meta, engine_name → float tensor). Skips unmapped aux tensors."""
    meta, tensors, data_start = read_gguf_header(path)
    state: dict[str, torch.Tensor] = {}
    with path.open("rb") as f:
        for info in tensors:
            engine = map_gguf_name(info.name)
            if engine is None:
                continue
            n_elem = 1
            for d in info.dims:
                n_elem *= int(d)
            nbytes = _block_size(info.ggml_type, n_elem)
            f.seek(data_start + info.offset)
            blob = f.read(nbytes)
            if info.ggml_type == GGUF_F32:
                t = torch.frombuffer(bytearray(blob), dtype=torch.float32).clone()
            elif info.ggml_type == GGUF_F16:
                t = torch.frombuffer(bytearray(blob), dtype=torch.float16).float().clone()
            elif info.ggml_type == GGUF_BF16:
                t = torch.frombuffer(bytearray(blob), dtype=torch.bfloat16).float().clone()
            elif info.ggml_type == GGUF_Q8_0:
                t = _dequant_q8_0(blob, n_elem)
            elif info.ggml_type == GGUF_Q4_0:
                t = _dequant_q4_0(blob, n_elem)
            else:
                raise ValueError(f"unsupported GGUF type {info.ggml_type} for {info.name}")
            # GGUF stores dims reversed vs PyTorch.
            shape = tuple(reversed(info.dims)) if info.dims else (n_elem,)
            t = t[:n_elem].view(*shape)
            state[engine] = t
    meta["_gguf_has_output"] = "lm_head.weight" in state
    return meta, state


def write_gguf_f32(path: Path, tensors: dict[str, torch.Tensor], meta: dict[str, Any]) -> None:
    """Write a tiny F32 GGUF (tests). `tensors` keyed by GGUF names."""
    kv_items = list(meta.items())
    # header + kv + tensor infos, then aligned data
    buf = bytearray()
    buf += GGUF_MAGIC
    buf += struct.pack("<I", 3)
    buf += struct.pack("<QQ", len(tensors), len(kv_items))

    def put_str(b: bytearray, s: str) -> None:
        raw = s.encode("utf-8")
        b += struct.pack("<Q", len(raw))
        b += raw

    def put_value(b: bytearray, value: Any) -> None:
        if isinstance(value, bool):
            b += struct.pack("<I", _BOOL)
            b += struct.pack("<B", int(value))
        elif isinstance(value, int):
            b += struct.pack("<I", _U32)
            b += struct.pack("<I", value)
        elif isinstance(value, float):
            b += struct.pack("<I", _F32)
            b += struct.pack("<f", value)
        elif isinstance(value, str):
            b += struct.pack("<I", _STR)
            put_str(b, value)
        else:
            raise TypeError(type(value))

    for k, v in kv_items:
        put_str(buf, k)
        put_value(buf, v)

    blobs: list[bytes] = []
    infos: list[tuple[str, tuple[int, ...], int]] = []
    offset = 0
    alignment = 32
    for name, tensor in tensors.items():
        t = tensor.detach().float().contiguous()
        # GGUF dims are reversed
        dims = tuple(reversed(t.shape))
        payload = t.numpy().tobytes()
        pad = (alignment - (offset % alignment)) % alignment
        offset += pad
        infos.append((name, dims, offset))
        blobs.append((b"\x00" * pad) + payload)
        offset += len(payload)

    for name, dims, off in infos:
        put_str(buf, name)
        buf += struct.pack("<I", len(dims))
        for d in dims:
            buf += struct.pack("<Q", int(d))
        buf += struct.pack("<I", GGUF_F32)
        buf += struct.pack("<Q", off)

    alignment = 32
    pad = (alignment - (len(buf) % alignment)) % alignment
    buf += b"\x00" * pad
    for blob in blobs:
        buf += blob
    path.write_bytes(bytes(buf))
