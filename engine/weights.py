"""Load safetensors, rename HF keys → engine keys, cast, place on device."""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from safetensors.torch import load_file

from engine.config import ModelConfig


def _map_llama_hf_name(hf_name: str) -> str | None:
    """Map one Llama-style HF tensor name to our engine name."""
    if hf_name == "model.embed_tokens.weight":
        return "embed.weight"
    if hf_name == "model.norm.weight":
        return "final_norm.weight"
    if hf_name == "lm_head.weight":
        return "lm_head.weight"

    m = re.match(r"^model\.layers\.(\d+)\.(.+)$", hf_name)
    if not m:
        return None
    i, rest = m.group(1), m.group(2)
    table = {
        "self_attn.q_proj.weight": f"layers.{i}.attn.q.weight",
        "self_attn.k_proj.weight": f"layers.{i}.attn.k.weight",
        "self_attn.v_proj.weight": f"layers.{i}.attn.v.weight",
        "self_attn.o_proj.weight": f"layers.{i}.attn.o.weight",
        "mlp.gate_proj.weight": f"layers.{i}.mlp.gate.weight",
        "mlp.up_proj.weight": f"layers.{i}.mlp.up.weight",
        "mlp.down_proj.weight": f"layers.{i}.mlp.down.weight",
        "input_layernorm.weight": f"layers.{i}.input_norm.weight",
        "post_attention_layernorm.weight": f"layers.{i}.post_attn_norm.weight",
    }
    return table.get(rest)


def _map_nemotron_h_hf_name(hf_name: str) -> str | None:
    """Map Nemotron-H (Nano 3 / Super family) HF names → engine names."""
    if hf_name == "backbone.embeddings.weight":
        return "embed.weight"
    if hf_name == "backbone.norm_f.weight":
        return "final_norm.weight"
    if hf_name == "lm_head.weight":
        return "lm_head.weight"

    m = re.match(r"^backbone\.layers\.(\d+)\.(.+)$", hf_name)
    if not m:
        return None
    i, rest = m.group(1), m.group(2)

    if rest == "norm.weight":
        return f"layers.{i}.input_norm.weight"

    # Attention mixer
    attn = {
        "mixer.q_proj.weight": f"layers.{i}.attn.q.weight",
        "mixer.k_proj.weight": f"layers.{i}.attn.k.weight",
        "mixer.v_proj.weight": f"layers.{i}.attn.v.weight",
        "mixer.o_proj.weight": f"layers.{i}.attn.o.weight",
    }
    if rest in attn:
        return attn[rest]

    # Dense MLP mixer (pattern '-')
    mlp = {
        "mixer.up_proj.weight": f"layers.{i}.mlp.up.weight",
        "mixer.down_proj.weight": f"layers.{i}.mlp.down.weight",
    }
    if rest in mlp:
        return mlp[rest]

    # MoE
    if rest == "mixer.gate.weight":
        return f"layers.{i}.moe.gate.weight"
    if rest == "mixer.gate.e_score_correction_bias":
        return f"layers.{i}.moe.gate.e_score_correction_bias"
    if rest == "mixer.shared_experts.up_proj.weight":
        return f"layers.{i}.moe.shared.up.weight"
    if rest == "mixer.shared_experts.down_proj.weight":
        return f"layers.{i}.moe.shared.down.weight"
    em = re.match(r"^mixer\.experts\.(\d+)\.(up_proj|down_proj)\.weight$", rest)
    if em:
        e, which = em.group(1), em.group(2)
        short = "up" if which == "up_proj" else "down"
        return f"layers.{i}.moe.experts.{e}.{short}.weight"

    # Mamba-2 mixer
    mamba = {
        "mixer.in_proj.weight": f"layers.{i}.mamba.in_proj.weight",
        "mixer.out_proj.weight": f"layers.{i}.mamba.out_proj.weight",
        "mixer.conv1d.weight": f"layers.{i}.mamba.conv1d.weight",
        "mixer.conv1d.bias": f"layers.{i}.mamba.conv1d.bias",
        "mixer.A_log": f"layers.{i}.mamba.A_log",
        "mixer.D": f"layers.{i}.mamba.D",
        "mixer.dt_bias": f"layers.{i}.mamba.dt_bias",
        "mixer.norm.weight": f"layers.{i}.mamba.norm.weight",
    }
    if rest in mamba:
        return mamba[rest]

    return None


def map_hf_name(hf_name: str, config: ModelConfig | None = None) -> str | None:
    """Dispatch HF→engine rename from auto-detected recipe."""
    recipe = getattr(config, "recipe_id", "") if config else ""
    if not recipe and config is not None:
        from engine.detect import detect_recipe_id

        recipe = detect_recipe_id(config.raw or {"model_type": config.model_type})
    if recipe == "nemotron_h" or (hf_name.startswith("backbone.") and recipe != "llama"):
        return _map_nemotron_h_hf_name(hf_name)
    return _map_llama_hf_name(hf_name)


# Backward-compatible private name used in older call sites / tests.
_map_hf_name = _map_llama_hf_name


def _resolve_dtype(name: str) -> torch.dtype:
    key = name.lower().replace("torch.", "")
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float": torch.float32,
    }
    if key not in mapping:
        raise ValueError(f"unsupported torch_dtype={name!r}")
    return mapping[key]


def _shard_paths(model_dir: Path) -> list[Path]:
    index_path = model_dir / "model.safetensors.index.json"
    single = model_dir / "model.safetensors"
    if index_path.is_file():
        with index_path.open("r", encoding="utf-8") as f:
            index = json.load(f)
        weight_map: dict[str, str] = index["weight_map"]
        return sorted({model_dir / shard for shard in weight_map.values()})
    if single.is_file():
        return [single]
    shards = sorted(model_dir.glob("model-*.safetensors"))
    if shards:
        return shards
    raise FileNotFoundError(
        f"no safetensors weights under {model_dir} "
        "(expected model.safetensors or sharded model-*.safetensors)"
    )


def load_weight_index(model_dir: str | Path) -> dict[str, str]:
    """HF name → shard filename (no tensor load)."""
    model_dir = Path(model_dir)
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        with index_path.open("r", encoding="utf-8") as f:
            return dict(json.load(f)["weight_map"])
    single = model_dir / "model.safetensors"
    if single.is_file():
        from safetensors import safe_open

        with safe_open(str(single), framework="pt") as f:
            return {k: single.name for k in f.keys()}
    raise FileNotFoundError(f"no weight index under {model_dir}")


def validate_name_map(
    config: ModelConfig, hf_names: list[str] | set[str]
) -> dict[str, str]:
    """Map all HF names; ensure they cover expected_shapes exactly.

    Returns engine_name → hf_name. Raises on unknown / missing / extra keys.
    """
    mapped: dict[str, str] = {}
    unknown: list[str] = []
    for hf_name in sorted(hf_names):
        engine_name = map_hf_name(hf_name, config)
        if engine_name is None:
            unknown.append(hf_name)
            continue
        if engine_name in mapped:
            raise RuntimeError(
                f"name map collision on {engine_name}: "
                f"{mapped[engine_name]} vs {hf_name}"
            )
        mapped[engine_name] = hf_name

    if unknown:
        preview = ", ".join(unknown[:8])
        more = f" (+{len(unknown) - 8} more)" if len(unknown) > 8 else ""
        raise KeyError(f"unmapped HF tensors: {preview}{more}")

    expected = set(config.expected_shapes())
    got = set(mapped)
    if config.tie_word_embeddings:
        expected.discard("lm_head.weight")

    missing = sorted(expected - got)
    extra = sorted(got - expected)
    if missing:
        raise KeyError(f"missing weights after map: {missing[:12]}")
    if extra:
        raise KeyError(f"unexpected weights after map: {extra[:12]}")
    return mapped


def load_hf_state_cpu(model_dir: str | Path) -> dict[str, torch.Tensor]:
    """Deserialize all shards → CPU tensors still keyed by HF names."""
    model_dir = Path(model_dir)
    state: dict[str, torch.Tensor] = {}
    for shard in _shard_paths(model_dir):
        piece = load_file(str(shard), device="cpu")
        overlap = set(state) & set(piece)
        if overlap:
            raise RuntimeError(f"duplicate tensors across shards: {sorted(overlap)[:5]}")
        state.update(piece)
    return state


def rename_state(
    hf_state: dict[str, torch.Tensor],
    config: ModelConfig | None = None,
) -> dict[str, torch.Tensor]:
    """Apply HF → engine name map. Errors on unknown or colliding keys."""
    out: dict[str, torch.Tensor] = {}
    unknown: list[str] = []
    for hf_name, tensor in hf_state.items():
        engine_name = map_hf_name(hf_name, config)
        if engine_name is None:
            unknown.append(hf_name)
            continue
        if engine_name in out:
            raise RuntimeError(f"name map collision on {engine_name}")
        out[engine_name] = tensor
    if unknown:
        preview = ", ".join(unknown[:8])
        more = f" (+{len(unknown) - 8} more)" if len(unknown) > 8 else ""
        raise KeyError(f"unmapped HF tensors: {preview}{more}")
    return out


def validate_shapes(config: ModelConfig, state: dict[str, torch.Tensor]) -> None:
    expected = config.expected_shapes()
    missing = sorted(set(expected) - set(state))
    extra = sorted(set(state) - set(expected))

    if config.tie_word_embeddings and "lm_head.weight" in missing:
        missing = [m for m in missing if m != "lm_head.weight"]
    if config.tie_word_embeddings and "lm_head.weight" in extra:
        extra = [e for e in extra if e != "lm_head.weight"]

    if missing:
        raise KeyError(f"missing weights after map: {missing[:12]}")
    if extra:
        raise KeyError(f"unexpected weights after map: {extra[:12]}")

    bad: list[str] = []
    for name, want in expected.items():
        if name not in state:
            continue
        got = tuple(state[name].shape)
        if got != want:
            bad.append(f"{name}: got {got}, want {want}")
    if config.tie_word_embeddings and "lm_head.weight" in state:
        if tuple(state["lm_head.weight"].shape) != tuple(state["embed.weight"].shape):
            bad.append("lm_head.weight shape != embed.weight (tie_word_embeddings)")
    if bad:
        raise ValueError("shape mismatches:\n  " + "\n  ".join(bad[:20]))


def apply_tied_embeddings(
    config: ModelConfig, state: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Ensure lm_head exists; share storage with embed when tied."""
    if config.tie_word_embeddings:
        state = dict(state)
        state["lm_head.weight"] = state["embed.weight"]
    elif "lm_head.weight" not in state:
        raise KeyError("lm_head.weight missing and tie_word_embeddings is false")
    return state


def cast_and_to_device(
    state: dict[str, torch.Tensor],
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Move tensors to device/dtype; preserve embed↔lm_head tying by shared storage."""
    tied = (
        "lm_head.weight" in state
        and state["lm_head.weight"].data_ptr() == state["embed.weight"].data_ptr()
    )
    out: dict[str, torch.Tensor] = {}
    for name, tensor in state.items():
        if tied and name == "lm_head.weight":
            continue
        # Keep router bias / gate weights that are float32 in the checkpoint.
        if tensor.dtype == torch.float32 and name.endswith(
            ("e_score_correction_bias", "moe.gate.weight")
        ):
            out[name] = tensor.to(device=device)
        else:
            out[name] = tensor.to(device=device, dtype=dtype)
    if tied:
        out["lm_head.weight"] = out["embed.weight"]
    elif "lm_head.weight" in state and "lm_head.weight" not in out:
        out["lm_head.weight"] = state["lm_head.weight"].to(device=device, dtype=dtype)
    return out


def count_params(state: dict[str, torch.Tensor]) -> int:
    """Count unique parameter storage (tied lm_head not double-counted)."""
    seen: set[int] = set()
    total = 0
    for tensor in state.values():
        ptr = tensor.data_ptr()
        if ptr in seen:
            continue
        seen.add(ptr)
        total += tensor.numel()
    return total


def _cast_one(
    name: str, tensor: torch.Tensor, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    if tensor.dtype == torch.float32 and name.endswith(
        ("e_score_correction_bias", "moe.gate.weight")
    ):
        return tensor.to(device=device)
    return tensor.to(device=device, dtype=dtype)


def load_weights(
    model_dir: str | Path,
    config: ModelConfig,
    device: str | torch.device = "cuda",
    dtype: str | torch.dtype | None = None,
) -> dict[str, torch.Tensor]:
    """Deserialize shard-by-shard onto device so we do not hold a CPU+GPU copy."""
    model_dir = Path(model_dir)
    device = torch.device(device)
    if dtype is None:
        dtype = _resolve_dtype(config.torch_dtype)
    elif isinstance(dtype, str):
        dtype = _resolve_dtype(dtype)

    index = load_weight_index(model_dir)
    validate_name_map(config, index.keys())

    shards = _shard_paths(model_dir)
    state: dict[str, torch.Tensor] = {}
    for i, shard in enumerate(shards, start=1):
        print(f"  shard {i}/{len(shards)} {shard.name}", flush=True)
        piece = load_file(str(shard), device="cpu")
        for hf_name, tensor in piece.items():
            engine_name = map_hf_name(hf_name, config)
            if engine_name is None:
                raise KeyError(f"unmapped HF tensor during load: {hf_name}")
            if engine_name in state:
                raise RuntimeError(f"name map collision on {engine_name}")
            state[engine_name] = _cast_one(engine_name, tensor, dtype, device)
        del piece

    validate_shapes(config, state)
    state = apply_tied_embeddings(config, state)
    return state
