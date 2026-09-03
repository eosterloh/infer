"""Agent-facing plug-and-play API.

Agents pass a model directory (config.json + weights + tokenizer).
The engine reads config.json, auto-detects the recipe, and generates.

North star: Nemotron NVFP4 on DGX Spark — MTP lands with Super later.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

import torch

from engine.config import ModelConfig
from engine.detect import (
    UnsupportedRecipeError,
    can_run as recipe_can_run,
    detect_missing,
    detect_quant_flags,
    detect_recipe_id,
)
from engine.generate import generate_greedy
from engine.model import DecoderModel
from engine.schedule import FfnKind, MixerKind
from engine.tokenizer import Tokenizer
from engine.weights import count_params, load_weights


@dataclass(frozen=True)
class Capabilities:
    """What this checkpoint can do — agents branch on these flags."""

    recipe_id: str
    model_type: str
    architectures: tuple[str, ...]
    can_run: bool
    missing: tuple[str, ...]
    dense_mlp: bool
    moe: bool
    mamba2: bool
    attention: bool
    rope: bool
    mtp: bool
    nvfp4: bool
    fp8: bool
    hybrid_pattern: str | None
    num_layers: int
    hidden_size: int
    vocab_size: int
    max_position_embeddings: int
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def inspect_capabilities(model_dir: str | Path) -> Capabilities:
    """Parse config only — no weight load. Safe for agents to probe first.

    Raises UnsupportedRecipeError if config.json is not a recipe we implement.
    """
    model_dir = Path(model_dir)
    config = ModelConfig.from_pretrained(model_dir)
    recipe_id = config.recipe_id or detect_recipe_id(config.raw)
    layers = config.layers
    has_attn = any(s.mixer == MixerKind.ATTENTION for s in layers)
    has_mamba = any(s.mixer == MixerKind.MAMBA2 for s in layers)
    has_moe = any(s.ffn == FfnKind.MOE for s in layers)
    has_dense = any(s.ffn == FfnKind.DENSE_MLP for s in layers)
    rope = config.pos_kind == "rope"

    nvfp4, fp8, mtp = detect_quant_flags(config.raw or {}, model_dir.name)
    missing = detect_missing(config.raw or {}, model_dir.name)
    runnable = recipe_can_run(recipe_id, missing)

    notes: list[str] = []
    if has_mamba:
        notes.append("mamba2 scheduled — sequential SSM path")
    if any(s.mixer == MixerKind.GATED_DELTANET for s in layers):
        notes.append("gated deltanet scheduled — sequential linear-attention path")
    if "vision" in missing:
        notes.append("vision advertised — text greedy only; vision tower not wired")
    if has_moe:
        notes.append("moe scheduled — router+shared expert path")
    if (model_dir / "chat_template.jinja").is_file() or (
        (model_dir / "tokenizer_config.json").is_file()
        and "chat_template" in (model_dir / "tokenizer_config.json").read_text(encoding="utf-8", errors="ignore")
    ):
        notes.append("chat template present — generate() wraps raw prompts")
    if (model_dir / "generation_config.json").is_file():
        notes.append("generation_config.json eos/stop ids used automatically")
    if "mtp_decode" in missing:
        notes.append("MTP advertised — greedy still works; speculative decode not wired")
    if nvfp4:
        notes.append("NVFP4 advertised — dequant-on-load (not fused)")
    if fp8:
        notes.append("FP8 advertised — dequant-on-load (not fused)")

    return Capabilities(
        recipe_id=recipe_id,
        model_type=config.model_type,
        architectures=config.architectures,
        can_run=runnable,
        missing=missing,
        dense_mlp=has_dense,
        moe=has_moe,
        mamba2=has_mamba,
        attention=has_attn,
        rope=rope,
        mtp=mtp,
        nvfp4=nvfp4,
        fp8=fp8,
        hybrid_pattern=config.hybrid_override_pattern,
        num_layers=config.num_hidden_layers,
        hidden_size=config.hidden_size,
        vocab_size=config.vocab_size,
        max_position_embeddings=config.max_position_embeddings,
        notes=tuple(notes),
    )


@dataclass
class Engine:
    """Loaded engine handle for agents."""

    model_dir: Path
    config: ModelConfig
    model: DecoderModel
    tokenizer: Tokenizer
    capabilities: Capabilities
    n_params: int

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 32,
        use_cache: bool = True,
        apply_chat_template: bool | None = None,
        enable_thinking: bool = False,
    ) -> str:
        return "".join(
            generate_greedy(
                self.model,
                self.tokenizer,
                prompt,
                max_new_tokens=max_new_tokens,
                use_cache=use_cache,
                apply_chat_template=apply_chat_template,
                enable_thinking=enable_thinking,
            )
        )

    def stream(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 32,
        use_cache: bool = True,
        apply_chat_template: bool | None = None,
        enable_thinking: bool = False,
    ) -> Iterator[str]:
        yield from generate_greedy(
            self.model,
            self.tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            use_cache=use_cache,
            apply_chat_template=apply_chat_template,
            enable_thinking=enable_thinking,
        )

    def info(self) -> dict[str, Any]:
        return {
            "model_dir": str(self.model_dir),
            "params": self.n_params,
            "device": str(self.model.device),
            "dtype": str(self.model.dtype),
            "config": self.config.summary(),
            "capabilities": self.capabilities.to_dict(),
        }


def load_engine(
    model_dir: str | Path,
    *,
    device: str | None = None,
    dtype: str | None = None,
) -> Engine:
    """One-call load for agents: folder → detect recipe → weights → tokenizer."""
    model_dir = Path(model_dir).expanduser().resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(model_dir)

    caps = inspect_capabilities(model_dir)
    if not caps.can_run:
        raise UnsupportedRecipeError(
            f"cannot load {model_dir.name}: recipe={caps.recipe_id} "
            f"missing={list(caps.missing)}. inspect_capabilities() for details."
        )

    config = ModelConfig.from_pretrained(model_dir)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    weights = load_weights(model_dir, config, device=device, dtype=dtype)
    model = DecoderModel(config, weights)
    tokenizer = Tokenizer.from_pretrained(model_dir)
    return Engine(
        model_dir=model_dir,
        config=config,
        model=model,
        tokenizer=tokenizer,
        capabilities=caps,
        n_params=count_params(weights),
    )
