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
from engine.generate import (
    generate_greedy,
    generate_mtp_greedy,
    generate_multimodal_greedy,
)
from engine.model import DecoderModel
from engine.mtp import Qwen35MTP
from engine.schedule import FfnKind, MixerKind
from engine.tokenizer import Tokenizer
from engine.vision import validate_qwen35_vision_weights
from engine.weights import count_params, load_hf_prefixes, load_weights


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
        notes.append("gated deltanet scheduled — chunked CUDA prefill + recurrent decode")
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
    elif mtp and recipe_id == "qwen3_5":
        notes.append("native MTP available — lossless greedy speculative decode")
    if config.raw.get("vision_config") and recipe_id == "qwen3_5":
        notes.append("image/video tower and multimodal M-RoPE available")
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
    vision_weights: dict[str, torch.Tensor] | None = None
    processor: Any | None = None
    mtp: Qwen35MTP | None = None
    last_mtp_stats: dict[str, int] | None = None

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 32,
        use_cache: bool = True,
        apply_chat_template: bool | None = None,
        enable_thinking: bool = False,
        num_speculative_tokens: int = 0,
    ) -> str:
        if num_speculative_tokens:
            if self.mtp is None:
                raise RuntimeError("this engine was not loaded with an MTP head")
            self.last_mtp_stats = {}
            return "".join(
                generate_mtp_greedy(
                    self.model,
                    self.mtp,
                    self.tokenizer,
                    prompt,
                    max_new_tokens=max_new_tokens,
                    num_speculative_tokens=num_speculative_tokens,
                    apply_chat_template=apply_chat_template,
                    enable_thinking=enable_thinking,
                    stats=self.last_mtp_stats,
                )
            )
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
        num_speculative_tokens: int = 0,
    ) -> Iterator[str]:
        if num_speculative_tokens:
            if self.mtp is None:
                raise RuntimeError("this engine was not loaded with an MTP head")
            self.last_mtp_stats = {}
            yield from generate_mtp_greedy(
                self.model,
                self.mtp,
                self.tokenizer,
                prompt,
                max_new_tokens=max_new_tokens,
                num_speculative_tokens=num_speculative_tokens,
                apply_chat_template=apply_chat_template,
                enable_thinking=enable_thinking,
                stats=self.last_mtp_stats,
            )
            return
        yield from generate_greedy(
            self.model,
            self.tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            use_cache=use_cache,
            apply_chat_template=apply_chat_template,
            enable_thinking=enable_thinking,
        )

    def generate_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        max_new_tokens: int = 32,
        enable_thinking: bool = False,
        num_speculative_tokens: int = 0,
    ) -> str:
        """Generate from Qwen processor-compatible text/image/video messages."""
        if self.processor is None or self.vision_weights is None:
            raise RuntimeError("this engine was not loaded with a vision tower")
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=enable_thinking,
        )
        if num_speculative_tokens:
            if self.mtp is None:
                raise RuntimeError("this engine was not loaded with an MTP head")
            self.last_mtp_stats = {}
            return "".join(
                generate_mtp_greedy(
                    self.model,
                    self.mtp,
                    self.tokenizer,
                    "",
                    max_new_tokens=max_new_tokens,
                    num_speculative_tokens=num_speculative_tokens,
                    stats=self.last_mtp_stats,
                    processor_inputs=dict(inputs),
                    vision_weights=self.vision_weights,
                )
            )
        return "".join(
            generate_multimodal_greedy(
                self.model,
                self.tokenizer,
                self.vision_weights,
                dict(inputs),
                max_new_tokens=max_new_tokens,
            )
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
    vision_weights = None
    processor = None
    mtp = None
    if config.recipe_id == "qwen3_5" and isinstance(
        config.raw.get("vision_config"), dict
    ):
        vision_weights = load_hf_prefixes(
            model_dir, ("model.visual.",), device=device, dtype=dtype
        )
        validate_qwen35_vision_weights(
            vision_weights, config.raw["vision_config"]
        )
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(str(model_dir))
    if config.recipe_id == "qwen3_5" and config.num_nextn_predict_layers:
        mtp_weights = load_hf_prefixes(
            model_dir, ("mtp.",), device=device, dtype=dtype
        )
        mtp = Qwen35MTP(config, weights, mtp_weights)
    total_params = count_params(weights)
    if vision_weights:
        total_params += count_params(vision_weights)
    if mtp is not None:
        total_params += count_params(mtp.root) + count_params(mtp.weights)
    return Engine(
        model_dir=model_dir,
        config=config,
        model=model,
        tokenizer=tokenizer,
        capabilities=caps,
        n_params=total_params,
        vision_weights=vision_weights,
        processor=processor,
        mtp=mtp,
    )
