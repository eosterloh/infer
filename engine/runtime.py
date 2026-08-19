"""Agent-facing plug-and-play API.

Point at a model directory → inspect capabilities → load → generate.
Designed so an agent can discover Mamba / MoE / MTP / quant from config
instead of hard-coding architecture switches.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch

from engine.cache import RuntimeState
from engine.capabilities import describe_capabilities
from engine.config import ModelConfig
from engine.generate import generate_greedy
from engine.model import DecoderModel
from engine.tokenizer import Tokenizer
from engine.weights import count_params, load_weights


@dataclass
class EngineHandle:
    """Loaded engine ready for agents to run."""

    model_dir: Path
    config: ModelConfig
    model: DecoderModel
    tokenizer: Tokenizer
    capabilities: dict[str, Any]

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 32,
        use_cache: bool = True,
    ) -> str:
        pieces = list(
            generate_greedy(
                self.model,
                self.tokenizer,
                prompt,
                max_new_tokens=max_new_tokens,
                use_cache=use_cache,
            )
        )
        return "".join(pieces)

    def stream(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 32,
        use_cache: bool = True,
    ) -> Iterator[str]:
        yield from generate_greedy(
            self.model,
            self.tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            use_cache=use_cache,
        )

    def num_params(self) -> int:
        return self.model.num_params()


def inspect_model(model_dir: str | Path) -> dict[str, Any]:
    """Read config only — no weight load. Safe for agents to probe first."""
    model_dir = Path(model_dir).expanduser().resolve()
    config = ModelConfig.from_pretrained(model_dir)
    caps = describe_capabilities(config)
    caps["model_dir"] = str(model_dir)
    caps["config_summary"] = config.summary()
    return caps


def load_engine(
    model_dir: str | Path,
    *,
    device: str | None = None,
    dtype: str | None = None,
) -> EngineHandle:
    """Load weights + tokenizer; return a handle agents can generate with."""
    model_dir = Path(model_dir).expanduser().resolve()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    config = ModelConfig.from_pretrained(model_dir)
    weights = load_weights(model_dir, config, device=device, dtype=dtype)
    model = DecoderModel(config, weights)
    tokenizer = Tokenizer.from_pretrained(model_dir)
    caps = describe_capabilities(config)
    caps["params"] = count_params(weights)
    caps["device"] = str(model.device)
    caps["loaded_dtype"] = str(model.dtype)
    return EngineHandle(
        model_dir=model_dir,
        config=config,
        model=model,
        tokenizer=tokenizer,
        capabilities=caps,
    )


def new_runtime_state(
    config: ModelConfig,
    *,
    batch_size: int = 1,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> RuntimeState:
    """Allocate hybrid KV + Mamba state for a generate session."""
    return RuntimeState(
        config, batch_size=batch_size, device=device, dtype=dtype
    )
