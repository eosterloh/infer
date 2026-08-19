"""Decoder model — scheduled forward with optional runtime state."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from engine.cache import KVCache, RuntimeState
from engine.config import ModelConfig
from engine.layers import build_inv_freq, build_rope_cos_sin, decoder_block, rms_norm
from engine.schedule import MixerKind, build_schedule


class DecoderModel:
    """Loaded weights + config-driven layer schedule; optional KV / hybrid state."""

    def __init__(self, config: ModelConfig, weights: dict[str, torch.Tensor]):
        self.config = config
        self.weights = weights
        sample = next(iter(weights.values()))
        self.device = sample.device
        self.dtype = sample.dtype
        self.layers = config.layers or build_schedule(config)
        # Nemotron-H has attention layers but no positional embeddings / RoPE.
        self.use_rope = (config.recipe_id == "llama") and any(
            s.mixer == MixerKind.ATTENTION for s in self.layers
        )
        self._inv_freq = (
            build_inv_freq(config, device=self.device) if self.use_rope else None
        )

    def make_cache(
        self,
        *,
        batch_size: int = 1,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> KVCache | RuntimeState:
        """Allocate the right runtime state for this architecture."""
        device = device or self.device
        dtype = dtype or self.dtype
        mt = (self.config.recipe_id or self.config.model_type or "").lower()
        if mt in {"nemotron_h", "nemotronh"} or any(
            s.mixer == MixerKind.MAMBA2 for s in self.layers
        ):
            return RuntimeState(
                self.config, batch_size=batch_size, device=device, dtype=dtype
            )
        return KVCache(
            self.config, batch_size=batch_size, device=device, dtype=dtype
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        cache: KVCache | RuntimeState | None = None,
    ) -> torch.Tensor:
        """input_ids: [B, S] → logits [B, S, V]."""
        if input_ids.dim() != 2:
            raise ValueError(f"expected input_ids [B, S], got {tuple(input_ids.shape)}")

        b, s = input_ids.shape
        start_pos = cache.seq_len() if cache is not None else 0

        x = self.weights["embed.weight"][input_ids]

        if self.use_rope:
            assert self._inv_freq is not None
            position_ids = torch.arange(
                start_pos, start_pos + s, device=input_ids.device, dtype=torch.long
            )[None, :].expand(b, -1)
            cos, sin = build_rope_cos_sin(self._inv_freq, position_ids, dtype=x.dtype)
        else:
            cos = sin = torch.empty(0, device=x.device, dtype=x.dtype)

        for spec in self.layers:
            x = decoder_block(
                x,
                self.weights,
                spec,
                cos,
                sin,
                self.config,
                cache=cache,
                use_rope=self.use_rope,
            )

        if cache is not None and hasattr(cache, "advance"):
            # Token cursor for hybrid (KV may not update until first attn layer).
            if isinstance(cache, RuntimeState):
                if start_pos == 0:
                    cache._token_len = s
                else:
                    cache.advance(s)

        x = rms_norm(x, self.weights["final_norm.weight"], self.config.rms_norm_eps)
        return F.linear(x, self.weights["lm_head.weight"])

    def num_params(self) -> int:
        from engine.weights import count_params

        return count_params(self.weights)


LlamaModel = DecoderModel
