"""Decoder model — scheduled forward with optional runtime state."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from engine.cache import KVCache, RuntimeState
from engine.config import ModelConfig
from engine.layers import (
    build_inv_freq,
    build_mrope_cos_sin,
    build_rope_cos_sin,
    decoder_block,
)
from engine.layers.norm import apply_norm
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
        self.use_rope = (config.pos_kind == "rope") and any(
            s.mixer == MixerKind.ATTENTION for s in self.layers
        )
        rope_dim = config.qk_rope_head_dim if config.attention_kind == "mla" else None
        self._inv_freq = None
        if self.use_rope:
            self._inv_freq = build_inv_freq(config, device=self.device)
            if rope_dim and self._inv_freq.numel() * 2 != rope_dim:
                # rebuild inv_freq for MLA rotary dim
                from engine.layers.rope import _inv_freq_default

                self._inv_freq = _inv_freq_default(
                    rope_dim, float(config.rope_theta), self.device
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
        hybrid = any(
            s.mixer in {MixerKind.MAMBA2, MixerKind.GATED_DELTANET} for s in self.layers
        )
        if mt in {"nemotron_h", "nemotronh", "qwen3_5", "qwen3_5_text"} or hybrid:
            return RuntimeState(
                self.config, batch_size=batch_size, device=device, dtype=dtype
            )
        return KVCache(
            self.config, batch_size=batch_size, device=device, dtype=dtype
        )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        cache: KVCache | RuntimeState | None = None,
        *,
        inputs_embeds: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        return_hidden: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Token IDs or precomputed embeddings → logits, optionally final hidden."""
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("provide exactly one of input_ids or inputs_embeds")
        if input_ids is not None and input_ids.dim() != 2:
            raise ValueError(f"expected input_ids [B, S], got {tuple(input_ids.shape)}")
        if inputs_embeds is not None and inputs_embeds.dim() != 3:
            raise ValueError(
                f"expected inputs_embeds [B,S,H], got {tuple(inputs_embeds.shape)}"
            )

        if inputs_embeds is None:
            assert input_ids is not None
            b, s = input_ids.shape
            x = self.weights["embed.weight"][input_ids]
        else:
            b, s, h = inputs_embeds.shape
            if h != self.config.hidden_size:
                raise ValueError(
                    f"embedding hidden {h} != config hidden {self.config.hidden_size}"
                )
            x = inputs_embeds
        start_pos = cache.seq_len() if cache is not None else 0
        effective_mask = attention_mask
        if cache is not None:
            effective_mask = cache.prepare_padding_mask(attention_mask, s)

        scale = self.config.embed_scale
        if scale != 1.0:
            x = x * scale
        if self.config.pos_kind == "learned":
            pos = torch.arange(
                start_pos, start_pos + s, device=x.device, dtype=torch.long
            )
            x = x + self.weights["pos_embed.weight"][pos]

        if self.use_rope:
            assert self._inv_freq is not None
            if position_ids is None:
                if effective_mask is not None:
                    position_ids = effective_mask.long().cumsum(-1) - 1
                    position_ids = position_ids.clamp_min(0)[:, -s:]
                else:
                    position_ids = torch.arange(
                        start_pos, start_pos + s, device=x.device, dtype=torch.long
                    )[None, :].expand(b, -1)
            if position_ids.dim() == 3:
                section = (
                    (self.config.rope_scaling or {}).get("mrope_section")
                    or (self.config.raw.get("rope_parameters") or {}).get("mrope_section")
                    or [11, 11, 10]
                )
                cos, sin = build_mrope_cos_sin(
                    self._inv_freq, position_ids, dtype=x.dtype, mrope_section=section
                )
            else:
                cos, sin = build_rope_cos_sin(
                    self._inv_freq, position_ids, dtype=x.dtype
                )
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
                attention_mask=effective_mask,
            )

        if cache is not None and hasattr(cache, "advance"):
            if isinstance(cache, RuntimeState):
                if start_pos == 0:
                    cache._token_len = s
                else:
                    cache.advance(s)

        pre_norm_hidden = x
        hidden = apply_norm(
            x,
            self.weights,
            "final_norm",
            self.config.rms_norm_eps,
            self.config.norm_kind,
        )
        logits = F.linear(hidden, self.weights["lm_head.weight"])
        if return_hidden:
            return logits, pre_norm_hidden
        return logits

    def num_params(self) -> int:
        from engine.weights import count_params

        return count_params(self.weights)


LlamaModel = DecoderModel
