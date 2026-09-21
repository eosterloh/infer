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
from engine.layers.gdn import fuse_split_conv1d
from engine.layers.linear import dense
from engine.layers.norm import apply_norm
from engine.schedule import MixerKind, build_schedule


class DecoderModel:
    """Loaded weights + config-driven layer schedule; optional KV / hybrid state."""

    def __init__(self, config: ModelConfig, weights: dict[str, torch.Tensor]):
        self.config = config
        self.weights = fuse_split_conv1d(weights)
        sample = next(iter(self.weights.values()))
        self.device = sample.device
        self.dtype = sample.dtype
        self.layers = config.layers or build_schedule(config)
        self.use_rope = (config.pos_kind == "rope") and any(
            s.mixer == MixerKind.ATTENTION for s in self.layers
        )
        rope_dim = config.qk_rope_head_dim if config.attention_kind == "mla" else None
        self._inv_freq = None
        self._local_inv_freq = None
        self._theta_inv_freq: dict[float, torch.Tensor] = {}
        if self.use_rope:
            self._inv_freq = build_inv_freq(config, device=self.device)
            if rope_dim and self._inv_freq.numel() * 2 != rope_dim:
                # rebuild inv_freq for MLA rotary dim
                from engine.layers.rope import _inv_freq_default

                self._inv_freq = _inv_freq_default(
                    rope_dim, float(config.rope_theta), self.device
                )
            from engine.layers.rope import _inv_freq_default

            rotary_dim = int(
                (rope_dim or config.head_dim)
                * float(getattr(config, "partial_rotary_factor", 1.0) or 1.0)
            )
            if rotary_dim < 2:
                rotary_dim = rope_dim or config.head_dim
            self._theta_inv_freq[float(config.rope_theta)] = self._inv_freq
            for theta in getattr(config, "layer_rope_theta", ()) or ():
                t = float(theta)
                if t and t not in self._theta_inv_freq:
                    self._theta_inv_freq[t] = _inv_freq_default(rotary_dim, t, self.device)
            if any("sliding" in str(t).lower() for t in (config.layer_types or ())):
                local_theta = 10000.0 if config.recipe_id == "gemma3" else None
                rp = (config.rope_scaling or config.raw.get("rope_parameters") or {})
                if isinstance(rp, dict):
                    sliding = rp.get("sliding_attention")
                    if isinstance(sliding, dict) and sliding.get("rope_theta") is not None:
                        local_theta = float(sliding["rope_theta"])
                if local_theta is not None:
                    if local_theta not in self._theta_inv_freq:
                        self._theta_inv_freq[local_theta] = _inv_freq_default(
                            rotary_dim, local_theta, self.device
                        )
                    self._local_inv_freq = self._theta_inv_freq[local_theta]

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
            s.mixer in {MixerKind.MAMBA2, MixerKind.MAMBA1, MixerKind.GATED_DELTANET}
            for s in self.layers
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
        kv_mask: torch.Tensor | None = None,
        return_hidden: bool = False,
        logits_to_keep: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Token IDs or precomputed embeddings → logits, optionally final hidden.

        ``logits_to_keep=n`` projects only the last ``n`` positions through the
        LM head. Decoding needs one; on a 512-token prefill the head is about a
        fifth of the arithmetic, so generate always asks for one.
        """
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
        emb_mul = float(getattr(self.config, "embedding_multiplier", 1.0) or 1.0)
        if emb_mul != 1.0:
            x = x * emb_mul
        if self.config.embed_norm:
            x = apply_norm(
                x, self.weights, "embed_norm", self.config.rms_norm_eps, self.config.norm_kind
            )
        if self.config.pos_kind == "learned":
            pos = torch.arange(
                start_pos, start_pos + s, device=x.device, dtype=torch.long
            )
            offset = int(getattr(self.config, "pos_offset", 0) or 0)
            x = x + self.weights["pos_embed.weight"][pos + offset]

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
            local_cos = local_sin = None
            theta_tables: dict[float, tuple[torch.Tensor, torch.Tensor]] = {
                float(self.config.rope_theta): (cos, sin),
            }
            if self._local_inv_freq is not None and position_ids.dim() != 3:
                local_cos, local_sin = build_rope_cos_sin(
                    self._local_inv_freq, position_ids, dtype=x.dtype
                )
            for theta, inv in self._theta_inv_freq.items():
                if theta not in theta_tables:
                    theta_tables[theta] = build_rope_cos_sin(inv, position_ids, dtype=x.dtype)
        else:
            cos = sin = torch.empty(0, device=x.device, dtype=x.dtype)
            local_cos = local_sin = None
            theta_tables = {}

        types = getattr(self.config, "layer_types", ()) or ()
        layer_thetas = getattr(self.config, "layer_rope_theta", ()) or ()
        for spec in self.layers:
            layer_cos, layer_sin = cos, sin
            if spec.index < len(layer_thetas):
                theta = float(layer_thetas[spec.index])
                if theta == 0.0:
                    layer_cos = layer_sin = torch.empty(0, device=x.device, dtype=x.dtype)
                elif theta in theta_tables:
                    layer_cos, layer_sin = theta_tables[theta]
            elif (
                local_cos is not None
                and spec.index < len(types)
                and "sliding" in str(types[spec.index]).lower()
            ):
                layer_cos, layer_sin = local_cos, local_sin
            x = decoder_block(
                x,
                self.weights,
                spec,
                layer_cos,
                layer_sin,
                self.config,
                cache=cache,
                use_rope=self.use_rope,
                attention_mask=effective_mask,
                kv_mask=kv_mask,
                q_positions=position_ids,
            )

        if cache is not None and hasattr(cache, "advance"):
            if isinstance(cache, RuntimeState):
                if start_pos == 0:
                    cache._token_len = s
                else:
                    cache.advance(s)

        pre_norm_hidden = x
        if logits_to_keep is not None and 0 < int(logits_to_keep) < x.shape[1]:
            x = x[:, -int(logits_to_keep) :]
        hidden = apply_norm(
            x,
            self.weights,
            "final_norm",
            self.config.rms_norm_eps,
            self.config.norm_kind,
        )
        logits = dense(
            hidden,
            self.weights["lm_head.weight"],
            self.weights.get("lm_head.bias"),
        )
        logits_scaling = float(getattr(self.config, "logits_scaling", 1.0) or 1.0)
        if logits_scaling != 1.0:
            logits = logits / logits_scaling
        logit_scale = float(getattr(self.config, "logit_scale", 1.0) or 1.0)
        if logit_scale != 1.0:
            logits = logits * logit_scale
        cap = getattr(self.config, "final_logit_softcapping", None)
        if cap:
            cap_f = float(cap)
            logits = torch.tanh(logits / cap_f) * cap_f
        if return_hidden:
            return logits, pre_norm_hidden
        return logits

    def num_params(self) -> int:
        from engine.weights import count_params

        return count_params(self.weights)


LlamaModel = DecoderModel
