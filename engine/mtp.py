"""Native Qwen3.5 / Qwen3.8 multi-token-prediction draft head."""

from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn.functional as F

from engine.cache import KVCache
from engine.config import ModelConfig
from engine.layers.block import decoder_block
from engine.layers.norm import gemma_rms_norm
from engine.layers.rope import (
    build_inv_freq,
    build_mrope_cos_sin,
    build_rope_cos_sin,
)
from engine.schedule import FfnKind, LayerSpec, MixerKind


_MTP_LAYER_MAP = {
    "input_layernorm.weight": "input_norm.weight",
    "post_attention_layernorm.weight": "post_attn_norm.weight",
    "self_attn.q_proj.weight": "attn.q.weight",
    "self_attn.k_proj.weight": "attn.k.weight",
    "self_attn.v_proj.weight": "attn.v.weight",
    "self_attn.o_proj.weight": "attn.o.weight",
    "self_attn.q_norm.weight": "attn.q_norm.weight",
    "self_attn.k_norm.weight": "attn.k_norm.weight",
    "mlp.gate_proj.weight": "mlp.gate.weight",
    "mlp.up_proj.weight": "mlp.up.weight",
    "mlp.down_proj.weight": "mlp.down.weight",
}


def qwen35_mtp_expected_shapes(
    config: ModelConfig,
) -> dict[str, tuple[int, ...]]:
    """Return the exact HF MTP tensor contract, excluding shared weights."""
    h = config.hidden_size
    inter = config.intermediate_size
    nq = config.num_attention_heads
    nkv = config.num_key_value_heads
    hd = config.head_dim
    expected = {
        "mtp.fc.weight": (h, 2 * h),
        "mtp.norm.weight": (h,),
        "mtp.pre_fc_norm_embedding.weight": (h,),
        "mtp.pre_fc_norm_hidden.weight": (h,),
    }
    for i in range(config.num_nextn_predict_layers or 0):
        prefix = f"mtp.layers.{i}"
        expected.update(
            {
                f"{prefix}.input_layernorm.weight": (h,),
                f"{prefix}.post_attention_layernorm.weight": (h,),
                f"{prefix}.self_attn.q_proj.weight": (2 * nq * hd, h),
                f"{prefix}.self_attn.k_proj.weight": (nkv * hd, h),
                f"{prefix}.self_attn.v_proj.weight": (nkv * hd, h),
                f"{prefix}.self_attn.o_proj.weight": (h, nq * hd),
                f"{prefix}.self_attn.q_norm.weight": (hd,),
                f"{prefix}.self_attn.k_norm.weight": (hd,),
                f"{prefix}.mlp.gate_proj.weight": (inter, h),
                f"{prefix}.mlp.up_proj.weight": (inter, h),
                f"{prefix}.mlp.down_proj.weight": (h, inter),
            }
        )
    return expected


class Qwen35MTP:
    """One full-attention draft layer sharing target embeddings and LM head."""

    def __init__(
        self,
        config: ModelConfig,
        target_weights: dict[str, torch.Tensor],
        hf_weights: dict[str, torch.Tensor],
    ):
        if config.recipe_id != "qwen3_5":
            raise ValueError("Qwen35MTP requires a qwen3_5 target config")
        self.target_config = config
        self.config = replace(
            config,
            num_hidden_layers=1,
            layers=(LayerSpec(0, MixerKind.ATTENTION, FfnKind.DENSE_MLP),),
            layer_types=("full_attention",),
            num_nextn_predict_layers=None,
        )
        self.embed = target_weights["embed.weight"]
        self.lm_head = target_weights["lm_head.weight"]
        self.device = self.embed.device
        self.dtype = self.embed.dtype
        self.root = {
            name: hf_weights[f"mtp.{name}"]
            for name in (
                "fc.weight",
                "norm.weight",
                "pre_fc_norm_embedding.weight",
                "pre_fc_norm_hidden.weight",
            )
        }
        self.weights: dict[str, torch.Tensor] = {}
        prefix = "mtp.layers.0."
        for hf_name, tensor in hf_weights.items():
            if not hf_name.startswith(prefix):
                continue
            rest = hf_name[len(prefix) :]
            mapped = _MTP_LAYER_MAP.get(rest)
            if mapped is not None:
                self.weights[f"layers.0.{mapped}"] = tensor

        expected = set(self.config.expected_shapes())
        expected.discard("embed.weight")
        expected.discard("final_norm.weight")
        expected.discard("lm_head.weight")
        missing = sorted(expected - set(self.weights))
        if missing:
            raise KeyError(f"MTP layer missing weights: {missing}")
        root_shapes = {
            name.removeprefix("mtp."): shape
            for name, shape in qwen35_mtp_expected_shapes(config).items()
            if not name.startswith("mtp.layers.")
        }
        for name, shape in root_shapes.items():
            if tuple(self.root[name].shape) != shape:
                raise ValueError(
                    f"MTP {name}: got {tuple(self.root[name].shape)}, expected {shape}"
                )
        expected_shapes = self.config.expected_shapes()
        for name, tensor in self.weights.items():
            shape = expected_shapes[name]
            if tuple(tensor.shape) != shape:
                raise ValueError(
                    f"MTP {name}: got {tuple(tensor.shape)}, expected {shape}"
                )
        self._inv_freq = build_inv_freq(self.config, self.device)

    def make_cache(self, batch_size: int = 1) -> KVCache:
        return KVCache(
            self.config,
            batch_size=batch_size,
            device=self.device,
            dtype=self.dtype,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        previous_hidden: torch.Tensor,
        *,
        cache: KVCache | None = None,
        position_ids: torch.Tensor | None = None,
        input_embeddings: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if input_ids.shape != previous_hidden.shape[:2]:
            raise ValueError(
                f"MTP ids/hidden mismatch: {input_ids.shape} vs {previous_hidden.shape}"
            )
        b, s = input_ids.shape
        start = cache.seq_len() if cache is not None else 0
        if position_ids is None:
            position_ids = torch.arange(
                start, start + s, device=self.device, dtype=torch.long
            )[None].expand(b, -1)

        embedding = (
            self.embed[input_ids]
            if input_embeddings is None
            else input_embeddings.to(device=self.device, dtype=self.dtype)
        )
        if embedding.shape != previous_hidden.shape:
            raise ValueError(
                f"MTP embeddings/hidden mismatch: {embedding.shape} "
                f"vs {previous_hidden.shape}"
            )
        embedding = gemma_rms_norm(
            embedding,
            self.root["pre_fc_norm_embedding.weight"],
            self.config.rms_norm_eps,
        )
        hidden = gemma_rms_norm(
            previous_hidden,
            self.root["pre_fc_norm_hidden.weight"],
            self.config.rms_norm_eps,
        )
        x = F.linear(
            torch.cat((embedding, hidden), dim=-1), self.root["fc.weight"]
        )

        if position_ids.dim() == 3:
            section = (
                (self.config.rope_scaling or {}).get("mrope_section")
                or [11, 11, 10]
            )
            cos, sin = build_mrope_cos_sin(
                self._inv_freq,
                position_ids,
                dtype=x.dtype,
                mrope_section=section,
            )
        else:
            cos, sin = build_rope_cos_sin(
                self._inv_freq, position_ids, dtype=x.dtype
            )
        x = decoder_block(
            x,
            self.weights,
            self.config.layers[0],
            cos,
            sin,
            self.config,
            cache=cache,
            use_rope=True,
        )
        hidden_out = gemma_rms_norm(
            x, self.root["norm.weight"], self.config.rms_norm_eps
        )
        return F.linear(hidden_out, self.lm_head), hidden_out
