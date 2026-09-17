"""Native multi-token-prediction draft heads.

Qwen3.5 uses a dedicated transformer + pre-fc norms. DeepSeek V3 and
Nemotron Super use enorm/hnorm/eh_proj fusion plus extra decoder layers
that share the target embed and LM head.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Protocol, runtime_checkable

import torch
import torch.nn.functional as F

from engine.cache import KVCache, RuntimeState
from engine.config import ModelConfig
from engine.layers.block import decoder_block
from engine.layers.norm import gemma_rms_norm, rms_norm
from engine.layers.rope import (
    _inv_freq_default,
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


@runtime_checkable
class NativeMTP(Protocol):
    """Draft head used by generate_mtp_greedy."""

    def make_cache(self, batch_size: int = 1) -> KVCache | RuntimeState: ...

    def forward(
        self,
        input_ids: torch.Tensor,
        previous_hidden: torch.Tensor,
        *,
        cache: KVCache | RuntimeState | None = None,
        position_ids: torch.Tensor | None = None,
        input_embeddings: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


_MTP_LAYER_KEY = re.compile(
    r"^mtp\.layers\.(\d+)\.(attn\.|mlp\.|moe\.|mamba\.|gdn\.|"
    r"input_norm\.|post_attn_norm\.|pre_ff_norm\.|post_ff_norm\.)"
)


def _norm_weight(
    x: torch.Tensor, weight: torch.Tensor, eps: float, kind: str
) -> torch.Tensor:
    if kind == "gemma_rms":
        return gemma_rms_norm(x, weight, eps)
    return rms_norm(x, weight, eps)


def _clone_layer_weights(
    source: dict[str, torch.Tensor], src_index: int, dst_index: int = 0
) -> dict[str, torch.Tensor]:
    prefix = f"layers.{src_index}."
    dest = f"layers.{dst_index}."
    return {
        dest + name[len(prefix) :]: tensor
        for name, tensor in source.items()
        if name.startswith(prefix)
    }


def _dedicated_mtp_indices(weights: dict[str, torch.Tensor]) -> list[int]:
    found: set[int] = set()
    for name in weights:
        match = _MTP_LAYER_KEY.match(name)
        if match:
            found.add(int(match.group(1)))
    return sorted(found)


def _strip_mtp_layer_prefix(
    weights: dict[str, torch.Tensor], src_index: int, dst_index: int
) -> dict[str, torch.Tensor]:
    prefix = f"mtp.layers.{src_index}."
    skip = ("enorm.", "hnorm.", "eh_proj.", "shared_head.")
    dest = f"layers.{dst_index}."
    out: dict[str, torch.Tensor] = {}
    for name, tensor in weights.items():
        if not name.startswith(prefix):
            continue
        rest = name[len(prefix) :]
        if rest.startswith(skip):
            continue
        out[dest + rest] = tensor
    return out


def _infer_layer_spec(weights: dict[str, torch.Tensor], index: int) -> LayerSpec:
    p = f"layers.{index}."
    has_attn = any(k.startswith(f"{p}attn.") for k in weights)
    has_mamba = any(k.startswith(f"{p}mamba.") for k in weights)
    has_gdn = any(k.startswith(f"{p}gdn.") for k in weights)
    has_moe = any(k.startswith(f"{p}moe.") for k in weights)
    has_mlp = any(k.startswith(f"{p}mlp.") for k in weights)
    if has_mamba:
        mixer = MixerKind.MAMBA2
    elif has_gdn:
        mixer = MixerKind.GATED_DELTANET
    elif has_attn:
        mixer = MixerKind.ATTENTION
    else:
        mixer = MixerKind.NONE
    if has_moe:
        ffn = FfnKind.MOE
    elif has_mlp:
        ffn = FfnKind.DENSE_MLP
    else:
        ffn = FfnKind.NONE
    if mixer == MixerKind.NONE and ffn == FfnKind.NONE:
        raise KeyError(f"MTP layer {index} has neither mixer nor FFN weights")
    return LayerSpec(index, mixer, ffn)


class NextnMTP:
    """DeepSeek / Nemotron enorm+hnorm+eh_proj draft stack sharing embed/lm_head."""

    def __init__(self, config: ModelConfig, target_weights: dict[str, torch.Tensor]):
        if not (config.num_nextn_predict_layers or 0):
            raise ValueError("NextnMTP requires num_nextn_predict_layers > 0")
        self.target_config = config
        self.embed = target_weights["embed.weight"]
        self.lm_head = target_weights.get("lm_head.weight", self.embed)
        self.device = self.embed.device
        self.dtype = self.embed.dtype
        self.norm_kind = config.norm_kind

        fusion_index = 0
        try:
            self.enorm = target_weights[f"mtp.layers.{fusion_index}.enorm.weight"]
            self.hnorm = target_weights[f"mtp.layers.{fusion_index}.hnorm.weight"]
            self.eh_proj = target_weights[f"mtp.layers.{fusion_index}.eh_proj.weight"]
        except KeyError as exc:
            raise KeyError(
                "DeepSeek/Nemotron MTP requires mtp.layers.0.{enorm,hnorm,eh_proj}"
            ) from exc

        head_key = f"mtp.layers.{fusion_index}.shared_head.norm.weight"
        self.head_norm = target_weights.get(
            head_key,
            torch.ones(config.hidden_size, device=self.device, dtype=self.dtype),
        )
        self.root = {
            "enorm.weight": self.enorm,
            "hnorm.weight": self.hnorm,
            "eh_proj.weight": self.eh_proj,
            "shared_head.norm.weight": self.head_norm,
        }

        dedicated = _dedicated_mtp_indices(target_weights)
        layer_weights: dict[str, torch.Tensor] = {}
        specs: list[LayerSpec] = []
        if dedicated:
            for dst, src in enumerate(dedicated):
                piece = _strip_mtp_layer_prefix(target_weights, src, dst)
                layer_weights.update(piece)
                specs.append(_infer_layer_spec(layer_weights, dst))
        else:
            if not config.layers:
                raise ValueError("NextnMTP fallback needs a target layer schedule")
            last = config.layers[-1]
            layer_weights.update(_clone_layer_weights(target_weights, last.index, 0))
            specs.append(LayerSpec(0, last.mixer, last.ffn))

        self.weights = layer_weights
        self.config = replace(
            config,
            num_hidden_layers=len(specs),
            layers=tuple(specs),
            num_nextn_predict_layers=None,
        )
        self.use_rope = config.pos_kind == "rope" and any(
            spec.mixer == MixerKind.ATTENTION for spec in specs
        )
        self._inv_freq = None
        if self.use_rope:
            if config.attention_kind == "mla" and config.qk_rope_head_dim:
                self._inv_freq = _inv_freq_default(
                    int(config.qk_rope_head_dim),
                    float(config.rope_theta),
                    self.device,
                )
            else:
                self._inv_freq = build_inv_freq(self.config, self.device)

    def make_cache(self, batch_size: int = 1) -> KVCache | RuntimeState:
        hybrid = any(
            spec.mixer in {MixerKind.MAMBA2, MixerKind.GATED_DELTANET}
            for spec in self.config.layers
        )
        if hybrid:
            return RuntimeState(
                self.config,
                batch_size=batch_size,
                device=self.device,
                dtype=self.dtype,
            )
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
        cache: KVCache | RuntimeState | None = None,
        position_ids: torch.Tensor | None = None,
        input_embeddings: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if input_ids.shape != previous_hidden.shape[:2]:
            raise ValueError(
                f"MTP ids/hidden mismatch: {input_ids.shape} vs {previous_hidden.shape}"
            )
        batch, seq = input_ids.shape
        start = cache.seq_len() if cache is not None else 0
        if position_ids is None:
            position_ids = torch.arange(
                start, start + seq, device=self.device, dtype=torch.long
            )[None].expand(batch, -1)

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
        embedding = _norm_weight(
            embedding, self.enorm, self.config.rms_norm_eps, self.norm_kind
        )
        hidden = _norm_weight(
            previous_hidden, self.hnorm, self.config.rms_norm_eps, self.norm_kind
        )
        hidden = F.linear(torch.cat((embedding, hidden), dim=-1), self.eh_proj)

        if self.use_rope and self._inv_freq is not None:
            if position_ids.dim() == 3:
                section = (
                    (self.config.rope_scaling or {}).get("mrope_section")
                    or [11, 11, 10]
                )
                cos, sin = build_mrope_cos_sin(
                    self._inv_freq,
                    position_ids,
                    dtype=hidden.dtype,
                    mrope_section=section,
                )
            else:
                cos, sin = build_rope_cos_sin(
                    self._inv_freq, position_ids, dtype=hidden.dtype
                )
        else:
            cos = sin = torch.empty(0, device=hidden.device, dtype=hidden.dtype)

        for spec in self.config.layers:
            hidden = decoder_block(
                hidden,
                self.weights,
                spec,
                cos,
                sin,
                self.config,
                cache=cache,
                use_rope=self.use_rope,
            )
        hidden_out = _norm_weight(
            hidden, self.head_norm, self.config.rms_norm_eps, self.norm_kind
        )
        return F.linear(hidden_out, self.lm_head), hidden_out


def build_mtp(
    config: ModelConfig,
    target_weights: dict[str, torch.Tensor],
    hf_weights: dict[str, torch.Tensor] | None = None,
) -> NativeMTP | None:
    """Construct a native MTP head when the recipe advertises one."""
    n_mtp = config.num_nextn_predict_layers or 0
    if n_mtp <= 0:
        return None
    if config.recipe_id == "qwen3_5":
        if hf_weights is None:
            raise ValueError("Qwen3.5 MTP requires separately loaded HF tensors")
        return Qwen35MTP(config, target_weights, hf_weights)
    if config.recipe_id in {"deepseek_v3", "deepseek_v2", "nemotron_h"}:
        return NextnMTP(config, target_weights)
    return None
