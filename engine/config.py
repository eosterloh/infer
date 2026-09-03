"""Parse config.json into a typed ModelConfig + layer schedule."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from engine.detect import detect_recipe_id
from engine.schedule import (
    FfnKind,
    LayerSpec,
    MixerKind,
    build_schedule,
    llama_dense_schedule,
)


@dataclass(frozen=True)
class ModelConfig:
    """Hyperparameters that size every weight matrix + layer recipe."""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    tie_word_embeddings: bool
    torch_dtype: str
    hidden_act: str = "silu"
    attention_bias: bool = False
    mlp_bias: bool = False
    bos_token_id: int | list[int] | None = None
    eos_token_id: int | list[int] | None = None
    rope_scaling: dict[str, Any] | None = None
    model_type: str = "llama"
    recipe_id: str = "llama"
    architectures: tuple[str, ...] = field(default_factory=tuple)
    layers: tuple[LayerSpec, ...] = field(default_factory=tuple)
    hybrid_override_pattern: str | None = None
    mamba_num_heads: int | None = None
    mamba_head_dim: int | None = None
    ssm_state_size: int | None = None
    n_groups: int | None = None
    conv_kernel: int | None = None
    n_routed_experts: int | None = None
    n_shared_experts: int | None = None
    num_experts_per_tok: int | None = None
    moe_intermediate_size: int | None = None
    moe_shared_expert_intermediate_size: int | None = None
    routed_scaling_factor: float | None = None
    mlp_hidden_act: str | None = None
    moe_latent_size: int | None = None
    q_lora_rank: int | None = None
    kv_lora_rank: int | None = None
    qk_nope_head_dim: int | None = None
    qk_rope_head_dim: int | None = None
    v_head_dim: int | None = None
    first_k_dense_replace: int | None = None
    num_nextn_predict_layers: int | None = None
    sliding_window: int | None = None
    qk_norm: bool = False
    intermediate_size_mlp: int | None = None
    layer_types: tuple[str, ...] = field(default_factory=tuple)
    attn_output_gate: bool = False
    partial_rotary_factor: float = 1.0
    linear_num_key_heads: int | None = None
    linear_num_value_heads: int | None = None
    linear_key_head_dim: int | None = None
    linear_value_head_dim: int | None = None
    linear_conv_kernel_dim: int | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def nq(self) -> int:
        return self.num_attention_heads

    @property
    def nkv(self) -> int:
        return self.num_key_value_heads

    @property
    def mamba_intermediate(self) -> int:
        if self.mamba_num_heads is None or self.mamba_head_dim is None:
            raise ValueError("mamba dims not set on config")
        return self.mamba_num_heads * self.mamba_head_dim

    @property
    def mamba_conv_dim(self) -> int:
        if self.n_groups is None or self.ssm_state_size is None:
            raise ValueError("mamba group/state dims not set on config")
        return self.mamba_intermediate + 2 * self.n_groups * self.ssm_state_size

    @property
    def gemma_rms(self) -> bool:
        return self.recipe_id == "gemma" or (self.model_type or "").startswith("gemma")

    @property
    def embed_scale(self) -> float:
        if self.gemma_rms:
            return float(self.hidden_size) ** 0.5
        return 1.0

    @property
    def pos_kind(self) -> str:
        if self.recipe_id == "gpt2":
            return "learned"
        if self.recipe_id == "nemotron_h":
            return "none"
        return "rope"

    @property
    def norm_kind(self) -> str:
        if self.recipe_id in {"gpt2", "gpt_neox"}:
            return "layer"
        if self.gemma_rms or self.recipe_id == "qwen3_5":
            return "gemma_rms"
        return "rms"

    @property
    def attention_kind(self) -> str:
        if self.recipe_id == "gpt2":
            return "gpt2"
        if self.recipe_id in {"phi3", "gpt_neox"}:
            return "fused_qkv"
        if self.recipe_id == "deepseek_v3":
            return "mla"
        if self.recipe_id == "gpt_oss":
            return "gqa_sinks"
        return "gqa"

    @property
    def moe_kind(self) -> str:
        if self.moe_latent_size:
            return "latent"
        if self.recipe_id == "mixtral":
            return "mixtral"
        if self.recipe_id == "llama4":
            return "llama4"
        if self.recipe_id == "gpt_oss":
            return "gpt_oss"
        if self.recipe_id == "deepseek_v3":
            return "deepseek"
        if self.recipe_id == "nemotron_h":
            return "nemotron"
        return "none"

    @property
    def uses_swiglu(self) -> bool:
        if self.recipe_id == "nemotron_h":
            return False
        if self.recipe_id in {"gpt2", "gpt_neox"}:
            return False
        return True

    @classmethod
    def from_pretrained(cls, model_dir: str | Path) -> ModelConfig:
        model_dir = Path(model_dir)
        path = model_dir / "config.json"
        if path.is_file():
            with path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
        else:
            from engine.gguf import find_gguf, gguf_meta_to_raw, read_gguf_header

            gguf = find_gguf(model_dir)
            if gguf is None:
                raise FileNotFoundError(f"missing config.json under {model_dir}")
            meta, tensors, _ = read_gguf_header(gguf)
            raw = gguf_meta_to_raw(meta)
            raw["tie_word_embeddings"] = not any(
                t.name in {"output.weight", "output"} for t in tensors
            )

        raw = normalize_raw(raw)
        raw = _merge_generation_config(model_dir, raw)
        recipe_id = detect_recipe_id(raw)

        hidden_size = int(raw["hidden_size"])
        nq = int(raw["num_attention_heads"])
        head_dim = int(raw.get("head_dim", hidden_size // nq))
        if head_dim * nq != hidden_size and "head_dim" not in raw and recipe_id not in {
            "deepseek_v3",
            "phi3",
            "qwen3_5",
        }:
            raise ValueError(
                f"hidden_size={hidden_size} not divisible by "
                f"num_attention_heads={nq}"
            )

        dtype = raw.get("torch_dtype", "bfloat16")
        if not isinstance(dtype, str):
            dtype = str(dtype)

        arches = raw.get("architectures") or []
        num_layers = int(raw["num_hidden_layers"])
        eps = float(
            raw.get("rms_norm_eps", raw.get("layer_norm_epsilon", raw.get("norm_eps", 1e-5)))
        )
        pattern = raw.get("hybrid_override_pattern")
        if pattern is not None:
            pattern = str(pattern)

        attn_bias = bool(raw.get("attention_bias", recipe_id in {"qwen2", "qwen3", "gpt2", "gpt_neox"}))
        qk_norm = bool(
            raw.get(
                "qk_norm",
                recipe_id in {"qwen3", "qwen3_5"} or raw.get("use_qk_norm", False),
            )
        )
        sliding = raw.get("sliding_window")
        sliding_i = int(sliding) if sliding not in (None, False) else None

        n_routed = _opt_int(raw, "n_routed_experts")
        moe_inter = _opt_int(raw, "moe_intermediate_size") or (
            int(raw["intermediate_size"]) if n_routed else None
        )
        shared_inter = _opt_int(raw, "moe_shared_expert_intermediate_size")
        if shared_inter is None and _opt_int(raw, "n_shared_experts"):
            shared_inter = (moe_inter or int(raw.get("intermediate_size", 0))) * int(
                raw["n_shared_experts"]
            )

        partial = cls(
            vocab_size=int(raw["vocab_size"]),
            hidden_size=hidden_size,
            intermediate_size=int(raw.get("intermediate_size", 0)),
            num_hidden_layers=num_layers,
            num_attention_heads=nq,
            num_key_value_heads=int(raw.get("num_key_value_heads", nq)),
            head_dim=head_dim,
            rms_norm_eps=eps,
            rope_theta=float(raw.get("rope_theta", 10000.0)),
            max_position_embeddings=int(raw.get("max_position_embeddings", 2048)),
            tie_word_embeddings=bool(raw.get("tie_word_embeddings", recipe_id == "gpt2")),
            torch_dtype=dtype,
            hidden_act=str(raw.get("hidden_act", "silu")),
            attention_bias=attn_bias,
            mlp_bias=bool(raw.get("mlp_bias", recipe_id in {"gpt2", "gpt_neox"})),
            bos_token_id=raw.get("bos_token_id"),
            eos_token_id=raw.get("eos_token_id"),
            rope_scaling=raw.get("rope_scaling") or raw.get("rope_parameters"),
            model_type=str(raw.get("model_type", "llama")),
            recipe_id=recipe_id,
            architectures=tuple(arches),
            layers=(),
            hybrid_override_pattern=pattern,
            mamba_num_heads=_opt_int(raw, "mamba_num_heads"),
            mamba_head_dim=_opt_int(raw, "mamba_head_dim"),
            ssm_state_size=_opt_int(raw, "ssm_state_size"),
            n_groups=_opt_int(raw, "n_groups"),
            conv_kernel=_opt_int(raw, "conv_kernel"),
            n_routed_experts=n_routed,
            n_shared_experts=_opt_int(raw, "n_shared_experts"),
            num_experts_per_tok=_opt_int(raw, "num_experts_per_tok"),
            moe_intermediate_size=moe_inter,
            moe_shared_expert_intermediate_size=shared_inter,
            routed_scaling_factor=(
                float(raw["routed_scaling_factor"])
                if "routed_scaling_factor" in raw
                else None
            ),
            mlp_hidden_act=(
                str(raw["mlp_hidden_act"]) if "mlp_hidden_act" in raw else None
            ),
            moe_latent_size=_opt_int(raw, "moe_latent_size"),
            q_lora_rank=_opt_int(raw, "q_lora_rank"),
            kv_lora_rank=_opt_int(raw, "kv_lora_rank"),
            qk_nope_head_dim=_opt_int(raw, "qk_nope_head_dim"),
            qk_rope_head_dim=_opt_int(raw, "qk_rope_head_dim"),
            v_head_dim=_opt_int(raw, "v_head_dim"),
            first_k_dense_replace=_opt_int(raw, "first_k_dense_replace"),
            num_nextn_predict_layers=_opt_int(raw, "num_nextn_predict_layers")
            or _opt_int(raw, "mtp_num_layers"),
            sliding_window=sliding_i,
            qk_norm=qk_norm,
            intermediate_size_mlp=_opt_int(raw, "intermediate_size_mlp"),
            layer_types=tuple(str(t) for t in (raw.get("layer_types") or ())),
            attn_output_gate=bool(
                raw.get("attn_output_gate", recipe_id == "qwen3_5")
            ),
            partial_rotary_factor=float(raw.get("partial_rotary_factor") or 1.0),
            linear_num_key_heads=_opt_int(raw, "linear_num_key_heads"),
            linear_num_value_heads=_opt_int(raw, "linear_num_value_heads"),
            linear_key_head_dim=_opt_int(raw, "linear_key_head_dim"),
            linear_value_head_dim=_opt_int(raw, "linear_value_head_dim"),
            linear_conv_kernel_dim=_opt_int(raw, "linear_conv_kernel_dim"),
            raw=raw,
        )
        schedule = build_schedule(partial)
        return cls(
            **{**partial.__dict__, "layers": schedule},
        )

    def expected_shapes(self) -> dict[str, tuple[int, ...]]:
        """Blueprint shapes for every weight we expect after the name map."""
        h = self.hidden_size
        i = self.intermediate_size
        v = self.vocab_size
        dh = self.head_dim
        nq = self.num_attention_heads
        nkv = self.num_key_value_heads
        schedule = self.layers or llama_dense_schedule(self.num_hidden_layers)
        ln = self.norm_kind == "layer"
        bias = self.attention_bias
        mlp_bias = self.mlp_bias

        shapes: dict[str, tuple[int, ...]] = {
            "embed.weight": (v, h),
            "final_norm.weight": (h,),
        }
        if ln:
            shapes["final_norm.bias"] = (h,)
        if self.pos_kind == "learned":
            shapes["pos_embed.weight"] = (self.max_position_embeddings, h)
        if not self.tie_word_embeddings:
            shapes["lm_head.weight"] = (v, h)

        for spec in schedule:
            p = f"layers.{spec.index}"
            if spec.mixer != MixerKind.NONE or spec.ffn != FfnKind.NONE:
                shapes[f"{p}.input_norm.weight"] = (h,)
                if ln:
                    shapes[f"{p}.input_norm.bias"] = (h,)

            if spec.mixer == MixerKind.ATTENTION:
                shapes.update(self._attn_shapes(p, nq, nkv, dh, h, bias))
            elif spec.mixer == MixerKind.MAMBA2:
                shapes.update(self._mamba_shapes(p))
            elif spec.mixer == MixerKind.GATED_DELTANET:
                shapes.update(self._gdn_shapes(p))

            if spec.ffn == FfnKind.DENSE_MLP:
                if spec.mixer != MixerKind.NONE:
                    shapes[f"{p}.post_attn_norm.weight"] = (h,)
                    if ln:
                        shapes[f"{p}.post_attn_norm.bias"] = (h,)
                shapes.update(self._mlp_shapes(p, h, i, mlp_bias))
                if self.recipe_id == "nemotron_h" or (
                    self.mlp_hidden_act and self.mlp_hidden_act != "silu"
                ):
                    shapes.pop(f"{p}.mlp.gate.weight", None)
            elif spec.ffn == FfnKind.MOE:
                if spec.mixer != MixerKind.NONE:
                    shapes[f"{p}.post_attn_norm.weight"] = (h,)
                    if ln:
                        shapes[f"{p}.post_attn_norm.bias"] = (h,)
                shapes.update(self._moe_shapes(p))

        n_mtp = self.num_nextn_predict_layers or 0
        for j in range(n_mtp):
            shapes[f"mtp.layers.{j}.enorm.weight"] = (h,)
            shapes[f"mtp.layers.{j}.hnorm.weight"] = (h,)
            shapes[f"mtp.layers.{j}.eh_proj.weight"] = (h, 2 * h)

        return shapes

    def _attn_shapes(
        self, p: str, nq: int, nkv: int, dh: int, h: int, bias: bool
    ) -> dict[str, tuple[int, ...]]:
        kind = self.attention_kind
        if kind == "gpt2":
            return {
                f"{p}.attn.c_attn.weight": (3 * h, h),
                f"{p}.attn.c_attn.bias": (3 * h,),
                f"{p}.attn.c_proj.weight": (h, h),
                f"{p}.attn.c_proj.bias": (h,),
            }
        if kind == "fused_qkv":
            if self.recipe_id == "gpt_neox":
                qkv = (3 * nq * dh, h)
            else:
                qkv = ((nq + 2 * nkv) * dh, h)
            out: dict[str, tuple[int, ...]] = {
                f"{p}.attn.qkv.weight": qkv,
                f"{p}.attn.o.weight": (h, nq * dh),
            }
            if bias:
                out[f"{p}.attn.qkv.bias"] = (qkv[0],)
                out[f"{p}.attn.o.bias"] = (h,)
            return out
        if kind == "mla":
            qk = (self.qk_nope_head_dim or dh) + (self.qk_rope_head_dim or dh)
            vdh = self.v_head_dim or dh
            kv_lora = self.kv_lora_rank or h
            rope_d = self.qk_rope_head_dim or dh
            nope = self.qk_nope_head_dim or dh
            shapes: dict[str, tuple[int, ...]] = {
                f"{p}.attn.kv_a.weight": (kv_lora + rope_d, h),
                f"{p}.attn.kv_a_norm.weight": (kv_lora,),
                f"{p}.attn.kv_b.weight": (nq * (nope + vdh), kv_lora),
                f"{p}.attn.o.weight": (h, nq * vdh),
            }
            if self.q_lora_rank:
                shapes[f"{p}.attn.q_a.weight"] = (self.q_lora_rank, h)
                shapes[f"{p}.attn.q_a_norm.weight"] = (self.q_lora_rank,)
                shapes[f"{p}.attn.q_b.weight"] = (nq * qk, self.q_lora_rank)
            else:
                shapes[f"{p}.attn.q.weight"] = (nq * qk, h)
            return shapes
        q_out = nq * dh * (2 if self.attn_output_gate else 1)
        shapes = {
            f"{p}.attn.q.weight": (q_out, h),
            f"{p}.attn.k.weight": (nkv * dh, h),
            f"{p}.attn.v.weight": (nkv * dh, h),
            f"{p}.attn.o.weight": (h, nq * dh),
        }
        if bias:
            shapes[f"{p}.attn.q.bias"] = (q_out,)
            shapes[f"{p}.attn.k.bias"] = (nkv * dh,)
            shapes[f"{p}.attn.v.bias"] = (nkv * dh,)
        if self.qk_norm:
            shapes[f"{p}.attn.q_norm.weight"] = (dh,)
            shapes[f"{p}.attn.k_norm.weight"] = (dh,)
        if kind == "gqa_sinks":
            shapes[f"{p}.attn.sinks"] = (nq,)
        return shapes

    def _gdn_shapes(self, p: str) -> dict[str, tuple[int, ...]]:
        assert self.linear_num_key_heads is not None
        assert self.linear_num_value_heads is not None
        assert self.linear_key_head_dim is not None
        assert self.linear_value_head_dim is not None
        assert self.linear_conv_kernel_dim is not None
        h = self.hidden_size
        key_dim = self.linear_num_key_heads * self.linear_key_head_dim
        value_dim = self.linear_num_value_heads * self.linear_value_head_dim
        conv_dim = key_dim * 2 + value_dim
        k = self.linear_conv_kernel_dim
        nv = self.linear_num_value_heads
        return {
            f"{p}.gdn.in_proj_qkv.weight": (conv_dim, h),
            f"{p}.gdn.in_proj_z.weight": (value_dim, h),
            f"{p}.gdn.in_proj_b.weight": (nv, h),
            f"{p}.gdn.in_proj_a.weight": (nv, h),
            f"{p}.gdn.conv1d.weight": (conv_dim, 1, k),
            f"{p}.gdn.A_log": (nv,),
            f"{p}.gdn.dt_bias": (nv,),
            f"{p}.gdn.norm.weight": (self.linear_value_head_dim,),
            f"{p}.gdn.out_proj.weight": (h, value_dim),
        }

    def _mlp_shapes(
        self, p: str, h: int, i: int, mlp_bias: bool
    ) -> dict[str, tuple[int, ...]]:
        if self.recipe_id == "phi3":
            return {
                f"{p}.mlp.gate_up.weight": (2 * i, h),
                f"{p}.mlp.down.weight": (h, i),
            }
        if self.recipe_id == "gpt2":
            return {
                f"{p}.mlp.c_fc.weight": (i, h),
                f"{p}.mlp.c_fc.bias": (i,),
                f"{p}.mlp.c_proj.weight": (h, i),
                f"{p}.mlp.c_proj.bias": (h,),
            }
        if self.recipe_id == "gpt_neox":
            out = {
                f"{p}.mlp.up.weight": (i, h),
                f"{p}.mlp.down.weight": (h, i),
            }
            if mlp_bias:
                out[f"{p}.mlp.up.bias"] = (i,)
                out[f"{p}.mlp.down.bias"] = (h,)
            return out
        dense_i = self.intermediate_size_mlp or i
        out = {
            f"{p}.mlp.gate.weight": (dense_i, h),
            f"{p}.mlp.up.weight": (dense_i, h),
            f"{p}.mlp.down.weight": (h, dense_i),
        }
        if mlp_bias:
            out[f"{p}.mlp.gate.bias"] = (dense_i,)
            out[f"{p}.mlp.up.bias"] = (dense_i,)
            out[f"{p}.mlp.down.bias"] = (h,)
        return out

    def _mamba_shapes(self, p: str) -> dict[str, tuple[int, ...]]:
        assert self.mamba_num_heads is not None
        assert self.mamba_head_dim is not None
        assert self.conv_kernel is not None
        h = self.hidden_size
        n_heads = self.mamba_num_heads
        inter = self.mamba_intermediate
        conv_dim = self.mamba_conv_dim
        proj = inter + conv_dim + n_heads
        k = self.conv_kernel
        return {
            f"{p}.mamba.in_proj.weight": (proj, h),
            f"{p}.mamba.out_proj.weight": (h, inter),
            f"{p}.mamba.conv1d.weight": (conv_dim, 1, k),
            f"{p}.mamba.conv1d.bias": (conv_dim,),
            f"{p}.mamba.A_log": (n_heads,),
            f"{p}.mamba.D": (n_heads,),
            f"{p}.mamba.dt_bias": (n_heads,),
            f"{p}.mamba.norm.weight": (inter,),
        }

    def _moe_shapes(self, p: str) -> dict[str, tuple[int, ...]]:
        assert self.n_routed_experts is not None
        n_e = self.n_routed_experts
        h = self.hidden_size
        mi = self.moe_intermediate_size or self.intermediate_size
        kind = self.moe_kind
        latent = self.moe_latent_size or h

        if kind in {"mixtral", "llama4", "gpt_oss"}:
            shapes: dict[str, tuple[int, ...]] = {
                f"{p}.moe.gate.weight": (n_e, h),
            }
            if kind == "gpt_oss":
                shapes[f"{p}.moe.gate.bias"] = (n_e,)
                shapes[f"{p}.moe.experts.gate_up.weight"] = (n_e, h, 2 * mi)
                shapes[f"{p}.moe.experts.gate_up.bias"] = (n_e, 2 * mi)
                shapes[f"{p}.moe.experts.down.weight"] = (n_e, mi, h)
                shapes[f"{p}.moe.experts.down.bias"] = (n_e, h)
                return shapes
            if kind == "llama4":
                shapes[f"{p}.moe.experts.gate_up.weight"] = (n_e, h, 2 * mi)
                shapes[f"{p}.moe.experts.down.weight"] = (n_e, mi, h)
                if self.n_shared_experts:
                    si = self.moe_shared_expert_intermediate_size or mi
                    shapes[f"{p}.moe.shared.gate.weight"] = (si, h)
                    shapes[f"{p}.moe.shared.up.weight"] = (si, h)
                    shapes[f"{p}.moe.shared.down.weight"] = (h, si)
                return shapes
            for e in range(n_e):
                shapes[f"{p}.moe.experts.{e}.gate.weight"] = (mi, h)
                shapes[f"{p}.moe.experts.{e}.up.weight"] = (mi, h)
                shapes[f"{p}.moe.experts.{e}.down.weight"] = (h, mi)
            return shapes

        if kind == "deepseek":
            shapes = {f"{p}.moe.gate.weight": (n_e, h)}
            if "e_score_correction_bias" in str(self.raw):
                shapes[f"{p}.moe.gate.e_score_correction_bias"] = (n_e,)
            for e in range(n_e):
                shapes[f"{p}.moe.experts.{e}.gate.weight"] = (mi, h)
                shapes[f"{p}.moe.experts.{e}.up.weight"] = (mi, h)
                shapes[f"{p}.moe.experts.{e}.down.weight"] = (h, mi)
            n_shared = self.n_shared_experts or 0
            if n_shared:
                si = self.moe_shared_expert_intermediate_size or mi * n_shared
                shapes[f"{p}.moe.shared.gate.weight"] = (si, h)
                shapes[f"{p}.moe.shared.up.weight"] = (si, h)
                shapes[f"{p}.moe.shared.down.weight"] = (h, si)
            return shapes

        # Nemotron-H (+ LatentMoE)
        assert self.moe_intermediate_size is not None
        assert self.moe_shared_expert_intermediate_size is not None
        si = self.moe_shared_expert_intermediate_size
        in_dim = latent if kind == "latent" else h
        shapes = {
            f"{p}.moe.gate.weight": (n_e, h),
            f"{p}.moe.gate.e_score_correction_bias": (n_e,),
            f"{p}.moe.shared.up.weight": (si, h),
            f"{p}.moe.shared.down.weight": (h, si),
        }
        if kind == "latent":
            shapes[f"{p}.moe.latent_down.weight"] = (latent, h)
            shapes[f"{p}.moe.latent_up.weight"] = (h, latent)
        for e in range(n_e):
            shapes[f"{p}.moe.experts.{e}.up.weight"] = (mi, in_dim)
            shapes[f"{p}.moe.experts.{e}.down.weight"] = (in_dim, mi)
        return shapes

    def summary(self) -> str:
        sched = self.layers or ()
        kinds: dict[str, int] = {}
        for spec in sched:
            key = f"{spec.mixer.value}+{spec.ffn.value}"
            kinds[key] = kinds.get(key, 0) + 1
        kind_s = ",".join(f"{k}×{n}" for k, n in sorted(kinds.items())) or "unset"
        return (
            f"type={self.model_type} recipe={self.recipe_id} layers={self.num_hidden_layers} "
            f"hidden={self.hidden_size} intermediate={self.intermediate_size} "
            f"heads={self.num_attention_heads}/{self.num_key_value_heads} "
            f"head_dim={self.head_dim} vocab={self.vocab_size} "
            f"ctx={self.max_position_embeddings} dtype={self.torch_dtype} "
            f"tie_embeddings={self.tie_word_embeddings} schedule=[{kind_s}]"
        )


def normalize_raw(raw: dict[str, Any]) -> dict[str, Any]:
    """Fold nested text_config and GPT-2 / Mixtral aliases into one dict."""
    raw = dict(raw)
    if "text_config" in raw and isinstance(raw["text_config"], dict):
        if "hidden_size" not in raw:
            text = dict(raw["text_config"])
            arches = raw.get("architectures")
            mt = raw.get("model_type")
            raw = {**raw, **text}
            if arches:
                raw["architectures"] = arches
            if mt and "model_type" not in text:
                raw["model_type"] = mt
    aliases = {
        "n_embd": "hidden_size",
        "n_head": "num_attention_heads",
        "n_layer": "num_hidden_layers",
        "n_positions": "max_position_embeddings",
        "n_inner": "intermediate_size",
        "num_local_experts": "n_routed_experts",
        "num_experts": "n_routed_experts",
        "n_ctx": "max_position_embeddings",
        "d_model": "hidden_size",
        "n_embed": "hidden_size",
    }
    for src, dst in aliases.items():
        if src in raw and (dst not in raw or raw[dst] in (None, 0)):
            raw[dst] = raw[src]
    if not raw.get("intermediate_size") and raw.get("hidden_size"):
        raw["intermediate_size"] = 4 * int(raw["hidden_size"])
    if "num_key_value_heads" not in raw and "num_attention_heads" in raw:
        raw["num_key_value_heads"] = raw["num_attention_heads"]
    if "vocab_size" not in raw and isinstance(raw.get("tokenizer.ggml.tokens"), list):
        raw["vocab_size"] = len(raw["tokenizer.ggml.tokens"])
    if "torch_dtype" not in raw and isinstance(raw.get("dtype"), str):
        raw["torch_dtype"] = raw["dtype"]
    rp = raw.get("rope_parameters")
    if isinstance(rp, dict):
        if rp.get("rope_theta") is not None:
            raw["rope_theta"] = rp["rope_theta"]
        if rp.get("partial_rotary_factor") is not None and "partial_rotary_factor" not in raw:
            raw["partial_rotary_factor"] = rp["partial_rotary_factor"]
    return raw


def _opt_int(raw: dict[str, Any], key: str) -> int | None:
    if key not in raw or raw[key] is None:
        return None
    return int(raw[key])


def _merge_generation_config(model_dir: Path, raw: dict[str, Any]) -> dict[str, Any]:
    """Folder-in stop tokens: generation_config.json wins over config.json."""
    path = model_dir / "generation_config.json"
    if not path.is_file():
        return raw
    with path.open("r", encoding="utf-8") as f:
        gen = json.load(f)
    merged = dict(raw)
    for key in ("eos_token_id", "bos_token_id", "pad_token_id"):
        if key in gen and gen[key] is not None:
            merged[key] = gen[key]
    return merged
