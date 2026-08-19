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
    # Nemotron-H / hybrid extras (unused for dense Llama)
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

    @classmethod
    def from_pretrained(cls, model_dir: str | Path) -> ModelConfig:
        model_dir = Path(model_dir)
        path = model_dir / "config.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing config.json under {model_dir}")

        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)

        recipe_id = detect_recipe_id(raw)

        hidden_size = int(raw["hidden_size"])
        nq = int(raw["num_attention_heads"])
        head_dim = int(raw.get("head_dim", hidden_size // nq))
        if head_dim * nq != hidden_size and "head_dim" not in raw:
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
            tie_word_embeddings=bool(raw.get("tie_word_embeddings", False)),
            torch_dtype=dtype,
            hidden_act=str(raw.get("hidden_act", "silu")),
            attention_bias=bool(raw.get("attention_bias", False)),
            mlp_bias=bool(raw.get("mlp_bias", False)),
            bos_token_id=raw.get("bos_token_id"),
            eos_token_id=raw.get("eos_token_id"),
            rope_scaling=raw.get("rope_scaling"),
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
            n_routed_experts=_opt_int(raw, "n_routed_experts"),
            n_shared_experts=_opt_int(raw, "n_shared_experts"),
            num_experts_per_tok=_opt_int(raw, "num_experts_per_tok"),
            moe_intermediate_size=_opt_int(raw, "moe_intermediate_size"),
            moe_shared_expert_intermediate_size=_opt_int(
                raw, "moe_shared_expert_intermediate_size"
            ),
            routed_scaling_factor=(
                float(raw["routed_scaling_factor"])
                if "routed_scaling_factor" in raw
                else None
            ),
            mlp_hidden_act=(
                str(raw["mlp_hidden_act"]) if "mlp_hidden_act" in raw else None
            ),
            raw=raw,
        )
        schedule = build_schedule(partial)
        return cls(
            vocab_size=partial.vocab_size,
            hidden_size=partial.hidden_size,
            intermediate_size=partial.intermediate_size,
            num_hidden_layers=partial.num_hidden_layers,
            num_attention_heads=partial.num_attention_heads,
            num_key_value_heads=partial.num_key_value_heads,
            head_dim=partial.head_dim,
            rms_norm_eps=partial.rms_norm_eps,
            rope_theta=partial.rope_theta,
            max_position_embeddings=partial.max_position_embeddings,
            tie_word_embeddings=partial.tie_word_embeddings,
            torch_dtype=partial.torch_dtype,
            hidden_act=partial.hidden_act,
            attention_bias=partial.attention_bias,
            mlp_bias=partial.mlp_bias,
            bos_token_id=partial.bos_token_id,
            eos_token_id=partial.eos_token_id,
            rope_scaling=partial.rope_scaling,
            model_type=partial.model_type,
            recipe_id=partial.recipe_id,
            architectures=partial.architectures,
            layers=schedule,
            hybrid_override_pattern=partial.hybrid_override_pattern,
            mamba_num_heads=partial.mamba_num_heads,
            mamba_head_dim=partial.mamba_head_dim,
            ssm_state_size=partial.ssm_state_size,
            n_groups=partial.n_groups,
            conv_kernel=partial.conv_kernel,
            n_routed_experts=partial.n_routed_experts,
            n_shared_experts=partial.n_shared_experts,
            num_experts_per_tok=partial.num_experts_per_tok,
            moe_intermediate_size=partial.moe_intermediate_size,
            moe_shared_expert_intermediate_size=partial.moe_shared_expert_intermediate_size,
            routed_scaling_factor=partial.routed_scaling_factor,
            mlp_hidden_act=partial.mlp_hidden_act,
            raw=partial.raw,
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

        shapes: dict[str, tuple[int, ...]] = {
            "embed.weight": (v, h),
            "final_norm.weight": (h,),
        }
        if not self.tie_word_embeddings:
            shapes["lm_head.weight"] = (v, h)

        for spec in schedule:
            p = f"layers.{spec.index}"
            # Prenorm for any non-empty layer
            if spec.mixer != MixerKind.NONE or spec.ffn != FfnKind.NONE:
                shapes[f"{p}.input_norm.weight"] = (h,)

            if spec.mixer == MixerKind.ATTENTION:
                shapes.update(
                    {
                        f"{p}.attn.q.weight": (nq * dh, h),
                        f"{p}.attn.k.weight": (nkv * dh, h),
                        f"{p}.attn.v.weight": (nkv * dh, h),
                        f"{p}.attn.o.weight": (h, nq * dh),
                    }
                )
            elif spec.mixer == MixerKind.MAMBA2:
                shapes.update(self._mamba_shapes(p))

            if spec.ffn == FfnKind.DENSE_MLP:
                # Llama stacked FFN uses post_attn_norm; Nemotron-H MLP-only
                # layers only have input_norm (already added).
                if spec.mixer != MixerKind.NONE:
                    shapes[f"{p}.post_attn_norm.weight"] = (h,)
                shapes.update(
                    {
                        f"{p}.mlp.gate.weight": (i, h),
                        f"{p}.mlp.up.weight": (i, h),
                        f"{p}.mlp.down.weight": (h, i),
                    }
                )
                # Nemotron dense MLP is up/down only (relu2) — no gate.
                # Detect via pattern char '-' / mlp_hidden_act without swiglu.
                if self.model_type == "nemotron_h" or (
                    self.mlp_hidden_act and self.mlp_hidden_act != "silu"
                ):
                    shapes.pop(f"{p}.mlp.gate.weight", None)
                    # NemotronHMLP: up (inter, h), down (h, inter)
                    # intermediate may be config.intermediate_size
            elif spec.ffn == FfnKind.MOE:
                shapes.update(self._moe_shapes(p))

        return shapes

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
        assert self.moe_intermediate_size is not None
        assert self.moe_shared_expert_intermediate_size is not None
        h = self.hidden_size
        n_e = self.n_routed_experts
        mi = self.moe_intermediate_size
        si = self.moe_shared_expert_intermediate_size
        shapes: dict[str, tuple[int, ...]] = {
            f"{p}.moe.gate.weight": (n_e, h),
            f"{p}.moe.gate.e_score_correction_bias": (n_e,),
            f"{p}.moe.shared.up.weight": (si, h),
            f"{p}.moe.shared.down.weight": (h, si),
        }
        for e in range(n_e):
            shapes[f"{p}.moe.experts.{e}.up.weight"] = (mi, h)
            shapes[f"{p}.moe.experts.{e}.down.weight"] = (h, mi)
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


def _opt_int(raw: dict[str, Any], key: str) -> int | None:
    if key not in raw or raw[key] is None:
        return None
    return int(raw[key])
