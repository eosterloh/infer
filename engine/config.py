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
    embedding_multiplier: float = 1.0
    residual_multiplier: float = 1.0
    attention_multiplier: float | None = None
    logits_scaling: float = 1.0
    logit_scale: float = 1.0
    attn_logit_softcapping: float | None = None
    final_logit_softcapping: float | None = None
    query_pre_attn_scalar: float | None = None
    clip_qkv: float | None = None
    residual_kind: str = "sequential"
    no_rope_layers: tuple[int, ...] = field(default_factory=tuple)
    layer_rope_theta: tuple[float, ...] = field(default_factory=tuple)
    attention_out_bias: bool | None = None
    norm_topk_prob: bool = True
    alibi: bool = False
    alibi_kind: str = "bloom"
    embed_norm: bool = False
    pos_offset: int = 0
    norm_has_bias: bool = True
    attn_sub_norm: bool = False
    ffn_sub_norm: bool = False
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
        return self.recipe_id in {"gemma", "gemma2", "gemma3", "qwen3_next", "qwen3_5_moe"} or (
            self.model_type or ""
        ).startswith("gemma")

    @property
    def embed_scale(self) -> float:
        if self.recipe_id in {"gemma", "gemma2", "gemma3"} or (
            (self.model_type or "").startswith("gemma")
        ):
            return float(self.hidden_size) ** 0.5
        return 1.0

    @property
    def pos_kind(self) -> str:
        if self.alibi or self.recipe_id in {"bloom", "mpt"}:
            return "alibi"
        if self.recipe_id in {"gpt2", "gpt_neo", "gpt_bigcode", "opt"}:
            return "learned"
        if self.recipe_id in {"nemotron_h", "jamba"}:
            return "none"
        return "rope"

    @property
    def norm_kind(self) -> str:
        if self.recipe_id == "olmo":
            return "olmo"
        if self.recipe_id == "cohere":
            return "cohere"
        if self.recipe_id == "cohere2_moe":
            if (self.raw or {}).get("rms_norm_eps") is not None:
                return "rms"
            return "cohere"
        if self.recipe_id == "nemotron":
            return "layer_1p"
        if self.recipe_id in {
            "gpt2",
            "gptj",
            "gpt_neo",
            "gpt_neox",
            "gpt_bigcode",
            "opt",
            "bloom",
            "falcon",
            "mpt",
            "starcoder2",
            "stablelm",
            "phi",
            "phimoe",
            "dbrx",
        }:
            return "layer"
        if self.gemma_rms or self.recipe_id == "qwen3_5":
            return "gemma_rms"
        return "rms"

    @property
    def attention_kind(self) -> str:
        if self.recipe_id == "gpt2":
            return "gpt2"
        if self.recipe_id == "gpt_bigcode":
            return "gpt_bigcode"
        if self.recipe_id in {"phi3", "phi4", "gpt_neox", "bloom", "falcon", "mpt", "dbrx"}:
            return "fused_qkv"
        if self.recipe_id in {"deepseek_v3", "deepseek_v2"}:
            return "mla"
        if self.recipe_id == "diffllama":
            return "diff"
        if self.recipe_id in {"gpt_oss", "granite_swa", "granitemoe_swa"}:
            return "gqa_sinks"
        return "gqa"

    @property
    def moe_kind(self) -> str:
        if self.moe_latent_size:
            return "latent"
        if self.recipe_id in {
            "mixtral",
            "qwen3_moe",
            "qwen2_moe",
            "olmoe",
            "flex_olmo",
            "granitemoe",
            "granitemoe_swa",
            "granitemoeshared",
            "phimoe",
            "hunyuan_v1_moe",
            "ernie4_5_moe",
            "cohere2_moe",
            "qwen3_next",
            "qwen3_5_moe",
            "jamba",
        }:
            return "mixtral"
        if self.recipe_id == "dbrx":
            return "dbrx"
        if self.recipe_id == "llama4":
            return "llama4"
        if self.recipe_id == "gpt_oss":
            return "gpt_oss"
        if self.recipe_id in {"deepseek_v3", "deepseek_v2", "glm4_moe", "exaone_moe"}:
            return "deepseek"
        if self.recipe_id == "nemotron_h":
            return "nemotron"
        return "none"

    @property
    def uses_swiglu(self) -> bool:
        if self.recipe_id == "nemotron_h":
            return False
        if self.recipe_id in {
            "gpt2",
            "gptj",
            "gpt_neo",
            "gpt_neox",
            "gpt_bigcode",
            "opt",
            "bloom",
            "falcon",
            "mpt",
            "starcoder2",
            "nemotron",
            "phi",
            "arcee",
        }:
            return False
        return True

    @property
    def rope_interleaved(self) -> bool:
        if self.recipe_id in {"cohere", "cohere2_moe", "glm", "gptj", "ernie4_5_moe"}:
            return True
        mt = (self.model_type or "").lower().replace("-", "_")
        if mt in {"helium"} or any(
            "helium" in str(a).lower() for a in (self.architectures or ())
        ):
            return True
        return False

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
            "deepseek_v2",
            "phi3",
            "phi4",
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
            raw.get(
                "rms_norm_eps",
                raw.get(
                    "layer_norm_epsilon",
                    raw.get(
                        "layer_norm_eps",
                        raw.get("norm_eps", raw.get("norm_epsilon", 1e-5)),
                    ),
                ),
            )
        )
        hybrid_pattern = raw.get("hybrid_override_pattern")
        if hybrid_pattern is not None:
            hybrid_pattern = str(hybrid_pattern)

        use_bias = bool(raw.get("use_bias", False))
        mt = str(raw.get("model_type") or "")
        attn_bias_default = recipe_id in {
            "qwen2",
            "qwen3",
            "qwen2_moe",
            "gpt2",
            "gpt_neox",
            "gpt_bigcode",
            "opt",
            "bloom",
            "starcoder2",
            "glm",
            "phi",
        } or use_bias or mt.lower() in {"seed_oss", "seedoss"}
        if "attention_bias" in raw:
            attn_bias = bool(raw["attention_bias"])
        elif "qkv_bias" in raw:
            attn_bias = bool(raw["qkv_bias"])
        elif "use_qkv_bias" in raw:
            attn_bias = bool(raw["use_qkv_bias"])
        else:
            attn_bias = attn_bias_default
        qk_norm_default = recipe_id in {
            "qwen3",
            "qwen3_5",
            "qwen3_moe",
            "qwen3_next",
            "qwen3_5_moe",
            "olmo2",
            "olmo3",
            "olmoe",
            "olmo_hybrid",
            "gemma3",
            "exaone4",
            "exaone_moe",
            "flex_olmo",
            "hunyuan_v1_moe",
        } or bool(raw.get("use_qk_norm", False))
        qk_norm = bool(raw.get("qk_norm", raw.get("qk_layernorm", qk_norm_default)))
        sliding = raw.get("sliding_window")
        sliding_i = int(sliding) if sliding not in (None, False) else None

        n_routed = _opt_int(raw, "n_routed_experts")
        moe_inter = _opt_int(raw, "moe_intermediate_size") or (
            int(raw["intermediate_size"]) if n_routed else None
        )
        shared_inter = _opt_int(raw, "moe_shared_expert_intermediate_size") or _opt_int(
            raw, "shared_expert_intermediate_size"
        ) or _opt_int(raw, "shared_intermediate_size")
        if recipe_id == "hunyuan_v1_moe":
            raw.setdefault("n_shared_experts", 1)
            if shared_inter is None:
                shared_inter = int(raw.get("intermediate_size", 0))
        if recipe_id == "cohere2_moe" and shared_inter is None and _opt_int(raw, "num_shared_experts"):
            shared_inter = int(raw.get("intermediate_size", 0)) * int(raw["num_shared_experts"])
        if recipe_id in {"qwen3_next", "qwen3_5_moe"}:
            raw.setdefault("n_shared_experts", 1)
            if shared_inter is None:
                shared_inter = _opt_int(raw, "shared_expert_intermediate_size") or int(
                    raw.get("moe_intermediate_size") or raw.get("intermediate_size") or 0
                )
        if recipe_id == "cohere2_moe" and raw.get("prefix_dense_intermediate_size"):
            raw.setdefault("intermediate_size_mlp", raw["prefix_dense_intermediate_size"])
        if shared_inter is None and _opt_int(raw, "n_shared_experts"):
            shared_inter = (moe_inter or int(raw.get("intermediate_size", 0))) * int(
                raw["n_shared_experts"]
            )
        partial_rotary = raw.get("partial_rotary_factor")
        if partial_rotary is None:
            if recipe_id in {"qwen3_5", "qwen3_next", "qwen3_5_moe"}:
                partial_rotary = 0.25
            elif recipe_id == "stablelm":
                partial_rotary = 0.25
            elif recipe_id in {"nemotron", "glm", "glm4_moe", "phi"}:
                partial_rotary = 0.5
            elif recipe_id == "gptj" and raw.get("rotary_dim") is not None:
                partial_rotary = float(raw["rotary_dim"]) / float(head_dim)
            else:
                partial_rotary = 1.0
        if recipe_id == "olmo_hybrid":
            raw.setdefault("linear_num_key_heads", nq)
            raw.setdefault("linear_num_value_heads", nq)
            n_lin_k = int(raw["linear_num_key_heads"])
            raw.setdefault(
                "linear_key_head_dim",
                int(0.75 * hidden_size / max(n_lin_k, 1)),
            )
            raw.setdefault(
                "linear_value_head_dim",
                2 * int(raw["linear_key_head_dim"]),
            )
            raw.setdefault("linear_conv_kernel_dim", 4)
            raw.setdefault("linear_allow_neg_eigval", True)
        linear_defaults = recipe_id in {"qwen3_5", "qwen3_next", "qwen3_5_moe"}
        hidden_act = str(
            raw.get("hidden_act")
            or raw.get("hidden_activation")
            or raw.get("activation_function")
            or raw.get("activation")
            or "silu"
        )
        if recipe_id == "bloom":
            hidden_act = "gelu_pytorch_tanh"
        elif recipe_id == "mpt":
            hidden_act = "gelu"
        elif recipe_id == "falcon" and hidden_act == "silu":
            hidden_act = "gelu"
        elif recipe_id == "gpt_bigcode" and hidden_act == "silu":
            hidden_act = str(raw.get("activation_function") or "gelu_pytorch_tanh")
        elif recipe_id in {"gptj", "gpt_neo"} and hidden_act == "silu":
            hidden_act = str(raw.get("activation_function") or "gelu_new")
        elif recipe_id == "opt" and hidden_act == "silu":
            hidden_act = str(raw.get("activation_function") or "relu")
        mlp_bias = bool(
            raw.get(
                "mlp_bias",
                recipe_id
                in {
                    "gpt2",
                    "gptj",
                    "gpt_neo",
                    "gpt_neox",
                    "gpt_bigcode",
                    "opt",
                    "bloom",
                    "phi",
                    "starcoder2",
                }
                or use_bias,
            )
        )
        if "attention_out_bias" in raw:
            attn_out_bias: bool | None = bool(raw["attention_out_bias"])
        elif recipe_id == "starcoder2" or use_bias:
            attn_out_bias = True
        elif recipe_id in {"phi", "gpt_neo", "opt", "bloom", "gpt_bigcode"}:
            attn_out_bias = True
        elif recipe_id == "gptj":
            attn_out_bias = False
        elif recipe_id == "falcon":
            attn_out_bias = bool(raw.get("bias", False))
        elif recipe_id == "mpt":
            attn_out_bias = False
        elif mt.lower() in {"seed_oss", "seedoss"}:
            attn_out_bias = False
        else:
            attn_out_bias = None

        layer_types = tuple(str(t) for t in (raw.get("layer_types") or ()))
        if not layer_types and recipe_id == "gpt_neo":
            attn_layers = raw.get("attention_layers")
            if not attn_layers and raw.get("attention_types"):
                attn_layers = []
                for item in raw["attention_types"]:
                    kinds, reps = item[0], item[1]
                    for _ in range(int(reps)):
                        attn_layers.extend(kinds)
            layer_types = tuple(
                "sliding_attention" if str(t) == "local" else "full_attention"
                for t in (attn_layers or ())
            )
            if sliding_i is None and raw.get("window_size") is not None:
                sliding_i = int(raw["window_size"])
        no_rope_raw = raw.get("no_rope_layers")
        if no_rope_raw is None and recipe_id == "smollm3":
            interval = int(raw.get("no_rope_layer_interval") or 4)
            no_rope_raw = [
                int((i + 1) % interval != 0) for i in range(num_layers)
            ]
        no_rope_layers = tuple(int(x) for x in (no_rope_raw or ()))
        if not layer_types and recipe_id == "gemma2":
            layer_types = tuple(
                "sliding_attention" if (i + 1) % 2 else "full_attention"
                for i in range(num_layers)
            )
        if not layer_types and recipe_id == "gemma3":
            sw_pattern = int(raw.get("sliding_window_pattern") or 6)
            layer_types = tuple(
                "sliding_attention" if (i + 1) % sw_pattern else "full_attention"
                for i in range(num_layers)
            )
        if not layer_types and recipe_id == "olmo3":
            layer_types = tuple(
                "sliding_attention" if (i + 1) % 4 != 0 else "full_attention"
                for i in range(num_layers)
            )
        if not layer_types and recipe_id in {"granite_swa", "granitemoe_swa"}:
            layer_types = tuple(
                "full_attention" if i % 4 == 0 else "sliding_attention"
                for i in range(num_layers)
            )
        if not layer_types and recipe_id in {"exaone4", "exaone_moe"}:
            sw_pattern = raw.get("sliding_window_pattern") or 4
            if isinstance(sw_pattern, str):
                pattern = sw_pattern.upper()
                cycling = (pattern * ((num_layers // max(len(pattern), 1)) + 1))[:num_layers]
                layer_types = tuple(
                    "sliding_attention" if ch == "L" else "full_attention" for ch in cycling
                )
            else:
                step = int(sw_pattern) or 4
                layer_types = tuple(
                    "sliding_attention" if (i + 1) % step != 0 else "full_attention"
                    for i in range(num_layers)
                )
        if not layer_types and (mt.lower() in {"cohere2"} or recipe_id == "cohere2_moe"):
            sw_pattern = int(raw.get("sliding_window_pattern") or 4)
            layer_types = tuple(
                "sliding_attention" if (i + 1) % sw_pattern else "full_attention"
                for i in range(num_layers)
            )
        if not layer_types and recipe_id == "smollm3" and no_rope_layers:
            sw = sliding_i
            use_sw = bool(raw.get("use_sliding_window")) and sw is not None
            types: list[str] = []
            for i, has_rope in enumerate(no_rope_layers):
                if use_sw and not has_rope:
                    types.append("sliding_attention")
                else:
                    types.append("full_attention")
            layer_types = tuple(types)

        layer_rope_raw = raw.get("layer_rope_theta")
        layer_rope_theta = tuple(float(x) for x in (layer_rope_raw or ()))
        if not no_rope_layers and layer_rope_theta:
            no_rope_layers = tuple(int(t != 0.0) for t in layer_rope_theta)
        if (
            not no_rope_layers
            and layer_types
            and sliding_i
            and (
                mt.lower() in {"cohere2"}
                or recipe_id in {"exaone4", "exaone_moe", "cohere2_moe"}
            )
        ):
            no_rope_layers = tuple(
                int("sliding" in str(t).lower()) for t in layer_types
            )
        if not layer_rope_theta and layer_types:
            rp = raw.get("rope_parameters") or raw.get("rope_scaling") or {}
            sliding_th = None
            full_th = float(raw.get("rope_theta", 10000.0))
            if isinstance(rp, dict):
                sliding = rp.get("sliding_attention")
                if isinstance(sliding, dict) and sliding.get("rope_theta") is not None:
                    sliding_th = float(sliding["rope_theta"])
                full = rp.get("full_attention")
                if isinstance(full, dict) and full.get("rope_theta") is not None:
                    full_th = float(full["rope_theta"])
            if sliding_th is None and recipe_id == "gemma3" and any(
                "sliding" in str(t).lower() for t in layer_types
            ):
                sliding_th = 10000.0
            if sliding_th is not None:
                layer_rope_theta = tuple(
                    sliding_th if "sliding" in str(t).lower() else full_th
                    for t in layer_types
                )

        mt = str(raw.get("model_type") or "")
        if recipe_id in {"gemma2", "gemma3"} or (
            recipe_id == "glm" and (mt.startswith("glm4") or any("glm4" in str(a).lower() for a in arches))
        ):
            residual_kind = "gemma2"
        elif recipe_id == "olmo_hybrid":
            residual_kind = "olmo_hybrid"
        elif recipe_id in {"olmo2", "olmo3", "flex_olmo", "exaone4"}:
            residual_kind = "post_norm"
        elif recipe_id == "gpt_neox" and bool(
            raw.get("use_parallel_residual", True)
        ):
            # GPT-NeoX keeps a second layernorm for the parallel FFN branch;
            # Phi / StableLM / Cohere reuse the one normed hidden state.
            residual_kind = "parallel_dual"
        elif recipe_id in {"cohere", "cohere2_moe", "phi", "gptj"} or bool(raw.get("use_parallel_residual")):
            residual_kind = "parallel"
        elif recipe_id == "falcon" and bool(raw.get("new_decoder_architecture", False)):
            residual_kind = "parallel_dual"
        elif recipe_id == "falcon" and bool(raw.get("parallel_attn", True)):
            residual_kind = "parallel"
        elif recipe_id == "opt" and not bool(raw.get("do_layer_norm_before", True)):
            residual_kind = "post_norm"
        else:
            residual_kind = "sequential"

        granite_ids = {"granite", "granite_swa", "granitemoe", "granitemoe_swa", "granitemoeshared"}
        granite_default = 1.0 if recipe_id in granite_ids else None
        attn_mul = raw.get("attention_multiplier", granite_default)
        if recipe_id == "gpt_neo" and "attention_multiplier" not in raw:
            attn_mul = 1.0
        falcon_alibi = bool(raw.get("alibi", False)) if recipe_id == "falcon" else False
        use_alibi = recipe_id in {"bloom", "mpt"} or falcon_alibi
        if recipe_id == "mpt":
            alibi_kind = "mpt"
        elif falcon_alibi:
            alibi_kind = "falcon"
        else:
            alibi_kind = "bloom"
        embed_norm = recipe_id == "bloom"
        pos_offset = 2 if recipe_id == "opt" else 0
        norm_has_bias = recipe_id not in {"mpt", "dbrx"}
        attn_sub_norm = recipe_id == "bitnet"
        ffn_sub_norm = recipe_id == "bitnet"
        q_pre = raw.get("query_pre_attn_scalar")
        clip = raw.get("clip_qkv")
        attn_soft = raw.get("attn_logit_softcapping")
        final_soft = raw.get("final_logit_softcapping")
        logit_scale_default = 0.0625 if recipe_id in {"cohere", "cohere2_moe"} else 1.0
        n_routed = n_routed or _opt_int(raw, "num_experts")

        partial = cls(
            vocab_size=int(raw["vocab_size"]),
            hidden_size=hidden_size,
            intermediate_size=int(raw.get("intermediate_size", 0)),
            num_hidden_layers=num_layers,
            num_attention_heads=nq,
            num_key_value_heads=_kv_heads(recipe_id, raw, nq),
            head_dim=head_dim,
            rms_norm_eps=eps,
            rope_theta=float(raw.get("rope_theta", 10000.0)),
            max_position_embeddings=int(raw.get("max_position_embeddings", 2048)),
            tie_word_embeddings=bool(raw.get("tie_word_embeddings", recipe_id == "gpt2")),
            torch_dtype=dtype,
            hidden_act=hidden_act,
            attention_bias=attn_bias,
            mlp_bias=mlp_bias,
            bos_token_id=raw.get("bos_token_id"),
            eos_token_id=raw.get("eos_token_id"),
            rope_scaling=raw.get("rope_scaling") or raw.get("rope_parameters"),
            model_type=str(raw.get("model_type", "llama")),
            recipe_id=recipe_id,
            architectures=tuple(arches),
            layers=(),
            hybrid_override_pattern=hybrid_pattern,
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
            or _opt_int(raw, "mtp_num_layers")
            or _opt_int(raw, "mtp_num_hidden_layers")
            or _opt_int(raw, "num_mtp_layers"),
            sliding_window=sliding_i,
            qk_norm=qk_norm,
            intermediate_size_mlp=_opt_int(raw, "intermediate_size_mlp"),
            layer_types=layer_types,
            attn_output_gate=bool(
                raw.get(
                    "attn_output_gate",
                    recipe_id in {"qwen3_5", "qwen3_next", "qwen3_5_moe"},
                )
            ),
            partial_rotary_factor=float(partial_rotary),
            linear_num_key_heads=_opt_int(raw, "linear_num_key_heads")
            or (16 if linear_defaults else None),
            linear_num_value_heads=_opt_int(raw, "linear_num_value_heads")
            or (32 if linear_defaults else None),
            linear_key_head_dim=_opt_int(raw, "linear_key_head_dim")
            or (128 if linear_defaults else None),
            linear_value_head_dim=_opt_int(raw, "linear_value_head_dim")
            or (128 if linear_defaults else None),
            linear_conv_kernel_dim=_opt_int(raw, "linear_conv_kernel_dim")
            or (4 if linear_defaults else None),
            embedding_multiplier=float(raw.get("embedding_multiplier", 1.0) or 1.0),
            residual_multiplier=float(raw.get("residual_multiplier", 1.0) or 1.0),
            attention_multiplier=(float(attn_mul) if attn_mul is not None else None),
            logits_scaling=float(raw.get("logits_scaling", 1.0) or 1.0),
            logit_scale=float(raw.get("logit_scale", logit_scale_default) or logit_scale_default),
            attn_logit_softcapping=(float(attn_soft) if attn_soft is not None else None),
            final_logit_softcapping=(float(final_soft) if final_soft is not None else None),
            query_pre_attn_scalar=(float(q_pre) if q_pre is not None else None),
            clip_qkv=(float(clip) if clip is not None else None),
            residual_kind=residual_kind,
            no_rope_layers=no_rope_layers,
            layer_rope_theta=layer_rope_theta,
            attention_out_bias=attn_out_bias,
            norm_topk_prob=bool(
                raw.get(
                    "norm_topk_prob",
                    recipe_id
                    not in {"qwen2_moe", "qwen3_moe", "olmoe", "flex_olmo", "jamba"},
                )
            ),
            alibi=use_alibi,
            alibi_kind=alibi_kind,
            embed_norm=embed_norm,
            pos_offset=pos_offset,
            norm_has_bias=norm_has_bias,
            attn_sub_norm=attn_sub_norm,
            ffn_sub_norm=ffn_sub_norm,
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
        ln = self.norm_kind in {"layer", "layer_1p"}
        ln_bias = ln and bool(getattr(self, "norm_has_bias", True))
        skip_all_norms = self.norm_kind == "olmo"
        residual_kind = getattr(self, "residual_kind", "sequential") or "sequential"
        skip_input = skip_all_norms or residual_kind == "post_norm"
        skip_post_attn = skip_all_norms or residual_kind == "parallel"
        extra_pre_ff = residual_kind == "gemma2"
        extra_post_ff = residual_kind in {"gemma2", "post_norm"}
        bias = self.attention_bias
        mlp_bias = self.mlp_bias

        shapes: dict[str, tuple[int, ...]] = {
            "embed.weight": (v, h),
        }
        if not skip_all_norms:
            shapes["final_norm.weight"] = (h,)
            if ln_bias:
                shapes["final_norm.bias"] = (h,)
        if self.embed_norm:
            shapes["embed_norm.weight"] = (h,)
            if ln_bias:
                shapes["embed_norm.bias"] = (h,)
        if self.pos_kind == "learned":
            shapes["pos_embed.weight"] = (self.max_position_embeddings + int(self.pos_offset), h)
        if not self.tie_word_embeddings:
            shapes["lm_head.weight"] = (v, h)
        if self.recipe_id in {"phi", "gptj"}:
            shapes["lm_head.bias"] = (v,)

        for spec in schedule:
            p = f"layers.{spec.index}"
            layer_residual = residual_kind
            if residual_kind == "olmo_hybrid":
                layer_residual = (
                    "sequential"
                    if spec.mixer == MixerKind.GATED_DELTANET
                    else "post_norm"
                )
            skip_input_l = skip_all_norms or layer_residual == "post_norm"
            skip_post_attn_l = skip_all_norms or layer_residual == "parallel"
            extra_pre_ff_l = layer_residual == "gemma2"
            extra_post_ff_l = layer_residual in {"gemma2", "post_norm"}
            if (
                not skip_input_l
                and (spec.mixer != MixerKind.NONE or spec.ffn != FfnKind.NONE)
            ):
                shapes[f"{p}.input_norm.weight"] = (h,)
                if ln_bias:
                    shapes[f"{p}.input_norm.bias"] = (h,)

            if spec.mixer == MixerKind.ATTENTION:
                shapes.update(self._attn_shapes(p, nq, nkv, dh, h, bias))
            elif spec.mixer == MixerKind.MAMBA2:
                shapes.update(self._mamba_shapes(p))
            elif spec.mixer == MixerKind.MAMBA1:
                shapes.update(self._mamba1_shapes(p))
            elif spec.mixer == MixerKind.GATED_DELTANET:
                shapes.update(self._gdn_shapes(p))

            if spec.ffn == FfnKind.DENSE_MLP:
                if spec.mixer != MixerKind.NONE and not skip_post_attn_l:
                    shapes[f"{p}.post_attn_norm.weight"] = (h,)
                    if ln_bias:
                        shapes[f"{p}.post_attn_norm.bias"] = (h,)
                if extra_pre_ff_l:
                    shapes[f"{p}.pre_ff_norm.weight"] = (h,)
                    if ln_bias:
                        shapes[f"{p}.pre_ff_norm.bias"] = (h,)
                if extra_post_ff_l:
                    shapes[f"{p}.post_ff_norm.weight"] = (h,)
                    if ln_bias:
                        shapes[f"{p}.post_ff_norm.bias"] = (h,)
                shapes.update(self._mlp_shapes(p, h, i, mlp_bias))
                if (
                    self.recipe_id in {"nemotron_h", "nemotron"}
                    or not self.uses_swiglu
                    or (self.mlp_hidden_act and self.mlp_hidden_act != "silu")
                ):
                    shapes.pop(f"{p}.mlp.gate.weight", None)
                    shapes.pop(f"{p}.mlp.gate.bias", None)
            elif spec.ffn == FfnKind.MOE:
                if spec.mixer != MixerKind.NONE and not skip_post_attn_l:
                    shapes[f"{p}.post_attn_norm.weight"] = (h,)
                    if ln_bias:
                        shapes[f"{p}.post_attn_norm.bias"] = (h,)
                if extra_post_ff_l:
                    shapes[f"{p}.post_ff_norm.weight"] = (h,)
                    if ln_bias:
                        shapes[f"{p}.post_ff_norm.bias"] = (h,)
                shapes.update(self._moe_shapes(p))
            if self.attn_sub_norm and spec.mixer == MixerKind.ATTENTION:
                shapes[f"{p}.attn.sub_norm.weight"] = (h,)
            if self.ffn_sub_norm and spec.ffn == FfnKind.DENSE_MLP:
                shapes[f"{p}.mlp.sub_norm.weight"] = (self.intermediate_size_mlp or i,)

        # Qwen3.5's complete MTP transformer is loaded by engine.mtp, not as
        # three legacy inline tensors in the target backbone state.
        n_mtp = 0 if self.recipe_id == "qwen3_5" else (self.num_nextn_predict_layers or 0)
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
        if kind == "gpt_bigcode":
            kv = nkv * dh
            return {
                f"{p}.attn.c_attn.weight": (h + 2 * kv, h),
                f"{p}.attn.c_attn.bias": (h + 2 * kv,),
                f"{p}.attn.c_proj.weight": (h, h),
                f"{p}.attn.c_proj.bias": (h,),
            }
        if kind == "fused_qkv":
            raw = self.raw or {}
            if self.recipe_id in {"gpt_neox", "bloom"}:
                qkv = (3 * nq * dh, h)
            elif self.recipe_id == "falcon":
                new_dec = bool(raw.get("new_decoder_architecture", False))
                multi_q = bool(raw.get("multi_query", True))
                if new_dec:
                    qkv = ((nq + 2 * nkv) * dh, h)
                elif multi_q:
                    qkv = ((nq + 2) * dh, h)
                else:
                    qkv = (3 * nq * dh, h)
            elif self.recipe_id == "mpt":
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
        if self.attention_out_bias:
            shapes[f"{p}.attn.o.bias"] = (h,)
        if self.qk_norm:
            if self.recipe_id in {"olmo2", "olmo3", "olmoe", "flex_olmo", "olmo_hybrid"}:
                shapes[f"{p}.attn.q_norm.weight"] = (nq * dh,)
                shapes[f"{p}.attn.k_norm.weight"] = (nkv * dh,)
            else:
                shapes[f"{p}.attn.q_norm.weight"] = (dh,)
                shapes[f"{p}.attn.k_norm.weight"] = (dh,)
            if self.recipe_id == "phi":
                shapes[f"{p}.attn.q_norm.bias"] = (dh,)
                shapes[f"{p}.attn.k_norm.bias"] = (dh,)
        if kind == "gqa_sinks":
            shapes[f"{p}.attn.sinks"] = (nq,)
        if kind == "diff":
            shapes[f"{p}.attn.lambda_q1"] = (dh,)
            shapes[f"{p}.attn.lambda_k1"] = (dh,)
            shapes[f"{p}.attn.lambda_q2"] = (dh,)
            shapes[f"{p}.attn.lambda_k2"] = (dh,)
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
        common = {
            f"{p}.gdn.conv1d.weight": (conv_dim, 1, k),
            f"{p}.gdn.A_log": (nv,),
            f"{p}.gdn.dt_bias": (nv,),
            f"{p}.gdn.norm.weight": (self.linear_value_head_dim,),
            f"{p}.gdn.out_proj.weight": (h, value_dim),
        }
        if self.recipe_id == "olmo_hybrid":
            return {
                f"{p}.gdn.q.weight": (key_dim, h),
                f"{p}.gdn.k.weight": (key_dim, h),
                f"{p}.gdn.v.weight": (value_dim, h),
                f"{p}.gdn.in_proj_z.weight": (value_dim, h),
                f"{p}.gdn.in_proj_b.weight": (nv, h),
                f"{p}.gdn.in_proj_a.weight": (nv, h),
                **common,
            }
        if self.recipe_id == "qwen3_next":
            return {
                f"{p}.gdn.in_proj_qkvz.weight": (key_dim * 2 + value_dim * 2, h),
                f"{p}.gdn.in_proj_ba.weight": (nv * 2, h),
                **common,
            }
        return {
            f"{p}.gdn.in_proj_qkv.weight": (conv_dim, h),
            f"{p}.gdn.in_proj_z.weight": (value_dim, h),
            f"{p}.gdn.in_proj_b.weight": (nv, h),
            f"{p}.gdn.in_proj_a.weight": (nv, h),
            **common,
        }

    def _mlp_shapes(
        self, p: str, h: int, i: int, mlp_bias: bool
    ) -> dict[str, tuple[int, ...]]:
        if self.recipe_id in {"phi3", "phi4", "glm"}:
            return {
                f"{p}.mlp.gate_up.weight": (2 * i, h),
                f"{p}.mlp.down.weight": (h, i),
            }
        if self.recipe_id in {"gpt2", "starcoder2", "gpt_bigcode"}:
            return {
                f"{p}.mlp.c_fc.weight": (i, h),
                f"{p}.mlp.c_fc.bias": (i,),
                f"{p}.mlp.c_proj.weight": (h, i),
                f"{p}.mlp.c_proj.bias": (h,),
            }
        if self.recipe_id in {
            "gpt_neox",
            "gptj",
            "gpt_neo",
            "opt",
            "bloom",
            "falcon",
            "mpt",
            "nemotron",
        }:
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

    def _mamba1_shapes(self, p: str) -> dict[str, tuple[int, ...]]:
        raw = self.raw or {}
        h = self.hidden_size
        expand = int(raw.get("mamba_expand") or 2)
        inter = expand * h
        state = int(raw.get("mamba_d_state") or self.ssm_state_size or 16)
        kernel = int(raw.get("mamba_d_conv") or self.conv_kernel or 4)
        dt_rank = raw.get("mamba_dt_rank") or "auto"
        if dt_rank == "auto" or dt_rank is None:
            dt_rank = max(int((h + 15) // 16), 1)
        else:
            dt_rank = int(dt_rank)
        use_conv_bias = bool(raw.get("mamba_conv_bias", True))
        use_proj_bias = bool(raw.get("mamba_proj_bias", False))
        out: dict[str, tuple[int, ...]] = {
            f"{p}.mamba1.in_proj.weight": (2 * inter, h),
            f"{p}.mamba1.out_proj.weight": (h, inter),
            f"{p}.mamba1.conv1d.weight": (inter, 1, kernel),
            f"{p}.mamba1.x_proj.weight": (dt_rank + 2 * state, inter),
            f"{p}.mamba1.dt_proj.weight": (inter, dt_rank),
            f"{p}.mamba1.dt_proj.bias": (inter,),
            f"{p}.mamba1.A_log": (inter, state),
            f"{p}.mamba1.D": (inter,),
            f"{p}.mamba1.dt_norm.weight": (dt_rank,),
            f"{p}.mamba1.b_norm.weight": (state,),
            f"{p}.mamba1.c_norm.weight": (state,),
        }
        if use_conv_bias:
            out[f"{p}.mamba1.conv1d.bias"] = (inter,)
        if use_proj_bias:
            out[f"{p}.mamba1.in_proj.bias"] = (2 * inter,)
            out[f"{p}.mamba1.out_proj.bias"] = (h,)
        return out

    def _moe_shapes(self, p: str) -> dict[str, tuple[int, ...]]:
        assert self.n_routed_experts is not None
        n_e = self.n_routed_experts
        h = self.hidden_size
        mi = self.moe_intermediate_size or self.intermediate_size
        kind = self.moe_kind
        latent = self.moe_latent_size or h

        if kind == "dbrx":
            return {
                f"{p}.moe.gate.weight": (n_e, h),
                f"{p}.moe.experts.w1.weight": (n_e * mi, h),
                f"{p}.moe.experts.v1.weight": (n_e * mi, h),
                f"{p}.moe.experts.w2.weight": (n_e * mi, h),
            }

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
            if self.recipe_id in {
                "qwen3_moe",
                "qwen2_moe",
                "olmoe",
                "flex_olmo",
                "granitemoe",
                "granitemoe_swa",
                "granitemoeshared",
                "phimoe",
                "hunyuan_v1_moe",
                "ernie4_5_moe",
                "cohere2_moe",
                "qwen3_next",
                "qwen3_5_moe",
                "jamba",
            }:
                # Transformers 5.x packed expert tensors: [E, 2I, H] / [E, H, I]
                shapes[f"{p}.moe.experts.gate_up.weight"] = (n_e, 2 * mi, h)
                shapes[f"{p}.moe.experts.down.weight"] = (n_e, h, mi)
                if self.recipe_id in {"qwen2_moe", "qwen3_next", "qwen3_5_moe"}:
                    si = self.moe_shared_expert_intermediate_size or self.intermediate_size
                    shapes[f"{p}.moe.shared.gate.weight"] = (si, h)
                    shapes[f"{p}.moe.shared.up.weight"] = (si, h)
                    shapes[f"{p}.moe.shared.down.weight"] = (h, si)
                    shapes[f"{p}.moe.shared_gate.weight"] = (1, h)
                if self.recipe_id == "cohere2_moe" and self.n_shared_experts:
                    si = self.moe_shared_expert_intermediate_size or (
                        self.intermediate_size * int(self.n_shared_experts)
                    )
                    shapes[f"{p}.moe.shared.gate.weight"] = (si, h)
                    shapes[f"{p}.moe.shared.up.weight"] = (si, h)
                    shapes[f"{p}.moe.shared.down.weight"] = (h, si)
                if self.recipe_id in {"granitemoeshared", "granitemoe_swa"}:
                    si = self.moe_shared_expert_intermediate_size or 0
                    if si:
                        shapes[f"{p}.moe.shared.gate_up.weight"] = (2 * si, h)
                        shapes[f"{p}.moe.shared.down.weight"] = (h, si)
                if self.recipe_id == "ernie4_5_moe":
                    shapes[f"{p}.moe.gate.e_score_correction_bias"] = (n_e,)
                if self.recipe_id in {"ernie4_5_moe", "hunyuan_v1_moe"} and self.n_shared_experts:
                    si = self.moe_shared_expert_intermediate_size or (
                        self.intermediate_size * int(self.n_shared_experts)
                    )
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
            if "e_score_correction_bias" in str(self.raw) or self.recipe_id in {
                "glm4_moe",
                "exaone_moe",
            }:
                shapes[f"{p}.moe.gate.e_score_correction_bias"] = (n_e,)
            if self.recipe_id in {"glm4_moe", "exaone_moe"}:
                shapes[f"{p}.moe.experts.gate_up.weight"] = (n_e, 2 * mi, h)
                shapes[f"{p}.moe.experts.down.weight"] = (n_e, h, mi)
            else:
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
        outer_model_type = str(raw.get("model_type", ""))
        if "hidden_size" not in raw or any(
            outer_model_type.startswith(p)
            for p in (
                "qwen3_5",
                "qwen3_5_moe",
                "gemma3",
                "mistral3",
                "internvl",
                "qwen2_vl",
                "qwen2_5_vl",
                "kimi",
                "phi4_multimodal",
                "exaone4_5",
            )
        ):
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
        "ffn_dim": "intermediate_size",
        "ffn_hidden_size": "intermediate_size",
        "num_local_experts": "n_routed_experts",
        "num_experts": "n_routed_experts",
        "moe_num_experts": "n_routed_experts",
        "moe_k": "num_experts_per_tok",
        "moe_topk": "num_experts_per_tok",
        "moe_top_k": "num_experts_per_tok",
        "moe_num_experts": "n_routed_experts",
        "moe_num_shared_experts": "n_shared_experts",
        "num_shared_experts": "n_shared_experts",
        "shared_expert_intermediate_size": "moe_shared_expert_intermediate_size",
        "shared_intermediate_size": "moe_shared_expert_intermediate_size",
        "norm_epsilon": "layer_norm_epsilon",
        "n_ctx": "max_position_embeddings",
        "max_seq_len": "max_position_embeddings",
        "d_model": "hidden_size",
        "n_embed": "hidden_size",
        "num_heads": "num_attention_heads",
        "n_heads": "num_attention_heads",
        "num_layers": "num_hidden_layers",
        "n_layers": "num_hidden_layers",
        "num_kv_heads": "num_key_value_heads",
        "window_size": "sliding_window",
        "mamba_d_state": "ssm_state_size",
        "mamba_d_conv": "conv_kernel",
        "ffn_hidden_size": "moe_intermediate_size",
    }
    for src, dst in aliases.items():
        if src in raw and (dst not in raw or raw[dst] in (None, 0)):
            raw[dst] = raw[src]
    if not raw.get("intermediate_size") and raw.get("hidden_size"):
        exp = raw.get("expansion_ratio")
        raw["intermediate_size"] = (
            int(exp) * int(raw["hidden_size"]) if exp else 4 * int(raw["hidden_size"])
        )
    attn_cfg = raw.get("attn_config")
    if isinstance(attn_cfg, dict):
        if raw.get("clip_qkv") is None and attn_cfg.get("clip_qkv") is not None:
            raw["clip_qkv"] = attn_cfg["clip_qkv"]
        if raw.get("alibi_bias_max") is None and attn_cfg.get("alibi_bias_max") is not None:
            raw["alibi_bias_max"] = attn_cfg["alibi_bias_max"]
        if raw.get("softmax_scale") is None and attn_cfg.get("softmax_scale") is not None:
            raw["attention_multiplier"] = attn_cfg["softmax_scale"]
        if raw.get("num_key_value_heads") is None and attn_cfg.get("kv_n_heads") is not None:
            raw["num_key_value_heads"] = attn_cfg["kv_n_heads"]
        if raw.get("rope_theta") is None and attn_cfg.get("rope_theta") is not None:
            raw["rope_theta"] = attn_cfg["rope_theta"]
    ffn_cfg = raw.get("ffn_config")
    if isinstance(ffn_cfg, dict):
        if raw.get("moe_num_experts") is None and ffn_cfg.get("moe_num_experts") is not None:
            raw["moe_num_experts"] = ffn_cfg["moe_num_experts"]
        if raw.get("n_routed_experts") is None and ffn_cfg.get("moe_num_experts") is not None:
            raw["n_routed_experts"] = ffn_cfg["moe_num_experts"]
        if raw.get("moe_top_k") is None and ffn_cfg.get("moe_top_k") is not None:
            raw["moe_top_k"] = ffn_cfg["moe_top_k"]
        if raw.get("num_experts_per_tok") is None and ffn_cfg.get("moe_top_k") is not None:
            raw["num_experts_per_tok"] = ffn_cfg["moe_top_k"]
        if not raw.get("intermediate_size") and ffn_cfg.get("ffn_hidden_size"):
            raw["intermediate_size"] = ffn_cfg["ffn_hidden_size"]
        if raw.get("moe_intermediate_size") is None and ffn_cfg.get("ffn_hidden_size"):
            raw["moe_intermediate_size"] = ffn_cfg["ffn_hidden_size"]
        if raw.get("moe_normalize_expert_weights") is None:
            raw["moe_normalize_expert_weights"] = ffn_cfg.get("moe_normalize_expert_weights", 1.0)
    if "attention_bias" not in raw and "bias" in raw:
        raw["attention_bias"] = bool(raw["bias"])
        raw.setdefault("mlp_bias", bool(raw["bias"]))
    if "num_key_value_heads" not in raw:
        if raw.get("num_kv_heads") is not None:
            raw["num_key_value_heads"] = raw["num_kv_heads"]
        elif raw.get("multi_query"):
            raw["num_key_value_heads"] = 1
        elif "num_attention_heads" in raw:
            raw["num_key_value_heads"] = raw["num_attention_heads"]
    if "vocab_size" not in raw and isinstance(raw.get("tokenizer.ggml.tokens"), list):
        raw["vocab_size"] = len(raw["tokenizer.ggml.tokens"])
    if "torch_dtype" not in raw and isinstance(raw.get("dtype"), str):
        raw["torch_dtype"] = raw["dtype"]
    rp = raw.get("rope_parameters")
    if isinstance(rp, dict):
        if rp.get("rope_theta") is not None:
            raw["rope_theta"] = rp["rope_theta"]
        else:
            nested = rp.get("full_attention")
            if isinstance(nested, dict) and nested.get("rope_theta") is not None:
                raw["rope_theta"] = nested["rope_theta"]
        if rp.get("partial_rotary_factor") is not None and "partial_rotary_factor" not in raw:
            raw["partial_rotary_factor"] = rp["partial_rotary_factor"]
    return raw


def _kv_heads(recipe_id: str, raw: dict[str, Any], nq: int) -> int:
    if recipe_id in {"gptj", "gpt_neo", "opt", "bloom", "mpt"}:
        return nq
    if recipe_id == "falcon" and raw.get("multi_query") and not raw.get(
        "new_decoder_architecture"
    ):
        return 1
    if recipe_id == "gpt_bigcode" and raw.get("multi_query", True):
        return 1
    return int(raw.get("num_key_value_heads", nq))


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
