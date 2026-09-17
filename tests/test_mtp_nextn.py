"""DeepSeek V3 and Nemotron Super native MTP."""

from __future__ import annotations

from pathlib import Path

import torch

from engine.config import ModelConfig
from engine.detect import detect_missing
from engine.generate import generate_greedy, generate_mtp_greedy
from engine.model import DecoderModel
from engine.mtp import NextnMTP
from engine.synth import random_engine_weights, write_config
from engine.tokenizer import Tokenizer


def _deepseek_raw() -> dict:
    return {
        "architectures": ["DeepseekV3ForCausalLM"],
        "model_type": "deepseek_v3",
        "vocab_size": 48,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "head_dim": 8,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "max_position_embeddings": 64,
        "tie_word_embeddings": False,
        "torch_dtype": "float32",
        "q_lora_rank": 16,
        "kv_lora_rank": 16,
        "qk_nope_head_dim": 8,
        "qk_rope_head_dim": 8,
        "v_head_dim": 8,
        "first_k_dense_replace": 1,
        "num_nextn_predict_layers": 1,
    }


def _super_raw() -> dict:
    return {
        "architectures": ["NemotronHForCausalLM"],
        "model_type": "nemotron_h",
        "vocab_size": 48,
        "hidden_size": 16,
        "intermediate_size": 8,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "layer_norm_epsilon": 1e-5,
        "max_position_embeddings": 64,
        "tie_word_embeddings": False,
        "torch_dtype": "float32",
        "hybrid_override_pattern": "E",
        "n_routed_experts": 4,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 8,
        "moe_shared_expert_intermediate_size": 16,
        "moe_latent_size": 8,
        "routed_scaling_factor": 1.0,
        "mlp_hidden_act": "relu2",
        "n_group": 1,
        "topk_group": 1,
        "num_nextn_predict_layers": 1,
    }


def test_deepseek_mtp_not_missing() -> None:
    assert "mtp_decode" not in detect_missing(_deepseek_raw(), "deepseek-v3-mtp")


def test_deepseek_mtp_greedy_is_lossless(tmp_path: Path) -> None:
    torch.manual_seed(11)
    cfg = ModelConfig.from_pretrained(write_config(tmp_path, _deepseek_raw()))
    target = random_engine_weights(cfg)
    model = DecoderModel(cfg, target)
    mtp = NextnMTP(cfg, target)
    tokenizer = Tokenizer.from_pretrained(tmp_path)
    expected = "".join(
        generate_greedy(
            model, tokenizer, "nextn", max_new_tokens=8, apply_chat_template=False
        )
    )
    actual = "".join(
        generate_mtp_greedy(
            model,
            mtp,
            tokenizer,
            "nextn",
            max_new_tokens=8,
            num_speculative_tokens=3,
            apply_chat_template=False,
        )
    )
    assert actual == expected


def test_deepseek_mtp_prefill_matches_full(tmp_path: Path) -> None:
    torch.manual_seed(5)
    cfg = ModelConfig.from_pretrained(write_config(tmp_path, _deepseek_raw()))
    target = random_engine_weights(cfg)
    mtp = NextnMTP(cfg, target)
    ids = torch.randint(0, cfg.vocab_size, (1, 5))
    previous = torch.randn(1, 5, cfg.hidden_size)
    full_logits, full_hidden = mtp.forward(ids, previous)
    cache = mtp.make_cache()
    logits_a, hidden_a = mtp.forward(ids[:, :3], previous[:, :3], cache=cache)
    logits_b, hidden_b = mtp.forward(ids[:, 3:], previous[:, 3:], cache=cache)
    assert torch.allclose(
        full_logits, torch.cat((logits_a, logits_b), dim=1), atol=1e-4, rtol=1e-4
    )
    assert torch.allclose(
        full_hidden, torch.cat((hidden_a, hidden_b), dim=1), atol=1e-4, rtol=1e-4
    )


def test_super_mtp_greedy_is_lossless(tmp_path: Path) -> None:
    torch.manual_seed(9)
    cfg = ModelConfig.from_pretrained(write_config(tmp_path, _super_raw()))
    target = random_engine_weights(cfg)
    model = DecoderModel(cfg, target)
    mtp = NextnMTP(cfg, target)
    tokenizer = Tokenizer.from_pretrained(tmp_path)
    expected = "".join(
        generate_greedy(
            model, tokenizer, "super", max_new_tokens=7, apply_chat_template=False
        )
    )
    actual = "".join(
        generate_mtp_greedy(
            model,
            mtp,
            tokenizer,
            "super",
            max_new_tokens=7,
            num_speculative_tokens=2,
            apply_chat_template=False,
        )
    )
    assert actual == expected
