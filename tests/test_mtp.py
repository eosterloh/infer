"""Qwen3.5 native MTP draft layer."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from engine.config import ModelConfig
from engine.generate import generate_greedy, generate_mtp_greedy
from engine.model import DecoderModel
from engine.mtp import Qwen35MTP
from engine.layers.norm import gemma_rms_norm
from engine.synth import random_engine_weights, write_config
from engine.tokenizer import Tokenizer


def _config(tmp_path: Path) -> ModelConfig:
    raw = {
        "architectures": ["Qwen3_5ForCausalLM"],
        "model_type": "qwen3_5",
        "vocab_size": 48,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "max_position_embeddings": 64,
        "tie_word_embeddings": False,
        "torch_dtype": "float32",
        "layer_types": ["full_attention"],
        "attn_output_gate": True,
        "partial_rotary_factor": 0.5,
        "mtp_num_hidden_layers": 1,
    }
    return ModelConfig.from_pretrained(write_config(tmp_path, raw))


def _mtp_hf_weights(
    cfg: ModelConfig,
    target: dict[str, torch.Tensor],
    *,
    source_layer: int = 0,
) -> dict[str, torch.Tensor]:
    h = cfg.hidden_size
    result = {
        "mtp.fc.weight": torch.randn(h, 2 * h) * 0.02,
        "mtp.norm.weight": torch.zeros(h),
        "mtp.pre_fc_norm_embedding.weight": torch.zeros(h),
        "mtp.pre_fc_norm_hidden.weight": torch.zeros(h),
    }
    reverse = {
        "input_norm.weight": "input_layernorm.weight",
        "post_attn_norm.weight": "post_attention_layernorm.weight",
        "attn.q.weight": "self_attn.q_proj.weight",
        "attn.k.weight": "self_attn.k_proj.weight",
        "attn.v.weight": "self_attn.v_proj.weight",
        "attn.o.weight": "self_attn.o_proj.weight",
        "attn.q_norm.weight": "self_attn.q_norm.weight",
        "attn.k_norm.weight": "self_attn.k_norm.weight",
        "mlp.gate.weight": "mlp.gate_proj.weight",
        "mlp.up.weight": "mlp.up_proj.weight",
        "mlp.down.weight": "mlp.down_proj.weight",
    }
    for name, tensor in target.items():
        layer_prefix = f"layers.{source_layer}."
        if not name.startswith(layer_prefix):
            continue
        rest = name[len(layer_prefix) :]
        if rest in reverse:
            result[f"mtp.layers.0.{reverse[rest]}"] = tensor.clone()
    return result


def test_mtp_prefill_decode_matches_full(tmp_path: Path) -> None:
    torch.manual_seed(3)
    cfg = _config(tmp_path)
    target = random_engine_weights(cfg)
    mtp = Qwen35MTP(cfg, target, _mtp_hf_weights(cfg, target))
    ids = torch.randint(0, cfg.vocab_size, (1, 5))
    previous = torch.randn(1, 5, cfg.hidden_size)

    full_logits, full_hidden = mtp.forward(ids, previous)
    embedded_logits, embedded_hidden = mtp.forward(
        ids, previous, input_embeddings=target["embed.weight"][ids]
    )
    assert torch.equal(full_logits, embedded_logits)
    assert torch.equal(full_hidden, embedded_hidden)
    cache = mtp.make_cache()
    logits_a, hidden_a = mtp.forward(ids[:, :4], previous[:, :4], cache=cache)
    logits_b, hidden_b = mtp.forward(ids[:, 4:], previous[:, 4:], cache=cache)

    assert torch.allclose(
        full_logits, torch.cat((logits_a, logits_b), dim=1), atol=1e-4, rtol=1e-4
    )
    assert torch.allclose(
        full_hidden, torch.cat((hidden_a, hidden_b), dim=1), atol=1e-4, rtol=1e-4
    )


def test_target_exposes_pre_final_norm_hidden(tmp_path: Path) -> None:
    torch.manual_seed(6)
    cfg = _config(tmp_path)
    target = random_engine_weights(cfg)
    model = DecoderModel(cfg, target)
    ids = torch.randint(0, cfg.vocab_size, (1, 4))
    logits, hidden = model.forward(ids, return_hidden=True)
    normed = gemma_rms_norm(
        hidden, target["final_norm.weight"], cfg.rms_norm_eps
    )
    assert torch.allclose(logits, F.linear(normed, target["lm_head.weight"]))
    assert not torch.allclose(hidden, normed)


def test_mtp_greedy_is_lossless(tmp_path: Path) -> None:
    torch.manual_seed(8)
    cfg = _config(tmp_path)
    target = random_engine_weights(cfg)
    model = DecoderModel(cfg, target)
    mtp = Qwen35MTP(cfg, target, _mtp_hf_weights(cfg, target))
    tokenizer = Tokenizer.from_pretrained(tmp_path)

    expected = "".join(
        generate_greedy(
            model,
            tokenizer,
            "native mtp",
            max_new_tokens=8,
            apply_chat_template=False,
        )
    )
    actual = "".join(
        generate_mtp_greedy(
            model,
            mtp,
            tokenizer,
            "native mtp",
            max_new_tokens=8,
            num_speculative_tokens=3,
            apply_chat_template=False,
        )
    )
    assert actual == expected


def test_mtp_greedy_hybrid_rollback_is_lossless(tmp_path: Path) -> None:
    torch.manual_seed(12)
    raw = {
        "architectures": ["Qwen3_5ForCausalLM"],
        "model_type": "qwen3_5",
        "vocab_size": 48,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "max_position_embeddings": 64,
        "tie_word_embeddings": False,
        "torch_dtype": "float32",
        "layer_types": ["linear_attention", "full_attention"],
        "attn_output_gate": True,
        "partial_rotary_factor": 0.5,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "linear_key_head_dim": 8,
        "linear_value_head_dim": 8,
        "linear_conv_kernel_dim": 4,
        "mtp_num_hidden_layers": 1,
    }
    cfg = ModelConfig.from_pretrained(write_config(tmp_path, raw))
    target = random_engine_weights(cfg)
    model = DecoderModel(cfg, target)
    mtp = Qwen35MTP(
        cfg, target, _mtp_hf_weights(cfg, target, source_layer=1)
    )
    tokenizer = Tokenizer.from_pretrained(tmp_path)

    expected = "".join(
        generate_greedy(
            model,
            tokenizer,
            "hybrid",
            max_new_tokens=9,
            apply_chat_template=False,
        )
    )
    actual = "".join(
        generate_mtp_greedy(
            model,
            mtp,
            tokenizer,
            "hybrid",
            max_new_tokens=9,
            num_speculative_tokens=3,
            apply_chat_template=False,
        )
    )
    assert actual == expected

    short = torch.tensor([[5, 6, 7]])
    long = torch.tensor([[8, 9, 10, 11, 12]])
    padded = torch.tensor([[0, 0, 5, 6, 7], long[0].tolist()])
    mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
    batch_logits = model.forward(padded, attention_mask=mask)
    short_logits = model.forward(short)
    long_logits = model.forward(long)
    assert torch.allclose(batch_logits[0, -3:], short_logits[0], atol=1e-4, rtol=1e-4)
    assert torch.allclose(batch_logits[1], long_logits[0], atol=1e-4, rtol=1e-4)
