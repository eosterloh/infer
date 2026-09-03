"""End-to-end Qwen3.5 hybrid text logits against Transformers."""

from __future__ import annotations

import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

from engine.config import ModelConfig
from engine.maps import map_hf_name
from engine.model import DecoderModel
from engine.synth import write_config


def test_qwen35_hybrid_logits_match_transformers(tmp_path) -> None:
    torch.manual_seed(21)
    raw = {
        "architectures": ["Qwen3_5ForCausalLM"],
        "model_type": "qwen3_5",
        "vocab_size": 64,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "max_position_embeddings": 128,
        "tie_word_embeddings": False,
        "torch_dtype": "float32",
        "hidden_act": "silu",
        "layer_types": ["linear_attention", "full_attention"],
        "attn_output_gate": True,
        "partial_rotary_factor": 0.5,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 8,
        "linear_value_head_dim": 8,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
    }
    config = ModelConfig.from_pretrained(write_config(tmp_path, raw))
    hf_config = Qwen3_5TextConfig(**raw)
    hf_config._attn_implementation = "eager"
    reference = Qwen3_5ForCausalLM(hf_config).eval()

    weights: dict[str, torch.Tensor] = {}
    for name, tensor in reference.state_dict().items():
        mapped = map_hf_name(name, config)
        assert mapped is not None, name
        weights[mapped] = tensor.detach()
    assert set(weights) == set(config.expected_shapes())

    model = DecoderModel(config, weights)
    input_ids = torch.randint(0, config.vocab_size, (1, 7))
    with torch.inference_mode():
        expected = reference(input_ids, use_cache=False).logits
        actual = model.forward(input_ids)
    assert torch.allclose(actual, expected, atol=3e-5, rtol=3e-5)
    assert torch.equal(actual.argmax(dim=-1), expected.argmax(dim=-1))
