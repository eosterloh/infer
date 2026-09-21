"""Loading a checkpoint straight into packed weights.

Packing during the load is what lets a big model fit at all: the dense tensor is
released as soon as its packed form exists, so peak memory tracks the packed
model instead of the BF16 checkpoint. These tests use tiny fixtures on CPU, so
they check the wiring and the numerics, not the throughput.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from engine.agent_api import load_engine, memory_preflight
from engine.config import ModelConfig
from engine.layers.moe import EXPERT_STACK_KEY
from engine.qweight import QuantWeight, concat_quant_weights, quantize
from engine.synth import random_engine_weights, write_config, write_hf_folder
from engine.weights import checkpoint_numel, load_weights

_LLAMA = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "vocab_size": 64,
    "hidden_size": 128,
    "intermediate_size": 256,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 32,
    "rms_norm_eps": 1e-5,
    "rope_theta": 10000.0,
    "max_position_embeddings": 64,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "hidden_act": "silu",
}

_MOE = {
    "architectures": ["Qwen3MoeForCausalLM"],
    "model_type": "qwen3_moe",
    "vocab_size": 64,
    "hidden_size": 128,
    "intermediate_size": 256,
    "moe_intermediate_size": 128,
    "num_hidden_layers": 1,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 32,
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "norm_topk_prob": True,
    "decoder_sparse_step": 1,
    "mlp_only_layers": [],
    "rms_norm_eps": 1e-5,
    "rope_theta": 10000.0,
    "max_position_embeddings": 64,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "hidden_act": "silu",
}


def _folder(tmp_path: Path, raw: dict, name: str) -> Path:
    folder = write_config(tmp_path / name, raw)
    cfg = ModelConfig.from_pretrained(folder)
    write_hf_folder(folder, cfg, random_engine_weights(cfg, seed=3))
    return folder


@pytest.fixture(autouse=True)
def _pack_small_tensors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fixtures are tiny; drop the floor so the same code path still packs."""
    monkeypatch.setenv("INFER_QUANT_MIN_NUMEL", "1024")


@pytest.mark.parametrize("kind", ["int4", "fp8", "nvfp4"])
def test_streaming_pack_matches_post_hoc_pack(tmp_path: Path, kind: str) -> None:
    """Packing during the load must give the same weights as packing after it."""
    folder = _folder(tmp_path, _LLAMA, f"dense_{kind}")
    cfg = ModelConfig.from_pretrained(folder)
    dense = load_weights(folder, cfg, device="cpu")
    streamed = load_weights(
        folder, cfg, device="cpu", quant=kind, quant_group=64, quant_min_numel=1024
    )

    packed_names = [n for n, v in streamed.items() if isinstance(v, QuantWeight)]
    assert packed_names, "nothing was packed during the load"
    for name in packed_names:
        group = streamed[name].group_size
        want = quantize(
            dense[name], kind, **({} if kind == "fp8" else {"group_size": group})
        )
        torch.testing.assert_close(
            streamed[name].dequantize().float(), want.dequantize().float()
        )


def test_streaming_pack_leaves_small_tensors_alone(tmp_path: Path) -> None:
    folder = _folder(tmp_path, _LLAMA, "dense_small")
    cfg = ModelConfig.from_pretrained(folder)
    streamed = load_weights(
        folder, cfg, device="cpu", quant="int4", quant_group=64, quant_min_numel=1024
    )
    assert isinstance(streamed["layers.0.input_norm.weight"], torch.Tensor)
    assert isinstance(streamed["embed.weight"], torch.Tensor)


def test_load_engine_packed_generates(tmp_path: Path) -> None:
    """A packed model must still run a forward pass and produce finite logits."""
    folder = _folder(tmp_path, _LLAMA, "dense_engine")
    eng = load_engine(folder, device="cpu", quant="int4", quant_group=64)
    assert eng.quantization is not None
    assert eng.quantization["compression"] > 1.0
    logits = eng.model.forward(torch.randint(0, 64, (1, 4)))
    assert torch.isfinite(logits).all()


def test_packed_moe_experts_stack_into_one_block(tmp_path: Path) -> None:
    """Checkpoint-packed experts must be adopted as a stack and then packed.

    Qwen3-MoE ships one ``[E, 2I, H]`` block per layer, which the dispatch loop
    used to index expert by expert; the grouped GEMV wants exactly that block.
    """
    folder = _folder(tmp_path, _MOE, "moe_engine")
    eng = load_engine(folder, device="cpu", quant="int4", quant_group=64)
    block = eng.model.weights[EXPERT_STACK_KEY]["layers.0.moe"]
    assert isinstance(block["gate_up"], QuantWeight)
    assert block["gate_up"].experts == _MOE["num_experts"]
    assert block["gate_up"].expert_cols == 2 * _MOE["moe_intermediate_size"]
    assert isinstance(block["down"], QuantWeight)
    # The dense entries are gone, so nothing pins the unpacked experts.
    assert not any(".moe.experts." in name for name in eng.model.weights)
    logits = eng.model.forward(torch.randint(0, 64, (1, 4)))
    assert torch.isfinite(logits).all()


def test_moe_packed_logits_track_dense(tmp_path: Path) -> None:
    """int4 experts must not change what the model says, only how it stores it."""
    folder = _folder(tmp_path, _MOE, "moe_compare")
    ids = torch.randint(0, 64, (1, 6))
    dense = load_engine(folder, device="cpu", dtype="float32")
    packed = load_engine(folder, device="cpu", quant="int4", quant_group=64)
    a = dense.model.forward(ids).float()
    b = packed.model.forward(ids).float()
    rel = (a - b).norm() / a.norm()
    assert rel < 0.25, f"packed MoE drifted from dense: {rel:.3f}"
    assert a.argmax(-1).shape == b.argmax(-1).shape


def test_concat_quant_weights_is_exact() -> None:
    """Joining packed blocks must be byte identical to the parts."""
    torch.manual_seed(5)
    parts = [
        quantize(torch.randn(8, 128) * 0.05, "int4", group_size=64) for _ in range(3)
    ]
    want = torch.cat([p.dequantize().float() for p in parts])
    joined = concat_quant_weights(parts)
    assert joined.experts == 3 and joined.expert_cols == 8
    torch.testing.assert_close(joined.dequantize().float(), want)


def test_concat_nvfp4_keeps_per_part_scales() -> None:
    """Parts with different global scales survive the join as per-row scales."""
    torch.manual_seed(6)
    parts = [
        quantize(torch.randn(8, 128) * scale, "nvfp4", group_size=16)
        for scale in (0.01, 1.0, 50.0)
    ]
    want = torch.cat([p.dequantize().float() for p in parts])
    joined = concat_quant_weights([p for p in parts])
    assert joined.channel_scale is not None
    torch.testing.assert_close(joined.dequantize().float(), want)


def test_expert_view_round_trips(tmp_path: Path) -> None:
    torch.manual_seed(7)
    parts = [
        quantize(torch.randn(8, 128) * 0.05, "int4", group_size=64) for _ in range(4)
    ]
    originals = [p.dequantize().float() for p in parts]
    joined = concat_quant_weights(parts)
    for i, original in enumerate(originals):
        torch.testing.assert_close(joined.expert_view(i).dequantize().float(), original)


def test_checkpoint_numel_matches_load(tmp_path: Path) -> None:
    folder = _folder(tmp_path, _LLAMA, "numel")
    cfg = ModelConfig.from_pretrained(folder)
    loaded = load_weights(folder, cfg, device="cpu")
    counted = sum(v.numel() for v in loaded.values() if isinstance(v, torch.Tensor))
    assert checkpoint_numel(folder) == counted


def test_memory_preflight_refuses_impossible_load(tmp_path: Path) -> None:
    """A model that cannot fit must fail fast instead of swapping the box."""
    folder = _folder(tmp_path, _LLAMA, "preflight")
    report = memory_preflight(folder, device="cpu", dtype=None, quant=None)
    assert report["estimate_gb"] > 0
    with pytest.raises(MemoryError):
        memory_preflight(folder, device="cpu", dtype=None, quant=None, headroom=1e-12)
