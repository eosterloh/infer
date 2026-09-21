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


def test_checkpoint_profile_reports_load_transients(tmp_path: Path) -> None:
    """The preflight needs the shard and one layer's experts, not just totals."""
    from engine.weights import checkpoint_profile

    folder = _folder(tmp_path, _MOE, "profile_moe")
    profile = checkpoint_profile(folder)
    assert profile["numel"] == checkpoint_numel(folder)
    # One shard here, so it accounts for the whole file.
    assert profile["shard_bytes"] == profile["bytes"]
    # Qwen3-MoE ships packed [E, 2I, H] / [E, H, I] expert blocks per layer, and
    # the fixture has one layer, so that is every expert byte in the file.
    width = profile["bytes"] // profile["numel"]
    per_expert = 3 * _MOE["moe_intermediate_size"] * _MOE["hidden_size"]
    assert profile["expert_layer_bytes"] == width * _MOE["num_experts"] * per_expert

    dense = checkpoint_profile(_folder(tmp_path, _LLAMA, "profile_dense"))
    assert dense["expert_layer_bytes"] == 0


def test_preflight_counts_the_stacking_transient(tmp_path: Path) -> None:
    """Stacking experts needs room for a second copy of one layer's worth."""
    folder = _folder(tmp_path, _MOE, "preflight_moe")
    with_stack = memory_preflight(folder, device="cpu", dtype=None, quant=None)
    assert with_stack["transient_gb"] > 0
    assert with_stack["estimate_gb"] > with_stack["resident_gb"]


def test_memory_floor_stops_a_load_before_the_pool_runs_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guard is what turns a frozen machine into a stack trace."""
    from engine import memory

    monkeypatch.setattr(memory, "available_bytes", lambda: int(1e9))
    monkeypatch.setenv("INFER_MEM_FLOOR_GB", "6")
    with pytest.raises(MemoryError, match="stopping before the pool"):
        memory.check_floor("shard 1/4")

    monkeypatch.setenv("INFER_MEM_FLOOR_GB", "0")
    memory.check_floor("shard 1/4")

    monkeypatch.setenv("INFER_MEM_FLOOR_GB", "6")
    monkeypatch.setattr(memory, "available_bytes", lambda: None)
    memory.check_floor("unknown platform")


def test_the_floor_reclaims_torch_reserve_before_it_refuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cached block is not a shortage on a pool shared with the host.

    Stacking a 30B checkpoint's experts frees one layer at a time, and every
    freed tensor stayed in torch's allocator, where MemAvailable stops counting
    it. The guard saw 5.7 GB and stopped a load that had 60 GB of its own reserve
    to give back.
    """
    from engine import memory

    reads = iter([int(1e9), int(80e9)])
    released: list[bool] = []
    monkeypatch.setattr(memory, "available_bytes", lambda: next(reads))
    monkeypatch.setattr(memory, "_release_cached_blocks", lambda: released.append(True))
    monkeypatch.setenv("INFER_MEM_FLOOR_GB", "6")
    memory.check_floor("stacking experts for layers.45.moe")
    assert released == [True], "the guard refused without reclaiming first"

    # Still short after reclaiming is a real shortage, and still raises.
    monkeypatch.setattr(memory, "available_bytes", lambda: int(1e9))
    with pytest.raises(MemoryError, match="stopping before the pool"):
        memory.check_floor("stacking experts for layers.45.moe")


def test_fused_path_chunks_past_the_kernels_row_budget() -> None:
    """The row cap is a measured choice, not the kernel's register count."""
    from engine.kernels import available
    from engine.qweight import KERNEL_ROW_LIMIT, fused_qlinear, python_dequantize, quantize

    if not available():
        pytest.skip("the fused path is the compiled op")

    torch.manual_seed(4)
    w = torch.randn(320, 512, dtype=torch.bfloat16) * 0.02
    for kind, group in (("int4", 128), ("nvfp4", 16), ("fp8", 512)):
        qw = quantize(w, kind=kind, group_size=group)
        dense = python_dequantize(qw).to(torch.bfloat16)
        for shape in ((1, 512), (KERNEL_ROW_LIMIT, 512), (KERNEL_ROW_LIMIT * 4 + 1, 512), (2, 7, 512)):
            x = torch.randn(*shape, dtype=torch.bfloat16)
            got = fused_qlinear(x, qw)
            assert got is not None, (kind, shape)
            want = torch.nn.functional.linear(x, dense)
            assert got.shape == want.shape
            scale = want.float().abs().max().item()
            assert (got.float() - want.float()).abs().max().item() / scale < 2e-2


def test_fused_path_declines_shapes_it_cannot_pack() -> None:
    from engine.kernels import available
    from engine.qweight import fused_qlinear, quantize

    if not available():
        pytest.skip("the fused path is the compiled op")

    qw = quantize(torch.randn(64, 512, dtype=torch.bfloat16) * 0.02, kind="int4", group_size=128)
    assert fused_qlinear(torch.randn(4, 256, dtype=torch.bfloat16), qw) is None  # wrong width
    assert fused_qlinear(torch.randn(4, 512, dtype=torch.float32), qw) is None  # wrong dtype


@pytest.mark.parametrize("kind", ["int4", "nvfp4", "fp8"])
def test_dequantize_honors_the_dtype_it_was_asked_for(kind: str) -> None:
    """out_dtype is a request, not a hint.

    The unpack kernel writes bf16 or fp16 and takes a single flag to choose, so
    asking it for fp32 used to return bf16 silently — which quietly rounds every
    weight in what callers use as their exact reference.
    """
    from engine.qweight import quantize

    torch.manual_seed(7)
    w = torch.randn(64, 128, dtype=torch.bfloat16) * 0.05
    qw = quantize(w, kind=kind, group_size=64)
    for dtype in (torch.float32, torch.float64, torch.bfloat16):
        assert qw.dequantize(out_dtype=dtype).dtype is dtype
    # And the fp32 unpack has to be the exact one, not a widened bf16 copy.
    exact = qw.dequantize(out_dtype=torch.float32)
    assert not torch.equal(exact, exact.to(torch.bfloat16).float()), (
        "fp32 unpack came back through bf16"
    )


def test_preflight_trusts_the_kernel_on_an_integrated_gpu(tmp_path: Path, monkeypatch) -> None:
    """The CUDA driver's "free" is not the budget when the pool is shared.

    On the GB10 the device pool is the host pool, and the driver counts only
    unused pages: right after a 55 GB checkpoint read it said 40 GB free while
    the kernel said 122 GB of the same 131 GB was allocatable. Believing the
    driver refused every model loaded after a large read.
    """
    from engine import agent_api

    class Props:
        is_integrated = 1

    monkeypatch.setattr(agent_api.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(agent_api.torch.cuda, "get_device_properties", lambda _: Props())
    monkeypatch.setattr(
        agent_api.torch.cuda, "mem_get_info", lambda: (40 * 10**9, 131 * 10**9)
    )
    monkeypatch.setattr("engine.memory.available_bytes", lambda: 122 * 10**9)

    folder = _folder(tmp_path, _LLAMA, "integrated")
    report = agent_api.memory_preflight(folder, device="cuda", dtype="bfloat16", quant=None)
    assert report["free_gb"] > 100, report

    # A discrete GPU has its own pool, so there the driver is the authority.
    Props.is_integrated = 0
    report = agent_api.memory_preflight(folder, device="cuda", dtype="bfloat16", quant=None)
    assert report["free_gb"] < 50, report
