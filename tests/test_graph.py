"""The captured decode step, and the conditions it refuses to run under.

Capture is opt-in and fragile by nature — it bakes in buffer addresses and
shapes — so the interesting behavior on a machine without CUDA is that it
declines cleanly and the caller decodes eagerly. The replay itself can only be
checked on the Spark; what is checked here is that nothing about arming it
changes the tokens.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from engine.agent_api import load_engine
from engine.config import ModelConfig
from engine.graph import GraphDecoder, enabled
from engine.kernels import available as kernels_available
from engine.synth import random_engine_weights, write_config, write_hf_folder

_LLAMA = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "vocab_size": 64,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "rms_norm_eps": 1e-5,
    "rope_theta": 10000.0,
    "max_position_embeddings": 128,
    "tie_word_embeddings": False,
    "torch_dtype": "float32",
    "hidden_act": "silu",
}


def _engine(tmp_path: Path):
    folder = write_config(tmp_path / "graph_llama", _LLAMA)
    cfg = ModelConfig.from_pretrained(folder)
    write_hf_folder(folder, cfg, random_engine_weights(cfg, seed=7))
    return load_engine(folder, device="cpu", dtype="float32")


def test_graphs_are_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INFER_CUDA_GRAPH", raising=False)
    assert enabled() is False
    monkeypatch.setenv("INFER_CUDA_GRAPH", "1")
    assert enabled() is torch.cuda.is_available()


def test_create_declines_without_a_capturable_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No CUDA, no budget, or no KV buffers: return None, do not raise."""
    monkeypatch.setenv("INFER_CUDA_GRAPH", "1")
    engine = _engine(tmp_path)
    model = engine.model
    cache = model.make_cache(batch_size=1, device=model.device, dtype=model.dtype)

    assert GraphDecoder.create(model, cache, length=0, budget=8) is None  # cold cache
    assert GraphDecoder.create(model, None, length=0, budget=8) is None
    model.forward(torch.randint(0, 64, (1, 6)), cache=cache, logits_to_keep=1)
    assert GraphDecoder.create(model, cache, length=6, budget=0) is None
    if not torch.cuda.is_available():
        assert GraphDecoder.create(model, cache, length=6, budget=8) is None


def test_arming_graphs_does_not_change_the_tokens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Whether capture succeeds or declines, greedy output must be identical."""
    engine = _engine(tmp_path)
    monkeypatch.delenv("INFER_CUDA_GRAPH", raising=False)
    eager = engine.generate("hi", max_new_tokens=8, apply_chat_template=False)
    monkeypatch.setenv("INFER_CUDA_GRAPH", "1")
    armed = engine.generate("hi", max_new_tokens=8, apply_chat_template=False)
    assert armed == eager


def test_graph_mode_cache_writes_land_at_the_slot(tmp_path: Path) -> None:
    """The mechanism capture depends on: a device-held write index.

    No CUDA needed to check the bookkeeping — a fixed window out, the write at
    the slot the runner owns, and the mask left alone.
    """
    engine = _engine(tmp_path)
    model = engine.model
    cache = model.make_cache(batch_size=1, device=model.device, dtype=model.dtype)
    model.forward(torch.randint(0, 64, (1, 4)), cache=cache, logits_to_keep=1)
    kv = getattr(cache, "kv", cache)
    kv.reserve(64)

    slot = torch.tensor([4], dtype=torch.long, device=model.device)
    kv.enable_graph_mode(64, slot)
    assert kv.graph_mode is True
    k_new = torch.full((1, 2, 1, 16), 0.5, dtype=model.dtype)
    v_new = torch.full((1, 2, 1, 16), -0.5, dtype=model.dtype)
    k_all, v_all = kv.update(0, k_new, v_new)
    assert k_all.shape[2] == 64, "graph mode must expose a fixed window"
    torch.testing.assert_close(k_all[:, :, 4], k_new[:, :, 0])
    torch.testing.assert_close(v_all[:, :, 4], v_new[:, :, 0])

    # Buffers are zeroed, not uninitialized: attention reads the whole window,
    # and NaN garbage past the live length would poison the softmax.
    assert torch.count_nonzero(k_all[:, :, 5:]) == 0

    slot.fill_(5)
    k_all, _ = kv.update(0, k_new * 2, v_new * 2)
    torch.testing.assert_close(k_all[:, :, 5], (k_new * 2)[:, :, 0])
    torch.testing.assert_close(k_all[:, :, 4], k_new[:, :, 0])

    kv.disable_graph_mode()
    assert kv.graph_mode is False


_GRAPH_SHAPES = {
    # A window shorter than the captured buffer: the case where measuring the
    # window from the buffer's end instead of the query's position masks off
    # every real key and the layer returns zeros.
    "sliding": {**_LLAMA, "sliding_window": 8, "use_sliding_window": True},
    # MLA and the differential path build their masks by hand, so they are the
    # two most likely to ignore the validity mask a fixed buffer needs.
    "mla": {
        **_LLAMA,
        "architectures": ["DeepseekV3ForCausalLM"],
        "model_type": "deepseek_v3",
        "q_lora_rank": 16,
        "kv_lora_rank": 16,
        "qk_nope_head_dim": 8,
        "qk_rope_head_dim": 8,
        "v_head_dim": 8,
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 32,
        "first_k_dense_replace": 1,
    },
    "diff": {
        **_LLAMA,
        "architectures": ["DiffLlamaForCausalLM"],
        "model_type": "diffllama",
    },
    # A mixer whose state is updated in place, which is what capture requires of
    # it: the recorded graph writes through the address it saw at capture time.
    "hybrid": {
        **_LLAMA,
        "architectures": ["NemotronHForCausalLM"],
        "model_type": "nemotron_h",
        "layer_norm_epsilon": 1e-5,
        "hybrid_override_pattern": "M*",
        "mamba_num_heads": 4,
        "mamba_head_dim": 16,
        "ssm_state_size": 16,
        "n_groups": 1,
        "conv_kernel": 4,
        "mamba_hidden_act": "silu",
        "chunk_size": 16,
    },
}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="capture needs a GPU")
@pytest.mark.parametrize("shape", sorted(_GRAPH_SHAPES))
def test_replayed_tokens_match_the_eager_ones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shape: str
) -> None:
    """Capture has to produce the eager tokens, and has to actually happen.

    ``capture()`` checks its own first replay and declines on a mismatch, so a
    bug shows up as a decline rather than a wrong answer — which is why this
    asserts capture succeeded before it asserts anything about the tokens.
    """
    monkeypatch.setenv("INFER_CUDA_GRAPH", "1")
    folder = write_config(tmp_path / shape, _GRAPH_SHAPES[shape])
    cfg = ModelConfig.from_pretrained(folder)
    write_hf_folder(folder, cfg, random_engine_weights(cfg, seed=7))
    engine = load_engine(folder, device="cuda", dtype="bfloat16")
    model = engine.model
    prompt = torch.randint(0, 64, (1, 12), device="cuda")

    def eager_tokens(count: int) -> list[int]:
        cache = model.make_cache(batch_size=1, device=model.device, dtype=model.dtype)
        logits = model.forward(prompt, cache=cache, logits_to_keep=1)
        out = [int(logits[0, -1].argmax().item())]
        for _ in range(count - 1):
            step = torch.tensor([[out[-1]]], device="cuda")
            logits = model.forward(step, cache=cache, logits_to_keep=1)
            out.append(int(logits[0, -1].argmax().item()))
        return out

    want = eager_tokens(6)

    cache = model.make_cache(batch_size=1, device=model.device, dtype=model.dtype)
    model.forward(prompt, cache=cache, logits_to_keep=1)
    runner = GraphDecoder.create(model, cache, length=prompt.shape[1], budget=8)
    if runner is None:
        # The one thing that legitimately blocks these shapes is an MoE layer
        # with no fused dispatch behind it, which is INFER_KERNELS=0. Capture
        # has to decline there rather than tear down mid-capture, so a None here
        # is the expected answer and the eager tokens are still the answer.
        assert not kernels_available(), f"{shape}: capture declined with kernels built"
        return
    assert runner is not None, f"{shape}: nothing about this cache blocks capture"
    assert runner.capture(want[0]), (
        f"{shape}: capture declined, so its first replay disagreed with eager"
    )
    got = [want[0]]
    while len(got) < len(want) and runner.room():
        got.append(runner.step())
    runner.release()
    assert got == want, f"{shape}: replay diverged from eager at {got} vs {want}"
