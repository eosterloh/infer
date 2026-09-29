"""MLA decode must not build the score matrix, and must still match the one that did."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from engine.cache import KVCache
from engine.layers.attention import decode_attend
from engine.layers.mla import mla_attention


def _config(vdh: int) -> SimpleNamespace:
    return SimpleNamespace(
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        v_head_dim=vdh,
        kv_lora_rank=16,
        rms_norm_eps=1e-5,
    )


def _weights(cfg: SimpleNamespace, device: torch.device, dtype: torch.dtype) -> dict:
    torch.manual_seed(3)
    h = 32
    nq = cfg.num_attention_heads
    nope = cfg.qk_nope_head_dim
    rope = cfg.qk_rope_head_dim
    rank = cfg.kv_lora_rank
    vdh = cfg.v_head_dim

    def w(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, device=device, dtype=dtype)

    return {
        "layers.0.attn.q.weight": w(nq * (nope + rope), h),
        "layers.0.attn.kv_a.weight": w(rank + rope, h),
        "layers.0.attn.kv_a_norm.weight": w(rank),
        "layers.0.attn.kv_b.weight": w(nq * (nope + vdh), rank),
        "layers.0.attn.o.weight": w(h, nq * vdh),
    }


def _no_worse(got: torch.Tensor, python: torch.Tensor, exact: torch.Tensor) -> None:
    """Same bar as the other kernels: no further from fp32 than the eager path."""
    exact = exact.float()
    kernel_err = (got.float() - exact).abs()
    python_err = (python.float() - exact).abs()
    slack = 1.5
    assert kernel_err.max() <= python_err.max() * slack + 1e-6, (
        f"max {kernel_err.max():.3e} vs eager {python_err.max():.3e}"
    )
    assert kernel_err.mean() <= python_err.mean() * slack + 1e-9, (
        f"mean {kernel_err.mean():.3e} vs eager {python_err.mean():.3e}"
    )


def _run(cfg, weights, x: torch.Tensor, step: torch.Tensor):
    cache = KVCache(
        cfg, batch_size=1, device=x.device, dtype=x.dtype, max_seq_len=32
    )
    empty = torch.empty(0, device=x.device)
    prefill = mla_attention(x, weights, 0, empty, empty, cfg, cache=cache)
    decoded = mla_attention(step, weights, 0, empty, empty, cfg, cache=cache)
    return prefill, decoded


@pytest.mark.parametrize("vdh", [8, 16])
def test_mla_fast_path_matches_the_eager_scores(
    monkeypatch: pytest.MonkeyPatch, vdh: int
) -> None:
    """V shorter than QK is the real MLA layout; equal dims can use flash-decode."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    cfg = _config(vdh)
    weights = _weights(cfg, device, dtype)
    torch.manual_seed(11)
    x = torch.randn(1, 5, 32, device=device, dtype=dtype)
    step = torch.randn(1, 1, 32, device=device, dtype=dtype)

    monkeypatch.setenv("INFER_ATTENTION", "eager")
    eager = _run(cfg, weights, x, step)
    exact = _run(
        cfg,
        {key: value.float() for key, value in weights.items()},
        x.float(),
        step.float(),
    )
    monkeypatch.setenv("INFER_ATTENTION", "auto")
    fast = _run(cfg, weights, x, step)

    if dtype == torch.float32:
        torch.testing.assert_close(fast[0], exact[0], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(fast[1], exact[1], atol=1e-4, rtol=1e-4)
        return
    _no_worse(fast[0], eager[0], exact[0])
    _no_worse(fast[1], eager[1], exact[1])


def test_equal_head_dims_take_the_decode_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The equal-dim case is the one attn_decode can run; a silent SDPA fallback is not it."""
    if not torch.cuda.is_available():
        pytest.skip("attn_decode is checked on the GPU")
    from engine.kernels import available

    if not available():
        pytest.skip("extension not built")

    seen: list[int] = []
    real = decode_attend

    def wrapped(q, k, v, **kwargs):
        seen.append(q.shape[-1])
        return real(q, k, v, **kwargs)

    monkeypatch.setattr("engine.layers.mla.decode_attend", wrapped)
    monkeypatch.setenv("INFER_ATTENTION", "auto")
    device = torch.device("cuda")
    cfg = _config(16)
    weights = _weights(cfg, device, torch.bfloat16)
    x = torch.randn(1, 3, 32, device=device, dtype=torch.bfloat16)
    step = torch.randn(1, 1, 32, device=device, dtype=torch.bfloat16)
    _run(cfg, weights, x, step)
    assert seen, "decode never consulted the kernel"
    assert all(dim == 16 for dim in seen)
