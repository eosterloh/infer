"""Compiled kernels must match the Python reference they replace.

The CPU cases run everywhere. The CUDA cases are the ones that matter on the
Spark: they compare the hand-written kernels against the PyTorch expressions
the engine used before, on the shapes real checkpoints produce.
"""

from __future__ import annotations

import pytest
import torch

from engine.kernels import (
    act_and_mul,
    act_mul,
    available,
    fused_add_rms_norm,
    load_extension,
    python_act_mul,
    python_rms_norm,
    python_silu_mul,
    rms_norm,
    rope_inplace,
    silu_mul,
)
from engine.layers.rope import apply_rope

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)

BF16_TOL = dict(atol=6e-3, rtol=6e-3)


def assert_no_worse_than_python(
    got: torch.Tensor, python: torch.Tensor, exact: torch.Tensor
) -> None:
    """Judge a kernel against fp32 truth, not against a lossy bf16 expression.

    The kernels accumulate in fp32 and round once; the PyTorch expressions they
    replace round after every elementwise step. Demanding a bitwise match with
    the lossier path would be demanding that the kernel be less accurate, so
    the requirement is that the kernel lands no further from the exact answer.
    """
    exact = exact.float()
    kernel_err = (got.float() - exact).abs()
    python_err = (python.float() - exact).abs()
    assert kernel_err.max() <= python_err.max() * 1.5 + 1e-6, (
        f"kernel max error {kernel_err.max():.3e} vs python {python_err.max():.3e}"
    )
    assert kernel_err.mean() <= python_err.mean() * 1.5 + 1e-9, (
        f"kernel mean error {kernel_err.mean():.3e} vs python {python_err.mean():.3e}"
    )


def test_python_rms_norm_finite() -> None:
    x = torch.randn(2, 3, 16)
    w = torch.randn(16)
    y = python_rms_norm(x, w, 1e-5)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_compiled_rms_norm_matches_python() -> None:
    if load_extension() is None:
        pytest.skip("infer_kernels extension did not compile")
    torch.manual_seed(0)
    x = torch.randn(4, 8, 32)
    w = torch.randn(32)
    ref = python_rms_norm(x, w, 1e-5)
    got = rms_norm(x, w, 1e-5)
    assert torch.allclose(got, ref, atol=2e-5, rtol=2e-5)


def test_compiled_silu_mul_matches_python() -> None:
    if load_extension() is None:
        pytest.skip("infer_kernels extension did not compile")
    torch.manual_seed(1)
    gate = torch.randn(3, 5, 17)
    up = torch.randn(3, 5, 17)
    ref = python_silu_mul(gate, up)
    got = silu_mul(gate, up)
    assert torch.allclose(got, ref, atol=2e-5, rtol=2e-5)


def test_rms_norm_weight_offset_is_gemma_form() -> None:
    torch.manual_seed(2)
    x = torch.randn(2, 4, 64)
    w = torch.randn(64)
    ref = python_rms_norm(x, 1.0 + w, 1e-6)
    got = rms_norm(x, w, 1e-6, weight_offset=1.0)
    assert torch.allclose(got, ref, atol=2e-5, rtol=2e-5)


@cuda_only
def test_cuda_extension_loaded() -> None:
    ext = load_extension()
    assert ext is not None, "CUDA build of infer_kernels failed"
    assert ext.has_cuda(), "extension compiled without CUDA sources"
    assert available()


@cuda_only
@pytest.mark.parametrize("hidden", [64, 2048, 3072, 4096, 5120, 8192])
@pytest.mark.parametrize("rows", [1, 7, 512])
def test_cuda_rms_norm_matches_python(hidden: int, rows: int) -> None:
    torch.manual_seed(hidden + rows)
    x = torch.randn(1, rows, hidden, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(hidden, device="cuda", dtype=torch.bfloat16)
    got = rms_norm(x, w, 1e-5)
    assert got.dtype == x.dtype
    torch.testing.assert_close(got, python_rms_norm(x, w, 1e-5), **BF16_TOL)
    assert_no_worse_than_python(
        got, python_rms_norm(x, w, 1e-5), python_rms_norm(x.float(), w.float(), 1e-5)
    )


@cuda_only
def test_cuda_rms_norm_unaligned_hidden() -> None:
    torch.manual_seed(3)
    x = torch.randn(2, 3, 130, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(130, device="cuda", dtype=torch.bfloat16)
    torch.testing.assert_close(rms_norm(x, w, 1e-5), python_rms_norm(x, w, 1e-5), **BF16_TOL)


@cuda_only
def test_cuda_rms_norm_float32() -> None:
    torch.manual_seed(4)
    x = torch.randn(4, 16, 512, device="cuda")
    w = torch.randn(512, device="cuda")
    torch.testing.assert_close(
        rms_norm(x, w, 1e-6), python_rms_norm(x, w, 1e-6), atol=1e-5, rtol=1e-5
    )


@cuda_only
@pytest.mark.parametrize("offset", [0.0, 1.0])
def test_cuda_fused_add_rms_norm_matches_python(offset: float) -> None:
    torch.manual_seed(5)
    x = torch.randn(1, 9, 4096, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    w = torch.randn(4096, device="cuda", dtype=torch.bfloat16)

    want_residual = residual + x
    want = python_rms_norm(want_residual, w, 1e-5, offset)
    exact = python_rms_norm(
        residual.float() + x.float(), w.float(), 1e-5, offset
    )

    normed, new_residual = fused_add_rms_norm(
        x.clone(), residual.clone(), w, 1e-5, offset
    )
    torch.testing.assert_close(new_residual, want_residual, **BF16_TOL)
    assert_no_worse_than_python(normed, want, exact)


@cuda_only
@pytest.mark.parametrize("act", ["silu", "gelu_tanh", "gelu", "relu2"])
def test_cuda_act_mul_matches_python(act: str) -> None:
    torch.manual_seed(6)
    gate = torch.randn(1, 13, 8960, device="cuda", dtype=torch.bfloat16)
    up = torch.randn_like(gate)
    assert_no_worse_than_python(
        act_mul(gate, up, act),
        python_act_mul(gate, up, act),
        python_act_mul(gate.float(), up.float(), act),
    )


@cuda_only
def test_cuda_act_and_mul_splits_packed_projection() -> None:
    torch.manual_seed(7)
    packed = torch.randn(1, 5, 2 * 4096, device="cuda", dtype=torch.bfloat16)
    gate, up = packed.chunk(2, dim=-1)
    torch.testing.assert_close(
        act_and_mul(packed, "silu"), python_act_mul(gate, up, "silu"), **BF16_TOL
    )


@cuda_only
def test_cuda_act_mul_odd_width_falls_back_to_scalar_path() -> None:
    torch.manual_seed(8)
    gate = torch.randn(2, 3, 101, device="cuda", dtype=torch.bfloat16)
    up = torch.randn_like(gate)
    torch.testing.assert_close(
        act_mul(gate, up, "silu"), python_act_mul(gate, up, "silu"), **BF16_TOL
    )


def _rope_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    interleaved: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """apply_rope with the compiled path disabled."""
    from engine.layers import rope as rope_mod

    original = rope_mod.rope_inplace
    rope_mod.rope_inplace = lambda *a, **kw: False
    try:
        return apply_rope(q.clone(), k.clone(), cos, sin, interleaved=interleaved)
    finally:
        rope_mod.rope_inplace = original


@cuda_only
@pytest.mark.parametrize("interleaved", [False, True])
@pytest.mark.parametrize("seq", [1, 37, 512])
def test_cuda_rope_matches_python(interleaved: bool, seq: int) -> None:
    torch.manual_seed(9)
    b, nq, nkv, hd = 1, 32, 8, 128
    # Shapes the engine actually produces: a transposed view of the projection.
    q = torch.randn(b, seq, nq, hd, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    k = torch.randn(b, seq, nkv, hd, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    freqs = torch.randn(b, seq, hd // 2, device="cuda", dtype=torch.bfloat16)
    cos = torch.cat((freqs, freqs), dim=-1).cos().to(torch.bfloat16)
    sin = torch.cat((freqs, freqs), dim=-1).sin().to(torch.bfloat16)

    want_q, want_k = _rope_reference(q, k, cos, sin, interleaved)
    exact_q, exact_k = _rope_reference(
        q.float(), k.float(), cos.float(), sin.float(), interleaved
    )
    assert rope_inplace(q, k, cos, sin, interleaved=interleaved)
    assert_no_worse_than_python(q, want_q, exact_q)
    assert_no_worse_than_python(k, want_k, exact_k)


@cuda_only
def test_cuda_rope_partial_rotary_leaves_tail_untouched() -> None:
    torch.manual_seed(10)
    b, heads, seq, hd, rot = 1, 4, 6, 128, 64
    q = torch.randn(b, seq, heads, hd, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    k = torch.randn(b, seq, heads, hd, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    half = torch.randn(b, seq, rot // 2, device="cuda", dtype=torch.bfloat16)
    cos = torch.cat((half, half), dim=-1).cos().to(torch.bfloat16)
    sin = torch.cat((half, half), dim=-1).sin().to(torch.bfloat16)

    tail_before = q[..., rot:].clone()
    want_q, want_k = _rope_reference(q, k, cos, sin, False)
    exact_q, exact_k = _rope_reference(
        q.float(), k.float(), cos.float(), sin.float(), False
    )
    assert rope_inplace(q, k, cos, sin, interleaved=False)
    assert_no_worse_than_python(q, want_q, exact_q)
    assert_no_worse_than_python(k, want_k, exact_k)
    torch.testing.assert_close(q[..., rot:], tail_before)
