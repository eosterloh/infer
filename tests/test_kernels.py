"""C++ kernels must match the Python reference."""

from __future__ import annotations

import pytest
import torch

from engine.kernels import load_extension, python_rms_norm, python_silu_mul, rms_norm, silu_mul


def test_python_rms_norm_finite() -> None:
    x = torch.randn(2, 3, 16)
    w = torch.randn(16)
    y = python_rms_norm(x, w, 1e-5)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_compiled_rms_norm_matches_python() -> None:
    ext = load_extension()
    if ext is None:
        pytest.skip("infer_kernels C++ extension did not compile")
    torch.manual_seed(0)
    x = torch.randn(4, 8, 32)
    w = torch.randn(32)
    ref = python_rms_norm(x, w, 1e-5)
    got = rms_norm(x, w, 1e-5)
    assert torch.allclose(got, ref, atol=2e-5, rtol=2e-5)


def test_compiled_silu_mul_matches_python() -> None:
    ext = load_extension()
    if ext is None:
        pytest.skip("infer_kernels C++ extension did not compile")
    torch.manual_seed(1)
    gate = torch.randn(3, 5, 17)
    up = torch.randn(3, 5, 17)
    ref = python_silu_mul(gate, up)
    got = silu_mul(gate, up)
    assert torch.allclose(got, ref, atol=2e-5, rtol=2e-5)
