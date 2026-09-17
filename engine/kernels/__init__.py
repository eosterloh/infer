"""Optional compiled kernels. Python math remains the reference."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

_EXT = None
_LOAD_ATTEMPTED = False


def _extension_dir() -> Path:
    return Path(__file__).resolve().parent


def load_extension():
    """Compile or reuse the C++ extension. Returns the module or None."""
    global _EXT, _LOAD_ATTEMPTED
    if _LOAD_ATTEMPTED:
        return _EXT
    _LOAD_ATTEMPTED = True
    if os.environ.get("INFER_KERNELS", "1") == "0":
        return None
    try:
        from torch.utils.cpp_extension import load
    except ImportError:
        return None
    src = _extension_dir() / "csrc" / "ops.cpp"
    if not src.is_file():
        return None
    exe_bin = str(Path(sys.prefix) / "bin")
    os.environ["PATH"] = exe_bin + os.pathsep + os.environ.get("PATH", "")
    build = _extension_dir() / "_build"
    build.mkdir(parents=True, exist_ok=True)
    extra_cflags = ["-O3"]
    extra_cuda = []
    sources = [str(src)]
    cu = _extension_dir() / "csrc" / "ops.cu"
    nvcc = os.environ.get("CUDA_HOME") or os.environ.get("CUDACXX")
    if torch.cuda.is_available() and nvcc and cu.is_file():
        sources.append(str(cu))
        extra_cflags.append("-DWITH_CUDA")
        extra_cuda = ["-O3", "--use_fast_math"]
    try:
        _EXT = load(
            name="infer_kernels",
            sources=sources,
            extra_cflags=extra_cflags,
            extra_cuda_cflags=extra_cuda,
            build_directory=str(build),
            verbose=os.environ.get("INFER_KERNELS_VERBOSE") == "1",
        )
    except Exception:
        _EXT = None
    return _EXT


def python_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    orig_dtype = x.dtype
    x_f = x.float()
    var = x_f.pow(2).mean(dim=-1, keepdim=True)
    x_f = x_f * torch.rsqrt(var + eps)
    return (x_f * weight.float()).to(orig_dtype)


def python_silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.silu(gate.float()).to(dtype=gate.dtype) * up


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    ext = load_extension()
    if ext is not None and not x.is_cuda:
        try:
            return ext.rms_norm(x, weight, float(eps))
        except Exception:
            pass
    return python_rms_norm(x, weight, eps)


def silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    ext = load_extension()
    if ext is not None and not gate.is_cuda:
        try:
            return ext.silu_mul(gate, up)
        except Exception:
            pass
    return python_silu_mul(gate, up)
