"""Compiled kernels. Python math stays the reference and the fallback.

The extension registers ``torch.ops.infer.*``. Every wrapper here tries the
compiled op and drops back to the PyTorch expression it replaces, so an engine
on a machine without nvcc behaves the same, only slower.

Environment:
    INFER_KERNELS=0          never load the extension
    INFER_KERNELS_VERBOSE=1  print the build log
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

_EXT = None
_LOAD_ATTEMPTED = False
_SOURCES = ("ops.cpp",)
_CUDA_SOURCES = ("norm.cu", "activation.cu", "rope.cu", "gemv.cu")


def _extension_dir() -> Path:
    return Path(__file__).resolve().parent


def _arch_flags() -> list[str]:
    """Target the local GPU; GB10 is sm_121."""
    if os.environ.get("TORCH_CUDA_ARCH_LIST"):
        return []
    try:
        major, minor = torch.cuda.get_device_capability()
    except Exception:
        return []
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
    return []


def load_extension():
    """Compile or reuse the extension. Returns the module or None."""
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
    csrc = _extension_dir() / "csrc"
    sources = [str(csrc / name) for name in _SOURCES]
    if not all(Path(s).is_file() for s in sources):
        return None
    # nvcc lives outside the venv on most CUDA installs.
    os.environ["PATH"] = os.pathsep.join(
        [str(Path(sys.prefix) / "bin"), "/usr/local/cuda/bin", os.environ.get("PATH", "")]
    )
    build = _extension_dir() / "_build"
    build.mkdir(parents=True, exist_ok=True)
    extra_cflags = ["-O3", "-std=c++17"]
    extra_cuda_cflags: list[str] = []
    cuda_available = False
    try:
        cuda_available = torch.cuda.is_available()
    except Exception:
        cuda_available = False
    if cuda_available and all((csrc / name).is_file() for name in _CUDA_SOURCES):
        from torch.utils.cpp_extension import CUDA_HOME

        if CUDA_HOME or os.environ.get("CUDA_HOME"):
            _arch_flags()
            sources += [str(csrc / name) for name in _CUDA_SOURCES]
            extra_cflags.append("-DWITH_CUDA")
            extra_cuda_cflags = [
                "-O3",
                "-DWITH_CUDA",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "--expt-relaxed-constexpr",
            ]
    try:
        _EXT = load(
            name="infer_kernels",
            sources=sources,
            extra_cflags=extra_cflags,
            extra_cuda_cflags=extra_cuda_cflags,
            extra_include_paths=[str(csrc)],
            build_directory=str(build),
            verbose=os.environ.get("INFER_KERNELS_VERBOSE") == "1",
        )
    except Exception as exc:  # pragma: no cover - build environment dependent
        if os.environ.get("INFER_KERNELS_VERBOSE") == "1":
            print(f"[infer.kernels] build failed: {exc}", file=sys.stderr)
        _EXT = None
    return _EXT


def _ops():
    """The registered op namespace, or None when the build is unavailable."""
    if load_extension() is None:
        return None
    return getattr(torch.ops, "infer", None)


def available() -> bool:
    return _ops() is not None


# --- Python reference implementations ---------------------------------


def python_rms_norm(
    x: torch.Tensor, weight: torch.Tensor, eps: float, weight_offset: float = 0.0
) -> torch.Tensor:
    orig_dtype = x.dtype
    x_f = x.float()
    var = x_f.pow(2).mean(dim=-1, keepdim=True)
    x_f = x_f * torch.rsqrt(var + eps)
    return (x_f * (weight.float() + weight_offset)).to(orig_dtype)


def python_silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.silu(gate.float()).to(dtype=gate.dtype) * up


def _python_activate(x: torch.Tensor, act: str) -> torch.Tensor:
    if act in {"silu", "swiglu"}:
        return torch.nn.functional.silu(x)
    if act in {"gelu_tanh", "gelu_pytorch_tanh", "gelu_new"}:
        return torch.nn.functional.gelu(x, approximate="tanh")
    if act in {"gelu", "gelu_erf"}:
        return torch.nn.functional.gelu(x)
    if act in {"relu2", "relu_squared", "squared_relu"}:
        return torch.square(torch.nn.functional.relu(x))
    raise ValueError(f"unsupported fused activation: {act!r}")


def python_act_mul(gate: torch.Tensor, up: torch.Tensor, act: str) -> torch.Tensor:
    return (_python_activate(gate.float(), act) * up.float()).to(dtype=gate.dtype)


# --- public wrappers --------------------------------------------------


def rms_norm(
    x: torch.Tensor, weight: torch.Tensor, eps: float, weight_offset: float = 0.0
) -> torch.Tensor:
    ops = _ops()
    if ops is not None and x.dtype == weight.dtype:
        try:
            return ops.rms_norm(x, weight, float(eps), float(weight_offset))
        except Exception:
            pass
    return python_rms_norm(x, weight, eps, weight_offset)


def fused_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    weight_offset: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``residual += x`` then ``x = rms_norm(residual)``, both in place.

    Returns ``(normed, residual)``. The caller keeps the residual stream alive
    across the block, which is what makes the fusion possible.
    """
    ops = _ops()
    if (
        ops is not None
        and x.is_contiguous()
        and residual.is_contiguous()
        and x.dtype == weight.dtype
        and x.shape == residual.shape
    ):
        try:
            ops.fused_add_rms_norm(x, residual, weight, float(eps), float(weight_offset))
            return x, residual
        except Exception:
            pass
    residual = residual + x
    return python_rms_norm(residual, weight, eps, weight_offset), residual


def act_mul(gate: torch.Tensor, up: torch.Tensor, act: str = "silu") -> torch.Tensor:
    ops = _ops()
    if ops is not None and gate.dtype == up.dtype:
        try:
            return ops.act_mul(gate, up, act)
        except Exception:
            pass
    return python_act_mul(gate, up, act)


def act_and_mul(gate_up: torch.Tensor, act: str = "silu") -> torch.Tensor:
    """Split a packed ``[..., 2 * inter]`` projection and fuse the gate."""
    ops = _ops()
    if ops is not None:
        try:
            return ops.act_and_mul(gate_up, act)
        except Exception:
            pass
    gate, up = gate_up.chunk(2, dim=-1)
    return python_act_mul(gate, up, act)


def silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    ops = _ops()
    if ops is not None and gate.dtype == up.dtype:
        try:
            return ops.act_mul(gate, up, "silu")
        except Exception:
            pass
    return python_silu_mul(gate, up)


def gemv(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor | None:
    """Batch-1 ``x @ weight.T``, or None when the kernel does not apply."""
    ops = _ops()
    if ops is None or not x.is_cuda:
        return None
    if weight.dim() != 2 or not weight.is_contiguous():
        return None
    if x.numel() != weight.shape[1] or x.dtype != weight.dtype:
        return None
    if weight.dtype not in (torch.bfloat16, torch.float16):
        return None
    if bias is not None and (bias.dtype != weight.dtype or bias.numel() != weight.shape[0]):
        return None
    try:
        return ops.gemv(x, weight, bias)
    except Exception:
        return None


def rope_inplace(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    interleaved: bool = False,
) -> bool:
    """Rotate q and k in place. False means the caller must use the Python path."""
    ops = _ops()
    if ops is None:
        return False
    if q.dim() != 4 or k.dim() != 4 or cos.dim() != 3 or sin.dim() != 3:
        return False
    if q.stride(-1) != 1 or k.stride(-1) != 1:
        return False
    if q.dtype != cos.dtype or q.dtype != k.dtype:
        return False
    if cos.shape[-1] % 2 or cos.shape[-1] > q.shape[-1]:
        return False
    try:
        ops.rope_inplace(q, k, cos, sin, bool(interleaved))
        return True
    except Exception:
        return False
