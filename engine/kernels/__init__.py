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
_CUDA_SOURCES = (
    "norm.cu",
    "activation.cu",
    "rope.cu",
    "gemv.cu",
    "quant_gemv.cu",
    "moe.cu",
    "mamba.cu",
    "attn_decode.cu",
    "gdn.cu",
)

# The scan kernel is instantiated for these ratios only; see the
# INFER_SCAN_CASE list in csrc/mamba.cu. Anything else hard-errors inside the
# op, which this wrapper would then swallow into the slow Python scan.
_SCAN_D_PER_WARP = (1, 2, 4, 8, 16)
_SCAN_N_PER_LANE = (1, 2, 4)

# A routed row becomes gridDim.y in the MoE kernels, which the hardware caps
# at 65535; the .cu files check the same bound.
_MAX_ROUTED_ROWS = 65535


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


def python_gated_rms_norm(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    group: int,
    *,
    gate_first: bool,
) -> torch.Tensor:
    shape = x.shape
    xf = x.float().reshape(-1, group)
    gf = torch.nn.functional.silu(gate.float().reshape(-1, group))
    value = xf * gf if gate_first else xf
    inv = torch.rsqrt(value.pow(2).mean(dim=-1, keepdim=True) + eps)
    out = value * inv * weight.float().reshape(1, group)
    if not gate_first:
        out = out * gf
    return out.reshape(shape).to(x.dtype)


def gated_rms_norm(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    group: int,
    *,
    gate_first: bool,
) -> torch.Tensor:
    """RMS norm fused with its SiLU gate, as the recurrent mixers need it.

    ``gate_first`` normalizes ``x * silu(gate)`` (Mamba-2); otherwise the gate
    multiplies the normalized value (Gated DeltaNet).
    """
    ops = _ops()
    if ops is not None and x.dtype == weight.dtype and x.shape == gate.shape:
        try:
            return ops.gated_rms_norm(
                x, gate, weight, float(eps), int(group), bool(gate_first)
            )
        except Exception:
            pass
    return python_gated_rms_norm(x, gate, weight, eps, group, gate_first=gate_first)


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


def moe_gemv(
    x: torch.Tensor,
    w: torch.Tensor,
    row_expert: torch.Tensor,
    row_input: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Per-row expert GEMV against a stacked ``[E, N, K]`` weight.

    ``row_expert[r]`` picks the expert for output row ``r`` and ``row_input[r]``
    the activation row, so routing never leaves the device. None means the
    caller must take the Python path.
    """
    ops = _ops()
    if ops is None or w.dim() != 3 or x.dim() != 2:
        return None
    if x.device.type not in ("cuda", "cpu"):
        return None
    if x.dtype != w.dtype or x.dtype not in (torch.bfloat16, torch.float16):
        return None
    if not x.is_contiguous() or not w.is_contiguous():
        return None
    if row_expert.numel() > _MAX_ROUTED_ROWS:
        return None
    try:
        return ops.moe_gemv(x, w, row_expert.to(torch.int32), row_input, bias)
    except Exception:
        return None


def qmoe_gemv(
    x: torch.Tensor,
    qw,
    row_expert: torch.Tensor,
    row_input: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Per-row expert GEMV against a packed ``[E * N, K]`` expert stack.

    ``qw`` is a ``QuantWeight`` whose rows are the experts' output rows laid end
    to end. None means the caller must take the dense path.
    """
    ops = _ops()
    if ops is None or x.dim() != 2 or x.device.type not in ("cuda", "cpu"):
        return None
    if getattr(qw, "expert_cols", 0) <= 0 or x.shape[-1] != qw.in_features:
        return None
    if x.dtype not in (torch.bfloat16, torch.float16) or qw.in_features % 64:
        return None
    if row_expert.numel() > _MAX_ROUTED_ROWS:
        return None
    try:
        from engine.qweight import _KIND_CODE

        return ops.qmoe_gemv(
            x.contiguous(),
            qw.qweight,
            qw.scales,
            qw.zeros,
            qw.channel_scale,
            row_expert.to(torch.int32),
            row_input if row_input is None else row_input.to(torch.int32),
            _KIND_CODE[qw.kind],
            qw.group_size,
            qw.expert_cols,
            qw.in_features,
            float(qw.global_scale),
        )
    except Exception:
        return None


def moe_combine(
    expert_out: torch.Tensor, weights: torch.Tensor, topk: int
) -> torch.Tensor:
    """``sum_k weights[t, k] * expert_out[t * topk + k]``."""
    ops = _ops()
    if ops is not None and expert_out.device.type in ("cuda", "cpu"):
        try:
            return ops.moe_combine(expert_out, weights, int(topk))
        except Exception:
            pass
    tokens = expert_out.shape[0] // topk
    grouped = expert_out.view(tokens, topk, -1).float()
    return (grouped * weights.reshape(tokens, topk, 1).float()).sum(1).to(expert_out.dtype)


def mamba2_scan(
    x: torch.Tensor,
    dt_raw: torch.Tensor,
    dt_bias: torch.Tensor,
    a_log: torch.Tensor,
    b_mat: torch.Tensor,
    c_mat: torch.Tensor,
    d_skip: torch.Tensor,
    state: torch.Tensor,
    *,
    has_state: bool,
    dt_lo: float = 0.0,
    dt_hi: float = float("inf"),
) -> torch.Tensor | None:
    """Selective scan over the whole sequence, state updated in place.

    ``x`` is ``[B, S, H, D]``, ``b_mat``/``c_mat`` are ``[B, S, G, N]``, and
    ``dt_raw`` is pre-softplus. The kernel keeps the recurrent state in
    registers, so it handles prefill and decode with one launch per layer.
    None means the caller must run the Python scan.
    """
    ops = _ops()
    if ops is None or not x.is_cuda:
        return None
    head_dim, state_size = x.shape[-1], b_mat.shape[-1]
    if head_dim % 8 or state_size % 32:
        return None
    if head_dim // 8 not in _SCAN_D_PER_WARP or state_size // 32 not in _SCAN_N_PER_LANE:
        return None
    if not (x.is_contiguous() and b_mat.is_contiguous() and c_mat.is_contiguous()):
        return None
    if not dt_raw.is_contiguous():
        return None
    if state.dtype != torch.float32 or not state.is_contiguous():
        return None
    if x.dtype != b_mat.dtype or x.dtype != c_mat.dtype or x.dtype != dt_raw.dtype:
        return None
    try:
        return ops.mamba2_scan(
            x,
            dt_raw,
            dt_bias.float(),
            a_log.float(),
            b_mat,
            c_mat,
            d_skip.float(),
            state,
            bool(has_state),
            float(dt_lo),
            float(dt_hi),
        )
    except Exception:
        return None


def gdn_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_log: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
) -> torch.Tensor | None:
    """One gated-delta step per head, state updated in place.

    ``q``/``k`` are ``[B, H, Dk]`` (already normalized and scaled), ``v`` is
    ``[B, H, Dv]``, gates are ``[B, H]``, and ``state`` is ``[B, H, Dk, Dv]``
    fp32. None means the caller must run the Python recurrence.
    """
    ops = _ops()
    if ops is None or q.device.type not in ("cuda", "cpu"):
        return None
    if q.dim() != 3 or v.dim() != 3 or state.dim() != 4:
        return None
    if q.dtype != torch.float32 or state.dtype != torch.float32:
        return None
    if v.shape[-1] > 1024 or q.shape[-1] > 4096:
        return None
    tensors = (q, k, v, g_log, beta, state)
    if not all(t.is_contiguous() and t.dtype == torch.float32 for t in tensors):
        return None
    try:
        return ops.gdn_decode(q, k, v, g_log, beta, state)
    except Exception:
        return None


def attn_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float,
    kv_mask: torch.Tensor | None = None,
    sinks: torch.Tensor | None = None,
    window: int = 0,
    softcap: float = 0.0,
) -> torch.Tensor | None:
    """Attention for one query position, straight out of the cache.

    ``q`` is ``[B, nq, D]`` and ``k``/``v`` are the cache views ``[B, nkv, L, D]``
    — GQA stays unexpanded and no mask of length L is ever built. Returns
    ``[B, nq, D]``, or None when the caller must use SDPA.
    """
    ops = _ops()
    if ops is None or q.dim() != 3 or k.dim() != 4 or v.dim() != 4:
        return None
    if q.device.type not in ("cuda", "cpu"):
        return None
    if q.dtype != k.dtype or q.dtype != v.dtype or not q.is_floating_point():
        return None
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        return None
    head_dim = q.shape[-1]
    if head_dim > 256 or v.shape[-1] != head_dim:
        return None
    heads, kv_heads = q.shape[1], k.shape[1]
    if kv_heads == 0 or heads % kv_heads:
        return None
    if kv_mask is not None:
        kv_mask = kv_mask.bool().contiguous()
        if kv_mask.shape[-1] != k.shape[2]:
            return None
    try:
        return ops.attn_decode(
            q,
            k,
            v,
            kv_mask,
            sinks,
            float(scale),
            int(window or 0),
            float(softcap or 0.0),
        )
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
