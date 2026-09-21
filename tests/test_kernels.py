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


# --- quantized weights ------------------------------------------------


QUANT_KINDS = ["int4", "nvfp4", "fp8"]


def _quant_available() -> bool:
    return available()


@pytest.mark.parametrize("kind", QUANT_KINDS)
def test_dequant_matches_python_reference(kind: str) -> None:
    """The kernel and the PyTorch unpack must agree bit for bit."""
    from engine.qweight import python_dequantize, quantize

    torch.manual_seed(11)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    w = torch.randn(256, 192, device=device, dtype=torch.bfloat16) * 0.02
    qw = quantize(w, kind, group_size=64) if kind == "int4" else quantize(w, kind)
    torch.testing.assert_close(
        qw.dequantize().float(), python_dequantize(qw).to(torch.bfloat16).float()
    )


@pytest.mark.parametrize("kind", QUANT_KINDS)
def test_quantization_round_trip_stays_close(kind: str) -> None:
    """Packing has to preserve the weight well enough to be useful."""
    from engine.qweight import quantize

    torch.manual_seed(12)
    w = torch.randn(512, 256, dtype=torch.bfloat16) * 0.02
    qw = quantize(w, kind)
    back = qw.dequantize().float()
    err = (back - w.float()).norm() / w.float().norm()
    limit = 0.05 if kind == "fp8" else 0.16
    assert err < limit, f"{kind} round trip error {err:.4f}"


@pytest.mark.parametrize("kind", QUANT_KINDS)
@pytest.mark.parametrize("rows", [1, 4])
def test_qgemv_matches_dequantized_linear(kind: str, rows: int) -> None:
    """The fused GEMV must equal a linear against the unpacked weight."""
    from engine.qweight import python_dequantize, qlinear, quantize

    torch.manual_seed(13)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    w = torch.randn(320, 128, device=device, dtype=torch.bfloat16) * 0.02
    x = torch.randn(rows, 128, device=device, dtype=torch.bfloat16)
    qw = quantize(w, kind, group_size=64) if kind == "int4" else quantize(w, kind)
    got = qlinear(x, qw)
    want = torch.nn.functional.linear(x.float(), python_dequantize(qw))
    rel = (got.float() - want).norm() / want.norm()
    assert rel < 5e-3, f"{kind} qgemv rel error {rel:.2e}"


@pytest.mark.parametrize("kind", QUANT_KINDS)
def test_qgemv_bias_is_added(kind: str) -> None:
    from engine.qweight import qlinear, quantize

    torch.manual_seed(14)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    w = torch.randn(128, 128, device=device, dtype=torch.bfloat16) * 0.02
    x = torch.randn(1, 128, device=device, dtype=torch.bfloat16)
    bias = torch.randn(128, device=device, dtype=torch.bfloat16)
    qw = quantize(w, kind)
    torch.testing.assert_close(
        qlinear(x, qw, bias).float(),
        (qlinear(x, qw).float() + bias.float()),
        atol=6e-3,
        rtol=6e-3,
    )


def test_dense_dispatches_quantized_weights() -> None:
    """`dense` has to accept a packed weight wherever a tensor was allowed."""
    from engine.layers.linear import dense
    from engine.qweight import quantize

    torch.manual_seed(15)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    w = torch.randn(192, 128, device=device, dtype=torch.bfloat16) * 0.02
    x = torch.randn(1, 3, 128, device=device, dtype=torch.bfloat16)
    qw = quantize(w, "fp8")
    out = dense(x, qw)
    assert out.shape == (1, 3, 192)
    reference = torch.nn.functional.linear(x, qw.dequantize())
    rel = (out.float() - reference.float()).norm() / reference.float().norm()
    assert rel < 5e-3


def test_quantize_state_dict_only_touches_projections() -> None:
    from engine.qweight import QuantWeight
    from engine.quantize import quantize_state_dict

    torch.manual_seed(16)
    weights = {
        "embed.weight": torch.randn(1024, 512, dtype=torch.bfloat16),
        "layers.0.attn.q.weight": torch.randn(512, 512, dtype=torch.bfloat16),
        "layers.0.input_norm.weight": torch.randn(512, dtype=torch.bfloat16),
        "layers.0.attn.q.bias": torch.randn(512, dtype=torch.bfloat16),
        "lm_head.weight": torch.randn(1024, 512, dtype=torch.bfloat16),
    }
    out = quantize_state_dict(dict(weights), kind="int4", min_numel=1024)
    assert isinstance(out["layers.0.attn.q.weight"], QuantWeight)
    assert isinstance(out["lm_head.weight"], QuantWeight)
    assert isinstance(out["embed.weight"], torch.Tensor)
    assert isinstance(out["layers.0.input_norm.weight"], torch.Tensor)
    assert isinstance(out["layers.0.attn.q.bias"], torch.Tensor)


# --- MoE grouped GEMV -------------------------------------------------


def test_moe_gemv_matches_per_expert_loop() -> None:
    """Grouped dispatch must equal running each routed row on its own."""
    from engine.kernels import moe_gemv

    torch.manual_seed(17)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    experts, n, k, rows = 8, 64, 96, 5
    w = torch.randn(experts, n, k, device=device, dtype=torch.bfloat16) * 0.05
    x = torch.randn(3, k, device=device, dtype=torch.bfloat16)
    row_expert = torch.randint(0, experts, (rows,), device=device, dtype=torch.int32)
    row_input = torch.randint(0, 3, (rows,), device=device, dtype=torch.int32)

    got = moe_gemv(x, w, row_expert, row_input)
    if got is None:
        pytest.skip("moe_gemv unavailable")
    want = torch.stack(
        [
            torch.nn.functional.linear(
                x[int(row_input[r])].float(), w[int(row_expert[r])].float()
            )
            for r in range(rows)
        ]
    )
    rel = (got.float() - want).norm() / want.norm()
    assert rel < 5e-3, f"moe_gemv rel error {rel:.2e}"


def test_moe_combine_matches_weighted_sum() -> None:
    from engine.kernels import moe_combine

    torch.manual_seed(18)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokens, topk, n = 4, 3, 32
    parts = torch.randn(tokens * topk, n, device=device, dtype=torch.bfloat16)
    weights = torch.rand(tokens, topk, device=device, dtype=torch.bfloat16)
    got = moe_combine(parts, weights, topk)
    want = (parts.view(tokens, topk, n).float() * weights.view(tokens, topk, 1).float()).sum(1)
    torch.testing.assert_close(got.float(), want, atol=6e-3, rtol=6e-3)


def test_fused_dispatch_matches_reference_dispatch() -> None:
    """The stacked path and the per-expert loop must agree on the same weights."""
    from engine.layers.moe import _dispatch_experts, expert_mlp, fused_dispatch

    torch.manual_seed(19)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    experts, inter, hidden, topk, tokens = 6, 64, 96, 2, 4
    up = torch.randn(experts, inter, hidden, device=device, dtype=torch.bfloat16) * 0.05
    down = torch.randn(experts, hidden, inter, device=device, dtype=torch.bfloat16) * 0.05
    x = torch.randn(tokens, hidden, device=device, dtype=torch.bfloat16)
    idx = torch.randint(0, experts, (tokens, topk), device=device)
    wts = torch.rand(tokens, topk, device=device, dtype=torch.bfloat16)

    fused = fused_dispatch(x, idx, wts, {"up": up, "down": down}, "relu2")
    assert fused is not None
    loop = _dispatch_experts(
        x, idx, wts, experts, lambda e, tok: expert_mlp(tok, up[e], down[e], "relu2")
    )
    rel = (fused.float() - loop.float()).norm() / loop.float().norm()
    assert rel < 2e-2, f"fused vs loop rel error {rel:.2e}"


def test_stack_moe_experts_replaces_per_expert_keys() -> None:
    from engine.layers.moe import EXPERT_STACK_KEY
    from engine.quantize import stack_moe_experts

    torch.manual_seed(20)
    weights: dict[str, object] = {"embed.weight": torch.randn(8, 4)}
    for e in range(3):
        weights[f"layers.1.moe.experts.{e}.up.weight"] = torch.randn(6, 4)
        weights[f"layers.1.moe.experts.{e}.down.weight"] = torch.randn(4, 6)
    report = stack_moe_experts(weights)
    assert report["layers"] == 1
    assert not any(".experts.0." in name for name in weights)
    stacks = weights[EXPERT_STACK_KEY]["layers.1.moe"]
    assert tuple(stacks["up"].shape) == (3, 6, 4)
    assert tuple(stacks["down"].shape) == (3, 4, 6)


# --- Mamba-2 selective scan -------------------------------------------


def _reference_scan(
    x: torch.Tensor,
    dt_raw: torch.Tensor,
    dt_bias: torch.Tensor,
    a_log: torch.Tensor,
    b_mat: torch.Tensor,
    c_mat: torch.Tensor,
    d_skip: torch.Tensor,
    state: torch.Tensor,
    has_state: bool,
    dt_lo: float = 0.0,
    dt_hi: float = float("inf"),
) -> tuple[torch.Tensor, torch.Tensor]:
    """The step-at-a-time PyTorch scan the kernel replaces, in fp32."""
    b, s, heads, head_dim = x.shape
    groups, n = b_mat.shape[2], b_mat.shape[3]
    reps = heads // groups
    a = -torch.exp(a_log.float())
    dt = torch.nn.functional.softplus(dt_raw.float() + dt_bias.float())
    dt = torch.clamp(dt, dt_lo, dt_hi)
    b_h = b_mat.float().repeat_interleave(reps, dim=2)
    c_h = c_mat.float().repeat_interleave(reps, dim=2)
    acc = state.clone() if has_state else torch.zeros_like(state)
    ys = []
    for t in range(s):
        dt_t = dt[:, t][:, :, None].expand(b, heads, head_dim)
        da = torch.exp(dt_t[..., None] * a[None, :, None, None])
        db = dt_t[..., None] * b_h[:, t][:, :, None, :]
        x_t = x[:, t].float()
        acc = acc * da + db * x_t[..., None]
        y_t = torch.einsum("bhdn,bhn->bhd", acc, c_h[:, t])
        ys.append(y_t + x_t * d_skip.float()[None, :, None])
    return torch.stack(ys, dim=1), acc


def _scan_inputs(seq: int, dtype: torch.dtype, device: str, heads: int = 16):
    head_dim, groups, n = 64, 2, 128
    gen = torch.Generator(device="cpu").manual_seed(seq + 101)
    mk = lambda *shape: torch.randn(*shape, generator=gen).to(device=device, dtype=dtype)
    return dict(
        x=mk(1, seq, heads, head_dim),
        dt_raw=mk(1, seq, heads),
        dt_bias=torch.randn(heads, generator=gen).to(device),
        a_log=torch.randn(heads, generator=gen).to(device),
        b_mat=mk(1, seq, groups, n),
        c_mat=mk(1, seq, groups, n),
        d_skip=torch.randn(heads, generator=gen).to(device),
    )


@pytest.mark.parametrize("seq", [1, 7, 64])
def test_mamba2_scan_matches_sequential_reference(seq: int) -> None:
    """Fused scan must match the step loop for both prefill and decode shapes."""
    from engine.kernels import mamba2_scan

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        pytest.skip("scan kernel is CUDA only; CPU op is the reference itself")
    args = _scan_inputs(seq, torch.bfloat16, device)
    state = torch.randn(
        1, args["x"].shape[2], args["x"].shape[3], args["b_mat"].shape[3], device=device
    )
    want_y, want_state = _reference_scan(state=state, has_state=True, **args)

    got_state = state.clone()
    got = mamba2_scan(state=got_state, has_state=True, **args)
    if got is None:
        pytest.skip("mamba2_scan unavailable")

    assert_no_worse_than_python(got, want_y.to(torch.bfloat16), want_y)
    torch.testing.assert_close(got_state, want_state, atol=2e-3, rtol=2e-3)


def test_mamba2_scan_from_zero_state() -> None:
    """has_state=False must behave like a zeroed state, not read the buffer."""
    from engine.kernels import mamba2_scan

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        pytest.skip("scan kernel is CUDA only")
    args = _scan_inputs(9, torch.bfloat16, device)
    shape = (1, args["x"].shape[2], args["x"].shape[3], args["b_mat"].shape[3])
    want_y, want_state = _reference_scan(
        state=torch.zeros(shape, device=device), has_state=False, **args
    )
    # Garbage in the buffer must not leak into the result.
    got_state = torch.full(shape, 1e4, device=device)
    got = mamba2_scan(state=got_state, has_state=False, **args)
    if got is None:
        pytest.skip("mamba2_scan unavailable")
    assert_no_worse_than_python(got, want_y.to(torch.bfloat16), want_y)
    torch.testing.assert_close(got_state, want_state, atol=2e-3, rtol=2e-3)


def test_mamba2_scan_chained_decode_equals_prefill() -> None:
    """Token-by-token decode must land on the same state as one long prefill."""
    from engine.kernels import mamba2_scan

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        pytest.skip("scan kernel is CUDA only")
    seq = 12
    args = _scan_inputs(seq, torch.bfloat16, device)
    shape = (1, args["x"].shape[2], args["x"].shape[3], args["b_mat"].shape[3])

    whole_state = torch.zeros(shape, device=device)
    whole = mamba2_scan(state=whole_state, has_state=False, **args)
    if whole is None:
        pytest.skip("mamba2_scan unavailable")

    step_state = torch.zeros(shape, device=device)
    steps = []
    for t in range(seq):
        piece = {
            k: (v[:, t : t + 1].contiguous() if v.dim() > 1 else v) for k, v in args.items()
        }
        steps.append(mamba2_scan(state=step_state, has_state=t > 0, **piece))
    chained = torch.cat(steps, dim=1)
    torch.testing.assert_close(chained.float(), whole.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(step_state, whole_state, atol=2e-3, rtol=2e-3)


# --- quantized MoE experts --------------------------------------------


def test_quantized_expert_stack_matches_dense_dispatch() -> None:
    """Packing the [E, N, K] blocks must not change what the MoE computes much."""
    from engine.layers.moe import EXPERT_STACK_KEY, fused_dispatch
    from engine.quantize import quantize_state_dict, stack_moe_experts

    torch.manual_seed(21)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    experts, inter, hidden, topk, tokens = 4, 128, 256, 2, 3
    weights: dict[str, object] = {}
    for e in range(experts):
        weights[f"layers.0.moe.experts.{e}.up.weight"] = (
            torch.randn(inter, hidden, device=device, dtype=torch.bfloat16) * 0.05
        )
        weights[f"layers.0.moe.experts.{e}.down.weight"] = (
            torch.randn(hidden, inter, device=device, dtype=torch.bfloat16) * 0.05
        )
    stack_moe_experts(weights)
    dense_stack = {k: v.clone() for k, v in weights[EXPERT_STACK_KEY]["layers.0.moe"].items()}

    x = torch.randn(tokens, hidden, device=device, dtype=torch.bfloat16)
    idx = torch.randint(0, experts, (tokens, topk), device=device)
    wts = torch.rand(tokens, topk, device=device, dtype=torch.bfloat16)
    want = fused_dispatch(x, idx, wts, dense_stack, "relu2")
    assert want is not None

    packed = quantize_state_dict(weights, kind="int4", group_size=64, min_numel=0)
    stack = packed[EXPERT_STACK_KEY]["layers.0.moe"]
    assert stack["up"].expert_cols == inter, "expert stack was not packed"
    got = fused_dispatch(x, idx, wts, stack, "relu2")
    assert got is not None

    # The packed path must reproduce what the same weights produce once
    # unpacked; the gap to the original bf16 block is 4-bit rounding, which
    # relu2 squares, so only the unpacked comparison is a kernel assertion.
    unpacked = {
        field: value.dequantize().reshape(experts, -1, value.in_features)
        for field, value in stack.items()
    }
    exact = fused_dispatch(x, idx, wts, unpacked, "relu2")
    assert exact is not None
    rel = (got.float() - exact.float()).norm() / exact.float().norm()
    assert rel < 2e-2, f"packed vs unpacked dispatch rel error {rel:.2e}"
    drift = (got.float() - want.float()).norm() / want.float().norm()
    assert drift < 0.3, f"int4 experts drifted too far from bf16: {drift:.2e}"


def test_qmoe_gemv_matches_dequantized_weight() -> None:
    """Routed packed GEMV must match linear() against the unpacked block."""
    from engine.kernels import qmoe_gemv
    from engine.qweight import quantize

    torch.manual_seed(22)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    experts, n, k, rows = 3, 64, 128, 5
    block = torch.randn(experts, n, k, device=device, dtype=torch.bfloat16) * 0.05
    qw = quantize(block.reshape(experts * n, k), "int4", group_size=64)
    qw.experts, qw.expert_cols = experts, n
    x = torch.randn(4, k, device=device, dtype=torch.bfloat16)
    row_expert = torch.randint(0, experts, (rows,), device=device, dtype=torch.int32)
    row_input = torch.randint(0, 4, (rows,), device=device, dtype=torch.int32)

    got = qmoe_gemv(x, qw, row_expert, row_input)
    if got is None:
        pytest.skip("qmoe_gemv unavailable")
    unpacked = qw.dequantize().reshape(experts, n, k)
    want = torch.stack(
        [
            torch.nn.functional.linear(
                x[int(row_input[r])].float(), unpacked[int(row_expert[r])].float()
            )
            for r in range(rows)
        ]
    )
    rel = (got.float() - want).norm() / want.norm()
    assert rel < 5e-3, f"qmoe_gemv rel error {rel:.2e}"
