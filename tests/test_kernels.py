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
# Some cases assert the compiled op ran at all, which INFER_KERNELS=0 (how the
# baseline benchmark runs) deliberately makes impossible.
compiled_only = pytest.mark.skipif(
    not available(), reason="needs the compiled extension"
)

BF16_TOL = dict(atol=6e-3, rtol=6e-3)

def registered_op(name: str):
    """The raw op, bypassing the wrapper's device guard and its fallback."""
    load_extension()
    namespace = getattr(torch.ops, "infer", None)
    return getattr(namespace, name, None) if namespace is not None else None




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


@compiled_only
@pytest.mark.parametrize("act", ["silu", "gelu_tanh", "gelu", "relu2"])
@pytest.mark.parametrize("inter", [4096, 4051])
def test_act_and_mul_every_activation_and_the_scalar_tail(act: str, inter: int) -> None:
    """Four near-identical switch arms, and a width the vector path cannot take."""
    op = registered_op("act_and_mul")
    if op is None:
        pytest.skip("extension unavailable")
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gate = torch.randn(1, 3, inter, device=device, dtype=torch.bfloat16)
    up = torch.randn(1, 3, inter, device=device, dtype=torch.bfloat16)
    got = op(torch.cat([gate, up], dim=-1).contiguous(), act)
    assert_no_worse_than_python(
        got,
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
@pytest.mark.parametrize("b", [1, 3])
def test_cuda_rope_matches_python(interleaved: bool, seq: int, b: int) -> None:
    """A batch of one leaves the per-row angle stride at zero, i.e. broadcast.

    With more than one sequence in flight the rows sit at different positions, so
    each needs its own angles; getting that wrong gives every row sequence 0's
    rotation, which looks right until two requests share a batch.
    """
    torch.manual_seed(9)
    nq, nkv, hd = 32, 8, 128
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



# --- dense GEMV ------------------------------------------------------


@compiled_only
@pytest.mark.parametrize("k", [512, 1024, 1032, 2560, 4104, 8192])
@pytest.mark.parametrize("n", [1, 7, 4096])
def test_gemv_matches_linear(k: int, n: int) -> None:
    """The decode GEMV: widths that cross both unroll thresholds and the tail.

    launch_gemv picks among three unroll factors at 128 and 64 vectors, so these
    widths straddle the boundaries, and 1032 and 4104 leave a remainder the
    vector loop cannot cover.
    """
    op = registered_op("gemv")
    if op is None:
        pytest.skip("extension unavailable")
    torch.manual_seed(40)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(1, k, device=device, dtype=torch.bfloat16)
    w = torch.randn(n, k, device=device, dtype=torch.bfloat16) * 0.05

    got = op(x, w, None)
    exact = torch.nn.functional.linear(x.float(), w.float())
    python = torch.nn.functional.linear(x, w)
    assert_no_worse_than_python(got, python, exact)


@compiled_only
def test_gemv_adds_bias_and_keeps_shape() -> None:
    op = registered_op("gemv")
    if op is None:
        pytest.skip("extension unavailable")
    torch.manual_seed(41)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(1, 2048, device=device, dtype=torch.bfloat16)
    w = torch.randn(512, 2048, device=device, dtype=torch.bfloat16) * 0.05
    bias = torch.randn(512, device=device, dtype=torch.bfloat16)
    got = op(x, w, bias)
    assert got.shape == (1, 512)
    torch.testing.assert_close(
        got.float(), (op(x, w, None).float() + bias.float()), **BF16_TOL
    )


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
def test_dequant_handles_a_vocabulary_sized_weight(kind: str) -> None:
    """Past 32768 rows the kernel strides, which no small shape exercises."""
    from engine.qweight import python_dequantize, quantize

    torch.manual_seed(11)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    w = torch.randn(40000, 128, device=device, dtype=torch.bfloat16) * 0.02
    qw = quantize(w, kind, group_size=64)
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


@compiled_only
@pytest.mark.parametrize("kind", QUANT_KINDS)
@pytest.mark.parametrize("rows", [1, 4, 6, 12, 17, 32, 70])
def test_qgemv_matches_dequantized_linear(kind: str, rows: int) -> None:
    """The fused GEMV must equal a linear against the unpacked weight.

    The row counts walk every register bucket the launcher selects, including 6
    and 12 — a two-token draft and an eight-token one — and 70, which chunks.
    """
    from engine.qweight import fused_qlinear, python_dequantize, quantize

    torch.manual_seed(13)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 1024 wide, so a lane's 32 weights do not cover the row in one pass and the
    # k loop runs more than once.
    w = torch.randn(320, 1024, device=device, dtype=torch.bfloat16) * 0.02
    x = torch.randn(rows, 1024, device=device, dtype=torch.bfloat16)
    qw = quantize(w, kind, group_size=64) if kind == "int4" else quantize(w, kind)
    got = fused_qlinear(x, qw)
    assert got is not None
    want = torch.nn.functional.linear(x.float(), python_dequantize(qw))
    rel = (got.float() - want).norm() / want.norm()
    assert rel < 5e-3, f"{kind} qgemv rel error {rel:.2e}"


@compiled_only
@pytest.mark.parametrize("group", [32, 64, 128])
@pytest.mark.parametrize("k", [128, 2048, 4096])
def test_qgemv_int4_group_sizes_line_up_with_scales(group: int, k: int) -> None:
    """Every group the loader can choose, at widths that make the k loop wrap."""
    from engine.qweight import fused_qlinear, python_dequantize, quantize

    torch.manual_seed(15)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    w = torch.randn(192, k, device=device, dtype=torch.bfloat16) * 0.02
    x = torch.randn(3, k, device=device, dtype=torch.bfloat16)
    qw = quantize(w, "int4", group_size=group)
    got = fused_qlinear(x, qw)
    assert got is not None
    want = torch.nn.functional.linear(x.float(), python_dequantize(qw))
    rel = (got.float() - want).norm() / want.norm()
    assert rel < 5e-3, f"int4 group {group} k {k} rel error {rel:.2e}"


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
    experts, n, k, rows = 8, 64, 100, 5  # 100 is not a whole number of vectors
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


@compiled_only
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
    assert rel < 5e-3, f"fused vs loop rel error {rel:.2e}"


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


def _scan_inputs(
    seq: int,
    dtype: torch.dtype,
    device: str,
    heads: int = 16,
    head_dim: int = 64,
    n: int = 128,
    groups: int = 2,
    batch: int = 1,
):
    gen = torch.Generator(device="cpu").manual_seed(seq + 101)
    mk = lambda *shape: torch.randn(*shape, generator=gen).to(device=device, dtype=dtype)
    return dict(
        x=mk(batch, seq, heads, head_dim),
        dt_raw=mk(batch, seq, heads),
        dt_bias=torch.randn(heads, generator=gen).to(device),
        a_log=torch.randn(heads, generator=gen).to(device),
        b_mat=mk(batch, seq, groups, n),
        c_mat=mk(batch, seq, groups, n),
        d_skip=torch.randn(heads, generator=gen).to(device),
    )


def _scan_op(state, has_state, x, dt_raw, dt_bias, a_log, b_mat, c_mat, d_skip):
    """The scan through the op itself: the wrapper refuses anything but CUDA."""
    op = registered_op("mamba2_scan")
    if op is None:
        pytest.skip("extension unavailable")
    return op(x, dt_raw, dt_bias, a_log, b_mat, c_mat, d_skip, state, has_state, 0.0, 100.0)


@compiled_only
@pytest.mark.parametrize("head_dim,n", [(64, 128), (128, 128), (64, 64), (32, 128), (128, 32)])
@pytest.mark.parametrize("groups", [1, 2])
def test_mamba2_scan_register_layouts_all_agree(head_dim: int, n: int, groups: int) -> None:
    """Each (head_dim, state) pair is its own template; only one was ever run.

    A single group is what most Nemotron-H layers use, and it changes which head
    reads which B/C row.
    """
    args = _scan_inputs(
        5, torch.float32, "cuda" if torch.cuda.is_available() else "cpu",
        heads=8, head_dim=head_dim, n=n, groups=groups, batch=2,
    )
    state = torch.randn(2, 8, head_dim, n, device=args["x"].device)
    want_y, want_state = _reference_scan(
        state=state, has_state=True, dt_lo=0.0, dt_hi=100.0, **args
    )
    got_state = state.clone()
    got = _scan_op(got_state, True, **args)
    torch.testing.assert_close(got.float(), want_y.float(), atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(got_state.float(), want_state.float(), atol=2e-4, rtol=2e-4)


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


@compiled_only
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
    assert rel < 5e-3, f"packed vs unpacked dispatch rel error {rel:.2e}"
    drift = (got.float() - want.float()).norm() / want.float().norm()
    assert drift < 0.3, f"int4 experts drifted too far from bf16: {drift:.2e}"


@pytest.mark.parametrize("kind", QUANT_KINDS)
def test_qmoe_gemv_matches_dequantized_weight(kind: str) -> None:
    """Routed packed GEMV must match linear() against the unpacked block.

    The block is built the way the loader builds it — each expert packed on its
    own, then joined — because for NVFP4 that join folds every part's global
    scale into a per-row scale, and the kernel reads scales differently as a
    result.
    """
    from engine.kernels import qmoe_gemv
    from engine.qweight import concat_quant_weights, quantize

    torch.manual_seed(22)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    experts, n, k, rows = 4, 64, 256, 6
    block = torch.randn(experts, n, k, device=device, dtype=torch.bfloat16) * 0.05
    group = 64 if kind == "int4" else (16 if kind == "nvfp4" else k)
    qw = concat_quant_weights(
        [quantize(block[e].contiguous(), kind, group_size=group) for e in range(experts)]
    )
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
    assert rel < 5e-3, f"{kind} qmoe_gemv rel error {rel:.2e}"


# --- gated RMS norm ---------------------------------------------------


def test_gated_rms_norm_matches_mamba_expression() -> None:
    """gate_first must match mamba_ssm's norm_before_gate=False form."""
    from engine.kernels import gated_rms_norm

    torch.manual_seed(23)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    group, rows = 128, 6
    x = torch.randn(1, rows, group * 2, device=device, dtype=torch.float32)
    gate = torch.randn_like(x)
    w = torch.randn(group, device=device, dtype=torch.float32)

    y = x * torch.nn.functional.silu(gate)
    y = y.reshape(1, rows, 2, group)
    want = (y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + 1e-5)).reshape(x.shape) * w.repeat(2)
    got = gated_rms_norm(x, gate, w, 1e-5, group, gate_first=True)
    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)


def test_gated_rms_norm_matches_gdn_expression() -> None:
    """gate_after must match Gated DeltaNet's norm-then-gate form."""
    from engine.kernels import gated_rms_norm

    torch.manual_seed(24)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    heads, dv = 4, 64
    x = torch.randn(1, 3, heads, dv, device=device, dtype=torch.float32)
    z = torch.randn_like(x)
    w = torch.randn(dv, device=device, dtype=torch.float32)

    normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
    want = normed * w * torch.nn.functional.silu(z)
    got = gated_rms_norm(x, z, w, 1e-6, dv, gate_first=False)
    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)


@cuda_only
def test_gated_rms_norm_bf16_no_worse_than_python() -> None:
    from engine.kernels import gated_rms_norm, python_gated_rms_norm

    torch.manual_seed(25)
    group = 256
    x = torch.randn(1, 8, group, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn_like(x)
    w = torch.randn(group, device="cuda", dtype=torch.bfloat16)
    got = gated_rms_norm(x, gate, w, 1e-5, group, gate_first=True)
    python = python_gated_rms_norm(x, gate, w, 1e-5, group, gate_first=True)
    exact = python_gated_rms_norm(
        x.float(), gate.float(), w.float(), 1e-5, group, gate_first=True
    )
    assert_no_worse_than_python(got, python, exact)


# --- flash decode attention -------------------------------------------


def _reference_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float,
    window: int = 0,
    kv_mask: torch.Tensor | None = None,
    sinks: torch.Tensor | None = None,
    softcap: float = 0.0,
) -> torch.Tensor:
    """The expression the kernel replaces: fp32 scores, mask, softmax, matmul."""
    from engine.layers.attention import repeat_kv

    group = q.shape[1] // k.shape[1]
    kf = repeat_kv(k.float(), group)
    vf = repeat_kv(v.float(), group)
    total = k.shape[2]
    scores = torch.matmul(q.float(), kf.transpose(-2, -1)) * scale
    if softcap:
        scores = torch.tanh(scores / softcap) * softcap
    keep = torch.ones(1, 1, 1, total, dtype=torch.bool, device=q.device)
    if window and window < total:
        pos = torch.arange(total, device=q.device)
        keep = keep & (pos >= total - window).view(1, 1, 1, total)
    if kv_mask is not None:
        keep = keep & kv_mask.bool().view(-1, 1, 1, total)
    scores = scores.masked_fill(~keep, float("-inf"))
    if sinks is not None:
        sink = sinks.float().view(1, -1, 1, 1).expand(q.shape[0], q.shape[1], 1, 1)
        w = torch.softmax(torch.cat([scores, sink], dim=-1), dim=-1)[..., :total]
    else:
        w = torch.softmax(scores, dim=-1)
    return torch.matmul(torch.nan_to_num(w), vf)


@pytest.mark.parametrize(
    "heads,kv_heads,head_dim",
    [(8, 8, 64), (16, 4, 128), (6, 2, 96), (32, 1, 128), (8, 8, 80), (4, 4, 256)],
)
@compiled_only
def test_attn_decode_matches_reference(heads: int, kv_heads: int, head_dim: int) -> None:
    """GQA, odd head dims, and long caches must all land on the same answer."""
    from engine.kernels import attn_decode

    torch.manual_seed(31)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    total = 300
    q = torch.randn(1, heads, head_dim, device=device)
    k = torch.randn(1, kv_heads, total, head_dim, device=device)
    v = torch.randn(1, kv_heads, total, head_dim, device=device)
    scale = head_dim**-0.5
    got = attn_decode(q, k, v, scale=scale)
    assert got is not None
    want = _reference_decode(q.unsqueeze(2), k, v, scale=scale)[:, :, 0]
    torch.testing.assert_close(got, want, atol=2e-5, rtol=2e-5)


@compiled_only
def test_attn_decode_honors_window_mask_sinks_and_softcap() -> None:
    from engine.kernels import attn_decode

    torch.manual_seed(32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    heads, kv_heads, head_dim, total = 8, 2, 64, 200
    q = torch.randn(2, heads, head_dim, device=device)
    k = torch.randn(2, kv_heads, total, head_dim, device=device)
    v = torch.randn(2, kv_heads, total, head_dim, device=device)
    scale = head_dim**-0.5
    mask = torch.zeros(2, total, dtype=torch.bool, device=device)
    mask[0, :140] = True
    mask[1, :90] = True
    sinks = torch.randn(heads, device=device)

    for window, sink, cap in ((64, None, 0.0), (0, sinks, 0.0), (0, None, 30.0), (128, sinks, 50.0)):
        got = attn_decode(
            q, k, v, scale=scale, kv_mask=mask, sinks=sink, window=window, softcap=cap
        )
        assert got is not None
        want = _reference_decode(
            q.unsqueeze(2),
            k,
            v,
            scale=scale,
            window=window,
            kv_mask=mask,
            sinks=sink,
            softcap=cap,
        )[:, :, 0]
        torch.testing.assert_close(got, want, atol=2e-5, rtol=2e-5)


@compiled_only
def test_attn_decode_empty_mask_gives_zeros() -> None:
    """Every key masked is the padded-row case; the eager path zeroes it."""
    from engine.kernels import attn_decode

    device = "cuda" if torch.cuda.is_available() else "cpu"
    q = torch.randn(1, 4, 64, device=device)
    k = torch.randn(1, 4, 32, 64, device=device)
    v = torch.randn(1, 4, 32, 64, device=device)
    mask = torch.zeros(1, 32, dtype=torch.bool, device=device)
    got = attn_decode(q, k, v, scale=0.125, kv_mask=mask)
    assert got is not None
    assert torch.count_nonzero(got) == 0


@cuda_only
def test_attn_decode_bf16_no_worse_than_python() -> None:
    from engine.kernels import attn_decode

    torch.manual_seed(33)
    heads, kv_heads, head_dim, total = 32, 8, 128, 1024
    q = torch.randn(1, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, kv_heads, total, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, kv_heads, total, head_dim, device="cuda", dtype=torch.bfloat16)
    scale = head_dim**-0.5
    got = attn_decode(q, k, v, scale=scale)
    assert got is not None
    python = _reference_decode(q.unsqueeze(2), k, v, scale=scale)[:, :, 0].to(torch.bfloat16)
    exact = _reference_decode(
        q.float().unsqueeze(2), k.float(), v.float(), scale=scale
    )[:, :, 0]
    assert_no_worse_than_python(got, python, exact)


@cuda_only
def test_attn_decode_reads_a_cache_view() -> None:
    """The engine passes ``buf[:, :, :len]``; a strided view must still work."""
    from engine.kernels import attn_decode

    torch.manual_seed(34)
    heads, kv_heads, head_dim, cap, live = 8, 2, 64, 512, 130
    q = torch.randn(1, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    k_buf = torch.randn(1, kv_heads, cap, head_dim, device="cuda", dtype=torch.bfloat16)
    v_buf = torch.randn(1, kv_heads, cap, head_dim, device="cuda", dtype=torch.bfloat16)
    k, v = k_buf[:, :, :live], v_buf[:, :, :live]
    assert not k.is_contiguous()
    got = attn_decode(q, k, v, scale=head_dim**-0.5)
    assert got is not None
    want = _reference_decode(q.unsqueeze(2), k, v, scale=head_dim**-0.5)[:, :, 0]
    torch.testing.assert_close(got.float(), want.float(), atol=6e-3, rtol=6e-3)


# --- gated delta step -------------------------------------------------


@compiled_only
@pytest.mark.parametrize("b,heads,dk,dv", [(2, 6, 128, 64), (1, 4, 4096, 128), (3, 32, 128, 128)])
def test_gdn_decode_matches_recurrence(b: int, heads: int, dk: int, dv: int) -> None:
    """The fused step must reproduce one iteration of the Python recurrence.

    4096 is the widest key the kernel accepts and puts 32 KB into shared memory,
    which no other shape here comes close to.
    """
    from engine.kernels import gdn_decode

    torch.manual_seed(41)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    q = torch.randn(b, heads, dk, device=device)
    k = torch.randn(b, heads, dk, device=device)
    v = torch.randn(b, heads, dv, device=device)
    g_log = -torch.rand(b, heads, device=device)
    beta = torch.rand(b, heads, device=device)
    state = torch.randn(b, heads, dk, dv, device=device)

    rec = state.clone() * g_log.exp()[:, :, None, None]
    kv_mem = (rec * k[..., None]).sum(-2)
    delta = (v - kv_mem) * beta[..., None]
    rec = rec + k[..., None] * delta[..., None, :]
    want = (rec * q[..., None]).sum(-2)

    got = gdn_decode(q, k, v, g_log, beta, state)
    assert got is not None
    torch.testing.assert_close(got, want, atol=2e-5, rtol=2e-5)
    # The state carries the next token, so it has to be updated in place.
    torch.testing.assert_close(state, rec, atol=2e-5, rtol=2e-5)


@compiled_only
def test_gdn_decode_matches_layer_recurrent_path() -> None:
    """Whole-mixer check: fused step vs the engine's own Python recurrence."""
    from engine.layers.gdn import _gated_delta_recurrent, _gated_delta_step

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    b, heads, dk, dv = 1, 4, 64, 64
    query = torch.randn(b, 1, heads, dk, device=device)
    key = torch.randn(b, 1, heads, dk, device=device)
    value = torch.randn(b, 1, heads, dv, device=device)
    g_log = -torch.rand(b, 1, heads, device=device)
    beta = torch.rand(b, 1, heads, device=device)
    state = torch.randn(b, heads, dk, dv, device=device)

    want, want_state = _gated_delta_recurrent(
        query, key, value, g_log, beta, state.clone()
    )
    fused = _gated_delta_step(query, key, value, g_log, beta, state.clone())
    assert fused is not None
    got, got_state = fused
    torch.testing.assert_close(got, want, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(got_state, want_state, atol=2e-5, rtol=2e-5)


@compiled_only
@pytest.mark.parametrize("kv_len", [1, 63, 64, 65, 127, 128, 129, 1000])
def test_attn_decode_across_split_boundaries(kv_len: int) -> None:
    """How many chunks the kernel splits into is the GPU's choice, not the test's.

    pick_splits reads the SM count, so these lengths straddle the 64-key chunk
    boundary to make the merge step run at more than one partition count.
    """
    from engine.kernels import attn_decode

    torch.manual_seed(33)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    heads, head_dim = 32, 128
    q = torch.randn(1, heads, head_dim, device=device)
    k = torch.randn(1, heads, kv_len, head_dim, device=device)
    v = torch.randn(1, heads, kv_len, head_dim, device=device)
    scale = head_dim**-0.5
    got = attn_decode(q, k, v, scale=scale)
    assert got is not None
    want = _reference_decode(q.unsqueeze(2), k, v, scale=scale)[:, :, 0]
    torch.testing.assert_close(got, want, atol=2e-5, rtol=2e-5)


@compiled_only
def test_moe_gemv_handles_a_real_expert_width() -> None:
    """The tiny shape leaves most lanes idle; a real expert does not."""
    from engine.kernels import moe_gemv

    torch.manual_seed(18)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    experts, n, k, rows = 4, 256, 2048, 32
    w = torch.randn(experts, n, k, device=device, dtype=torch.bfloat16) * 0.02
    x = torch.randn(8, k, device=device, dtype=torch.bfloat16)
    row_expert = torch.randint(0, experts, (rows,), device=device, dtype=torch.int32)
    row_input = torch.randint(0, 8, (rows,), device=device, dtype=torch.int32)
    got = moe_gemv(x, w, row_expert, row_input)
    assert got is not None
    wf, xf = w.float(), x.float()
    want = torch.stack([
        torch.nn.functional.linear(xf[int(row_input[r])], wf[int(row_expert[r])])
        for r in range(rows)
    ])
    rel = (got.float() - want).norm() / want.norm()
    assert rel < 6e-3, f"moe_gemv rel error {rel:.2e}"


@compiled_only
def test_moe_combine_keeps_the_routers_precision() -> None:
    """Routing weights are fp32 by design and must not be rounded on the way in.

    A BF16 output cannot be compared against an fp32 reference tightly enough to
    see this, so instead this asks which reference the kernel agrees with: the one
    that keeps the router's fp32 weights, or the one that rounds them first. The
    two are about 1.5e-2 apart here, so the answer is unambiguous.
    """
    op = registered_op("moe_combine")
    if op is None:
        pytest.skip("extension unavailable")
    torch.manual_seed(44)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokens, topk, n = 5, 4, 128
    parts = torch.randn(tokens * topk, n, device=device, dtype=torch.bfloat16)
    # Values a BF16 mantissa cannot hold, which is what a real softmax produces.
    weights = torch.softmax(
        torch.randn(tokens, topk, device=device, dtype=torch.float32) * 2, dim=-1
    )
    got = op(parts, weights, topk).float()

    def combine(w: torch.Tensor) -> torch.Tensor:
        return (
            (parts.view(tokens, topk, n).float() * w.float()[..., None])
            .sum(1)
            .to(torch.bfloat16)
            .float()
        )

    kept, rounded = combine(weights), combine(weights.to(torch.bfloat16))
    apart = (kept - rounded).abs().max().item()
    assert apart > 0, "pick weights that BF16 actually cannot represent"
    near = (got - kept).abs().max().item()
    far = (got - rounded).abs().max().item()
    assert near < far / 4, (
        f"combine rounded the router's weights: {near:.2e} from the fp32 answer, "
        f"{far:.2e} from the BF16 one"
    )
