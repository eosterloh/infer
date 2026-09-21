#!/usr/bin/env python3
"""Does every registered op really run on the GPU, and give the right answer?

The wrappers in engine.kernels all fall back to PyTorch when anything goes
wrong, which is the right behavior for an engine and the wrong behavior for
evidence: a build that produced no CUDA code would still pass the suite, because
the parity tests would be comparing the Python reference against itself.

So this calls torch.ops.infer.* directly — no wrapper, no fallback — on CUDA
tensors, checks each result against the expression it replaces, and exits
non-zero if any op is missing, throws, or disagrees. One line per op.

    scripts/kernel_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.kernels import load_extension  # noqa: E402
from engine.quantize import group_for_quant  # noqa: E402
from engine.qweight import _KIND_CODE, python_dequantize, quantize  # noqa: E402

DEV = "cuda"
DT = torch.bfloat16
results: list[tuple[str, bool, str]] = []


def check(name: str, got: torch.Tensor, want: torch.Tensor, tol: float = 2e-2) -> None:
    """Relative max error against the reference expression."""
    scale = want.float().abs().max().item() or 1.0
    err = (got.float() - want.float()).abs().max().item() / scale
    ok = err < tol and torch.isfinite(got.float()).all().item()
    results.append((name, ok, f"rel err {err:.2e}"))


def failed(name: str, exc: BaseException) -> None:
    results.append((name, False, f"{type(exc).__name__}: {exc}"))


def _rms(x: torch.Tensor, w: torch.Tensor, eps: float, offset: float = 0.0) -> torch.Tensor:
    f = x.float()
    normed = f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + eps)
    return (normed * (w.float() + offset)).to(x.dtype)


def main() -> int:  # noqa: C901 - a flat list of independent checks
    global DEV
    self_test = "--self-test" in sys.argv
    if self_test:
        # Runs the same reference expressions against the CPU implementations, so
        # the script itself can be checked on a machine with no GPU.
        DEV = "cpu"
    elif not torch.cuda.is_available():
        print("no CUDA device: this proves nothing on a CPU box")
        return 1
    if load_extension() is None:
        print("FAIL: the extension did not build")
        return 1
    ops = torch.ops.infer
    if DEV == "cuda":
        print(f"device      {torch.cuda.get_device_name()}")
        caps = "".join(str(v) for v in torch.cuda.get_device_capability())
        print(f"capability  sm_{caps}")
    else:
        print("device      cpu (self-test: reference expressions only)")
    print(f"torch       {torch.__version__} (cuda {torch.version.cuda})")
    print()

    h, seq = 4096, 7
    x = torch.randn(1, seq, h, device=DEV, dtype=DT)
    w = torch.randn(h, device=DEV, dtype=DT)

    try:
        check("rms_norm", ops.rms_norm(x, w, 1e-5, 0.0), _rms(x, w, 1e-5))
    except Exception as exc:
        failed("rms_norm", exc)

    try:
        check("rms_norm (gemma offset)", ops.rms_norm(x, w, 1e-5, 1.0), _rms(x, w, 1e-5, 1.0))
    except Exception as exc:
        failed("rms_norm (gemma offset)", exc)

    try:
        hidden, residual = x.clone(), torch.randn_like(x)
        want_res = (hidden.float() + residual.float()).to(DT)
        ops.fused_add_rms_norm(hidden, residual, w, 1e-5, 0.0)
        check("fused_add_rms_norm", hidden, _rms(want_res, w, 1e-5))
        check("fused_add_rms_norm (residual)", residual, want_res)
    except Exception as exc:
        failed("fused_add_rms_norm", exc)

    gate = torch.randn(1, seq, h, device=DEV, dtype=DT)
    up = torch.randn(1, seq, h, device=DEV, dtype=DT)
    for act, ref in (
        ("silu", lambda g: F.silu(g.float())),
        ("gelu_tanh", lambda g: F.gelu(g.float(), approximate="tanh")),
        ("relu2", lambda g: F.relu(g.float()).pow(2)),
    ):
        try:
            want = (ref(gate) * up.float()).to(DT)
            check(f"act_mul[{act}]", ops.act_mul(gate, up, act), want)
        except Exception as exc:
            failed(f"act_mul[{act}]", exc)

    try:
        packed = torch.cat([gate, up], dim=-1)
        want = (F.silu(gate.float()) * up.float()).to(DT)
        check("act_and_mul", ops.act_and_mul(packed, "silu"), want)
    except Exception as exc:
        failed("act_and_mul", exc)

    group, groups = 128, 4
    gx = torch.randn(1, seq, group * groups, device=DEV, dtype=torch.float32)
    gg = torch.randn_like(gx)
    gw = torch.randn(group, device=DEV, dtype=torch.float32)
    try:
        # Mamba-2's norm_before_gate=False: gate first, then normalize per group.
        inner = (gx * F.silu(gg)).reshape(1, seq, groups, group)
        want = inner * torch.rsqrt(inner.pow(2).mean(-1, keepdim=True) + 1e-5)
        want = want.reshape(gx.shape) * gw.repeat(groups)
        check("gated_rms_norm[gate first]", ops.gated_rms_norm(gx, gg, gw, 1e-5, group, True), want)
    except Exception as exc:
        failed("gated_rms_norm[gate first]", exc)

    try:
        # Gated DeltaNet: normalize, scale, then gate.
        normed = gx * torch.rsqrt(
            gx.reshape(1, seq, groups, group).pow(2).mean(-1, keepdim=True).repeat_interleave(
                group, dim=-1
            ).reshape(gx.shape) + 1e-6
        )
        want = normed * gw.repeat(groups) * F.silu(gg)
        check("gated_rms_norm[gate after]", ops.gated_rms_norm(gx, gg, gw, 1e-6, group, False), want)
    except Exception as exc:
        failed("gated_rms_norm[gate after]", exc)

    hd, heads, kvh = 128, 8, 2
    freqs = torch.randn(1, seq, hd // 2, device=DEV, dtype=torch.float32)
    # The engine passes full-width cos/sin, the halves duplicated.
    cos = torch.cat((freqs, freqs), dim=-1).cos().to(DT)
    sin = torch.cat((freqs, freqs), dim=-1).sin().to(DT)
    q = torch.randn(1, seq, heads, hd, device=DEV, dtype=DT).transpose(1, 2).contiguous()
    k = torch.randn(1, seq, kvh, hd, device=DEV, dtype=DT).transpose(1, 2).contiguous()

    def rotate_half(t: torch.Tensor) -> torch.Tensor:
        half = t.shape[-1] // 2
        flipped = torch.cat((-t.float()[..., half:], t.float()[..., :half]), dim=-1)
        return (t.float() * cos.float()[:, None] + flipped * sin.float()[:, None]).to(DT)

    try:
        want_q, want_k = rotate_half(q), rotate_half(k)
        qc, kc = q.clone(), k.clone()
        ops.rope_inplace(qc, kc, cos, sin, False)
        check("rope_inplace (q)", qc, want_q)
        check("rope_inplace (k)", kc, want_k)
    except Exception as exc:
        failed("rope_inplace", exc)

    try:
        wm = torch.randn(2048, 4096, device=DEV, dtype=DT) * 0.02
        xv = torch.randn(1, 4096, device=DEV, dtype=DT)
        check("gemv", ops.gemv(xv, wm, None), F.linear(xv, wm))
    except Exception as exc:
        failed("gemv", exc)

    for kind in ("int4", "nvfp4", "fp8"):
        group = group_for_quant(kind, 4096, 128)
        qw = quantize(torch.randn(1024, 4096, device=DEV, dtype=DT) * 0.02,
                      kind=kind, group_size=group)
        dense = python_dequantize(qw).to(DT)
        try:
            got = ops.dequant(
                qw.qweight, qw.scales, qw.zeros, qw.channel_scale, _KIND_CODE[kind],
                qw.group_size, qw.out_features, qw.in_features,
                float(qw.global_scale), 0,
            )
            check(f"dequant[{kind}]", got, dense, tol=1e-2)
        except Exception as exc:
            failed(f"dequant[{kind}]", exc)
        for rows in (1, 32):
            try:
                xq = torch.randn(rows, 4096, device=DEV, dtype=DT)
                got = ops.qgemv(
                    xq, qw.qweight, qw.scales, qw.zeros, qw.channel_scale,
                    _KIND_CODE[kind], qw.group_size, qw.out_features, qw.in_features,
                    float(qw.global_scale),
                )
                check(f"qgemv[{kind}, {rows} rows]", got, F.linear(xq, dense))
            except Exception as exc:
                failed(f"qgemv[{kind}, {rows} rows]", exc)
        del qw, dense
        if DEV == "cuda":
            torch.cuda.empty_cache()

    try:
        experts, n, kk, rows = 4, 512, 1024, 6
        stack = torch.randn(experts, n, kk, device=DEV, dtype=DT) * 0.05
        xm = torch.randn(3, kk, device=DEV, dtype=DT)
        row_expert = torch.randint(0, experts, (rows,), device=DEV, dtype=torch.int32)
        row_input = torch.randint(0, 3, (rows,), device=DEV, dtype=torch.int32)
        got = ops.moe_gemv(xm, stack, row_expert, row_input, None)
        want = torch.stack([
            F.linear(xm[int(row_input[i])], stack[int(row_expert[i])]) for i in range(rows)
        ])
        check("moe_gemv", got, want)
    except Exception as exc:
        failed("moe_gemv", exc)

    try:
        topk = 2
        parts = torch.randn(4 * topk, 128, device=DEV, dtype=DT)
        weights = torch.rand(4, topk, device=DEV, dtype=DT)
        want = (parts.view(4, topk, 128).float() * weights.float().unsqueeze(-1)).sum(1)
        check("moe_combine", ops.moe_combine(parts, weights, topk), want.to(DT))
    except Exception as exc:
        failed("moe_combine", exc)

    try:
        experts, n, kk = 4, 256, 1024
        block = torch.randn(experts * n, kk, device=DEV, dtype=DT) * 0.05
        qw = quantize(block, kind="nvfp4", group_size=group_for_quant("nvfp4", kk, 128))
        qw.experts, qw.expert_cols = experts, n
        dense = python_dequantize(qw).to(DT)
        rows = 5
        xm = torch.randn(rows, kk, device=DEV, dtype=DT)
        row_expert = torch.randint(0, experts, (rows,), device=DEV, dtype=torch.int32)
        got = ops.qmoe_gemv(
            xm, qw.qweight, qw.scales, qw.zeros, qw.channel_scale, row_expert, None,
            _KIND_CODE["nvfp4"], qw.group_size, n, kk, float(qw.global_scale),
        )
        want = torch.stack([
            F.linear(xm[i], dense[int(row_expert[i]) * n : (int(row_expert[i]) + 1) * n])
            for i in range(rows)
        ])
        check("qmoe_gemv[nvfp4]", got, want)
    except Exception as exc:
        failed("qmoe_gemv[nvfp4]", exc)

    try:
        heads, dh, dn, groups, steps = 16, 64, 128, 2, 5
        gen = torch.Generator(device="cpu").manual_seed(0)
        mk = lambda *shape: torch.randn(*shape, generator=gen).to(DEV, DT)
        xs, dt_raw = mk(1, steps, heads, dh), mk(1, steps, heads)
        dt_bias = torch.randn(heads, generator=gen).to(DEV)
        a_log = torch.randn(heads, generator=gen).to(DEV)
        d_skip = torch.randn(heads, generator=gen).to(DEV)
        b, c = mk(1, steps, groups, dn), mk(1, steps, groups, dn)
        state = torch.zeros(1, heads, dh, dn, device=DEV, dtype=torch.float32)

        # The recurrence the kernel fuses, stepped in Python.
        want = torch.zeros_like(xs, dtype=torch.float32)
        ref_state = state.clone()
        for t in range(steps):
            dt = F.softplus(dt_raw[0, t].float() + dt_bias.float()).clamp(0.001, 100.0)
            decay = torch.exp(-a_log.float().exp() * dt)
            bt = b[0, t].float().repeat_interleave(heads // groups, dim=0)
            ct = c[0, t].float().repeat_interleave(heads // groups, dim=0)
            ref_state = ref_state[0] * decay[:, None, None] + (
                xs[0, t].float() * dt[:, None]
            )[..., None] * bt[:, None, :]
            ref_state = ref_state[None]
            want[0, t] = torch.einsum("hdn,hn->hd", ref_state[0], ct) + (
                d_skip.float()[:, None] * xs[0, t].float()
            )
        got = ops.mamba2_scan(
            xs, dt_raw, dt_bias, a_log, b, c, d_skip, state, False, 0.001, 100.0
        )
        check("mamba2_scan", got, want.to(DT), tol=5e-2)
        check("mamba2_scan (state)", state, ref_state, tol=5e-2)
    except Exception as exc:
        failed("mamba2_scan", exc)

    try:
        heads, kvh, hd, past = 8, 2, 128, 37
        q = torch.randn(1, heads, hd, device=DEV, dtype=torch.float32)
        kc = torch.randn(1, kvh, past, hd, device=DEV, dtype=torch.float32)
        vc = torch.randn(1, kvh, past, hd, device=DEV, dtype=torch.float32)
        scale = hd ** -0.5
        got = ops.attn_decode(q, kc, vc, None, None, scale, 0, 0.0)
        want = F.scaled_dot_product_attention(
            q.unsqueeze(2),
            kc.repeat_interleave(heads // kvh, dim=1),
            vc.repeat_interleave(heads // kvh, dim=1),
            scale=scale,
        )[:, :, 0]
        check("attn_decode", got, want)
    except Exception as exc:
        failed("attn_decode", exc)

    try:
        heads, dk, dv = 4, 64, 64
        q = torch.randn(1, heads, dk, device=DEV, dtype=torch.float32)
        k = torch.randn(1, heads, dk, device=DEV, dtype=torch.float32)
        v = torch.randn(1, heads, dv, device=DEV, dtype=torch.float32)
        g_log = -torch.rand(1, heads, device=DEV, dtype=torch.float32)
        beta = torch.rand(1, heads, device=DEV, dtype=torch.float32)
        state = torch.randn(1, heads, dk, dv, device=DEV, dtype=torch.float32) * 0.1
        ref_state = state.clone()
        got = ops.gdn_decode(q, k, v, g_log, beta, state)
        decay = g_log.exp()[..., None, None] * ref_state
        kv = torch.einsum("bhk,bhkv->bhv", k, decay)
        delta = (v - kv) * beta[..., None]
        want_state = decay + k[..., None] * delta[..., None, :]
        want = torch.einsum("bhk,bhkv->bhv", q, want_state)
        check("gdn_decode", got, want)
        check("gdn_decode (state)", state, want_state)
    except Exception as exc:
        failed("gdn_decode", exc)

    width = max(len(name) for name, _, _ in results) + 2
    for name, ok, note in results:
        print(f"{'ok  ' if ok else 'FAIL'}  {name:<{width}}{note}")
    bad = [name for name, ok, _ in results if not ok]
    print()
    print(f"{len(results) - len(bad)}/{len(results)} CUDA ops ran and matched")
    if bad:
        print("failed: " + ", ".join(bad))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
