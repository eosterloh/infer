"""The extension's wiring, checked without a compiler.

Every kernel crosses three files: the .cu that defines it, ops.cpp that declares
and registers it, and the Python wrapper that calls it with a fallback. A typo in
any one of them fails at a different time — link error, dispatch error, or a
silent fallback that quietly runs the slow path forever. The Mac cannot compile
CUDA, so these read the sources instead and catch the drift here rather than on
the Spark, where every failed build is a round trip.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from engine.kernels import _CUDA_SOURCES

CSRC = Path(__file__).resolve().parents[1] / "engine" / "kernels" / "csrc"
OPS = (CSRC / "ops.cpp").read_text()

# Ops that deliberately have no CUDA implementation of their own.
_CPU_ONLY: set[str] = set()


def _registered() -> list[str]:
    return re.findall(r'm\.def\(\s*"(\w+)\(', OPS.replace("\n", " ").replace('"      "', ""))


def _impls(dispatch_key: str) -> dict[str, str]:
    block = re.search(
        rf"TORCH_LIBRARY_IMPL\(infer, {dispatch_key}, m\)\s*\{{(.*?)\n\}}", OPS, re.S
    )
    assert block, f"no {dispatch_key} implementation block"
    return dict(re.findall(r'm\.impl\("(\w+)",\s*TORCH_FN\(infer::(\w+)\)\)', block.group(1)))


def _param_types(params: str) -> list[str]:
    """Parameter types with the names dropped, so naming cannot fail the match."""
    depth = 0
    current = ""
    out = []
    for char in params + ",":
        if char in "<(":
            depth += 1
        elif char in ">)":
            depth -= 1
        if char == "," and depth == 0:
            token = " ".join(current.split())
            if token:
                # "const at::Tensor& q" -> "const at::Tensor&"
                out.append(" ".join(token.split(" ")[:-1]) or token)
            current = ""
        else:
            current += char
    return out


def _signature(text: str, name: str, *, declaration: bool) -> str | None:
    tail = ";" if declaration else r"\{"
    found = re.search(
        rf"(?:at::Tensor|void)\s+{name}\s*\((.*?)\)\s*{tail}", text, re.S
    )
    return None if not found else str(_param_types(found.group(1)))


def _cuda_text() -> dict[str, str]:
    return {name: (CSRC / name).read_text() for name in _CUDA_SOURCES}


def test_every_registered_op_has_both_implementations() -> None:
    ops = _registered()
    assert len(ops) > 10, f"suspiciously few ops parsed: {ops}"
    cpu, cuda = _impls("CPU"), _impls("CUDA")
    for op in ops:
        assert op in cpu, f"{op} has no CPU implementation"
        if op not in _CPU_ONLY:
            assert op in cuda, f"{op} has no CUDA implementation"
    assert set(cuda) - set(ops) == set(), "CUDA impl for an unregistered op"


@pytest.mark.parametrize("op", sorted(set(_impls("CUDA"))))
def test_cuda_prototypes_match_their_definitions(op: str) -> None:
    """A prototype that drifts from its .cu definition is a link error."""
    func = _impls("CUDA")[op]
    if func.endswith("_shim"):
        # A shim lives in ops.cpp and forwards to the real kernel; check the
        # kernel it calls, otherwise this op is the one whose prototype drift
        # nothing catches until the link step on the GPU box.
        body = re.search(rf"{func}\s*\(.*?\)\s*\{{(.*?)\n\}}", OPS, re.S)
        assert body, f"{func} has no body in ops.cpp"
        called = re.search(r"return\s+(\w+_cuda)\s*\(", body.group(1))
        assert called, f"{func} forwards to no *_cuda function"
        func = called.group(1)
    declared = _signature(OPS, func, declaration=True)
    assert declared, f"{func} is registered but never declared in ops.cpp"
    defined = [
        _signature(text, func, declaration=False)
        for text in _cuda_text().values()
    ]
    defined = [sig for sig in defined if sig]
    assert defined, f"{func} is declared but defined in no compiled .cu source"
    assert declared in defined, (
        f"{func} prototype {declared} does not match its definition {defined}"
    )


def test_every_source_file_is_compiled() -> None:
    """A .cu that nobody lists is a kernel that silently never runs."""
    on_disk = {p.name for p in CSRC.glob("*.cu")}
    assert on_disk == set(_CUDA_SOURCES), (
        f"sources on disk {sorted(on_disk)} != compiled {sorted(_CUDA_SOURCES)}"
    )


def test_every_op_is_reachable_from_python() -> None:
    """Each registered op needs a wrapper, or nothing will ever call it."""
    wrappers = (Path(__file__).resolve().parents[1] / "engine" / "kernels" / "__init__.py").read_text()
    callers = wrappers + (
        Path(__file__).resolve().parents[1] / "engine" / "qweight.py"
    ).read_text()
    for op in _registered():
        assert f"ops.{op}(" in callers, f"{op} is registered but never called"
