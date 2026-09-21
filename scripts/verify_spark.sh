#!/usr/bin/env bash
# The whole evidence set for the GB10 kernel work, in one run on the Spark.
#
#   1. the CUDA extension builds for sm_121 and loads in this torch
#   2. every kernel matches the PyTorch reference (tests/test_kernels.py)
#   3. the full suite is green
#   4. baseline (INFER_KERNELS=0, eager) vs kernels vs graph vs quant, per model
#   5. a markdown table with fingerprints, so speed is checked against output
#
#   scripts/verify_spark.sh [small|all|big]
set -u

SET="${1:-small}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
cd "$ROOT"
mkdir -p bench

step() { echo; echo "######## $* ########"; }
fail=0

step "1. build + load the extension"
INFER_KERNELS=1 "$PY" - <<'EOF' || fail=1
import torch
from engine import kernels
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
print("capability", torch.cuda.get_device_capability(0) if torch.cuda.is_available() else "-")
ext = kernels.load_extension()
assert ext is not None, "extension failed to build"
print("ops:", sorted(n for n in dir(torch.ops.infer) if not n.startswith("_")))
EOF

step "2. kernel parity"
"$PY" -m pytest tests/test_kernels.py -q || fail=1

step "2b. quantized GEMV / MoE parity + bandwidth"
"$PY" scripts/check_quant.py || fail=1

step "3. full suite"
"$PY" -m pytest tests -q || fail=1

step "3b. full suite with graphs armed"
INFER_CUDA_GRAPH=1 "$PY" -m pytest tests -q || fail=1

step "3c. full suite with the extension off (the fallbacks the baseline uses)"
INFER_KERNELS=0 "$PY" -m pytest tests -q || fail=1

step "3d. decode roofline — achieved bandwidth against the 273 GB/s bus"
MODELS="${MODELS_DIR:-$HOME/models}"
for dir in "$MODELS"/*/; do
  [ -f "$dir/config.json" ] || continue
  name="$(basename "$dir")"
  "$PY" scripts/profile_decode.py --model "$dir" --steps 16 \
    >"bench/roofline.$name.txt" 2>&1 || continue
  sed -n '1,12p' "bench/roofline.$name.txt"
done

step "4a. the engine as it was before this work (a worktree at \$PRE_REV)"
# "Before" has to mean the old code, not the new code with the extension turned
# off: the static KV cache, the SDPA path and the sliced LM head are part of what
# was added. So the same bench script runs against a worktree of the pre-work
# revision, and its results are appended to the same JSONL.
PRE_REV="${PRE_REV:-4453002}"
PRE_DIR="${PRE_DIR:-/tmp/infer-pre}"
if [ ! -f "$PRE_DIR/engine/model.py" ]; then
  git worktree add --detach "$PRE_DIR" "$PRE_REV" || echo "  (worktree failed; skipping)"
fi
if [ -f "$PRE_DIR/engine/model.py" ]; then
  mkdir -p "$PRE_DIR/scripts" "$PRE_DIR/bench"
  cp scripts/bench_engine.py scripts/bench_sweep.sh "$PRE_DIR/scripts/"
  (cd "$PRE_DIR" && PYTHON="$PY" bash scripts/bench_sweep.sh origin "$SET")
  if [ -f "$PRE_DIR/bench/results.jsonl" ]; then
    cat "$PRE_DIR/bench/results.jsonl" >> bench/results.jsonl
    rm -f "$PRE_DIR/bench/results.jsonl"
  fi
fi

step "4b. same code, extension off (separates the Python rework from the kernels)"
INFER_KERNELS=0 scripts/bench_sweep.sh baseline "$SET"

step "4c. kernels sweep"
INFER_KERNELS=1 scripts/bench_sweep.sh kernels "$SET"

step "4d. kernels + cuda graph sweep"
INFER_KERNELS=1 scripts/bench_sweep.sh graph "$SET" --graph

step "4e. nvfp4 sweep"
INFER_KERNELS=1 scripts/bench_sweep.sh nvfp4 "$SET" --quant nvfp4

step "5. report — strict, so a faster wrong answer fails here"
"$PY" scripts/bench_report.py --baseline origin --strict \
  --tag baseline --tag kernels --tag graph --tag nvfp4 --out bench/REPORT.md || fail=1

echo
if [ "$fail" -ne 0 ]; then
  echo "VERIFY: correctness steps FAILED"
  exit 1
fi
echo "VERIFY: correctness steps passed; see bench/REPORT.md"
