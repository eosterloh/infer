#!/usr/bin/env bash
# Benchmark every model folder we have, one process each so a load failure or
# an OOM cannot take the rest of the sweep with it.
#
#   scripts/bench_sweep.sh <tag> [small|all|big] [extra bench_engine flags...]
set -u

TAG="${1:-run}"
SET="${2:-small}"
shift 2 2>/dev/null || true
EXTRA=("$@")
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
MODELS="${MODELS_DIR:-$HOME/models}"

SMALL=(
  "gpt2 128 32"
  "pythia-410m 128 32"
  "Qwen3-0.6B 512 32"
  "TinyLlama-1.1B-Chat-v1.0 512 32"
  "Llama-3.2-1B-Instruct 512 32"
  "Qwen2.5-1.5B-Instruct 512 32"
  "SmolLM2-1.7B-Instruct 512 32"
  "Llama-3.2-3B-Instruct 512 32"
  "Phi-3-mini-4k-instruct 512 32"
  "Yi-1.5-6B-Chat 512 32"
  "Mistral-7B-Instruct-v0.3 512 32"
)

BIG=(
  "NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 256 16"
  "Qwen3.8-27B 256 16"
)

case "$SET" in
  small) LIST=("${SMALL[@]}") ;;
  big) LIST=("${BIG[@]}") ;;
  all) LIST=("${SMALL[@]}" "${BIG[@]}") ;;
  *) echo "usage: $0 <tag> [small|all|big]" >&2; exit 2 ;;
esac

for entry in "${LIST[@]}"; do
  set -- $entry
  name="$1"; prefill="$2"; decode="$3"
  dir="$MODELS/$name"
  if [ ! -d "$dir" ]; then
    echo "skip $name (not downloaded)"
    continue
  fi
  echo "=== $name (prefill $prefill, decode $decode) ${EXTRA[*]:-} ==="
  mkdir -p "$ROOT/bench"
  "$PY" "$ROOT/scripts/bench_engine.py" \
    --model "$dir" --prefill "$prefill" --decode "$decode" \
    --reps 2 --tag "$TAG" ${EXTRA[@]+"${EXTRA[@]}"} \
    >/dev/null 2>"$ROOT/bench/$name.$TAG.err"
  status=$?
  if [ $status -ne 0 ]; then
    echo "  FAILED (exit $status): $(tail -3 "$ROOT/bench/$name.$TAG.err" | tr '\n' ' ')"
  else
    echo "  ok"
  fi
done
