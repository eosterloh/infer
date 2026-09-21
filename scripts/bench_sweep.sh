#!/usr/bin/env bash
# Benchmark every model folder we have, one process each so a load failure or
# an OOM cannot take the rest of the sweep with it.
#
#   scripts/bench_sweep.sh <tag> [small|all|big] [extra bench_engine flags...]
#
# The list is discovered, not hardcoded: any folder under $MODELS_DIR with a
# config.json is a candidate, split into small and big by checkpoint size. A
# hardcoded list silently skips the family whose folder was named differently,
# and a sweep that skips a family is not evidence about that family.
set -u

TAG="${1:-run}"
SET="${2:-small}"
shift 2 2>/dev/null || true
EXTRA=("$@")
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
MODELS="${MODELS_DIR:-$HOME/models}"
# Anything at or above this many bytes of weights is "big": shorter prompts,
# fewer decode steps, and excluded from the default sweep.
BIG_BYTES="${BIG_BYTES:-16000000000}"

if [ ! -d "$MODELS" ]; then
  echo "no model directory at $MODELS (set MODELS_DIR)" >&2
  exit 2
fi

weights_bytes() {
  local total
  total=$(find "$1" -maxdepth 2 \( -name '*.safetensors' -o -name '*.gguf' -o -name '*.bin' \) \
    -printf '%s\n' 2>/dev/null | awk '{s+=$1} END {print s+0}')
  if [ -z "$total" ] || [ "$total" = "0" ]; then
    total=$(( $(du -sk "$1" 2>/dev/null | cut -f1) * 1024 ))
  fi
  echo "$total"
}

LIST=()
for dir in "$MODELS"/*/; do
  name="$(basename "$dir")"
  [ -f "$dir/config.json" ] || continue
  bytes="$(weights_bytes "$dir")"
  [ "$bytes" -gt 0 ] || continue
  if [ "$bytes" -ge "$BIG_BYTES" ]; then
    class=big; prefill=256; decode=16
  else
    class=small; prefill=512; decode=32
  fi
  case "$SET" in
    small) [ "$class" = small ] || continue ;;
    big) [ "$class" = big ] || continue ;;
    all) ;;
    *) echo "usage: $0 <tag> [small|all|big]" >&2; exit 2 ;;
  esac
  # gpt2 and pythia predate long contexts; keep the prompt inside their window.
  case "$name" in
    gpt2*|*pythia*|*Pythia*) prefill=128 ;;
  esac
  LIST+=("$name $prefill $decode $bytes")
done

if [ "${#LIST[@]}" -eq 0 ]; then
  echo "nothing to benchmark in $MODELS for set '$SET'; it holds:" >&2
  ls -1 "$MODELS" >&2
  exit 2
fi

mkdir -p "$ROOT/bench"
for entry in "${LIST[@]}"; do
  set -- $entry
  name="$1"; prefill="$2"; decode="$3"; bytes="$4"
  echo "=== $name ($(( bytes / 1000000000 )) GB, prefill $prefill, decode $decode) ${EXTRA[*]:-} ==="
  "$PY" "$ROOT/scripts/bench_engine.py" \
    --model "$MODELS/$name" --prefill "$prefill" --decode "$decode" \
    --reps 2 --tag "$TAG" ${EXTRA[@]+"${EXTRA[@]}"} \
    >/dev/null 2>"$ROOT/bench/$name.$TAG.err"
  status=$?
  if [ $status -ne 0 ]; then
    echo "  FAILED (exit $status): $(tail -3 "$ROOT/bench/$name.$TAG.err" | tr '\n' ' ')"
  else
    echo "  ok"
  fi
done
