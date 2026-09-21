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

# -L so the sizes resolve through the HF cache's symlinks into blobs/: without
# it every symlinked checkpoint measures a few hundred bytes and a 30B model
# classifies as "small", which is how a big model ends up with a 512-token
# prompt and an OOM instead of a row.
# -L so the sizes resolve through the HF cache's symlinks, and the index first
# where there is one: Mistral-7B ships a 14 GB consolidated copy beside its 14 GB
# shards, and summing both classified a 7B model as too big to benchmark.
weights_bytes() {
  local dir="$1" index
  for index in "$dir"/model.safetensors.index.json "$dir"/pytorch_model.bin.index.json; do
    [ -f "$index" ] || continue
    "$PY" - "$index" <<'PY' && return 0
import json, os, sys
index = sys.argv[1]
folder = os.path.dirname(index)
with open(index) as fh:
    files = set(json.load(fh).get("weight_map", {}).values())
print(sum(os.path.getsize(os.path.realpath(os.path.join(folder, f)))
          for f in files if os.path.exists(os.path.join(folder, f))))
PY
  done
  find -L "$dir" -maxdepth 2 \( -name '*.safetensors' -o -name '*.gguf' -o -name '*.bin' \) \
    -printf '%s\n' 2>/dev/null | awk '{s+=$1} END {print s+0}'
}

LIST=()
SKIPPED=()
while IFS= read -r -d '' cfg; do
  dir="$(dirname "$cfg")"
  name="$(basename "$dir")"
  bytes="$(weights_bytes "$dir")"
  # A config.json with no weights beside it is a tokenizer-only or partial
  # download. Benchmarking it just spends a process to print a load error.
  if [ "${bytes:-0}" -le 0 ]; then
    SKIPPED+=("$name (no weight files)")
    continue
  fi
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
  # Tab-separated: model directories with spaces in the name are common enough
  # in an HF cache that word-splitting the list would drop them.
  LIST+=("$name	$prefill	$decode	$bytes")
done < <(find -L "$MODELS" -mindepth 2 -maxdepth 2 -name config.json -print0 2>/dev/null)

if [ "${#SKIPPED[@]}" -ne 0 ]; then
  printf 'skipping: %s\n' "${SKIPPED[@]}" >&2
fi

if [ "${#LIST[@]}" -eq 0 ]; then
  echo "nothing to benchmark in $MODELS for set '$SET'; it holds:" >&2
  ls -1 "$MODELS" >&2
  exit 2
fi

mkdir -p "$ROOT/bench"
failed=0
for entry in "${LIST[@]}"; do
  IFS=$'\t' read -r name prefill decode bytes <<<"$entry"
  echo "=== $name ($(( bytes / 1000000000 )) GB, prefill $prefill, decode $decode) ${EXTRA[*]:-} ==="
  "$PY" "$ROOT/scripts/bench_engine.py" \
    --model "$MODELS/$name" --prefill "$prefill" --decode "$decode" \
    --reps 2 --tag "$TAG" ${EXTRA[@]+"${EXTRA[@]}"} \
    >/dev/null 2>"$ROOT/bench/$name.$TAG.err"
  status=$?
  if [ $status -ne 0 ]; then
    echo "  FAILED (exit $status): $(tail -3 "$ROOT/bench/$name.$TAG.err" | tr '\n' ' ')"
    failed=$((failed + 1))
  else
    echo "  ok"
  fi
done

# The sweep's own exit code reports how many models produced no row, so a tag
# that collapses entirely cannot look like a clean run in the log above it.
if [ "$failed" -ne 0 ]; then
  echo "$TAG: $failed of ${#LIST[@]} models produced no row" >&2
  exit 1
fi
