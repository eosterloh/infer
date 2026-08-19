# infer — DIY hybrid decoder engine

Load HF safetensors, run a from-scratch decoder forward, and greedy-generate
with a KV cache. Today: dense Llama (Phase 0–2 + Phase R schedule refactor).
Next: Nemotron Nano hybrid (Mamba / Attn / MoE), then Super (+ MTP).

## Setup

```bash
cd ~/Projects/infer
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=~/Projects/infer
```

## Run (plug-and-play)

**Inspect** a checkpoint (config only — no weight load):

```bash
python -m engine.chat --model testdata/nemotron3-nano-30b-a3b --inspect
```

**Load + generate** (Spark, Llama 1B already on disk):

```bash
python -m engine.chat --model ~/models/Llama-3.2-1B-Instruct --device cuda \
  --prompt "The capital of France is" --max-new-tokens 32
```

**From Python (agents):**

```python
from engine.agent_api import inspect_capabilities, load_engine

print(inspect_capabilities("testdata/nemotron3-nano-30b-a3b").to_dict())
# can_run / recipe_id / missing tell the agent if this folder will load
eng = load_engine("~/models/Llama-3.2-1B-Instruct", device="cuda")
print(eng.generate("The capital of France is", max_new_tokens=16))
```

## Tests

```bash
# unit tests (schedule, agent caps, MoE, Mamba, Nano name-map, dense forward)
pytest -q

# also run integration if weights exist
pytest -q -m integration

# Spark CUDA Llama load+generate
pytest -q -m spark
```

Download weights (gated — accept the license on HF, then login):

```bash
huggingface-cli login
python scripts/download_llama.py --out ~/models/Llama-3.2-1B-Instruct
python scripts/download_nano.py --out ~/models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
```

## Phase 0 load

```bash
source .venv/bin/activate
export PYTHONPATH=~/Projects/infer
python -m engine.chat --model ~/models/Llama-3.2-1B-Instruct --device cuda
```

Success looks like `load_ok=true` plus param count / dtype / device.

## Generate (Phase 2 — KV cache on by default)

```bash
python -m engine.chat --model ~/models/Llama-3.2-1B-Instruct --device cuda \
  --prompt "The capital of France is" --max-new-tokens 32
```

Disable cache (Phase 1 full recompute): add `--no-cache`.

Parity:

```bash
python scripts/parity_check.py --model ~/models/Llama-3.2-1B-Instruct --device cuda
python scripts/cache_parity_check.py --model ~/models/Llama-3.2-1B-Instruct --device cuda
```

### Smoke test without Meta gate access

A tiny public Llama-shaped fixture is under `testdata/tiny-random-llama`
(from `hf-internal-testing/tiny-random-LlamaForCausalLM`). Same load path:

```bash
export PYTHONPATH=~/Projects/infer
python -m engine.chat --model testdata/tiny-random-llama --device cuda --dtype bfloat16
```

## Agent plug-and-play

```python
from engine.agent_api import inspect_capabilities, load_engine

caps = inspect_capabilities("~/models/...")          # config only
eng = load_engine("~/models/Llama-3.2-1B-Instruct")  # full load
print(eng.info()["capabilities"])
print(eng.generate("The capital of France is", max_new_tokens=16))
```

CLI inspect (no weights):

```bash
python -m engine.chat --model testdata/nemotron3-nano-30b-a3b --inspect
```

North star: **Nemotron NVFP4 on DGX Spark**, agent-runnable (MTP with Super later).

## Hybrid schedule (Phase R / N0 / N2 / N3)

Dense Llama still works. Nemotron-H configs (`hybrid_override_pattern`) build a
per-layer schedule (`M`=Mamba2, `E`=MoE, `*`=Attention).

```bash
python scripts/schedule_smoke.py
python scripts/nano_map_check.py --model testdata/nemotron3-nano-30b-a3b
python scripts/moe_smoke.py
```

Full Nano weight load belongs on the Spark (VRAM / disk). MoE path is
implemented (synthetic smoke). Next: Mamba-2 + dual cache (N3), then quant,
then Super/MTP.

## Layout

| Path | Role |
|---|---|
| `engine/config.py` | `config.json` → `ModelConfig` + layer schedule |
| `engine/schedule.py` | `LayerSpec` (mixer + FFN kinds per layer) |
| `engine/weights.py` | safetensors → rename → validate → bf16 → CUDA |
| `engine/tokenizer.py` | HF tokenizer wrapper |
| `engine/layers/` | RMSNorm, RoPE, GQA, SwiGLU; MoE/Mamba stubs |
| `engine/model.py` | `DecoderModel` scheduled forward → logits |
| `engine/cache.py` | per-layer K/V cache |
| `engine/generate.py` | greedy generate (cached prefill/decode) |
| `engine/chat.py` | load + optional `--prompt` |
| `scripts/parity_check.py` | vs HuggingFace |
| `scripts/nano_map_check.py` | Nano config + index name-map check |
| `testdata/nemotron3-nano-30b-a3b/` | Nano `config.json` + weight index |
| `docs/llama-3.2-1b-config.annotated.md` | commented config walkthrough |

## Load pipeline

```text
disk safetensors → CPU (HF names) → rename map → shape checks
  → bf16 cast → .to(cuda) → GPU tensors ready for later matmuls
config.json → ModelConfig → expected shapes
tokenizer files → encode/decode
```
