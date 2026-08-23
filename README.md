# infer — DIY hybrid decoder

From-scratch inference: HuggingFace folder in (`config.json` + safetensors) → logits → greedy tokens.

Works today: drop in `config.json` + weights (safetensors or GGUF). Recipes: Llama, Mistral, Qwen2/3, Yi, Gemma, Phi-3, Mixtral, Llama 4, GPT-2, GPT-NeoX, GPT-OSS, DeepSeek V3, Nemotron-H (Nano + Super LatentMoE). NVFP4/FP8 dequant on load; MTP weights load (greedy uses the main head).  
North star: **Nemotron NVFP4 fused on DGX Spark**, agent-runnable.

## How to read `engine/`

Read this **in order**. It is the same story as `~/Projects/Scratch/inference.py`, split into modules. Stop after Pass 1 if you only want Llama. Come back with questions.

Skip `__init__.py`, `__main__.py`, `__pycache__`.

### Pass 1 — Llama (the path you already understand)

Same loop you know: embed → (norm → attention → residual → norm → MLP → residual) × N → final norm → lm_head → argmax.

| # | File | What to look for |
|---|---|---|
| 1 | [`engine/chat.py`](engine/chat.py) | CLI. `--inspect` vs load + `--prompt`. Starts at `main()`. |
| 2 | [`engine/detect.py`](engine/detect.py) | Folder in, no register. `config.json` → `llama` or `nemotron_h`. |
| 3 | [`engine/config.py`](engine/config.py) | Sizes from `config.json` (`hidden_size`, heads, layers). Skim `from_pretrained`; skip `expected_shapes` until weights. |
| 4 | [`engine/tokenizer.py`](engine/tokenizer.py) | Text ↔ ids. Short. |
| 5 | [`engine/weights.py`](engine/weights.py) | HF tensor names → engine names. Read `_map_llama_hf_name` first; ignore Nemotron map. Then `load_weights`. |
| 6 | [`engine/layers/norm.py`](engine/layers/norm.py) | RMSNorm. |
| 7 | [`engine/layers/rope.py`](engine/layers/rope.py) | Cos/sin + `apply_rope`. |
| 8 | [`engine/layers/attention.py`](engine/layers/attention.py) | **Q, K, V live here.** RoPE, causal mask, softmax, mix V, `o_proj`. Cache append is extra vs scratch. |
| 9 | [`engine/layers/mlp.py`](engine/layers/mlp.py) | SwiGLU. One token, no other rows. |
| 10 | [`engine/schedule.py`](engine/schedule.py) | Llama = every layer `attention + dense_mlp`. |
| 11 | [`engine/layers/block.py`](engine/layers/block.py) | One layer: mixer then FFN, both with residual. This is the `x = x + attn; x = x + mlp` you already have. |
| 12 | [`engine/model.py`](engine/model.py) | `DecoderModel.forward`: embed, RoPE tables, **for spec in layers**, lm_head. |
| 13 | [`engine/cache.py`](engine/cache.py) | `KVCache` only (top of file). Prefill writes K/V; decode appends. Scratch file had none of this. |
| 14 | [`engine/generate.py`](engine/generate.py) | `generate_greedy`: encode → forward → argmax → append. `use_cache=True` is the fast path. |

After Pass 1 you can trace: `python -m engine.chat --model ~/models/Llama-3.2-1B-Instruct --prompt "..."`.

### Pass 2 — Hybrid + agents (when you come back)

| File | What to look for |
|---|---|
| [`engine/schedule.py`](engine/schedule.py) | `hybrid_override_pattern`: `M` / `E` / `*` |
| [`engine/layers/moe.py`](engine/layers/moe.py) | Router + top-k + shared expert (replaces dense MLP on `E` layers) |
| [`engine/layers/mamba2.py`](engine/layers/mamba2.py) | SSM mixer (replaces attention on `M` layers) |
| [`engine/cache.py`](engine/cache.py) | `RuntimeState` = KV + Mamba conv/SSM |
| [`engine/agent_api.py`](engine/agent_api.py) | `inspect_capabilities` / `load_engine` — public agent surface |
| [`engine/capabilities.py`](engine/capabilities.py) | Older capability dict (Spark also has this) |
| [`engine/runtime.py`](engine/runtime.py) | Older handle API; prefer `agent_api.py` if they overlap |

### Not yet

NVFP4/FP8 dequant-on-load. MTP speculative decode is listed in `missing` (`mtp_decode`) but greedy still runs.

## Run (Spark)

```bash
cd ~/Projects/infer
source .venv/bin/activate
export PYTHONPATH=~/Projects/infer

python -m engine.chat --model testdata/nemotron3-nano-30b-a3b --inspect

python -m engine.chat --model ~/models/Llama-3.2-1B-Instruct --device cuda \
  --prompt "The capital of France is" --max-new-tokens 32
```

```python
from engine.agent_api import inspect_capabilities, load_engine

print(inspect_capabilities("testdata/nemotron3-nano-30b-a3b").to_dict())
eng = load_engine("~/models/Llama-3.2-1B-Instruct", device="cuda")
print(eng.generate("The capital of France is", max_new_tokens=16))
```

Toy walkthrough (no engine modules): `~/Projects/Scratch/inference.py`.

## Tests

```bash
pytest -q
pytest -q -m spark    # CUDA Llama 1B
```

```bash
python scripts/download_llama.py --out ~/models/Llama-3.2-1B-Instruct
python scripts/download_nano.py --out ~/models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
```

Parity vs HuggingFace:

```bash
python scripts/parity_check.py --model ~/models/Llama-3.2-1B-Instruct --device cuda
python scripts/cache_parity_check.py --model ~/models/Llama-3.2-1B-Instruct --device cuda
```
