# infer — DIY hybrid decoder

From-scratch inference: HuggingFace folder in (`config.json` + safetensors) → logits → greedy or sampled tokens.

Works today: drop in `config.json` + weights (safetensors or GGUF). Recipes: Llama, Mistral, Qwen2/3, Qwen2-MoE, Qwen3-MoE, Qwen3.5/Qwen3.8 (hybrid text, image/video, native MTP), Qwen3-Next, Qwen3.5-MoE, Yi, Gemma / Gemma2 / Gemma3, Phi / Phi-3 / Phi-4, Mixtral, Llama 4, GPT-2 / GPT-J / GPT-Neo / GPT-NeoX / GPT-BigCode, OPT, Bloom, Falcon, MPT, BitNet, GPT-OSS, DeepSeek V2 / V3, Granite / Granite SWA / GraniteMoE / GraniteMoE-SWA / GraniteMoE-Shared, OLMo / OLMo2 / OLMo3 / OLMoE / FlexOlmo / OLMo-Hybrid, SmolLM3, StarCoder2, Nemotron (dense), Cohere / Cohere2 / Cohere2-MoE, GLM / GLM4 / GLM4-MoE, StableLM, EXAONE 4 / EXAONE MoE, Arcee, PhiMoE, Hunyuan V1 MoE, ERNIE 4.5 MoE, DBRX, DiffLlama, Jamba (Mamba-1), Nemotron-H (Nano + Super LatentMoE). Llama-identical checkpoints (Helium, ERNIE 4.5 dense, Hunyuan dense, Seed-OSS, CWM, Ministral / Mistral3 text) auto-detect onto Llama/Mistral. Qwen2-VL / InternVL / Kimi text backbones alias onto Qwen2 / Llama / DeepSeek V3. Native MTP on Qwen3.5, DeepSeek V3, and Nemotron Super. Vision generate for Qwen3.5, Qwen2-VL / Qwen2.5-VL, Gemma3, Llama 4, and Mistral3. Sampling (`temperature`, `top_k`, `top_p`, `seed`) on decode. C++ RMSNorm / SiLU-mul kernels with a Python fallback. NVFP4/FP8 dequant on load.
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
| [`engine/layers/gdn.py`](engine/layers/gdn.py) | Qwen Gated DeltaNet reference + chunked CUDA prefill |
| [`engine/vision.py`](engine/vision.py) | Qwen image/video tower, merger, M-RoPE positions |
| [`engine/mtp.py`](engine/mtp.py) | Native Qwen MTP draft layer sharing embed/lm_head |
| [`engine/cache.py`](engine/cache.py) | `RuntimeState` = KV + Mamba conv/SSM |
| [`engine/agent_api.py`](engine/agent_api.py) | `inspect_capabilities` / `load_engine` — public agent surface |
| [`engine/capabilities.py`](engine/capabilities.py) | Older capability dict (Spark also has this) |
| [`engine/runtime.py`](engine/runtime.py) | Older handle API; prefer `agent_api.py` if they overlap |

## Run (Spark)

```bash
cd ~/Projects/infer
source .venv/bin/activate
export PYTHONPATH=~/Projects/infer

python -m engine.chat --model testdata/nemotron3-nano-30b-a3b --inspect

python -m engine.chat --model ~/models/Llama-3.2-1B-Instruct --device cuda \
  --prompt "The capital of France is" --max-new-tokens 32 --temperature 0.8 --top-p 0.9 --seed 0

python -m engine.chat --model ~/models/Qwen3.8-27B --device cuda \
  --prompt "Write one sentence." --max-new-tokens 32 --mtp-draft-tokens 3
```

```python
from engine.agent_api import inspect_capabilities, load_engine

print(inspect_capabilities("testdata/nemotron3-nano-30b-a3b").to_dict())
eng = load_engine("~/models/Llama-3.2-1B-Instruct", device="cuda")
print(eng.generate("The capital of France is", max_new_tokens=16))
print(eng.generate("The capital of France is", max_new_tokens=16, temperature=0.8, top_p=0.9, seed=0))

# Qwen image/video messages use the checkpoint's official AutoProcessor.
# content may contain PIL images, local paths, URLs, or videos accepted by it.
qwen = load_engine("~/models/Qwen3.8-27B", device="cuda")
messages = [{"role": "user", "content": [
    {"type": "image", "image": image},
    {"type": "text", "text": "What is shown?"},
]}]
print(qwen.generate_messages(messages, max_new_tokens=32))
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
