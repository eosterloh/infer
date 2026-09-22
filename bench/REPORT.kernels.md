# infer benchmarks — spark-6b20

Every number here was measured on the host named above. Each row is one checkpoint from disk: a prefill of the stated length, then a fixed number of greedy decode steps, best of the repetitions.

Each table compares one configuration against `baseline`:

- `baseline` — this engine with the extension off, so the Python rework shows separately
- `kernels` — the compiled kernels, BF16 weights
- `graph` — the kernels plus a captured CUDA graph for the decode step

The **output** column is the correctness check, and it is deliberately not an exact match on the whole greedy chain. Reassociating a reduction moves a logit by about a BF16 ulp, and sixteen steps of greedy decoding turn one near-tie into a completely different sentence. So a configuration passes on its first token plus its top-5 logits staying within a tolerance that scales with their magnitude, and the table says where the chain split when it did. A configuration that changes precision is reported and never failed: its job is to be faster and cheaper, and how much answer that costs is the finding, not a bug.

Engine at `aae0929`.

## family coverage

| family | models measured |
|---|---|
| hybrid recurrent (Mamba2 / GDN scan) | NVIDIA-Nemotron-3-Nano-30B-A3B-BF16, Qwen3.8-27B |
| MLA (compressed KV cache) | *no checkpoint of this family on this host* |
| sparse MoE dispatch | NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 |
| legacy attention (learned norms / parallel residual) | gpt2, pythia-410m |
| sliding window / soft-capped attention | gemma-2-2b-it |
| dense attention + MLP | Llama-3.2-1B-Instruct, Llama-3.2-3B-Instruct, Mistral-7B-Instruct-v0.3, Phi-3-mini-4k-instruct, Qwen2.5-1.5B-Instruct, Qwen3-0.6B, SmolLM2-1.7B-Instruct, TinyLlama-1.1B-Chat-v1.0, Yi-1.5-6B-Chat |

## how much of the memory bus is left

Measured read bandwidth on this host is **247.3 GB/s** (`scripts/bench_bandwidth.py`), against 273 GB/s on the specification. A BF16 decode step reads every weight once, so the ceiling below is `params x 2 bytes / bandwidth` and no kernel goes past it. Packed and MoE runs are left out: neither moves two bytes per parameter.

| model | GB read per token | ceiling tok/s | best measured | of ceiling |
|---|---|---|---|---|
| Qwen3.8-27B | 55.56 | 4.5 | 4.46 | 100% |
| Yi-1.5-6B-Chat | 12.12 | 20.4 | 19.1 | 94% |
| Mistral-7B-Instruct-v0.3 | 14.50 | 17.1 | 15.69 | 92% |
| gemma-2-2b-it | 5.23 | 47.3 | 42.75 | 90% |
| Llama-3.2-3B-Instruct | 6.43 | 38.5 | 34.3 | 89% |
| Qwen2.5-1.5B-Instruct | 3.09 | 80.1 | 68.93 | 86% |
| Phi-3-mini-4k-instruct | 7.64 | 32.4 | 27.39 | 85% |
| Llama-3.2-1B-Instruct | 2.47 | 100.1 | 78.5 | 78% |
| SmolLM2-1.7B-Instruct | 3.42 | 72.3 | 55.94 | 77% |
| TinyLlama-1.1B-Chat-v1.0 | 2.20 | 112.4 | 83.03 | 74% |
| pythia-410m | 0.81 | 305.1 | 222.48 | 73% |
| Qwen3-0.6B | 1.19 | 207.4 | 140.41 | 68% |
| gpt2 | 0.25 | 993.7 | 546.09 | 55% |

## kernels vs baseline

| model | params | decode tok/s base | decode tok/s new | speedup | prefill tok/s base | prefill tok/s new | speedup | output |
|---|---|---|---|---|---|---|---|---|
| Llama-3.2-1B-Instruct | 1.24B | 62.7 | 78.26 | 1.25x | 15275.57 | 24532.65 | 1.61x | same (Δlogit 0.125) |
| Llama-3.2-3B-Instruct | 3.21B | 25.33 | 33.38 | 1.32x | 7034.6 | 10486.0 | 1.49x | same (Δlogit 0.125) |
| Mistral-7B-Instruct-v0.3 | 7.25B | 14.13 | 15.48 | 1.10x | 3092.1 | 4499.36 | 1.46x | same (Δlogit 0.000) |
| NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 | 31.58B | 8.98 | 29.44 | 3.28x | 294.11 | 606.56 | 2.06x | same (Δlogit 0.250) |
| Phi-3-mini-4k-instruct | 3.82B | 21.3 | 27.25 | 1.28x | 5526.93 | 7976.73 | 1.44x | same head, chain splits at 5/16 (Δlogit 0.000) |
| Qwen2.5-1.5B-Instruct | 1.54B | 41.59 | 65.48 | 1.57x | 7181.83 | 10305.38 | 1.43x | same (Δlogit 0.000) |
| Qwen3-0.6B | 0.60B | 75.76 | 127.36 | 1.68x | 19178.94 | 32598.01 | 1.70x | same (Δlogit 0.125) |
| Qwen3.8-27B | 27.78B | 4.11 | 4.37 | 1.06x | 472.54 | 471.96 | 1.00x | same (Δlogit 0.062) |
| SmolLM2-1.7B-Instruct | 1.71B | 44.6 | 55.42 | 1.24x | 10599.56 | 16684.14 | 1.57x | same (Δlogit 0.125) |
| TinyLlama-1.1B-Chat-v1.0 | 1.10B | 62.9 | 83.03 | 1.32x | 15844.42 | 25019.41 | 1.58x | same (Δlogit 0.062) |
| Yi-1.5-6B-Chat | 6.06B | 14.89 | 18.84 | 1.27x | 3809.47 | 5512.2 | 1.45x | same head, chain splits at 6/16 (Δlogit 0.125) |
| gemma-2-2b-it | 2.61B | 27.77 | 40.61 | 1.46x | 5482.08 | 7699.35 | 1.40x | same (Δlogit 0.125) |
| gpt2 | 0.12B | 456.89 | 489.67 | 1.07x | 51360.3 | 50548.22 | 0.98x | same head, chain splits at 9/16 (Δlogit 0.000) |
| pythia-410m | 0.41B | 163.02 | 197.17 | 1.21x | 17204.15 | 18857.88 | 1.10x | same (Δlogit 0.070) |

## graph vs baseline

| model | params | decode tok/s base | decode tok/s new | speedup | prefill tok/s base | prefill tok/s new | speedup | output |
|---|---|---|---|---|---|---|---|---|
| Llama-3.2-1B-Instruct | 1.24B | 62.7 | 78.5 | 1.25x | 15275.57 | 24390.57 | 1.60x | same (Δlogit 0.125) |
| Llama-3.2-3B-Instruct | 3.21B | 25.33 | 34.3 | 1.35x | 7034.6 | 10503.59 | 1.49x | same (Δlogit 0.125) |
| Mistral-7B-Instruct-v0.3 | 7.25B | 14.13 | 15.69 | 1.11x | 3092.1 | 4494.39 | 1.45x | same (Δlogit 0.000) |
| NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 | 31.58B | 8.98 | 32.2 | 3.59x | 294.11 | 610.98 | 2.08x | same (Δlogit 0.250) |
| Phi-3-mini-4k-instruct | 3.82B | 21.3 | 27.39 | 1.29x | 5526.93 | 8008.3 | 1.45x | same head, chain splits at 5/16 (Δlogit 0.000) |
| Qwen2.5-1.5B-Instruct | 1.54B | 41.59 | 68.93 | 1.66x | 7181.83 | 10199.3 | 1.42x | same (Δlogit 0.000) |
| Qwen3-0.6B | 0.60B | 75.76 | 140.41 | 1.85x | 19178.94 | 32624.7 | 1.70x | same (Δlogit 0.125) |
| Qwen3.8-27B | 27.78B | 4.11 | 4.46 | 1.09x | 472.54 | 535.8 | 1.13x | same (Δlogit 0.062) |
| SmolLM2-1.7B-Instruct | 1.71B | 44.6 | 55.94 | 1.25x | 10599.56 | 16696.33 | 1.58x | same (Δlogit 0.125) |
| TinyLlama-1.1B-Chat-v1.0 | 1.10B | 62.9 | 82.95 | 1.32x | 15844.42 | 24774.58 | 1.56x | same (Δlogit 0.062) |
| Yi-1.5-6B-Chat | 6.06B | 14.89 | 19.1 | 1.28x | 3809.47 | 5601.52 | 1.47x | same head, chain splits at 6/16 (Δlogit 0.125) |
| gemma-2-2b-it | 2.61B | 27.77 | 42.75 | 1.54x | 5482.08 | 7764.77 | 1.42x | same (Δlogit 0.125) |
| gpt2 | 0.12B | 456.89 | 546.09 | 1.20x | 51360.3 | 50452.93 | 0.98x | same head, chain splits at 9/16 (Δlogit 0.000) |
| pythia-410m | 0.41B | 163.02 | 222.48 | 1.36x | 17204.15 | 20003.34 | 1.16x | same (Δlogit 0.070) |

