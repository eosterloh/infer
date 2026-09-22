# infer benchmarks — spark-6b20

Every number here was measured on the host named above. Each row is one checkpoint from disk: a prefill of the stated length, then a fixed number of greedy decode steps, best of the repetitions.

Each table compares one configuration against `origin`:

- `origin` — the engine before any of this work, at the revision named in the sweep
- `baseline` — this engine with the extension off, so the Python rework shows separately
- `kernels` — the compiled kernels, BF16 weights
- `graph` — the kernels plus a captured CUDA graph for the decode step
- `nvfp4` — the kernels with weights packed to NVFP4 at load

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

## baseline vs origin

| model | params | decode tok/s base | decode tok/s new | speedup | prefill tok/s base | prefill tok/s new | speedup | output |
|---|---|---|---|---|---|---|---|---|
| Llama-3.2-1B-Instruct | 1.24B | 57.07 | 62.7 | 1.10x | 9542.11 | 15275.57 | 1.60x | same (Δlogit 0.125) |
| Llama-3.2-3B-Instruct | 3.21B | 23.22 | 25.33 | 1.09x | 4969.5 | 7034.6 | 1.42x | same (Δlogit 0.062) |
| Mistral-7B-Instruct-v0.3 | 7.25B | 13.27 | 14.13 | 1.06x | 2493.52 | 3092.1 | 1.24x | same (Δlogit 0.000) |
| NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 | 31.58B | — | 8.98 | — | — | 294.11 | — | no origin row |
| Phi-3-mini-4k-instruct | 3.82B | 20.48 | 21.3 | 1.04x | 4344.02 | 5526.93 | 1.27x | same head, chain splits at 5/16 (Δlogit 0.250) |
| Qwen2.5-1.5B-Instruct | 1.54B | 42.72 | 41.59 | 0.97x | 8366.11 | 7181.83 | 0.86x | same (Δlogit 0.125) |
| Qwen3-0.6B | 0.60B | — | 75.76 | — | — | 19178.94 | — | no origin row |
| Qwen3.8-27B | 27.78B | — | 4.11 | — | — | 472.54 | — | no origin row |
| SmolLM2-1.7B-Instruct | 1.71B | 40.21 | 44.6 | 1.11x | 6609.1 | 10599.56 | 1.60x | same (Δlogit 0.125) |
| TinyLlama-1.1B-Chat-v1.0 | 1.10B | 57.3 | 62.9 | 1.10x | 8113.61 | 15844.42 | 1.95x | same (Δlogit 0.062) |
| Yi-1.5-6B-Chat | 6.06B | 13.9 | 14.89 | 1.07x | 2858.58 | 3809.47 | 1.33x | same (Δlogit 0.125) |
| gemma-2-2b-it | 2.61B | — | 27.77 | — | — | 5482.08 | — | no origin row |
| gpt2 | 0.12B | — | 456.89 | — | — | 51360.3 | — | no origin row |
| pythia-410m | 0.41B | — | 163.02 | — | — | 17204.15 | — | no origin row |

## kernels vs origin

| model | params | decode tok/s base | decode tok/s new | speedup | prefill tok/s base | prefill tok/s new | speedup | output |
|---|---|---|---|---|---|---|---|---|
| Llama-3.2-1B-Instruct | 1.24B | 57.07 | 78.26 | 1.37x | 9542.11 | 24532.65 | 2.57x | same (Δlogit 0.125) |
| Llama-3.2-3B-Instruct | 3.21B | 23.22 | 33.38 | 1.44x | 4969.5 | 10486.0 | 2.11x | same (Δlogit 0.125) |
| Mistral-7B-Instruct-v0.3 | 7.25B | 13.27 | 15.48 | 1.17x | 2493.52 | 4499.36 | 1.80x | same (Δlogit 0.000) |
| NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 | 31.58B | — | 29.44 | — | — | 606.56 | — | no origin row |
| Phi-3-mini-4k-instruct | 3.82B | 20.48 | 27.25 | 1.33x | 4344.02 | 7976.73 | 1.84x | same (Δlogit 0.250) |
| Qwen2.5-1.5B-Instruct | 1.54B | 42.72 | 65.48 | 1.53x | 8366.11 | 10305.38 | 1.23x | same (Δlogit 0.125) |
| Qwen3-0.6B | 0.60B | — | 127.36 | — | — | 32598.01 | — | no origin row |
| Qwen3.8-27B | 27.78B | — | 4.37 | — | — | 471.96 | — | no origin row |
| SmolLM2-1.7B-Instruct | 1.71B | 40.21 | 55.42 | 1.38x | 6609.1 | 16684.14 | 2.52x | same (Δlogit 0.125) |
| TinyLlama-1.1B-Chat-v1.0 | 1.10B | 57.3 | 83.03 | 1.45x | 8113.61 | 25019.41 | 3.08x | same (Δlogit 0.062) |
| Yi-1.5-6B-Chat | 6.06B | 13.9 | 18.84 | 1.36x | 2858.58 | 5512.2 | 1.93x | same head, chain splits at 6/16 (Δlogit 0.125) |
| gemma-2-2b-it | 2.61B | — | 40.61 | — | — | 7699.35 | — | no origin row |
| gpt2 | 0.12B | — | 489.67 | — | — | 50548.22 | — | no origin row |
| pythia-410m | 0.41B | — | 197.17 | — | — | 18857.88 | — | no origin row |

## graph vs origin

| model | params | decode tok/s base | decode tok/s new | speedup | prefill tok/s base | prefill tok/s new | speedup | output |
|---|---|---|---|---|---|---|---|---|
| Llama-3.2-1B-Instruct | 1.24B | 57.07 | 78.5 | 1.38x | 9542.11 | 24390.57 | 2.56x | same (Δlogit 0.125) |
| Llama-3.2-3B-Instruct | 3.21B | 23.22 | 34.3 | 1.48x | 4969.5 | 10503.59 | 2.11x | same (Δlogit 0.125) |
| Mistral-7B-Instruct-v0.3 | 7.25B | 13.27 | 15.69 | 1.18x | 2493.52 | 4494.39 | 1.80x | same (Δlogit 0.000) |
| NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 | 31.58B | — | 32.2 | — | — | 610.98 | — | no origin row |
| Phi-3-mini-4k-instruct | 3.82B | 20.48 | 27.39 | 1.34x | 4344.02 | 8008.3 | 1.84x | same (Δlogit 0.250) |
| Qwen2.5-1.5B-Instruct | 1.54B | 42.72 | 68.93 | 1.61x | 8366.11 | 10199.3 | 1.22x | same (Δlogit 0.125) |
| Qwen3-0.6B | 0.60B | — | 140.41 | — | — | 32624.7 | — | no origin row |
| Qwen3.8-27B | 27.78B | — | 4.46 | — | — | 535.8 | — | no origin row |
| SmolLM2-1.7B-Instruct | 1.71B | 40.21 | 55.94 | 1.39x | 6609.1 | 16696.33 | 2.53x | same (Δlogit 0.125) |
| TinyLlama-1.1B-Chat-v1.0 | 1.10B | 57.3 | 82.95 | 1.45x | 8113.61 | 24774.58 | 3.05x | same (Δlogit 0.062) |
| Yi-1.5-6B-Chat | 6.06B | 13.9 | 19.1 | 1.37x | 2858.58 | 5601.52 | 1.96x | same head, chain splits at 6/16 (Δlogit 0.125) |
| gemma-2-2b-it | 2.61B | — | 42.75 | — | — | 7764.77 | — | no origin row |
| gpt2 | 0.12B | — | 546.09 | — | — | 50452.93 | — | no origin row |
| pythia-410m | 0.41B | — | 222.48 | — | — | 20003.34 | — | no origin row |

## nvfp4 vs origin

| model | params | decode tok/s base | decode tok/s new | speedup | prefill tok/s base | prefill tok/s new | speedup | output |
|---|---|---|---|---|---|---|---|---|
| Llama-3.2-1B-Instruct | 1.50B | 57.07 | 173.0 | 3.03x | 9542.11 | 14699.77 | 1.54x | nvfp4: 16/16 tokens kept, top-5 reordered |
| Llama-3.2-3B-Instruct | 3.61B | 23.22 | 87.68 | 3.78x | 4969.5 | 5922.57 | 1.19x | nvfp4: 16/16 tokens kept, Δlogit 0.688 |
| Mistral-7B-Instruct-v0.3 | 7.25B | 13.27 | 46.68 | 3.52x | 2493.52 | 2662.64 | 1.07x | nvfp4: 16/16 tokens kept, Δlogit 0.562 |
| NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 | 31.58B | — | 50.93 | — | — | 632.06 | — | no origin row |
| Phi-3-mini-4k-instruct | 3.82B | 20.48 | 68.22 | 3.33x | 4344.02 | 4735.52 | 1.09x | nvfp4: 5/16 tokens kept, Δlogit 1.000 |
| Qwen2.5-1.5B-Instruct | 1.78B | 42.72 | 135.62 | 3.17x | 8366.11 | 7827.4 | 0.94x | nvfp4: 6/16 tokens kept, Δlogit 1.000 |
| Qwen3-0.6B | 0.75B | — | 211.04 | — | — | 23493.51 | — | no origin row |
| Qwen3.8-27B | 27.78B | — | 9.05 | — | — | 367.72 | — | no origin row |
| SmolLM2-1.7B-Instruct | 1.81B | 40.21 | 121.59 | 3.02x | 6609.1 | 10364.21 | 1.57x | nvfp4: 3/16 tokens kept, top-5 reordered |
| TinyLlama-1.1B-Chat-v1.0 | 1.10B | 57.3 | 159.84 | 2.79x | 8113.61 | 16885.5 | 2.08x | nvfp4: 2/16 tokens kept, top-5 reordered |
| Yi-1.5-6B-Chat | 6.06B | 13.9 | 54.32 | 3.91x | 2858.58 | 3254.19 | 1.14x | nvfp4: 2/16 tokens kept, top-5 reordered |
| gemma-2-2b-it | 3.20B | — | 97.15 | — | — | 5387.39 | — | no origin row |
| gpt2 | 0.16B | — | 620.35 | — | — | 37979.87 | — | no origin row |
| pythia-410m | 0.41B | — | 294.51 | — | — | 14923.08 | — | no origin row |

# versus `baseline`

Same engine, same checkpoints, `baseline` as the before. This is what the kernels are worth on their own, and unlike the tables above it covers every model measured.

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

## nvfp4 vs baseline

| model | params | decode tok/s base | decode tok/s new | speedup | prefill tok/s base | prefill tok/s new | speedup | output |
|---|---|---|---|---|---|---|---|---|
| Llama-3.2-1B-Instruct | 1.50B | 62.7 | 173.0 | 2.76x | 15275.57 | 14699.77 | 0.96x | nvfp4: 16/16 tokens kept, top-5 reordered |
| Llama-3.2-3B-Instruct | 3.61B | 25.33 | 87.68 | 3.46x | 7034.6 | 5922.57 | 0.84x | nvfp4: 16/16 tokens kept, Δlogit 0.688 |
| Mistral-7B-Instruct-v0.3 | 7.25B | 14.13 | 46.68 | 3.30x | 3092.1 | 2662.64 | 0.86x | nvfp4: 16/16 tokens kept, Δlogit 0.562 |
| NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 | 31.58B | 8.98 | 50.93 | 5.67x | 294.11 | 632.06 | 2.15x | nvfp4: 2/16 tokens kept, top-5 reordered |
| Phi-3-mini-4k-instruct | 3.82B | 21.3 | 68.22 | 3.20x | 5526.93 | 4735.52 | 0.86x | nvfp4: 5/16 tokens kept, Δlogit 1.000 |
| Qwen2.5-1.5B-Instruct | 1.78B | 41.59 | 135.62 | 3.26x | 7181.83 | 7827.4 | 1.09x | nvfp4: 6/16 tokens kept, Δlogit 1.000 |
| Qwen3-0.6B | 0.75B | 75.76 | 211.04 | 2.79x | 19178.94 | 23493.51 | 1.22x | nvfp4: 9/16 tokens kept, Δlogit 2.000 |
| Qwen3.8-27B | 27.78B | 4.11 | 9.05 | 2.20x | 472.54 | 367.72 | 0.78x | nvfp4: 16/16 tokens kept, Δlogit 0.188 |
| SmolLM2-1.7B-Instruct | 1.81B | 44.6 | 121.59 | 2.73x | 10599.56 | 10364.21 | 0.98x | nvfp4: 3/16 tokens kept, top-5 reordered |
| TinyLlama-1.1B-Chat-v1.0 | 1.10B | 62.9 | 159.84 | 2.54x | 15844.42 | 16885.5 | 1.07x | nvfp4: 2/16 tokens kept, top-5 reordered |
| Yi-1.5-6B-Chat | 6.06B | 14.89 | 54.32 | 3.65x | 3809.47 | 3254.19 | 0.85x | nvfp4: 2/16 tokens kept, top-5 reordered |
| gemma-2-2b-it | 3.20B | 27.77 | 97.15 | 3.50x | 5482.08 | 5387.39 | 0.98x | nvfp4: 4/16 tokens kept, Δlogit 0.938 |
| gpt2 | 0.16B | 456.89 | 620.35 | 1.36x | 51360.3 | 37979.87 | 0.74x | nvfp4: 1/16 tokens kept, top-5 reordered |
| pythia-410m | 0.41B | 163.02 | 294.51 | 1.81x | 17204.15 | 14923.08 | 0.87x | nvfp4: 7/16 tokens kept, top-5 reordered |

