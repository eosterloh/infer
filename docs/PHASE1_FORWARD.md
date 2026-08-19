# Phase 1 — Transformer forward (no KV cache yet)

**Status:** plan only (Phase 0 load is done).  
**Model:** `meta-llama/Llama-3.2-1B-Instruct` at `~/models/Llama-3.2-1B-Instruct`.  
**Stack:** PyTorch + CUDA, our weight dict from Phase 0.

---

## Goal

Turn loaded weights into **logits**.

Given token ids, run the full Llama forward pass (embed → 16 layers → norm → lm_head) and get a score for every vocab token at every position.

**Definition of done**

1. `model.forward(input_ids)` returns logits shaped `[batch, seq, vocab_size]`.
2. On a short fixed prompt, our greedy next-token (and preferably full greedy continuation for ~32 tokens) **matches HuggingFace `LlamaForCausalLM`** within floating-point noise (same top-1; logit MSE tiny).
3. CLI can print: prompt → next token id/text (still OK if slow — we recompute attention every step).
4. **No KV cache yet.** `cache` arg stays unused / ignored. That is Phase 2.

Phase 1 is “the math works.” Phase 2 is “decode is fast.”

---

## Where we are vs where we’re going

```text
Phase 0  LOAD     weights + config + tokenizer on CUDA     ✅ done
Phase 1  FORWARD  input_ids → logits (full recompute)      ← this doc
Phase 2  DECODE   KV cache + streaming generate
Phase 3  KNOBS    dtype / window / paged / eviction / …
```

Real engines also separate these: you can have correct weights and still be wrong in attention; you can have correct forward and still be slow without KV.

---

## Mental model: what “forward” means

The model is a function:

```text
token ids  →  vectors  →  16× (attend + MLP)  →  vocab scores (logits)
```

Nothing mystical: **matrix multiply, add, normalize, softmax**, repeated. The “intelligence” is entirely in the weight numbers we already loaded.

For Llama-3.2-1B Instruct the blueprint is:

| Knob | Value |
|---|---|
| layers | 16 |
| hidden `H` | 2048 |
| MLP `I` | 8192 |
| Q heads / KV heads | 32 / 8 (GQA) |
| head dim | 64 |
| vocab | 128256 |
| act | SiLU (SwiGLU MLP) |
| norm | RMSNorm |
| position | RoPE (Llama3 scaling in config) |

---

## Data flow (one forward call)

```text
input_ids: int64 [B, S]
      │
      │  embed.weight[id]     lookup rows
      ▼
x: bf16 [B, S, H]
      │
      │  for layer in 0 .. 15:
      │       x = x + Attention(RMSNorm(x))
      │       x = x + MLP(RMSNorm(x))
      ▼
x: [B, S, H]
      │
      │  final_norm
      │  lm_head  (tied to embed for this model)
      ▼
logits: [B, S, V]
```

**Residual connections** (`x = x + …`) are why deep stacks train/infer stably: each block adds a delta, it doesn’t replace the stream wholesale.

**Causal mask:** position `t` may only attend to positions `≤ t`. Without that, the model would cheat by looking at the future.

---

## Block 1 — Embedding

```text
x[b, s, :] = embed.weight[ input_ids[b, s] ]
```

Each token id picks one row of a `[V, H]` table. That vector is the starting residual stream.

Our Phase 0 key: `embed.weight`.

---

## Block 2 — RMSNorm

Before attention and before MLP (pre-norm Llama):

```text
RMSNorm(x) = (x / sqrt(mean(x²) + eps)) * weight
```

- No mean-centering (unlike LayerNorm) — that’s the “RMS” part.
- `weight` is a learned `[H]` scale (`input_norm`, `post_attn_norm`, `final_norm`).
- `eps` from config (`1e-5`).

Do this in float32 accumulate if needed, then cast back to bf16 — common numerical hygiene.

---

## Block 3 — Attention (GQA + RoPE)

This is the expensive / subtle part.

### 3a. Project Q, K, V

```text
q = x @ W_q^T     → [B, S, Nq * Dh] = [B, S, 2048]
k = x @ W_k^T     → [B, S, Nkv * Dh] = [B, S, 512]
v = x @ W_v^T     → [B, S, Nkv * Dh] = [B, S, 512]
```

Reshape to heads:

```text
q: [B, S, 32, 64] → [B, 32, S, 64]
k: [B, S,  8, 64] → [B,  8, S, 64]
v: [B, S,  8, 64] → [B,  8, S, 64]
```

**GQA:** 32 query heads share 8 KV heads (4 Q heads per KV group). At attention time we **repeat** K/V along the head axis (or use grouped math) so each Q head has a KV partner.

Phase 0 keys: `layers.{i}.attn.{q,k,v}.weight`.

### 3b. RoPE — rotary position embeddings

Before attention scores, rotate Q and K by a position-dependent angle so the model knows *order* without absolute position embeddings glued onto `x`.

Intuition: each head_dim pair `(x0, x1)` is treated like a 2D vector and rotated by `θ(position, frequency)`. Relative position then shows up naturally in `q·k`.

Llama 3.2 also has **`rope_scaling`** in config (`rope_type: llama3`, factor 32, …) so long context (up to 128k claim) stretches the frequencies. Phase 1 must implement this correctly or long prompts diverge from HF even if short ones look fine.

Implementation plan:

1. Build cos/sin caches from `rope_theta` + `rope_scaling` + positions `0..S-1`.
2. Apply to `q` and `k` (not `v`).

### 3c. Scores, mask, softmax, weighted V

```text
scores = (q @ k^T) / sqrt(Dh)     # [B, Nq, S, S]
scores = scores + causal_mask     # -inf where j > i
attn   = softmax(scores)
out    = attn @ v                 # [B, Nq, S, Dh]
```

Merge heads → `[B, S, H]`, then:

```text
y = out @ W_o^T                   # o_proj
```

Phase 0 key: `layers.{i}.attn.o.weight`.

### 3d. Residual

```text
x = x + y
```

---

## Block 4 — MLP (SwiGLU)

```text
h = RMSNorm(x)
gate = h @ W_gate^T               # [B, S, I]
up   = h @ W_up^T                 # [B, S, I]
hidden = silu(gate) * up          # SwiGLU
y = hidden @ W_down^T             # [B, S, H]
x = x + y
```

`silu(z) = z * sigmoid(z)`.

Phase 0 keys: `layers.{i}.mlp.{gate,up,down}.weight`.

---

## Block 5 — Final norm + lm_head

After layer 15:

```text
x = RMSNorm(x)                    # final_norm.weight
logits = x @ W_lm^T               # [B, S, V]
```

For this checkpoint `tie_word_embeddings=true`, so `W_lm` **is** `embed.weight` (we already aliased `lm_head.weight` in Phase 0).

Logits are **not** probabilities. Softmax(logits) would be a distribution; for greedy decode we only need `argmax` on the last position:

```text
next_id = argmax(logits[0, -1, :])
```

---

## Phase 1 intentionally ignores KV cache

Naive decode for `T` new tokens:

```text
for each new token:
    forward(entire sequence so far)   # O(S²) attention each time
```

That’s correct but wasteful — we redo attention over the whole prefix every step. **Phase 2** stores K/V from past tokens and only computes the new row. Phase 1 still does full recompute so we can prove numerics without cache bugs masking math bugs.

---

## Implementation order (build sequence)

Do these as small, testable slices — don’t write all 16 layers before checking embed.

| Step | Deliverable | How to know it’s right |
|---|---|---|
| 1 | `rms_norm(x, weight, eps)` | Match HF on a random tensor |
| 2 | `apply_rope(q, k, positions, config)` | Match HF cos/sin application on short S |
| 3 | `attention(x, layer_weights)` one layer | Match HF hidden states after layer 0 attn |
| 4 | `mlp(x, layer_weights)` | Match HF after layer 0 MLP |
| 5 | Full `layer(x)` × 16 + final + lm_head | Match HF logits on short prompt |
| 6 | Greedy loop (no cache) | Same token ids as HF for ~32 steps |
| 7 | Wire CLI (`--prompt`, print continuation) | Human-visible smoke |

Suggested new/touched files:

```text
engine/layers.py      # rmsnorm, rope, attention, mlp (pure functions or small modules)
engine/model.py       # real forward using weights dict
engine/generate.py    # greedy_from_logits / generate_greedy (still no KV)
engine/chat.py        # optional: --prompt path for Phase 1 demo
scripts/parity_check.py  # compare vs transformers LlamaForCausalLM
```

Keep using the **flat weight dict** from Phase 0 (learning-friendly). Optionally wrap into `nn.Module` later; not required for Phase 1.

---

## Parity methodology (don’t skip this)

HF is the oracle:

```python
# pseudocode
hf = AutoModelForCausalLM.from_pretrained(path, torch_dtype=bfloat16).cuda().eval()
ours = LlamaModel(config, weights)

ids = tokenizer.encode("The capital of France is")
with torch.no_grad():
    logits_hf = hf(ids).logits
    logits_ours = ours.forward(ids)

# checks
assert torch.equal(logits_hf.argmax(-1), logits_ours.argmax(-1))  # ideal
# or: max |logit diff| / MSE under a tight threshold
```

Debug strategy if mismatch:

1. Compare after embed.
2. Compare after layer 0 attention residual.
3. Compare after layer 0 MLP residual.
4. Binary-search first diverging layer.
5. Usual culprits: RoPE (especially llama3 scaling), GQA repeat, causal mask, RMSNorm in bf16, wrong transpose on `nn.Linear` vs explicit `@ W.T`.

**Linear layout note:** HF `nn.Linear` stores `weight` as `[out, in]` and computes `x @ W.T`. Our safetensors are that same layout. Explicit matmul must use `.T` (or `F.linear`).

---

## How this compares to “real” inference engines

| Concern | Phase 1 (us) | llama.cpp / vLLM |
|---|---|---|
| Correct matmul graph | Yes — goal | Yes |
| KV cache | No | Yes |
| Custom CUDA kernels | No — PyTorch | Often yes |
| Batching many users | No | Yes (especially vLLM) |
| Quantized weights | No | Common |
| RoPE / GQA / SwiGLU | Must match | Same algorithms |

We’re building the **reference graph**. Engines optimize the same graph.

---

## Explicit non-goals for Phase 1

- KV cache / paged attention  
- Sampling beyond greedy (temp/top-p can be a tiny add-on if parity is green)  
- Chat template polish (nice-to-have once greedy works)  
- FlashAttention / SDPA required (using `torch.nn.functional.scaled_dot_product_attention` with `is_causal=True` is OK if parity holds)  
- MoE / Mamba / NVFP4  

---

## Done checklist

- [ ] RMSNorm, RoPE (llama3 scaling), GQA attention, SwiGLU MLP implemented  
- [ ] `LlamaModel.forward` returns `[B,S,V]` logits  
- [ ] Parity script vs HF on short prompt (top-1 + logit tolerance)  
- [ ] Greedy ~32-token match vs HF  
- [ ] CLI demo path that prints generated text (slow OK)  
- [ ] Cache still unused; Phase 2 doc/plan next  

---

## One-sentence summary

**Phase 1 implements the deterministic Llama recipe that turns our Phase 0 GPU tensors into logits, proven by matching HuggingFace, before we complicate life with a KV cache.**
