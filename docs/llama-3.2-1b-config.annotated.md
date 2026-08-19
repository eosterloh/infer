# Llama-3.2-1B-Instruct — annotated config.json

JSON cannot contain comments; this markdown mirrors the public 1B Instruct
blueprint with explanations. Exact values come from Meta's `config.json`.

```json
{
  "architectures": ["LlamaForCausalLM"],
  "model_type": "llama",
  "vocab_size": 128256,
  "hidden_size": 2048,
  "intermediate_size": 8192,
  "num_hidden_layers": 16,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "head_dim": 64,
  "hidden_act": "silu",
  "rms_norm_eps": 1e-05,
  "rope_theta": 500000.0,
  "rope_scaling": {
    "rope_type": "llama3",
    "factor": 32.0,
    "original_max_position_embeddings": 8192,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0
  },
  "max_position_embeddings": 131072,
  "tie_word_embeddings": true,
  "torch_dtype": "bfloat16",
  "attention_bias": false,
  "mlp_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 128000,
  "eos_token_id": [128001, 128008, 128009],
  "use_cache": true,
  "initializer_range": 0.02,
  "pretraining_tp": 1
}
```

## Field notes

| Field | Meaning |
|---|---|
| `hidden_size` | Residual stream width **H** (2048) |
| `intermediate_size` | MLP width **I** (8192) |
| `num_hidden_layers` | Stack depth **L** (16) |
| `num_attention_heads` / `num_key_value_heads` | GQA: 32 Q heads, 8 KV heads |
| `head_dim` | 64; `32 * 64 = 2048` |
| `tie_word_embeddings` | `lm_head` shares storage with `embed_tokens` |
| `torch_dtype` | Phase 0 loads everything as BF16 |
| `rope_*` | Stored in Phase 0; used when RoPE lands in Phase 1+ |

## Expected weight shapes (from this config)

- `embed`: `[128256, 2048]`
- `q`: `[2048, 2048]`, `k`/`v`: `[512, 2048]` (8 × 64)
- `o`: `[2048, 2048]`
- `gate`/`up`: `[8192, 2048]`, `down`: `[2048, 8192]`
- norms: `[2048]`
- × 16 layers; `lm_head` tied to embed
