#!/usr/bin/env python3
"""Compare our forward / greedy decode against HuggingFace LlamaForCausalLM."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Allow `python scripts/parity_check.py` from repo root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.config import ModelConfig
from engine.model import LlamaModel
from engine.weights import load_weights


def _dtype_from_name(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


@torch.inference_mode()
def greedy_ids(
    logits_fn,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_ids: set[int],
) -> list[int]:
    tokens = input_ids
    out: list[int] = []
    for _ in range(max_new_tokens):
        logits = logits_fn(tokens)
        next_id = int(torch.argmax(logits[0, -1, :]).item())
        out.append(next_id)
        tokens = torch.cat(
            [tokens, torch.tensor([[next_id]], device=tokens.device, dtype=tokens.dtype)],
            dim=1,
        )
        if next_id in eos_ids:
            break
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="HF vs DIY Llama parity check")
    p.add_argument(
        "--model",
        type=Path,
        default=Path.home() / "models" / "Llama-3.2-1B-Instruct",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default=None, choices=["bfloat16", "float16", "float32"])
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--logit-atol", type=float, default=2e-2)
    args = p.parse_args()

    model_dir = args.model.expanduser().resolve()
    config = ModelConfig.from_pretrained(model_dir)
    dtype_name = args.dtype or config.torch_dtype
    dtype = _dtype_from_name(dtype_name)
    device = torch.device(args.device)

    print(f"model_dir={model_dir}")
    print(f"device={device} dtype={dtype_name}")

    tok = AutoTokenizer.from_pretrained(str(model_dir))
    ids = tok.encode(args.prompt, add_special_tokens=True)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    print(f"prompt={args.prompt!r} n_tokens={len(ids)}")

    print("loading HF…")
    hf = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=dtype,
        attn_implementation="eager",
    ).to(device)
    hf.eval()

    print("loading DIY…")
    weights = load_weights(model_dir, config, device=device, dtype=dtype_name)
    ours = LlamaModel(config, weights)

    hf_logits = hf(input_ids).logits
    our_logits = ours.forward(input_ids)

    diff = (hf_logits.float() - our_logits.float()).abs()
    max_diff = float(diff.max().item())
    mean_diff = float(diff.mean().item())
    hf_arg = hf_logits[0].argmax(dim=-1)
    our_arg = our_logits[0].argmax(dim=-1)
    argmax_match = bool(torch.equal(hf_arg, our_arg))

    print(f"logit_max_abs_diff={max_diff:.6g}")
    print(f"logit_mean_abs_diff={mean_diff:.6g}")
    print(f"per_position_argmax_match={argmax_match}")

    if max_diff > args.logit_atol and not argmax_match:
        # Debug ladder: find first position where top-1 disagrees
        disagree = (hf_arg != our_arg).nonzero(as_tuple=False)
        if len(disagree):
            i = int(disagree[0].item())
            print(
                f"first_argmax_mismatch pos={i} "
                f"hf={int(hf_arg[i])} ours={int(our_arg[i])}"
            )
        print("FAIL: logits diverge beyond tolerance without matching argmax")
        return 1

    eos = config.eos_token_id
    if eos is None:
        eos_ids: set[int] = set()
    elif isinstance(eos, int):
        eos_ids = {eos}
    else:
        eos_ids = set(int(x) for x in eos)

    hf_gen = greedy_ids(
        lambda t: hf(t).logits, input_ids, args.max_new_tokens, eos_ids
    )
    our_gen = greedy_ids(
        lambda t: ours.forward(t), input_ids, args.max_new_tokens, eos_ids
    )
    print(f"hf_greedy_ids={hf_gen}")
    print(f"our_greedy_ids={our_gen}")
    print(f"hf_text={tok.decode(hf_gen, skip_special_tokens=False)!r}")
    print(f"our_text={tok.decode(our_gen, skip_special_tokens=False)!r}")

    if hf_gen != our_gen:
        print("FAIL: greedy token ids differ")
        return 1

    print("parity_ok=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
