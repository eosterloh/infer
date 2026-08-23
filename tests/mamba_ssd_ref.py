"""HF Nemotron-H torch_forward SSD scan (prefill), for parity tests.

Copied from modeling_nemotron_h.py torch_forward else-branch — sequential
SSM in engine/layers/mamba2.py must match this within float32 noise.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def pad_tensor_by_size(input_tensor: torch.Tensor, pad_size: int) -> torch.Tensor:
    pad_shape = (
        (0, 0, 0, 0, 0, pad_size, 0, 0)
        if len(input_tensor.shape) == 4
        else (0, 0, 0, pad_size, 0, 0)
    )
    return F.pad(input_tensor, pad_shape, mode="constant", value=0)


def reshape_into_chunks(
    input_tensor: torch.Tensor, pad_size: int, chunk_size: int
) -> torch.Tensor:
    input_tensor = pad_tensor_by_size(input_tensor, pad_size)
    if len(input_tensor.shape) == 3:
        return input_tensor.reshape(
            input_tensor.shape[0], -1, chunk_size, input_tensor.shape[2]
        )
    return input_tensor.reshape(
        input_tensor.shape[0],
        -1,
        chunk_size,
        input_tensor.shape[2],
        input_tensor.shape[3],
    )


def segment_sum(input_tensor: torch.Tensor) -> torch.Tensor:
    chunk_size = input_tensor.size(-1)
    input_tensor = input_tensor[..., None].expand(*input_tensor.size(), chunk_size)
    mask = torch.tril(
        torch.ones(
            chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool
        ),
        diagonal=-1,
    )
    input_tensor = input_tensor.masked_fill(~mask, 0)
    tensor_segsum = torch.cumsum(input_tensor, dim=-2)
    mask = torch.tril(
        torch.ones(
            chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool
        ),
        diagonal=0,
    )
    return tensor_segsum.masked_fill(~mask, -torch.inf)


def ssd_scan(
    x_h: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B_h: torch.Tensor,
    C_h: torch.Tensor,
    D: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """x_h: [B,S,H,D], dt: [B,S,H] (already softplus), A: [H] (negative)."""
    batch_size, seq_len, n_heads, head_dim = x_h.shape
    hidden_states = x_h.float()
    B = B_h.float()
    C = C_h.float()
    pad_size = (chunk_size - seq_len % chunk_size) % chunk_size

    D_residual = D[..., None] * pad_tensor_by_size(hidden_states, pad_size)

    hidden_states = hidden_states * dt[..., None]
    A = A.to(hidden_states.dtype) * dt

    hidden_states, A, B, C = [
        reshape_into_chunks(t, pad_size, chunk_size)
        for t in (hidden_states, A, B, C)
    ]

    A = A.permute(0, 3, 1, 2)
    A_cumsum = torch.cumsum(A, dim=-1)
    L = torch.exp(segment_sum(A))

    G_intermediate = C[:, :, :, None, :, :] * B[:, :, None, :, :, :]
    G = G_intermediate.sum(dim=-1)
    M_intermediate = G[..., None] * L.permute(0, 2, 3, 4, 1)[..., None]
    M = M_intermediate.sum(dim=-1)
    Y_diag = (M[..., None] * hidden_states[:, :, None]).sum(dim=3)

    decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
    B_decay = B * decay_states.permute(0, -2, -1, 1)[..., None]
    states = (B_decay[..., None, :] * hidden_states[..., None]).sum(dim=2)

    previous_states = torch.zeros_like(states[:, :1])
    states = torch.cat([previous_states, states], dim=1)
    decay_chunk = torch.exp(segment_sum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
    decay_chunk = decay_chunk.transpose(1, 3)
    new_states = (decay_chunk[..., None, None] * states[:, :, None, ...]).sum(dim=1)
    states, _ssm_state = new_states[:, :-1], new_states[:, -1]

    state_decay_out = torch.exp(A_cumsum)
    C_times_states = C[..., None, :] * states[:, :, None, ...]
    state_decay_out_permuted = state_decay_out.permute(0, 2, 3, 1)
    Y_off = C_times_states.sum(-1) * state_decay_out_permuted[..., None]

    y = Y_diag + Y_off
    y = y.reshape(batch_size, -1, n_heads, head_dim)
    y = y + D_residual
    if pad_size > 0:
        y = y[:, :seq_len, :, :]
    return y.reshape(batch_size, seq_len, n_heads * head_dim)
