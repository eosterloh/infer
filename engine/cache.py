"""Runtime state — KV cache for attention + conv/SSM state for Mamba-2."""

from __future__ import annotations

import torch

from engine.config import ModelConfig
from engine.schedule import MixerKind, build_schedule


class KVCache:
    """Per-layer attention K/V cache (RoPE'd K/V before GQA repeat).

    Layout: k, v : [batch, n_kv_heads, seq, head_dim]
    """

    def __init__(
        self,
        config: ModelConfig,
        *,
        batch_size: int = 1,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.config = config
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.dtype = dtype
        self.n_layers = config.num_hidden_layers
        self.n_kv = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.k: list[torch.Tensor | None] = [None] * self.n_layers
        self.v: list[torch.Tensor | None] = [None] * self.n_layers
        self._seq_len = 0
        self.padding_mask: torch.Tensor | None = None

    def empty(self) -> bool:
        return self._seq_len == 0

    def seq_len(self) -> int:
        return self._seq_len

    def clear(self) -> None:
        self.k = [None] * self.n_layers
        self.v = [None] * self.n_layers
        self._seq_len = 0
        self.padding_mask = None

    def truncate(self, seq_len: int) -> None:
        """Discard cached positions at and after ``seq_len``."""
        seq_len = int(seq_len)
        if seq_len < 0 or seq_len > self._seq_len:
            raise ValueError(f"cannot truncate KV length {self._seq_len} to {seq_len}")
        self.k = [
            tensor[:, :, :seq_len].contiguous() if tensor is not None else None
            for tensor in self.k
        ]
        self.v = [
            tensor[:, :, :seq_len].contiguous() if tensor is not None else None
            for tensor in self.v
        ]
        if self.padding_mask is not None:
            self.padding_mask = self.padding_mask[:, :seq_len].contiguous()
        self._seq_len = seq_len

    def prepare_padding_mask(
        self, mask: torch.Tensor | None, new_tokens: int
    ) -> torch.Tensor | None:
        """Install or extend one key-padding mask before layer KV updates."""
        if mask is None:
            if self.padding_mask is not None:
                ones = torch.ones(
                    self.batch_size,
                    new_tokens,
                    device=self.device,
                    dtype=self.padding_mask.dtype,
                )
                self.padding_mask = torch.cat((self.padding_mask, ones), dim=1)
            return self.padding_mask
        if mask.dim() != 2 or mask.shape[0] != self.batch_size:
            raise ValueError(f"expected attention_mask [B,S], got {tuple(mask.shape)}")
        mask = mask.to(device=self.device)
        total = self._seq_len + new_tokens
        if mask.shape[1] == total:
            self.padding_mask = mask
        elif mask.shape[1] == new_tokens:
            prefix = self.padding_mask
            if prefix is None and self._seq_len:
                prefix = torch.ones(
                    self.batch_size,
                    self._seq_len,
                    device=self.device,
                    dtype=mask.dtype,
                )
            self.padding_mask = (
                torch.cat((prefix, mask), dim=1) if prefix is not None else mask
            )
        else:
            raise ValueError(
                f"attention_mask length {mask.shape[1]} != new {new_tokens} "
                f"or total {total}"
            )
        return self.padding_mask

    def update(
        self,
        layer: int,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if layer < 0 or layer >= self.n_layers:
            raise IndexError(f"layer {layer} out of range 0..{self.n_layers - 1}")
        if k_new.shape[0] != self.batch_size:
            raise ValueError(
                f"batch {k_new.shape[0]} != cache batch_size {self.batch_size}"
            )
        if k_new.dim() != 4 or v_new.dim() != 4:
            raise ValueError(f"expected k/v [B, heads, S, dim], got {k_new.shape} {v_new.shape}")
        if k_new.shape[2] != v_new.shape[2]:
            raise ValueError(f"k/v seq mismatch: {k_new.shape} vs {v_new.shape}")

        if self.k[layer] is None:
            self.k[layer] = k_new
            self.v[layer] = v_new
        else:
            self.k[layer] = torch.cat([self.k[layer], k_new], dim=2)
            self.v[layer] = torch.cat([self.v[layer], v_new], dim=2)

        self._seq_len = int(self.k[layer].shape[2])
        return self.k[layer], self.v[layer]


class RuntimeState:
    """Unified decode state for hybrid models (attention KV + Mamba conv/SSM).

    Agents / generate() pass one object; layers that don't need a slot ignore it.
    """

    def __init__(
        self,
        config: ModelConfig,
        *,
        batch_size: int = 1,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        ssm_dtype: torch.dtype = torch.float32,
    ):
        self.config = config
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.dtype = dtype
        self.ssm_dtype = ssm_dtype
        self.layers = config.layers or build_schedule(config)
        self.n_layers = config.num_hidden_layers

        self.kv = KVCache(
            config, batch_size=batch_size, device=device, dtype=dtype
        )

        # Per-layer Mamba state (None for non-mamba layers)
        self.conv_states: list[torch.Tensor | None] = [None] * self.n_layers
        self.ssm_states: list[torch.Tensor | None] = [None] * self.n_layers
        # Per-layer: True after that mixer has seen at least one token.
        # A global flag is wrong on hybrid models — later Mamba layers would
        # take the decode path during a 1-token prefill.
        self._mamba_ready: list[bool] = [False] * self.n_layers
        self._token_len = 0

        for spec in self.layers:
            if spec.mixer == MixerKind.MAMBA2:
                if config.mamba_num_heads is None or config.mamba_head_dim is None:
                    raise ValueError("mamba dims required for RuntimeState on hybrid model")
                if config.conv_kernel is None or config.ssm_state_size is None:
                    raise ValueError("conv_kernel / ssm_state_size required")
                conv_dim = config.mamba_conv_dim
                k = config.conv_kernel
                n = config.ssm_state_size
                n_heads = config.mamba_num_heads
                hd = config.mamba_head_dim
                i = spec.index
                self.conv_states[i] = torch.zeros(
                    batch_size, conv_dim, k, device=self.device, dtype=dtype
                )
                self.ssm_states[i] = torch.zeros(
                    batch_size, n_heads, hd, n, device=self.device, dtype=ssm_dtype
                )
            elif spec.mixer == MixerKind.GATED_DELTANET:
                if (
                    config.linear_num_key_heads is None
                    or config.linear_num_value_heads is None
                    or config.linear_key_head_dim is None
                    or config.linear_value_head_dim is None
                    or config.linear_conv_kernel_dim is None
                ):
                    raise ValueError("linear_* dims required for Gated DeltaNet cache")
                key_dim = config.linear_num_key_heads * config.linear_key_head_dim
                value_dim = config.linear_num_value_heads * config.linear_value_head_dim
                conv_dim = key_dim * 2 + value_dim
                k = config.linear_conv_kernel_dim
                i = spec.index
                self.conv_states[i] = torch.zeros(
                    batch_size, conv_dim, k, device=self.device, dtype=dtype
                )
                self.ssm_states[i] = torch.zeros(
                    batch_size,
                    config.linear_num_value_heads,
                    config.linear_key_head_dim,
                    config.linear_value_head_dim,
                    device=self.device,
                    dtype=ssm_dtype,
                )

    @property
    def _mamba_has_state(self) -> bool:
        return any(self._mamba_ready)

    def mamba_ready(self, layer: int) -> bool:
        return bool(self._mamba_ready[layer])

    def empty(self) -> bool:
        return self._token_len == 0 and self.kv.empty() and not self._mamba_has_state

    def seq_len(self) -> int:
        return max(self._token_len, self.kv.seq_len())

    @property
    def padding_mask(self) -> torch.Tensor | None:
        return self.kv.padding_mask

    def prepare_padding_mask(
        self, mask: torch.Tensor | None, new_tokens: int
    ) -> torch.Tensor | None:
        return self.kv.prepare_padding_mask(mask, new_tokens)

    def advance(self, n_tokens: int) -> None:
        """Record that `n_tokens` were consumed (prefill sets absolute via replace)."""
        self._token_len += int(n_tokens)

    def clear(self) -> None:
        self.kv.clear()
        self._mamba_ready = [False] * self.n_layers
        self._token_len = 0
        for i, s in enumerate(self.conv_states):
            if s is not None:
                self.conv_states[i] = torch.zeros_like(s)
        for i, s in enumerate(self.ssm_states):
            if s is not None:
                self.ssm_states[i] = torch.zeros_like(s)

    def snapshot(self) -> dict[str, object]:
        """Copy fixed-size hybrid state; KV rollback only needs its sequence length."""
        return {
            "token_len": self._token_len,
            "kv_len": self.kv.seq_len(),
            "ready": list(self._mamba_ready),
            "conv": [s.clone() if s is not None else None for s in self.conv_states],
            "ssm": [s.clone() if s is not None else None for s in self.ssm_states],
        }

    def restore(self, snapshot: dict[str, object]) -> None:
        """Restore a snapshot after speculative tokens are rejected."""
        self.kv.truncate(int(snapshot["kv_len"]))
        self._token_len = int(snapshot["token_len"])
        self._mamba_ready = list(snapshot["ready"])  # type: ignore[arg-type]
        self.conv_states = list(snapshot["conv"])  # type: ignore[arg-type]
        self.ssm_states = list(snapshot["ssm"])  # type: ignore[arg-type]

    # --- attention passthrough ---
    def update(
        self,
        layer: int,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.kv.update(layer, k_new, v_new)

    # --- mamba ---
    def update_conv_prefill(self, layer: int, conv_state: torch.Tensor) -> None:
        """Replace conv state after a prefill (shape [B, conv_dim, kernel])."""
        self.conv_states[layer] = conv_state.to(
            device=self.device, dtype=self.dtype
        )
        self._mamba_ready[layer] = True

    def update_conv_step(self, layer: int, x_t: torch.Tensor) -> torch.Tensor:
        """Roll conv cache and insert new token features. x_t: [B, 1, conv_dim]."""
        state = self.conv_states[layer]
        assert state is not None
        state = state.roll(shifts=-1, dims=-1)
        state[:, :, -1] = x_t[:, 0, :].to(dtype=state.dtype)
        self.conv_states[layer] = state
        self._mamba_ready[layer] = True
        return state

    def update_ssm(self, layer: int, ssm_state: torch.Tensor) -> None:
        self.ssm_states[layer] = ssm_state.to(
            device=self.device, dtype=self.ssm_dtype
        )
        self._mamba_ready[layer] = True
