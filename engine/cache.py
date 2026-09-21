"""Runtime state — KV cache for attention + conv/SSM state for Mamba-2."""

from __future__ import annotations

import torch

from engine.config import ModelConfig
from engine.schedule import MixerKind, build_schedule


CACHE_BLOCK = 256


def _round_up(value: int, block: int = CACHE_BLOCK) -> int:
    return ((value + block - 1) // block) * block


class KVCache:
    """Per-layer attention K/V cache (RoPE'd K/V before GQA repeat).

    Layout: k, v : [batch, n_kv_heads, capacity, head_dim]

    The buffers are allocated once and written in place. Growing the sequence
    costs one slice assignment, not a reallocate-and-copy of the whole cache,
    and a fixed ``max_seq_len`` keeps the pointers stable enough to capture a
    decode step in a CUDA graph.
    """

    def __init__(
        self,
        config: ModelConfig,
        *,
        batch_size: int = 1,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        max_seq_len: int | None = None,
    ):
        self.config = config
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.dtype = dtype
        self.n_layers = config.num_hidden_layers
        self.n_kv = config.num_key_value_heads
        self.head_dim = config.head_dim
        # Buffers are [B, heads, capacity, dim]; `k`/`v` stay the public views.
        self._k_buf: list[torch.Tensor | None] = [None] * self.n_layers
        self._v_buf: list[torch.Tensor | None] = [None] * self.n_layers
        self._layer_len: list[int] = [0] * self.n_layers
        self._capacity = 0
        self.max_seq_len = int(max_seq_len) if max_seq_len else None
        self._seq_len = 0
        self.padding_mask: torch.Tensor | None = None
        # Graph mode: writes go to a slot held on the GPU and reads expose a
        # fixed window, so a decode step has static shapes and can be captured.
        self._graph_window: int | None = None
        self._graph_slot: torch.Tensor | None = None

    # --- buffer views -------------------------------------------------
    @property
    def k(self) -> list[torch.Tensor | None]:
        return [
            buf[:, :, : self._layer_len[i]] if buf is not None else None
            for i, buf in enumerate(self._k_buf)
        ]

    @property
    def v(self) -> list[torch.Tensor | None]:
        return [
            buf[:, :, : self._layer_len[i]] if buf is not None else None
            for i, buf in enumerate(self._v_buf)
        ]

    def capacity(self) -> int:
        return self._capacity

    def empty(self) -> bool:
        return self._seq_len == 0

    def seq_len(self) -> int:
        return self._seq_len

    def clear(self) -> None:
        self._layer_len = [0] * self.n_layers
        self._seq_len = 0
        self.padding_mask = None

    def reset_buffers(self) -> None:
        """Drop the allocations as well as the lengths."""
        self._k_buf = [None] * self.n_layers
        self._v_buf = [None] * self.n_layers
        self._capacity = 0
        self.clear()

    def truncate(self, seq_len: int) -> None:
        """Discard cached positions at and after ``seq_len``."""
        seq_len = int(seq_len)
        if self._seq_len == 0:
            return
        if seq_len < 0 or seq_len > self._seq_len:
            raise ValueError(f"cannot truncate KV length {self._seq_len} to {seq_len}")
        # Rolling back is just a length change; the stale tail is never read.
        self._layer_len = [min(length, seq_len) for length in self._layer_len]
        if self.padding_mask is not None:
            self.padding_mask = self.padding_mask[:, :seq_len].contiguous()
        self._seq_len = seq_len

    def _grow(self, needed: int) -> None:
        if needed <= self._capacity:
            return
        if self.max_seq_len is not None:
            if needed > self.max_seq_len:
                raise ValueError(
                    f"KV cache needs {needed} positions > max_seq_len {self.max_seq_len}"
                )
            target = self.max_seq_len
        else:
            target = max(_round_up(needed), self._capacity * 2)
        for i, buf in enumerate(self._k_buf):
            if buf is None:
                continue
            for store in (self._k_buf, self._v_buf):
                old = store[i]
                assert old is not None
                shape = list(old.shape)
                shape[2] = target
                new = torch.zeros(shape, device=old.device, dtype=old.dtype)
                length = self._layer_len[i]
                if length:
                    new[:, :, :length] = old[:, :, :length]
                store[i] = new
        self._capacity = target

    def _allocate_layer(
        self, layer: int, k_new: torch.Tensor, v_new: torch.Tensor, needed: int
    ) -> None:
        target = (
            self.max_seq_len
            if self.max_seq_len is not None
            else max(_round_up(needed), self._capacity)
        )
        if self.max_seq_len is not None and needed > self.max_seq_len:
            raise ValueError(
                f"KV cache needs {needed} positions > max_seq_len {self.max_seq_len}"
            )
        # MLA keys and values have different head dims, so size each separately.
        # Zeroed, not empty: graph-mode attention reads a fixed window that can
        # run past the live length, and masked-out NaN garbage would poison the
        # softmax even though its weight is zero.
        for store, ref in ((self._k_buf, k_new), (self._v_buf, v_new)):
            store[layer] = torch.zeros(
                (ref.shape[0], ref.shape[1], target, ref.shape[3]),
                device=ref.device,
                dtype=ref.dtype,
            )
        self._capacity = max(self._capacity, target)

    # --- CUDA graph mode ----------------------------------------------
    def enable_graph_mode(self, window: int, slot: torch.Tensor) -> None:
        """Freeze shapes for capture: write at ``slot``, read ``[:window]``.

        ``slot`` is a device tensor so the write index can change between
        replays without re-capturing, and ``window`` is a fixed power-of-block
        length so every tensor in the step keeps its shape.
        """
        window = int(window)
        if window > self._capacity:
            raise ValueError(f"graph window {window} > capacity {self._capacity}")
        if slot.device != self.device or slot.dtype != torch.int64:
            raise ValueError("graph slot must be an int64 tensor on the cache device")
        self._graph_window = window
        self._graph_slot = slot

    def disable_graph_mode(self) -> None:
        self._graph_window = None
        self._graph_slot = None

    @property
    def graph_mode(self) -> bool:
        return self._graph_window is not None

    def set_length(self, seq_len: int) -> None:
        """Set the live length directly (graph replays bypass ``update``)."""
        seq_len = int(seq_len)
        if seq_len > self._capacity:
            raise ValueError(f"length {seq_len} > capacity {self._capacity}")
        self._layer_len = [seq_len] * self.n_layers
        self._seq_len = seq_len

    def prepare_padding_mask(
        self, mask: torch.Tensor | None, new_tokens: int
    ) -> torch.Tensor | None:
        """Install or extend one key-padding mask before layer KV updates."""
        if self._graph_window is not None:
            # The runner owns the mask buffer; growing it here would allocate.
            return self.padding_mask
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

        if self._graph_window is not None:
            k_buf = self._k_buf[layer]
            v_buf = self._v_buf[layer]
            if k_buf is None or v_buf is None:
                raise RuntimeError("graph mode requires a warmed cache")
            slot = self._graph_slot
            assert slot is not None
            # index_copy_ keeps the write position in a device tensor, so the
            # captured graph can target a different row on every replay.
            k_buf.index_copy_(2, slot, k_new.to(k_buf.dtype))
            v_buf.index_copy_(2, slot, v_new.to(v_buf.dtype))
            window = self._graph_window
            return k_buf[:, :, :window], v_buf[:, :, :window]

        base = self._layer_len[layer]
        end = base + int(k_new.shape[2])
        if self._k_buf[layer] is None:
            self._allocate_layer(layer, k_new, v_new, end)
        elif end > self._capacity:
            self._grow(end)
        k_buf = self._k_buf[layer]
        v_buf = self._v_buf[layer]
        assert k_buf is not None and v_buf is not None
        k_buf[:, :, base:end] = k_new
        v_buf[:, :, base:end] = v_new
        self._layer_len[layer] = end
        self._seq_len = end
        return k_buf[:, :, :end], v_buf[:, :, :end]

    def begin_speculative(self) -> None:
        if getattr(self, "_spec_len", None) is not None:
            raise RuntimeError("speculative transaction already active")
        self._spec_len = self._seq_len

    def commit_speculative(self, accepted_tokens: int) -> None:
        if getattr(self, "_spec_len", None) is None:
            raise RuntimeError("no speculative transaction active")
        accepted_tokens = int(accepted_tokens)
        if accepted_tokens < 1:
            raise ValueError("verification must commit at least its target seed")
        self.truncate(int(self._spec_len) + accepted_tokens)
        self._spec_len = None

    def finish_speculative(self) -> None:
        if getattr(self, "_spec_len", None) is None:
            raise RuntimeError("no speculative transaction active")
        self._spec_len = None

    def cancel_speculative(self) -> None:
        spec_len = getattr(self, "_spec_len", None)
        if spec_len is None:
            return
        self.truncate(int(spec_len))
        self._spec_len = None


def _clone_states(states) -> list[torch.Tensor | None]:
    """Detach a list of mixer state buffers from whoever else holds them."""
    return [s.clone() if s is not None else None for s in states]


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
        self._spec_base: dict[str, object] | None = None
        self._spec_gdn: dict[int, tuple[list[torch.Tensor], torch.Tensor]] = {}

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
            elif spec.mixer == MixerKind.MAMBA1:
                raw = config.raw or {}
                expand = int(raw.get("mamba_expand") or 2)
                inter = expand * config.hidden_size
                k = int(raw.get("mamba_d_conv") or config.conv_kernel or 4)
                n = int(raw.get("mamba_d_state") or config.ssm_state_size or 16)
                i = spec.index
                self.conv_states[i] = torch.zeros(
                    batch_size, inter, k, device=self.device, dtype=dtype
                )
                self.ssm_states[i] = torch.zeros(
                    batch_size, inter, n, device=self.device, dtype=ssm_dtype
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

    def enable_graph_mode(self, window: int, slot: torch.Tensor) -> None:
        self.kv.enable_graph_mode(window, slot)

    def disable_graph_mode(self) -> None:
        self.kv.disable_graph_mode()

    @property
    def graph_mode(self) -> bool:
        return self.kv.graph_mode

    def set_length(self, seq_len: int) -> None:
        if self.kv.capacity():
            self.kv.set_length(seq_len)
        self._token_len = int(seq_len)

    def advance(self, n_tokens: int) -> None:
        """Record that `n_tokens` were consumed (prefill sets absolute via replace)."""
        self._token_len += int(n_tokens)

    def truncate(self, seq_len: int) -> None:
        """Discard cached positions at and after ``seq_len``."""
        seq_len = int(seq_len)
        if self.kv.seq_len() > 0:
            self.kv.truncate(seq_len)
        self._token_len = seq_len

    @property
    def is_speculating(self) -> bool:
        return self._spec_base is not None

    def begin_speculative(self) -> None:
        """Begin a target verification transaction without cloning fixed state."""
        if self._spec_base is not None:
            raise RuntimeError("speculative transaction already active")
        if any(spec.mixer == MixerKind.MAMBA2 for spec in self.layers):
            raise NotImplementedError(
                "transactional speculative commit is implemented for Gated DeltaNet"
            )
        # Cloned, not aliased: the mixers write their state buffers in place, so
        # a reference would be overwritten by the very tokens this transaction
        # may have to roll back.
        self._spec_base = {
            "token_len": self._token_len,
            "kv_len": self.kv.seq_len(),
            "ready": list(self._mamba_ready),
            "conv": _clone_states(self.conv_states),
            "ssm": _clone_states(self.ssm_states),
        }
        self._spec_gdn = {}

    def record_gdn_speculation(
        self,
        layer: int,
        recurrent_states: list[torch.Tensor],
        mixed_inputs: torch.Tensor,
    ) -> None:
        if self._spec_base is None:
            return
        self._spec_gdn[layer] = (recurrent_states, mixed_inputs)

    def commit_speculative(self, accepted_tokens: int) -> None:
        """Commit one accepted verification prefix across KV, conv, and SSM."""
        if self._spec_base is None:
            raise RuntimeError("no speculative transaction active")
        accepted_tokens = int(accepted_tokens)
        if accepted_tokens < 1:
            raise ValueError("verification must commit at least its target seed")
        base = self._spec_base
        kv_target = int(base["kv_len"]) + accepted_tokens
        if self.kv.seq_len() > 0:
            self.kv.truncate(kv_target)
        self._token_len = int(base["token_len"]) + accepted_tokens
        base_conv = list(base["conv"])  # type: ignore[arg-type]
        for layer, (states, mixed) in self._spec_gdn.items():
            if accepted_tokens > len(states):
                raise ValueError(
                    f"accepted {accepted_tokens} exceeds GDN trajectory {len(states)}"
                )
            self.ssm_states[layer] = states[accepted_tokens - 1]
            previous = base_conv[layer]
            assert previous is not None
            joined = torch.cat(
                (previous, mixed[:, :accepted_tokens].transpose(1, 2)), dim=-1
            )
            self.conv_states[layer] = joined[
                :, :, -previous.shape[-1] :
            ].contiguous()
            self._mamba_ready[layer] = True
        self._spec_base = None
        self._spec_gdn = {}

    def finish_speculative(self) -> None:
        """Keep the full verified sequence and release transaction metadata."""
        if self._spec_base is None:
            raise RuntimeError("no speculative transaction active")
        self._spec_base = None
        self._spec_gdn = {}

    def cancel_speculative(self) -> None:
        """Restore state references if verification raises before commit."""
        if self._spec_base is None:
            return
        base = self._spec_base
        self.kv.truncate(int(base["kv_len"]))
        self._token_len = int(base["token_len"])
        self._mamba_ready = list(base["ready"])  # type: ignore[arg-type]
        self.conv_states = _clone_states(base["conv"])  # type: ignore[arg-type]
        self.ssm_states = _clone_states(base["ssm"])  # type: ignore[arg-type]
        self._spec_base = None
        self._spec_gdn = {}

    def clear(self) -> None:
        self.kv.clear()
        self._mamba_ready = [False] * self.n_layers
        self._token_len = 0
        self._spec_base = None
        self._spec_gdn = {}
        for s in self.conv_states:
            if s is not None:
                s.zero_()
        for s in self.ssm_states:
            if s is not None:
                s.zero_()

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
        """Restore a snapshot after speculative tokens are rejected.

        Copies out of the snapshot: mixer state is now updated in place, so
        adopting the snapshot's tensors directly would let the next token
        overwrite the snapshot and make a second restore read post-rollback
        state.
        """
        self.kv.truncate(int(snapshot["kv_len"]))
        self._token_len = int(snapshot["token_len"])
        self._mamba_ready = list(snapshot["ready"])  # type: ignore[arg-type]
        self.conv_states = _clone_states(snapshot["conv"])  # type: ignore[arg-type]
        self.ssm_states = _clone_states(snapshot["ssm"])  # type: ignore[arg-type]

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
        self._write_state(self.conv_states, layer, conv_state, self.dtype)
        self._mamba_ready[layer] = True

    def update_conv_step(self, layer: int, x_t: torch.Tensor) -> torch.Tensor:
        """Roll conv cache and insert new token features. x_t: [B, 1, conv_dim]."""
        state = self.conv_states[layer]
        assert state is not None
        # In place: the buffer address has to survive a CUDA graph capture, so
        # the rolled copy is written back instead of rebinding the slot.
        state.copy_(state.roll(shifts=-1, dims=-1))
        state[:, :, -1] = x_t[:, 0, :].to(dtype=state.dtype)
        self._mamba_ready[layer] = True
        return state

    def update_ssm(self, layer: int, ssm_state: torch.Tensor) -> None:
        self._write_state(self.ssm_states, layer, ssm_state, self.ssm_dtype)
        self._mamba_ready[layer] = True

    def mark_mamba_ready(self, layer: int) -> None:
        """Record that a kernel updated this layer's state buffer in place."""
        self._mamba_ready[layer] = True

    def _write_state(
        self,
        store: list[torch.Tensor | None],
        layer: int,
        value: torch.Tensor,
        dtype: torch.dtype,
    ) -> None:
        current = store[layer]
        if current is not None and current.shape == value.shape:
            current.copy_(value)
        else:
            store[layer] = value.to(device=self.device, dtype=dtype)
