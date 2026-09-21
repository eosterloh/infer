"""Capture one decode step as a CUDA graph and replay it per token.

A decode step on this engine is a few hundred small kernels. Each one costs the
host a handful of microseconds to submit, and on a 1B model that submission time
is larger than the GPU work it describes — the GPU finishes early and waits for
the next launch. A captured graph replays the whole step with one submission, so
the step costs what the memory traffic costs.

What the capture requires, and what the rest of the engine had to provide:
    * fixed addresses — the KV cache writes through ``index_copy_`` at a slot
      held in a device tensor, and the mixer states are updated in place
    * fixed shapes — attention reads a fixed KV window with a validity mask
      instead of a slice that grows every token
    * no host round trips — routing, sampling and the position update all stay
      on the device, so MoE models need the fused dispatch path

Anything that does not hold (a Python expert loop, a mixer that reallocates)
makes capture fail, and the caller falls back to the eager step.
"""

from __future__ import annotations

import os

import torch

BLOCK = 256


def enabled() -> bool:
    """Graphs are opt-in: INFER_CUDA_GRAPH=1, and need a CUDA device."""
    return os.environ.get("INFER_CUDA_GRAPH", "0") == "1" and torch.cuda.is_available()


def _round_up(value: int, block: int = BLOCK) -> int:
    return ((int(value) + block - 1) // block) * block


def _needs_expert_loop(model) -> bool:
    """Whether this model's MoE layers would dispatch experts from Python.

    The reference dispatch asks ``torch.where`` which tokens an expert received,
    and that has to know the answer on the host, so it synchronizes. A capture
    does not survive one: it ends in cudaErrorStreamCaptureInvalidated, thrown by
    ``capture_end`` rather than by the call that caused it. Declining here says
    so where the reason is still legible, instead of leaving a torn-down capture
    and a generic exception for the caller to interpret.
    """
    from engine.kernels import available
    from engine.layers.moe import expert_stack
    from engine.schedule import FfnKind

    weights = getattr(model, "weights", None)
    routed = [
        spec
        for spec in (getattr(model.config, "layers", None) or [])
        if spec.ffn is FfnKind.MOE
    ]
    if not weights or not routed:
        return False
    # Stacked weights are what the fused kernel reads, and the kernel is what
    # keeps the routing on the device; either one missing is the Python loop.
    if not available():
        return True
    return any(expert_stack(weights, f"layers.{spec.index}") is None for spec in routed)


class GraphDecoder:
    """One captured greedy decode step, replayed until the window runs out."""

    @classmethod
    def create(cls, model, cache, *, length: int, budget: int):
        """A decoder for this cache, or None when capture cannot apply."""
        if not enabled() or budget <= 0 or cache is None:
            return None
        kv = getattr(cache, "kv", cache)
        if not hasattr(kv, "enable_graph_mode") or kv.capacity() == 0:
            return None
        if _needs_expert_loop(model):
            return None
        return cls(model, cache, length=length, budget=budget)

    def __init__(self, model, cache, *, length: int, budget: int, block: int = BLOCK):
        self.model = model
        self.cache = cache
        self.kv = getattr(cache, "kv", cache)
        self.device = model.device
        want = _round_up(length + budget + 1, block)
        self.window = min(want, self.kv.max_seq_len or want)
        self.length = int(length)
        self.graph: torch.cuda.CUDAGraph | None = None
        self.ids = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        self.positions = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        self.slot = torch.zeros(1, dtype=torch.long, device=self.device)
        self.kv_mask = torch.zeros((1, self.window), dtype=torch.bool, device=self.device)
        self.logits: torch.Tensor | None = None

    # --- capture ------------------------------------------------------
    def capture(self, token_id: int) -> bool:
        """Warm up, capture, and verify. False means the caller stays eager."""
        if not torch.cuda.is_available() or self.window <= self.length:
            return False
        try:
            self.kv.reserve(self.window)
        except Exception:
            return False

        want = self._eager_reference(token_id)
        if want is None:
            return False

        self._saved = self._save()
        try:
            self._warmup(token_id)
            graph = torch.cuda.CUDAGraph()
            self._arm(token_id)
            with torch.cuda.graph(graph):
                self._step_body()
            self.graph = graph
        except Exception:
            self.graph = None
            self.kv.disable_graph_mode()
            self._restore(self._saved)
            return False

        # A captured step that disagrees with the eager one is worse than no
        # capture at all, so prove the first replay before trusting it.
        self._arm(token_id)
        self.graph.replay()
        got = int(self.ids.item())
        self._restore(self._saved)
        if got != want:
            self.graph = None
            self.kv.disable_graph_mode()
            return False
        self._arm(token_id)
        return True

    def _eager_reference(self, token_id: int) -> int | None:
        """The token the eager path produces, with the cache left untouched."""
        saved = self._save()
        try:
            step = torch.tensor([[token_id]], dtype=torch.long, device=self.device)
            logits = self.model.forward(step, cache=self.cache, logits_to_keep=1)
            return int(logits[0, -1].argmax().item())
        except Exception:
            return None
        finally:
            self._restore(saved)

    def _save(self) -> dict:
        state = {"length": self.cache.seq_len()}
        if hasattr(self.cache, "snapshot"):
            state["hybrid"] = self.cache.snapshot()
        return state

    def _restore(self, state: dict) -> None:
        """Undo whatever a warmup or reference step did to the cache."""
        if "hybrid" in state:
            self.cache.restore(state["hybrid"])
        self.kv.set_length(int(state["length"]))
        if hasattr(self.cache, "set_length"):
            self.cache.set_length(int(state["length"]))

    def _arm(self, token_id: int) -> None:
        """Point every buffer at the step that follows ``self.length``."""
        # A hybrid mixer advances its state every time the step runs, and arming
        # happens once per warmup pass and again before the verifying replay.
        # Without putting the state back, each of those leaves the recurrence a
        # token further along than the eager reference it is checked against.
        saved = getattr(self, "_saved", None)
        if saved is not None and "hybrid" in saved:
            self.cache.restore(saved["hybrid"])
        self.ids.fill_(int(token_id))
        self.positions.fill_(self.length)
        self.slot.fill_(self.length)
        self.kv_mask.zero_()
        self.kv_mask[:, : self.length + 1] = True
        self.kv.enable_graph_mode(self.window, self.slot)
        self.kv.set_length(self.length)
        self.kv.padding_mask = None
        if hasattr(self.cache, "set_length"):
            self.cache.set_length(self.length)

    def _warmup(self, token_id: int) -> None:
        """Run the step off-stream first; capture records an already-warm path."""
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self._arm(token_id)
                self._step_body()
        torch.cuda.current_stream().wait_stream(stream)

    def _step_body(self) -> None:
        """The captured region: one token in, one token out, state advanced."""
        logits = self.model.forward(
            self.ids,
            cache=self.cache,
            position_ids=self.positions,
            kv_mask=self.kv_mask,
            logits_to_keep=1,
        )
        self.logits = logits
        nxt = logits[0, -1].argmax()
        # Everything below stays on the device so a replay needs no host work.
        self.positions.add_(1)
        self.slot.add_(1)
        self.kv_mask.index_fill_(1, self.slot, True)
        self.ids.copy_(nxt.reshape(1, 1))

    # --- replay -------------------------------------------------------
    @property
    def ready(self) -> bool:
        return self.graph is not None

    def room(self) -> bool:
        """Whether another token still fits the captured window."""
        return self.length + 1 < self.window

    def step(self) -> int:
        """Replay one token. Returns the id the step produced."""
        assert self.graph is not None
        self.graph.replay()
        self.length += 1
        return int(self.ids.item())

    def release(self) -> None:
        self.kv.disable_graph_mode()
        if hasattr(self.cache, "set_length"):
            self.cache.set_length(self.length)
        self.graph = None
