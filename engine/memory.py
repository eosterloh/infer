"""How much room is left, and when to stop asking for more.

The Spark shares one 128 GB pool between CPU and GPU. A load that overruns it
does not fail the way an allocation on a discrete GPU fails: there is no device
limit to hit, so the kernel starts reclaiming, then swapping, and the machine
stops answering SSH — which costs a power cycle, not a stack trace. So the load
path watches the free pool at every stage that can grow it and gives up while a
clean failure is still possible.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_FLOOR_GB = 6.0


def available_bytes() -> int | None:
    """Memory that can be handed out now, or None when it cannot be read.

    Only ``MemAvailable`` counts: it is the kernel's own estimate of what is
    allocatable without swapping. Free pages alone — all a Mac will tell us —
    hover near zero on a healthy machine because the page cache holds the rest,
    so treating that number as a budget would refuse every load. Nowhere but
    Linux gets a guard, which is fine: the pool this protects is the Spark's.
    """
    meminfo = Path("/proc/meminfo")
    if not meminfo.is_file():
        return None
    try:
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def total_bytes() -> int | None:
    try:
        return int(os.sysconf("SC_PHYS_PAGES")) * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return None


def floor_bytes() -> float:
    """The reserve we refuse to spend. 0 disables the guard."""
    try:
        gb = float(os.environ.get("INFER_MEM_FLOOR_GB", DEFAULT_FLOOR_GB))
    except ValueError:
        gb = DEFAULT_FLOOR_GB
    return gb * 1e9


def _release_cached_blocks() -> None:
    """Hand torch's unused device blocks back to the kernel.

    ``del`` on a CUDA tensor returns its block to torch's caching allocator, not
    to the system. On a discrete GPU that is invisible and free. Here the device
    pool *is* the host pool, so a cached block is a page MemAvailable no longer
    counts: stacking a 30B checkpoint's experts frees one layer at a time and
    still walked the pool down to 5.7 GB, because every freed expert stayed in
    the reserve. Giving them back costs a synchronize, which is nothing next to
    the load it unblocks.
    """
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def check_floor(stage: str) -> None:
    """Raise while the machine is still responsive enough to raise.

    Called between stages of a load — after a shard, after an expert stack —
    because those are the points where the next step is about to allocate a
    large, known-bad amount on top of what is already resident.
    """
    floor = floor_bytes()
    if floor <= 0:
        return
    free = available_bytes()
    if free is None or free >= floor:
        return
    # Reclaim before refusing: the shortage may be entirely torch's own reserve.
    _release_cached_blocks()
    free = available_bytes()
    if free is None or free >= floor:
        return
    raise MemoryError(
        f"only {free / 1e9:.1f} GB left after {stage}; stopping before the pool "
        f"is exhausted (floor {floor / 1e9:.1f} GB, INFER_MEM_FLOOR_GB=0 to disable)"
    )
