"""Reach the engine's random number generator from Python.

`tools/engine_seed` preloads a replacement for `std::random_device`, which is
where `libcg` gets its entropy. This module is the handle on it: `pin(seed)`
resets the stream so the next battle deals the same cards it dealt last time,
and `available()` says whether the preload is actually in the process — the
answer is False for anything launched without `DYLD_INSERT_LIBRARIES`
(`LD_PRELOAD` on Linux), and a harness that silently ran unpinned would report
reproducibility it does not have.

`scripts/run_pinned.sh` sets the environment; `ptcg.arena.play_game` calls
`pin` when the preload is present.
"""

from __future__ import annotations

import ctypes
import os

_LIB = None
_CHECKED = False


def _lib():
    global _LIB, _CHECKED
    if _CHECKED:
        return _LIB
    _CHECKED = True
    # The preload is already mapped into this process; a null-path handle
    # searches every image in it, so there is no path to get wrong here.
    #
    # Except on Windows, where there is no such thing: `CDLL(None)` raises
    # TypeError rather than returning the main image, and the preload mechanism
    # this depends on (DYLD_INSERT_LIBRARIES / LD_PRELOAD) does not exist at
    # all. Bail before touching ctypes — otherwise every seeded play_game()
    # dies, which is what happened to Jon's Windows box the moment this merged.
    if os.name == "nt":
        return None
    try:
        h = ctypes.CDLL(None)
        h.cabt_engine_seed_present.restype = ctypes.c_int
        if h.cabt_engine_seed_present() != 1:
            return None
        h.cabt_engine_seed.argtypes = [ctypes.c_ulonglong]
        h.cabt_engine_seed.restype = None
        h.cabt_engine_state.restype = ctypes.c_ulonglong
        h.cabt_engine_draws.restype = ctypes.c_ulonglong
        _LIB = h
    # Broad by design: this is a capability probe for an optional local-gating
    # feature. Any failure means "not pinned", and must never take the harness
    # down with it.
    except Exception:
        _LIB = None
    return _LIB


def available() -> bool:
    """Is the engine RNG under our control in this process?"""
    return _lib() is not None


def pin(seed: int) -> bool:
    """Reset the engine RNG to `seed`. False means the preload is absent."""
    h = _lib()
    if h is None:
        return False
    h.cabt_engine_seed(ctypes.c_ulonglong(int(seed) & 0xFFFFFFFFFFFFFFFF))
    return True


def state() -> int | None:
    h = _lib()
    return None if h is None else int(h.cabt_engine_state())


def draws() -> int | None:
    """How many numbers the engine has drawn. The instrument that found the
    `agent_ptr` leak: a game replayed in one process consumed 1,394 draws,
    then 1,901, then 1,242, which is what said the second run of a game was
    not the same game."""
    h = _lib()
    return None if h is None else int(h.cabt_engine_draws())


def require() -> None:
    """Refuse to measure on an unpinned engine."""
    if not available():
        raise RuntimeError(
            "engine RNG is not pinned: run under scripts/run_pinned.sh, or "
            "build the preload with tools/engine_seed/build.sh. Unpinned, "
            "`seed` controls nothing and two runs of one file are two "
            "different experiments (D66).")


def env_ready() -> bool:
    """Did the caller at least ask for the preload?"""
    return bool(os.environ.get("DYLD_INSERT_LIBRARIES")
                or os.environ.get("LD_PRELOAD"))
