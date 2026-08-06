"""Deck creation (chunk 1 of 3): archipelago GA over decks.

Depends on the licensed battle engine, which is never committed: copy the
competition `sample_submission/` into `engine/` (see README). This package
finds the engine's `cg` wrapper there and puts it on sys.path.

Depends on deck utilization (chunk 2) only through the pilot interface:
any callable(obs_dict) -> list[int]. A baseline greedy pilot ships in
`pilots.py` so creation runs stand-alone.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def _ensure_engine() -> None:
    try:
        import cg  # noqa: F401
        return
    except ImportError:
        pass
    for cand in (_ROOT / "engine",
                 _ROOT / "engine" / "sample_submission",
                 _ROOT / "engine" / "sample_submission" / "sample_submission"):
        if (cand / "cg" / "__init__.py").exists():
            sys.path.insert(0, str(cand))
            return
    raise ImportError(
        "PTCG engine not found. Copy the competition sample_submission/ "
        "into engine/ (it is licensed competition-use-only and gitignored)."
    )


_ensure_engine()
