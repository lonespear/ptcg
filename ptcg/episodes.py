"""Read Kaggle `cabt` episode replays into deck lists and match outcomes.

Each replay is a full 60-card decklist for both players plus who won, so the
public episode dumps are a direct readout of the live metagame. Card ids in the
replay join to `Card ID` in the competition CSV (verified against max HP).

    from ptcg.episodes import parse_episode, iter_episode_files
    ep = parse_episode(path)
"""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

EPISODE_ROOT = Path(
    os.environ.get(
        "PTCG_EPISODE_DIR",
        Path.home() / ".cache" / "kagglehub" / "datasets" / "kaggle",
    )
)


@dataclass
class Episode:
    episode_id: int
    agents: list[str]
    rewards: list[int]
    decks: list[Counter] = field(default_factory=list)
    n_steps: int = 0

    @property
    def winner(self) -> int | None:
        """Index of the winning player, or None for a draw/no-contest.

        A null reward means that agent errored or timed out; the opponent is
        credited with the win only if they finished with a real score.
        """
        if len(self.rewards) != 2:
            return None
        a, b = self.rewards
        if a is None and b is None:
            return None
        if a is None:
            return 1 if b is not None else None
        if b is None:
            return 0
        return None if a == b else (0 if a > b else 1)


def iter_episode_files(day: str | None = None) -> Iterator[Path]:
    """Yield episode JSON paths from the downloaded daily dumps."""
    pattern = (f"pokemon-tcg-ai-battle-episodes-{day}" if day
               else "pokemon-tcg-ai-battle-episodes-*")
    for ds in sorted(EPISODE_ROOT.glob(pattern)):
        if ds.name.endswith("index"):
            continue
        yield from sorted(ds.glob("versions/*/*.json"))


def parse_episode(path: str | Path) -> Episode | None:
    """Parse one replay. Returns None if it is malformed or has no decks.

    Only the head of the file is needed — `info`, `rewards` and `steps[0]` all
    precede the bulk of the replay — but the JSON is parsed in full because the
    files are small enough (~4 MB) that streaming is not worth the fragility.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None

    info = d.get("info") or {}
    steps = d.get("steps") or []
    if not steps:
        return None

    # steps[0][0].visualize[0].action is [deck_p0, deck_p1], each 60 card ids.
    decks: list[Counter] = []
    try:
        vis = steps[0][0]["visualize"][0]["action"]
        if isinstance(vis, list) and len(vis) == 2:
            decks = [Counter(x) for x in vis
                     if isinstance(x, list) and all(isinstance(i, int) for i in x)]
    except (KeyError, IndexError, TypeError):
        decks = []
    if len(decks) != 2:
        return None

    return Episode(
        episode_id=int(info.get("EpisodeId") or d.get("id", 0) or 0),
        agents=list(info.get("TeamNames") or []),
        rewards=list(d.get("rewards") or []),
        decks=decks,
        n_steps=len(steps),
    )
