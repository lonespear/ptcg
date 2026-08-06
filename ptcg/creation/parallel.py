"""Parallel fitness evaluation: one engine per worker process.

The cabt engine allows one battle at a time per process, so parallelism
means processes: each worker bootstraps its own engine, builds its own
panel pilots (specialists included), and evaluates whole decks. Deck
evaluations are independent, so speedup is near-linear until the pool
outruns the per-era batch size.
"""

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

_STATE: dict = {}


def _init_worker(root: str, panel_json: str, generalist: str,
                 games: int, slow_games: int) -> None:
    os.chdir(root)
    sys.path.insert(0, root)
    import ptcg.creation  # noqa: F401 — engine bootstrap
    from ptcg.creation.pilots import GreedyPilot, JonDayPilot
    from ptcg.creation.specialist_panel import make_panel_pilots

    factories = {
        "greedy": lambda s: GreedyPilot(seed=s),
        "jon": lambda s: JonDayPilot(seed=s, search=False),
    }
    factory = factories[generalist]
    panel = json.loads(panel_json)
    _STATE["panel"] = panel
    _STATE["pilots"] = make_panel_pilots(panel, factory)
    _STATE["candidate"] = factory(101 + os.getpid() % 97)
    _STATE["slow"] = {i for i, e in enumerate(panel)
                     if e["pilot"].get("specialist") == "codex_alakazam"}
    _STATE["games"] = games
    _STATE["slow_games"] = slow_games


def _eval_deck(deck: list[int]) -> tuple[tuple, float]:
    from ptcg.creation.harness import play_match
    st = _STATE
    score = 0.0
    for i, (entry, pilot) in enumerate(zip(st["panel"], st["pilots"])):
        n = st["slow_games"] if i in st["slow"] else st["games"]
        m = play_match(st["candidate"], pilot, deck, entry["deck"], n)
        score += m.win_rate(0) * entry["weight"]
    return tuple(sorted(deck)), score


class ParallelFitness:
    """Evaluate batches of decks across a process pool."""

    def __init__(self, panel: list[dict], generalist: str,
                 games_per_opponent: int, workers: int, root: str):
        slow_games = max(6, games_per_opponent // 4)
        self.pool = ProcessPoolExecutor(
            max_workers=workers, mp_context=get_context("spawn"),
            initializer=_init_worker,
            initargs=(root, json.dumps(panel), generalist,
                      games_per_opponent, slow_games))

    def evaluate_many(self, decks: list[list[int]]) -> dict[tuple, float]:
        unique = {tuple(sorted(d)): d for d in decks}
        results = self.pool.map(_eval_deck, list(unique.values()),
                                chunksize=1)
        return dict(results)

    def close(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)
