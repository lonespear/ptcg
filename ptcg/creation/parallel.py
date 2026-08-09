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
    # Screen-tier panel (seeded v3): the slow codex cells are played by the
    # generalist so a greedy screening pass stays cheap; every
    # selection-deciding (real-pilot) eval sees the full specialist panel.
    _STATE["pilots_screen"] = [
        factory(700 + i) if i in _STATE["slow"] else pl
        for i, pl in enumerate(_STATE["pilots"])]
    _STATE["games"] = games
    _STATE["slow_games"] = slow_games
    _STATE["factories"] = factories
    _STATE["cand_pilots"] = {}


def _cand_pilot(kind: str):
    """Candidate-side pilot by kind: 'greedy'/'jon' factories, anything
    else an external specialist name (external/<kind>_agent.py). Broken
    externals fall back to 'jon' — degraded, not dead."""
    st = _STATE
    if kind not in st["cand_pilots"]:
        if kind in st["factories"]:
            st["cand_pilots"][kind] = st["factories"][kind](
                301 + os.getpid() % 97)
        else:
            try:
                from ptcg.creation.pilots import ExternalPilot
                st["cand_pilots"][kind] = ExternalPilot(
                    os.path.join("external", f"{kind}_agent.py"))
            except Exception:  # noqa: BLE001
                st["cand_pilots"][kind] = st["factories"]["jon"](
                    301 + os.getpid() % 97)
    return st["cand_pilots"][kind]


TERM_LAMBDA = 0.15  # D18: deck-out/bench-out losses are a defect, not noise
EXHAUSTION = ("deck-out", "no active Pokemon")


def _eval_deck(deck: list[int]) -> tuple[tuple, tuple]:
    from ptcg.creation.harness import play_match
    st = _STATE
    score, losses, exh = 0.0, 0, 0
    profile = []           # per-panel-entry win rate: the matchup vector
    for i, (entry, pilot) in enumerate(zip(st["panel"], st["pilots"])):
        n = st["slow_games"] if i in st["slow"] else st["games"]
        m = play_match(st["candidate"], pilot, deck, entry["deck"], n)
        wr = m.win_rate(0)
        profile.append(round(wr, 4))
        score += wr * entry["weight"]
        for g in m.games:
            if g.winner == 1:
                losses += 1
                exh += g.reason in EXHAUSTION
    frag = exh / losses if losses else 0.0
    return tuple(sorted(deck)), (score - TERM_LAMBDA * frag, score, frag,
                                 profile)


def _eval_deck_profile(args: tuple) -> tuple[tuple, tuple]:
    """Two-tier worker: matchup profile + exhaustion share for one deck
    under an explicit candidate pilot and games count. Fitness weighting
    happens in the parent — the profile is pilot- and games-dependent but
    island-independent."""
    from ptcg.creation.harness import play_match
    deck, pilot_kind, games = args
    st = _STATE
    cand = _cand_pilot(pilot_kind)
    screen = pilot_kind == "greedy"     # screen tier: cheap panel (above)
    opp_pilots = st["pilots_screen"] if screen else st["pilots"]
    losses, exh = 0, 0
    profile = []
    for i, (entry, pilot) in enumerate(zip(st["panel"], opp_pilots)):
        n = games if screen else (
            max(3, games // 4) if i in st["slow"] else games)
        m = play_match(cand, pilot, deck, entry["deck"], n)
        profile.append(round(m.win_rate(0), 4))
        for g in m.games:
            if g.winner == 1:
                losses += 1
                exh += g.reason in EXHAUSTION
    frag = exh / losses if losses else 0.0
    return (pilot_kind, tuple(sorted(deck))), (profile, frag)


def _eval_deck_reasons(args: tuple) -> list[dict]:
    """Deep-eval worker: per-opponent W/L/draw/capped + termination-reason
    counts for one deck at an explicit games count (quarter rate for the
    slow specialist, as in fitness). A pilot kind may ride along; default
    is the generalist candidate pilot."""
    from ptcg.creation.harness import play_match
    deck, games = args[0], args[1]
    cand = _cand_pilot(args[2]) if len(args) > 2 else _STATE["candidate"]
    st = _STATE
    out = []
    for i, (entry, pilot) in enumerate(zip(st["panel"], st["pilots"])):
        n = max(6, games // 4) if i in st["slow"] else games
        m = play_match(cand, pilot, deck, entry["deck"], n)
        po = {"w": 0, "l": 0, "draw": 0, "capped": 0,
              "win_reasons": {}, "loss_reasons": {}}
        for g in m.games:
            if g.winner == 0:
                po["w"] += 1
                po["win_reasons"][g.reason] = \
                    po["win_reasons"].get(g.reason, 0) + 1
            elif g.winner == 1:
                po["l"] += 1
                po["loss_reasons"][g.reason] = \
                    po["loss_reasons"].get(g.reason, 0) + 1
            elif g.winner == 2:
                po["draw"] += 1
            else:
                po["capped"] += 1
        out.append(po)
    return out


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

    def evaluate_reasons(self, decks: list[list[int]], games: int,
                         pilot_kinds: list[str] | None = None
                         ) -> list[list[dict]]:
        """Deep-eval a batch: per-deck lists of per-opponent tallies with
        termination reasons, order-aligned with `decks`. `pilot_kinds`
        (optional, aligned) selects each deck's candidate pilot."""
        if pilot_kinds is None:
            items = [(d, games) for d in decks]
        else:
            items = [(d, games, k) for d, k in zip(decks, pilot_kinds)]
        return list(self.pool.map(_eval_deck_reasons, items, chunksize=1))

    def evaluate_profiles(self, items: list[tuple[list[int], str, int]]
                          ) -> dict[tuple, tuple]:
        """Two-tier evals: {(pilot_kind, deck_key): (profile, frag)} for
        (deck, pilot_kind, games) items; duplicates collapse."""
        unique = {(k, tuple(sorted(d))): (d, k, g) for d, k, g in items}
        results = self.pool.map(_eval_deck_profile, list(unique.values()),
                                chunksize=1)
        return dict(results)

    def close(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)
