"""Run agents against each other on the local engine.

The engine keeps a single global battle pointer, so only one battle can be in
flight per process. Parallelism comes from running several processes, not
several battles.

Agents use the same signature the competition expects:

    agent(obs_dict: dict) -> list[int]

so whatever wins here is what gets submitted, unchanged.
"""

from __future__ import annotations

import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ENGINE_DIR = Path(os.environ.get(
    "PTCG_ENGINE_DIR", Path(__file__).resolve().parents[1] / "engine"))
if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))

Agent = Callable[[dict], list[int]]

MAX_STEPS = 20_000


@dataclass
class GameResult:
    winner: int | None
    turns: int
    steps: int
    error: str | None = None


def load_deck(path: str | Path) -> list[int]:
    """Read a 60-line deck.csv of card ids."""
    text = Path(path).read_text()
    deck = [int(x) for x in text.replace(",", "\n").split() if x.strip()]
    if len(deck) != 60:
        raise ValueError(f"{path}: expected 60 cards, got {len(deck)}")
    return deck


def _reset_engine_agent() -> None:
    """Give this game its own engine-side search agent.

    `cg.api` allocates one `AgentStart()` handle the first time anything
    searches and then holds it in a module global for the life of the process,
    so every game a worker plays searches through state the game before it
    left behind. That is the second half of D60's worker-count defect: which
    games shared a process was set by `--workers`, so the same seed block
    scored differently at 2 workers and at 4. Measured: replaying one seed in
    a process gave three different games, and gives the same game every time
    with this reset in front of it.

    The handle costs ~25 KB and a few microseconds, and the engine exports no
    matching free, so a 600-game worker carries ~15 MB of dead handles. That
    is the price of a game that does not depend on its neighbours.
    """
    try:
        import cg.api as _capi
        if hasattr(_capi, "agent_ptr"):
            del _capi.agent_ptr
    except Exception:
        pass


def play_game(agent0: Agent, agent1: Agent, deck0: list[int], deck1: list[int],
              seed: int | None = None) -> GameResult:
    """Play one full battle and report who won.

    `seed` reaches the engine only when the `tools/engine_seed` preload is in
    the process. Without it this line seeds Python's global `random`, which
    the engine does not use — `libcg` draws from `std::random_device` and
    exports no seed entry point — so every "seeded" game was in fact a fresh
    deal and no run of any gate ever reproduced another (D66). Pinning per
    GAME rather than per process is also what makes a result independent of
    `--workers`: the engine's stream would otherwise carry from one game to
    the next inside a worker, so which games shared a process changed what
    was dealt (D60).
    """
    from cg.game import battle_finish, battle_select, battle_start
    from ptcg import engine_seed

    _reset_engine_agent()
    if seed is not None:
        random.seed(seed)
        engine_seed.pin(seed)

    agents = (agent0, agent1)
    obs, start = battle_start(list(deck0), list(deck1))
    if obs is None:
        return GameResult(None, 0, 0,
                          f"start failed (player {start.errorPlayer}, "
                          f"type {start.errorType})")

    steps = 0
    try:
        while steps < MAX_STEPS:
            cur = obs.get("current") or {}
            sel = obs.get("select")
            if sel is None or not sel.get("option"):
                break
            who = cur.get("yourIndex", 0)
            try:
                choice = agents[who](obs)
            except Exception as e:  # a crashing agent forfeits, as on Kaggle
                return GameResult(1 - who, cur.get("turn", 0), steps,
                                  f"agent {who} raised {type(e).__name__}: {e}")
            # An empty selection is the engine-sanctioned answer whenever the
            # prompt's minCount is 0 ("done benching", declining a TO_HAND or
            # an ATTACH_TO), and the competition's own reference agent returns
            # it. Treating [] as a forfeit scored 7,726 legal moves across 97
            # result files as opponent crashes, and our agent cannot emit []
            # by construction, so the penalty fell only on opponents: it read
            # as a 24.9% opponent forfeit rate against our 0.0%, and inflated
            # archaludon from a true 0.32 to 0.63.
            if not isinstance(choice, list):
                return GameResult(1 - who, cur.get("turn", 0), steps,
                                  f"agent {who} returned {choice!r}")
            if not choice and (sel.get("minCount") or 0) > 0:
                return GameResult(1 - who, cur.get("turn", 0), steps,
                                  f"agent {who} returned {choice!r} "
                                  f"(minCount={sel.get('minCount')})")
            try:
                obs = battle_select(choice)
            except (ValueError, IndexError) as e:
                return GameResult(1 - who, cur.get("turn", 0), steps,
                                  f"agent {who} illegal move: {type(e).__name__}")
            steps += 1

        cur = obs.get("current") or {}
        result = cur.get("result")
        winner = result if result in (0, 1) else None
        return GameResult(winner, cur.get("turn", 0), steps)
    finally:
        battle_finish()


def match(agent0: Agent, agent1: Agent, deck0: list[int], deck1: list[int],
          games: int = 100, seed0: int = 0, swap_sides: bool = True) -> dict:
    """Play a match, alternating who moves first to cancel the first-player edge."""
    wins = [0, 0]
    draws = 0
    errors: list[str] = []
    turns: list[int] = []

    for g in range(games):
        flip = swap_sides and (g % 2 == 1)
        a0, a1 = (agent1, agent0) if flip else (agent0, agent1)
        d0, d1 = (deck1, deck0) if flip else (deck0, deck1)
        r = play_game(a0, a1, d0, d1, seed=seed0 + g)
        if r.error:
            errors.append(r.error)
        if r.winner is None:
            draws += 1
        else:
            # Map the seat back to the agent that sat in it.
            actual = (1 - r.winner) if flip else r.winner
            wins[actual] += 1
            turns.append(r.turns)

    played = wins[0] + wins[1]
    return {
        "games": games, "decided": played, "draws": draws,
        "agent0_wins": wins[0], "agent1_wins": wins[1],
        "agent0_win_rate": wins[0] / played if played else float("nan"),
        "median_turns": sorted(turns)[len(turns) // 2] if turns else None,
        "errors": errors[:5], "n_errors": len(errors),
    }


def match_sprt(agent0: Agent, agent1: Agent, deck0: list[int], deck1: list[int],
               p0: float = 0.50, p1: float = 0.55, max_games: int = 4000,
               seed0: int = 0, report_every: int = 100) -> dict:
    """Play until the evidence decides, instead of a fixed game count.

    Stops as soon as Wald's test accepts "A is better" or "A is not better",
    which is both faster than a fixed 500 games when the effect is large and
    more trustworthy when it is small.
    """
    from ptcg.sprt import SPRT

    test = SPRT(p0=p0, p1=p1)
    draws = 0
    errors: list[str] = []
    turns: list[int] = []

    for g in range(max_games):
        flip = g % 2 == 1
        a0, a1 = (agent1, agent0) if flip else (agent0, agent1)
        d0, d1 = (deck1, deck0) if flip else (deck0, deck1)
        r = play_game(a0, a1, d0, d1, seed=seed0 + g)
        if r.error:
            errors.append(r.error)
        if r.winner is None:
            draws += 1
            continue
        actual = (1 - r.winner) if flip else r.winner
        turns.append(r.turns)
        res = test.update(actual == 0)
        if report_every and (g + 1) % report_every == 0:
            print(f"    {test.describe()}")
        if res.decision != "continue":
            break

    res = test.result()
    return {
        "decision": res.decision, "llr": res.llr,
        "games": res.n, "draws": draws,
        "agent0_wins": res.wins, "agent1_wins": res.losses,
        "agent0_win_rate": res.win_rate,
        "median_turns": sorted(turns)[len(turns) // 2] if turns else None,
        "errors": errors[:5], "n_errors": len(errors),
    }


def random_agent(obs: dict) -> list[int]:
    """Uniformly random legal choice — the baseline every agent must beat."""
    sel = obs.get("select")
    if sel is None:
        raise RuntimeError("random_agent called with no select; deck goes first")
    n = len(sel["option"])
    k = max(1, min(sel.get("maxCount", 1), n))
    return random.sample(range(n), k)
