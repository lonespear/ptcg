"""How do our games end, versus how real games end?

The distributional check flags our self-play games at ~7 turns against a real
median of 13, and that is measured *after* the shuffle fix — so it is not the
Energy-hand bug. A simulated world that terminates at a tempo the real one never
does invalidates everything measured inside it, so the termination mode gets
named before any more search work is queued behind it.

Pokémon TCG ends three ways: someone takes all their prizes, someone has no
Pokémon left to promote, or someone cannot draw. Each points at a different
suspect.

    python scripts/termination_modes.py --games 60 --episodes 200
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "engine"))

from ptcg.arena import load_deck  # noqa: E402
from ptcg.episodes import iter_episode_files  # noqa: E402


def load_agent(path: Path, name: str):
    cwd = os.getcwd()
    os.chdir(path.parent)
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod.agent
    finally:
        os.chdir(cwd)


def classify(cur: dict) -> str:
    """Name the ending from the final board state."""
    players = cur.get("players") or []
    if len(players) < 2:
        return "unknown"
    result = cur.get("result")
    modes = []
    for i, p in enumerate(players):
        prize_left = len([c for c in (p.get("prize") or [])])
        board = (p.get("active") or []) + (p.get("bench") or [])
        alive = [m for m in board if m]
        if prize_left == 0:
            modes.append(f"p{i}_took_all_prizes")
        if not alive:
            modes.append(f"p{i}_no_pokemon")
        if (p.get("deckCount") or 0) == 0:
            modes.append(f"p{i}_decked_out")
    if not modes:
        return "other/unresolved"
    # Report it from the loser's perspective, which is what actually ended it.
    loser = 1 - result if result in (0, 1) else None
    for m in modes:
        if loser is not None and m.startswith(f"p{loser}_") and "prizes" not in m:
            return m.split("_", 1)[1]
    for m in modes:
        if "prizes" in m:
            return "prizes_taken"
    return modes[0].split("_", 1)[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="agent/main.py")
    ap.add_argument("--deck", default="agent/deck.csv")
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--episodes", type=int, default=200)
    args = ap.parse_args()

    # ---- real replays ----------------------------------------------------
    real: Counter = Counter()
    real_turns: list[int] = []
    n = 0
    for path in iter_episode_files():
        if n >= args.episodes:
            break
        try:
            ep = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            continue
        steps = ep.get("steps") or []
        if not steps:
            continue
        n += 1
        cur = None
        for step in reversed(steps):
            for seat in (0, 1):
                try:
                    c = step[seat]["observation"]["current"]
                except (KeyError, IndexError, TypeError):
                    continue
                if c:
                    cur = c
                    break
            if cur:
                break
        if cur:
            real[classify(cur)] += 1
            real_turns.append(cur.get("turn") or 0)

    # ---- our self-play ---------------------------------------------------
    from cg.game import battle_finish, battle_select, battle_start
    agent = load_agent(ROOT / args.agent, "term_agent")
    deck = load_deck(ROOT / args.deck)
    mine: Counter = Counter()
    mine_turns: list[int] = []
    for g in range(args.games):
        random.seed(30_000 + g)
        obs, _ = battle_start(list(deck), list(deck))
        if obs is None:
            continue
        steps = 0
        while steps < 5000:
            sel = obs.get("select")
            if sel is None or not sel.get("option"):
                break
            obs = battle_select(agent(obs))
            steps += 1
        cur = obs.get("current") or {}
        mine[classify(cur)] += 1
        mine_turns.append(cur.get("turn") or 0)
        battle_finish()

    def show(label, c: Counter, turns: list[int]) -> None:
        tot = sum(c.values()) or 1
        med = sorted(turns)[len(turns) // 2] if turns else 0
        print(f"\n=== {label} (n={tot}, median turn {med}) ===")
        for k, v in c.most_common():
            print(f"  {k:<22} {v:>5} ({v/tot:.1%})")

    show(f"real replays", real, real_turns)
    show(f"self-play ({args.agent})", mine, mine_turns)


if __name__ == "__main__":
    main()
