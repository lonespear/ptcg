"""Check the simulated world against the real one, distributionally.

The Energy-hand bug — every rollout modelled an opponent holding nothing but
basic Energy — was found by reading code, not by playing games. It would have
shown up immediately in a distribution check: simulated opponents attaching
Energy every turn while never benching a Pokémon is a wild anomaly against
57,000 real games.

We already did this check once by accident. Games ending on turn 5 against a
real-replay median of 146 steps was exactly this signal, and it caught the first
broken benchmark. This makes it deliberate and repeatable, so it can be re-run
after any change to determinization or the rollout policy.

Compares our self-play games against mined replays on:
  * game length in steps and turns
  * Energy attachments per turn
  * Supporter/Item plays per turn
  * hand size trajectory
  * prize-trade tempo

    python scripts/validate_rollout.py --games 40 --episodes 200
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "engine"))

from ptcg.arena import load_deck  # noqa: E402
from ptcg.episodes import iter_episode_files  # noqa: E402

# LogType values we care about (see cg/api.py).
LOG_TURN_END, LOG_PLAY, LOG_ATTACH, LOG_ATTACK = 3, 10, 11, 15


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


def summarise(name: str, stats: dict) -> None:
    print(f"\n=== {name} ===")
    for k in ("steps", "turns", "attach_per_turn", "play_per_turn",
              "attack_per_turn", "hand_size"):
        xs = stats.get(k) or []
        if not xs:
            continue
        xs = sorted(xs)
        med = xs[len(xs) // 2]
        mean = statistics.mean(xs)
        print(f"  {k:<18} n={len(xs):<6} median={med:>7.1f}  mean={mean:>7.2f}"
              f"  p10={xs[len(xs)//10]:>6.1f}  p90={xs[9*len(xs)//10]:>6.1f}")


def replay_stats(limit: int) -> dict:
    stats: dict = defaultdict(list)
    n = 0
    for path in iter_episode_files():
        if n >= limit:
            break
        try:
            ep = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            continue
        steps = ep.get("steps") or []
        if not steps:
            continue
        n += 1
        stats["steps"].append(len(steps))
        counts = defaultdict(int)
        last_turn = 0
        for step in steps:
            try:
                obs = step[0]["observation"]
            except (KeyError, IndexError, TypeError):
                continue
            cur = obs.get("current") or {}
            last_turn = max(last_turn, cur.get("turn") or 0)
            for lg in obs.get("logs") or []:
                counts[lg.get("type")] += 1
            players = cur.get("players") or []
            if players and isinstance(players[0].get("handCount"), int):
                stats["hand_size"].append(players[0]["handCount"])
        stats["turns"].append(last_turn)
        t = max(last_turn, 1)
        stats["attach_per_turn"].append(counts[LOG_ATTACH] / t)
        stats["play_per_turn"].append(counts[LOG_PLAY] / t)
        stats["attack_per_turn"].append(counts[LOG_ATTACK] / t)
    print(f"replays read: {n}")
    return stats


def selfplay_stats(agent, deck: list[int], games: int) -> dict:
    from cg.game import battle_finish, battle_select, battle_start
    stats: dict = defaultdict(list)
    for g in range(games):
        random.seed(20_000 + g)
        obs, _ = battle_start(list(deck), list(deck))
        if obs is None:
            continue
        counts = defaultdict(int)
        steps = 0
        last_turn = 0
        while steps < 5000:
            sel = obs.get("select")
            if sel is None or not sel.get("option"):
                break
            cur = obs.get("current") or {}
            last_turn = max(last_turn, cur.get("turn") or 0)
            for lg in obs.get("logs") or []:
                counts[lg.get("type")] += 1
            players = cur.get("players") or []
            if players and isinstance(players[0].get("handCount"), int):
                stats["hand_size"].append(players[0]["handCount"])
            obs = battle_select(agent(obs))
            steps += 1
        battle_finish()
        stats["steps"].append(steps)
        stats["turns"].append(last_turn)
        t = max(last_turn, 1)
        stats["attach_per_turn"].append(counts[LOG_ATTACH] / t)
        stats["play_per_turn"].append(counts[LOG_PLAY] / t)
        stats["attack_per_turn"].append(counts[LOG_ATTACK] / t)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="agent/main.py")
    ap.add_argument("--deck", default="agent/deck.csv")
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--episodes", type=int, default=200)
    args = ap.parse_args()

    real = replay_stats(args.episodes)
    agent = load_agent(ROOT / args.agent, "rollout_agent")
    mine = selfplay_stats(agent, load_deck(ROOT / args.deck), args.games)

    summarise("real replays", real)
    summarise(f"self-play ({args.agent})", mine)

    print("\n=== ratio (self-play / real), 1.0 means we look like the field ===")
    for k in ("steps", "turns", "attach_per_turn", "play_per_turn",
              "attack_per_turn", "hand_size"):
        a, b = mine.get(k) or [], real.get(k) or []
        if not a or not b:
            continue
        ma, mb = statistics.median(a), statistics.median(b)
        flag = "" if 0.6 <= (ma / mb if mb else 0) <= 1.6 else "   <-- ANOMALY"
        print(f"  {k:<18} {ma:>7.2f} / {mb:>7.2f} = {ma/mb if mb else 0:>5.2f}{flag}")


if __name__ == "__main__":
    main()
