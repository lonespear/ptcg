"""How often the postures and the protection rule actually fire, counted.

D25: every behaviour is a counted telemetry event. This plays the agent on a
decklist against an opponent and prints its own counters afterwards — posture
activations by archetype, gust selects the named order reordered, exposure and
last-attacker positions, and the per-decision cost.

    python scripts/posture_telemetry.py --deck external/grimmsnarl/deck.csv \
        --opponent external/grimmsnarl/deck.csv --games 20
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "engine"))

from ptcg.arena import load_deck, play_game  # noqa: E402


def load_agent(path: Path, name: str):
    cwd = os.getcwd()
    os.chdir(path.parent)
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        os.chdir(cwd)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="agent/main.py")
    ap.add_argument("--deck", default="external/grimmsnarl/deck.csv")
    ap.add_argument("--opponent", default=None,
                    help="the other seat's list; defaults to a mirror")
    ap.add_argument("--opponent-agent", default=None,
                    help="the other seat's agent; defaults to a self-match")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    mine = load_agent(ROOT / args.agent, "tele_a")
    other = (load_agent(ROOT / args.opponent_agent, "tele_b")
             if args.opponent_agent else mine)
    d0 = load_deck(ROOT / args.deck)
    d1 = load_deck(ROOT / (args.opponent or args.deck))

    wins = decided = 0
    t0 = time.time()
    for g in range(args.games):
        r = play_game(mine.agent, other.agent, d0, d1, seed=g)
        if r.winner is not None:
            decided += 1
            wins += 1 if r.winner == 0 else 0
    secs = time.time() - t0

    post = mine.TELEMETRY_POSTURE
    prot = mine.TELEMETRY_PROT
    dec = max(post["decisions"], 1)
    scored = max(prot["scored"], 1)
    print(f"{args.games} games in {secs:.1f}s "
          f"({1000 * secs / args.games:.0f} ms a game); "
          f"seat 0 won {wins} of {decided} decided")
    print(f"\nposture, over {post['decisions']} agent calls:")
    print(f"  matchup posture on   {post['active']} "
          f"({100.0 * post['active'] / dec:.1f}%)")
    for a, n in sorted(post["by_archetype"].items(), key=lambda kv: -kv[1]):
        print(f"    {a:28s} {n:7d}  ({100.0 * n / dec:.1f}%)")
    print(f"  card selects with a named target of theirs "
          f"{post['gust_selects']}")
    print(f"  options the named order reordered            "
          f"{post['gust_targeted']}")
    print(f"  postures.json missing                        "
          f"{post['specs_missing']}")
    print(f"\nprotection, over {prot['scored']} scored positions:")
    print(f"  at least one attacker exposed  {prot['exposed']} "
          f"({100.0 * prot['exposed'] / scored:.1f}%)")
    print(f"  last attacker exposed          {prot['last_attacker']} "
          f"({100.0 * prot['last_attacker'] / scored:.2f}%)")
    print(f"\ntrajectory: {mine.TELEMETRY_TRAJ}")
    print(f"search: {mine.TELEMETRY_OPP['searches']} searches, "
          f"{mine.TELEMETRY_OPP['overrides']} overrides "
          f"({mine.TELEMETRY_OPP['overrides'] / max(mine.TELEMETRY_OPP['searches'], 1):.4f})")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "agent": args.agent, "deck": args.deck,
            "opponent_deck": args.opponent or args.deck,
            "opponent_agent": args.opponent_agent,
            "games": args.games, "seconds": round(secs, 1),
            "ms_per_game": round(1000 * secs / max(args.games, 1), 1),
            "seat0_wins": wins, "decided": decided,
            "env": {k: os.environ[k] for k in sorted(os.environ)
                    if k.startswith("CABT_")},
            "posture": post, "protection": prot,
            "trajectory": mine.TELEMETRY_TRAJ,
            "search": mine.TELEMETRY_OPP,
        }, indent=1))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
