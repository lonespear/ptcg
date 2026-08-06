"""Play two agent versions against each other on the same deck.

Same deck both sides means the result is pure policy difference - the deck
cannot flatter either one.

    python scripts/compare_agents.py --a agent/main.py --b build/agents/v3.py
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "engine"))

from ptcg.arena import load_deck, match, match_sprt, random_agent  # noqa: E402


def load_agent(path: Path, name: str):
    """Import an agent, with the cwd set to its own folder.

    Rule-based agents read deck.csv at import time, so they only load correctly
    from beside their own deck.
    """
    import os
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="agent/main.py")
    ap.add_argument("--b", default="build/agents/v3.py")
    ap.add_argument("--deck", default="agent/deck.csv")
    ap.add_argument("--deck-b", default=None,
                    help="B's deck; defaults to A's, which isolates policy")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0,
                    help="first game seed; change it for an independent sample")
    ap.add_argument("--vs-random", action="store_true")
    ap.add_argument("--sprt", action="store_true",
                    help="stop when the evidence decides, not at a game count")
    ap.add_argument("--p0", type=float, default=0.50,
                    help="SPRT null: win rate meaning 'no improvement'")
    ap.add_argument("--p1", type=float, default=0.55,
                    help="SPRT alternative: win rate worth shipping")
    args = ap.parse_args()

    deck = load_deck(ROOT / args.deck)
    deck_b = load_deck(ROOT / args.deck_b) if args.deck_b else list(deck)
    a = load_agent(ROOT / args.a, "agent_a")

    if args.vs_random:
        print(f"{args.a} vs random - {args.games} games, mirror deck")
        res = match(a, random_agent, deck, list(deck), games=args.games, seed0=args.seed)
    else:
        b = load_agent(ROOT / args.b, "agent_b")
        same = "same deck" if args.deck_b is None else "each with its own deck"
        if args.sprt:
            print(f"{args.a}  vs  {args.b} - SPRT p0={args.p0} p1={args.p1}, "
                  f"{same}, max {args.games}")
            res = match_sprt(a, b, deck, deck_b, p0=args.p0, p1=args.p1,
                             max_games=args.games, seed0=args.seed)
            print(f"\n  decision     : {res['decision']}  "
                  f"(llr {res['llr']:+.2f})")
        else:
            print(f"{args.a}  vs  {args.b} - {args.games} games, {same}")
            res = match(a, b, deck, deck_b, games=args.games, seed0=args.seed)

    print(f"  A wins       : {res['agent0_wins']}")
    print(f"  B wins       : {res['agent1_wins']}")
    print(f"  A win rate   : {res['agent0_win_rate']:.3f}")
    print(f"  median turns : {res['median_turns']}")
    if res["n_errors"]:
        print(f"  errors       : {res['n_errors']}")
        for e in res["errors"]:
            print(f"    - {e}")


if __name__ == "__main__":
    main()
