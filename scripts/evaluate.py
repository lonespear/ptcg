"""Play an agent against a baseline and report the win rate.

    python scripts/evaluate.py --games 200

Sides are swapped every other game, so the first-player advantage cancels out
and the number reported is the agent's true edge.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "engine"))
sys.path.insert(0, str(ROOT / "agent"))

from ptcg.arena import load_deck, match, random_agent  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--deck", default=str(ROOT / "agent" / "deck.csv"))
    ap.add_argument("--opponent-deck", default=None,
                    help="defaults to the same deck, so only policy differs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--opponent", choices=("random", "heuristic"),
                    default="random",
                    help="heuristic vs heuristic isolates deck strength")
    args = ap.parse_args()

    from main import agent as heuristic_agent

    deck = load_deck(args.deck)
    opp_deck = load_deck(args.opponent_deck) if args.opponent_deck else list(deck)
    opp = heuristic_agent if args.opponent == "heuristic" else random_agent

    same_deck = "mirror deck" if opp_deck == deck else "different decks"
    print(f"heuristic vs {args.opponent} — {args.games} games, {same_deck}")
    res = match(heuristic_agent, opp, deck, opp_deck,
                games=args.games, seed0=args.seed)

    print(f"\n  agent0 wins    : {res['agent0_wins']}")
    print(f"  opponent wins  : {res['agent1_wins']}")
    print(f"  draws          : {res['draws']}")
    print(f"  win rate       : {res['agent0_win_rate']:.3f}")
    print(f"  median turns   : {res['median_turns']}")
    if res["n_errors"]:
        print(f"  errors         : {res['n_errors']}")
        for e in res["errors"]:
            print(f"    - {e}")


if __name__ == "__main__":
    main()
