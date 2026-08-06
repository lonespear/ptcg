"""Round-robin candidate decks against each other, all piloted by our agent.

Deck strength is not separable from the policy that plays it, so the only
meaningful ranking is the one measured with the agent we will actually submit.

    python scripts/tournament.py --games 60
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "engine"))
sys.path.insert(0, str(ROOT / "agent"))

from ptcg.arena import load_deck, match  # noqa: E402


def collect_decks() -> dict[str, list[int]]:
    decks: dict[str, list[int]] = {}
    for p in sorted((ROOT / "build" / "decks").glob("*.csv")):
        decks[p.stem] = load_deck(p)
    sample = ROOT / "build" / "sample_deck.csv"
    if sample.exists():
        decks["sample"] = load_deck(sample)
    meta = ROOT / "build" / "meta_lucario.csv"
    if meta.exists():
        decks["meta_lucario"] = load_deck(meta)
    return decks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=60)
    args = ap.parse_args()

    from main import agent as heuristic

    decks = collect_decks()
    if len(decks) < 2:
        sys.exit("need at least two decks in build/decks/")
    print(f"{len(decks)} decks, {args.games} games per pair, "
          f"{len(list(itertools.combinations(decks, 2))) * args.games} games total\n")

    wins = {k: 0 for k in decks}
    played = {k: 0 for k in decks}
    for a, b in itertools.combinations(decks, 2):
        res = match(heuristic, heuristic, decks[a], decks[b],
                    games=args.games, seed0=0)
        wa, wb = res["agent0_wins"], res["agent1_wins"]
        wins[a] += wa
        wins[b] += wb
        played[a] += wa + wb
        played[b] += wa + wb
        print(f"  {a:<16} {wa:>3} - {wb:<3} {b}")

    print("\n=== standings ===")
    table = sorted(decks, key=lambda k: -(wins[k] / played[k] if played[k] else 0))
    for k in table:
        wr = wins[k] / played[k] if played[k] else 0
        print(f"  {k:<16} {wr:.3f}  ({wins[k]}/{played[k]})")
    print(f"\nbest: {table[0]}")


if __name__ == "__main__":
    main()
