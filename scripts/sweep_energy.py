"""How much Energy does a greedy pilot actually want?

The v2 deck (20 Basics, 24 Energy) lost 40-160 to the sample deck (6 Basics,
35 Energy). The hypothesis that explains it: with one attachment per turn, the
only thing that matters is how fast Mega Abomasnow ex (350 HP, 200 damage for
3 Water) comes online, and every non-Energy card is a turn where the attachment
does not happen.

This sweeps Energy count against that same core and lets the round-robin decide.

    python scripts/sweep_energy.py --games 80
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "engine"))
sys.path.insert(0, str(ROOT / "agent"))

from ptcg.arena import load_deck, match  # noqa: E402


def build(by_name: dict[str, int], recipe: dict[str, int]) -> list[int]:
    deck: list[int] = []
    for name, n in recipe.items():
        cid = by_name.get(name)
        if cid is None:
            raise KeyError(name)
        deck.extend([cid] * n)
    if len(deck) != 60:
        raise ValueError(f"{sum(recipe.values())} cards for {recipe}")
    return deck


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=80)
    args = ap.parse_args()

    from cg.sim import lib
    by_name: dict[str, int] = {}
    for c in json.loads(lib.AllCard().decode()):
        by_name.setdefault(c["name"], c["cardId"])

    E = "Basic {W} Energy"
    variants = {
        # core is always 4 Snover + 4 Mega Abomasnow ex
        "e52_bare":   {"Snover": 4, "Mega Abomasnow ex": 4, E: 52},
        "e44_poffin": {"Snover": 4, "Mega Abomasnow ex": 4,
                       "Buddy-Buddy Poffin": 4, "Poké Pad": 4, E: 44},
        "e40_kyogre": {"Snover": 4, "Mega Abomasnow ex": 4, "Kyogre": 4,
                       "Buddy-Buddy Poffin": 4, E: 44},
        "e35_support": {"Snover": 4, "Mega Abomasnow ex": 4, "Kyogre": 4,
                        "Poké Pad": 4, "Buddy-Buddy Poffin": 4,
                        "Lillie's Determination": 4, E: 36},
    }

    decks = {}
    for label, recipe in variants.items():
        try:
            decks[label] = build(by_name, recipe)
        except (KeyError, ValueError) as e:
            print(f"  [skip] {label}: {e}")
    decks["sample"] = load_deck(ROOT / "build" / "sample_deck.csv")

    from main import agent as heuristic

    n_pairs = len(list(itertools.combinations(decks, 2)))
    print(f"{len(decks)} decks, {args.games} games/pair, "
          f"{n_pairs * args.games} games\n")

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
        print(f"  {a:<14} {wa:>3} - {wb:<3} {b}")

    print("\n=== standings ===")
    order = sorted(decks, key=lambda k: -(wins[k] / max(played[k], 1)))
    for k in order:
        print(f"  {k:<14} {wins[k] / max(played[k],1):.3f}  "
              f"({wins[k]}/{played[k]})")

    best = order[0]
    print(f"\nbest: {best}")
    if best != "sample":
        (ROOT / "build" / f"{best}.csv").write_text(
            "\n".join(str(c) for c in decks[best]) + "\n")
        print(f"wrote build/{best}.csv")


if __name__ == "__main__":
    main()
