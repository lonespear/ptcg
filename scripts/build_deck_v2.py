"""Build the Abomasnow deck the trace said we needed, and A/B it against the sample.

The sample deck wins the internal tournament but runs only 6 Basic Pokémon in
60 cards. The game trace showed why that matters: games end at turn 3-5 because
a player has no Pokémon left to promote, not because of prizes. Against a random
opponent that favours us; against a competent one it is a liability.

This keeps the Snover / Mega Abomasnow ex core (350 HP, 200 damage for 3 Water)
and spends the chaff slots on more Water Basics, so there is always something to
promote.

    python scripts/build_deck_v2.py --games 200
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "engine"))
sys.path.insert(0, str(ROOT / "agent"))

from ptcg.arena import load_deck, match  # noqa: E402

RECIPE = {
    # core: the 350 HP / 200 damage engine
    "Snover": 4,
    "Mega Abomasnow ex": 4,
    # extra Basics so the bench is never empty
    "Kyogre": 4,
    "Terapagos": 4,
    "Misty's Lapras": 4,
    "Tauros": 4,
    # support a greedy agent cannot misplay
    "Poké Pad": 4,
    "Buddy-Buddy Poffin": 4,
    "Lillie's Determination": 4,
    # the rest is Energy
    "Basic {W} Energy": 24,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    from cg.sim import lib
    cards = json.loads(lib.AllCard().decode())
    by_name: dict[str, int] = {}
    for c in cards:
        by_name.setdefault(c["name"], c["cardId"])

    deck: list[int] = []
    missing = []
    for name, n in RECIPE.items():
        cid = by_name.get(name)
        if cid is None:
            missing.append(name)
            continue
        deck.extend([cid] * n)
    if missing:
        sys.exit(f"card names not found: {missing}")
    if len(deck) != 60:
        sys.exit(f"recipe totals {len(deck)} cards, need 60")

    basics = sum(n for nm, n in RECIPE.items()
                 if nm in ("Snover", "Kyogre", "Terapagos", "Misty's Lapras",
                           "Tauros"))
    print(f"built 60 cards, {basics} Basic Pokémon "
          f"(sample deck has 6)\n")

    out = ROOT / "build" / "decks_v2.csv"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(str(c) for c in deck) + "\n")

    sample = load_deck(ROOT / "build" / "sample_deck.csv")
    from main import agent as heuristic

    print(f"A/B vs sample deck, both piloted by the heuristic "
          f"({args.games} games)")
    res = match(heuristic, heuristic, deck, sample, games=args.games, seed0=0)
    print(f"  v2 deck   : {res['agent0_wins']}")
    print(f"  sample    : {res['agent1_wins']}")
    print(f"  v2 win rate: {res['agent0_win_rate']:.3f}")
    print(f"  median turns: {res['median_turns']}")

    if args.write:
        if res["agent0_win_rate"] > 0.5:
            (ROOT / "agent" / "deck.csv").write_text(
                "\n".join(str(c) for c in deck) + "\n")
            print("\nwrote agent/deck.csv (v2 wins)")
        else:
            print("\nNOT written — v2 does not beat the sample deck")


if __name__ == "__main__":
    main()
