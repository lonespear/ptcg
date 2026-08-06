"""Export a decklist from the mined field, and explain why it beats the field.

    python scripts/pick_field_deck.py --rank 3 --write
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "engine"))

from ptcg.meta import signature_to_deck  # noqa: E402

HISTORY = ROOT / "data" / "history_decklists.csv"
OUT = ROOT / "agent" / "deck.csv"

ENERGY_NAME = {0: "Colorless", 1: "Grass", 2: "Fire", 3: "Water", 4: "Lightning",
               5: "Psychic", 6: "Fighting", 7: "Darkness", 8: "Metal",
               9: "Dragon", 10: "Rainbow", 11: "Team Rocket"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=3,
                    help="index into the most-played decklists")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    from cg.sim import lib
    cards = {c["cardId"]: c for c in json.loads(lib.AllCard().decode())}

    df = pd.read_csv(HISTORY)
    g = (df.groupby("signature", as_index=False)
         .agg(plays=("decks", "sum"), wins=("wins", "sum"),
              archetype=("archetype", "first")))
    top = g.nlargest(12, "plays").reset_index(drop=True)
    print("most-played decklists:")
    for i, r in top.iterrows():
        print(f"  [{i}] {r.archetype:<26} plays={r.plays:<6} "
              f"wr={r.wins / r.plays:.3f}")

    pick = top.iloc[args.rank]
    deck = signature_to_deck(pick.signature)
    counts = Counter(deck)

    print(f"\n=== [{args.rank}] {pick.archetype} — {pick.plays} plays, "
          f"{pick.wins / pick.plays:.3f} real win rate ===")
    attack_types: Counter = Counter()
    for cid, n in counts.most_common():
        c = cards.get(cid, {})
        bits = []
        if c.get("hp"):
            bits.append(f"hp={c['hp']}")
        et = c.get("energyType")
        if et:
            bits.append(ENERGY_NAME.get(et, str(et)))
            if c.get("cardType") == 0:
                attack_types[ENERGY_NAME.get(et, str(et))] += n
        w = c.get("weakness")
        if w:
            bits.append(f"weak:{ENERGY_NAME.get(w, w)}")
        print(f"  {n}x  {c.get('name', cid):<30} {' '.join(bits)}")
    print(f"  total {len(deck)} cards, {len(counts)} distinct")
    print(f"\n  Pokémon types in this deck: {dict(attack_types)}")

    # Why does it beat the field? Show what the field is weak to.
    print("\n  field weaknesses (top decks, their Pokémon's weakness):")
    for j, r in top.head(6).iterrows():
        wk: Counter = Counter()
        for cid in set(signature_to_deck(r.signature)):
            c = cards.get(cid, {})
            if c.get("cardType") == 0 and c.get("weakness"):
                wk[ENERGY_NAME.get(c["weakness"], c["weakness"])] += 1
        print(f"    [{j}] {r.archetype:<26} weak to {dict(wk)}")

    if args.write:
        OUT.write_text("\n".join(str(c) for c in deck) + "\n")
        print(f"\nwrote {OUT.relative_to(ROOT)}")
    else:
        print("\n(not written — pass --write)")


if __name__ == "__main__":
    main()
