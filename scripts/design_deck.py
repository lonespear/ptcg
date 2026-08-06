"""Design a deck around what the heuristic agent can actually pilot.

The mined leaderboard decks assume a skilled pilot: evolution lines timed
correctly, Energy spread deliberately, Supporters sequenced. Our agent does none
of that, and piloting a top player's Mega Lucario list it loses 27-73 to the
trivial sample deck.

So this picks for pilotability instead:
  - Basic Pokémon only, so there is no evolution line to misplay
  - one Energy type, so every attachment is a useful attachment
  - attacks that are cheap and unconditional, so "attack with the biggest
    number" is the right policy

    python scripts/design_deck.py --write
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "engine"))

from ptcg import load_cards  # noqa: E402

OUT = ROOT / "agent" / "deck.csv"

# Basic Energy card ids, one per type, from the engine's card table.
ENERGY_BY_TYPE = {
    1: 1,   # Grass
    2: 2,   # Fire
    3: 3,   # Water
    4: 4,   # Lightning
    5: 5,   # Psychic
    6: 6,   # Fighting
    7: 7,   # Darkness
    8: 8,   # Metal
}


def engine_tables() -> tuple[dict, dict]:
    from cg.sim import lib
    cards = {c["cardId"]: c for c in json.loads(lib.AllCard().decode())}
    attacks = {a["attackId"]: a for a in json.loads(lib.AllAttack().decode())}
    return cards, attacks


def candidates(cards: dict, attacks: dict, eff) -> list[dict]:
    """Basic Pokémon with a cheap, unconditional, hard-hitting attack."""
    import pandas as pd
    drawback = {n: d for n, d in zip(eff["effect_name"], eff["drawback"])
                if not pd.isna(d)}
    rows = []
    for c in cards.values():
        if c.get("cardType") != 0 or not c.get("basic"):
            continue
        hp = c.get("hp") or 0
        best = None
        for aid in c.get("attacks") or []:
            a = attacks.get(aid)
            if not a:
                continue
            cost = len(a.get("energies") or [])
            dmg = a.get("damage") or 0
            if cost == 0 or dmg == 0:
                continue
            # Text on an attack almost always means a condition or a cost.
            if a.get("text"):
                continue
            if drawback.get(a.get("name")):
                continue
            score = dmg / cost
            if best is None or score > best["dpe"]:
                best = {"attack": a["name"], "cost": cost, "damage": dmg,
                        "dpe": score, "energies": a.get("energies") or []}
        if best is None:
            continue
        types = {e for e in best["energies"] if e != 0}
        rows.append({
            "card_id": c["cardId"], "name": c["name"], "hp": hp,
            "ex": bool(c.get("ex")), "retreat": c.get("retreatCost") or 0,
            "energy_type": (sorted(types)[0] if types else 0),
            **best,
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    _, eff = load_cards()
    cards, attacks = engine_tables()
    rows = candidates(cards, attacks, eff)

    # Prefer single-prize attackers: our agent will lose Pokémon, and giving up
    # two prizes each time is how a mediocre pilot loses fastest.
    rows.sort(key=lambda r: (-r["dpe"], -r["hp"]))
    print(f"{len(rows)} Basic Pokémon with an unconditional priced attack\n")
    print(f"{'name':<28}{'hp':>5}{'cost':>6}{'dmg':>6}{'dpe':>7}  ex  type")
    for r in rows[:args.top]:
        print(f"{r['name']:<28}{r['hp']:>5}{r['cost']:>6}{r['damage']:>6}"
              f"{r['dpe']:>7.0f}  {'Y' if r['ex'] else '-'}   {r['energy_type']}")

    build_variants(rows, cards, args.write)


# Simple support that a greedy agent cannot misplay: pure draw and "put a Basic
# onto your Bench". Anything needing a target or a discard is left out.
SAFE_SUPPORT = {
    "Poké Pad": 4,
    "Buddy-Buddy Poffin": 4,
    "Lillie's Determination": 4,
}


def build_variants(rows: list[dict], cards: dict, write: bool) -> None:
    """Emit candidate 60-card builds that differ only in their resource ratios."""
    name_to_id = {c["name"]: cid for cid, c in cards.items()}
    out_dir = ROOT / "build" / "decks"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Group single-prize Basics by Energy type; a mono-type deck means every
    # Energy attachment is live.
    by_type: dict[int, list[dict]] = {}
    for r in rows:
        if r["ex"] or r["energy_type"] == 0:
            continue
        by_type.setdefault(r["energy_type"], []).append(r)

    best_type, best_pool = None, None
    for etype, pool in by_type.items():
        pool.sort(key=lambda r: (-r["dpe"], -r["hp"]))
        strength = sum(r["dpe"] for r in pool[:6])
        if best_pool is None or strength > best_pool:
            best_type, best_pool = etype, strength
    pool = sorted(by_type[best_type], key=lambda r: (-r["dpe"], -r["hp"]))
    energy_id = ENERGY_BY_TYPE[best_type]

    print(f"\nstrongest mono-type pool: energy type {best_type}")
    for r in pool[:6]:
        print(f"  {r['name']:<24} hp={r['hp']:<4} {r['cost']}E -> {r['damage']}")

    variants = {
        # (attackers, copies each, energy) -> the rest is safe support
        "aggro_24p_36e": (6, 4, 36, False),
        "aggro_24p_20e": (6, 4, 20, True),
        "aggro_16p_28e": (4, 4, 28, True),
        "aggro_20p_24e": (5, 4, 24, True),
    }

    written = {}
    for label, (n_atk, copies, n_energy, support) in variants.items():
        deck: Counter = Counter()
        for r in pool[:n_atk]:
            deck[r["card_id"]] = copies
        deck[energy_id] = n_energy
        if support:
            for nm, cnt in SAFE_SUPPORT.items():
                cid = name_to_id.get(nm)
                if cid is not None and sum(deck.values()) + cnt <= 60:
                    deck[cid] = cnt
        short = 60 - sum(deck.values())
        if short > 0:
            deck[energy_id] += short
        elif short < 0:
            deck[energy_id] += short
        if sum(deck.values()) != 60 or deck[energy_id] < 0:
            print(f"  [skip] {label}: could not reach exactly 60")
            continue
        flat = [cid for cid, n in deck.items() for _ in range(n)]
        (out_dir / f"{label}.csv").write_text(
            "\n".join(str(c) for c in flat) + "\n")
        written[label] = deck
        print(f"\n=== {label} ===")
        for cid, n in deck.most_common():
            print(f"  {n}x  {cards[cid]['name']}")

    print(f"\nwrote {len(written)} candidate decks to build/decks/")
    if write and written:
        label = next(iter(written))
        flat = [cid for cid, n in written[label].items() for _ in range(n)]
        OUT.write_text("\n".join(str(c) for c in flat) + "\n")
        print(f"wrote {label} to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
