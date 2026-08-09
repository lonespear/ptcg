"""Rebuild the deck against the field we actually play, on a prize budget.

The autopsy says where the rating goes: Archaludon ~0.29 and Mega Lucario 0-for-6,
together ~30% of our games at about 0.2, against 0.750 on the Grimmsnarl games
the deck was designed for.

Both losses are one mechanic. Mega Brave does **270 for two Energy** and Teal
Mask Ogerpon ex has **210 HP**; the Archaludon lists run Cinderace and Ogerpon is
weak to **Fire**. Nothing reasonable survives 270, so armour is the wrong axis.

The right axis is **prize liability**. Ogerpon concedes two prizes, so a
one-shot deck needs three knockouts to win. Every single-prize attacker we lead
with makes them take six instead — the same one-shot machine doing double the
work — while the Grass Weakness math keeps deleting Grimmsnarl's board.

So variants are indexed by how many two-prize bodies the list can afford, and
scored on the **matchup vector**, not just the weighted scalar: a flat 0.55
everywhere is worth more against a moving field than a spiky 0.75/0.30.

Success bar (from the encounter weights): Archaludon and Lucario **above ~0.45**
without Grimmsnarl dropping below **~0.65**. That takes ~0.52 overall to ~0.61.

    python scripts/rebuild_v2.py --games 60
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "engine"))

from ptcg.arena import load_deck, play_game  # noqa: E402

POOL = ROOT / "data" / "analysis" / "autopsy_pool.json"
GRASS = 1


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


def card_tables():
    from cg.sim import lib
    cards = {c["cardId"]: c for c in json.loads(lib.AllCard().decode())}
    attacks = {a["attackId"]: a for a in json.loads(lib.AllAttack().decode())}
    return cards, attacks


def prize_liability(deck, cards) -> float:
    """Expected prizes conceded per Pokémon body in the list."""
    bodies = [cards[c] for c in deck if cards.get(c, {}).get("cardType") == 0]
    if not bodies:
        return 0.0
    tot = sum(3 if b.get("megaEx") else 2 if b.get("ex") else 1 for b in bodies)
    return tot / len(bodies)


def single_prize_grass(cards, attacks):
    """Single-prize Grass Basics that can attack on Grass Energy."""
    out = []
    for c in cards.values():
        if c.get("cardType") != 0 or not c.get("basic"):
            continue
        if c.get("ex") or c.get("megaEx") or c.get("energyType") != GRASS:
            continue
        best, cost = 0, 99
        for aid in c.get("attacks") or []:
            a = attacks.get(aid) or {}
            en = a.get("energies") or []
            if not en or not all(e in (0, GRASS) for e in en):
                continue
            if (a.get("damage") or 0) > best:
                best, cost = a["damage"], len(en)
        if best <= 0:
            continue
        out.append({"id": c["cardId"], "name": c["name"], "hp": c.get("hp") or 0,
                    "dmg": best, "cost": cost})
    out.sort(key=lambda r: (-(r["dmg"] / max(r["cost"], 1)), -r["hp"]))
    return out


def build_variants(base, cards, attacks):
    """Trade Ogerpon copies and spare Trainers for single-prize Grass bodies."""
    pool = single_prize_grass(cards, attacks)[:3]
    if not pool:
        return {}
    counts = Counter(base)
    ogerpon = [c for c in counts
               if cards.get(c, {}).get("name", "").startswith("Teal Mask Ogerpon")]
    if not ogerpon:
        return {}
    og = ogerpon[0]
    trainers = sorted((n, c) for c, n in counts.items()
                      if cards.get(c, {}).get("cardType") in (1, 2, 3, 4))

    variants = {"current": list(base)}
    # (ogerpon copies kept, single-prize bodies added)
    for keep_og, add in ((4, 6), (3, 8), (2, 10), (4, 10)):
        d = Counter(base)
        need = add
        d[og] = keep_og
        need -= (4 - keep_og) * 0 + 0
        freed = 4 - keep_og
        take = add - freed
        for n, cid in trainers:
            while d.get(cid, 0) > 0 and take > 0:
                d[cid] -= 1
                if d[cid] == 0:
                    del d[cid]
                take -= 1
            if take <= 0:
                break
        if take > 0:
            continue
        per = add // len(pool)
        for p in pool:
            d[p["id"]] = d.get(p["id"], 0) + per
        short = 60 - sum(d.values())
        d[pool[0]["id"]] += short
        if sum(d.values()) != 60:
            continue
        if any(v > 4 for c, v in d.items()
               if cards.get(c, {}).get("cardType") != 5):
            continue
        variants[f"og{keep_og}_sp{add}"] = [c for c, n in d.items()
                                            for _ in range(n)]
    return variants


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="agent/main.py")
    ap.add_argument("--deck", default="agent/deck.csv")
    ap.add_argument("--games", type=int, default=60)
    args = ap.parse_args()

    if not POOL.exists():
        sys.exit("run scripts/build_autopsy_pool.py first")
    pool = json.loads(POOL.read_text(encoding="utf-8"))
    opps = pool["pool"]
    cards, attacks = card_tables()
    base = load_deck(ROOT / args.deck)
    variants = build_variants(base, cards, attacks)
    if len(variants) < 2:
        sys.exit("could not build variants")

    agent = load_agent(ROOT / args.agent, "rb_agent")
    print(f"pool: {pool['band']}, {len(opps)} opponents\n")
    print(f"{'variant':<14}{'pk':>4}{'prz/body':>10}  " +
          "".join(f"{o['archetype'][:11]:>13}" for o in opps) + f"{'WTD':>8}")

    for label, deck in variants.items():
        npk = sum(1 for c in deck if cards.get(c, {}).get("cardType") == 0)
        liab = prize_liability(deck, cards)
        vec, wtd = [], 0.0
        for o in opps:
            w = l = 0
            for g in range(args.games):
                second = g % 2 == 1
                if second:
                    r = play_game(agent, agent, o["deck"], deck, seed=800 + g)
                    won = r.winner == 1
                else:
                    r = play_game(agent, agent, deck, o["deck"], seed=800 + g)
                    won = r.winner == 0
                w += won
                l += not won
            wr = w / max(w + l, 1)
            vec.append(wr)
            wtd += wr * o["share"]
        print(f"{label:<14}{npk:>4}{liab:>10.2f}  " +
              "".join(f"{v:>13.3f}" for v in vec) + f"{wtd:>8.3f}")

    print("\nBar: Archaludon and Lucario > 0.45, Grimmsnarl still > 0.65.")
    print("Prefer a FLAT vector over a spiky one at equal weighted score — the "
          "field moves, and a 0.75/0.30 split is a bet that it will not.")


if __name__ == "__main__":
    main()
