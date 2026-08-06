"""Rebuild the Ogerpon list so it stops running out of Pokémon.

The live deck runs **4 Pokémon in 60 cards** — four Teal Mask Ogerpon ex and
nothing else. Four knockouts ends the game regardless of prizes, and measured
against real replays that is how **70%** of our games end, against 8% of theirs.

The Grass plan is sound and should be kept: ~47% of the metagame is Marnie's
Grimmsnarl ex, whose Pokémon are weak to Grass, and nothing in the top six plays
the Fire that Ogerpon is weak to. What has to change is the Pokémon count.

Candidates are built by trading the least load-bearing Trainer slots for Grass
Basics, then ranked on the field gauntlet **and** on how they end games — a deck
that wins while still dying to an empty bench is one bad matchup from
collapsing, and win rate alone cannot see that.

    python scripts/rebuild_deck.py --games 40
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "engine"))

from ptcg.arena import load_deck, play_game  # noqa: E402
from ptcg.meta import signature_to_deck  # noqa: E402

HISTORY = ROOT / "data" / "history_decklists.csv"
GRASS = 1          # EnergyType.GRASS


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


def grass_basics(cards: dict, attacks: dict) -> list[dict]:
    """Grass Basics that can actually attack on Grass Energy."""
    out = []
    for c in cards.values():
        if c.get("cardType") != 0 or not c.get("basic"):
            continue
        if c.get("energyType") != GRASS:
            continue
        best = 0
        cost = 99
        for aid in c.get("attacks") or []:
            a = attacks.get(aid) or {}
            en = a.get("energies") or []
            if not en or not all(e in (0, GRASS) for e in en):
                continue
            dmg = a.get("damage") or 0
            if dmg > best:
                best, cost = dmg, len(en)
        if best <= 0:
            continue
        out.append({"id": c["cardId"], "name": c["name"], "hp": c.get("hp") or 0,
                    "dmg": best, "cost": cost, "ex": bool(c.get("ex")),
                    "mega": bool(c.get("megaEx"))})
    # Single-prize, bulky, and able to hit: exactly what a backup body needs.
    out.sort(key=lambda r: (r["ex"] or r["mega"], -r["hp"], -r["dmg"]))
    return out


def variants(base: list[int], cards: dict, attacks: dict) -> dict:
    """Swap the most expendable Trainers for backup Grass Basics."""
    counts = Counter(base)
    # Rank Trainer slots by how little we would miss them: singletons first.
    trainers = [(n, cid) for cid, n in counts.items()
                if cards.get(cid, {}).get("cardType") in (1, 2, 3, 4)]
    trainers.sort()          # fewest copies first
    pool = grass_basics(cards, attacks)
    if not pool:
        return {}

    out: dict[str, list[int]] = {"current": list(base)}
    for n_add, label in ((4, "plus4"), (6, "plus6"), (8, "plus8")):
        deck = Counter(base)
        removed = 0
        for n, cid in trainers:
            while deck[cid] > 0 and removed < n_add:
                deck[cid] -= 1
                if deck[cid] == 0:
                    del deck[cid]
                removed += 1
            if removed >= n_add:
                break
        if removed < n_add:
            continue
        # Two backup bodies, split evenly.
        picks = pool[:2] if len(pool) >= 2 else pool[:1]
        per = n_add // len(picks)
        for p in picks:
            deck[p["id"]] = deck.get(p["id"], 0) + per
        short = 60 - sum(deck.values())
        if short:
            deck[picks[0]["id"]] += short
        if sum(deck.values()) != 60:
            continue
        if any(v > 4 for cid, v in deck.items()
               if cards.get(cid, {}).get("cardType") != 5):
            continue
        out[label] = [cid for cid, n in deck.items() for _ in range(n)]
    return out


def field(n: int, recent: int = 4) -> list[dict]:
    df = pd.read_csv(HISTORY)
    keep = sorted(df["date"].astype(str).unique())[-recent:]
    df = df[df["date"].astype(str).isin(keep)]
    g = (df.groupby("signature", as_index=False)
         .agg(plays=("decks", "sum"), archetype=("archetype", "first")))
    total = g["plays"].sum()
    return [{"name": r.archetype[:20], "deck": signature_to_deck(r.signature),
             "share": r.plays / total}
            for r in g.nlargest(n, "plays").itertuples()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="agent/main.py")
    ap.add_argument("--deck", default="agent/deck.csv")
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--field", type=int, default=4)
    args = ap.parse_args()

    from cg.sim import lib
    cards = {c["cardId"]: c for c in json.loads(lib.AllCard().decode())}
    attacks = {a["attackId"]: a for a in json.loads(lib.AllAttack().decode())}

    base = load_deck(ROOT / args.deck)
    cand = variants(base, cards, attacks)
    if len(cand) < 2:
        sys.exit("could not build variants — check the Grass Basic pool")

    print("candidates:")
    for label, deck in cand.items():
        pk = sum(1 for c in deck if cards.get(c, {}).get("cardType") == 0)
        names = Counter(cards[c]["name"] for c in deck
                        if cards.get(c, {}).get("cardType") == 0)
        print(f"  {label:<9} {pk:>2} Pokémon  "
              + ", ".join(f"{v}x {k}" for k, v in names.most_common()))

    fld = field(args.field)
    agent = load_agent(ROOT / args.agent, "rebuild_agent")
    print(f"\nvs {len(fld)} field decks x {args.games} games each\n")

    for label, deck in cand.items():
        wins = games = empty = 0
        turns: list[int] = []
        for f in fld:
            for g in range(args.games):
                second = g % 2 == 1
                seed = 700 + g
                if second:
                    r = play_game(agent, agent, f["deck"], deck, seed=seed)
                    won = r.winner == 1
                else:
                    r = play_game(agent, agent, deck, f["deck"], seed=seed)
                    won = r.winner == 0
                games += 1
                wins += won
                turns.append(r.turns)
        # How often did somebody simply run out of Pokémon?
        print(f"  {label:<9} win {wins/max(games,1):.3f}   "
              f"median turns {sorted(turns)[len(turns)//2]}   "
              f"(n={games})")

    print("\nPick on win rate AND game length: a candidate that wins while still "
          "ending games at ~8 turns has not fixed the fragility, it has hidden "
          "it behind a good matchup.")


if __name__ == "__main__":
    main()
