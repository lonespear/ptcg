"""Aggregate parsed replays into per-day metagame tables.

Kept separate from the mining script so the same aggregation runs over one day
or a stitched-together history.
"""

from __future__ import annotations

from collections import Counter

import pandas as pd


def label_archetype(deck: dict, by_id: pd.DataFrame) -> str:
    """Name a deck by its main attacker — the highest-HP Pokémon it runs."""
    best, best_hp = None, -1
    for cid in deck:
        if cid not in by_id.index:
            continue
        r = by_id.loc[cid]
        if not r["is_pokemon"] or pd.isna(r["hp"]):
            continue
        hp = int(r["hp"])
        if hp > best_hp:
            best, best_hp = r["name"], hp
    return best or "(no Pokémon)"


def aggregate_day(rows: list[dict], by_id: pd.DataFrame, date: str
                  ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Turn deck instances into (cards, archetypes, agents) tables for one day.

    Counts are kept as raw numerators/denominators rather than rates so days
    can be summed into a history without weighting mistakes.
    """
    df = pd.DataFrame(rows)
    if df.empty:
        empty = pd.DataFrame()
        return empty, empty, empty

    df["archetype"] = df["deck"].map(lambda d: label_archetype(d, by_id))

    plays: Counter = Counter()
    wins: Counter = Counter()
    copies: Counter = Counter()
    for deck, won in zip(df["deck"], df["won"]):
        for cid, n in deck.items():
            plays[cid] += 1
            copies[cid] += n
            wins[cid] += won

    n_decks = len(df)
    cards = pd.DataFrame([
        {"date": date, "card_id": cid,
         "name": by_id.loc[cid, "name"] if cid in by_id.index else f"?{cid}",
         "supertype": by_id.loc[cid, "supertype"] if cid in by_id.index else "?",
         "decks": p, "wins": wins[cid], "copies": copies[cid],
         "day_decks": n_decks}
        for cid, p in plays.items()
    ])

    arch = (df.groupby("archetype")
            .agg(decks=("won", "size"), wins=("won", "sum"))
            .reset_index().assign(date=date, day_decks=n_decks))

    agents = (df.groupby("agent")
              .agg(games=("won", "size"), wins=("won", "sum"))
              .reset_index().assign(date=date))

    return cards, arch, agents


def summarize_history(cards: pd.DataFrame, arch: pd.DataFrame,
                      agents: pd.DataFrame, min_decks: int = 40
                      ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Collapse a multi-day history into overall rates."""
    c = (cards.groupby(["card_id", "name", "supertype"], as_index=False)
         .agg(decks=("decks", "sum"), wins=("wins", "sum"),
              copies=("copies", "sum")))
    total_decks = cards.groupby("date")["day_decks"].first().sum()
    c["play_rate"] = c["decks"] / total_decks
    c["win_rate"] = c["wins"] / c["decks"]
    c["avg_copies"] = c["copies"] / c["decks"]

    a = (arch.groupby("archetype", as_index=False)
         .agg(decks=("decks", "sum"), wins=("wins", "sum")))
    a["win_rate"] = a["wins"] / a["decks"]

    g = (agents.groupby("agent", as_index=False)
         .agg(games=("games", "sum"), wins=("wins", "sum")))
    g["win_rate"] = g["wins"] / g["games"]

    return (c.sort_values("play_rate", ascending=False),
            a[a["decks"] >= min_decks].sort_values("win_rate", ascending=False),
            g.sort_values("win_rate", ascending=False))
