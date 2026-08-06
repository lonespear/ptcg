"""Mine the public episode replays into a picture of the live metagame.

Every replay carries both players' full 60-card decks and who won, so the dumps
are a direct readout of what the leaderboard is actually playing — and what is
actually winning.

    python scripts/mine_meta.py            # all downloaded days
    python scripts/mine_meta.py --limit 800

Writes data/meta_decks.csv (one row per deck instance) and data/meta_cards.csv
(one row per card), then renders figures.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ptcg import load_cards
from ptcg.episodes import iter_episode_files, parse_episode
from ptcg.viz import (INK_MUTED, INK_SECONDARY, SERIES, SURFACE, despine,
                      label_bars_h, use_style)

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
DATA_DIR = ROOT / "data"
FIG_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# A deck instance is one player's 60 cards in one match. The same agent plays
# many matches, so these are weighted by activity — that is the meta *as
# encountered at the table*, which is what an agent has to beat.
MIN_DECKS_FOR_WINRATE = 40


def _job(path):
    ep = parse_episode(path)
    if ep is None or ep.winner is None or len(ep.agents) != 2:
        return None
    return [
        {"episode_id": ep.episode_id, "agent": ep.agents[i], "player": i,
         "won": int(ep.winner == i), "n_steps": ep.n_steps,
         "deck": dict(ep.decks[i])}
        for i in (0, 1)
    ]


def collect(limit: int | None) -> pd.DataFrame:
    files = list(iter_episode_files())
    if limit:
        files = files[:limit]
    print(f"parsing {len(files)} episode files...")
    rows = []
    with ProcessPoolExecutor() as pool:
        for i, res in enumerate(pool.map(_job, files, chunksize=16), 1):
            if res:
                rows.extend(res)
            if i % 1000 == 0:
                print(f"  {i}/{len(files)}")
    df = pd.DataFrame(rows)
    print(f"-> {len(df)} deck instances from {df['episode_id'].nunique()} matches")
    return df


def label_archetype(deck: dict, by_id: pd.DataFrame) -> str:
    """Name a deck by its main attacker — the highest-HP Pokémon it runs."""
    best, best_hp = None, -1
    for cid, n in deck.items():
        if cid not in by_id.index:
            continue
        r = by_id.loc[cid]
        if not r["is_pokemon"] or pd.isna(r["hp"]):
            continue
        # Prefer evolved / high-HP attackers over the basics that fetch them.
        hp = int(r["hp"])
        if hp > best_hp:
            best, best_hp = r["name"], hp
    return best or "(no Pokémon)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    use_style()
    cards, _ = load_cards()
    by_id = cards.set_index("card_id")

    df = collect(args.limit)
    if df.empty:
        print("no episodes parsed — is the episode dataset downloaded?")
        return

    df["archetype"] = df["deck"].map(lambda d: label_archetype(d, by_id))
    df[["episode_id", "agent", "player", "won", "n_steps", "archetype"]].to_csv(
        DATA_DIR / "meta_decks.csv", index=False)

    n_decks = len(df)
    print(f"\noverall: {n_decks} deck instances, "
          f"{df['agent'].nunique()} distinct agents")
    print(f"median match length: {df['n_steps'].median():.0f} steps")

    # ---- card-level play rate and win rate --------------------------------
    plays: Counter = Counter()
    wins: Counter = Counter()
    copies: Counter = Counter()
    for deck, won in zip(df["deck"], df["won"]):
        for cid, n in deck.items():
            plays[cid] += 1
            copies[cid] += n
            wins[cid] += won

    card_rows = []
    for cid, p in plays.items():
        if cid not in by_id.index:
            continue
        r = by_id.loc[cid]
        card_rows.append({
            "card_id": cid, "name": r["name"], "stage": r["stage"],
            "supertype": r["supertype"], "is_ex": bool(r["is_ex"]),
            "decks": p, "play_rate": p / n_decks,
            "win_rate": wins[cid] / p, "avg_copies": copies[cid] / p,
        })
    cdf = pd.DataFrame(card_rows).sort_values("play_rate", ascending=False)
    cdf.to_csv(DATA_DIR / "meta_cards.csv", index=False)

    print(f"\ndistinct cards seen in play: {len(cdf)} of {len(cards)} in the pool "
          f"({len(cdf) / len(cards):.1%})")
    print("\n=== most-played cards ===")
    print(cdf.head(20)[["name", "supertype", "play_rate", "avg_copies",
                        "win_rate"]].to_string(index=False))

    strong = cdf[cdf["decks"] >= MIN_DECKS_FOR_WINRATE]
    print(f"\n=== highest win rate (>= {MIN_DECKS_FOR_WINRATE} decks) ===")
    print(strong.nlargest(15, "win_rate")[
        ["name", "supertype", "decks", "play_rate", "win_rate"]].to_string(index=False))
    print(f"\n=== lowest win rate (>= {MIN_DECKS_FOR_WINRATE} decks) ===")
    print(strong.nsmallest(10, "win_rate")[
        ["name", "supertype", "decks", "play_rate", "win_rate"]].to_string(index=False))

    fig_card_map(strong)
    fig_top_cards(cdf)
    fig_archetypes(df)


def _annotate_spread(ax, picks: pd.DataFrame, min_gap_frac: float = 0.052) -> None:
    """Label points with a leader line, pushing labels apart so none collide.

    Labels sit left of high-play-rate points and right of the rest, so nothing
    runs off either edge of the axes.
    """
    y0, y1 = ax.get_ylim()
    x0, x1 = ax.get_xlim()
    gap = (y1 - y0) * min_gap_frac
    picks = picks.sort_values("win_rate").reset_index(drop=True)

    placed_y: list[float] = []
    for _, r in picks.iterrows():
        y = float(r["win_rate"])
        if placed_y and y - placed_y[-1] < gap:
            y = placed_y[-1] + gap
        placed_y.append(y)

    x_mid = (x0 + x1) / 2
    for (_, r), ly in zip(picks.iterrows(), placed_y):
        px, py = float(r["play_rate"]), float(r["win_rate"])
        right = px < x_mid
        lx = px + (x1 - x0) * (0.022 if right else -0.022)
        ax.annotate(
            r["name"], xy=(px, py), xytext=(lx, ly),
            ha="left" if right else "right", va="center",
            fontsize=8, color=INK_SECONDARY,
            arrowprops=dict(arrowstyle="-", color=INK_MUTED,
                            linewidth=0.7, shrinkA=0, shrinkB=3),
        )


def fig_card_map(strong: pd.DataFrame) -> None:
    """The meta map: how often a card is played vs how often it wins."""
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    for i, (label, grp) in enumerate(strong.groupby(
            np.where(strong["supertype"] == "Pokémon", "Pokémon", "Trainer / Energy"))):
        ax.scatter(grp["play_rate"], grp["win_rate"], s=26, alpha=0.7,
                   color=SERIES[i], label=label, linewidths=0.5,
                   edgecolors=SURFACE)
    ax.axhline(0.5, color=INK_MUTED, linewidth=1, linestyle=(0, (4, 4)))
    ax.text(0.995, 0.503, "even", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=8, color=INK_MUTED)
    # Fix the limits before annotating so label placement has a stable frame.
    ax.margins(x=0.14, y=0.12)
    ax.autoscale_view()

    # Label only the extremes — a name on every point would be unreadable.
    picks = pd.concat([strong.nlargest(4, "play_rate"),
                       strong.nlargest(4, "win_rate"),
                       strong.nsmallest(3, "win_rate")]).drop_duplicates("card_id")
    _annotate_spread(ax, picks)

    ax.set_xlabel("play rate (share of decks running it)")
    ax.set_ylabel("win rate of decks running it")
    ax.set_title("Staples converge to 50% — the edge is in what few decks run")
    ax.legend(loc="lower right")
    despine(ax)
    fig.savefig(FIG_DIR / "08_meta_card_map.png")
    plt.close(fig)
    print(f"  [figure] figures/08_meta_card_map.png")


def fig_top_cards(cdf: pd.DataFrame) -> None:
    top = cdf.head(18).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.8, 5.6))
    ax.barh(top["name"], top["play_rate"] * 100, height=0.62, color=SERIES[0])
    label_bars_h(ax, (top["play_rate"] * 100).values, fmt="{:.0f}%")
    ax.set_xlim(0, top["play_rate"].max() * 100 * 1.15)
    ax.set_xlabel("share of decks running the card")
    ax.set_title("The consensus staples")
    ax.grid(axis="y", visible=False)
    despine(ax)
    fig.savefig(FIG_DIR / "09_meta_staples.png")
    plt.close(fig)
    print(f"  [figure] figures/09_meta_staples.png")


def fig_archetypes(df: pd.DataFrame) -> None:
    arch = df.groupby("archetype").agg(
        decks=("won", "size"), win_rate=("won", "mean")).reset_index()
    arch = arch[arch["decks"] >= MIN_DECKS_FOR_WINRATE].nlargest(15, "decks")
    print("\n=== archetypes by main attacker ===")
    print(arch.sort_values("decks", ascending=False).to_string(index=False))

    a = arch.sort_values("decks").iloc[-14:]
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    # Polarity, not identity — the documented diverging pair, split at even.
    colors = [SERIES[0] if w >= 0.5 else "#e34948" for w in a["win_rate"]]
    ax.barh(a["archetype"], a["win_rate"] * 100, height=0.62, color=colors)
    ax.axvline(50, color=INK_MUTED, linewidth=1, linestyle=(0, (4, 4)), zorder=3)
    ax.set_xlim(0, 100)
    # Labels in a fixed column so they never collide with the 50% reference.
    for i, (w, n) in enumerate(zip(a["win_rate"], a["decks"])):
        ax.text(72, i, f"{w * 100:.0f}%   n={n}", va="center", ha="left",
                fontsize=9, color=INK_SECONDARY)
    ax.set_xlabel("win rate")
    ax.set_title("Archetype win rate, by the deck's biggest attacker")
    ax.grid(axis="y", visible=False)
    despine(ax)
    fig.savefig(FIG_DIR / "10_archetype_winrate.png")
    plt.close(fig)
    print(f"  [figure] figures/10_archetype_winrate.png")


if __name__ == "__main__":
    main()
