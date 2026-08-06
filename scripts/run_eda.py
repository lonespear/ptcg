"""First EDA pass over the competition card pool.

Writes figures to figures/ and prints the numbers behind each one, so the
findings in the writeup can be traced back to a specific table.

    python scripts/run_eda.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ptcg import load_cards
from ptcg.viz import (CMAP_BLUE, INK_MUTED, INK_SECONDARY, SERIES, SURFACE,
                      despine, label_bars_h, use_style)

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
FIG_DIR.mkdir(exist_ok=True)

ENERGY_ORDER = ["Grass", "Fire", "Water", "Lightning", "Psychic",
                "Fighting", "Darkness", "Metal", "Dragon", "Colorless"]


def head(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def save(fig, name: str) -> None:
    path = FIG_DIR / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  [figure] {path.relative_to(FIG_DIR.parent)}")


# ---------------------------------------------------------------------------
# 1. What is actually in the legal pool?
# ---------------------------------------------------------------------------
def fig_pool_composition(cards: pd.DataFrame) -> None:
    head("1. Card pool composition")
    comp = cards["stage"].value_counts()
    print(comp.to_string())
    print(f"\ntotal unique cards: {len(cards)}")
    print(f"rule-box cards (ex / Mega ex / ACE SPEC): {int(cards['has_rule_box'].sum())} "
          f"({cards['has_rule_box'].mean():.1%})")

    order = comp.sort_values()
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.barh(order.index, order.values, height=0.62, color=SERIES[0])
    label_bars_h(ax, order.values)
    ax.set_xlim(0, order.max() * 1.12)
    ax.set_xlabel("cards")
    ax.set_title("Basic Pokémon dominate the legal pool")
    ax.grid(axis="y", visible=False)
    despine(ax)
    save(fig, "01_pool_composition.png")


# ---------------------------------------------------------------------------
# 2. HP vs the prize cost of a knockout — the central deckbuilding tradeoff
# ---------------------------------------------------------------------------
def fig_hp_vs_prizes(cards: pd.DataFrame) -> None:
    head("2. HP distribution: rule-box vs single-prize Pokémon")
    pk = cards[cards["is_pokemon"] & cards["hp"].notna()].copy()
    pk["prizes"] = np.where(pk["has_rule_box"], "Rule box (2+ prizes)", "Single prize")
    print(pk.groupby("prizes")["hp"].describe()[["count", "mean", "50%", "max"]].to_string())

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bins = np.arange(30, 401, 20)
    for i, (label, grp) in enumerate(pk.groupby("prizes")):
        ax.hist(grp["hp"].astype(float), bins=bins, histtype="step",
                linewidth=2, color=SERIES[i], label=label)
    ax.set_xlabel("HP")
    ax.set_ylabel("cards")
    ax.set_title("Rule-box Pokémon buy HP with prize risk")
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)
    despine(ax)
    save(fig, "02_hp_vs_prizes.png")

    # The number that matters: HP bought per extra prize given up.
    med_rb = pk.loc[pk["has_rule_box"], "hp"].median()
    med_sp = pk.loc[~pk["has_rule_box"], "hp"].median()
    print(f"\nmedian HP  rule-box {med_rb:.0f}  vs  single-prize {med_sp:.0f} "
          f"(+{med_rb - med_sp:.0f} HP for the second prize)")


# ---------------------------------------------------------------------------
# 3. The attack efficiency frontier
# ---------------------------------------------------------------------------
def fig_efficiency(cards: pd.DataFrame, eff: pd.DataFrame) -> None:
    head("3. Attack efficiency — damage per energy")
    atk = eff[(eff["effect_kind"] == "Attack") & eff["damage_base"].notna()
              & (eff["cost_total"] > 0)].merge(
        cards[["card_id", "has_rule_box", "hp", "type_name"]], on="card_id", how="left")

    tier = atk.groupby("cost_total").agg(
        attacks=("damage_base", "size"),
        median_damage=("damage_base", "median"),
        median_dpe=("damage_per_energy", "median"),
        max_damage=("damage_base", "max"),
    )
    print(tier.to_string())

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for i, (label, grp) in enumerate(
            atk.groupby(np.where(atk["has_rule_box"], "Rule box", "Single prize"))):
        jitter = (np.random.default_rng(0 + i).random(len(grp)) - 0.5) * 0.28
        ax.scatter(grp["cost_total"] + jitter, grp["damage_base"].astype(float),
                   s=16, alpha=0.55, color=SERIES[i], label=label,
                   linewidths=0.5, edgecolors=SURFACE)
    med = atk.groupby("cost_total")["damage_base"].median()
    ax.plot(med.index, med.values, color=INK_SECONDARY, linewidth=2,
            marker="o", markersize=5, label="median damage", zorder=5)
    ax.set_xlabel("energy cost (total symbols)")
    ax.set_ylabel("base damage")
    ax.set_title("Damage scales with energy, but the spread is the strategy")
    ax.set_xticks(sorted(atk["cost_total"].unique()))
    ax.legend(loc="upper left")
    despine(ax)
    save(fig, "03_efficiency_frontier.png")

    head("3b. Efficiency leaders — and what they actually pay")
    flat = atk[atk["damage_mod"] == "flat"]
    print(f"conditional attacks: {flat['is_conditional'].mean():.1%} of flat-damage attacks")
    print("\ndrawback types among the top 40 by damage/energy:")
    print(flat.nlargest(40, "damage_per_energy")["drawback"]
          .fillna("(none)").value_counts().to_string())

    best = flat.nlargest(15, "damage_per_energy")
    print("\ntop 15 by raw damage/energy:")
    print(best[["name", "effect_name", "cost_total", "damage_base",
                "damage_per_energy", "drawback"]].to_string(index=False))

    clean = flat[~flat["is_conditional"]].nlargest(15, "damage_per_energy")
    print("\ntop 15 with NO drawback — the honest efficiency ranking:")
    print(clean[["name", "effect_name", "cost_total", "damage_base",
                 "damage_per_energy", "has_rule_box"]].to_string(index=False))

    # Two panels, same x-scale: the naive ranking beside the one that survives
    # reading the card text. The gap between them is the whole point.
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2), sharex=True)
    xmax = float(best["damage_per_energy"].max()) * 1.18
    for ax, sub, title, color in (
        (axes[0], best, "Ranked by damage per energy", SERIES[1]),
        (axes[1], clean, "…after removing attacks with drawbacks", SERIES[0]),
    ):
        b = sub.iloc[::-1]
        lbl = [f"{n} · {m}" for n, m in zip(b["name"], b["effect_name"])]
        ax.barh(lbl, b["damage_per_energy"].astype(float), height=0.62, color=color)
        label_bars_h(ax, b["damage_per_energy"].astype(float).values, fmt="{:.0f}")
        ax.set_xlim(0, xmax)
        ax.set_xlabel("damage per energy")
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
        despine(ax)
    fig.suptitle("Every top-efficiency attack pays a hidden cost", x=0.005,
                 ha="left", fontsize=13, fontweight="600")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "03b_top_efficiency.png")


# ---------------------------------------------------------------------------
# 4. Type supply and the weakness graph — where the metagame targeting lives
# ---------------------------------------------------------------------------
def fig_type_and_weakness(cards: pd.DataFrame) -> None:
    head("4. Type supply and weakness structure")
    pk = cards[cards["is_pokemon"] & cards["type_name"].notna()].copy()
    supply = pk["type_name"].value_counts()
    print("cards per type:")
    print(supply.to_string())

    weak = pk[pk["weakness_name"].notna()]
    exposure = weak["weakness_name"].value_counts()
    print("\nPokémon weak to each attacking type:")
    print(exposure.to_string())

    types = [t for t in ENERGY_ORDER if t in set(supply.index) | set(exposure.index)]
    mat = (weak.groupby(["type_name", "weakness_name"]).size()
           .unstack(fill_value=0).reindex(index=types, columns=types, fill_value=0))

    fig, ax = plt.subplots(figsize=(7.6, 6.0))
    im = ax.imshow(mat.values, cmap=CMAP_BLUE, aspect="auto")
    ax.set_xticks(range(len(mat.columns)), mat.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(mat.index)), mat.index)
    ax.set_xlabel("weak to (attacking type)")
    ax.set_ylabel("defending Pokémon type")
    ax.set_title("The weakness graph is lopsided, not symmetric")
    hi = mat.values.max()
    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            v = mat.values[r, c]
            if v:
                ax.text(c, r, str(v), ha="center", va="center", fontsize=8,
                        color="#ffffff" if v > hi * 0.55 else INK_SECONDARY)
    ax.grid(False)
    despine(ax, keep=())
    fig.colorbar(im, ax=ax, shrink=0.7, label="cards")
    save(fig, "04_weakness_matrix.png")

    # Supply vs exposure: an attacking type is good if many targets are weak to it
    # and the pool actually offers attackers of that type.
    comp = pd.DataFrame({"supply": supply, "targets_weak_to_it": exposure}).fillna(0)
    comp["exposure_ratio"] = comp["targets_weak_to_it"] / len(weak)
    print("\nsupply vs exposure:")
    print(comp.sort_values("targets_weak_to_it", ascending=False).to_string())

    order = exposure.reindex([t for t in ENERGY_ORDER if t in exposure.index]).sort_values()
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.barh(order.index, order.values, height=0.62, color=SERIES[0])
    label_bars_h(ax, order.values)
    ax.set_xlim(0, order.max() * 1.12)
    ax.set_xlabel("Pokémon in the pool weak to this type")
    ax.set_title("Attacking types are not equally rewarded")
    ax.grid(axis="y", visible=False)
    despine(ax)
    save(fig, "05_weakness_exposure.png")


# ---------------------------------------------------------------------------
# 5. What do Trainer cards actually do?
# ---------------------------------------------------------------------------
EFFECT_PATTERNS = {
    "Draw cards": r"\bdraw\b",
    "Search deck": r"search your deck",
    "Energy acceleration": r"attach .*energy.*(from your (hand|discard)|to)",
    "Switch / gust": r"\bswitch\b|\bto the active spot\b",
    "Heal / remove damage": r"\bheal\b|remove .*damage counter",
    "Opponent disruption": r"your opponent('|’)s hand|opponent discards|shuffle .*opponent",
    "Recover from discard": r"from your discard pile.*(into your hand|to your hand)",
}


def fig_trainer_taxonomy(cards: pd.DataFrame, eff: pd.DataFrame) -> None:
    head("5. Trainer / support effect taxonomy")
    trainers = cards[cards["supertype"] == "Trainer"]
    text = (eff.merge(trainers[["card_id", "stage"]], on="card_id")
            .assign(text=lambda d: d["text"].fillna("").str.lower()))
    counts = {k: int(text["text"].str.contains(p, regex=True, na=False).sum())
              for k, p in EFFECT_PATTERNS.items()}
    ser = pd.Series(counts).sort_values()
    print(ser.sort_values(ascending=False).to_string())
    print(f"\ntrainer cards: {len(trainers)}  |  trainer effect rows: {len(text)}")

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.barh(ser.index, ser.values, height=0.62, color=SERIES[0])
    label_bars_h(ax, ser.values)
    ax.set_xlim(0, max(ser.max(), 1) * 1.14)
    ax.set_xlabel("Trainer cards mentioning this effect")
    ax.set_title("Consistency, not damage, is what Trainers buy")
    ax.grid(axis="y", visible=False)
    despine(ax)
    save(fig, "06_trainer_taxonomy.png")


# ---------------------------------------------------------------------------
# 6. Mobility: retreat cost
# ---------------------------------------------------------------------------
def fig_retreat(cards: pd.DataFrame) -> None:
    head("6. Retreat cost distribution")
    pk = cards[cards["is_pokemon"] & cards["retreat"].notna()]
    dist = pk["retreat"].value_counts().sort_index()
    print(dist.to_string())
    print(f"\nfree-retreat Pokémon: {int(dist.get(0, 0))} ({dist.get(0, 0) / len(pk):.1%})")

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.bar(dist.index.astype(int), dist.values, width=0.58, color=SERIES[0])
    for x, v in zip(dist.index.astype(int), dist.values):
        ax.text(x, v + dist.max() * 0.02, str(v), ha="center", va="bottom",
                fontsize=9, color=INK_SECONDARY)
    ax.set_ylim(0, dist.max() * 1.14)
    ax.set_xlabel("retreat cost (energy)")
    ax.set_ylabel("Pokémon")
    ax.set_title("Most Pokémon are cheap to reposition")
    ax.grid(axis="x", visible=False)
    despine(ax)
    save(fig, "07_retreat_cost.png")


def main() -> None:
    use_style()
    cards, eff = load_cards("EN")
    print(f"loaded {len(cards)} unique cards, {len(eff)} attacks/abilities")

    fig_pool_composition(cards)
    fig_hp_vs_prizes(cards)
    fig_efficiency(cards, eff)
    fig_type_and_weakness(cards)
    fig_trainer_taxonomy(cards, eff)
    fig_retreat(cards)

    print(f"\nAll figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
