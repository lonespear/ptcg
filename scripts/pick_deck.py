"""Choose a decklist from the mined history and write it to agent/deck.csv.

A decklist that wins on the leaderboard is a tested artifact. Rather than
guessing at 60 cards, take the best-performing list that has enough games behind
it to be believable, and use a Wilson lower bound so a 5-game 100% list does not
outrank a 300-game 58% list.

    python scripts/pick_deck.py                # show the shortlist
    python scripts/pick_deck.py --write        # write the top pick to agent/deck.csv
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "engine"))

from ptcg import load_cards  # noqa: E402
from ptcg.meta import signature_to_deck  # noqa: E402

DECKS_CSV = ROOT / "data" / "history_decklists.csv"
OUT = ROOT / "agent" / "deck.csv"


def wilson_lower(wins: int, n: int, z: float = 1.96) -> float:
    """Lower bound of the win-rate confidence interval — penalises small samples."""
    if n == 0:
        return 0.0
    p = wins / n
    d = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return (centre - margin) / d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--min-games", type=int, default=30)
    ap.add_argument("--rank", type=int, default=0,
                    help="which shortlist entry to write (0 = best)")
    args = ap.parse_args()

    if not DECKS_CSV.exists():
        sys.exit(f"{DECKS_CSV} not found — run scripts/mine_day.py first")

    df = pd.read_csv(DECKS_CSV)
    # The same list can appear on several days; pool them.
    g = (df.groupby("signature", as_index=False)
         .agg(decks=("decks", "sum"), wins=("wins", "sum"),
              archetype=("archetype", "first"), n_agents=("n_agents", "max"),
              days=("date", "nunique")))
    g["win_rate"] = g["wins"] / g["decks"]
    g["score"] = [wilson_lower(w, n) for w, n in zip(g["wins"], g["decks"])]

    elig = g[g["decks"] >= args.min_games].sort_values("score", ascending=False)
    print(f"{len(g)} distinct decklists, {len(elig)} with >= {args.min_games} games\n")
    print(elig.head(12)[["archetype", "decks", "wins", "win_rate", "score",
                         "n_agents", "days"]].to_string(index=False))

    if elig.empty:
        sys.exit("\nno decklist meets the games threshold yet")

    pick = elig.iloc[args.rank]
    deck = signature_to_deck(pick["signature"])
    cards, _ = load_cards()
    by_id = cards.set_index("card_id")

    print(f"\n=== pick: {pick['archetype']} — {pick['decks']} games, "
          f"{pick['win_rate']:.1%} win rate (lower bound {pick['score']:.1%}) ===")
    counts: dict[int, int] = {}
    for c in deck:
        counts[c] = counts.get(c, 0) + 1
    rows = []
    for cid, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        name = by_id.loc[cid, "name"] if cid in by_id.index else f"?{cid}"
        stage = by_id.loc[cid, "stage"] if cid in by_id.index else "?"
        rows.append((n, name, stage))
    for n, name, stage in rows:
        print(f"  {n}x  {name:<34} {stage}")
    print(f"  total {len(deck)} cards, {len(counts)} distinct")

    if args.write:
        OUT.write_text("\n".join(str(c) for c in deck) + "\n")
        print(f"\nwrote {OUT.relative_to(ROOT)}")
    else:
        print("\n(not written — pass --write)")


if __name__ == "__main__":
    main()
