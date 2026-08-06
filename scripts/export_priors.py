"""Export the mined decklists as a compact prior the agent can ship.

The agent needs to guess the opponent's hidden cards to search. It cannot read
data/ at run time, so the top decklists get baked into a small JSON that rides
along in the submission bundle.

    python scripts/export_priors.py --top 40
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "history_decklists.csv"
OUT = ROOT / "agent" / "deck_priors.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--min-plays", type=int, default=5)
    args = ap.parse_args()

    if not SRC.exists():
        sys.exit(f"{SRC} not found — run scripts/mine_day.py first")

    df = pd.read_csv(SRC)
    g = (df.groupby("signature", as_index=False)
         .agg(plays=("decks", "sum"), wins=("wins", "sum"),
              archetype=("archetype", "first")))
    g = g[g["plays"] >= args.min_plays].nlargest(args.top, "plays")

    entries = []
    for r in g.itertuples():
        counts: dict[str, int] = {}
        for part in r.signature.split(","):
            cid, n = part.split(":")
            counts[cid] = counts.get(cid, 0) + int(n)
        entries.append({"c": counts, "p": int(r.plays),
                        "w": int(r.wins), "a": str(r.archetype)})

    total = int(g["plays"].sum())
    OUT.write_text(json.dumps({"total": total, "decks": entries},
                              separators=(",", ":")))
    print(f"wrote {OUT.relative_to(ROOT)}: {len(entries)} decklists, "
          f"{total} observed plays, {OUT.stat().st_size / 1024:.0f} KB")
    print("\ncoverage of the metagame by these lists:")
    all_plays = df.groupby("signature")["decks"].sum().sum()
    print(f"  {total}/{all_plays} = {total / all_plays:.1%}")
    for r in g.head(6).itertuples():
        print(f"  {r.archetype:<28} {r.plays:>6} plays  "
              f"{r.wins / r.plays:.3f} wr")


if __name__ == "__main__":
    main()
