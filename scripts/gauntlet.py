"""Score candidate decks against the *actual* metagame, weighted by how often
each opponent deck is really played.

Two benchmarks have now misled us in opposite directions:

  1. Our own weak agent said the vendor sample deck was best (0.883). Live: 545.
  2. The official reference agent said Mega Lucario ex was better (0.276 vs
     0.194). Live: 306 — worse.

The second failure is a monoculture: the reference agent plays one archetype, so
"beats the reference agent" measures one matchup, not the field. The real field
is 34.5% Marnie's Grimmsnarl ex, and we never tested against it once.

This builds the opponent pool from the mined decklists, weights each by its true
play share, and reports a field-weighted win rate.

    python scripts/gauntlet.py --games 60
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "engine"))

from ptcg.arena import load_deck, match  # noqa: E402
from ptcg.meta import signature_to_deck  # noqa: E402

HISTORY = ROOT / "data" / "history_decklists.csv"


def load_agent(path: Path, name: str):
    import os
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


def build_field(n: int, recent_days: int = 0) -> list[dict]:
    """The most-played decklists, with their real share of the metagame.

    Share is computed over the most recent `recent_days` only when given: the
    field moves fast. Marnie's Grimmsnarl ex ran at 63% of all decks in late
    July and 30% by 2026-08-05, so pooling the whole history overstates it by
    a factor of two and mis-weights every matchup that involves it.
    """
    df = pd.read_csv(HISTORY)
    if recent_days:
        keep = sorted(df["date"].astype(str).unique())[-recent_days:]
        df = df[df["date"].astype(str).isin(keep)]
        print(f"field weighted over {len(keep)} most recent days: "
              f"{keep[0]} .. {keep[-1]}")
    g = (df.groupby("signature", as_index=False)
         .agg(plays=("decks", "sum"), wins=("wins", "sum"),
              archetype=("archetype", "first")))
    total = g["plays"].sum()
    top = g.nlargest(n, "plays")
    return [{"name": f"{r.archetype[:24]}", "deck": signature_to_deck(r.signature),
             "share": r.plays / total, "plays": int(r.plays),
             "wr": r.wins / r.plays}
            for r in top.itertuples()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--field", type=int, default=6)
    ap.add_argument("--agent", default="build/agents/v4.py")
    ap.add_argument("--recent-days", type=int, default=4,
                    help="weight the field over the N most recent days (0 = all)")
    args = ap.parse_args()

    field = build_field(args.field, args.recent_days)
    covered = sum(f["share"] for f in field)
    print(f"field: {len(field)} decks covering {covered:.1%} of real play\n")
    for f in field:
        print(f"  {f['name']:<26} share={f['share']:.1%}  "
              f"plays={f['plays']:<6} real wr={f['wr']:.3f}")

    candidates = {
        "sample": ROOT / "build" / "sample_deck.csv",
        "lucario_ref": ROOT / "build" / "agents" / "reference_deck.csv",
    }
    decks = {k: load_deck(v) for k, v in candidates.items() if v.exists()}
    # The field's own top decks are candidates too — often the right answer is
    # simply to play what wins.
    for f in field:
        decks[f"field:{f['name'][:18]}"] = f["deck"]

    pilot = load_agent(ROOT / args.agent, "pilot")

    print(f"\n{len(decks)} candidates x {len(field)} opponents x {args.games} "
          f"games = {len(decks)*len(field)*args.games} games\n")

    rows = []
    for name, deck in decks.items():
        weighted = 0.0
        detail = []
        for f in field:
            res = match(pilot, pilot, deck, f["deck"],
                        games=args.games, seed0=0)
            wr = res["agent0_win_rate"]
            if wr != wr:      # NaN
                wr = 0.5
            weighted += wr * f["share"]
            detail.append(f"{wr:.2f}")
        weighted /= covered
        rows.append((weighted, name, detail))
        print(f"  {name:<28} field-weighted {weighted:.3f}   "
              f"[{' '.join(detail)}]")

    rows.sort(reverse=True)
    print("\n=== ranking (field-weighted win rate) ===")
    for w, name, _ in rows:
        print(f"  {name:<28} {w:.3f}")
    print(f"\nbest: {rows[0][1]}")
    print("\nCaveat: our own agent pilots both sides, so the field is played "
          "worse than real opponents play it. This ranks decks under *our* "
          "piloting, which is the right question for what we submit, but the "
          "absolute win rates are optimistic.")


if __name__ == "__main__":
    main()
