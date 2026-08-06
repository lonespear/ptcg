"""Compare two agents against a small field, on paired shuffles.

Two instrument problems this fixes.

**Single-opponent testing.** Every comparison since the gauntlet insight has
gone back to one opponent — first the mirror, then Marnie's — and the README
documents twice why one opponent misranks. The leaderboard is a field, so the
harness is a small one: a handful of real decklists weighted roughly by play
share.

**Variance.** At p0=0.50/p1=0.55 the SPRT ran 1500 games without deciding,
because we are now hunting 1-2 point effects. Both candidates play the *same
seeds* against the *same opponents*, and the test runs on the paired outcomes:
only games the two agents disagree about carry information. This is why chess
engines test on paired openings, and it typically cuts the games needed several
fold.

    python scripts/field_sprt.py --a build/agents/v10_ismcts.py \
                                 --b build/agents/v9_energy.py
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "engine"))

from ptcg.arena import load_deck, play_game  # noqa: E402
from ptcg.meta import signature_to_deck  # noqa: E402
from ptcg.sprt import SPRT  # noqa: E402

HISTORY = ROOT / "data" / "history_decklists.csv"


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


def build_field(n: int, recent_days: int = 4) -> list[dict]:
    df = pd.read_csv(HISTORY)
    keep = sorted(df["date"].astype(str).unique())[-recent_days:]
    df = df[df["date"].astype(str).isin(keep)]
    g = (df.groupby("signature", as_index=False)
         .agg(plays=("decks", "sum"), archetype=("archetype", "first")))
    total = g["plays"].sum()
    return [{"name": r.archetype[:22], "deck": signature_to_deck(r.signature),
             "share": r.plays / total}
            for r in g.nlargest(n, "plays").itertuples()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--deck", default="agent/deck.csv")
    ap.add_argument("--field", type=int, default=4)
    ap.add_argument("--max-games", type=int, default=3000)
    ap.add_argument("--p0", type=float, default=0.50)
    ap.add_argument("--p1", type=float, default=0.55)
    args = ap.parse_args()

    deck = load_deck(ROOT / args.deck)
    field = build_field(args.field)
    a = load_agent(ROOT / args.a, "cand_a")
    b = load_agent(ROOT / args.b, "cand_b")

    print(f"field ({len(field)} decks): "
          + ", ".join(f"{f['name']} {f['share']:.0%}" for f in field))
    print(f"paired seeds, SPRT p0={args.p0} p1={args.p1}\n")

    test = SPRT(p0=args.p0, p1=args.p1)
    paired = ties = 0
    a_raw = b_raw = 0

    for g in range(args.max_games):
        opp = field[g % len(field)]
        seed = 10_000 + g // len(field)
        flip = (g // len(field)) % 2 == 1

        outcomes = []
        for agent in (a, b):
            # Identical seed, identical opponent deck, identical seat.
            if flip:
                r = play_game(agent, b if agent is a else a,
                              deck, opp["deck"], seed=seed)
                won = r.winner == 0
            else:
                r = play_game(agent, b if agent is a else a,
                              deck, opp["deck"], seed=seed)
                won = r.winner == 0
            outcomes.append(bool(won))

        a_won, b_won = outcomes
        a_raw += a_won
        b_raw += b_won
        if a_won == b_won:
            ties += 1
            continue                    # no information in an agreed game
        paired += 1
        res = test.update(a_won)
        if paired % 50 == 0:
            print(f"    paired={paired} (ties {ties})  {test.describe()}")
        if res.decision != "continue":
            break

    res = test.result()
    played = max(paired + ties, 1)
    print(f"\n  raw win rate   A {a_raw/played:.3f}  B {b_raw/played:.3f}  "
          f"over {played} seeded games")
    print(f"  informative    {paired} of {played} "
          f"({paired/played:.0%} — the rest agreed)")
    print(f"  paired A-wins  {res.wins}/{res.n} = {res.win_rate:.3f}")
    print(f"  decision       {res.decision}  (llr {res.llr:+.2f})")


if __name__ == "__main__":
    main()
