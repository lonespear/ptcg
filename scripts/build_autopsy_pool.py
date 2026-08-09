"""Build the opponent pool from games we actually played, not from global mining.

The replay-mined field describes the *population*: all 6,500 teams, dominated by
the copy-paste middle. Matchmaking does not draw from it. It pairs us by rating,
and the two disagree violently — Archaludon is **0.31% of mined replays and 14%
of our own games**, a 45x gap, and it happens to run the Fire attacker our Grass
deck is weak to.

Every gauntlet score and GA fitness built on the mined weights was therefore
optimising against a field we do not play. This rebuilds the pool from
`data/analysis/ladder_autopsy.json`, which carries the actual opposing decklist
and both ratings for every episode we have.

Because rating decides who we meet, the pool is band-filterable: optimise for
the band you are climbing *into*, not the one you are leaving.

    python scripts/build_autopsy_pool.py                    # all bands
    python scripts/build_autopsy_pool.py --band 700-900     # where we live now
    python scripts/build_autopsy_pool.py --min-rating 800   # where we are going
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

AUTOPSY = ROOT / "data" / "analysis" / "ladder_autopsy.json"
OUT = ROOT / "data" / "analysis" / "autopsy_pool.json"


def deck_from_counts(counts: dict) -> list[int]:
    out: list[int] = []
    for cid, n in counts.items():
        out.extend([int(cid)] * int(n))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", default=None,
                    help="keep only opponents in a rating band, e.g. 700-900")
    ap.add_argument("--min-rating", type=float, default=None)
    ap.add_argument("--max-rating", type=float, default=None)
    ap.add_argument("--top", type=int, default=8,
                    help="how many distinct lists to keep in the pool")
    ap.add_argument("--min-encounters", type=int, default=2)
    args = ap.parse_args()

    if not AUTOPSY.exists():
        sys.exit(f"{AUTOPSY} not found — it comes from Austin's branch")
    recs = json.loads(AUTOPSY.read_text(encoding="utf-8"))["records"]

    lo, hi = args.min_rating, args.max_rating
    if args.band:
        a, _, b = args.band.partition("-")
        lo, hi = float(a), float(b)

    kept = []
    for r in recs:
        s = r.get("opp_score")
        if lo is not None and (s is None or s < lo):
            continue
        if hi is not None and (s is None or s > hi):
            continue
        if not r.get("opp_deck"):
            continue
        kept.append(r)

    label = args.band or (f">={lo}" if lo else "all bands")
    print(f"{len(kept)} of {len(recs)} episodes in {label}")
    if not kept:
        sys.exit("no episodes in that band")

    # One entry per exact decklist we actually met.
    by_sig: dict = defaultdict(lambda: {"n": 0, "wins": 0, "arch": None,
                                        "deck": None, "ratings": []})
    for r in kept:
        e = by_sig[str(r["opp_deck_sig"])]
        e["n"] += 1
        e["wins"] += 1 if r.get("our_reward") == 1 else 0
        e["arch"] = e["arch"] or r.get("archetype")
        e["deck"] = e["deck"] or deck_from_counts(r["opp_deck"])
        if r.get("opp_score") is not None:
            e["ratings"].append(r["opp_score"])

    entries = [e for e in by_sig.values()
               if e["n"] >= args.min_encounters and e["deck"]
               and len(e["deck"]) == 60]
    entries.sort(key=lambda e: -e["n"])
    entries = entries[:args.top]
    total = sum(e["n"] for e in entries)

    print(f"\npool: {len(entries)} lists covering {total} of {len(kept)} "
          f"episodes ({total/len(kept):.0%})\n")
    print(f"{'archetype':<38}{'met':>5}{'share':>8}{'our wr':>9}{'opp rating':>12}")
    pool = []
    for e in entries:
        share = e["n"] / total
        wr = e["wins"] / e["n"]
        avg = sum(e["ratings"]) / len(e["ratings"]) if e["ratings"] else 0.0
        print(f"{(e['arch'] or '?')[:37]:<38}{e['n']:>5}{share:>8.1%}"
              f"{wr:>9.3f}{avg:>12.0f}")
        pool.append({"archetype": e["arch"], "deck": e["deck"],
                     "encounters": e["n"], "share": share,
                     "our_win_rate": wr, "mean_opp_rating": avg})

    OUT.write_text(json.dumps({"band": label, "episodes": len(kept),
                               "pool": pool}, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(ROOT)}")

    # The comparison that motivates all of this.
    try:
        import pandas as pd
        a = pd.read_csv(ROOT / "data" / "history_archetypes.csv")
        mined = a.groupby("archetype").decks.sum()
        mined = mined / mined.sum()
        print("\nmined share vs what we actually meet:")
        print(f"  {'archetype':<34}{'mined':>9}{'faced':>9}{'ratio':>8}")
        for e in entries[:6]:
            name = (e["arch"] or "").split("/")[0].split("-")[0]
            m = mined[mined.index.str.contains(name, case=False, na=False)].sum()
            f = e["n"] / total
            ratio = (f / m) if m else float("inf")
            print(f"  {(e['arch'] or '?')[:33]:<34}{m:>9.2%}{f:>9.1%}"
                  f"{ratio:>8.1f}x")
    except Exception as exc:
        print(f"(mined comparison skipped: {type(exc).__name__})")


if __name__ == "__main__":
    main()
