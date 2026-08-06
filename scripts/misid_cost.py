"""When the posterior names the wrong deck, how wrong is it?

Frequency of misidentification is not the same as its cost. The metagame holds
several near-duplicate lists — two Marnie's Grimmsnarl ex builds at 41% and 6%
of play differ by a handful of cards and play identically — so confusing them
costs almost nothing. Confusing Grimmsnarl for Garchomp means the search
defends against the wrong game plan entirely.

This measures, for each wrong top-1 pick: how many of the 60 cards the guessed
list shares with the true one, and whether the archetype label matches. Also
splits by opponent strength, because coverage measured against the field as
played is dominated by the copy-paste middle of the leaderboard — and as our
rating climbs we meet more of the customised tail.

    python scripts/misid_cost.py --episodes 300
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ptcg.episodes import iter_episode_files  # noqa: E402
from ptcg.opponent import DeckPredictor  # noqa: E402


def overlap(a: Counter, b: Counter) -> int:
    """How many of the 60 cards the two lists agree on."""
    return sum(min(a.get(k, 0), b.get(k, 0)) for k in set(a) | set(b))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--strong-wr", type=float, default=0.55,
                    help="agent win rate that counts as a strong opponent")
    args = ap.parse_args()

    pred = DeckPredictor.from_history()
    cand_by_key = {frozenset(c.counts.items()): c for c in pred.candidates}

    # Agent strength, as a stand-in for rating (the replays carry names, and our
    # mined history carries each name's win rate).
    ag = pd.read_csv(ROOT / "data" / "history_agents.csv")
    ag = ag.groupby("agent", as_index=False).agg(games=("games", "sum"),
                                                 wins=("wins", "sum"))
    ag["wr"] = ag["wins"] / ag["games"]
    strength = {r.agent: (r.wr, r.games) for r in ag.itertuples()}

    wrong_overlap: list[int] = []
    same_arch = cross_arch = 0
    by_band: defaultdict[str, list[int]] = defaultdict(list)
    cover_band: defaultdict[str, list[int]] = defaultdict(list)
    episodes = 0

    for path in iter_episode_files():
        if episodes >= args.episodes:
            break
        try:
            ep = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            continue
        try:
            vis = ep["steps"][0][0]["visualize"][0]["action"]
            decks = [Counter(v) for v in vis]
            names = ep["info"]["TeamNames"]
        except (KeyError, IndexError, TypeError):
            continue
        if len(decks) != 2 or len(names) != 2:
            continue
        episodes += 1

        for seat in (0, 1):
            opp_seat = 1 - seat
            truth = decks[opp_seat]
            key = frozenset(truth.items())
            wr, games = strength.get(names[opp_seat], (0.5, 0))
            band = ("strong" if (games >= 50 and wr >= args.strong_wr)
                    else "field")
            cover_band[band].append(1 if key in cand_by_key else 0)
            if key not in cand_by_key:
                continue

            # Sample a mid-early decision, where uncertainty actually lives.
            picked = None
            for step in ep.get("steps") or []:
                try:
                    obs = step[seat]["observation"]
                except (KeyError, IndexError, TypeError):
                    continue
                if not obs or obs.get("select") is None:
                    continue
                turn = (obs.get("current") or {}).get("turn")
                if turn is not None and 1 <= turn <= 3:
                    picked = obs
                    break
            if picked is None:
                continue
            post = pred.posterior(picked)
            if not post:
                continue
            top = post[0][0]
            hit = frozenset(top.counts.items()) == key
            by_band[band].append(1 if hit else 0)
            if hit:
                continue
            ov = overlap(top.counts, truth)
            wrong_overlap.append(ov)
            true_arch = cand_by_key[key].archetype
            if top.archetype == true_arch:
                same_arch += 1
            else:
                cross_arch += 1

    print(f"episodes read: {episodes}\n")
    print("coverage by opponent strength (deck present in our prior):")
    for band, xs in cover_band.items():
        if xs:
            print(f"  {band:<7} n={len(xs):<6} {sum(xs)/len(xs):.3f}")

    print("\ntop-1 accuracy at turns 1-3, by opponent strength:")
    for band, xs in by_band.items():
        if xs:
            print(f"  {band:<7} n={len(xs):<6} {sum(xs)/len(xs):.3f}")

    n = len(wrong_overlap)
    if n:
        wrong_overlap.sort()
        print(f"\nwhen top-1 is WRONG (n={n}), cards shared with the true "
              f"60-card list:")
        print(f"  median {wrong_overlap[n//2]}   "
              f"p25 {wrong_overlap[n//4]}   p75 {wrong_overlap[3*n//4]}   "
              f"min {wrong_overlap[0]}   max {wrong_overlap[-1]}")
        print(f"  same archetype  : {same_arch} ({same_arch/n:.1%}) "
              f"-> near-harmless aliasing")
        print(f"  cross archetype : {cross_arch} ({cross_arch/n:.1%}) "
              f"-> defending the wrong game plan")


if __name__ == "__main__":
    main()
