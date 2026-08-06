"""Measure the opponent model against ground truth, offline.

Every mined replay contains *both* players' real decklists, so the opponent
model can be scored directly instead of inferred through game outcomes — which
is the noisiest possible instrument and costs SPRT games we do not have.

For each decision point in a replay we feed the acting player's own view into
the posterior and compare against what the opponent was actually playing.
Reports:

  * top-1 accuracy by turn — how fast inference converges
  * calibration — when the posterior says 0.7, is it right 70% of the time?
  * coverage — how often the true deck is not in our candidate set at all,
    which is an error no amount of clever inference can fix
  * how long genuine uncertainty lasts, which is what decides whether averaging
    over several determinizations can pay for itself

    python scripts/validate_posterior.py --episodes 400
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ptcg.episodes import iter_episode_files  # noqa: E402
from ptcg.opponent import DeckPredictor  # noqa: E402


def true_decks(episode: dict) -> list[Counter] | None:
    try:
        vis = episode["steps"][0][0]["visualize"][0]["action"]
    except (KeyError, IndexError, TypeError):
        return None
    if not isinstance(vis, list) or len(vis) != 2:
        return None
    if not all(isinstance(v, list) for v in vis):
        return None
    return [Counter(v) for v in vis]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--top-k", type=int, default=3)
    args = ap.parse_args()

    pred = DeckPredictor.from_history()
    print(f"{len(pred.candidates)} candidate decklists in the prior\n")
    cand_keys = {frozenset(c.counts.items()): i
                 for i, c in enumerate(pred.candidates)}

    by_turn_hits: defaultdict[int, int] = defaultdict(int)
    by_turn_n: defaultdict[int, int] = defaultdict(int)
    by_turn_topk: defaultdict[int, int] = defaultdict(int)
    by_turn_mass: defaultdict[int, float] = defaultdict(float)
    calib_hits: defaultdict[int, int] = defaultdict(int)
    calib_n: defaultdict[int, int] = defaultdict(int)
    covered = uncovered = 0
    episodes = 0

    for path in iter_episode_files():
        if episodes >= args.episodes:
            break
        try:
            ep = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            continue
        decks = true_decks(ep)
        if decks is None:
            continue
        episodes += 1

        truth_key = [frozenset(d.items()) for d in decks]
        in_prior = [k in cand_keys for k in truth_key]
        covered += sum(in_prior)
        uncovered += sum(not x for x in in_prior)

        seen_turn: set[tuple[int, int]] = set()
        for step in ep.get("steps") or []:
            for seat in (0, 1):
                try:
                    obs = step[seat]["observation"]
                except (KeyError, IndexError, TypeError):
                    continue
                if not obs or obs.get("select") is None:
                    continue
                cur = obs.get("current") or {}
                turn = cur.get("turn")
                if turn is None or (seat, turn) in seen_turn:
                    continue
                # The opponent of `seat` is the deck we are trying to name.
                opp_seat = 1 - seat
                if not in_prior[opp_seat]:
                    continue
                seen_turn.add((seat, turn))

                post = pred.posterior(obs)
                if not post:
                    continue
                target = truth_key[opp_seat]
                top = frozenset(post[0][0].counts.items())
                hit = top == target
                by_turn_n[turn] += 1
                by_turn_hits[turn] += hit
                by_turn_topk[turn] += any(
                    frozenset(c.counts.items()) == target
                    for c, _ in post[:args.top_k])
                mass = sum(w for c, w in post
                           if frozenset(c.counts.items()) == target)
                by_turn_mass[turn] += mass
                bucket = min(int(post[0][1] * 10), 9)
                calib_n[bucket] += 1
                calib_hits[bucket] += hit

    print(f"episodes read: {episodes}")
    tot = covered + uncovered
    if tot:
        print(f"coverage: the true deck is in our prior for "
              f"{covered}/{tot} player-slots ({covered/tot:.1%})")
        print(f"  -> {uncovered/tot:.1%} of opponents are decks we cannot name "
              f"at all, a floor on inference error\n")

    print("top-1 accuracy by turn (over slots whose deck IS in the prior):")
    print(f"{'turn':>5}{'n':>8}{'top1':>8}{'top3':>8}{'mass@truth':>12}")
    for turn in sorted(by_turn_n):
        n = by_turn_n[turn]
        if n < 20 or turn > 14:
            continue
        print(f"{turn:>5}{n:>8}{by_turn_hits[turn]/n:>8.2f}"
              f"{by_turn_topk[turn]/n:>8.2f}{by_turn_mass[turn]/n:>12.2f}")

    print("\ncalibration (posterior weight on its own top pick vs how often "
          "that pick is right):")
    print(f"{'claimed':>10}{'n':>8}{'actual':>9}")
    for b in sorted(calib_n):
        n = calib_n[b]
        if n < 20:
            continue
        print(f"{b/10:.1f}-{b/10+0.1:.1f}{n:>8}{calib_hits[b]/n:>9.2f}")


if __name__ == "__main__":
    main()
