"""How well does our agent pilot each opponent's deck? And does that predict
how wrong our local matchup numbers are?

The blocking problem: every local matchup drives BOTH sides with our own agent,
so the harness reports 0.892 where the ladder says 0.519 — and the error scales
with how much skill the opponent's deck needs (+0.25 on Grimmsnarl, +0.78 on
Mega Lucario). We cannot fix that by choosing better decks, only by having a
better driver, and we do not have one.

But we can measure the driver, using the instrument that has been most reliable
in this project: real replays as ground truth. For every decision a real player
made with deck X, ask our agent what it would have done. The agreement rate is a
direct, submission-free measure of how well we pilot X.

If agreement predicts the local-vs-ladder gap across the pool, then it is a
**correction factor** — and more usefully, an advance warning that a given
local matchup should not be trusted.

    python scripts/pilot_gap.py --episodes 250
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "engine"))

from ptcg.episodes import iter_episode_files  # noqa: E402

POOL = ROOT / "data" / "analysis" / "autopsy_pool.json"


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


def deck_key(counts: Counter) -> frozenset:
    return frozenset(counts.items())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="agent/main.py")
    ap.add_argument("--episodes", type=int, default=250)
    ap.add_argument("--max-decisions", type=int, default=60,
                    help="cap per episode; search is not free")
    args = ap.parse_args()

    pool = json.loads(POOL.read_text(encoding="utf-8"))["pool"] \
        if POOL.exists() else []
    # Match a replay deck to a pool entry by exact composition.
    want = {deck_key(Counter(p["deck"])): p for p in pool}
    print(f"{len(want)} pool decks to look for\n")

    agent = load_agent(ROOT / args.agent, "gap_agent")

    agree: defaultdict[str, int] = defaultdict(int)
    total: defaultdict[str, int] = defaultdict(int)
    seen_eps = matched = 0

    for path in iter_episode_files():
        if seen_eps >= args.episodes:
            break
        try:
            ep = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            continue
        seen_eps += 1
        try:
            decks = ep["steps"][0][0]["visualize"][0]["action"]
        except (KeyError, IndexError, TypeError):
            continue
        if not isinstance(decks, list) or len(decks) != 2:
            continue

        for seat in (0, 1):
            k = deck_key(Counter(decks[seat]))
            entry = want.get(k)
            if entry is None:
                continue
            matched += 1
            label = entry["archetype"][:30]
            n = 0
            for step in ep.get("steps") or []:
                if n >= args.max_decisions:
                    break
                try:
                    cell = step[seat]
                    obs = cell.get("observation") or {}
                    act = cell.get("action")
                except (KeyError, IndexError, TypeError):
                    continue
                sel = obs.get("select")
                if not sel or not sel.get("option") or act is None:
                    continue
                if not isinstance(act, list) or not act:
                    continue
                try:
                    ours = agent(obs)
                except Exception:
                    continue
                total[label] += 1
                agree[label] += int(sorted(ours) == sorted(act))
                n += 1

    print(f"read {seen_eps} episodes, matched {matched} player-slots to the pool\n")
    if not total:
        print("No pool deck appeared in the episodes on disk.")
        print("The pool comes from 08-07+ ladder games; the episodes cached "
              "here are 08-04. Re-run scripts/mine_day.py --keep for a day that "
              "overlaps, or widen the match from exact list to archetype.")
        return

    print(f"{'deck we are piloting':<32}{'decisions':>11}{'agreement':>11}")
    rows = []
    for label in sorted(total, key=lambda x: -total[x]):
        a, t = agree[label], total[label]
        print(f"{label:<32}{t:>11}{a/t:>11.3f}")
        rows.append((label, a / t))

    # Does piloting skill predict how wrong the local number is?
    by_arch = {p["archetype"][:30]: p for p in pool}
    local = {"Grimmsnarl/Munkidori": 1.000, "Dudunsparce-Alakazam": 0.840,
             "Archaludon-Duraludon-Cinderace": 0.860, "Mega Lucario": 0.780}
    pairs = []
    for label, ag in rows:
        p = by_arch.get(label)
        loc = local.get(label)
        if p and loc is not None:
            pairs.append((ag, loc - p["our_win_rate"]))
    if len(pairs) >= 3:
        n = len(pairs)
        mx = sum(x for x, _ in pairs) / n
        my = sum(y for _, y in pairs) / n
        cov = sum((x - mx) * (y - my) for x, y in pairs)
        vx = sum((x - mx) ** 2 for x, _ in pairs) ** 0.5
        vy = sum((y - my) ** 2 for _, y in pairs) ** 0.5
        r = cov / (vx * vy) if vx and vy else 0.0
        print(f"\ncorrelation(agreement, local-minus-ladder gap) = {r:+.3f} "
              f"over {n} decks")
        print("A strong negative r means: the worse we pilot a deck, the more "
              "our local number overstates that matchup — which makes "
              "agreement a usable trust score for any future local result.")


if __name__ == "__main__":
    main()
