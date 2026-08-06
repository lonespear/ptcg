"""CLI for the deck-creation GA (ptcg/creation/).

Fitness against the real mined field, under the baseline pilot:
  python scripts/run_deck_ga.py --mode mono --hours 8 \
      --panel-from agent/deck_priors.json

`--panel-from` accepts GA checkpoint latest.json files and/or a
deck_priors.json (top mined lists join the panel, play-weighted top 4).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ptcg.creation.ga import run


def expand_priors(paths: list[str] | None) -> list[str] | None:
    """deck_priors.json entries become temp checkpoint-shaped panels."""
    if not paths:
        return paths
    out = []
    for path in paths:
        p = Path(path)
        try:
            d = json.loads(p.read_text())
        except Exception:
            out.append(path)
            continue
        if "decks" in d and "total" in d:  # deck_priors schema
            top = sorted(d["decks"], key=lambda e: -e["p"])[:4]
            shaped = {"islands": {
                e["a"]: {"best": e["w"] / e["p"],
                         "best_deck": [int(c) for c, n in e["c"].items()
                                       for _ in range(n)]}
                for e in top}}
            tmp = p.parent / f".panel_{p.stem}.json"
            tmp.write_text(json.dumps(shaped))
            out.append(str(tmp))
        else:
            out.append(path)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["mono", "multi"], required=True)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--pop", type=int, default=12)
    ap.add_argument("--games", type=int, default=24)
    ap.add_argument("--eras", type=int, default=10_000)
    ap.add_argument("--hours", type=float, default=8.0)
    ap.add_argument("--migrate-every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--panel-seed", type=int, default=42)
    ap.add_argument("--git", action="store_true")
    ap.add_argument("--panel-from", nargs="*", default=None)
    args = ap.parse_args()

    run_id = args.run_id or f"{args.mode}_ga"
    run(mode=args.mode, run_dir=Path("runs") / run_id, pop_size=args.pop,
        games_per_opponent=args.games, max_eras=args.eras, hours=args.hours,
        migrate_every=args.migrate_every, seed=args.seed,
        panel_seed=args.panel_seed, git_commit=args.git,
        panel_from=expand_priors(args.panel_from))
