"""CLI for the seeded archipelago v3 (D38-D41): five chains x
(explore + refine), stratified LB panel, two-tier pilot split.

  python scripts/run_seeded.py --run-id seeded_overnight \
      --founders runs/seeded_overnight/founders.json --workers 7
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ptcg.creation.seeded import CHAINS, run_seeded

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="seeded_overnight")
    ap.add_argument("--panel", default="data/panel_lb.json")
    ap.add_argument("--hours", type=float, default=10.5,
                    help="GA wall budget; deep eval runs after")
    ap.add_argument("--wall-hours", type=float, default=11.5,
                    help="total wall budget incl. the deep final eval")
    ap.add_argument("--screen-games", type=int, default=8)
    ap.add_argument("--real-games", type=int, default=6)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--seed", type=int, default=74)
    ap.add_argument("--plateau", type=int, default=12)
    ap.add_argument("--migrate-every", type=int, default=10)
    ap.add_argument("--migrate-top", type=int, default=2)
    ap.add_argument("--floor-wr", type=float, default=0.35)
    ap.add_argument("--explore-pop", type=int, default=24)
    ap.add_argument("--refine-pop", type=int, default=10)
    ap.add_argument("--real-top-explore", type=int, default=6)
    ap.add_argument("--real-top-refine", type=int, default=4)
    ap.add_argument("--founders",
                    default="runs/seeded_overnight/founders.json")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--deep-top", type=int, default=3)
    ap.add_argument("--deep-block", type=int, default=100)
    ap.add_argument("--deep-max", type=int, default=400)
    ap.add_argument("--real-pilot", action="append", default=[],
                    help="chain=kind override, e.g. rainbow-Kanga=kanga; "
                         "repeatable")
    args = ap.parse_args()

    overrides = {}
    for spec in args.real_pilot:
        sk, _, kind = spec.partition("=")
        if sk not in CHAINS or not kind:
            raise SystemExit(f"bad --real-pilot {spec!r}; chains: "
                             f"{sorted(CHAINS)}")
        overrides[sk] = kind

    run_seeded(run_dir=Path("runs") / args.run_id,
               panel_path=Path(args.panel), hours=args.hours,
               wall_hours=args.wall_hours,
               screen_games=args.screen_games, real_games=args.real_games,
               workers=args.workers, seed=args.seed,
               plateau_window=args.plateau,
               migrate_every=args.migrate_every,
               migrate_top=args.migrate_top, floor_wr=args.floor_wr,
               explore_pop=args.explore_pop, refine_pop=args.refine_pop,
               real_top_explore=args.real_top_explore,
               real_top_refine=args.real_top_refine,
               founders=json.load(open(args.founders))
               if args.founders else None,
               resume=args.resume, deep_top=args.deep_top,
               deep_block=args.deep_block, deep_max=args.deep_max,
               pilot_overrides=overrides)
