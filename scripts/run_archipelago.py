"""CLI for the archipelago GA (creation v2). Fitness = play-weighted win
rate vs the mined field, under the injected pilot.

  python scripts/run_archipelago.py --hours 8 --git
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ptcg.creation.archipelago import run_archipelago

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="archipelago")
    ap.add_argument("--priors", default="agent/deck_priors.json")
    ap.add_argument("--hours", type=float, default=8.0)
    ap.add_argument("--phase-a-frac", type=float, default=0.7)
    ap.add_argument("--pop", type=int, default=10)
    ap.add_argument("--games", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plateau", type=int, default=15)
    ap.add_argument("--git", action="store_true")
    args = ap.parse_args()

    run_archipelago(run_dir=Path("runs") / args.run_id,
                    priors_path=Path(args.priors), hours=args.hours,
                    phase_a_frac=args.phase_a_frac, pop_size=args.pop,
                    games_per_opponent=args.games, seed=args.seed,
                    plateau_window=args.plateau, git_commit=args.git)
