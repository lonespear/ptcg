"""CLI for the archipelago GA (creation v2). Fitness = play-weighted win
rate vs the mined field, under the injected pilot.

  python scripts/run_archipelago.py --hours 8 --git
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ptcg.creation.archipelago import run_archipelago
from ptcg.creation.pilots import GreedyPilot, JonDayPilot

PILOTS = {
    "greedy": lambda s: GreedyPilot(seed=s),
    "jon": lambda s: JonDayPilot(seed=s, search=False),
    "jon-search": lambda s: JonDayPilot(seed=s, search=True),
}

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
    ap.add_argument("--pilot", choices=list(PILOTS), default="jon")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seed-deck", default=None,
                    help="csv/json 60-card list; rebuild mode seeds matching islands")
    ap.add_argument("--resume", action="store_true",
                    help="continue from runs/<run-id>/state.json; loses at "
                         "most the era in flight when the run was killed")
    ap.add_argument("--git", action="store_true")
    ap.add_argument("--mono-types", default=None,
                    help="comma-separated type names (e.g. Grass,Metal): "
                         "restrict mono islands to these types")
    ap.add_argument("--mono-only", action="store_true",
                    help="monotype sprint: no founding burst, no Phase B; "
                         "reclaimed time funds a deep final panel evaluation")
    ap.add_argument("--wall-hours", type=float, default=None,
                    help="total wall budget incl. deep eval (mono-only)")
    ap.add_argument("--deep-top", type=int, default=3,
                    help="elites per mono set in the deep final evaluation")
    ap.add_argument("--preload-archive", default=None,
                    help="comma-separated archive.jsonl paths: warm the "
                         "fitness cache from earlier runs")
    ap.add_argument("--seed-elites", default=None,
                    help="final_eval.json from a mono sprint: seed matching "
                         "mono islands from its elites")
    ap.add_argument("--v2", action="store_true",
                    help="D38 seeded-archipelago behaviors: tournament-3 "
                         "explore / truncation-4 refine selection, top-2 "
                         "migration, migration-armed patience with the "
                         "below-floor reseed rule")
    ap.add_argument("--explore-pop", type=int, default=None)
    ap.add_argument("--refine-pop", type=int, default=None)
    ap.add_argument("--migrate-every", type=int, default=None)
    ap.add_argument("--migrate-top", type=int, default=1)
    ap.add_argument("--floor-wr", type=float, default=0.35)
    ap.add_argument("--founders", default=None,
                    help="founders.json: {set_key: [[60 card ids], ...]}; "
                         "a spec-Ogerpon key adds the specialty island")
    ap.add_argument("--panel-top-n", type=int, default=8,
                    help="panel entries taken from --priors (use the panel "
                         "file's full length for stratified panels)")
    args = ap.parse_args()

    _json = __import__('json')
    seed_elites = None
    if args.seed_elites:
        seed_elites = _json.load(open(args.seed_elites))["elites"]

    run_archipelago(run_dir=Path("runs") / args.run_id,
                    priors_path=Path(args.priors), hours=args.hours,
                    phase_a_frac=args.phase_a_frac, pop_size=args.pop,
                    games_per_opponent=args.games, seed=args.seed,
                    plateau_window=args.plateau, git_commit=args.git,
                    pilot_factory=PILOTS[args.pilot], workers=args.workers,
                    generalist_name=args.pilot if args.pilot in ('jon','greedy') else 'jon',
                    resume=args.resume,
                    mono_types=args.mono_types.split(",")
                    if args.mono_types else None,
                    mono_only=args.mono_only, wall_hours=args.wall_hours,
                    deep_top=args.deep_top,
                    v2=args.v2, explore_pop=args.explore_pop,
                    refine_pop=args.refine_pop,
                    migrate_every=args.migrate_every,
                    migrate_top=args.migrate_top, floor_wr=args.floor_wr,
                    founders=_json.load(open(args.founders))
                    if args.founders else None,
                    panel_top_n=args.panel_top_n,
                    preload_archives=[Path(x) for x in
                                      args.preload_archive.split(",")]
                    if args.preload_archive else None,
                    seed_elites=seed_elites,
                    seed_deck=_json.load(open(args.seed_deck))
                    if args.seed_deck and args.seed_deck.endswith('.json')
                    else [int(x) for x in open(args.seed_deck).read().split()]
                    if args.seed_deck else None)
