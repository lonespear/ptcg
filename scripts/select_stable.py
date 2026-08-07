"""Build the D30 deck stable from overnight archipelago archives.

Merges one or more runs' archive.jsonl files (laptop + Sebastian), keeps
legal decks above the fitness floor, greedily selects ~100 members with
decorrelated weakness vectors (ptcg/creation/portfolio.py), and writes
stable.json: per member the 60-card list, D18 fitness components, matchup
profile against the named panel, and (optionally, --curves) the goldfish
threat curve. Fitted per-deck weight vectors are the morning tuning step
(ptcg/creation/tuning.py), not this script.

  python scripts/select_stable.py runs/stable_r1/archive.jsonl \
      [runs/stable_r1_seb/archive.jsonl ...] \
      --panel runs/stable_r1/panel.json --size 100 --floor 0.15 \
      --out runs/stable_r1/stable.json [--curves]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ptcg.creation.portfolio import load_archive, select_stable  # noqa: E402
from ptcg.creation.validator import validate                     # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("archives", nargs="+", help="archive.jsonl paths")
    ap.add_argument("--panel", default=None,
                    help="panel.json of the run, for profile column names")
    ap.add_argument("--size", type=int, default=100)
    ap.add_argument("--floor", type=float, default=None,
                    help="absolute D18 fitness floor")
    ap.add_argument("--top-frac", type=float, default=None,
                    help="alternative floor: keep this fraction by fitness")
    ap.add_argument("--gamma", type=float, default=0.1,
                    help="decorrelation pressure (fitness units per unit "
                         "of max weakness correlation)")
    ap.add_argument("--curves", action="store_true",
                    help="attach goldfish threat curves (50 games/member)")
    ap.add_argument("--out", default="stable.json")
    args = ap.parse_args()

    cands = load_archive([Path(p) for p in args.archives])
    legal = [c for c in cands if validate(c["deck"]).legal]
    print(f"{len(cands)} archived decks, {len(legal)} legal", flush=True)

    floor = args.floor
    if floor is None and args.top_frac:
        ranked = sorted((c["fitness"] for c in legal), reverse=True)
        floor = ranked[max(0, int(len(ranked) * args.top_frac) - 1)]
        print(f"top-frac {args.top_frac} -> floor {floor:.4f}", flush=True)

    stable = select_stable(legal, size=args.size, fitness_floor=floor,
                           gamma=args.gamma)

    panel_names = None
    if args.panel:
        panel = json.loads(Path(args.panel).read_text())
        panel_names = [{"name": e["name"], "weight": e["weight"],
                        "pilot": e.get("pilot", {}).get("specialist",
                                                        "generalist")}
                       for e in panel]
    stable["panel"] = panel_names
    stable["weights"] = None    # per-deck fitted vectors: morning step (D30)

    if args.curves and stable["members"]:
        from ptcg.creation.goldfish import profile as goldfish_profile
        for m in stable["members"]:
            try:
                m["threat_curve"] = goldfish_profile(m["deck"], n_games=50)
            except Exception as exc:  # noqa: BLE001 — curve is optional cargo
                m["threat_curve"] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"curve {m['rank'] + 1}/{len(stable['members'])}",
                  flush=True)

    out = Path(args.out)
    out.write_text(json.dumps(stable, indent=1))
    s = stable["summary"]
    print(f"stable -> {out}: n={s['n']} fitness {s['fitness_min']:.3f}.."
          f"{s['fitness_max']:.3f} (mean {s['fitness_mean']:.3f}), "
          f"pairwise weakness corr mean={s['pairwise_corr_mean']} "
          f"max={s['pairwise_corr_max']}", flush=True)
