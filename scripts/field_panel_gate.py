"""Gate an arm on the FIELD panel — real ladder decks, frozen-copy pilots.

One `scripts/specialist_gate.py` run per cell in `data/analysis/field_panel.json`,
scored under D54 (clean games only) at D51 power (600 games a cell minimum),
then aggregated two ways:

  * POOLED CLEAN — every clean game counts once. The honest sample statistic.
  * FIELD-WEIGHTED — cells reweighted to the archetype's share of the 700-1000
    band of our real ladder episodes. This is the number that is supposed to
    track the ladder, and the number the validation in E16 checks.

Usage, gating a candidate against the v7 baseline:

    python scripts/field_panel_gate.py --arm /path/to/cand/main.py \
        --tag cand --games 600 --workers 3 --seed 96000

    python scripts/field_panel_gate.py --arm agent/main.py \
        --tag v7 --games 600 --workers 3 --seed 96000

    python scripts/field_panel_gate.py --report v7 cand

Baselines must be pinned to a frozen copy or a commit, never to the working
tree (D50's rule). `--arm agent/main.py` is only legitimate when agent/main.py
is the thing under test.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
PANEL = ROOT / "data/analysis/field_panel.json"
RESULTS = ROOT / "data/analysis/field_panel_runs"


def run(args) -> None:
    panel = json.loads(Path(args.panel).read_text())
    RESULTS.mkdir(parents=True, exist_ok=True)
    cells = panel["cells"]
    if args.cells:
        want = set(args.cells.split(","))
        cells = [c for c in cells if c["cell"] in want]
    for c in cells:
        out = RESULTS / f"{args.tag}_{c['cell']}.json"
        if out.exists() and not args.force:
            print(f"skip {c['cell']} (have {out.name})", flush=True)
            continue
        cmd = [PY, str(ROOT / "scripts/specialist_gate.py"),
               "--a", args.arm, "--deck-a", args.deck_a,
               "--specialist", c["pilot"], "--deck-specialist", c["deck_json"],
               "--games", str(args.games), "--workers", str(args.workers),
               "--seed", str(args.seed), "--out", str(out)]
        print(f"\n=== {args.tag} / {c['cell']} ({c['archetype']}) ===",
              flush=True)
        t0 = time.time()
        subprocess.run(cmd, cwd=ROOT, check=True)
        print(f"  cell done in {time.time()-t0:.0f}s", flush=True)


def _load(tag: str, panel: dict) -> list[dict]:
    rows = []
    for c in panel["cells"]:
        f = RESULTS / f"{tag}_{c['cell']}.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        rows.append({**c, "tag": tag,
                     "wr": d["a_win_rate_clean"], "n": d["clean_games"],
                     "wins": d["a_clean"], "se": d["clean_se"],
                     "forfeit_rate": d.get("forfeit_rate"),
                     "b_forfeits": d.get("b_error_forfeits"),
                     "a_forfeits": d.get("a_error_forfeits"),
                     "all_games_wr": d.get("a_win_rate"),
                     "decided": d.get("decided")})
    return rows


def _agg(rows: list[dict]) -> dict:
    n = sum(r["n"] for r in rows)
    w = sum(r["wins"] for r in rows)
    pooled = w / n if n else float("nan")
    tw = sum(r["weight"] for r in rows)
    fw = sum(r["weight"] * r["wr"] for r in rows) / tw if tw else float("nan")
    # Weighted SE: sum of (w_i/W)^2 * p_i(1-p_i)/n_i
    var = sum((r["weight"] / tw) ** 2 * r["wr"] * (1 - r["wr"]) / r["n"]
              for r in rows if r["n"]) if tw else float("nan")
    return {"pooled_clean": round(pooled, 4), "clean_games": n,
            "pooled_se": round(math.sqrt(pooled * (1 - pooled) / n), 4) if n else None,
            "field_weighted": round(fw, 4),
            "field_weighted_se": round(math.sqrt(var), 4),
            "weight_covered": round(tw, 4),
            "forfeit_games": sum(r["decided"] - r["n"] for r in rows),
            "decided": sum(r["decided"] for r in rows)}


def report(args) -> None:
    panel = json.loads(Path(args.panel).read_text())
    allrows = {t: _load(t, panel) for t in args.report}
    print(f"\nFIELD PANEL — {panel['n_field_episodes']} real ladder episodes, "
          f"{len(panel['cells'])} cells, weights from the 700-1000 band\n")
    hdr = f"{'cell':17s} {'archetype':26s} {'wt':>6s}"
    for t in args.report:
        hdr += f" {t+' wr':>12s} {'n':>6s}"
    print(hdr)
    print("-" * len(hdr))
    for c in panel["cells"]:
        line = f"{c['cell']:17s} {c['archetype'][:26]:26s} {c['weight']:6.3f}"
        for t in args.report:
            r = next((x for x in allrows[t] if x["cell"] == c["cell"]), None)
            line += (f" {r['wr']:12.4f} {r['n']:6d}" if r
                     else f" {'-':>12s} {'-':>6s}")
        print(line)
    print()
    summary = {}
    for t in args.report:
        if not allrows[t]:
            continue
        a = _agg(allrows[t])
        summary[t] = a
        print(f"{t}: pooled clean {a['pooled_clean']:.4f} "
              f"+- {a['pooled_se']:.4f} over {a['clean_games']} games   |   "
              f"FIELD-WEIGHTED {a['field_weighted']:.4f} "
              f"+- {a['field_weighted_se']:.4f} "
              f"(covers {a['weight_covered']:.1%} of the band)")
        print(f"    forfeits: {a['forfeit_games']}/{a['decided']} decided "
              f"({a['forfeit_games']/max(a['decided'],1):.2%}) — the old "
              f"external panel ran 24.9% (D54)")
    if len(args.report) == 2:
        t0, t1 = args.report
        if t0 in summary and t1 in summary:
            d = summary[t1]["field_weighted"] - summary[t0]["field_weighted"]
            se = math.sqrt(summary[t0]["field_weighted_se"] ** 2
                           + summary[t1]["field_weighted_se"] ** 2)
            print(f"\n  {t1} - {t0} field-weighted: {d:+.4f} "
                  f"({d*100:+.1f} pt), SE {se:.4f}, z = {d/se:+.2f}")
    if args.out:
        Path(args.out).write_text(json.dumps(
            {"summary": summary,
             "cells": {t: allrows[t] for t in args.report}}, indent=1))
        print(f"\nwrote {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=str(PANEL))
    ap.add_argument("--arm", default=None, help="path to the arm's main.py")
    ap.add_argument("--deck-a", default=str(ROOT / "agent/deck.csv"))
    ap.add_argument("--tag", default=None, help="name this arm's result files")
    ap.add_argument("--games", type=int, default=600)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=96000)
    ap.add_argument("--cells", default=None, help="comma list, default all")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report", nargs="*", default=None,
                    help="tags to tabulate instead of running")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.report:
        report(args)
    else:
        if not (args.arm and args.tag):
            ap.error("--arm and --tag are required to run cells")
        run(args)


if __name__ == "__main__":
    main()
