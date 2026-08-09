"""Validate the field panel against the ladder results it is supposed to predict.

A panel is only worth gating on if it reproduces what we already know. This
compares, per archetype, the panel's cell win rate against our REAL win rate
against that archetype on the ladder, and reports:

  * LEVEL — field-weighted panel win rate vs the live win rate over the same
    episodes. A panel that says we win 85% when the ladder says 47% is broken
    no matter how pretty its cells are.
  * SHAPE — Spearman rank correlation between cell win rate and live win rate.
    Gating compares arms, so a constant level offset is survivable and a
    rank inversion is not.
  * CONTAMINATION — forfeit rate, the D54 defect the old panel had at 24.9%.

    python scripts/field_panel_validate.py --tag v7
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data/analysis/field_panel_runs"


def spearman(xs, ys) -> float:
    def rank(v):
        s = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(s):
            j = i
            while j + 1 < len(s) and v[s[j + 1]] == v[s[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[s[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx)
                    * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v7")
    ap.add_argument("--panel", default=str(ROOT / "data/analysis/field_panel.json"))
    ap.add_argument("--corpus", default=str(ROOT / "data/analysis/field_decks.json"))
    ap.add_argument("--live-subs", default=None,
                    help="restrict live comparison to these of our submissions")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    panel = json.loads(Path(args.panel).read_text())
    corpus = json.loads(Path(args.corpus).read_text())
    eps = corpus["episodes"]
    if args.live_subs:
        keep = {int(s) for s in args.live_subs.split(",")}
        eps = [e for e in eps if e["our_sub"] in keep]

    live_w, live_n = Counter(), Counter()
    for e in eps:
        live_n[e["opp_archetype"]] += 1
        live_w[e["opp_archetype"]] += 1 if e["won"] else 0

    rows = []
    for c in panel["cells"]:
        f = RESULTS / f"{args.tag}_{c['cell']}.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        a = c["archetype"]
        rows.append({
            "cell": c["cell"], "archetype": a, "weight": c["weight"],
            "panel_wr": d["a_win_rate_clean"], "panel_n": d["clean_games"],
            "panel_se": d["clean_se"],
            "forfeit_rate": d.get("forfeit_rate", 0.0),
            "live_wr": (live_w[a] / live_n[a]) if live_n[a] else None,
            "live_n": live_n[a],
        })
    if not rows:
        raise SystemExit(f"no results for tag {args.tag} in {RESULTS}")

    tw = sum(r["weight"] for r in rows)
    fw = sum(r["weight"] * r["panel_wr"] for r in rows) / tw
    fw_se = math.sqrt(sum((r["weight"] / tw) ** 2 * r["panel_wr"]
                          * (1 - r["panel_wr"]) / r["panel_n"] for r in rows))
    pooled_n = sum(r["panel_n"] for r in rows)
    pooled = sum(r["panel_wr"] * r["panel_n"] for r in rows) / pooled_n
    forf = sum(r["forfeit_rate"] * r["panel_n"] for r in rows) / pooled_n

    live_all = sum(1 for e in eps if e["won"]) / len(eps)
    # Live rate over just the archetypes the panel covers, weighted the same way.
    cover = {r["archetype"] for r in rows}
    ce = [e for e in eps if e["opp_archetype"] in cover]
    live_cov = sum(1 for e in ce if e["won"]) / len(ce) if ce else float("nan")
    live_w_match = (sum(r["weight"] * r["live_wr"] for r in rows if r["live_wr"] is not None)
                    / sum(r["weight"] for r in rows if r["live_wr"] is not None))

    both = [r for r in rows if r["live_wr"] is not None and r["live_n"] >= 3]
    rho = spearman([r["panel_wr"] for r in both], [r["live_wr"] for r in both])

    print(f"\nFIELD PANEL VALIDATION — arm {args.tag}, "
          f"{len(eps)} live episodes\n")
    print(f"{'cell':17s} {'archetype':26s} {'wt':>5s} {'panel':>8s} {'n':>6s} "
          f"{'live':>8s} {'n':>4s} {'delta':>7s}")
    print("-" * 92)
    for r in sorted(rows, key=lambda x: -x["weight"]):
        lv = f"{r['live_wr']:.3f}" if r["live_wr"] is not None else "-"
        dl = (f"{r['panel_wr']-r['live_wr']:+.3f}"
              if r["live_wr"] is not None else "-")
        print(f"{r['cell']:17s} {r['archetype'][:26]:26s} {r['weight']:5.3f} "
              f"{r['panel_wr']:8.4f} {r['panel_n']:6d} {lv:>8s} "
              f"{r['live_n']:4d} {dl:>7s}")

    print(f"\nLEVEL")
    print(f"  panel field-weighted   {fw:.4f} +- {fw_se:.4f}")
    print(f"  panel pooled clean     {pooled:.4f}  (n={pooled_n})")
    print(f"  live, same weighting   {live_w_match:.4f}")
    print(f"  live, covered archetypes {live_cov:.4f}  (n={len(ce)})")
    print(f"  live, all episodes     {live_all:.4f}  (n={len(eps)})")
    print(f"  LEVEL OFFSET (panel - live, same weighting) "
          f"{fw - live_w_match:+.4f}")
    print(f"\nSHAPE")
    print(f"  Spearman rho (cells with >=3 live episodes, n={len(both)}) "
          f"{rho:+.3f}")
    print(f"\nCONTAMINATION")
    print(f"  panel forfeit rate {forf:.2%}  (old external panel: 24.9%, "
          f"ladder: 0.0%)")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"tag": args.tag, "rows": rows,
             "field_weighted": round(fw, 4), "field_weighted_se": round(fw_se, 4),
             "pooled_clean": round(pooled, 4), "pooled_n": pooled_n,
             "live_weighted": round(live_w_match, 4),
             "live_covered": round(live_cov, 4), "live_all": round(live_all, 4),
             "level_offset": round(fw - live_w_match, 4),
             "spearman": round(rho, 3), "forfeit_rate": round(forf, 4)},
            indent=1))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
