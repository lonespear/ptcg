"""Price the C1/C2 trajectory features on real games, or refuse to ship them.

D30's discipline: an evaluator weight is a fitted coefficient or it is not a
weight. So this computes the agent's own trajectory features — the exact
functions `agent/main.py` runs on the ladder, imported from the shipped file
rather than reimplemented — retrospectively over the mined positions sample,
and fits

    won ~ shipped evaluator margin terms + the trajectory features

by the same logistic machinery `ptcg/advantage.py` used for the shipped vector
(Newton-Raphson MLE, Wald intervals), on one position per episode at rating
>= 1000, with turn fixed effects and `went_first`. Coefficients are rescaled
into evaluator units by anchoring prize_diff at its 1000.

A feature whose 95% interval covers zero is reported as measured-null and does
not get a weight.

    python scripts/fit_trajectory_features.py
    python scripts/fit_trajectory_features.py --out data/analysis/trajectory_fit.json
"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "engine"))

from ptcg.advantage import RATING_CUT, coef_table, logit_fit  # noqa: E402

MINED = ROOT / "data" / "mined"
AGENT = ROOT / "agent" / "main.py"

# The terms the shipped evaluator already scores (agent/main.py WEIGHTS).
BASE_TERMS = ["prize_diff", "hp_diff", "energy_diff", "bench_diff",
              "damage_diff"]
TRAJ_TERMS = ["online_lead", "energy_traj_t2", "threat_traj_t2"]


def load_agent():
    """Import the shipped agent from beside its own deck.csv."""
    cwd = os.getcwd()
    os.chdir(AGENT.parent)
    try:
        spec = importlib.util.spec_from_file_location("shipped_agent", AGENT)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["shipped_agent"] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        os.chdir(cwd)


def side_totals(player: dict) -> tuple[float, float, float, float]:
    hp = energy = damage = 0.0
    mons = 0
    for zone in ("active", "bench"):
        for mon in player.get(zone) or []:
            if not mon:
                continue
            mons += 1
            cur_hp = float(mon.get("hp") or 0)
            hp += cur_hp
            damage += float(mon.get("maxHp") or 0) - cur_hp
            energy += len(mon.get("energies") or [])
    bench = len([m for m in (player.get("bench") or []) if m])
    return hp, energy, bench, damage


def build_rows(agent, dates: list[str] | None) -> tuple[list[dict], dict]:
    paths = sorted(MINED.glob("*/positions.jsonl.gz"))
    if dates:
        paths = [p for p in paths if p.parent.name in dates]
    rows: list[dict] = []
    timing = {"n": 0, "seconds": 0.0}
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                obs = rec.get("observation") or {}
                cur = obs.get("current") or {}
                players = cur.get("players") or []
                if len(players) != 2 or rec.get("won", -1) < 0:
                    continue
                me = cur.get("yourIndex")
                if me not in (0, 1):
                    me = int(rec.get("seat", 0))
                mine, theirs = players[me], players[1 - me]

                # Exactly the live order: labels once, then the features.
                t0 = time.perf_counter()
                agent._refresh_traj_arch(obs)
                try:
                    lead, e_t2, thr_t2 = agent._trajectory_terms(cur, mine,
                                                                 theirs)
                except Exception:
                    lead = e_t2 = thr_t2 = 0.0
                timing["seconds"] += time.perf_counter() - t0
                timing["n"] += 1

                # The same differential at other horizons, so "the projection
                # is doing the work" is a measurement and not a claim.
                thr_k = {}
                try:
                    (sm, gm, _), (st, gt, _) = agent._traj_projection(
                        cur, mine, theirs)
                    for k in (0, 1, 3):
                        thr_k[f"threat_traj_k{k}"] = (
                            agent._threat_at(sm, gm, k)
                            - agent._threat_at(st, gt, k))
                except Exception:
                    thr_k = {f"threat_traj_k{k}": 0.0 for k in (0, 1, 3)}

                hp_m, en_m, bench_m, dmg_m = side_totals(mine)
                hp_t, en_t, bench_t, dmg_t = side_totals(theirs)
                rows.append({
                    "episode_id": int(rec["episode_id"]),
                    "turn": int(rec.get("turn", -1)),
                    "rating": float(rec.get("agent_rating") or 0.0),
                    "went_first": int(rec.get("went_first", -1)),
                    "won": int(rec["won"]),
                    "our_archetype": rec.get("our_archetype"),
                    "opp_archetype": rec.get("opp_archetype"),
                    "label_us": agent._TRAJ_ARCH["us"],
                    "label_them": agent._TRAJ_ARCH["them"],
                    "prize_diff": len(theirs.get("prize") or [])
                                  - len(mine.get("prize") or []),
                    "hp_diff": hp_m - hp_t,
                    "energy_diff": en_m - en_t,
                    "bench_diff": bench_m - bench_t,
                    "damage_diff": dmg_t - dmg_m,
                    "no_active_me": 0.0 if (mine.get("active") or []) else 1.0,
                    "online_lead": float(lead),
                    "energy_traj_t2": float(e_t2),
                    "threat_traj_t2": float(thr_t2),
                    **thr_k,
                })
    return rows, timing


def dedup(rows: list[dict]) -> list[dict]:
    """One position an episode, the earliest — `ptcg/advantage.py` §A2's rule."""
    keep: dict[int, dict] = {}
    for r in rows:
        cur = keep.get(r["episode_id"])
        if cur is None or r["turn"] < cur["turn"]:
            keep[r["episode_id"]] = r
    return sorted(keep.values(), key=lambda r: (r["episode_id"],))


def fit(sel: list[dict], terms: list[str]) -> dict:
    cols = [np.ones(len(sel))]
    names = ["const"]
    for tm in terms:
        cols.append(np.array([r[tm] for r in sel], dtype=float))
        names.append(tm)
    na = np.array([r["no_active_me"] for r in sel], dtype=float)
    if na.std() > 0 and na.sum() >= 10:
        cols.append(na)
        names.append("no_active_me")
    cols.append(np.array([r["went_first"] for r in sel], dtype=float))
    names.append("went_first")
    turns = sorted({r["turn"] for r in sel})
    for tv in turns[1:]:
        cols.append(np.array([1.0 if r["turn"] == tv else 0.0 for r in sel]))
        names.append(f"turn={tv}")
    X = np.column_stack(cols)
    y = np.array([r["won"] for r in sel], dtype=float)
    f = logit_fit(X, y)
    tab = coef_table(f, names, X)
    anchor = next(r for r in tab if r["term"] == "prize_diff")["beta"]
    scaled = {}
    for r in tab:
        if r["term"] in terms + ["no_active_me"] and anchor:
            lo = 1000.0 * r["ci_lo"] / anchor
            hi = 1000.0 * r["ci_hi"] / anchor
            scaled[r["term"]] = {
                "weight": round(1000.0 * r["beta"] / anchor, 2),
                "ci_lo": round(min(lo, hi), 2),
                "ci_hi": round(max(lo, hi), 2),
                "excludes_zero": bool(r["sig"]),
            }
    return {"n": len(sel), "base_rate_win": round(float(y.mean()), 4),
            "mcfadden_r2": round(f["mcfadden_r2"], 4),
            "loglik": round(f["loglik"], 3),
            "coefficients": tab,
            "weights_prize_1000": scaled}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", nargs="*", default=None)
    ap.add_argument("--out", default=str(ROOT / "data" / "analysis"
                                         / "trajectory_fit.json"))
    args = ap.parse_args()

    agent = load_agent()
    if not agent._curves():
        sys.exit("no trajectory curves loaded — the fit would be meaningless")
    if not agent._accel_rates():
        sys.exit("no energy-mechanics KB loaded — the fit would be meaningless")

    rows, timing = build_rows(agent, args.dates)
    print(f"{len(rows)} positions read; feature cost "
          f"{1000 * timing['seconds'] / max(timing['n'], 1):.3f} ms each")

    sel = [r for r in dedup(rows)
           if r["rating"] >= RATING_CUT and r["turn"] > 0
           and r["went_first"] >= 0]
    print(f"{len(sel)} episodes after the rating cut and the one-per-episode "
          f"rule")

    labelled = sum(1 for r in sel if r["label_them"] != "(all)")
    exact = sum(1 for r in sel if r["label_them"] == r["opp_archetype"])
    print(f"opponent label: {labelled} of {len(sel)} named, "
          f"{exact} matching the replay's own archetype "
          f"({100.0 * exact / max(len(sel), 1):.1f}%)")

    base = fit(sel, BASE_TERMS)
    full = fit(sel, BASE_TERMS + TRAJ_TERMS)
    solo = {t: fit(sel, BASE_TERMS + [t]) for t in TRAJ_TERMS}

    lr = 2.0 * (full["loglik"] - base["loglik"])
    print(f"\nbase   n={base['n']}  McFadden R2 {base['mcfadden_r2']}")
    print(f"full   n={full['n']}  McFadden R2 {full['mcfadden_r2']}  "
          f"LR chi2(3) = {lr:.2f}")

    print(f"\n{'term':18s} {'beta':>9s} {'95% CI':>22s} {'weight':>10s} "
          f"{'CI (weight)':>22s}")
    for r in full["coefficients"]:
        if r["term"] not in BASE_TERMS + TRAJ_TERMS + ["no_active_me"]:
            continue
        w = full["weights_prize_1000"].get(r["term"], {})
        print(f"{r['term']:18s} {r['beta']:+9.4f} "
              f"[{r['ci_lo']:+8.4f},{r['ci_hi']:+8.4f}] "
              f"{w.get('weight', 0):10.2f} "
              f"[{w.get('ci_lo', 0):+9.2f},{w.get('ci_hi', 0):+9.2f}]"
              f"{'' if w.get('excludes_zero') else '   NULL'}")

    print("\nsame feature, entered alone on top of the shipped terms:")
    for t in TRAJ_TERMS:
        w = solo[t]["weights_prize_1000"][t]
        print(f"  {t:18s} weight {w['weight']:9.2f} "
              f"[{w['ci_lo']:+9.2f},{w['ci_hi']:+9.2f}]"
              f"{'' if w['excludes_zero'] else '   NULL'}")

    # Does the projection earn its place, or is this the present board?
    horizons = {}
    for k, name in ((0, "threat_traj_k0"), (1, "threat_traj_k1"),
                    (2, "threat_traj_t2"), (3, "threat_traj_k3")):
        f = fit(sel, BASE_TERMS + [name])
        horizons[name] = dict(f["weights_prize_1000"][name],
                              loglik=f["loglik"])
    pair = fit(sel, BASE_TERMS + ["threat_traj_k0", "threat_traj_t2"])
    print("\ndamage-potential differential, by how far ahead it looks:")
    for name, w in horizons.items():
        print(f"  {name:18s} weight {w['weight']:7.2f} "
              f"[{w['ci_lo']:+7.2f},{w['ci_hi']:+7.2f}]  "
              f"loglik {w['loglik']:.2f}"
              f"{'' if w['excludes_zero'] else '   NULL'}")
    print("  present board and t+2 in the model together:")
    for name in ("threat_traj_k0", "threat_traj_t2"):
        w = pair["weights_prize_1000"][name]
        print(f"    {name:16s} weight {w['weight']:7.2f} "
              f"[{w['ci_lo']:+7.2f},{w['ci_hi']:+7.2f}]"
              f"{'' if w['excludes_zero'] else '   NULL'}")

    # The model that actually ships: the evaluator's terms plus the features
    # whose intervals excluded zero. Its coefficient is the shipped weight.
    survivors = [t for t in TRAJ_TERMS
                 if full["weights_prize_1000"][t]["excludes_zero"]]
    shipping = fit(sel, BASE_TERMS + survivors) if survivors else None
    if shipping:
        print(f"\nshipping model ({', '.join(survivors)}):")
        for t in survivors:
            w = shipping["weights_prize_1000"][t]
            print(f"  WEIGHTS[{t!r}] = {w['weight']}   "
                  f"[{w['ci_lo']:+.2f}, {w['ci_hi']:+.2f}]")

    corr_terms = BASE_TERMS + TRAJ_TERMS
    M = np.column_stack([[r[t] for r in sel] for t in corr_terms])
    C = np.corrcoef(M, rowvar=False)
    print("\ncorrelation with the shipped terms:")
    for i, t in enumerate(TRAJ_TERMS):
        j = len(BASE_TERMS) + i
        print(f"  {t:18s} " + "  ".join(
            f"{b}={C[j, k]:+.2f}" for k, b in enumerate(BASE_TERMS)))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "spec": {"sample": "data/mined/*/positions.jsonl.gz",
                 "rule": "one position per episode (earliest turn), "
                         "rating >= 1000, turn fixed effects, went_first",
                 "base_terms": BASE_TERMS, "trajectory_terms": TRAJ_TERMS,
                 "anchor": "prize_diff = 1000",
                 "feature_source": "agent/main.py::_trajectory_terms"},
        "n_positions_read": len(rows),
        "feature_ms_each": round(1000 * timing["seconds"]
                                 / max(timing["n"], 1), 4),
        "label_match_rate": round(exact / max(len(sel), 1), 4),
        "base": base, "full": full,
        "solo": {t: solo[t]["weights_prize_1000"][t] for t in TRAJ_TERMS},
        "horizons": horizons,
        "present_and_t2_together": {
            n: pair["weights_prize_1000"][n]
            for n in ("threat_traj_k0", "threat_traj_t2")},
        "shipping_model": shipping,
        "shipped_weights": {t: shipping["weights_prize_1000"][t]["weight"]
                            for t in survivors} if shipping else {},
        "measured_null": {t: full["weights_prize_1000"][t]
                          for t in TRAJ_TERMS if t not in survivors},
        "lr_chi2_3": round(lr, 3),
    }, indent=1))
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
