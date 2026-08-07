"""Price the attacker-protection feature on real games, or refuse to ship it.

`data/analysis/DECK_REFINEMENT.md`: 68% of this deck's losses are "no active
Pokemon". The evaluator scored HP and bench width and neither of them says
which of our Pokemon does not survive the opponent's next turn, which is what
that loss mode is made of. `agent/main.py::_exposure` is the feature that
does — our attack-capable Pokemon whose remaining HP is at or under the damage
their archetype can afford one of its own turns from now, counting a benched
Pokemon only where their lists carry a gust.

Same discipline as `fit_trajectory_features.py`, same machinery, same sample:
the agent's own function is imported from the shipped file rather than
reimplemented, computed retrospectively over the mined positions, and fitted

    won ~ shipped evaluator terms (including the C2 term) + attackers_exposed

by `ptcg/advantage.py`'s logistic (Newton-Raphson MLE, Wald intervals), one
position per episode at rating >= 1000, with turn fixed effects and
went_first, rescaled into evaluator units by anchoring prize_diff at 1000. An
interval that covers zero is reported as measured-null and gets no weight.

The last-attacker rule is reported here and is NOT fitted here: it is a
statement about what a line should avoid leaving true, and the retrospective
sample holds one position per episode with no counterfactual attached to it.
What this script can say about it, and does, is how often the position arises
and how those episodes ended.

    python scripts/fit_protection_feature.py
    python scripts/fit_protection_feature.py --out data/analysis/protection_fit.json
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

# The terms the shipped evaluator scores, the C2 term included: the protection
# feature has to earn its interval on top of what already runs, not against a
# stripped-down vector it can beat by proxying for something already priced.
BASE_TERMS = ["prize_diff", "hp_diff", "energy_diff", "bench_diff",
              "damage_diff", "threat_traj_k3"]
NEW_TERMS = ["attackers_exposed", "attackers_exposed_diff"]


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
    for zone in ("active", "bench"):
        for mon in player.get(zone) or []:
            if not mon:
                continue
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

                # Exactly the live order: labels once (they key both the curve
                # and the gust-reach lookup), then the features.
                t0 = time.perf_counter()
                agent._refresh_traj_arch(obs)
                try:
                    thr, exposed, capable = agent._threat_and_exposure(
                        cur, mine, theirs)
                except Exception:
                    thr, exposed, capable = 0.0, 0, 0
                # Their exposure under our threat, the mirror of the feature,
                # so "ours minus theirs" can be priced beside "ours".
                try:
                    (sm, gm, _), (st, gt, _) = agent._traj_projection(
                        cur, mine, theirs)
                    ours_out = agent._threat_at(sm, gm, 1)
                    reach_them = agent._GUST_REACH.get(
                        agent._TRAJ_ARCH["us"], agent._GUST_REACH_DEFAULT)
                    exposed_t, capable_t = agent._exposure(
                        theirs, ours_out, reach_them)
                except Exception:
                    exposed_t, capable_t = 0, 0
                timing["seconds"] += time.perf_counter() - t0
                timing["n"] += 1

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
                    "label_them": agent._TRAJ_ARCH["them"],
                    "prize_diff": len(theirs.get("prize") or [])
                                  - len(mine.get("prize") or []),
                    "hp_diff": hp_m - hp_t,
                    "energy_diff": en_m - en_t,
                    "bench_diff": bench_m - bench_t,
                    "damage_diff": dmg_t - dmg_m,
                    "no_active_me": 0.0 if (mine.get("active") or []) else 1.0,
                    "threat_traj_k3": float(thr),
                    "attackers_exposed": float(exposed),
                    "attackers_capable": float(capable),
                    "attackers_exposed_them": float(exposed_t),
                    "attackers_exposed_diff": float(exposed_t - exposed),
                    "last_attacker": 1.0 if (capable == 1 and exposed == 1)
                                     else 0.0,
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


def fit(sel: list[dict], terms: list[str], arch_fe: bool = False) -> dict:
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
    if arch_fe:
        # Which deck the seat is playing. Wide cheap benches and small tanky
        # boards are archetype properties, and both of them move the exposure
        # count without any of it being about danger, so the sign of this
        # feature is only readable with the deck held fixed.
        seen = sorted({(r["our_archetype"] or "?") for r in sel})
        for a in seen[1:]:
            col = np.array([1.0 if (r["our_archetype"] or "?") == a else 0.0
                            for r in sel])
            if col.std() > 0 and col.sum() >= 20:
                cols.append(col)
                names.append(f"arch={a}")
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
                                         / "protection_fit.json"))
    args = ap.parse_args()

    agent = load_agent()
    if not agent._curves():
        sys.exit("no trajectory curves loaded — the fit would be meaningless")
    if not agent._posture_specs():
        sys.exit("no postures.json loaded — gust reach would be a guess")

    rows, timing = build_rows(agent, args.dates)
    print(f"{len(rows)} positions read; feature cost "
          f"{1000 * timing['seconds'] / max(timing['n'], 1):.3f} ms each")

    sel = [r for r in dedup(rows)
           if r["rating"] >= RATING_CUT and r["turn"] > 0
           and r["went_first"] >= 0]
    print(f"{len(sel)} episodes after the rating cut and the one-per-episode "
          f"rule")

    # What the feature says about the sample before any model sees it.
    n_exp = sum(1 for r in sel if r["attackers_exposed"] > 0)
    n_last = sum(1 for r in sel if r["last_attacker"])
    won_exp = [r["won"] for r in sel if r["attackers_exposed"] > 0]
    won_safe = [r["won"] for r in sel if r["attackers_exposed"] == 0]
    won_last = [r["won"] for r in sel if r["last_attacker"]]
    print(f"\nexposure on {n_exp} of {len(sel)} positions "
          f"({100.0 * n_exp / max(len(sel), 1):.1f}%); "
          f"win rate {np.mean(won_exp) if won_exp else float('nan'):.3f} "
          f"exposed against "
          f"{np.mean(won_safe) if won_safe else float('nan'):.3f} not")
    print(f"last-attacker exposure on {n_last} positions "
          f"({100.0 * n_last / max(len(sel), 1):.2f}%); "
          f"win rate {np.mean(won_last) if won_last else float('nan'):.3f}")
    mean_cap = float(np.mean([r["attackers_capable"] for r in sel]))
    print(f"attack-capable Pokemon in play, mean {mean_cap:.2f}")
    # The one-per-episode rule is a fitting rule, not a rate: over every
    # position read, this is how often the agent would meet each state.
    all_exp = sum(1 for r in rows if r["attackers_exposed"] > 0)
    all_last = sum(1 for r in rows if r["last_attacker"])
    print(f"over all {len(rows)} positions read: exposure "
          f"{100.0 * all_exp / max(len(rows), 1):.1f}%, last-attacker "
          f"{100.0 * all_last / max(len(rows), 1):.2f}%")

    base = fit(sel, BASE_TERMS)
    full = fit(sel, BASE_TERMS + NEW_TERMS)
    solo = {t: fit(sel, BASE_TERMS + [t]) for t in NEW_TERMS}

    lr = 2.0 * (full["loglik"] - base["loglik"])
    print(f"\nbase   n={base['n']}  McFadden R2 {base['mcfadden_r2']}")
    print(f"full   n={full['n']}  McFadden R2 {full['mcfadden_r2']}  "
          f"LR chi2({len(NEW_TERMS)}) = {lr:.2f}")

    print(f"\n{'term':24s} {'beta':>9s} {'95% CI':>22s} {'weight':>10s} "
          f"{'CI (weight)':>22s}")
    for r in full["coefficients"]:
        if r["term"] not in BASE_TERMS + NEW_TERMS + ["no_active_me"]:
            continue
        w = full["weights_prize_1000"].get(r["term"], {})
        print(f"{r['term']:24s} {r['beta']:+9.4f} "
              f"[{r['ci_lo']:+8.4f},{r['ci_hi']:+8.4f}] "
              f"{w.get('weight', 0):10.2f} "
              f"[{w.get('ci_lo', 0):+9.2f},{w.get('ci_hi', 0):+9.2f}]"
              f"{'' if w.get('excludes_zero') else '   NULL'}")

    print("\nsame feature, entered alone on top of the shipped terms:")
    for t in NEW_TERMS:
        w = solo[t]["weights_prize_1000"][t]
        print(f"  {t:24s} weight {w['weight']:9.2f} "
              f"[{w['ci_lo']:+9.2f},{w['ci_hi']:+9.2f}]"
              f"{'' if w['excludes_zero'] else '   NULL'}")

    # The model that ships: the shipped terms plus whichever survives alone.
    # `attackers_exposed` is the named feature; the differential is reported
    # beside it so a reader can see whether the count only works as a proxy
    # for the same thing on their side.
    ship = solo["attackers_exposed"]["weights_prize_1000"]["attackers_exposed"]
    print(f"\nWEIGHTS['attackers_exposed'] = {ship['weight']}   "
          f"[{ship['ci_lo']:+.2f}, {ship['ci_hi']:+.2f}]"
          f"{'' if ship['excludes_zero'] else '   MEASURED-NULL, no weight'}")

    # Robustness, because a count of bodies is also a description of a deck.
    # Three cuts of the same feature: with the seat's archetype held fixed,
    # on our own list alone, and on the narrow boards the last-attacker rule
    # is about.
    cuts = {
        "archetype_fixed_effects": (sel, True),
        "our_list_only": ([r for r in sel
                           if r["our_archetype"] == "Teal Mask Ogerpon ex"],
                          False),
        "narrow_boards_capable_le_2": ([r for r in sel
                                        if r["attackers_capable"] <= 2],
                                       False),
    }
    robust = {}
    print("\nrobustness — the same feature, entered alone:")
    for name, (rows_c, fe) in cuts.items():
        if len(rows_c) < 200:
            print(f"  {name:28s} n={len(rows_c)} — too few to fit")
            continue
        try:
            f = fit(rows_c, BASE_TERMS + ["attackers_exposed"], arch_fe=fe)
        except Exception as exc:
            print(f"  {name:28s} n={len(rows_c)} — fit failed: {exc}")
            continue
        w = f["weights_prize_1000"]["attackers_exposed"]
        robust[name] = dict(w, n=len(rows_c))
        print(f"  {name:28s} n={len(rows_c):5d}  weight {w['weight']:8.2f} "
              f"[{w['ci_lo']:+8.2f},{w['ci_hi']:+8.2f}]"
              f"{'' if w['excludes_zero'] else '   NULL'}")

    corr_terms = BASE_TERMS + NEW_TERMS
    M = np.column_stack([[r[t] for r in sel] for t in corr_terms])
    C = np.corrcoef(M, rowvar=False)
    print("\ncorrelation with the shipped terms:")
    for i, t in enumerate(NEW_TERMS):
        j = len(BASE_TERMS) + i
        print(f"  {t:24s} " + "  ".join(
            f"{b}={C[j, k]:+.2f}" for k, b in enumerate(BASE_TERMS)))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "spec": {"sample": "data/mined/*/positions.jsonl.gz",
                 "rule": "one position per episode (earliest turn), "
                         "rating >= 1000, turn fixed effects, went_first",
                 "base_terms": BASE_TERMS, "new_terms": NEW_TERMS,
                 "anchor": "prize_diff = 1000",
                 "feature_source": "agent/main.py::_exposure via "
                                   "_threat_and_exposure"},
        "n_positions_read": len(rows),
        "feature_ms_each": round(1000 * timing["seconds"]
                                 / max(timing["n"], 1), 4),
        "descriptive": {
            "n_episodes": len(sel),
            "share_exposed": round(n_exp / max(len(sel), 1), 4),
            "win_rate_exposed": round(float(np.mean(won_exp)), 4)
                                if won_exp else None,
            "win_rate_not_exposed": round(float(np.mean(won_safe)), 4)
                                    if won_safe else None,
            "share_exposed_all_positions": round(all_exp / max(len(rows), 1), 4),
            "share_last_attacker_all_positions": round(
                all_last / max(len(rows), 1), 5),
            "share_last_attacker": round(n_last / max(len(sel), 1), 5),
            "win_rate_last_attacker": round(float(np.mean(won_last)), 4)
                                      if won_last else None,
            "mean_attackers_capable": round(mean_cap, 3),
        },
        "base": base, "full": full,
        "solo": {t: solo[t]["weights_prize_1000"][t] for t in NEW_TERMS},
        "robustness": robust,
        "shipped_weight": (ship["weight"] if ship["excludes_zero"] else 0.0),
        "shipped_interval": ship,
        "lr_chi2": round(lr, 3),
    }, indent=1))
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
