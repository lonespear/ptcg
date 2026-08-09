"""Phase-conditional weight vectors: choose the phase variable, then price it.

The turn-checkpoint fits in `data/analysis/REPORT.md` showed the advantage
coefficients drift across the game (bench early, the prize race late), and the
tree-leaf refusal (playbook entry 7) says unscreened flexible structure is not
how that drift gets captured. The sanctioned form is a mixture of linear
experts with a NAMED gate: 2-3 phases, each phase a linear vector fitted by
exactly the machinery that fitted the shipped v2 vector.

Two decisions are made here, both on evidence:

  1. THE GATE. Candidates are prizes-remaining bands (total and the min of
     the two sides — both computable from the observation the evaluator
     already reads) and turn bands. Each candidate partitions the SAME
     sample (one position per episode, earliest turn, rating >= 1000 — the
     v2 rule) and fits one logistic per phase on the train days
     (07-31..08-05); the held-out day (08-06, the tree-leaf split) scores
     them. The winner is the candidate with the best held-out log-likelihood,
     and a candidate only beats "global" if it actually transfers.

  2. THE VECTORS. The chosen gate is refitted per phase on all seven days —
     the citation fit, same terms the evaluator scores (prize, hp, energy,
     bench, damage, threat_traj at k=3), turn fixed effects, went_first,
     no_active where it varies — and rescaled into evaluator units by
     anchoring prize_diff at 1000 per phase. Per term, the exact v2 rule
     (the WEIGHTS comment block documents it): the incumbent value stands
     unless the fitted 95% interval EXCLUDES it — that is how energy stayed
     30 in v2 against a significant 70 [8.7, 131.4] in the richer fit, and
     how hp moved off 1.0 to 2.6. Where the interval rejects the incumbent
     the fitted point estimate ships, cited with its interval. The threat
     term is under the same rule, so it goes per-phase only if some phase's
     interval excludes the global 1.71.

    python scripts/fit_phase_weights.py
    python scripts/fit_phase_weights.py --rows-cache /tmp/phase_rows.jsonl
"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "engine"))

from ptcg.advantage import RATING_CUT, Z95, coef_table, logit_fit  # noqa: E402

MINED = ROOT / "data" / "mined"
AGENT = ROOT / "agent" / "main.py"
HELD_OUT_DAY = "2026-08-06"

# The terms the shipped evaluator scores, as differentials, plus C2's term.
BASE_TERMS = ["prize_diff", "hp_diff", "energy_diff", "bench_diff",
              "damage_diff", "threat_traj_k3"]

# The incumbent vector these fits are measured against (agent/main.py WEIGHTS).
INCUMBENT = {"prize": 1000.0, "hp": 2.6, "energy": 30.0, "bench": 153.0,
             "damage": 4.2, "threat_traj": 1.71}
TERM_TO_KEY = {"hp_diff": "hp", "energy_diff": "energy", "bench_diff": "bench",
               "damage_diff": "damage", "threat_traj_k3": "threat_traj"}


def load_agent():
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


def build_rows(agent, dates=None) -> list[dict]:
    paths = sorted(MINED.glob("*/positions.jsonl.gz"))
    if dates:
        paths = [p for p in paths if p.parent.name in dates]
    rows: list[dict] = []
    for path in paths:
        date = path.parent.name
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

                agent._refresh_traj_arch(obs)
                try:
                    (sm, gm, _), (st, gt, _) = agent._traj_projection(
                        cur, mine, theirs)
                    thr3 = (agent._threat_at(sm, gm, 3)
                            - agent._threat_at(st, gt, 3))
                except Exception:
                    thr3 = 0.0

                hp_m, en_m, bench_m, dmg_m = side_totals(mine)
                hp_t, en_t, bench_t, dmg_t = side_totals(theirs)
                pl_m = len(mine.get("prize") or [])
                pl_t = len(theirs.get("prize") or [])
                rows.append({
                    "episode_id": int(rec["episode_id"]),
                    "date": date,
                    "turn": int(rec.get("turn", -1)),
                    "rating": float(rec.get("agent_rating") or 0.0),
                    "went_first": int(rec.get("went_first", -1)),
                    "won": int(rec["won"]),
                    "prizes_left_me": pl_m, "prizes_left_them": pl_t,
                    "prize_diff": pl_t - pl_m,
                    "hp_diff": hp_m - hp_t,
                    "energy_diff": en_m - en_t,
                    "bench_diff": bench_m - bench_t,
                    "damage_diff": dmg_t - dmg_m,
                    "no_active_me": 0.0 if (mine.get("active") or []) else 1.0,
                    "threat_traj_k3": float(thr3),
                })
    return rows


def dedup(rows: list[dict]) -> list[dict]:
    """One position an episode, the earliest — the v2 sample rule."""
    keep: dict[int, dict] = {}
    for r in rows:
        cur = keep.get(r["episode_id"])
        if cur is None or r["turn"] < cur["turn"]:
            keep[r["episode_id"]] = r
    return sorted(keep.values(), key=lambda r: r["episode_id"])


# ---------------------------------------------------------------------------
# candidate gates: named, and computable from what the evaluator already reads
# ---------------------------------------------------------------------------

def phase_prize_total(r, cuts) -> int:
    """Bands over total prizes remaining (mine + theirs), 12 down to 2."""
    total = r["prizes_left_me"] + r["prizes_left_them"]
    for i, c in enumerate(cuts):
        if total >= c:
            return i
    return len(cuts)


def phase_prize_min(r, cuts) -> int:
    """Bands over the leading side's prizes remaining."""
    m = min(r["prizes_left_me"], r["prizes_left_them"])
    for i, c in enumerate(cuts):
        if m >= c:
            return i
    return len(cuts)


def phase_turn(r, cuts) -> int:
    for i, c in enumerate(cuts):
        if r["turn"] <= c:
            return i
    return len(cuts)


# ---------------------------------------------------------------------------
# fitting
# ---------------------------------------------------------------------------

def design(sel: list[dict], terms: list[str], turn_levels: list[int]):
    cols = [np.ones(len(sel))]
    names = ["const"]
    for tm in terms:
        v = np.array([r[tm] for r in sel], dtype=float)
        if v.std() == 0:
            continue
        cols.append(v)
        names.append(tm)
    na = np.array([r["no_active_me"] for r in sel], dtype=float)
    if na.std() > 0 and na.sum() >= 10:
        cols.append(na)
        names.append("no_active_me")
    cols.append(np.array([r["went_first"] for r in sel], dtype=float))
    names.append("went_first")
    for tv in turn_levels:
        v = np.array([1.0 if r["turn"] == tv else 0.0 for r in sel])
        if v.std() == 0:
            continue
        cols.append(v)
        names.append(f"turn={tv}")
    return np.column_stack(cols), names


def fit_phase(sel: list[dict], terms: list[str]) -> dict:
    turn_levels = sorted({r["turn"] for r in sel})[1:]
    X, names = design(sel, terms, turn_levels)
    y = np.array([r["won"] for r in sel], dtype=float)
    f = logit_fit(X, y)
    tab = coef_table(f, names, X)
    anchor = next((r for r in tab if r["term"] == "prize_diff"), None)
    scaled = {}
    if anchor and anchor["beta"]:
        for r in tab:
            if r["term"] in terms and r["term"] != "prize_diff":
                lo = 1000.0 * r["ci_lo"] / anchor["beta"]
                hi = 1000.0 * r["ci_hi"] / anchor["beta"]
                scaled[r["term"]] = {
                    "weight": round(1000.0 * r["beta"] / anchor["beta"], 2),
                    "ci_lo": round(min(lo, hi), 2),
                    "ci_hi": round(max(lo, hi), 2),
                    "excludes_zero": bool(r["sig"]),
                }
    return {"n": len(sel), "base_rate_win": round(float(y.mean()), 4),
            "mcfadden_r2": round(f["mcfadden_r2"], 4),
            "loglik": round(f["loglik"], 3),
            "beta": {nm: float(b) for nm, b in zip(names, f["beta"])},
            "coefficients": tab,
            "prize_anchor_beta": anchor["beta"] if anchor else None,
            "prize_anchor_ci": ([anchor["ci_lo"], anchor["ci_hi"]]
                                if anchor else None),
            "weights_prize_1000": scaled}


def predict(beta: dict, r: dict, terms: list[str]) -> float:
    eta = beta.get("const", 0.0)
    for tm in terms:
        eta += beta.get(tm, 0.0) * r[tm]
    eta += beta.get("no_active_me", 0.0) * r["no_active_me"]
    eta += beta.get("went_first", 0.0) * r["went_first"]
    eta += beta.get(f"turn={r['turn']}", 0.0)
    eta = max(-35.0, min(35.0, eta))
    return 1.0 / (1.0 + math.exp(-eta))


def score_held_out(fits: list[dict], phase_fn, test: list[dict],
                   terms: list[str]) -> dict:
    ll = brier = 0.0
    ps, ys = [], []
    for r in test:
        ph = phase_fn(r)
        ph = min(ph, len(fits) - 1)
        p = predict(fits[ph]["beta"], r, terms)
        p = min(max(p, 1e-9), 1 - 1e-9)
        y = r["won"]
        ll += y * math.log(p) + (1 - y) * math.log(1 - p)
        brier += (p - y) ** 2
        ps.append(p)
        ys.append(y)
    ps, ys = np.array(ps), np.array(ys)
    pos, neg = ps[ys == 1], ps[ys == 0]
    if len(pos) and len(neg):
        auc = (np.sum(pos[:, None] > neg[None, :])
               + 0.5 * np.sum(pos[:, None] == neg[None, :])) \
              / (len(pos) * len(neg))
    else:
        auc = float("nan")
    return {"n": len(test), "loglik": round(ll, 3),
            "brier": round(brier / len(test), 5), "auc": round(float(auc), 5)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows-cache", default=None,
                    help="jsonl cache of extracted rows; created if absent")
    ap.add_argument("--out", default=str(ROOT / "data" / "analysis"
                                         / "phase_weights_fit.json"))
    args = ap.parse_args()

    rows = None
    if args.rows_cache and Path(args.rows_cache).exists():
        rows = [json.loads(l) for l in open(args.rows_cache)]
        print(f"{len(rows)} rows from cache")
    if rows is None:
        agent = load_agent()
        if not agent._curves() or not agent._accel_rates():
            sys.exit("trajectory KB missing — the threat term would be zero")
        rows = build_rows(agent)
        print(f"{len(rows)} positions read")
        if args.rows_cache:
            with open(args.rows_cache, "w") as fh:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")

    sel = [r for r in dedup(rows)
           if r["rating"] >= RATING_CUT and r["turn"] > 0
           and r["went_first"] >= 0]
    print(f"{len(sel)} episodes after the rating cut and one-per-episode rule")

    train = [r for r in sel if r["date"] != HELD_OUT_DAY]
    test = [r for r in sel if r["date"] == HELD_OUT_DAY]
    print(f"train {len(train)} (through 08-05), held out {len(test)} (08-06)")

    # distributions, so the cut points below are readable against the data
    from collections import Counter
    tot = Counter(r["prizes_left_me"] + r["prizes_left_them"] for r in sel)
    mn = Counter(min(r["prizes_left_me"], r["prizes_left_them"]) for r in sel)
    tn = Counter(min(r["turn"], 15) for r in sel)
    print("total prizes remaining:", dict(sorted(tot.items())))
    print("min prizes remaining:  ", dict(sorted(mn.items())))
    print("turn (15+ pooled):     ", dict(sorted(tn.items())))

    candidates = {
        "global": (lambda r: 0, 1),
        # total prizes remaining: 12..11 = nothing has died yet, 10..7 = the
        # exchange is on, <=6 = half the prizes are gone and the race can end
        "prize_total_2@10": (lambda r: phase_prize_total(r, [10]), 2),
        "prize_total_2@8": (lambda r: phase_prize_total(r, [8]), 2),
        "prize_total_3@11_7": (lambda r: phase_prize_total(r, [11, 7]), 3),
        "prize_total_3@10_7": (lambda r: phase_prize_total(r, [10, 7]), 3),
        # the leading side's pile: 6..5 opening, 4..3 middle, <=2 closing
        "prize_min_2@5": (lambda r: phase_prize_min(r, [5]), 2),
        "prize_min_3@5_3": (lambda r: phase_prize_min(r, [5, 3]), 3),
        # turn bands around the checkpoint drift the A1 fits showed
        "turn_2@5": (lambda r: phase_turn(r, [5]), 2),
        "turn_3@3_7": (lambda r: phase_turn(r, [3, 7]), 3),
        "turn_3@4_8": (lambda r: phase_turn(r, [4, 8]), 3),
    }

    results = {}
    for name, (fn, k) in candidates.items():
        fits, ok = [], True
        for ph in range(k):
            part = [r for r in train if fn(r) == ph]
            if len(part) < 300:
                ok = False
                break
            fits.append(fit_phase(part, BASE_TERMS))
        if not ok:
            results[name] = {"note": "a phase fell under 300 train episodes"}
            continue
        held = score_held_out(fits, fn, test, BASE_TERMS)
        results[name] = {
            "phases": k,
            "train_n": [f["n"] for f in fits],
            "train_loglik": round(sum(f["loglik"] for f in fits), 3),
            "held_out": held,
        }
        print(f"{name:22s} train n {results[name]['train_n']}  "
              f"held-out ll {held['loglik']:9.3f}  auc {held['auc']:.4f}  "
              f"brier {held['brier']:.5f}")

    base_ll = results["global"]["held_out"]["loglik"]
    for name, r in results.items():
        if "held_out" in r:
            r["held_out"]["delta_ll_vs_global"] = round(
                r["held_out"]["loglik"] - base_ll, 3)

    # the winner: best held-out log-likelihood among candidates that beat global
    contenders = {n: r for n, r in results.items()
                  if "held_out" in r and n != "global"}
    best = max(contenders, key=lambda n: contenders[n]["held_out"]["loglik"])
    chosen = best if contenders[best]["held_out"]["loglik"] > base_ll else "global"
    print(f"\nchosen gate: {chosen} "
          f"(held-out delta vs global "
          f"{results[chosen]['held_out']['delta_ll_vs_global'] if chosen != 'global' else 0.0})")

    out = {"spec": {
        "sample": "data/mined/*/positions.jsonl.gz, one position per episode "
                  "(earliest turn), rating >= 1000, went_first known",
        "split": f"train through 08-05, held out {HELD_OUT_DAY}",
        "terms": BASE_TERMS,
        "anchor": "prize_diff = 1000 per phase",
        "incumbent": INCUMBENT},
        "n_train": len(train), "n_test": len(test),
        "distributions": {"prize_total": dict(sorted(tot.items())),
                          "prize_min": dict(sorted(mn.items())),
                          "turn_15plus_pooled": dict(sorted(tn.items()))},
        "selection": results, "chosen_gate": chosen}

    # citation fits: the chosen gate on all seven days
    if chosen != "global":
        fn, k = candidates[chosen]
        final = []
        for ph in range(k):
            part = [r for r in sel if fn(r) == ph]
            f = fit_phase(part, BASE_TERMS)
            final.append(f)
            print(f"\nphase {ph}: n = {f['n']}, base rate {f['base_rate_win']}, "
                  f"McFadden R2 {f['mcfadden_r2']}")
            for tm, w in f["weights_prize_1000"].items():
                print(f"  {tm:16s} {w['weight']:9.2f} "
                      f"[{w['ci_lo']:+9.2f}, {w['ci_hi']:+9.2f}]"
                      f"{'' if w['excludes_zero'] else '   NULL'}")

        # the v2 rule per term: the incumbent stands unless the interval
        # excludes it; where it does, the fitted point estimate ships
        vectors, moves = [], []
        for ph, f in enumerate(final):
            v = {"prize": 1000.0, "no_active": 4000.0}
            for tm, key in TERM_TO_KEY.items():
                w = f["weights_prize_1000"].get(tm)
                if w is None:
                    v[key] = INCUMBENT[key]
                elif w["ci_lo"] <= INCUMBENT[key] <= w["ci_hi"]:
                    v[key] = INCUMBENT[key]     # the data declines to move it
                else:
                    v[key] = w["weight"]
                    moves.append({"phase": ph, "term": key,
                                  "from": INCUMBENT[key], "to": w["weight"],
                                  "ci": [w["ci_lo"], w["ci_hi"]],
                                  "excludes_zero": w["excludes_zero"]})
            vectors.append(v)
        out["final_fits"] = final
        out["moves_off_incumbent"] = moves
        out["vectors"] = vectors
        print("\nvectors (incumbent stands unless its value sits outside "
              "the fitted interval):")
        for ph, v in enumerate(vectors):
            print(f"  phase {ph}: {v}")
        for m in moves:
            print(f"  moved: phase {m['phase']} {m['term']} "
                  f"{m['from']} -> {m['to']} CI {m['ci']}"
                  f"{'' if m['excludes_zero'] else '  (null vs zero)'}")

    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
