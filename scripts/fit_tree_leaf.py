"""Fit the D34 tree leaf on the mined positions, or refuse to ship it.

D33 closed the 2-ply axis on the leaf: three refusals of a deeper search, each
removing the excuse the last had used. The sanctioned escalation is a richer
leaf over the SAME named features — a gradient-boosted forest fitted on real
ladder outcomes rather than a dot product with fitted coefficients.

The sample and the rule are the ones every fit in this repo uses
(`ptcg/advantage.py` §A2, `fit_trajectory_features.py`,
`fit_protection_feature.py`): one position per episode, the earliest, rating
>= 1000. The split is by DAY and not at random, because two positions from
one day share a metagame and a random split would let the model read the
validation day's field off the training rows: train 2026-07-31..08-05,
validate 2026-08-06.

The features are the agent's own — `agent/main.py::_tree_features`, imported
from the shipped file rather than reimplemented, so a training-time feature
and a run-time feature cannot drift.

Two stages, because the two interpreters this laptop has do not overlap: the
agent needs 3.10+ syntax and the venv has no sklearn; system python3.9 has
sklearn and cannot import the agent.

    ~/Desktop/PTCG_AI/.venv/bin/python scripts/fit_tree_leaf.py features
    /usr/bin/python3                   scripts/fit_tree_leaf.py fit

Stage one writes data/analysis/tree_features.csv (features, label, day, and
the shipped linear margin over the same position). Stage two fits the forest,
scores it against the linear baselines on the held-out day, and writes
data/analysis/tree_leaf.json — the bundled forest — plus tree_leaf_fit.json,
the report.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MINED = ROOT / "data" / "mined"
AGENT = ROOT / "agent" / "main.py"
ANALYSIS = ROOT / "data" / "analysis"
FEATURES_CSV = ANALYSIS / "tree_features.csv"
LEAF_JSON = ANALYSIS / "tree_leaf.json"
REPORT_JSON = ANALYSIS / "tree_leaf_fit.json"

RATING_CUT = 1000.0
TRAIN_DAYS = ("2026-07-31", "2026-08-01", "2026-08-02", "2026-08-03",
              "2026-08-04", "2026-08-05")
VALID_DAYS = ("2026-08-06",)


# --------------------------------------------------------------- stage one

def load_agent():
    """Import the shipped agent from beside its own deck.csv.

    `engine/` goes on the path first, because `cg.api` is where the card and
    attack metadata comes from and the projection features are computed off
    it. Import the agent without it and `CG_AVAILABLE` is False, every attack
    profile is empty, and the projection silently returns zeros — a fit on
    features the live agent never sees.
    """
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "engine"))
    import ptcg.creation  # noqa: F401  — engine bootstrap
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


def stage_features() -> None:
    agent = load_agent()
    if not agent.CG_AVAILABLE:
        sys.exit("cg.api did not import — the projection features would be "
                 "zeros and the fit would be on something the agent never sees")
    if not agent._curves():
        sys.exit("no trajectory curves loaded — the features would be wrong")
    if not agent._accel_rates():
        sys.exit("no energy-mechanics KB loaded — the features would be wrong")

    names = list(agent.TREE_FEATURES)
    rows: list[dict] = []
    timing = {"n": 0, "seconds": 0.0}
    for path in sorted(MINED.glob("*/positions.jsonl.gz")):
        day = path.parent.name
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

                # The live order: the archetype labels first (they key the
                # curve and the gust-reach lookup), then the features.
                t0 = time.perf_counter()
                agent._refresh_traj_arch(obs)
                try:
                    x = agent._tree_features(cur, me)
                except Exception:
                    continue
                timing["seconds"] += time.perf_counter() - t0
                timing["n"] += 1
                try:
                    linear = agent._margin(cur, me)
                except Exception:
                    linear = 0.0

                # `rec_turn` and not `turn`: `turn` is a FEATURE name, and a
                # metadata column of the same name would silently overwrite it
                # and hand the fit fifteen features while the run-time walk
                # indexed sixteen.
                row = {
                    "episode_id": int(rec["episode_id"]),
                    "day": day,
                    "rec_turn": int(rec.get("turn", -1)),
                    "rating": float(rec.get("agent_rating") or 0.0),
                    "went_first": int(rec.get("went_first", -1)),
                    "won": int(rec["won"]),
                    "linear_margin": float(linear),
                }
                row.update({n: v for n, v in zip(names, x)})
                rows.append(row)

    # One position an episode, the earliest — the rule every fit here uses.
    keep: dict[int, dict] = {}
    for r in rows:
        cur = keep.get(r["episode_id"])
        if cur is None or r["rec_turn"] < cur["rec_turn"]:
            keep[r["episode_id"]] = r
    sel = [r for r in sorted(keep.values(), key=lambda r: r["episode_id"])
           if r["rating"] >= RATING_CUT and r["rec_turn"] > 0
           and r["went_first"] >= 0]

    ANALYSIS.mkdir(parents=True, exist_ok=True)
    cols = (["episode_id", "day", "rec_turn", "rating", "went_first", "won",
             "linear_margin"] + names)
    with FEATURES_CSV.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in sel:
            w.writerow(r)
    print(f"{len(rows)} positions read; feature cost "
          f"{1000 * timing['seconds'] / max(timing['n'], 1):.3f} ms each")
    print(f"{len(sel)} episodes after the rating cut and the one-per-episode "
          f"rule")
    by_day: dict[str, int] = {}
    for r in sel:
        by_day[r["day"]] = by_day.get(r["day"], 0) + 1
    for d in sorted(by_day):
        tag = ("train" if d in TRAIN_DAYS
               else "VALID" if d in VALID_DAYS else "unused")
        print(f"  {d}  {by_day[d]:5d}  {tag}")
    print(f"wrote {FEATURES_CSV.relative_to(ROOT)}")


# --------------------------------------------------------------- stage two

def read_features(drop=()):
    import numpy as np
    with FEATURES_CSV.open() as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        sys.exit(f"{FEATURES_CSV} is empty — run the features stage first")
    meta = {"episode_id", "day", "rec_turn", "rating", "went_first", "won",
            "linear_margin"}
    canonical = [c for c in rows[0] if c not in meta]
    names = [c for c in canonical if c not in drop]
    X = np.array([[float(r[n]) for n in names] for r in rows])
    y = np.array([int(r["won"]) for r in rows], dtype=float)
    day = np.array([r["day"] for r in rows])
    lin = np.array([float(r["linear_margin"]) for r in rows])
    return X, y, day, lin, names, canonical, rows


def auc(y, s) -> float:
    """Mann-Whitney AUC with ties averaged."""
    import numpy as np
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    pos, neg = y.sum(), (1 - y).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def brier(y, p) -> float:
    return float(((p - y) ** 2).mean())


def logloss(y, p) -> float:
    import numpy as np
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def calibration_curve(y, p, bins=10) -> list[dict]:
    import numpy as np
    out = []
    edges = np.quantile(p, [i / bins for i in range(bins + 1)])
    edges[0], edges[-1] = -1.0, 2.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1])
        if not m.any():
            continue
        out.append({"n": int(m.sum()),
                    "p_mean": round(float(p[m].mean()), 4),
                    "won_rate": round(float(y[m].mean()), 4)})
    return out


def ece(curve) -> float:
    n = sum(b["n"] for b in curve) or 1
    return sum(b["n"] * abs(b["p_mean"] - b["won_rate"]) for b in curve) / n


def export_forest(model, scale: float, names: list[str],
                  n_trees: int, canonical: list[str]) -> dict:
    """sklearn's GradientBoostingClassifier as parallel arrays per tree.

    Each tree is [feature, threshold, left, right]; a leaf carries feature -1
    and its value in `threshold`, which is what lets the run-time walk read
    one array instead of two. The raw decision function is
    `bias + learning_rate * sum(leaf values)` — sklearn's own, so the
    pure-python evaluator and the library agree by construction rather than by
    approximation.
    """
    # The exported split indices point into the FULL run-time vector
    # (`agent/main.py::_tree_features`), never into whatever subset a variant
    # was fitted on. A screening arm that drops a feature therefore changes
    # the forest and nothing else — the agent computes the same sixteen
    # numbers either way and the walk simply never asks for the dropped one.
    remap = [canonical.index(n) for n in names]
    trees = []
    for est in model.estimators_[:n_trees, 0]:
        t = est.tree_
        feat = [remap[int(f)] if int(f) >= 0 else -1 for f in t.feature]
        thr = [float(v) for v in t.threshold]
        left = [int(v) for v in t.children_left]
        right = [int(v) for v in t.children_right]
        for i, f in enumerate(feat):
            if f < 0:                       # leaf: value in place of threshold
                thr[i] = float(t.value[i][0][0])
                left[i] = right[i] = -1
        # Full precision, deliberately. A threshold is the midpoint between
        # two observed values and rounding it moves the split; on features
        # whose neighbouring values differ in the seventh decimal (the
        # projection ones do) that flips a comparison, and the pure-python
        # walk stops being the library. Six-decimal thresholds cost the
        # 685-tree screening arm its 1e-6 verification and bought 40 KB.
        trees.append([feat, thr, left, right])
    return {
        "spec": {
            "source": "scripts/fit_tree_leaf.py",
            "sample": "data/mined/*/positions.jsonl.gz",
            "rule": "one position per episode (earliest turn), rating >= 1000",
            "train_days": list(TRAIN_DAYS), "valid_days": list(VALID_DAYS),
            "features": "agent/main.py::_tree_features (order is definitive)",
        },
        "features": canonical,
        "features_used": names,
        "bias": _init_raw(model),
        "learning_rate": float(model.learning_rate),
        "scale": round(scale, 4),
        "n_trees": len(trees),
        "max_depth": int(model.max_depth),
        "trees": trees,
    }


def _init_raw(model) -> float:
    """The ensemble's constant term, in log-odds."""
    import numpy as np
    raw = model._raw_predict_init(np.zeros((1, model.n_features_in_)))
    return float(np.ravel(raw)[0])


def walk(forest: dict, x) -> float:
    """The bundled pure-python evaluator, run here to verify it."""
    total = 0.0
    for feat, thr, left, right in forest["trees"]:
        node = 0
        f = feat[node]
        while f >= 0:
            node = left[node] if x[f] <= thr[node] else right[node]
            f = feat[node]
        total += thr[node]
    return forest["bias"] + forest["learning_rate"] * total


def stage_fit(args) -> None:
    import numpy as np
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression

    X, y, day, lin, names, canonical, _rows = read_features(tuple(args.drop))
    if args.drop:
        print(f"dropped {', '.join(args.drop)} — fitting on {len(names)} of "
              f"{len(canonical)} features")
    tr = np.isin(day, TRAIN_DAYS)
    va = np.isin(day, VALID_DAYS)
    print(f"train {tr.sum()} episodes over {len(TRAIN_DAYS)} days, "
          f"validate {va.sum()} on {VALID_DAYS[0]}")
    print(f"base rate: train {y[tr].mean():.4f}  valid {y[va].mean():.4f}")

    model = GradientBoostingClassifier(
        n_estimators=args.trees, max_depth=args.depth,
        learning_rate=args.lr, subsample=args.subsample,
        min_samples_leaf=args.min_leaf, random_state=0)
    model.fit(X[tr], y[tr])

    # Early stopping on the held-out DAY: the tree count that minimises
    # validation log-loss, and the ensemble is truncated there before export.
    stages = list(model.staged_decision_function(X[va]))
    losses = [logloss(y[va], 1.0 / (1.0 + np.exp(-np.ravel(s))))
              for s in stages]
    best_n = int(np.argmin(losses)) + 1
    print(f"early stopping: {best_n} of {args.trees} trees "
          f"(valid log-loss {losses[best_n - 1]:.4f}, "
          f"{args.trees} trees would be {losses[-1]:.4f})")

    raw_va = np.ravel(stages[best_n - 1])
    p_tree = 1.0 / (1.0 + np.exp(-raw_va))

    # --- the honest baselines, on the same split -------------------------
    # 1. The shipped linear margin, its own weights, mapped to P(win) by a
    #    one-parameter logistic fitted on the TRAINING days only. AUC does not
    #    depend on that map; Brier does, and refusing it a calibration would
    #    be scoring the baseline on a scale it never claimed.
    lin_map = LogisticRegression(C=1e6, max_iter=1000)
    lin_map.fit(lin[tr].reshape(-1, 1) / 1000.0, y[tr])
    p_lin = lin_map.predict_proba(lin[va].reshape(-1, 1) / 1000.0)[:, 1]

    # 2. A logistic regression over the SAME sixteen features, standardised —
    #    the linear model given exactly the information the forest gets. This
    #    is the comparison that isolates the functional form from the feature
    #    set, and it is the one that says whether the interactions are real.
    mu, sd = X[tr].mean(axis=0), X[tr].std(axis=0)
    sd[sd == 0] = 1.0
    lin_full = LogisticRegression(C=1.0, max_iter=5000)
    lin_full.fit((X[tr] - mu) / sd, y[tr])
    p_linfull = lin_full.predict_proba((X[va] - mu) / sd)[:, 1]

    scores = {}
    for tag, p in (("tree", p_tree), ("linear_margin", p_lin),
                   ("linear_same_features", p_linfull)):
        curve = calibration_curve(y[va], p)
        scores[tag] = {"auc": round(auc(y[va], p), 4),
                       "brier": round(brier(y[va], p), 4),
                       "logloss": round(logloss(y[va], p), 4),
                       "ece": round(ece(curve), 4),
                       "calibration": curve}
    base = brier(y[va], np.full(len(y[va]), y[tr].mean()))
    scores["base_rate"] = {"brier": round(base, 4), "auc": 0.5}

    # A held-out day is 1,286 episodes and the gaps below are hundredths, so
    # the differences get an interval rather than a point. Paired bootstrap:
    # one resample of the validation episodes scores every model, so the
    # sampling noise the three share cancels out of the difference.
    rs = np.random.RandomState(7)
    boot = {k: {"auc": [], "brier": []}
            for k in ("linear_margin", "linear_same_features")}
    n_va = len(y[va])
    yv = y[va]
    for _ in range(2000):
        idx_b = rs.randint(0, n_va, n_va)
        if yv[idx_b].sum() in (0, len(idx_b)):
            continue
        a_t, b_t = auc(yv[idx_b], p_tree[idx_b]), brier(yv[idx_b],
                                                        p_tree[idx_b])
        for tag, p in (("linear_margin", p_lin),
                       ("linear_same_features", p_linfull)):
            boot[tag]["auc"].append(a_t - auc(yv[idx_b], p[idx_b]))
            boot[tag]["brier"].append(b_t - brier(yv[idx_b], p[idx_b]))
    deltas = {}
    print("\ntree minus baseline on the held-out day, paired bootstrap "
          "(2,000 resamples):")
    for tag in boot:
        d = {}
        for metric in ("auc", "brier"):
            v = np.array(boot[tag][metric])
            d[metric] = {"delta": round(float(v.mean()), 4),
                         "ci_lo": round(float(np.percentile(v, 2.5)), 4),
                         "ci_hi": round(float(np.percentile(v, 97.5)), 4)}
            d[metric]["excludes_zero"] = bool(
                d[metric]["ci_lo"] > 0 or d[metric]["ci_hi"] < 0)
        deltas[tag] = d
        print(f"  vs {tag:22s} dAUC {d['auc']['delta']:+.4f} "
              f"[{d['auc']['ci_lo']:+.4f},{d['auc']['ci_hi']:+.4f}]"
              f"{'' if d['auc']['excludes_zero'] else '  NULL'}"
              f"   dBrier {d['brier']['delta']:+.4f} "
              f"[{d['brier']['ci_lo']:+.4f},{d['brier']['ci_hi']:+.4f}]"
              f"{'' if d['brier']['excludes_zero'] else '  NULL'}")

    print(f"\n{'model':24s} {'AUC':>8s} {'Brier':>8s} {'logloss':>9s} "
          f"{'ECE':>7s}")
    for tag in ("tree", "linear_margin", "linear_same_features"):
        s = scores[tag]
        print(f"{tag:24s} {s['auc']:8.4f} {s['brier']:8.4f} "
              f"{s['logloss']:9.4f} {s['ece']:7.4f}")
    print(f"{'base rate':24s} {0.5:8.4f} {base:8.4f}")

    print("\ncalibration, held-out day, deciles of predicted P(win):")
    print(f"  {'n':>6s} {'predicted':>10s} {'observed':>9s}")
    for b in scores["tree"]["calibration"]:
        print(f"  {b['n']:6d} {b['p_mean']:10.4f} {b['won_rate']:9.4f}")

    # TREE_SCALE: put the leaf's spread on the linear margin's, so the trade
    # against a decided determinization's +/-1e6 stays where the shipped
    # evaluator had it. Monotone, so it cannot move an override decision.
    raw_all = np.ravel(list(model.staged_decision_function(X))[best_n - 1])
    scale = float(lin.std() / max(raw_all.std(), 1e-9))
    print(f"\nTREE_SCALE {scale:.2f}  (leaf sd {raw_all.std():.3f} log-odds "
          f"against the linear margin's {lin.std():.1f})")

    forest = export_forest(model, scale, names, best_n, canonical)
    forest["fit"] = {k: {kk: vv for kk, vv in v.items() if kk != "calibration"}
                     for k, v in scores.items()}
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    out_leaf = ANALYSIS / args.name
    out_leaf.write_text(json.dumps(forest, separators=(",", ":")))
    size = out_leaf.stat().st_size / 1024.0
    print(f"wrote {out_leaf.relative_to(ROOT)} ({size:.0f} KB, "
          f"{best_n} trees, depth {args.depth})")

    # --- the pure-python evaluator must be the library, not an approximation
    idx = np.random.RandomState(0).choice(len(X), size=min(1000, len(X)),
                                          replace=False)
    lib = np.ravel(list(model.staged_decision_function(X[idx]))[best_n - 1])
    Xfull = np.array([[float(r[n]) for n in canonical] for r in _rows])
    mine = np.array([walk(forest, list(Xfull[i])) for i in idx])
    worst = float(np.abs(lib - mine).max())

    # sklearn casts to float32 before it walks a tree; the bundled evaluator
    # is float64 arithmetic in the standard library. On a sample sitting
    # inside float32 epsilon of a split threshold the two take different
    # branches, and that is a property of the comparison and not a defect in
    # the export. So the 1e-6 assertion is made where both sides see the same
    # numbers -- inputs rounded through float32 -- and the raw float64 gap is
    # reported beside it as what it is.
    X32 = X.astype(np.float32).astype(np.float64)
    Xfull32 = Xfull.astype(np.float32).astype(np.float64)
    lib32 = np.ravel(list(model.staged_decision_function(X32[idx]))[best_n - 1])
    mine32 = np.array([walk(forest, list(Xfull32[i])) for i in idx])
    worst32 = float(np.abs(lib32 - mine32).max())
    n_boundary = int((np.abs(lib - mine) > 1e-9).sum())
    print(f"pure-python walk vs sklearn over {len(idx)} samples: "
          f"max |diff| {worst32:.3e} at matched precision "
          f"({worst:.3e} raw, {n_boundary} sample(s) on a float32 split "
          f"boundary)")
    if worst32 > 1e-6:
        sys.exit(f"the bundled evaluator disagrees with the library by "
                 f"{worst32}")

    t0 = time.perf_counter()
    for i in idx:
        walk(forest, list(Xfull[i]))
    ms = 1000.0 * (time.perf_counter() - t0) / len(idx)
    print(f"tree walk: {ms:.4f} ms an evaluation")

    (ANALYSIS / args.name.replace(".json", "_fit.json")).write_text(
        json.dumps({
        "spec": forest["spec"],
        "n_train": int(tr.sum()), "n_valid": int(va.sum()),
        "base_rate_train": round(float(y[tr].mean()), 4),
        "base_rate_valid": round(float(y[va].mean()), 4),
        "hyperparameters": {"trees_fitted": args.trees, "trees_kept": best_n,
                            "max_depth": args.depth,
                            "learning_rate": args.lr,
                            "subsample": args.subsample,
                            "min_samples_leaf": args.min_leaf},
        "features": canonical,
        "features_used": names,
        "dropped": list(args.drop),
        "scores": scores,
        "deltas_paired_bootstrap": deltas,
        "tree_scale": round(scale, 4),
        "leaf_sd_logodds": round(float(raw_all.std()), 4),
        "linear_margin_sd": round(float(lin.std()), 1),
        "walk_max_abs_diff": worst32,
        "walk_max_abs_diff_float64": worst,
        "walk_float32_boundary_samples": n_boundary,
        "walk_ms_per_eval": round(ms, 5),
        "json_kb": round(size, 1),
        "importances": {n: round(float(v), 4) for n, v in
                        sorted(zip(names, model.feature_importances_),
                               key=lambda kv: -kv[1])},
        }, indent=1))
    print(f"wrote {args.name.replace('.json', '_fit.json')}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("features", "fit"))
    ap.add_argument("--trees", type=int, default=400)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--subsample", type=float, default=0.8)
    ap.add_argument("--min-leaf", type=int, default=40)
    ap.add_argument("--drop", nargs="*", default=[],
                    help="feature names to withhold — the screening arms")
    ap.add_argument("--name", default="tree_leaf.json",
                    help="filename under data/analysis/ for the forest")
    args = ap.parse_args()
    if args.stage == "features":
        stage_features()
    else:
        stage_fit(args)


if __name__ == "__main__":
    main()
