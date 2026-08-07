"""Mine the dead D34 forest for interactions, then screen what it names.

The forest lost its gate (playbook entry 7, 0.2135 pooled) and the diagnosis
was structural: hundreds of unscreened local weights against a linear spine
whose every coefficient passed a decision-value screen. This script is the
salvage: the forest stays dead as a leaf and works here as a feature-discovery
instrument. Whatever pairwise structure let it describe the held-out day at
AUC 0.692 gets named, lifted into explicit terms, and screened the way every
shipped weight was screened — one candidate at a time, on top of the terms the
evaluator already scores, with the held-out day held out.

Two stages, both under system python3 (numpy/pandas; no sklearn needed — the
committed forest JSON and tree_features.csv are the whole input, so the
analysis reproduces from the repo alone):

    /usr/bin/python3 scripts/mine_interactions.py shap
    /usr/bin/python3 scripts/mine_interactions.py screen

Stage one computes EXACT SHAP interaction values (Lundberg's pairwise
decomposition) over the held-out day, 2026-08-06, by extending the repo's own
path-dependent TreeSHAP (`scripts/shap_traces.py`) with present/absent
conditioning: for i != j,

    Phi[i, j] = (phi_j(f | x_i present) - phi_j(f | x_i absent)) / 2

where conditioning present forces every split on i down x's branch and
conditioning absent distributes every split on i by training cover. The
implementation is verified against brute-force enumeration of the interaction
index on a truncated forest, and the two independent computations of each pair
(condition on i; condition on j) are compared as a symmetry check on every
sample. Pairs are ranked by mean |Phi[i,j] + Phi[j,i]| over the held-out day
and written to data/analysis/interaction_shap.json.

Stage two builds the candidates — the top pairs as named product terms, plus
hinge terms max(0, f - theta) on prize_diff, energy_diff and threat_traj at
training-quartile knots — and screens each one exactly the way
`fit_trajectory_features.py` screened C1/C2: logistic on the training days
(07-31..08-05) over the terms the shipped evaluator scores, plus the one
candidate, with turn fixed effects and went_first; then the held-out day
scores the base model against the candidate model, paired bootstrap, and the
candidate needs BOTH its training coefficient interval and its held-out
improvement interval to exclude zero. Results to
data/analysis/interaction_screen.json. The decision-value verdict — whether
spending the resource produces the advantage, the test hand_diff failed at
0.360 — is an argument, not a fit, and lives in INTERACTION_MINING.md.

No gate games are run here. Nominations only.

A third candidate family, EXPECTED INCOMING DAMAGE, screens beside the mined
interactions (Austin's nomination, 2026-08-07). The shipped `_threat_at` is a
deterministic threshold — the hardest attack payable at t+k against the mean
growth curve — and it feeds the full board-level growth to every slot when it
checks payability, so a wide board is credited as if every attachment could
land everywhere at once. The family replaces both defects at once: per
opponent Pokemon, the probability its attacks are paid for by t+1 / t+2 under
a Poisson attach distribution whose mean is the same archetype growth curve
the shipped projection reads, times the damage — with growth allocated to ONE
attacker at a time (the max variant) or thinned across the attackers (the
split-sum variant), never copied to all slots. Stage three computes those
features over the mined positions with the agent's own machinery (the venv
interpreter, as in fit_tree_leaf.py):

    ~/Desktop/PTCG_AI/.venv/bin/python scripts/mine_interactions.py expected

and writes data/analysis/expected_incoming_features.csv; the screen stage
picks the file up automatically and screens the family under the identical
rules. The precedent the family answers to: `online_lead`, the threshold form
of turns-to-online, was fitted and measured null (C1), so the question the
screen decides is whether the expectation form prices where the threshold
form did not.

Stage four (Austin, 2026-08-07, two nominations that compose):

    ~/Desktop/PTCG_AI/.venv/bin/python scripts/mine_interactions.py evo

* EVOLUTION-AWARE THREAT LADDER — `_attack_profile` reads only a card's own
  printed attacks, so `_threat_at`'s ladder never sees a benched basic's
  future evolved attacks. The variant unions in attacks of evolutions
  reachable within the horizon (one step per own turn), priced by outs
  arithmetic over the side's inferred list (posterior top-1; our side by the
  same posterior over our own board, since the fit-time focal seat's exact
  list is unknown), with a clock-only variant as the pure-coverage envelope.
* DISCOUNTED THREAT INTEGRAL — sum over k=0..5 of gamma^k times the
  threat-at-k differential, on shipped and on evolution-aware profiles: a
  gradient with earlier-is-better pressure where the shipped term is a
  threshold at a single horizon. The gamma grid is profiled, not optimized.

The stage writes the per-k ladders to data/analysis/evo_threat_features.csv;
the screen stage builds every variant from it — swap arms (the variant
replaces threat_traj), side-by-side arms, the gamma profile, and incidence
stats so a null can be attributed (rare situations vs mispriced situations).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from itertools import combinations
from math import factorial
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ANALYSIS = ROOT / "data" / "analysis"
LEAF_JSON = ANALYSIS / "tree_leaf.json"
FEATURES_CSV = ANALYSIS / "tree_features.csv"
SHAP_OUT = ANALYSIS / "interaction_shap.json"
SCREEN_OUT = ANALYSIS / "interaction_screen.json"

TRAIN_DAYS = ("2026-07-31", "2026-08-01", "2026-08-02", "2026-08-03",
              "2026-08-04", "2026-08-05")
VALID_DAY = "2026-08-06"

# The terms the shipped evaluator scores (agent/main.py WEIGHTS): the five
# fitted margin differentials plus C2's threat_traj, which shipped. A
# candidate is screened on top of exactly this set, the same footing the
# trajectory and protection features were priced on.
BASE_TERMS = ["prize_diff", "hp_diff", "energy_diff", "bench_diff",
              "damage_diff", "threat_traj"]
HINGE_FEATURES = ("prize_diff", "energy_diff", "threat_traj")


# ------------------------------------------------------------------ forest IO

def load_forest():
    forest = json.loads(LEAF_JSON.read_text())
    names = forest["features"]
    with FEATURES_CSV.open() as fh:
        rows = list(csv.DictReader(fh))
    Xtr = np.array([[float(r[n]) for n in names] for r in rows
                    if r["day"] in TRAIN_DAYS])
    Xva = np.array([[float(r[n]) for n in names] for r in rows
                    if r["day"] == VALID_DAY])
    return forest, names, Xtr, Xva, rows


def node_cover(tree, X) -> list[float]:
    """Training-sample mass through each node (shap_traces.py's covers)."""
    feat, thr, left, right = tree
    cover = [0.0] * len(feat)
    for row in X:
        node = 0
        while True:
            cover[node] += 1.0
            f = feat[node]
            if f < 0:
                break
            node = left[node] if row[f] <= thr[node] else right[node]
    return cover


# ----------------------------------------------------- conditioned TreeSHAP

def tree_shap_cond(tree, cover, x, phi, ic=-1, present=True) -> None:
    """Path-dependent TreeSHAP over one tree, feature `ic` conditioned out.

    The recursion is `shap_traces.py::tree_shap` with one addition: a split on
    the conditioned feature is not a player. Present, the walk takes x's
    branch at weight one; absent, it takes both branches at their cover
    fractions, carried in the scalar `q` and applied at the leaf. The bypass
    edge enters the path as the (1, 1, -1) dummy the root already uses, which
    the Shapley weights are neutral to (w(s, D) + w(s+1, D) = w(s, D-1)), so
    the surviving features keep exactly their unconditioned combinatorics.
    With ic = -1 nothing is conditioned and this IS the plain algorithm.
    """
    feat, thr, left, right = tree

    def extend(m, pz, po, pi):
        m = [list(e) for e in m] + [[pi, pz, po, 1.0 if not m else 0.0]]
        l = len(m) - 1
        for i in range(l - 1, -1, -1):
            m[i + 1][3] += po * m[i][3] * (i + 1) / (l + 1)
            m[i][3] = pz * m[i][3] * (l - i) / (l + 1)
        return m

    def unwind(m, i):
        l = len(m) - 1
        n = m[l][3]
        m = [list(e) for e in m]
        for j in range(l - 1, -1, -1):
            if m[i][2] != 0:
                t = m[j][3]
                m[j][3] = n * (l + 1) / ((j + 1) * m[i][2])
                n = t - m[j][3] * m[i][1] * (l - j) / (l + 1)
            else:
                m[j][3] = m[j][3] * (l + 1) / (m[i][1] * (l - j))
        for j in range(i, l):
            m[j][0], m[j][1], m[j][2] = m[j + 1][0], m[j + 1][1], m[j + 1][2]
        return m[:l]

    def recurse(node, m, pz, po, pi, q):
        m = extend(m, pz, po, pi)
        f = feat[node]
        if f < 0:
            value = thr[node] * q
            for i in range(1, len(m)):
                w = sum(e[3] for e in unwind(m, i))
                phi[m[i][0]] += w * (m[i][2] - m[i][1]) * value
            return
        if f == ic:
            hot = left[node] if x[f] <= thr[node] else right[node]
            if present:
                recurse(hot, m, 1.0, 1.0, -1, q)
            else:
                cold = right[node] if x[f] <= thr[node] else left[node]
                cn = cover[node] or 1.0
                recurse(hot, m, 1.0, 1.0, -1, q * cover[hot] / cn)
                recurse(cold, m, 1.0, 1.0, -1, q * cover[cold] / cn)
            return
        hot = left[node] if x[f] <= thr[node] else right[node]
        cold = right[node] if x[f] <= thr[node] else left[node]
        iz = io = 1.0
        k = None
        for i in range(1, len(m)):
            if m[i][0] == f:
                k = i
                break
        if k is not None:
            iz, io = m[k][1], m[k][2]
            m = unwind(m, k)
        cn = cover[node] or 1.0
        recurse(hot, m, iz * cover[hot] / cn, io, f, q)
        recurse(cold, m, iz * cover[cold] / cn, 0.0, f, q)

    recurse(0, [], 1.0, 1.0, -1, 1.0)


def sample_interactions(forest, covers, tree_feats, x, n):
    """The full n x n interaction matrix at x, both triangles independently.

    Row i comes from conditioning on i, so Phi[i, j] and Phi[j, i] are two
    separate computations of the same quantity; their gap is returned as the
    symmetry error. Trees that never split on i contribute nothing to row i
    and are skipped — exact, because SHAP is additive across trees.
    """
    lr = forest["learning_rate"]
    phi = [0.0] * (n + 1)
    Phi = np.zeros((n, n))
    for tree, cover, feats in zip(forest["trees"], covers, tree_feats):
        tree_shap_cond(tree, cover, x, phi)          # unconditioned
        for ic in feats:
            pp = [0.0] * (n + 1)
            pa = [0.0] * (n + 1)
            tree_shap_cond(tree, cover, x, pp, ic, True)
            tree_shap_cond(tree, cover, x, pa, ic, False)
            for j in range(n):
                Phi[ic, j] += 0.5 * (pp[j] - pa[j])
    Phi *= lr
    phi_np = np.array(phi[:n]) * lr
    sym_err = float(np.abs(Phi - Phi.T).max())
    Phi = 0.5 * (Phi + Phi.T)
    np.fill_diagonal(Phi, 0.0)
    diag = phi_np - Phi.sum(axis=1)
    Phi[np.diag_indices(n)] = diag
    return Phi, phi_np, sym_err


# ------------------------------------------------- brute-force verification

def brute_interactions(forest, covers, x, n):
    """The interaction index from its definition, by coalition enumeration.

    Phi[i, j] = sum over S excluding i, j of |S|! (n - |S| - 2)! / (2 (n-1)!)
    times the discrete cross-difference f(S+ij) - f(S+i) - f(S+j) + f(S),
    with f(S) the cover-weighted expectation the path-dependent algorithm
    uses. Only affordable on a truncated forest; it exists so the conditioned
    recursion is checked against the definition rather than against itself.
    """
    def f_of(mask):
        total = 0.0
        for (feat, thr, left, right), cover in zip(forest["trees"], covers):
            def ev(node):
                f = feat[node]
                if f < 0:
                    return thr[node]
                if (mask >> f) & 1:
                    nxt = left[node] if x[f] <= thr[node] else right[node]
                    return ev(nxt)
                l, r = left[node], right[node]
                c = cover[node] or 1.0
                return (cover[l] * ev(l) + cover[r] * ev(r)) / c
            total += ev(0)
        return forest["bias"] + forest["learning_rate"] * total

    val = [f_of(m) for m in range(1 << n)]
    w = [factorial(s) * factorial(n - s - 2) / (2.0 * factorial(n - 1))
         for s in range(n - 1)]
    pop = [bin(m).count("1") for m in range(1 << n)]
    Phi = np.zeros((n, n))
    for i, j in combinations(range(n), 2):
        bi, bj = 1 << i, 1 << j
        acc = 0.0
        for m in range(1 << n):
            if m & (bi | bj):
                continue
            acc += w[pop[m]] * (val[m | bi | bj] - val[m | bi]
                                - val[m | bj] + val[m])
        Phi[i, j] = Phi[j, i] = acc
    return Phi


# ------------------------------------------------------------- stage one

_G = {}


def _init_worker(forest, covers, tree_feats, n):
    _G.update(forest=forest, covers=covers, tree_feats=tree_feats, n=n)


def _work(args):
    idx, x = args
    Phi, phi, sym = sample_interactions(_G["forest"], _G["covers"],
                                        _G["tree_feats"], x, _G["n"])
    return idx, Phi, np.abs(Phi), phi, sym


def stage_shap(args) -> None:
    import multiprocessing as mp

    forest, names, Xtr, Xva, _ = load_forest()
    n = len(names)
    print(f"forest: {len(forest['trees'])} trees; covers over {len(Xtr)} "
          f"training rows; {len(Xva)} held-out positions on {VALID_DAY}")
    covers = [node_cover(t, Xtr) for t in forest["trees"]]
    tree_feats = [sorted({f for f in t[0] if f >= 0}) for t in forest["trees"]]

    # --- verification, before anything is trusted ------------------------
    nt = args.verify_trees
    sub = {"trees": forest["trees"][:nt], "bias": forest["bias"],
           "learning_rate": forest["learning_rate"]}
    sub_cov = covers[:nt]
    sub_feats = tree_feats[:nt]
    worst = worst_sym = 0.0
    for k in range(args.verify_samples):
        xv = list(Xva[k])
        fast, _, sym = sample_interactions(sub, sub_cov, sub_feats, xv, n)
        slow = brute_interactions(sub, sub_cov, xv, n)
        off = ~np.eye(n, dtype=bool)
        worst = max(worst, float(np.abs(fast[off] - slow[off]).max()))
        worst_sym = max(worst_sym, sym)
    print(f"conditioned TreeSHAP vs brute-force interaction index, "
          f"{nt} trees x {args.verify_samples} samples: max |diff| "
          f"{worst:.3e}; max asymmetry {worst_sym:.3e}")
    if worst > 1e-8:
        sys.exit("the fast interaction path disagrees with the definition")

    # --- the full held-out day, in parallel, checkpointed ----------------
    scratch = Path(args.checkpoint) if args.checkpoint else None
    sum_abs = np.zeros((n, n))
    sum_phi_abs = np.zeros(n)
    sum_signed = np.zeros((n, n))
    max_sym = 0.0
    done = 0
    t0 = time.perf_counter()
    ctx = mp.get_context("fork")
    with ctx.Pool(args.workers, _init_worker,
                  (forest, covers, tree_feats, n)) as pool:
        jobs = [(i, list(Xva[i])) for i in range(len(Xva))]
        for idx, Phi, aPhi, phi, sym in pool.imap_unordered(
                _work, jobs, chunksize=16):
            sum_abs += aPhi
            sum_signed += Phi
            sum_phi_abs += np.abs(phi)
            max_sym = max(max_sym, sym)
            done += 1
            if done % 200 == 0:
                el = time.perf_counter() - t0
                print(f"  {done}/{len(Xva)} positions, {el:.0f}s, "
                      f"max asymmetry {max_sym:.2e}", flush=True)
                if scratch:
                    scratch.write_text(json.dumps(
                        {"done": done, "sum_abs": sum_abs.tolist(),
                         "sum_signed": sum_signed.tolist(),
                         "max_sym": max_sym}))
    mean_abs = sum_abs / done
    mean_signed = sum_signed / done
    mean_phi_abs = sum_phi_abs / done
    if max_sym > 1e-8:
        sys.exit(f"symmetry check failed at {max_sym}")

    pairs = []
    for i, j in combinations(range(n), 2):
        pairs.append({
            "pair": [names[i], names[j]],
            # Phi is symmetric, so the pair owns Phi[i,j] + Phi[j,i]: the mean
            # of |2 Phi| is the ranking statistic.
            "mean_abs_interaction": round(2.0 * mean_abs[i, j], 6),
            "mean_signed_interaction": round(2.0 * mean_signed[i, j], 6),
        })
    pairs.sort(key=lambda p: -p["mean_abs_interaction"])

    mains = sorted(
        [{"feature": names[i],
          "mean_abs_shap": round(float(mean_phi_abs[i]), 6),
          "mean_abs_self_interaction": round(float(mean_abs[i, i]), 6)}
         for i in range(n)], key=lambda r: -r["mean_abs_shap"])

    SHAP_OUT.write_text(json.dumps({
        "spec": {
            "source": "scripts/mine_interactions.py shap",
            "forest": "data/analysis/tree_leaf.json (the refused D34 leaf, "
                      "419 trees, used here as a discovery instrument only)",
            "sample": f"held-out day {VALID_DAY}, {done} positions, "
                      "one per episode (earliest turn), rating >= 1000",
            "method": "exact path-dependent TreeSHAP interaction values via "
                      "present/absent conditioning; verified against "
                      "brute-force coalition enumeration on a truncated "
                      "forest; both triangles computed independently as a "
                      "symmetry check on every position",
            "units": "log-odds of winning",
        },
        "verify": {"trees": nt, "samples": args.verify_samples,
                   "max_abs_diff_vs_brute_force": worst,
                   "max_asymmetry": max_sym},
        "n_positions": done,
        "main_effects": mains,
        "pairs": pairs,
    }, indent=1))
    el = time.perf_counter() - t0
    print(f"\n{done} positions in {el:.0f}s; top pairs by "
          f"mean |interaction| (log-odds):")
    for p in pairs[:10]:
        print(f"  {p['pair'][0]:22s} x {p['pair'][1]:22s} "
              f"|Phi| {p['mean_abs_interaction']:.4f}  "
              f"signed {p['mean_signed_interaction']:+.4f}")
    print(f"wrote {SHAP_OUT.relative_to(ROOT)}")


# ----------------------------------------- stage three: expected damage

EXPECTED_CSV = ANALYSIS / "expected_incoming_features.csv"
EXPECTED_TERMS = ("exp_in_t1_max", "exp_in_t2_max", "exp_in_t1_split",
                  "exp_in_t2_split", "exp_net_t2")


def _poisson_pmf(mu: float) -> list[float]:
    """P(B = b) for b = 0.., truncated where the tail is below 1e-9.

    Poisson because an attach count is a small-mean count process — the
    curves put board-level Energy gain near one a turn — and because the
    choice is documented here rather than hidden: the mean is the SAME
    archetype growth curve `_traj_growth` the shipped projection reads, so
    the family differs from `_threat_at` only in distribution and
    allocation, never in the growth data.
    """
    mu = max(float(mu), 0.0)
    if mu == 0.0:
        return [1.0]
    pmf = [math.exp(-mu)]
    b, cum = 0, pmf[0]
    while cum < 1.0 - 1e-9 and b < 60:
        b += 1
        pmf.append(pmf[-1] * mu / b)
        cum += pmf[-1]
    return pmf


def _expected_best_damage(profile, e: float, pmf) -> float:
    """E over the attach count of the hardest attack the budget covers.

    For a one-attack Pokemon this is exactly P(online) x damage; with more
    attacks it is the expectation of the best payable one, the same "hardest
    attack" notion `_threat_at` thresholds.
    """
    tot = 0.0
    for b, p in enumerate(pmf):
        budget = e + b
        best = 0.0
        for cost, dmg in profile:
            if cost <= budget and dmg > best:
                best = dmg
        tot += p * best
    return tot


def _expected_damage_features(slots, growth, attack_profile) -> dict:
    """(max, split) expected damage at t+1 and t+2 for one side.

    max: the full growth budget goes to ONE attacker, whichever it makes
    most dangerous — the feed order `energy_plan.plan` searches for, and the
    opposite of `_threat_at`'s copy-to-every-slot. split: the Poisson is
    thinned evenly across the attackers, so a wide board divides its
    attachments instead of multiplying them.
    """
    profs = []
    for e, cid in slots:
        p = attack_profile(int(cid))
        if p:
            profs.append((float(e), p))
    out = {}
    for k in (1, 2):
        if not profs:
            out[f"t{k}_max"] = out[f"t{k}_split"] = 0.0
            continue
        mu = max(float(growth(k)), 0.0)
        pmf = _poisson_pmf(mu)
        out[f"t{k}_max"] = max(_expected_best_damage(p, e, pmf)
                               for e, p in profs)
        pmf_s = _poisson_pmf(mu / len(profs))
        out[f"t{k}_split"] = sum(_expected_best_damage(p, e, pmf_s)
                                 for e, p in profs)
    return out


def _load_agent():
    """fit_tree_leaf.py's loader: the shipped agent beside its own deck.csv."""
    import importlib.util
    import os
    sys.path.insert(0, str(ROOT / "engine"))
    import ptcg.creation  # noqa: F401  — engine bootstrap
    agent_py = ROOT / "agent" / "main.py"
    cwd = os.getcwd()
    os.chdir(agent_py.parent)
    try:
        spec = importlib.util.spec_from_file_location("shipped_agent",
                                                      agent_py)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["shipped_agent"] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        os.chdir(cwd)


def stage_expected(args) -> None:
    import gzip

    agent = _load_agent()
    if not agent.CG_AVAILABLE:
        sys.exit("cg.api did not import — attack profiles would be empty")
    if not agent._curves():
        sys.exit("no trajectory curves loaded — the growth means would be "
                 "the flat fallback")
    if not agent._accel_rates():
        sys.exit("no energy-mechanics KB loaded")

    rows = []
    t_feat = 0.0
    n_feat = 0
    for path in sorted((ROOT / "data" / "mined").glob(
            "*/positions.jsonl.gz")):
        day = path.parent.name
        if day not in TRAIN_DAYS and day != VALID_DAY:
            continue
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
                t0 = time.perf_counter()
                agent._refresh_traj_arch(obs)
                try:
                    (sm, gm, _), (st, gt, _) = agent._traj_projection(
                        cur, mine, theirs)
                    inc = _expected_damage_features(st, gt,
                                                    agent._attack_profile)
                    out = _expected_damage_features(sm, gm,
                                                    agent._attack_profile)
                except Exception:
                    continue
                t_feat += time.perf_counter() - t0
                n_feat += 1
                rows.append({
                    "episode_id": int(rec["episode_id"]),
                    "day": day,
                    "rec_turn": int(rec.get("turn", -1)),
                    "rating": float(rec.get("agent_rating") or 0.0),
                    "went_first": int(rec.get("went_first", -1)),
                    "exp_in_t1_max": round(inc["t1_max"], 4),
                    "exp_in_t2_max": round(inc["t2_max"], 4),
                    "exp_in_t1_split": round(inc["t1_split"], 4),
                    "exp_in_t2_split": round(inc["t2_split"], 4),
                    "exp_net_t2": round(out["t2_max"] - inc["t2_max"], 4),
                })

    # One position an episode, the earliest — the rule every fit here uses.
    keep = {}
    for r in rows:
        cur = keep.get(r["episode_id"])
        if cur is None or r["rec_turn"] < cur["rec_turn"]:
            keep[r["episode_id"]] = r
    sel = [r for r in sorted(keep.values(), key=lambda r: r["episode_id"])
           if r["rating"] >= 1000.0 and r["rec_turn"] > 0
           and r["went_first"] >= 0]
    cols = ["episode_id", "day", "rec_turn"] + list(EXPECTED_TERMS)
    with EXPECTED_CSV.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sel:
            w.writerow(r)
    print(f"{len(rows)} positions read; feature cost "
          f"{1000 * t_feat / max(n_feat, 1):.3f} ms each")
    print(f"{len(sel)} episodes after the rating cut and the "
          f"one-per-episode rule")
    print(f"wrote {EXPECTED_CSV.relative_to(ROOT)}")


def load_expected(rows):
    """The expected-damage columns, aligned to tree_features.csv's rows."""
    if not EXPECTED_CSV.exists():
        return None
    with EXPECTED_CSV.open() as fh:
        by_ep = {int(r["episode_id"]): r for r in csv.DictReader(fh)}
    missing = [r for r in rows if int(r["episode_id"]) not in by_ep]
    if missing:
        sys.exit(f"{len(missing)} tree_features episodes have no "
                 f"expected-damage row — rerun the expected stage")
    return {t: np.array([float(by_ep[int(r["episode_id"])][t])
                         for r in rows]) for t in EXPECTED_TERMS}


# ------------------------------------- stage four: evolution-aware ladder

EVO_CSV = ANALYSIS / "evo_threat_features.csv"
# Per-horizon ladders for both sides, shipped and evolution-aware weighted
# (k = 0..TRAJ_HORIZON), plus the clock-only variant at the shipped k=3.
# The per-k columns exist so the DISCOUNTED THREAT INTEGRAL (Austin's second
# nomination) can be built at screen time for any gamma without re-mining:
# threat_integral(gamma) = sum over k of gamma^k x (ours(k) - theirs(k)).
EVO_KMAX = 5
EVO_COLS = tuple(f"{fam}_{side}_k{k}"
                 for k in range(EVO_KMAX + 1)
                 for fam in ("ship", "evow")
                 for side in ("us", "them")) + ("evof_us_k3", "evof_them_k3")


def _evo_map(cards) -> dict:
    """name -> [(evolution card id, steps)] within two stages.

    Built off the card table's own `evolvesFrom` names, forward: every card
    naming X as its pre-evolution is one step from X, and everything naming
    THAT card is two. Two is the cap the game rules set (basic, stage 1,
    stage 2), so the map is complete, not truncated.
    """
    fwd: dict[str, list[int]] = {}
    names: dict[int, str] = {}
    for cid, card in cards.items():
        nm = getattr(card, "name", None)
        if nm:
            names[cid] = nm
        ef = getattr(card, "evolvesFrom", None)
        if ef:
            fwd.setdefault(ef, []).append(cid)
    out: dict[str, list[tuple[int, int]]] = {}
    for nm in set(fwd) | set(names.values()):
        steps: list[tuple[int, int]] = []
        for e1 in fwd.get(nm, ()):
            steps.append((e1, 1))
            for e2 in fwd.get(names.get(e1, ""), ()):
                steps.append((e2, 2))
        if steps:
            out[nm] = steps
    return out


def _threat_at_evo(slots, growth, k: int, cards, evo_map, attack_profile,
                   avail) -> tuple[float, float]:
    """The shipped ladder with evolution coverage, (weighted, clock-only).

    Everything the shipped `_threat_at` does is kept — full board growth to
    every slot, hardest payable attack, max over slots — and one thing is
    added: each slot's profile unions in the attacks of evolutions reachable
    within the horizon. The evolution clock is one step per own turn (a
    Pokemon cannot evolve the turn it comes down, and these slots are already
    down, so s steps need s of the next k turns). Energy attached to the slot
    survives evolution, so the energy clock is unchanged. `avail` prices the
    evolution cards: the weighted value multiplies damage by P(every chain
    card is playable on schedule); the clock-only value counts any chain
    whose cards exist in the pool at all, the pure coverage repair.
    """
    gain = growth(k)
    best_w = best_f = 0.0
    for e, cid in slots:
        budget = e + gain
        for cost, dmg in attack_profile(int(cid)):
            if cost <= budget:
                if dmg > best_w:
                    best_w = dmg
                if dmg > best_f:
                    best_f = dmg
        nm = getattr(cards.get(int(cid)), "name", None)
        for evo_id, steps in (evo_map.get(nm or "", ()) or ()):
            if steps > k:
                continue
            p = avail(evo_id, steps, k)
            if p <= 0.0:
                continue
            for cost, dmg in attack_profile(evo_id):
                if cost <= budget:
                    if dmg * p > best_w:
                        best_w = dmg * p
                    if dmg > best_f:
                        best_f = dmg
    return best_w, best_f


def stage_evo(args) -> None:
    import gzip

    from ptcg.creation.outs import p_at_least_one

    agent = _load_agent()
    if not agent.CG_AVAILABLE:
        sys.exit("cg.api did not import")
    if not agent._curves():
        sys.exit("no trajectory curves loaded")
    cards, _attacks = agent._tables()
    evo_map = _evo_map(cards)
    K = agent.TRAJ_K
    n_evo_cards = sum(len(v) for v in evo_map.values())
    print(f"evolution map: {len(evo_map)} evolvable names, "
          f"{n_evo_cards} (evolution, step) edges; horizon K = {K}")

    def visible_counts(player, include_hand):
        out: dict[int, int] = {}
        zones = ["active", "bench", "discard"] + (
            ["hand"] if include_hand else [])
        for zone in zones:
            for card in (player.get(zone) or []):
                if card is None:
                    continue
                for cid in [card.get("id")] + \
                        [c.get("id") for c in (card.get("preEvolution")
                                               or []) if c]:
                    if cid:
                        out[int(cid)] = out.get(int(cid), 0) + 1
        return out

    def make_avail(pool_counts, seen, hand_counts, unseen_n, extra_draws,
                   name_counts):
        """P(the chain is playable on schedule) for one side.

        A copy in hand is certain. A copy still unseen prices by outs
        arithmetic: `outs` copies among `unseen_n` unseen cards, with the
        draws the schedule allows — the step-s card of an s-step chain must
        come down by own turn k - (steps - s), so it gets k - (steps - s)
        natural draws (one a turn), plus `extra_draws` for a hidden hand
        (their side: the hand they already hold is drawn cards we have not
        seen). Chain steps multiply — the independence approximation is
        stated, not hidden.
        """
        def avail(evo_id, steps, k):
            p = 1.0
            chain = ([evo_id] if steps == 1 else
                     [("name", getattr(cards.get(evo_id), "evolvesFrom",
                                       "") or ""), evo_id])
            for s, need in enumerate(chain, start=1):
                if isinstance(need, tuple):
                    nm = need[1]
                    in_hand = sum(c for i, c in hand_counts.items()
                                  if getattr(cards.get(i), "name", None)
                                  == nm)
                    outs = name_counts.get(nm, 0)
                else:
                    in_hand = hand_counts.get(need, 0)
                    outs = pool_counts.get(need, 0) - seen.get(need, 0)
                if in_hand > 0:
                    continue
                draws = k - (steps - s) + extra_draws
                ps = p_at_least_one(max(outs, 0), max(unseen_n, 1), draws)
                if ps <= 0.0:
                    return 0.0
                p *= ps
            return p
        return avail

    rows = []
    t_feat = 0.0
    n_feat = 0
    n_post_us = 0
    for path in sorted((ROOT / "data" / "mined").glob(
            "*/positions.jsonl.gz")):
        day = path.parent.name
        if day not in TRAIN_DAYS and day != VALID_DAY:
            continue
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
                t0 = time.perf_counter()
                agent._refresh_traj_arch(obs)
                try:
                    (sm, gm, _), (st, gt, _) = agent._traj_projection(
                        cur, mine, theirs)

                    # their pool: the posterior's top-1 list, the same
                    # inference _refresh_traj_arch keyed the curve on
                    try:
                        post = agent._deck_posterior(
                            obs, top_k=agent.POSTERIOR_TOP_K)
                        pool_t = dict(post[0][0]) if post else {}
                    except Exception:
                        pool_t = {}
                    # our pool: the same machinery pointed at our own board
                    # — at fit time the focal seat's exact list is unknown,
                    # so the runtime's "exact knowledge" is stood in for by
                    # the posterior over OUR visible cards (flipped seat)
                    try:
                        flipped = {"current": dict(cur,
                                                   yourIndex=1 - me)}
                        post_m = agent._deck_posterior(flipped, top_k=1)
                        pool_m = dict(post_m[0][0]) if post_m else {}
                        if pool_m:
                            n_post_us += 1
                    except Exception:
                        pool_m = {}

                    seen_m = visible_counts(mine, include_hand=True)
                    seen_t = visible_counts(theirs, include_hand=False)
                    hand_m = {}
                    for card in (mine.get("hand") or []):
                        if card and card.get("id"):
                            cid = int(card["id"])
                            hand_m[cid] = hand_m.get(cid, 0) + 1

                    def name_outs(pool_counts, seen):
                        out: dict[str, int] = {}
                        for i, c in pool_counts.items():
                            nm = getattr(cards.get(int(i)), "name", None)
                            if nm:
                                out[nm] = out.get(nm, 0) \
                                    + max(c - seen.get(int(i), 0), 0)
                        return out

                    deck_m = int(mine.get("deckCount") or 0)
                    deck_t = int(theirs.get("deckCount") or 0)
                    hand_t_n = int(theirs.get("handCount") or 0)
                    av_m = make_avail(pool_m, seen_m, hand_m, deck_m, 0,
                                      name_outs(pool_m, seen_m))
                    av_t = make_avail(pool_t, seen_t, {},
                                      deck_t + hand_t_n, hand_t_n,
                                      name_outs(pool_t, seen_t))
                    vals = {}
                    for k in range(EVO_KMAX + 1):
                        vals[f"ship_us_k{k}"] = agent._threat_at(sm, gm, k)
                        vals[f"ship_them_k{k}"] = agent._threat_at(st, gt,
                                                                   k)
                        uw, uf = _threat_at_evo(sm, gm, k, cards, evo_map,
                                                agent._attack_profile,
                                                av_m)
                        tw, tf = _threat_at_evo(st, gt, k, cards, evo_map,
                                                agent._attack_profile,
                                                av_t)
                        vals[f"evow_us_k{k}"] = uw
                        vals[f"evow_them_k{k}"] = tw
                        if k == K:
                            vals["evof_us_k3"] = uf
                            vals["evof_them_k3"] = tf
                except Exception:
                    continue
                t_feat += time.perf_counter() - t0
                n_feat += 1
                rows.append({
                    "episode_id": int(rec["episode_id"]),
                    "day": day,
                    "rec_turn": int(rec.get("turn", -1)),
                    "rating": float(rec.get("agent_rating") or 0.0),
                    "went_first": int(rec.get("went_first", -1)),
                    **{c: round(vals[c], 2) for c in EVO_COLS},
                })

    keep = {}
    for r in rows:
        cur_r = keep.get(r["episode_id"])
        if cur_r is None or r["rec_turn"] < cur_r["rec_turn"]:
            keep[r["episode_id"]] = r
    sel = [r for r in sorted(keep.values(), key=lambda r: r["episode_id"])
           if r["rating"] >= 1000.0 and r["rec_turn"] > 0
           and r["went_first"] >= 0]
    cols = ["episode_id", "day", "rec_turn"] + list(EVO_COLS)
    with EVO_CSV.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sel:
            w.writerow(r)
    print(f"{len(rows)} positions read; feature cost "
          f"{1000 * t_feat / max(n_feat, 1):.3f} ms each; our-side "
          f"posterior resolved on {n_post_us} of {n_feat}")
    print(f"{len(sel)} episodes after the rating cut and the "
          f"one-per-episode rule")
    print(f"wrote {EVO_CSV.relative_to(ROOT)}")


def load_evo(rows):
    """The evolution-ladder columns, aligned to tree_features.csv's rows."""
    if not EVO_CSV.exists():
        return None
    with EVO_CSV.open() as fh:
        by_ep = {int(r["episode_id"]): r for r in csv.DictReader(fh)}
    missing = [r for r in rows if int(r["episode_id"]) not in by_ep]
    if missing:
        sys.exit(f"{len(missing)} tree_features episodes have no "
                 f"evo-threat row — rerun the evo stage")
    return {t: np.array([float(by_ep[int(r["episode_id"])][t])
                         for r in rows]) for t in EVO_COLS}


# ------------------------------------------------------------- stage two

def auc(y, s) -> float:
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ss = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    pos, neg = y.sum(), (1 - y).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def logloss(y, p) -> float:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def stage_screen(args) -> None:
    from ptcg.advantage import Z95, coef_table, logit_fit

    forest, names, Xtr_f, Xva_f, rows = load_forest()
    shap_blob = json.loads(SHAP_OUT.read_text())

    day = np.array([r["day"] for r in rows])
    tr = np.isin(day, TRAIN_DAYS)
    va = day == VALID_DAY
    y = np.array([int(r["won"]) for r in rows], dtype=float)
    col = {n: np.array([float(r[n]) for r in rows]) for n in names}
    col["went_first"] = np.array([float(r["went_first"]) for r in rows])
    print(f"train {int(tr.sum())} episodes, validate {int(va.sum())} "
          f"on {VALID_DAY}")

    # --- candidates ------------------------------------------------------
    candidates: list[dict] = []
    for p in shap_blob["pairs"][:args.top_pairs]:
        a, b = p["pair"]
        candidates.append({
            "name": f"{a}*{b}", "kind": "product", "features": [a, b],
            "shap_mean_abs_interaction": p["mean_abs_interaction"],
            "shap_mean_signed_interaction": p["mean_signed_interaction"],
            "values": col[a] * col[b],
            "involves_hand": ("hand" in a) or ("hand" in b),
        })
    expected = load_expected(rows)
    if expected is not None:
        for t in EXPECTED_TERMS:
            candidates.append({
                "name": t, "kind": "expected", "features": [t],
                "values": expected[t], "involves_hand": False,
            })
    else:
        print("no expected_incoming_features.csv — the expected-damage "
              "family is not screened in this run")
    for f in HINGE_FEATURES:
        qs = np.quantile(col[f][tr], (0.25, 0.5, 0.75))
        knots = sorted({round(float(q), 2) for q in qs})
        for theta in knots:
            v = np.maximum(0.0, col[f] - theta)
            if v[tr].std() == 0 or (v[tr] > 0).mean() < 0.05:
                continue        # a knot in the tail is not a candidate
            candidates.append({
                "name": f"hinge({f},{theta:g})", "kind": "hinge",
                "features": [f], "knot": theta, "values": v,
                "involves_hand": False,
            })

    # --- the base model, fitted once -------------------------------------
    turns = sorted({int(col["turn"][k]) for k in np.where(tr)[0]})

    def design(extra=None):
        cols = [np.ones(len(y))]
        nm = ["const"]
        for t in BASE_TERMS:
            cols.append(col[t])
            nm.append(t)
        if extra is not None:
            cols.append(extra)
            nm.append("candidate")
        na = col["no_active_me"]
        if na[tr].std() > 0 and na[tr].sum() >= 10:
            cols.append(na)
            nm.append("no_active_me")
        cols.append(col["went_first"])
        nm.append("went_first")
        for tv in turns[1:]:
            cols.append((col["turn"] == tv).astype(float))
            nm.append(f"turn={tv}")
        return np.column_stack(cols), nm

    Xb, nb = design()
    fb = logit_fit(Xb[tr], y[tr])
    pb = 1.0 / (1.0 + np.exp(-np.clip(Xb[va] @ fb["beta"], -35, 35)))
    base_auc, base_ll = auc(y[va], pb), logloss(y[va], pb)
    print(f"base model held-out: AUC {base_auc:.4f}  logloss {base_ll:.4f}")

    # tercile edges on the training turns, for the sign-stability read
    t_edges = np.quantile(col["turn"][tr], (1 / 3, 2 / 3))

    results = []
    for c in candidates:
        v = c["values"]
        sd = float(v[tr].std())
        vs = v / sd                      # scale only — the term stays named
        Xc, nc = design(vs)
        fc = logit_fit(Xc[tr], y[tr])
        tab = coef_table(fc, nc, Xc[tr])
        crow = next(r for r in tab if r["term"] == "candidate")
        pc = 1.0 / (1.0 + np.exp(-np.clip(Xc[va] @ fc["beta"], -35, 35)))

        rs = np.random.RandomState(7)
        yv = y[va]
        d_auc, d_ll = [], []
        for _ in range(args.boot):
            ib = rs.randint(0, len(yv), len(yv))
            if yv[ib].sum() in (0, len(ib)):
                continue
            d_auc.append(auc(yv[ib], pc[ib]) - auc(yv[ib], pb[ib]))
            d_ll.append(logloss(yv[ib], pc[ib]) - logloss(yv[ib], pb[ib]))

        def ci(vals):
            a = np.array(vals)
            return {"delta": round(float(a.mean()), 5),
                    "ci_lo": round(float(np.percentile(a, 2.5)), 5),
                    "ci_hi": round(float(np.percentile(a, 97.5)), 5)}

        dA, dL = ci(d_auc), ci(d_ll)
        auc_up = dA["ci_lo"] > 0
        ll_down = dL["ci_hi"] < 0

        # sign stability across turn terciles, training days only
        signs = []
        for lo, hi in ((None, t_edges[0]), (t_edges[0], t_edges[1]),
                       (t_edges[1], None)):
            m = tr.copy()
            if lo is not None:
                m &= col["turn"] > lo
            if hi is not None:
                m &= col["turn"] <= hi
            if m.sum() < 300:
                signs.append(None)
                continue
            fs = logit_fit(Xc[m], y[m])
            ts = coef_table(fs, nc, Xc[m])
            r = next(t for t in ts if t["term"] == "candidate")
            signs.append({"n": int(m.sum()), "beta": r["beta"],
                          "sig": r["sig"]})

        corr = {t: round(float(np.corrcoef(v[tr], col[t][tr])[0, 1]), 3)
                for t in BASE_TERMS}
        res = {
            "name": c["name"], "kind": c["kind"],
            "features": c["features"],
            "knot": c.get("knot"),
            "shap_mean_abs_interaction":
                c.get("shap_mean_abs_interaction"),
            "shap_mean_signed_interaction":
                c.get("shap_mean_signed_interaction"),
            "scale_sd_train": round(sd, 4),
            "coef_per_sd": {"beta": crow["beta"], "se": crow["se"],
                            "ci_lo": crow["ci_lo"], "ci_hi": crow["ci_hi"],
                            "z": crow["z"],
                            "excludes_zero": bool(crow["sig"])},
            "heldout": {"auc": round(auc(yv, pc), 4),
                        "logloss": round(logloss(yv, pc), 4),
                        "delta_auc": dA, "delta_logloss": dL,
                        "auc_ci_excludes_zero": bool(auc_up),
                        "logloss_ci_excludes_zero": bool(ll_down)},
            "turn_tercile_betas": signs,
            "corr_with_base_terms": corr,
            "involves_hand": c["involves_hand"],
            "screen_pass_statistical": bool(
                crow["sig"] and (auc_up or ll_down)),
        }
        results.append(res)
        flag = "PASS(stat)" if res["screen_pass_statistical"] else "null"
        print(f"{c['name']:34s} beta/SD {crow['beta']:+.4f} "
              f"[{crow['ci_lo']:+.4f},{crow['ci_hi']:+.4f}] "
              f"dAUC {dA['delta']:+.4f} [{dA['ci_lo']:+.4f},{dA['ci_hi']:+.4f}] "
              f"dLL {dL['delta']:+.5f} [{dL['ci_lo']:+.5f},{dL['ci_hi']:+.5f}] "
              f" {flag}")

    # --- joint arms: is the forest's edge concentrated or diffuse? -------
    # The forest beat the same-features logistic on this day by +0.017 AUC
    # (tree_leaf_fit.json). If that edge lives in the top pairs, entering
    # them together should recover a piece of it; if the joint arms are null
    # too, the edge is diffuse — many small local corrections, no nameable
    # handful — and the nomination list is empty for a reason.
    joint = {}
    sets = {
        "products_top%d" % args.top_pairs:
            [c for c in candidates if c["kind"] == "product"],
        "hinges_all": [c for c in candidates if c["kind"] == "hinge"],
        "expected_family":
            [c for c in candidates if c["kind"] == "expected"],
        "all_candidates": candidates,
    }
    for tag, cs in sets.items():
        if not cs:
            continue
        extras = np.column_stack([c["values"] / c["values"][tr].std()
                                  for c in cs])
        cols = [np.ones(len(y))] + [col[t] for t in BASE_TERMS] \
            + [extras[:, k] for k in range(extras.shape[1])]
        na = col["no_active_me"]
        if na[tr].std() > 0 and na[tr].sum() >= 10:
            cols.append(na)
        cols.append(col["went_first"])
        for tv in turns[1:]:
            cols.append((col["turn"] == tv).astype(float))
        Xj = np.column_stack(cols)
        fj = logit_fit(Xj[tr], y[tr])
        pj = 1.0 / (1.0 + np.exp(-np.clip(Xj[va] @ fj["beta"], -35, 35)))
        rs = np.random.RandomState(7)
        yv = y[va]
        d_auc, d_ll = [], []
        for _ in range(args.boot):
            ib = rs.randint(0, len(yv), len(yv))
            if yv[ib].sum() in (0, len(ib)):
                continue
            d_auc.append(auc(yv[ib], pj[ib]) - auc(yv[ib], pb[ib]))
            d_ll.append(logloss(yv[ib], pj[ib]) - logloss(yv[ib], pb[ib]))

        def jci(vals):
            a = np.array(vals)
            return {"delta": round(float(a.mean()), 5),
                    "ci_lo": round(float(np.percentile(a, 2.5)), 5),
                    "ci_hi": round(float(np.percentile(a, 97.5)), 5)}

        joint[tag] = {"k_terms": len(cs),
                      "heldout_auc": round(auc(yv, pj), 4),
                      "heldout_logloss": round(logloss(yv, pj), 4),
                      "delta_auc": jci(d_auc), "delta_logloss": jci(d_ll)}
        print(f"joint {tag:22s} AUC {joint[tag]['heldout_auc']:.4f} "
              f"dAUC {joint[tag]['delta_auc']['delta']:+.4f} "
              f"[{joint[tag]['delta_auc']['ci_lo']:+.4f},"
              f"{joint[tag]['delta_auc']['ci_hi']:+.4f}] "
              f"dLL {joint[tag]['delta_logloss']['delta']:+.5f} "
              f"[{joint[tag]['delta_logloss']['ci_lo']:+.5f},"
              f"{joint[tag]['delta_logloss']['ci_hi']:+.5f}]")

    # --- the evolution-aware ladder: a swap, not a lift ------------------
    # This candidate is a coverage repair in an already-priced feature, so
    # the screen is different in shape: threat_traj is REPLACED by the
    # variant (and, in a second arm, both enter side by side), and the
    # incidence stats say whether a null is "the situation is rare" or "the
    # situation is priced already".
    evo = load_evo(rows)
    evo_out = {}
    if evo is not None:
        ship = evo["ship_us_k3"] - evo["ship_them_k3"]
        drift = float(np.abs(ship - col["threat_traj"]).max())
        variants = {
            "evo_weighted": evo["evow_us_k3"] - evo["evow_them_k3"],
            "evo_clock_only": evo["evof_us_k3"] - evo["evof_them_k3"],
        }
        pref = {"evo_weighted": "evow_{side}_k3",
                "evo_clock_only": "evof_{side}_k3"}
        inc = {"recomputed_shipped_max_drift": round(drift, 4)}
        for tag, v in variants.items():
            d = v - ship
            per_side = {
                s: evo[pref[tag].format(side=s)] - evo[f"ship_{s}_k3"]
                for s in ("us", "them")}
            inc[tag] = {
                "changed_share": round(float((d != 0).mean()), 4),
                "changed_share_train": round(float((d[tr] != 0).mean()), 4),
                "changed_share_valid": round(float((d[va] != 0).mean()), 4),
                "mean_abs_change": round(float(np.abs(d).mean()), 2),
                "mean_abs_change_when_changed": round(float(
                    np.abs(d[d != 0]).mean()), 2) if (d != 0).any() else 0.0,
                "p90_abs_change": round(float(np.quantile(np.abs(d), 0.9)),
                                        2),
                "changed_share_us_side": round(float(
                    (per_side["us"] != 0).mean()), 4),
                "changed_share_them_side": round(float(
                    (per_side["them"] != 0).mean()), 4),
            }

        def fit_arm(term_cols, term_names):
            cols_a = [np.ones(len(y))] + \
                [col[t] for t in BASE_TERMS if t != "threat_traj"] + \
                list(term_cols)
            nm = ["const"] + [t for t in BASE_TERMS
                              if t != "threat_traj"] + list(term_names)
            na = col["no_active_me"]
            if na[tr].std() > 0 and na[tr].sum() >= 10:
                cols_a.append(na)
                nm.append("no_active_me")
            cols_a.append(col["went_first"])
            nm.append("went_first")
            for tv in turns[1:]:
                cols_a.append((col["turn"] == tv).astype(float))
                nm.append(f"turn={tv}")
            Xa = np.column_stack(cols_a)
            fa = logit_fit(Xa[tr], y[tr])
            ta = coef_table(fa, nm, Xa[tr])
            pa = 1.0 / (1.0 + np.exp(-np.clip(Xa[va] @ fa["beta"], -35,
                                              35)))
            rs = np.random.RandomState(7)
            yv = y[va]
            d_auc, d_ll = [], []
            for _ in range(args.boot):
                ib = rs.randint(0, len(yv), len(yv))
                if yv[ib].sum() in (0, len(ib)):
                    continue
                d_auc.append(auc(yv[ib], pa[ib]) - auc(yv[ib], pb[ib]))
                d_ll.append(logloss(yv[ib], pa[ib])
                            - logloss(yv[ib], pb[ib]))

            def aci(vals):
                a = np.array(vals)
                return {"delta": round(float(a.mean()), 5),
                        "ci_lo": round(float(np.percentile(a, 2.5)), 5),
                        "ci_hi": round(float(np.percentile(a, 97.5)), 5)}

            terms_out = {}
            for name in term_names:
                r = next(t for t in ta if t["term"] == name)
                terms_out[name] = {
                    "beta_per_sd": r["beta"], "ci_lo": r["ci_lo"],
                    "ci_hi": r["ci_hi"], "z": r["z"],
                    "excludes_zero": bool(r["sig"])}
            return {"terms": terms_out,
                    "heldout_auc": round(auc(yv, pa), 4),
                    "heldout_logloss": round(logloss(yv, pa), 4),
                    "delta_auc_vs_base": aci(d_auc),
                    "delta_logloss_vs_base": aci(d_ll)}

        arms = {}
        for tag, v in variants.items():
            sdv = v[tr].std()
            arms[f"swap_{tag}"] = fit_arm([v / sdv], [f"threat_traj_{tag}"])
            sds = ship[tr].std()
            arms[f"both_{tag}"] = fit_arm(
                [ship / sds, v / sdv],
                ["threat_traj_shipped", f"threat_traj_{tag}"])

        # --- the discounted threat integral (Austin's second nomination) -
        # threat_integral(gamma) = sum over k=0..5 of gamma^k x (ours(k) -
        # theirs(k)): a gradient with earlier-is-better pressure where the
        # shipped term is a threshold at one horizon. The shipped k=3 point
        # term is the base model, so each swap arm IS the "integral
        # replaces the point term" comparison. The gamma grid is profiled
        # rather than optimized — five arms are reported with their own
        # intervals, and reading the best one off the held-out day is
        # selection; the profile's SHAPE is the finding, any single gamma's
        # delta is not.
        gammas = (0.4, 0.5, 0.6, 0.7, 0.8)
        fams = {
            "ship": {k: evo[f"ship_us_k{k}"] - evo[f"ship_them_k{k}"]
                     for k in range(EVO_KMAX + 1)},
            "evo": {k: evo[f"evow_us_k{k}"] - evo[f"evow_them_k{k}"]
                    for k in range(EVO_KMAX + 1)},
        }
        integral_profile = {}
        for fam, dd in fams.items():
            spread = np.max([dd[k] for k in dd], axis=0) \
                - np.min([dd[k] for k in dd], axis=0)
            inc[f"integral_{fam}"] = {
                "profile_nonflat_share": round(float(
                    (spread != 0).mean()), 4),
                "profile_spread_mean_when_nonflat": round(float(
                    spread[spread != 0].mean()), 2)
                if (spread != 0).any() else 0.0,
            }
            for g in gammas:
                vI = sum((g ** k) * dd[k] for k in dd)
                sdI = vI[tr].std()
                name = f"integral_{fam}_g{g:g}"
                arm = fit_arm([vI / sdI], [name])
                arm["corr_with_point_k3_train"] = round(float(
                    np.corrcoef(vI[tr], ship[tr])[0, 1]), 3)
                integral_profile[name] = arm

        evo_out = {"incidence": inc, "arms": arms,
                   "integral_profile": integral_profile}
        print("\nevolution-aware ladder:")
        print(f"  recomputed shipped ladder vs tree_features drift "
              f"{drift:.4f}")
        for tag in variants:
            i = inc[tag]
            print(f"  {tag}: changed on {100 * i['changed_share']:.1f}% "
                  f"of positions (us {100 * i['changed_share_us_side']:.1f}%"
                  f", them {100 * i['changed_share_them_side']:.1f}%), "
                  f"mean |change| when changed "
                  f"{i['mean_abs_change_when_changed']:.1f} damage")
        for name, a in {**arms, **integral_profile}.items():
            da, dl = a["delta_auc_vs_base"], a["delta_logloss_vs_base"]
            print(f"  {name:22s} AUC {a['heldout_auc']:.4f} "
                  f"dAUC {da['delta']:+.4f} [{da['ci_lo']:+.4f},"
                  f"{da['ci_hi']:+.4f}] dLL {dl['delta']:+.5f} "
                  f"[{dl['ci_lo']:+.5f},{dl['ci_hi']:+.5f}]"
                  + (f"  r(point)={a['corr_with_point_k3_train']}"
                     if "corr_with_point_k3_train" in a else ""))
            for t, r in a["terms"].items():
                print(f"      {t:28s} beta/SD {r['beta_per_sd']:+.4f} "
                      f"[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}]"
                      f"{'' if r['excludes_zero'] else '  NULL'}")
    else:
        print("no evo_threat_features.csv — the evolution ladder is not "
              "screened in this run")

    SCREEN_OUT.write_text(json.dumps({
        "spec": {
            "source": "scripts/mine_interactions.py screen",
            "sample": "data/analysis/tree_features.csv — one position per "
                      "episode (earliest turn), rating >= 1000",
            "split": {"train": list(TRAIN_DAYS), "valid": VALID_DAY},
            "base_model": BASE_TERMS + ["no_active_me", "went_first",
                                        "turn fixed effects"],
            "candidate_scaling": "divided by its training-day SD; the "
                                 "coefficient reads per SD of the raw term",
            "screen": "training Wald CI excludes zero AND held-out paired "
                      "bootstrap (delta AUC or delta logloss vs the base "
                      "model) excludes zero; the decision-value verdict is "
                      "separate and lives in INTERACTION_MINING.md",
            "bootstrap_resamples": args.boot,
        },
        "base_heldout": {"auc": round(base_auc, 4),
                         "logloss": round(base_ll, 4)},
        "n_train": int(tr.sum()), "n_valid": int(va.sum()),
        "turn_tercile_edges": [float(e) for e in t_edges],
        "candidates": results,
        "joint_arms": joint,
        "evolution_ladder": evo_out,
    }, indent=1))
    print(f"wrote {SCREEN_OUT.relative_to(ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("shap", "screen", "expected", "evo"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--verify-trees", type=int, default=12)
    ap.add_argument("--verify-samples", type=int, default=2)
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--top-pairs", type=int, default=5)
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    if args.stage == "shap":
        stage_shap(args)
    elif args.stage == "expected":
        stage_expected(args)
    elif args.stage == "evo":
        stage_evo(args)
    else:
        stage_screen(args)


if __name__ == "__main__":
    main()
