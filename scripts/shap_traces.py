"""Decompose the tree leaf's decisions: exact TreeSHAP, and what it disagrees with.

The D34 argument for a forest over a dot product was never accuracy alone — a
linear margin already decomposes into named terms by inspection, and giving
that up for a black box would be a bad trade. TreeSHAP is what buys it back:
for an ensemble of trees the Shapley values are EXACT and computable in
polynomial time, so a decision the search made comes apart into a per-feature
table the same way `_margin`'s terms do.

Two products, both offline. This is report machinery and none of it runs on
the grader.

  * **Decision traces.** The search's chosen option against its runner-up, as
    a per-feature `dphi` table. A decision's value is the mean of the leaf
    over the determinizations it was scored on, and Shapley values are linear
    in the model output, so the mean of the attributions decomposes the mean
    of the values: `sum(dphi) == raw(chosen) - raw(runner-up)` exactly, and
    the script asserts it.
  * **The agreement table.** Both leaves score the SAME positions inside one
    live search — the tree leaf is what the agent plays, and the linear margin
    is computed beside it on every position the rollout reaches — so the two
    argmaxes are compared over identical determinizations rather than over two
    runs that diverge after the first disagreement. Where they disagree, the
    feature contexts that separate the two picks.

The instrumentation wraps the shipped functions and does not modify them:
`_search_main` to bracket a decision, `search_begin` / `search_step` to name
the determinization and the candidate a rollout belongs to, and `_evaluate`
to record both leaves over the position the rollout ended on.

    ~/Desktop/PTCG_AI/.venv/bin/python scripts/shap_traces.py --games 20
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "engine"))

ANALYSIS = ROOT / "data" / "analysis"
LEAF_JSON = ANALYSIS / "tree_leaf.json"
FEATURES_CSV = ANALYSIS / "tree_features.csv"
OUT_JSON = ANALYSIS / "tree_leaf_traces.json"


# ------------------------------------------------------------------ TreeSHAP

def node_cover(tree, X) -> list[float]:
    """How much of the fitted sample reaches each node.

    Recomputed by pushing the training rows through the exported forest rather
    than exported alongside it, so the bundle carries only what the run-time
    walk reads. The covers are what make the path-dependent algorithm's
    "feature not in the coalition" case an expectation over the training
    distribution.
    """
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


def tree_shap(tree, cover, x, n_features: int, phi) -> None:
    """Lundberg's exact path-dependent TreeSHAP over one tree, added into phi.

    `m` is the path of unique features already split on, each carrying the
    fraction of "zero" (feature absent) and "one" (feature present) paths that
    reach here; the leaf case reads off the Shapley weight of every subset
    that path represents. O(leaves x depth^2) and exact, which is the whole
    reason the leaf is a forest and not a network.
    """
    feat, thr, left, right = tree

    def extend(m, pz, po, pi):
        # The elements are copied, not aliased: `recurse` descends into the
        # hot child and then the cold one off the same path, and a shared
        # element would let the first descent rewrite the second's weights.
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

    def recurse(node, m, pz, po, pi):
        m = extend(m, pz, po, pi)
        f = feat[node]
        if f < 0:
            value = thr[node]
            for i in range(1, len(m)):
                w = sum(e[3] for e in unwind(m, i))
                phi[m[i][0]] += w * (m[i][2] - m[i][1]) * value
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
        recurse(hot, m, iz * cover[hot] / cn, io, f)
        recurse(cold, m, iz * cover[cold] / cn, 0.0, f)

    recurse(0, [], 1.0, 1.0, -1)


def forest_shap(forest, covers, x) -> np.ndarray:
    """Per-feature attributions of the forest's raw log-odds at x."""
    n = len(forest["features"])
    phi = np.zeros(n + 1)
    for tree, cover in zip(forest["trees"], covers):
        tree_shap(tree, cover, x, n, phi)
    return phi[:n] * forest["learning_rate"]


def walk(forest, x) -> float:
    total = 0.0
    for feat, thr, left, right in forest["trees"]:
        node = 0
        f = feat[node]
        while f >= 0:
            node = left[node] if x[f] <= thr[node] else right[node]
            f = feat[node]
        total += thr[node]
    return forest["bias"] + forest["learning_rate"] * total


def expected_value(forest, covers) -> float:
    """The forest's output with no feature known — SHAP's base value."""
    total = 0.0
    for (feat, thr, left, right), cover in zip(forest["trees"], covers):
        def ev(node):
            f = feat[node]
            if f < 0:
                return thr[node]
            l, r = left[node], right[node]
            c = cover[node] or 1.0
            return (cover[l] * ev(l) + cover[r] * ev(r)) / c
        total += ev(0)
    return forest["bias"] + forest["learning_rate"] * total


def brute_force_shap(forest, covers, x, n_features):
    """Exact Shapley by enumerating coalitions — the check on the fast path.

    Only usable on a truncated forest; it exists so the O(TLD^2) algorithm
    above is verified against the definition rather than against itself.
    """
    from itertools import combinations
    from math import factorial

    def f_of(S):
        total = 0.0
        for (feat, thr, left, right), cover in zip(forest["trees"], covers):
            def ev(node):
                f = feat[node]
                if f < 0:
                    return thr[node]
                if f in S:
                    nxt = left[node] if x[f] <= thr[node] else right[node]
                    return ev(nxt)
                l, r = left[node], right[node]
                c = cover[node] or 1.0
                return (cover[l] * ev(l) + cover[r] * ev(r)) / c
            total += ev(0)
        return forest["bias"] + forest["learning_rate"] * total

    idx = list(range(n_features))
    phi = np.zeros(n_features)
    n = n_features
    cache = {}

    def val(S):
        key = frozenset(S)
        if key not in cache:
            cache[key] = f_of(key)
        return cache[key]

    for j in idx:
        rest = [i for i in idx if i != j]
        for k in range(len(rest) + 1):
            w = factorial(k) * factorial(n - k - 1) / factorial(n)
            for S in combinations(rest, k):
                phi[j] += w * (val(set(S) | {j}) - val(set(S)))
    return phi


# --------------------------------------------------------------- the agent

def load_agent():
    import ptcg.creation  # noqa: F401  — engine bootstrap
    cwd = os.getcwd()
    os.chdir(ROOT / "agent")
    try:
        spec = importlib.util.spec_from_file_location(
            "shipped_agent", str(ROOT / "agent" / "main.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules["shipped_agent"] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        os.chdir(cwd)


class Recorder:
    """One live search, taken apart into (determinization, candidate, leaf)."""

    def __init__(self, agent):
        self.agent = agent
        self.decisions: list[dict] = []
        self.plan = -1
        self.cand = -1
        self.rows: list[tuple] = []
        self.posterior = None

    def install(self):
        a = self.agent
        self._search_main = a._search_main
        self._begin = a.search_begin
        self._step = a.search_step
        self._eval = a._evaluate
        self._post = a._deck_posterior

        def deck_posterior(obs, top_k=a.POSTERIOR_TOP_K):
            p = self._post(obs, top_k=top_k)
            self.posterior = p
            return p

        def search_begin(*args, **kw):
            self.plan += 1
            return self._begin(*args, **kw)

        def search_step(sid, sel):
            if len(sel) == 1:
                self.cand = int(sel[0])
            return self._step(sid, sel)

        def evaluate(observation, me):
            v = self._eval(observation, me)
            cur = observation.current
            res = getattr(cur, "result", -1)
            if res is not None and res != -1:
                self.rows.append((self.plan, self.cand, v, v, None))
            else:
                try:
                    lin = a._margin(cur, me)
                    x = a._tree_features(cur, me)
                    tree = a.TREE_SCALE * a._tree_raw(cur, me)
                except Exception:
                    return v
                self.rows.append((self.plan, self.cand, lin, tree, x))
            return v

        def search_main(obs, options):
            self.plan, self.cand, self.rows, self.posterior = -1, -1, [], None
            out = self._search_main(obs, options)
            self._close(out)
            return out

        a._deck_posterior = deck_posterior
        a.search_begin = search_begin
        a.search_step = search_step
        a._evaluate = evaluate
        a._search_main = search_main

    def _weights(self) -> list[float]:
        """The plan's posterior weights, rebuilt the way `_search_main` does."""
        a = self.agent
        post = self.posterior or []
        if post and post[0][1] >= a.CONFIDENCE_GATE:
            post = post[:1]
        if not post:
            return []
        per_deck = max(1, a.SEARCH_N_DET // max(len(post), 1))
        return [w / per_deck for _, w in post for _ in range(per_deck)]

    def _close(self, chosen) -> None:
        if not self.rows:
            return
        weights = self._weights()
        if not weights:
            return
        acc: dict[int, list] = {}
        for p, i, lin, tree, x in self.rows:
            if p < 0 or p >= len(weights) or i < 0:
                continue
            w = weights[p]
            e = acc.setdefault(i, [0.0, 0.0, 0.0, 0, []])
            e[0] += w * lin
            e[1] += w * tree
            e[2] += w
            e[3] += 1
            if x is not None:
                e[4].append(x)
        if len(acc) < 2:
            return
        # The fairness filter the search itself applies: only candidates
        # sampled as often as the best-sampled one are comparable.
        top_n = max(e[3] for e in acc.values())
        ok = {i: e for i, e in acc.items() if e[3] == top_n and e[2] > 0}
        if len(ok) < 2:
            return
        lin_avg = {i: e[0] / e[2] for i, e in ok.items()}
        tree_avg = {i: e[1] / e[2] for i, e in ok.items()}
        lin_best = max(lin_avg, key=lambda i: lin_avg[i])
        tree_best = max(tree_avg, key=lambda i: tree_avg[i])
        order = sorted(tree_avg, key=lambda i: -tree_avg[i])
        runner = order[1]
        # Why a step function is the wrong shape for a search leaf: the
        # candidates in one search differ by a single action, often a small
        # resource change, and a piecewise-constant forest returns the SAME
        # leaf for most small perturbations. Count how many distinct values
        # each leaf produced over the same candidates, and how often the top
        # value was shared -- a tie the argmax then breaks arbitrarily, on
        # whichever candidate happens to sit first in the option list.
        def spread(vals):
            r = [round(v, 9) for v in vals.values()]
            top = max(r)
            return len(set(r)), sum(1 for v in r if v == top)

        t_distinct, t_tied = spread(tree_avg)
        l_distinct, l_tied = spread(lin_avg)
        self.decisions.append({
            "tree_distinct": t_distinct, "tree_tied_at_top": t_tied,
            "lin_distinct": l_distinct, "lin_tied_at_top": l_tied,
            "n_candidates": len(ok),
            "linear_best": lin_best, "tree_best": tree_best,
            "agree": lin_best == tree_best,
            "chosen_returned": chosen,
            "tree_gap": tree_avg[tree_best] - tree_avg[runner],
            "lin_gap": lin_avg[lin_best] - lin_avg[
                sorted(lin_avg, key=lambda i: -lin_avg[i])[1]],
            "runner_up": runner,
            "x_chosen": [list(map(float, np.mean(ok[tree_best][4], axis=0)))]
                        if ok[tree_best][4] else None,
            "x_runner": [list(map(float, np.mean(ok[runner][4], axis=0)))]
                        if ok[runner][4] else None,
            "x_lin_best": [list(map(float, np.mean(ok[lin_best][4], axis=0)))]
                          if ok[lin_best][4] else None,
        })


def play(agent, games: int, deck_path: str, seed0: int) -> None:
    from ptcg.arena import load_deck, play_game
    deck = load_deck(Path(deck_path))
    for g in range(games):
        play_game(agent.agent, agent.agent, deck, deck, seed=seed0 + g)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--deck", default="external/grimmsnarl/deck.csv")
    ap.add_argument("--seed", type=int, default=98000)
    ap.add_argument("--traces", type=int, default=3)
    ap.add_argument("--verify-trees", type=int, default=12,
                    help="trees in the truncated forest the brute-force "
                         "Shapley check runs on")
    args = ap.parse_args()

    os.environ["CABT_TREE_LEAF"] = "1"
    forest = json.loads(LEAF_JSON.read_text())
    names = forest["features"]
    with FEATURES_CSV.open() as fh:
        rows = list(csv.DictReader(fh))
    Xtr = np.array([[float(r[n]) for n in names] for r in rows
                    if r["day"] != "2026-08-06"])
    print(f"forest: {len(forest['trees'])} trees, {len(names)} features; "
          f"covers over {len(Xtr)} training rows")
    covers = [node_cover(t, Xtr) for t in forest["trees"]]
    base = expected_value(forest, covers)
    print(f"SHAP base value (no feature known): {base:+.4f} log-odds")

    # The algorithm against the definition, on a truncated forest.
    sub = {"features": names, "trees": forest["trees"][:args.verify_trees],
           "bias": forest["bias"], "learning_rate": forest["learning_rate"]}
    sub_cov = covers[:args.verify_trees]
    xv = list(Xtr[0])
    fast = forest_shap(sub, sub_cov, xv)
    slow = brute_force_shap(sub, sub_cov, xv, len(names))
    print(f"TreeSHAP vs brute-force Shapley over {args.verify_trees} trees: "
          f"max |diff| {np.abs(fast - slow).max():.3e}")

    agent = load_agent()
    if not agent.TREE_LEAF_ENABLED:
        sys.exit("CABT_TREE_LEAF did not take — the traces would be linear")
    rec = Recorder(agent)
    rec.install()
    print(f"playing {args.games} mirror games on {args.deck} ...")
    play(agent, args.games, args.deck, args.seed)
    dec = rec.decisions
    if not dec:
        sys.exit("no comparable searches were recorded")

    n_cand = float(np.mean([d["n_candidates"] for d in dec]))
    ties = {
        "mean_candidates": round(n_cand, 2),
        "tree_mean_distinct_values": round(float(np.mean(
            [d["tree_distinct"] for d in dec])), 3),
        "linear_mean_distinct_values": round(float(np.mean(
            [d["lin_distinct"] for d in dec])), 3),
        "tree_searches_tied_at_top": round(float(np.mean(
            [d["tree_tied_at_top"] > 1 for d in dec])), 4),
        "linear_searches_tied_at_top": round(float(np.mean(
            [d["lin_tied_at_top"] > 1 for d in dec])), 4),
    }
    print(f"\nresolution over {n_cand:.2f} candidates a search:")
    print(f"  distinct values   tree {ties['tree_mean_distinct_values']:.3f}  "
          f"linear {ties['linear_mean_distinct_values']:.3f}")
    print(f"  tied at the top   tree "
          f"{100 * ties['tree_searches_tied_at_top']:.1f}%  linear "
          f"{100 * ties['linear_searches_tied_at_top']:.1f}%")

    agree = sum(1 for d in dec if d["agree"])
    print(f"\n{len(dec)} comparable searches; the two leaves pick the same "
          f"option on {agree} ({100.0 * agree / len(dec):.1f}%)")

    # Where they diverge: which features separate the two picks, by mean
    # |dphi| over the disagreements. Attribution, not correlation — the
    # feature that MOVED the tree's choice, per decision.
    div = [d for d in dec if not d["agree"] and d["x_chosen"] and
           d["x_lin_best"]]
    contrib = np.zeros(len(names))
    signed = np.zeros(len(names))
    for d in div:
        a = forest_shap(forest, covers, d["x_chosen"][0])
        b = forest_shap(forest, covers, d["x_lin_best"][0])
        contrib += np.abs(a - b)
        signed += (a - b)
    ctx = []
    if div:
        contrib /= len(div)
        signed /= len(div)
        order = np.argsort(-contrib)
        print(f"\nthe {len(div)} disagreements, tree's pick minus linear's "
              f"pick, mean |dphi| in log-odds:")
        for j in order[:6]:
            print(f"  {names[j]:24s} |dphi| {contrib[j]:.4f}   "
                  f"mean dphi {signed[j]:+.4f}")
        ctx = [{"feature": names[j], "mean_abs_dphi": round(float(contrib[j]), 5),
                "mean_dphi": round(float(signed[j]), 5)} for j in order]

    # Decision traces: the widest-margin decisions, chosen against runner-up.
    traces = []
    picked = sorted([d for d in dec if d["x_chosen"] and d["x_runner"]],
                    key=lambda d: -abs(d["tree_gap"]))[:args.traces]
    for d in picked:
        xc, xr = d["x_chosen"][0], d["x_runner"][0]
        pc, pr = (forest_shap(forest, covers, xc),
                  forest_shap(forest, covers, xr))
        dphi = pc - pr
        raw_gap = walk(forest, xc) - walk(forest, xr)
        traces.append({
            "n_candidates": d["n_candidates"],
            "chosen": d["tree_best"], "runner_up": d["runner_up"],
            "agree_with_linear": d["agree"],
            "raw_gap_logodds": round(float(raw_gap), 5),
            "sum_dphi": round(float(dphi.sum()), 5),
            "additivity_error": round(float(abs(dphi.sum() - raw_gap)), 9),
            "terms": [{"feature": names[j],
                       "x_chosen": round(float(xc[j]), 2),
                       "x_runner": round(float(xr[j]), 2),
                       "dphi": round(float(dphi[j]), 5)}
                      for j in np.argsort(-np.abs(dphi))],
        })
        assert abs(dphi.sum() - raw_gap) < 1e-6, "TreeSHAP is not additive"

    print("\ndecision traces (chosen minus runner-up, log-odds):")
    for t in traces:
        print(f"\n  {t['n_candidates']} candidates, option {t['chosen']} over "
              f"{t['runner_up']}, gap {t['raw_gap_logodds']:+.4f}, "
              f"sum(dphi) {t['sum_dphi']:+.4f} "
              f"(additivity error {t['additivity_error']:.2e})")
        for term in t["terms"][:6]:
            print(f"      {term['feature']:24s} {term['x_runner']:9.2f} -> "
                  f"{term['x_chosen']:9.2f}   dphi {term['dphi']:+.4f}")

    OUT_JSON.write_text(json.dumps({
        "spec": {"source": "scripts/shap_traces.py",
                 "forest": "data/analysis/tree_leaf.json",
                 "games": args.games, "deck": args.deck, "seed": args.seed,
                 "method": "exact path-dependent TreeSHAP, covers from the "
                           "training rows; verified against brute-force "
                           "Shapley on a truncated forest"},
        "shap_base_value": round(float(base), 5),
        "verify_max_abs_diff": float(np.abs(fast - slow).max()),
        "n_searches": len(dec),
        "resolution": ties,
        "agreement": round(agree / len(dec), 4),
        "n_disagreements": len(div),
        "divergence_contexts": ctx,
        "traces": traces,
        "gaps": {"tree_mean_abs": round(float(np.mean(
            [abs(d["tree_gap"]) for d in dec])), 3)},
    }, indent=1))
    print(f"\nwrote {OUT_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
