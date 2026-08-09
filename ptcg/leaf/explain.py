"""D47 interpretability — per-decision feature attributions on named inputs.

For a sibling set from the training table, score every candidate with the
trained net and decompose the score DIFFERENCE between the net's pick and
the runner-up into per-feature contributions (gradient x input, exact for
the ReLU net at the evaluation point; the runtime exposes it).

    python -m ptcg.leaf.explain --model data/leaf_train/model_h64_multi.npz \
        --n 3 [--day 2026-08-07] [--arch "Marnie's Grimmsnarl ex"]

Output: readable text + a JSON blob per example for the Strategy report.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TRAIN_DIR = ROOT / "data" / "leaf_train"


def main() -> None:
    import pandas as pd
    from ptcg.leaf.runtime import NeuralLeaf
    from ptcg.leaf.features import FEATURE_NAMES

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--day", default="2026-08-07")
    ap.add_argument("--arch", default="")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--seed", type=int, default=47)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    net = NeuralLeaf(args.model)
    fcols = [f"f_{n}" for n in FEATURE_NAMES]
    path = TRAIN_DIR / args.day / "rows.parquet"
    d = pd.read_parquet(path)
    if args.arch:
        d = d[d.our_archetype == args.arch]
    d = d[d.terminal.abs() <= 0.5]
    sizes = d.groupby("group").size()
    ok = sizes[(sizes >= 3) & (sizes <= 8)].index
    rng = np.random.RandomState(args.seed)
    picks = rng.choice(ok, size=min(args.n, len(ok)), replace=False)

    blobs = []
    for gid in picks:
        g = d[d.group == gid].reset_index(drop=True)
        X = g[fcols].to_numpy(dtype=np.float32)
        scores = np.array([net.score_features(x) for x in X])
        order = np.argsort(-scores)
        best, second = order[0], order[1]
        chosen = int(g.is_chosen.to_numpy().argmax())
        meta = g.iloc[0]
        print(f"\n=== group {gid}  {meta.our_archetype} vs "
              f"{meta.opp_archetype}  turn {meta.turn}  "
              f"({meta.date}, rating {meta.agent_rating:.0f})")
        for i in order:
            tag = []
            if i == chosen:
                tag.append("FIELD CHOICE")
            if i == best:
                tag.append("net top")
            print(f"  cand {int(g.cand[i]):3d}  net {scores[i]:+.3f}  "
                  f"linear {g.linear_value[i]:+9.1f}  "
                  f"P(win) {net.win_prob(X[i]):.3f}  {' / '.join(tag)}")
        # why the top candidate beats the runner-up: attribution on the diff
        at_b = dict(net.attributions(X[best]))
        at_s = dict(net.attributions(X[second]))
        diff = {k: at_b[k] - at_s[k] for k in at_b}
        top = sorted(diff.items(), key=lambda t: -abs(t[1]))[:8]
        print("  top drivers of (net top - runner-up):")
        for name, v in top:
            raw_b = float(g[f"f_{name}"][best])
            raw_s = float(g[f"f_{name}"][second])
            print(f"    {name:<24} {v:+8.3f}   ({raw_b:.2f} vs {raw_s:.2f})")
        blobs.append({
            "group": int(gid), "date": str(meta.date),
            "our_archetype": str(meta.our_archetype),
            "opp_archetype": str(meta.opp_archetype),
            "turn": int(meta.turn),
            "candidates": [
                {"cand": int(g.cand[i]), "net": float(scores[i]),
                 "linear": float(g.linear_value[i]),
                 "p_win": float(net.win_prob(X[i])),
                 "is_chosen": bool(g.is_chosen[i])} for i in range(len(g))],
            "net_top_vs_runnerup_drivers": [
                {"feature": n, "delta_contrib": float(v)} for n, v in top],
        })
    if args.out:
        Path(args.out).write_text(json.dumps(blobs, indent=1))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
