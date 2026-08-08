"""D47 Phase 2 — train the neural leaf on the sibling-set table.

Two stages so the pandas/pyarrow interpreter and the torch interpreter need
not be the same one (fit falls back to pure-numpy Adam when torch is absent;
same model, same objective, slower):

    python -m ptcg.leaf.train prep  --out data/leaf_train/dataset.npz
    python -m ptcg.leaf.train fit   --data data/leaf_train/dataset.npz \
        --hidden 64 --objective multi --out data/leaf_train/model_h64_multi

Objectives (the D47 racing arms):
    rank   listwise softmax-CE over each sibling set: the chosen candidate's
           score against its siblings' — the generalization of pairwise
           logistic on score differences (fit the CHOICE, not the outcome).
    value  P(win) head alone, BCE on the game outcome.
    multi  rank + VALUE_LAMBDA * value, shared trunk, two heads.

Holdout is BY DAY: --holdout-day (default 2026-08-07) never trains.
Terminal leaves (|terminal| > 0.5) are dropped — the runtime scores them
with the +/-1e6 shortcut, never with the net. Groups keep >= 2 live rows
and their chosen row or are dropped whole.

The export is numpy-only: weights, feature means/stds and names in one
.npz + a .json report. `ptcg.leaf.runtime.NeuralLeaf` is the consumer.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TRAIN_DIR = ROOT / "data" / "leaf_train"
VALUE_LAMBDA = 0.25


# ------------------------------------------------------------------- prep

def stage_prep(args) -> None:
    import pandas as pd
    from ptcg.leaf.features import FEATURE_NAMES
    fcols = [f"f_{n}" for n in FEATURE_NAMES]
    paths = sorted(glob.glob(str(TRAIN_DIR / "*" / "rows.parquet")))
    if args.include:
        keep = set(args.include.split(","))
        paths = [p for p in paths if Path(p).parent.name in keep]
    frames = []
    meta_cols = ["group", "date", "our_archetype", "is_chosen", "won",
                 "terminal", "linear_value", "pool", "agent_rating"]
    for p in paths:
        d = pd.read_parquet(p, columns=meta_cols + fcols)
        print(f"  {Path(p).parent.name}: {len(d)} rows")
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    n0 = len(d)

    # Drop terminal leaves, then re-validate groups.
    d = d[d.terminal.abs() <= 0.5]
    g = d.groupby("group").agg(n=("is_chosen", "size"),
                               ch=("is_chosen", "sum"))
    good = g[(g.n >= 2) & (g.ch == 1)].index
    d = d[d.group.isin(good)].sort_values("group").reset_index(drop=True)
    print(f"rows {n0} -> {len(d)} after terminal/group pruning "
          f"({d.group.nunique()} groups)")

    days = sorted(d.date.unique())
    day_idx = {day: i for i, day in enumerate(days)}
    archs = sorted(d.our_archetype.unique())
    arch_idx = {a: i for i, a in enumerate(archs)}
    gids, ginv = np.unique(d.group.to_numpy(), return_inverse=True)

    X = d[fcols].to_numpy(dtype=np.float32)
    out = {
        "X": X,
        "group": ginv.astype(np.int64),
        "chosen": d.is_chosen.to_numpy(dtype=np.int8),
        "won": d.won.to_numpy(dtype=np.int8),
        "day": d.date.map(day_idx).to_numpy(dtype=np.int16),
        "arch": d.our_archetype.map(arch_idx).to_numpy(dtype=np.int16),
        "linear_value": d.linear_value.to_numpy(dtype=np.float64),
        "day_names": np.array(days),
        "arch_names": np.array(archs),
        "feature_names": np.array(FEATURE_NAMES),
    }
    np.savez_compressed(args.out, **out)
    print(f"wrote {args.out}: X {X.shape}, {len(gids)} groups, "
          f"days {days}")


# -------------------------------------------------------------- fit utils

def group_ptr(group: np.ndarray) -> np.ndarray:
    """CSR-style boundaries for consecutive equal group ids."""
    change = np.flatnonzero(np.diff(group)) + 1
    return np.concatenate([[0], change, [len(group)]])


def softmax_ce_grad(scores, ptr, chosen):
    """Listwise loss and dL/dscore, groups given by ptr. Vectorized via
    segment tricks (no per-group python loop)."""
    gid = np.repeat(np.arange(len(ptr) - 1), np.diff(ptr))
    smax = np.full(len(ptr) - 1, -np.inf)
    np.maximum.at(smax, gid, scores)
    ex = np.exp(scores - smax[gid])
    denom = np.zeros(len(ptr) - 1)
    np.add.at(denom, gid, ex)
    p = ex / denom[gid]
    loss = -np.log(np.maximum(p[chosen == 1], 1e-12)).sum()
    grad = p.copy()
    grad[chosen == 1] -= 1.0
    return loss, grad


class MLP:
    """d -> h -> h -> {score, value}. Pure numpy; Adam; float32."""

    def __init__(self, d, h, seed=0):
        r = np.random.RandomState(seed)
        def init(a, b):
            return (r.randn(a, b) * np.sqrt(2.0 / a)).astype(np.float32)
        self.p = {
            "W1": init(d, h), "b1": np.zeros(h, np.float32),
            "W2": init(h, h), "b2": np.zeros(h, np.float32),
            "ws": init(h, 1)[:, 0] / 10, "bs": np.float32(0.0),
            "wv": init(h, 1)[:, 0] / 10, "bv": np.float32(0.0),
        }
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0

    def forward(self, X):
        p = self.p
        h1 = np.maximum(X @ p["W1"] + p["b1"], 0)
        h2 = np.maximum(h1 @ p["W2"] + p["b2"], 0)
        s = h2 @ p["ws"] + p["bs"]
        v = h2 @ p["wv"] + p["bv"]
        return h1, h2, s, v

    def backward(self, X, h1, h2, gs, gv):
        p, g = self.p, {}
        g["ws"] = h2.T @ gs
        g["bs"] = gs.sum()
        g["wv"] = h2.T @ gv
        g["bv"] = gv.sum()
        dh2 = np.outer(gs, p["ws"]) + np.outer(gv, p["wv"])
        dh2[h2 <= 0] = 0
        g["W2"] = h1.T @ dh2
        g["b2"] = dh2.sum(0)
        dh1 = dh2 @ p["W2"].T
        dh1[h1 <= 0] = 0
        g["W1"] = X.T @ dh1
        g["b1"] = dh1.sum(0)
        return g

    def adam(self, g, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        for k in self.p:
            self.m[k] = b1 * self.m[k] + (1 - b1) * g[k]
            self.v[k] = b2 * self.v[k] + (1 - b2) * np.square(g[k])
            mhat = self.m[k] / (1 - b1 ** self.t)
            vhat = self.v[k] / (1 - b2 ** self.t)
            self.p[k] = (self.p[k] - lr * mhat / (np.sqrt(vhat) + eps)
                         ).astype(np.float32)


def eval_split(scores, values, ptr, chosen, won, linear, arch, arch_names):
    """Top-1 / pairwise vs the linear baseline, overall and per archetype."""
    gid = np.repeat(np.arange(len(ptr) - 1), np.diff(ptr))
    out = {}

    def top1(col):
        best = np.zeros(len(ptr) - 1, dtype=np.int64)
        cur = np.full(len(ptr) - 1, -np.inf)
        for i in range(len(col)):
            if col[i] > cur[gid[i]]:
                cur[gid[i]] = col[i]
                best[gid[i]] = i
        return chosen[best] == 1

    def pairwise(col):
        """P(chosen > sibling), ties count half."""
        wins = ties = tot = 0
        ch_val = np.zeros(len(ptr) - 1)
        ch_val[gid[chosen == 1]] = col[chosen == 1]
        others = chosen == 0
        v = col[others]
        c = ch_val[gid[others]]
        wins = (c > v).sum()
        ties = (c == v).sum()
        tot = len(v)
        return (wins + 0.5 * ties) / max(tot, 1)

    net_t1 = top1(scores)
    lin_t1 = top1(linear)
    out["n_groups"] = int(len(ptr) - 1)
    out["top1_net"] = round(float(net_t1.mean()), 4)
    out["top1_linear"] = round(float(lin_t1.mean()), 4)
    out["pairwise_net"] = round(float(pairwise(scores)), 4)
    out["pairwise_linear"] = round(float(pairwise(linear)), 4)

    # value head vs outcome, on rows with a real label
    lab = won >= 0
    if lab.sum() > 100:
        p = 1.0 / (1.0 + np.exp(-values[lab]))
        y = won[lab].astype(np.float64)
        order = np.argsort(p)
        ranks = np.empty(len(p)); ranks[order] = np.arange(1, len(p) + 1)
        n1, n0 = y.sum(), (1 - y).sum()
        auc = ((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))
        out["value_auc"] = round(float(auc), 4)
        out["value_logloss"] = round(float(
            -(y * np.log(np.maximum(p, 1e-12))
              + (1 - y) * np.log(np.maximum(1 - p, 1e-12))).mean()), 4)

    # per-archetype top-1 (groups keyed by their first row's archetype)
    garch = arch[ptr[:-1]]
    per = {}
    for a in np.unique(garch):
        m = garch == a
        if m.sum() < 25:
            continue
        per[str(arch_names[a])] = {
            "n": int(m.sum()),
            "top1_net": round(float(net_t1[m].mean()), 4),
            "top1_linear": round(float(lin_t1[m].mean()), 4)}
    out["per_archetype"] = per
    return out


def stage_fit(args) -> None:
    z = np.load(args.data, allow_pickle=True)
    X = z["X"]; group = z["group"]; chosen = z["chosen"].astype(np.int64)
    won = z["won"]; day = z["day"]; arch = z["arch"]
    linear = z["linear_value"]
    day_names = [str(x) for x in z["day_names"]]
    arch_names = [str(x) for x in z["arch_names"]]
    feature_names = [str(x) for x in z["feature_names"]]

    hold = day_names.index(args.holdout_day) if args.holdout_day in day_names else -1
    is_hold = day == hold
    print(f"{X.shape[0]} rows, {len(np.unique(group))} groups; "
          f"holdout day {args.holdout_day}: {int(is_hold.sum())} rows")

    mu = X[~is_hold].mean(0)
    sd = X[~is_hold].std(0)
    sd[sd < 1e-6] = 1.0
    Xs = ((X - mu) / sd).astype(np.float32)

    # contiguous-by-group is guaranteed by prep's sort; build ptrs per split
    def split_arrays(mask):
        idx = np.flatnonzero(mask)
        sub = group[idx]
        ptr = group_ptr(sub)
        return idx, ptr

    tr_idx, tr_ptr = split_arrays(~is_hold)
    ho_idx, ho_ptr = split_arrays(is_hold)

    model = MLP(X.shape[1], args.hidden, seed=args.seed)
    rng = np.random.RandomState(args.seed)
    n_groups = len(tr_ptr) - 1
    t0 = time.time()
    val_mask_tr = won[tr_idx] >= 0

    for epoch in range(args.epochs):
        order = rng.permutation(n_groups)
        tot_loss = 0.0
        nb = 0
        for start in range(0, n_groups, args.batch_groups):
            gs = order[start:start + args.batch_groups]
            rows = np.concatenate([np.arange(tr_ptr[g], tr_ptr[g + 1])
                                   for g in gs])
            sizes = tr_ptr[gs + 1] - tr_ptr[gs]
            bptr = np.concatenate([[0], np.cumsum(sizes)])
            bX = Xs[tr_idx[rows]]
            bch = chosen[tr_idx[rows]]
            bwon = won[tr_idx[rows]]
            h1, h2, s, v = model.forward(bX)
            gs_grad = np.zeros_like(s)
            gv_grad = np.zeros_like(v)
            loss = 0.0
            if args.objective in ("rank", "multi"):
                l, gr = softmax_ce_grad(s, bptr, bch)
                loss += l / len(gs)
                gs_grad = gr / len(gs)
            if args.objective in ("value", "multi"):
                lam = 1.0 if args.objective == "value" else VALUE_LAMBDA
                lab = bwon >= 0
                if lab.any():
                    p = 1.0 / (1.0 + np.exp(-v[lab]))
                    y = bwon[lab].astype(np.float32)
                    loss += lam * float(
                        -(y * np.log(np.maximum(p, 1e-12))
                          + (1 - y) * np.log(np.maximum(1 - p, 1e-12))).mean())
                    gv = np.zeros_like(v)
                    gv[lab] = lam * (p - y) / lab.sum()
                    gv_grad = gv
            grads = model.backward(bX, h1, h2, gs_grad.astype(np.float32),
                                   gv_grad.astype(np.float32))
            model.adam(grads, lr=args.lr)
            tot_loss += loss
            nb += 1
        # holdout read each epoch (the value-solo arm ranks by its win head)
        _, _, s_ho, v_ho = model.forward(Xs[ho_idx])
        if args.objective == "value":
            s_ho = v_ho
        ev = eval_split(s_ho, v_ho, ho_ptr, chosen[ho_idx], won[ho_idx],
                        linear[ho_idx], arch[ho_idx], arch_names)
        print(f"epoch {epoch}: loss {tot_loss / max(nb, 1):.4f}  "
              f"holdout top1 {ev['top1_net']} (lin {ev['top1_linear']})  "
              f"pair {ev['pairwise_net']} (lin {ev['pairwise_linear']})  "
              f"vAUC {ev.get('value_auc')}  {time.time() - t0:.0f}s",
              flush=True)

    # final evals
    _, _, s_tr, v_tr = model.forward(Xs[tr_idx])
    if args.objective == "value":
        s_tr = v_tr
    ev_tr = eval_split(s_tr, v_tr, tr_ptr, chosen[tr_idx], won[tr_idx],
                       linear[tr_idx], arch[tr_idx], arch_names)
    _, _, s_ho, v_ho = model.forward(Xs[ho_idx])
    if args.objective == "value":
        s_ho = v_ho
    ev_ho = eval_split(s_ho, v_ho, ho_ptr, chosen[ho_idx], won[ho_idx],
                       linear[ho_idx], arch[ho_idx], arch_names)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # a value-solo arm ranks leaves by its win head: export it as the score
    # head so the runtime consumer never needs to know which arm won
    ws_out = model.p["wv"] if args.objective == "value" else model.p["ws"]
    bs_out = model.p["bv"] if args.objective == "value" else model.p["bs"]
    np.savez(str(out) + ".npz",
             W1=model.p["W1"], b1=model.p["b1"],
             W2=model.p["W2"], b2=model.p["b2"],
             ws=ws_out, bs=bs_out,
             wv=model.p["wv"], bv=model.p["bv"],
             mu=mu.astype(np.float32), sd=sd.astype(np.float32),
             feature_names=np.array(feature_names))
    report = {"args": {k: (v if not isinstance(v, Path) else str(v))
                       for k, v in vars(args).items()},
              "n_rows": int(X.shape[0]), "train": ev_tr, "holdout": ev_ho}
    Path(str(out) + ".json").write_text(json.dumps(report, indent=1))
    print(json.dumps({"train": {k: v for k, v in ev_tr.items()
                                if k != "per_archetype"},
                      "holdout": ev_ho}, indent=1))
    print(f"wrote {out}.npz / .json")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--out", default=str(TRAIN_DIR / "dataset.npz"))
    p.add_argument("--include", default="",
                   help="comma list of leaf_train subdirs; default all")
    p = sub.add_parser("fit")
    p.add_argument("--data", default=str(TRAIN_DIR / "dataset.npz"))
    p.add_argument("--out", required=True)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--objective", choices=("rank", "value", "multi"),
                   default="multi")
    p.add_argument("--holdout-day", default="2026-08-07")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-groups", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=47)
    args = ap.parse_args()
    if args.cmd == "prep":
        stage_prep(args)
    else:
        stage_fit(args)


if __name__ == "__main__":
    main()
