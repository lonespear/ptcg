"""Diagnostic: damage/prize head fidelity in isolation.

The turn-level validation draws (prizes, chip) conditioned on the SAMPLED
attack indicator, so a head error and an action-model attack-rate error are
confounded there. This draws from the head conditioned on each real holdout
turn's REAL attacked flag instead: any residual gap is the head's own
(train = 2026-08-04/05 series pairs; holdout = 2026-08-06). Written after the
pre-registered v1 run; diagnostic only, gates nothing.

    /usr/bin/python3 futures/diag_damage_head.py
"""

from __future__ import annotations

import collections
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import fit_policy as fp
from sampler import TurnSampler
from validate_sampler import load_real_turns, load_real_damage, w1_hist

ARCHS = ["Marnie's Grimmsnarl ex", "Mega Lopunny ex", "Mega Kangaskhan ex",
         "Fezandipiti ex", "Dragapult ex", "Teal Mask Ogerpon ex",
         "Dudunsparce", "Archaludon ex", "FIELD"]


def main():
    rng = random.Random(20260807)
    sampler = TurnSampler(os.path.join(HERE, "policy_train.json"))
    real_turns, _ = load_real_turns()
    real_dmg = load_real_damage()
    by_arch = collections.defaultdict(list)
    for t in real_turns:
        if t["key"] in real_dmg:
            by_arch[t["arch"]].append(t)
            by_arch["FIELD"].append(t)
    out = {}
    for arch in ARCHS:
        rows = by_arch.get(arch, [])
        if not rows:
            continue
        r_chip, r_pr, s_chip, s_pr = [], [], [], []
        for row in rows:
            c, p = real_dmg[row["key"]]
            r_chip.append(c)
            r_pr.append(p)
            for _ in range(3):
                pp, cc = sampler._draw_damage(row["arch"], row["turn"],
                                              row["attacked"], rng)
                s_chip.append(cc)
                s_pr.append(pp)
        out[arch] = {
            "n": len(rows),
            "chip_mean_real": round(sum(r_chip) / len(r_chip), 1),
            "chip_mean_head": round(sum(s_chip) / len(s_chip), 1),
            "chip_w1": round(w1_hist(r_chip, s_chip), 1),
            "prizes_mean_real": round(sum(r_pr) / len(r_pr), 4),
            "prizes_mean_head": round(sum(s_pr) / len(s_pr), 4),
        }
        print("%-26s n=%6d chip %6.1f -> %6.1f (W1 %5.1f)  prizes %.3f -> %.3f"
              % (arch, len(rows), out[arch]["chip_mean_real"],
                 out[arch]["chip_mean_head"], out[arch]["chip_w1"],
                 out[arch]["prizes_mean_real"], out[arch]["prizes_mean_head"]))
    with open(os.path.join(HERE, "results_head_diag.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote futures/results_head_diag.json")


if __name__ == "__main__":
    main()
