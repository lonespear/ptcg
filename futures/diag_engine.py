"""Phase 0.5 post-verdict diagnostics (labeled as such; the gate is
validate_sampler_engine.py's and stands as computed).

Three questions, all answered with the same engine playout machinery:

1. WHERE does the attack-rate deficit come from? Per-visit end_turn hazard on
   multi-option menus, by within-turn ordinal, real vs sampled.
2. HOW MUCH of the prize deficit is attack timing vs attack quality?
   Prizes per attacking turn, real vs sampled (the conditional-KO rate).
3. IS the failure a one-rule fix? A variant sampler that never ends the turn
   while any other table type is on the menu (end_turn propensity zeroed on
   multi-type menus, the maximal version of raising the end hazard's floor),
   re-scored under the same pre-registered tolerances.

    ~/miniforge3/bin/python3 futures/diag_engine.py
"""

from __future__ import annotations

import collections
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import fit_policy as fp
import validate_sampler as vs
import validate_sampler_engine as ve
from sampler import TurnSampler


class NoEarlyEndSampler(TurnSampler):
    """end_turn only when it is the sole table type the menu offers."""

    def sample_action(self, band, arch, turn, ordinal, available, rng):
        avail = [t for t in available if t != "end_turn"]
        if not avail:
            return "end_turn"
        return super().sample_action(band, arch, turn, ordinal, avail, rng)


def run(sampler, roots, dturns, series_base, tag):
    per_turn = []
    _, _, disruption_ids = fp.card_tables()
    for key in sorted(roots):
        entry_k, rec = roots[key]
        dt = dturns.get(key)
        if dt is None or key not in series_base:
            continue
        obs = rec["observation"]
        players = obs["current"]["players"]
        me = obs["current"]["yourIndex"]
        try:
            post_me = ve.seat_posterior(players[me])
            import main as ag
            post_opp = ag._deck_posterior(obs, top_k=ve.POSTERIOR_TOP_K)
        except Exception:
            continue
        if not post_me or not post_opp:
            continue
        base_counters, base_prizes = series_base[key]
        playouts = []
        for j in range(ve.K):
            rng = random.Random((key[0] * 1000003) ^ (key[1] << 21)
                                ^ (key[2] << 24) ^ (j * 7919) ^ ve.SEED)
            po = ve.play_turn(rec, entry_k, sampler, disruption_ids,
                              post_me, post_opp, rng)
            if po is None or not po.get("completed"):
                continue
            po["chip"] = 10 * round(
                max(0, min(fp.CHIP_CAP, po["end_counters"] - base_counters))
                / 10)
            po["prizes"] = max(0, base_prizes - po["prizes_left"])
            playouts.append(po)
        if not playouts:
            continue
        suffix = [v for v in dt["visits"] if v[0] >= entry_k]
        prefix = [v for v in dt["visits"] if v[0] < entry_k]
        per_turn.append({
            "key": key, "arch": dt["arch"], "entry_k": entry_k,
            "real_suffix_multi": [t for _, t, m in suffix if m and t],
            "real_suffix_len": len(suffix),
            "real_attacked": dt["attacked"],
            "real_attached": dt["attached"],
            "real_attached_sfx": any(t == "attach" for _, t, _ in suffix),
            "prefix_attached": any(t == "attach" for _, t, _ in prefix),
            "playouts": playouts,
        })
    print("[%s] %d turns played" % (tag, len(per_turn)), file=sys.stderr)
    return per_turn


def conditional_prizes(per_turn, real_dmg):
    """prizes per turn, split by whether the turn attacked."""
    out = {}
    for side in ("real", "sampled"):
        n_atk = n_no = 0
        p_atk = p_no = 0.0
        for row in per_turn:
            if row["key"] not in real_dmg:
                continue
            if side == "real":
                _, p = real_dmg[row["key"]]
                if row["real_attacked"]:
                    n_atk += 1
                    p_atk += p
                else:
                    n_no += 1
                    p_no += p
            else:
                for po in row["playouts"]:
                    if po["attacked"]:
                        n_atk += 1
                        p_atk += po["prizes"]
                    else:
                        n_no += 1
                        p_no += po["prizes"]
        out[side] = {
            "n_attacking": n_atk,
            "prizes_per_attacking_turn": round(p_atk / n_atk, 4) if n_atk else None,
            "n_non_attacking": n_no,
            "prizes_per_non_attacking_turn": round(p_no / n_no, 4) if n_no else None,
        }
    return out


def end_hazard(per_turn, dturns):
    """P(end_turn | multi-option menu at ordinal k), real vs sampled.

    Real side counts only the suffix visits of the played turns, so both
    sides condition on the same turns and entry points.
    """
    real = collections.defaultdict(lambda: [0, 0])
    samp = collections.defaultdict(lambda: [0, 0])
    for row in per_turn:
        dt = dturns[row["key"]]
        for k, t, m in dt["visits"]:
            if k < row["entry_k"] or not m or not t:
                continue
            b = str(min(k, 3)) + ("+" if k >= 3 else "")
            real[b][1] += 1
            real[b][0] += t == "end_turn"
        for po in row["playouts"]:
            for n, t, k in po["visits"]:
                if n <= 1:
                    continue
                b = str(min(k, 3)) + ("+" if k >= 3 else "")
                samp[b][1] += 1
                samp[b][0] += t == "end_turn"
    return {b: {"real": round(real[b][0] / real[b][1], 4) if real[b][1] else None,
                "n_real": real[b][1],
                "sampled": round(samp[b][0] / samp[b][1], 4) if samp[b][1] else None,
                "n_sampled": samp[b][1]}
            for b in sorted(set(real) | set(samp))}


def main():
    t0 = time.time()
    print("loading holdout ...", file=sys.stderr)
    dturns, ordmap = ve.load_decision_turns()
    real_dmg = vs.load_real_damage()
    series_base = ve.load_series_base()
    roots = ve.load_position_roots(ordmap)
    disr_real, _ = vs.load_real_menus()
    disr_real = {("FIELD" if a == "*" else a): c for a, c in disr_real.items()}
    vol = collections.Counter(t["arch"] for t in dturns.values())
    total_turns = sum(vol.values())

    out = {"holdout_day": fp.HOLDOUT_DAY, "K": ve.K, "seed": ve.SEED,
           "note": "post-verdict diagnostics; the Phase 0.5 gate is "
                   "results_engine.json and stands as computed"}

    # --- baseline playouts (same seed: reproduces the run of record) --------
    base = TurnSampler(os.path.join(HERE, "policy_train.json"))
    per_turn = run(base, roots, dturns, series_base, "baseline")
    out["end_hazard_by_ordinal"] = end_hazard(per_turn, dturns)
    out["conditional_prizes_baseline"] = conditional_prizes(per_turn, real_dmg)

    # --- variant: never end early -------------------------------------------
    noend = NoEarlyEndSampler(os.path.join(HERE, "policy_train.json"))
    per_turn_v = run(noend, roots, dturns, series_base, "no-early-end")
    out["conditional_prizes_noearlyend"] = conditional_prizes(per_turn_v,
                                                              real_dmg)
    by_arch = collections.defaultdict(list)
    for row in per_turn_v:
        by_arch[row["arch"]].append(row)
    table = {}
    go_volume = 0
    for arch in ve.ARCHS + ["FIELD"]:
        rows = per_turn_v if arch == "FIELD" else by_arch.get(arch, [])
        if not rows:
            continue
        rec = ve.summarize(rows, real_dmg, disr_real, arch)
        if rec is None:
            continue
        table[arch] = rec
        if arch != "FIELD" and rec["verdict"] == "GO":
            go_volume += vol.get(arch, 0)
        print("  [no-early-end] %-26s n=%4d TV=%.3f chipΔ=%s atkΔ=%+.3f "
              "przΔ=%s  %s"
              % (arch, rec["n_real_turns"], rec["tv"],
                 "-" if rec["chip_mean_real"] is None else
                 "%+.1f" % (rec["chip_mean_sampled"] - rec["chip_mean_real"]),
                 rec["attack_rate_sampled"] - rec["attack_rate_real"],
                 "-" if rec["prizes_mean_real"] is None else
                 "%+.3f" % (rec["prizes_mean_sampled"]
                            - rec["prizes_mean_real"]),
                 rec["verdict"]), file=sys.stderr)
    table["_coverage_if_gated"] = round(go_volume / total_turns, 4)
    out["noearlyend_table"] = table

    dest = os.path.join(HERE, "results_engine_diag.json")
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote %s  (%.0fs)" % (dest, time.time() - t0))


if __name__ == "__main__":
    main()
