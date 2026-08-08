"""Phase 0 kill test: sampled opponent turns vs real held-out turns.

Everything here runs on HOLDOUT_DAY only (2026-08-06), which fit_policy.py
never read. For each archetype, every real holdout seat-turn provides the
conditioning (band, archetype, turn number) and the sampler draws K synthetic
turns per real one, so the two sides see the same conditioning distribution by
construction. Metrics and verdict criteria are pre-registered in
futures/README.md; this file computes, it does not decide.

    /usr/bin/python3 futures/validate_sampler.py
"""

from __future__ import annotations

import collections
import json
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import fit_policy as fp                      # shared helpers, no side effects
from sampler import TurnSampler, TYPES

ROOT = os.path.dirname(HERE)
HOLDOUT = fp.HOLDOUT_DAY
K = 3                                        # sampled turns per real turn
SEED = 20260807

ARCHS = ["Marnie's Grimmsnarl ex", "Mega Lopunny ex", "Mega Kangaskhan ex",
         "Fezandipiti ex", "Dragapult ex", "Teal Mask Ogerpon ex",
         "Dudunsparce", "Archaludon ex"]

# Pre-registered tolerances (README) — restated here only to print verdicts.
TOL_TV = 0.10
TOL_CHIP_W1 = 20.0
TOL_PRIZES = 0.05
TOL_ATTACK = 0.05
TOL_ATTACH = 0.05          # flag only
TOL_DISR = 0.03            # flag only
COVERAGE_GO = 0.70


def day_path(fname):
    return os.path.join(ROOT, "data", "mined", HOLDOUT, fname)


# ---------------------------------------------------------------------------
# real side
# ---------------------------------------------------------------------------
def load_real_turns():
    import pyarrow.parquet as pq
    cols = ["episode_id", "seat", "step", "turn", "context", "n_options",
            "chosen_option_type", "our_archetype", "agent_rating"]
    d = pq.read_table(day_path("decisions.parquet"), columns=cols).to_pydict()
    n = len(d["turn"])
    groups = collections.defaultdict(list)
    for i in range(n):
        if d["context"][i] != fp.CTX_MAIN or d["turn"][i] < 1:
            continue
        groups[(d["episode_id"][i], d["seat"][i], d["turn"][i])].append(i)
    turns = []
    ordmap = {}
    for (ep, seat, t), idxs in groups.items():
        idxs.sort(key=lambda i: d["step"][i])
        for k, i in enumerate(idxs):
            ordmap[(ep, seat, d["step"][i])] = k
        types_multi = [fp.TYPE_NAME.get(d["chosen_option_type"][i])
                       for i in idxs if d["n_options"][i] > 1]
        types_all = [fp.TYPE_NAME.get(d["chosen_option_type"][i]) for i in idxs]
        turns.append({
            "key": (ep, seat, t), "turn": t,
            "arch": d["our_archetype"][idxs[0]],
            "band": fp.band_of(d["agent_rating"][idxs[0]]),
            "types": [x for x in types_multi if x],
            "attacked": "attack" in types_all,
            "attached": "attach" in types_all,
        })
    return turns, ordmap


def load_real_damage():
    import pyarrow.parquet as pq
    cols = ["episode_id", "seat", "turn",
            "damage_dealt_cumulative", "damage_taken_cumulative",
            "prizes_remaining", "opp_prizes_remaining"]
    d = pq.read_table(day_path("series.parquet"), columns=cols).to_pydict()
    n = len(d["turn"])
    rows = {(d["episode_id"][i], d["seat"][i], d["turn"][i]): i
            for i in range(n)}
    out = {}
    for i in range(n):
        ep, s, t = d["episode_id"][i], d["seat"][i], d["turn"][i]
        j = rows.get((ep, 1 - s, t + 1))
        if j is None:
            continue
        chip = max(0, min(fp.CHIP_CAP,
                          d["damage_taken_cumulative"][j]
                          - d["damage_dealt_cumulative"][i]))
        prizes = max(0, d["prizes_remaining"][i]
                     - d["opp_prizes_remaining"][j])
        out[(ep, s, t)] = (10 * round(chip / 10), prizes)
    return out


def load_real_menus():
    """Holdout positions: disruption-per-play and attack-choice agreement."""
    import gzip
    cards, attacks, disruption_ids = fp.card_tables()
    disr = collections.defaultdict(collections.Counter)
    agree = collections.Counter()
    with gzip.open(day_path("positions.jsonl.gz"), "rt") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("context") != fp.CTX_MAIN:
                continue
            obs = r.get("observation")
            if isinstance(obs, str):
                obs = json.loads(obs)
            if not isinstance(obs, dict):
                continue
            opts = (obs.get("select") or {}).get("option") or []
            chosen = r.get("chosen") or []
            if len(opts) <= 1 or not chosen:
                continue
            ci = chosen[0]
            if not (0 <= ci < len(opts)):
                continue
            pick = opts[ci]
            arch = r.get("our_archetype")
            if pick.get("type") == fp.OPT_PLAY:
                cur = obs.get("current") or {}
                players = cur.get("players") or []
                me = cur.get("yourIndex", 0)
                if len(players) == 2:
                    hand = players[me].get("hand") or []
                    hi = pick.get("index")
                    if isinstance(hi, int) and 0 <= hi < len(hand):
                        disr[arch]["n_play"] += 1
                        disr["*"]["n_play"] += 1
                        if (hand[hi] or {}).get("id") in disruption_ids:
                            disr[arch]["disr"] += 1
                            disr["*"]["disr"] += 1
            if pick.get("type") == fp.OPT_ATTACK:
                _attack_agreement(obs, opts, ci, cards, attacks, agree)
    return disr, agree


def _attack_agreement(obs, opts, ci, cards, attacks, agree):
    """Does the field's pick match the sampler's rule: cheapest KO else max dmg."""
    cur = obs.get("current") or {}
    players = cur.get("players") or []
    me = cur.get("yourIndex", 0)
    if len(players) != 2:
        return
    act = (players[1 - me].get("active") or [None])
    target = act[0] if act else None
    thp = target.get("hp") if isinstance(target, dict) else None
    weak = None
    if isinstance(target, dict):
        weak = (cards.get(target.get("id")) or {}).get("weakness")

    def dmg(o):
        a = attacks.get(o.get("attackId"))
        if a is None:
            return 0
        d = a.get("damage") or 0
        if d and weak is not None and any(
                e == weak for e in (a.get("energies") or []) if e):
            d *= 2
        return d

    cand = [(i, dmg(o)) for i, o in enumerate(opts)
            if o.get("type") == fp.OPT_ATTACK]
    if len(cand) < 2:
        return
    kos = [(i, d) for i, d in cand if thp is not None and d >= thp]
    if kos:
        cheap = min(d for _, d in kos)
        rule = {i for i, d in kos if d == cheap}
    else:
        big = max(d for _, d in cand)
        rule = {i for i, d in cand if d == big}
    agree["n"] += 1
    agree["match"] += int(ci in rule)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def mix(seqs):
    c = collections.Counter()
    for s in seqs:
        c.update(s)
    n = sum(c.values())
    return ({t: c[t] / n for t in TYPES} if n else {}), n


def tv(a, b):
    return 0.5 * sum(abs(a.get(t, 0.0) - b.get(t, 0.0)) for t in TYPES)


def w1_hist(xs, ys, step=10):
    """Wasserstein-1 between two samples of multiples of `step`."""
    if not xs or not ys:
        return None
    ca, cb = collections.Counter(xs), collections.Counter(ys)
    grid = range(0, fp.CHIP_CAP + step, step)
    na, nb = len(xs), len(ys)
    Fa = Fb = 0.0
    dist = 0.0
    for g in grid:
        Fa += ca.get(g, 0) / na
        Fb += cb.get(g, 0) / nb
        dist += abs(Fa - Fb) * step
    return dist


def jeffreys(k, n):
    if n == 0:
        return (0.0, 1.0)
    from math import lgamma
    # 95% Jeffreys via normal approx fallback (no scipy): use Wilson interval
    p = k / n
    z = 1.96
    den = 1 + z * z / n
    ctr = (p + z * z / (2 * n)) / den
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, ctr - hw), min(1.0, ctr + hw))


def menu_replay(sampler, ordmap):
    """Decision-level test on REAL holdout menus: no availability model.

    For every holdout main menu with >1 option, the policy's distribution over
    the types actually offered is compared with the field's pick. This
    isolates the reply tables from the offline availability simulation: any
    divergence here is the tables', any divergence only in the turn-level test
    is the availability model's.
    """
    import gzip
    out = collections.defaultdict(lambda: {
        "n": 0, "top1": 0, "logloss": 0.0, "logloss_uniform": 0.0,
        "expected": collections.Counter(), "realized": collections.Counter()})
    with gzip.open(day_path("positions.jsonl.gz"), "rt") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("context") != fp.CTX_MAIN:
                continue
            obs = r.get("observation")
            if isinstance(obs, str):
                obs = json.loads(obs)
            if not isinstance(obs, dict):
                continue
            opts = (obs.get("select") or {}).get("option") or []
            chosen = r.get("chosen") or []
            if len(opts) <= 1 or not chosen:
                continue
            ci = chosen[0]
            if not (0 <= ci < len(opts)):
                continue
            k = ordmap.get((r["episode_id"], r["seat"], r["step"]))
            turn = r.get("turn")
            if k is None or not turn or turn < 1:
                continue
            offered = sorted({fp.TYPE_NAME[o.get("type")] for o in opts
                              if o.get("type") in fp.TYPE_NAME})
            picked = fp.TYPE_NAME.get(opts[ci].get("type"))
            if picked is None or len(offered) < 2:
                continue
            band = fp.band_of(r.get("agent_rating"))
            scores = sampler.type_scores(band, r.get("our_archetype"),
                                         turn, k)
            w = {t: scores.get(t, 0.0) for t in offered}
            tot = sum(w.values())
            if tot <= 0:
                continue
            p = {t: v / tot for t, v in w.items()}
            for arch in (r.get("our_archetype"), "FIELD"):
                rec = out[arch]
                rec["n"] += 1
                rec["top1"] += int(picked == max(p, key=p.get))
                rec["logloss"] += -math.log(max(p.get(picked, 0.0), 1e-9))
                rec["logloss_uniform"] += math.log(len(offered))
                rec["realized"][picked] += 1
                for t, v in p.items():
                    rec["expected"][t] += v
    table = {}
    for arch, rec in out.items():
        n = rec["n"]
        if n < 20:
            continue
        exp = {t: rec["expected"][t] / n for t in TYPES}
        rea = {t: rec["realized"][t] / n for t in TYPES}
        table[arch] = {
            "n_menus": n,
            "top1": round(rec["top1"] / n, 4),
            "logloss": round(rec["logloss"] / n, 4),
            "logloss_uniform": round(rec["logloss_uniform"] / n, 4),
            "mix_tv": round(tv(exp, rea), 4),
            "expected_mix": {t: round(v, 4) for t, v in exp.items() if v},
            "realized_mix": {t: round(v, 4) for t, v in rea.items() if v},
        }
    return table


# ---------------------------------------------------------------------------
def main():
    avail_model = "persistent" if "--persistent" in sys.argv else "iid"
    suffix = "_v2" if avail_model == "persistent" else ""
    rng = random.Random(SEED)
    sampler = TurnSampler(os.path.join(HERE, "policy_train.json"))
    print("loading holdout ...", file=sys.stderr)
    real_turns, ordmap = load_real_turns()
    real_dmg = load_real_damage()
    disr_real, agree = load_real_menus()

    by_arch = collections.defaultdict(list)
    for t in real_turns:
        by_arch[t["arch"]].append(t)
    total_turns = len(real_turns)

    results = {"holdout_day": HOLDOUT, "K": K, "seed": SEED,
               "avail_model": avail_model,
               "n_real_turns": total_turns, "archetypes": {}}
    go_volume = 0

    for arch in ARCHS + ["FIELD"]:
        rows = (real_turns if arch == "FIELD" else by_arch.get(arch, []))
        if not rows:
            continue
        cap = 20000
        if len(rows) > cap:
            rows = rng.sample(rows, cap)
        # --- sampled side, matched conditioning
        s_types, s_attack, s_attach, s_disr_turn = [], 0, 0, 0
        s_chip, s_prizes = [], []
        r_chip, r_prizes = [], []
        n_s = 0
        for row in rows:
            has_pair = row["key"] in real_dmg
            if has_pair:
                c, p = real_dmg[row["key"]]
                r_chip.append(c)
                r_prizes.append(p)
            for _ in range(K):
                st = sampler.sample_turn(row["band"], row["arch"],
                                         row["turn"], rng,
                                         avail_model=avail_model)
                n_s += 1
                s_types.append(st["actions"])
                s_attack += st["attacked"]
                s_attach += st["attached"]
                s_disr_turn += st["disrupted"]
                if has_pair:
                    s_chip.append(st["chip"])
                    s_prizes.append(st["prizes"])
        r_mix, r_nvis = mix([r["types"] for r in rows])
        s_mix, s_nvis = mix(s_types)
        d_tv = tv(r_mix, s_mix)
        r_atk = sum(r["attacked"] for r in rows) / len(rows)
        s_atk = s_attack / n_s
        r_att = sum(r["attached"] for r in rows) / len(rows)
        s_att = s_attach / n_s
        r_chip_mean = sum(r_chip) / len(r_chip) if r_chip else None
        s_chip_mean = sum(s_chip) / len(s_chip) if s_chip else None
        r_pr_mean = sum(r_prizes) / len(r_prizes) if r_prizes else None
        s_pr_mean = sum(s_prizes) / len(s_prizes) if s_prizes else None
        w1 = w1_hist(r_chip, s_chip)

        dr = disr_real.get(arch if arch != "FIELD" else "*",
                           collections.Counter())
        n_play, k_disr = dr.get("n_play", 0), dr.get("disr", 0)
        r_disr = k_disr / n_play if n_play else None
        ci = jeffreys(k_disr, n_play) if n_play else None
        if arch == "FIELD":
            s_disr = (sampler.disrupt.get("*") or {}).get("rate", 0.0)
        else:
            s_disr = sampler._disr_rate(arch)

        rec = {
            "n_real_turns": len(rows), "n_sampled_turns": n_s,
            "n_real_visits": r_nvis, "n_damage_pairs": len(r_chip),
            "other_backoff": sampler.arch_label(arch) == "OTHER",
            "action_mix_real": {t: round(v, 4) for t, v in r_mix.items()},
            "action_mix_sampled": {t: round(v, 4) for t, v in s_mix.items()},
            "tv": round(d_tv, 4),
            "chip_mean_real": None if r_chip_mean is None else round(r_chip_mean, 1),
            "chip_mean_sampled": None if s_chip_mean is None else round(s_chip_mean, 1),
            "chip_w1": None if w1 is None else round(w1, 1),
            "prizes_mean_real": None if r_pr_mean is None else round(r_pr_mean, 4),
            "prizes_mean_sampled": None if s_pr_mean is None else round(s_pr_mean, 4),
            "attack_rate_real": round(r_atk, 4),
            "attack_rate_sampled": round(s_atk, 4),
            "attach_rate_real": round(r_att, 4),
            "attach_rate_sampled": round(s_att, 4),
            "disr_per_play_real": None if r_disr is None else round(r_disr, 4),
            "disr_per_play_ci": None if ci is None else [round(x, 4) for x in ci],
            "disr_per_play_n": n_play,
            "disr_rate_sampled_head": round(s_disr, 4),
            "disr_turn_rate_sampled": round(s_disr_turn / n_s, 4),
        }
        # --- pre-registered verdict (README A-D)
        if r_chip_mean is not None:
            chip_tol = max(10.0, 0.15 * r_chip_mean)
            pass_b = (abs(s_chip_mean - r_chip_mean) <= chip_tol
                      and w1 is not None and w1 <= TOL_CHIP_W1)
            pass_c = abs(s_pr_mean - r_pr_mean) <= TOL_PRIZES
        else:
            pass_b = pass_c = False
        pass_a = d_tv <= TOL_TV
        pass_d = abs(s_atk - r_atk) <= TOL_ATTACK
        rec["criteria"] = {"A_action_mix": pass_a, "B_chip": pass_b,
                           "C_prizes": pass_c, "D_attack_rate": pass_d}
        rec["verdict"] = "GO" if (pass_a and pass_b and pass_c and pass_d) \
            else "NO-GO"
        rec["flags"] = {
            "attach_within_5pt": abs(s_att - r_att) <= TOL_ATTACH,
            "disruption_ok": (r_disr is None or ci is None
                              or ci[0] <= s_disr <= ci[1]
                              or abs(s_disr - r_disr) <= TOL_DISR),
        }
        results["archetypes"][arch] = rec
        if arch != "FIELD" and rec["verdict"] == "GO":
            go_volume += len(by_arch.get(arch, []))
        print("  %-26s n=%6d  TV=%.3f  chipΔ=%s  W1=%s  %s"
              % (arch, len(rows), d_tv,
                 "-" if r_chip_mean is None else "%+.1f" % (s_chip_mean - r_chip_mean),
                 "-" if w1 is None else "%.1f" % w1, rec["verdict"]),
              file=sys.stderr)

    covered = set(results["archetypes"]) - {"FIELD"}
    listed_volume = sum(len(by_arch.get(a, [])) for a in covered)
    results["attack_choice_agreement"] = {
        "n_multi_attack_menus": agree.get("n", 0),
        "agreement": (round(agree["match"] / agree["n"], 4)
                      if agree.get("n") else None),
        "gates": agree.get("n", 0) >= 30,
    }
    results["coverage"] = {
        "go_turn_share": round(go_volume / total_turns, 4),
        "listed_turn_share": round(listed_volume / total_turns, 4),
        "threshold": COVERAGE_GO,
    }
    results["phase0_verdict"] = ("GO" if go_volume / total_turns >= COVERAGE_GO
                                 else "NO-GO")

    if avail_model == "iid":          # decision-level test rides the v1 run
        print("menu replay ...", file=sys.stderr)
        results["menu_replay"] = menu_replay(sampler, ordmap)

    dest = os.path.join(HERE, "results%s.json" % suffix)
    with open(dest, "w") as fh:
        json.dump(results, fh, indent=1)
    print("phase0 verdict (%s availability): %s (GO archetypes cover %.1f%% "
          "of holdout turns)" % (avail_model, results["phase0_verdict"],
                                 100 * go_volume / total_turns))
    print("wrote %s" % dest)


if __name__ == "__main__":
    main()
