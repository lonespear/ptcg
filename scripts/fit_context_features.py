"""Fit and SCREEN the D46 context-feature slate for the linear pilot.

Five candidate feature bundles, each fitted by the advantage-regression
methodology of `fit_trajectory_features.py` / `fit_protection_feature.py` —
the agent's own functions imported from the shipped file where they exist,
logistic MLE from `ptcg/advantage.py`, one position per episode (earliest
turn) at rating >= 1000, turn fixed effects and went_first, coefficients
anchored at prize_diff = 1000 — and then screened the way the playbook's
entries 7-8 demand: on sign stability across days and archetypes, and on
decision relevance (does the term CHANGE what the search picks on real mined
positions), never on outcome-fit alone.

The slate (D46):
  1 status      the observation's status flags (both sides)
  2 pace        prizes-per-turn vs the archetype's pace curve (both sides)
  3 buildup     opponent bench evolution potential + their attach tempo
  4 hand        opponent hand size and its 2-turn delta
  5 horizon     own deck count / energy-in-deck vs estimated turns-to-end

Lookup tables (pace curves, per-turn series index for the lags, archetype
energy counts) come from `build_tables.py` over the three days that carry
series.parquet; pass --tables.

    python scripts/fit_context_features.py --tables <context_tables.json> \
        [--flips 400] [--out data/analysis/context_screen.json]
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
DAYS = ["2026-07-31", "2026-08-01", "2026-08-02", "2026-08-03",
        "2026-08-04", "2026-08-05", "2026-08-06"]
SERIES_DAYS = {"2026-08-04", "2026-08-05", "2026-08-06"}

# The terms the shipped evaluator scores, C2 included — a candidate earns its
# interval on top of what already runs (fit_protection_feature.py precedent).
BASE_TERMS = ["prize_diff", "hp_diff", "energy_diff", "bench_diff",
              "damage_diff", "threat_traj_k3"]

BUNDLES = {
    "status":  ["status_any_m", "status_any_t"],
    "pace":    ["pace_diff", "pace_dev_us", "pace_dev_them"],
    "buildup": ["evo_buildup_them", "opp_attach_tempo3"],
    "hand":    ["opp_hand", "opp_hand_delta2"],
    "horizon": ["deck_headroom", "energy_headroom"],
}
# Terms that need the series lag join (three days only).
LAG_TERMS = {"opp_attach_tempo3", "opp_hand_delta2"}

STATUS_FLAGS = ("asleep", "paralyzed", "confused", "poisoned", "burned")


def norm_tables(tables: dict) -> dict:
    """JSON round-trip leaves int keys as strings; fix the curve keys."""
    tables["pace_curves"] = {
        arch: {int(k): v for k, v in c.items()}
        for arch, c in tables["pace_curves"].items()}
    return tables


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


# --------------------------------------------------------------- the features
# Every function reads either observation form through agent._g, takes the
# per-position context dict, and is exactly what a live implementation would
# compute (histories the live agent would remember arrive frozen in ctx).

def _seat_turns(turn: int, first: int, seat: int) -> int:
    """How many of this seat's turns have started by game turn `turn`."""
    if turn <= 0:
        return 0
    return (turn + 1) // 2 if seat == first else turn // 2


def _curve_at(curves: dict, arch: str, st: int) -> float:
    c = curves.get(arch) or curves["(all)"]
    if not c:
        return 0.0
    ks = sorted(c)
    st = min(max(st, ks[0]), ks[-1])
    return c[st][0]


def _curve_rate(curves: dict, arch: str, st: int, span: int = 3) -> float:
    hi = _curve_at(curves, arch, st + span)
    lo = _curve_at(curves, arch, st)
    r = (hi - lo) / span
    if r < 0.12:                       # curve flat/exhausted: floor the rate
        r = 0.12
    return r


def feat_status(A, cur, mine, theirs, ctx) -> dict:
    out = {}
    for tag, p in (("m", mine), ("t", theirs)):
        flags = [1.0 if A._g(p, s, False) else 0.0 for s in STATUS_FLAGS]
        for s, v in zip(STATUS_FLAGS, flags):
            out[f"status_{s}_{tag}"] = v
        out[f"status_any_{tag}"] = 1.0 if any(flags) else 0.0
    return out


def feat_pace(A, cur, mine, theirs, ctx) -> dict:
    turn = A._g(cur, "turn", 0) or 0
    me, first = ctx["me"], ctx["first"]
    myt = max(_seat_turns(turn, first, me), 1)
    tht = max(_seat_turns(turn, first, 1 - me), 1)
    taken_m = 6 - len(A._g(mine, "prize", []) or [])
    taken_t = 6 - len(A._g(theirs, "prize", []) or [])
    curves = ctx["curves"]
    return {
        "pace_diff": taken_m / myt - taken_t / tht,
        "pace_dev_us": taken_m - _curve_at(curves, ctx["arch_us"], myt),
        "pace_dev_them": taken_t - _curve_at(curves, ctx["arch_them"], tht),
    }


def feat_evo_buildup(A, cur, mine, theirs, ctx) -> float:
    """Evolution upgrade potential assembling on the opponent's bench: for
    each benched Pokemon, the best printed-damage gain an evolution still
    available in their (posterior top-1) list would buy, summed over bench."""
    pool = ctx["pool_them"]
    if not pool:
        return 0.0
    cards, _ = A._tables()
    seen: dict = {}
    for zone in ("active", "bench", "discard"):
        for card in A._g(theirs, zone, []) or []:
            if card is None:
                continue
            cid = int(A._g(card, "id", 0) or 0)
            if cid:
                seen[cid] = seen.get(cid, 0) + 1
    total = 0.0
    emap = A._evo_map()
    for mon in A._g(theirs, "bench", []) or []:
        if mon is None:
            continue
        cid = int(A._g(mon, "id", 0) or 0)
        if not cid:
            continue
        prof = A._attack_profile(cid)
        cur_best = max((d for _, d in prof), default=0.0)
        nm = getattr(cards.get(cid), "name", None)
        best_gain = 0.0
        for evo_id, steps in (emap.get(nm or "", ()) or ()):
            if pool.get(evo_id, 0) - seen.get(evo_id, 0) <= 0:
                continue
            evo_best = max((d for _, d in A._attack_profile(evo_id)),
                           default=0.0)
            gain = evo_best - cur_best
            if gain > best_gain:
                best_gain = gain
        total += best_gain
    return total


def feat_tempo(A, cur, mine, theirs, ctx) -> float | None:
    """Opponent Energy attached over their last 3 turns (series lag join)."""
    lag = ctx.get("lag_opp_energy3")
    if lag is None:
        return None
    _, en_t, _, _ = A._side_totals(theirs)
    return en_t - lag


def feat_hand(A, cur, mine, theirs, ctx) -> dict:
    out = {"opp_hand": float(A._g(theirs, "handCount", 0) or 0)}
    lag = ctx.get("lag_opp_hand2")
    out["opp_hand_delta2"] = (out["opp_hand"] - lag) if lag is not None \
        else None
    return out


def feat_horizon(A, cur, mine, theirs, ctx) -> dict:
    """Own deck count and energy-in-deck against the estimated number of our
    turns left in the game — the leader's remaining prizes over their
    archetype pace-curve rate."""
    turn = A._g(cur, "turn", 0) or 0
    me, first, curves = ctx["me"], ctx["first"], ctx["curves"]
    myt = max(_seat_turns(turn, first, me), 1)
    tht = max(_seat_turns(turn, first, 1 - me), 1)
    pz_m = len(A._g(mine, "prize", []) or [])
    pz_t = len(A._g(theirs, "prize", []) or [])
    if pz_m <= pz_t:                   # leader = closer to their last prize
        lead_arch, lead_pz, lead_st = ctx["arch_us"], pz_m, myt
    else:
        lead_arch, lead_pz, lead_st = ctx["arch_them"], pz_t, tht
    est_turns = min(lead_pz / _curve_rate(curves, lead_arch, lead_st), 40.0)

    deck_m = float(A._g(mine, "deckCount", 0) or 0)
    energy_ids = ctx["energy_ids"]
    attached = hand_e = disc_e = 0
    for zone in ("active", "bench"):
        for mon in A._g(mine, zone, []) or []:
            if mon is None:
                continue
            ec = A._g(mon, "energyCards", None)
            attached += len(ec) if ec is not None \
                else len(A._g(mon, "energies", []) or [])
    for card in A._g(mine, "hand", []) or []:
        if card is not None and int(A._g(card, "id", 0) or 0) in energy_ids:
            hand_e += 1
    for card in A._g(mine, "discard", []) or []:
        if card is not None and int(A._g(card, "id", 0) or 0) in energy_ids:
            disc_e += 1
    deck_energy = max(ctx["arch_energy_us"] - attached - hand_e - disc_e, 0.0)
    return {
        "deck_headroom": deck_m - est_turns,
        "energy_headroom": deck_energy + hand_e - est_turns,
    }


def feat_deck_deficit(A, cur, mine, theirs, ctx) -> float:
    return min(feat_horizon(A, cur, mine, theirs, ctx)["deck_headroom"], 0.0)


def make_ctx(A, rec, obs, tables, energy_ids) -> dict:
    """Per-position constants: what the live agent would know or remember."""
    cur = obs.get("current") or {}
    me = cur.get("yourIndex")
    if me not in (0, 1):
        me = int(rec.get("seat", 0))
    first = cur.get("firstPlayer")
    if first not in (0, 1):
        first = me if rec.get("went_first", 0) == 1 else 1 - me
    arch_us = rec.get("our_archetype") or "(all)"
    arch_them = A._TRAJ_ARCH.get("them") or "(all)"
    try:
        post = A._deck_posterior(obs, top_k=1)
        pool_them = dict(post[0][0]) if post else {}
    except Exception:
        pool_them = {}
    ctx = {
        "me": me, "first": first,
        "arch_us": arch_us, "arch_them": arch_them,
        "curves": tables["pace_curves"],
        "arch_energy_us": tables["arch_energy"].get(arch_us, 10.0),
        "energy_ids": energy_ids,
        "pool_them": pool_them,
        "lag_opp_hand2": None, "lag_opp_energy3": None,
    }
    # Series lag joins, where the day carries series.parquet.
    sidx = tables.get("series_index") or {}
    key = f"{int(rec['episode_id'])}:{1 - me}"
    per_turn = sidx.get(key)
    if per_turn:
        turn = cur.get("turn", 0) or 0
        tht = _seat_turns(turn, first, 1 - me)
        row2 = per_turn.get(str(tht - 2))
        row3 = per_turn.get(str(tht - 3))
        if row2 is not None:
            ctx["lag_opp_hand2"] = float(row2[0])
        if row3 is not None:
            ctx["lag_opp_energy3"] = float(row3[1])
    return ctx


# ------------------------------------------------------------------ the rows

def build_rows(A, tables, energy_ids) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    timing = {"n": 0, "seconds": 0.0}
    for day in DAYS:
        path = MINED / day / "positions.jsonl.gz"
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

                A._refresh_traj_arch(obs)
                ctx = make_ctx(A, rec, obs, tables, energy_ids)

                t0 = time.perf_counter()
                try:
                    thr, _, _ = A._threat_and_exposure(cur, mine, theirs)
                except Exception:
                    thr = 0.0
                feats: dict = {}
                feats.update(feat_status(A, cur, mine, theirs, ctx))
                feats.update(feat_pace(A, cur, mine, theirs, ctx))
                feats["evo_buildup_them"] = feat_evo_buildup(
                    A, cur, mine, theirs, ctx)
                feats["opp_attach_tempo3"] = feat_tempo(
                    A, cur, mine, theirs, ctx)
                feats.update(feat_hand(A, cur, mine, theirs, ctx))
                feats.update(feat_horizon(A, cur, mine, theirs, ctx))
                timing["seconds"] += time.perf_counter() - t0
                timing["n"] += 1

                hp_m, en_m, bench_m, dmg_m = A._side_totals(mine)
                hp_t, en_t, bench_t, dmg_t = A._side_totals(theirs)
                rows.append({
                    "episode_id": int(rec["episode_id"]),
                    "date": day,
                    "turn": int(rec.get("turn", -1)),
                    "rating": float(rec.get("agent_rating") or 0.0),
                    "went_first": int(rec.get("went_first", -1)),
                    "won": int(rec["won"]),
                    "our_archetype": rec.get("our_archetype"),
                    "opp_archetype": rec.get("opp_archetype"),
                    "prize_diff": len(theirs.get("prize") or [])
                                  - len(mine.get("prize") or []),
                    "hp_diff": hp_m - hp_t,
                    "energy_diff": en_m - en_t,
                    "bench_diff": bench_m - bench_t,
                    "damage_diff": dmg_t - dmg_m,
                    "no_active_me": 0.0 if (mine.get("active") or []) else 1.0,
                    "threat_traj_k3": float(thr),
                    **feats,
                })
    return rows, timing


def dedup(rows: list[dict]) -> list[dict]:
    keep: dict[int, dict] = {}
    for r in rows:
        cur = keep.get(r["episode_id"])
        if cur is None or r["turn"] < cur["turn"]:
            keep[r["episode_id"]] = r
    return sorted(keep.values(), key=lambda r: r["episode_id"])


# ------------------------------------------------------------------- the fit

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
    for tv in sorted({r["turn"] for r in sel})[1:]:
        cols.append(np.array([1.0 if r["turn"] == tv else 0.0 for r in sel]))
        names.append(f"turn={tv}")
    if arch_fe:
        for a in sorted({(r["our_archetype"] or "?") for r in sel})[1:]:
            cols.append(np.array(
                [1.0 if (r["our_archetype"] or "?") == a else 0.0
                 for r in sel]))
            names.append(f"arch={a}")
    X = np.column_stack(cols)
    y = np.array([r["won"] for r in sel], dtype=float)
    f = logit_fit(X, y)
    tab = coef_table(f, names, X)
    anchor = next(r for r in tab if r["term"] == "prize_diff")["beta"]
    scaled = {}
    for r in tab:
        if r["term"] in terms and anchor:
            lo = 1000.0 * r["ci_lo"] / anchor
            hi = 1000.0 * r["ci_hi"] / anchor
            scaled[r["term"]] = {
                "weight": round(1000.0 * r["beta"] / anchor, 2),
                "ci_lo": round(min(lo, hi), 2),
                "ci_hi": round(max(lo, hi), 2),
                "excludes_zero": bool(r["sig"]),
            }
    return {"n": len(sel), "loglik": round(f["loglik"], 3),
            "mcfadden_r2": round(f["mcfadden_r2"], 4),
            "weights_prize_1000": scaled}


def rows_for(pool: list[dict], terms: list[str]) -> list[dict]:
    """One position per episode — the earliest at which every term in the
    bundle is defined (the lag joins only exist from the fourth own-turn on,
    so 'earliest overall' would throw those episodes away)."""
    ok = [r for r in pool if all(r.get(t) is not None for t in terms)]
    return dedup(ok)


def screen_bundle(pool: list[dict], name: str, terms: list[str]) -> dict:
    sub = rows_for(pool, terms)
    if len(sub) < 200:
        return {"bundle": name, "n": len(sub), "verdict": "DATA-NULL",
                "note": "fewer than 200 fittable episodes"}
    base = fit(sub, BASE_TERMS)
    solo = fit(sub, BASE_TERMS + terms)
    lr = 2.0 * (solo["loglik"] - base["loglik"])
    out = {"bundle": name, "n": len(sub),
           "lr_chi2": round(lr, 2), "df": len(terms),
           "pooled": solo["weights_prize_1000"]}

    # Sign stability across days.
    day_signs: dict = {t: [] for t in terms}
    for day in DAYS:
        dsub = [r for r in sub if r["date"] == day]
        if len(dsub) < 150:
            continue
        try:
            fd = fit(dsub, BASE_TERMS + terms)
        except Exception:
            continue
        for t in terms:
            w = fd["weights_prize_1000"].get(t)
            if w:
                day_signs[t].append((day, w["weight"]))
    out["by_day"] = {t: {d: w for d, w in v} for t, v in day_signs.items()}

    # Archetype fixed effects, and the cell that is our own deck.
    try:
        out["arch_fe"] = fit(sub, BASE_TERMS + terms,
                             arch_fe=True)["weights_prize_1000"]
    except Exception:
        out["arch_fe"] = {}
    oger = [r for r in sub
            if (r["our_archetype"] or "") == "Teal Mask Ogerpon ex"]
    if len(oger) >= 200:
        try:
            out["ogerpon_cell"] = dict(
                fit(oger, BASE_TERMS + terms)["weights_prize_1000"],
                n=len(oger))
        except Exception:
            out["ogerpon_cell"] = {}

    # The opponent-archetype cut, for opponent-facing features.
    counts: dict = {}
    for r in sub:
        a = r.get("opp_archetype") or "?"
        counts[a] = counts.get(a, 0) + 1
    out["by_opp_archetype"] = {}
    for a, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:4]:
        cell = [r for r in sub if (r.get("opp_archetype") or "?") == a]
        if len(cell) < 300:
            continue
        try:
            fc = fit(cell, BASE_TERMS + terms)["weights_prize_1000"]
            out["by_opp_archetype"][a] = {
                t: dict(fc[t], n=len(cell)) for t in terms if t in fc}
        except Exception:
            pass
    return out


# ---------------------------------------------------------- the flip screen

def flip_screen(A, positions: list[dict], tables, energy_ids,
                arms: dict[str, list[tuple]]) -> dict:
    """Does the fitted term CHANGE what the search picks on real positions?

    One search pass per position — the shipped plan, determinizations and
    rollouts — with `_evaluate` replaced by a recorder that logs, for every
    leaf the rollout reaches, the baseline margin AND each arm's feature
    delta at that same leaf. Baseline and arm picks are then argmaxes over
    identical worlds, the same construction the tree-leaf agreement table
    used, so the engine's internal randomness cancels exactly and a
    weight-zero arm flips nothing by construction. The naive
    replay-the-search-twice design reads a ~35% flip rate on a NULL arm
    (engine deals fresh hidden worlds each call); numbers from that design
    are noise and are not reported."""
    orig_eval = A._evaluate
    results = {name: {"n": 0, "flips": 0, "leaf_nonzero": 0,
                      "mean_abs_delta": 0.0}
               for name in arms}
    base_stats = {"n": 0, "search_overrides": 0}
    t0 = time.time()
    bucket: list = []

    def recorder(observation, me):
        cur = observation.current
        result = getattr(cur, "result", -1)
        if result is not None and result != -1:
            v = 1e6 if result == me else -1e6
            bucket.append((v, {name: 0.0 for name in arms}))
            return v
        base = A._margin(cur, me)
        deltas = {}
        players = A._g(cur, "players", []) or []
        if len(players) == 2:
            mine, theirs = players[me], players[1 - me]
            for name, featlist in arms.items():
                d = 0.0
                for fn, key, w in featlist:
                    if not w:
                        continue
                    try:
                        v = fn(A, cur, mine, theirs, recorder.ctx)
                        if isinstance(v, dict):
                            v = v.get(key)
                        if v is not None:
                            d += w * float(v)
                    except Exception:
                        pass
                deltas[name] = d
        bucket.append((base, deltas))
        return base

    for rec in positions:
        obs = rec["observation"]
        options = (obs.get("select") or {}).get("option") or []
        A._refresh_traj_arch(obs)
        recorder.ctx = make_ctx(A, rec, obs, tables, energy_ids)
        try:
            o = A.to_observation_class(obs)
            me = o.current.yourIndex
            rules_i = A._choose_main(options, obs)
            seen, n_deck, n_hand, n_prize = A._opponent_counts(obs)
            posterior = A._deck_posterior(obs, top_k=A.POSTERIOR_TOP_K)
            if posterior and posterior[0][1] >= A.CONFIDENCE_GATE:
                posterior = posterior[:1]
            if not posterior:
                continue
            cand, rep = A._dedup_options(obs, options)
            rules_i = rep.get(rules_i, rules_i)
            per_deck = max(1, A.SEARCH_N_DET // max(len(posterior), 1))
            plan = [(counts, weight / per_deck)
                    for counts, weight in posterior
                    for _ in range(per_deck)]
        except Exception:
            continue

        rng = A._position_rng(obs)
        A._SEARCH_ME = me
        A._evaluate = recorder
        acc = {name: {i: 0.0 for i in cand} for name in arms}
        acc_base = {i: 0.0 for i in cand}
        mass = {i: 0.0 for i in cand}
        n_eval = {i: 0 for i in cand}
        stats = {name: [0, 0.0, 0] for name in arms}  # nonzero, |d| sum, k
        try:
            for counts, weight in plan:
                try:
                    my_deck, my_prize = A._own_hidden(obs, rng)
                    od, oh, op = A._hidden_from(counts, seen, n_deck,
                                                n_hand, n_prize, rng)
                    root = A.search_begin(o, my_deck, my_prize, od, op,
                                          oh, [])
                except Exception:
                    continue
                try:
                    for i in cand:
                        from cg.api import search_step as _ss
                        bucket.clear()
                        try:
                            child = _ss(root.searchId, [i])
                            A._rollout_value(child, me,
                                             A._rules_choice_for, None)
                        except Exception:
                            continue
                        if not bucket:
                            continue
                        base_v, deltas = bucket[-1]
                        acc_base[i] += weight * base_v
                        for name in arms:
                            d = deltas.get(name, 0.0)
                            acc[name][i] += weight * (base_v + d)
                            st = stats[name]
                            if d:
                                st[0] += 1
                            st[1] += abs(d)
                            st[2] += 1
                        mass[i] += weight
                        n_eval[i] += 1
                finally:
                    try:
                        A.search_end()
                    except Exception:
                        pass
        finally:
            A._SEARCH_ME = None
            A._evaluate = orig_eval

        n_top = n_eval.get(rules_i, 0)
        if not n_top:
            continue
        evaluated = [i for i in cand
                     if n_eval[i] == n_top and mass[i] > 0.0]
        if rules_i not in evaluated:
            continue

        def pick(avg):
            best = max(evaluated, key=lambda i: avg[i])
            return rules_i if avg[best] < avg[rules_i] \
                + A.WEIGHTS["search_margin"] or best == rules_i else best

        avg_b = {i: acc_base[i] / mass[i] for i in evaluated}
        base_pick = pick(avg_b)
        base_stats["n"] += 1
        if base_pick != rules_i:
            base_stats["search_overrides"] += 1
        for name in arms:
            avg_a = {i: acc[name][i] / mass[i] for i in evaluated}
            r = results[name]
            r["n"] += 1
            if pick(avg_a) != base_pick:
                r["flips"] += 1
            nz, sabs, k = stats[name]
            if nz:
                r["leaf_nonzero"] += 1
            if k:
                r["mean_abs_delta"] += sabs / k

    for name, r in results.items():
        r["flip_rate"] = round(r["flips"] / r["n"], 4) if r["n"] else None
        r["leaf_nonzero_rate"] = round(r["leaf_nonzero"] / r["n"], 4) \
            if r["n"] else None
        r["mean_abs_delta"] = round(r["mean_abs_delta"] / r["n"], 1) \
            if r["n"] else None
    results["_base"] = dict(base_stats)
    results["_seconds"] = round(time.time() - t0, 1)
    return results


# ----------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", required=True)
    ap.add_argument("--out", default=str(ROOT / "data" / "analysis"
                                         / "context_screen.json"))
    ap.add_argument("--flips", type=int, default=0)
    ap.add_argument("--rows-cache", default=None,
                    help="write/reuse the computed feature rows (JSON)")
    args = ap.parse_args()

    tables = norm_tables(json.loads(Path(args.tables).read_text()))
    cards = json.load(open(ROOT / "data" / "engine_dump" / "cards.json"))
    energy_ids = {c["cardId"] for c in cards if c.get("cardType") == 5}

    A = load_agent()
    if not A._curves() or not A._accel_rates():
        sys.exit("agent KBs missing; the fit would be meaningless")

    cache = Path(args.rows_cache) if args.rows_cache else None
    if cache and cache.exists():
        rows = json.loads(cache.read_text())
        print(f"{len(rows)} rows from cache")
    else:
        rows, timing = build_rows(A, tables, energy_ids)
        print(f"{len(rows)} positions read; feature cost "
              f"{1000 * timing['seconds'] / max(timing['n'], 1):.3f} ms each")
        if cache:
            cache.write_text(json.dumps(rows))

    pool = [r for r in rows
            if r["rating"] >= RATING_CUT and r["turn"] > 0
            and r["went_first"] >= 0]
    sel = dedup(pool)
    print(f"{len(sel)} episodes after the rating cut and one-per-episode "
          f"({len(pool)} rated rows pooled for the per-bundle dedup)")

    # Status prevalence: reported whether or not the fit is possible.
    prevalence = {}
    for tag in ("m", "t"):
        for s in STATUS_FLAGS + ("any",):
            k = f"status_{s}_{tag}"
            prevalence[k] = int(sum(r.get(k) or 0 for r in rows))
    print("status prevalence over all rows:",
          {k: v for k, v in prevalence.items() if v})

    screens = {}
    for name, terms in BUNDLES.items():
        if name == "status":
            fitable = [t for t in terms
                       if sum(r.get(t) or 0 for r in sel) >= 50]
            if not fitable:
                screens[name] = {
                    "bundle": name, "verdict": "DATA-NULL",
                    "prevalence": prevalence,
                    "note": "no status flag reaches 50 occurrences in "
                            f"{len(sel)} fitted episodes"}
                continue
            terms = fitable
        screens[name] = screen_bundle(pool, name, terms)
        s = screens[name]
        print(f"\n== {name}  n={s.get('n')}  LR chi2={s.get('lr_chi2')}")
        for t, w in (s.get("pooled") or {}).items():
            days = s.get("by_day", {}).get(t, {})
            signs = "".join("+" if v > 0 else "-" for v in days.values())
            print(f"  {t:20s} {w['weight']:+9.2f} "
                  f"[{w['ci_lo']:+8.2f},{w['ci_hi']:+8.2f}]"
                  f"{'' if w['excludes_zero'] else '  NULL'}  days:{signs}")

    # Derived screening variants: deficit-only hinges for the horizon terms
    # (the resource-death story is about running OUT, and a linear term over
    # the whole range can mute a tail effect), and the big-and-growing
    # interaction the D46 hand entry names.
    for r in pool:
        dh, eh = r.get("deck_headroom"), r.get("energy_headroom")
        r["deck_deficit"] = min(dh, 0.0) if dh is not None else None
        r["energy_deficit"] = min(eh, 0.0) if eh is not None else None
        oh, d2 = r.get("opp_hand"), r.get("opp_hand_delta2")
        r["opp_hand_x_delta2"] = (oh * d2) if (oh is not None
                                              and d2 is not None) else None

    solo_specs = {
        "pace_dev_us_solo": ["pace_dev_us"],
        "pace_dev_them_solo": ["pace_dev_them"],
        "evo_buildup_solo": ["evo_buildup_them"],
        "tempo3_solo": ["opp_attach_tempo3"],
        "opp_hand_solo": ["opp_hand"],
        "hand_grow": ["opp_hand", "opp_hand_delta2", "opp_hand_x_delta2"],
        "deck_headroom_solo": ["deck_headroom"],
        "energy_headroom_solo": ["energy_headroom"],
        "deck_deficit_solo": ["deck_deficit"],
        "energy_deficit_solo": ["energy_deficit"],
    }
    for name, terms in solo_specs.items():
        screens[name] = screen_bundle(pool, name, terms)
        s = screens[name]
        print(f"\n== {name}  n={s.get('n')}  LR chi2={s.get('lr_chi2')}")
        for t in terms:
            w = (s.get("pooled") or {}).get(t)
            if not w:
                continue
            days = s.get("by_day", {}).get(t, {})
            signs = "".join("+" if v > 0 else "-" for v in days.values())
            fe = (s.get("arch_fe") or {}).get(t, {})
            print(f"  {t:20s} {w['weight']:+9.2f} "
                  f"[{w['ci_lo']:+8.2f},{w['ci_hi']:+8.2f}]"
                  f"{'' if w['excludes_zero'] else '  NULL'}  days:{signs}"
                  f"  archFE:{fe.get('weight')}")

    # Correlations with the shipped terms, over the full-sample episodes.
    cand_terms = ["pace_diff", "pace_dev_us", "pace_dev_them",
                  "evo_buildup_them", "opp_hand", "deck_headroom",
                  "energy_headroom"]
    M = {t: np.array([r[t] for r in sel], dtype=float)
         for t in BASE_TERMS + cand_terms}
    corr = {}
    for t in cand_terms:
        corr[t] = {b: round(float(np.corrcoef(M[t], M[b])[0, 1]), 3)
                   for b in BASE_TERMS}
    print("\ncorrelation with shipped terms:")
    for t, cs in corr.items():
        print(f"  {t:18s} " + "  ".join(f"{b}={v:+.2f}"
                                        for b, v in cs.items()))

    flips = None
    if args.flips:
        # Menu positions on our own archetype's seats (the deck the pilot
        # plays; _own_hidden reconstructs that list exactly).
        cand = []
        for day in DAYS:
            with gzip.open(MINED / day / "positions.jsonl.gz", "rt",
                           encoding="utf-8") as fh:
                for line in fh:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if (rec.get("our_archetype") or "") \
                            != "Teal Mask Ogerpon ex":
                        continue
                    obs = rec.get("observation") or {}
                    opts = (obs.get("select") or {}).get("option") or []
                    if len(opts) < 2 or not obs.get("search_begin_input"):
                        continue
                    if not any(o.get("type") in A.MAIN_PRIORITY
                               for o in opts):
                        continue
                    rec["date"] = day
                    cand.append(rec)
        step = max(len(cand) // args.flips, 1)
        sample = cand[::step][:args.flips]
        print(f"\nflip screen: {len(sample)} of {len(cand)} Ogerpon-seat "
              f"menu positions")

        def w_of(bundle, term):
            w = (screens.get(bundle, {}).get("pooled") or {}).get(term, {})
            # D30: a term whose interval covers zero carries no weight.
            return w.get("weight", 0.0) if w.get("excludes_zero") else 0.0

        # The arms are the D30-shaped shipping candidates: only terms whose
        # pooled interval excluded zero carry a weight, at the fitted value.
        arms = {
            "pace": [(feat_pace, "pace_diff", w_of("pace", "pace_diff")),
                     (feat_pace, "pace_dev_us", w_of("pace", "pace_dev_us")),
                     (feat_pace, "pace_dev_them",
                      w_of("pace", "pace_dev_them"))],
            "buildup": [(feat_evo_buildup, None,
                         w_of("buildup", "evo_buildup_them")),
                        (feat_tempo, None,
                         w_of("buildup", "opp_attach_tempo3"))],
            "opp_hand": [(feat_hand, "opp_hand",
                          w_of("opp_hand_solo", "opp_hand"))],
            "deck_deficit": [(feat_deck_deficit, None,
                              w_of("deck_deficit_solo", "deck_deficit"))],
        }
        arms = {k: v for k, v in arms.items()
                if any(w for _, _, w in v)} or {}
        # Weight-zero null arm: its flip rate is the replay noise floor, and
        # with the frozen clock it must read 0.
        arms["null"] = [(feat_hand, "opp_hand", 0.0)]
        for k, v in arms.items():
            print(f"  arm {k}: " + ", ".join(
                f"{key or fn.__name__}={w:+.1f}" for fn, key, w in v if w))
        flips = flip_screen(A, sample, tables, energy_ids, arms)
        print(json.dumps(flips, indent=1))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "spec": {
            "sample": "data/mined/*/positions.jsonl.gz, one position per "
                      "episode (earliest), rating >= 1000, turn FE, "
                      "went_first; base terms include the shipped C2",
            "base_terms": BASE_TERMS, "bundles": BUNDLES,
            "lag_terms_days": sorted(SERIES_DAYS),
            "anchor": "prize_diff = 1000",
        },
        "n_rows": len(rows), "n_episodes": len(sel),
        "status_prevalence": prevalence,
        "screens": screens,
        "correlations_with_shipped": corr,
        "flips": flips,
    }, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
