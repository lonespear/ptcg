"""D47 neural-leaf NAMED feature extractor.

One function, two callers, zero drift: the offline table builder
(`ptcg.leaf.build_table`) and the runtime evaluator (the scratch main.py
variant) both call `leaf_features(M, obs, me, lag_ctx)` with M = the loaded
agent module. Every feature is a named quantity the agent already computes
(the shipped linear/tree terms) or a named engineered term the D47 contract
lists: multi-turn lags, energy-attach acceleration, evolution-line state
including the Rare-Candy skip, scaling-attack potential, turn + prize band.

No raw state enters the model; the vector below is the complete input.

`lag_ctx` is a dict built per DECISION (not per leaf): the last three of our
own-turn board snapshots, oldest last:
    {"depth": int 0..3,
     "prizes_me": [l1, l2, l3], "prizes_them": [...],
     "energy_me": [...], "energy_them": [...],
     "hand_me":   [...], "hand_them":   [...],
     "bench_me":  [...], "bench_them":  [...]}
Missing lags are filled with the CURRENT leaf value (no-change assumption)
and `lag_depth` tells the net how many were real. The same convention runs
offline (built from the replay's turn stream) and live (built from the
observation stream the agent sees), so train and serve read the same thing.
"""

from __future__ import annotations

# The vector, in order. Index = position in the model input.
FEATURE_NAMES = (
    # core board state, both sides and diffs (the shipped linear terms,
    # levels included so the net can hold non-difference interactions)
    "prize_diff", "prizes_left_me", "prizes_left_them",
    "hp_me", "hp_them", "hp_diff",
    "energy_me", "energy_them", "energy_diff",
    "bench_me", "bench_them", "bench_diff",
    "damage_me", "damage_them", "damage_diff",
    "hand_me", "hand_them", "hand_diff",
    "deck_me", "deck_them",
    "no_active_me", "no_active_them",
    # game phase
    "turn", "my_turn_ordinal", "min_prizes", "race_phase",
    # trajectory / threat (the C2 machinery, both endpoints not just diffs)
    "threat_me_now", "threat_them_now", "threat_now_diff",
    "threat_me_k", "threat_them_k", "threat_traj",
    "online_me", "online_them", "online_lead",
    "energy_gain1_me", "energy_gain1_them",
    "energy_gaink_me", "energy_gaink_them", "energy_traj",
    # energy-attach acceleration (KB rate + realized lag deltas below)
    "accel_rate_me", "accel_rate_them",
    # exposure (protection feature, both directions)
    "attackers_exposed", "attackers_exposed_them",
    # evolution-line state, ours (private information included)
    "evo_slots_me", "evo_steps_min_me", "evo_avail_me",
    "evo_hand_me", "evo_stage2_hand_me", "evo_outs_me",
    "candy_hand_me", "candy_skip_me",
    # evolution-line state, theirs (visible only)
    "evo_slots_them", "evo_steps_min_them", "evo_avail_them",
    # 3-turn lags: prizes/energy/hand/bench, both sides
    "prizes_me_lag1", "prizes_me_lag2", "prizes_me_lag3",
    "prizes_them_lag1", "prizes_them_lag2", "prizes_them_lag3",
    "energy_me_lag1", "energy_me_lag2", "energy_me_lag3",
    "energy_them_lag1", "energy_them_lag2", "energy_them_lag3",
    "hand_me_lag1", "hand_me_lag2", "hand_me_lag3",
    "hand_them_lag1", "hand_them_lag2", "hand_them_lag3",
    "bench_me_lag1", "bench_me_lag2", "bench_me_lag3",
    "bench_them_lag1", "bench_them_lag2", "bench_them_lag3",
    "lag_depth",
    # realized attach acceleration off the lags (rate = 1-turn delta,
    # acceleration = delta of deltas)
    "energy_delta_me", "energy_delta_them",
    "energy_accel_me", "energy_accel_them",
)

N_FEATURES = len(FEATURE_NAMES)

_CANDY_ID_CACHE: list = []          # [id] once resolved; [-1] = not found


def _candy_id(M) -> int:
    if _CANDY_ID_CACHE:
        return _CANDY_ID_CACHE[0]
    cid = -1
    try:
        cards, _ = M._tables()
        for k, c in cards.items():
            if getattr(c, "name", "") == "Rare Candy":
                cid = int(k)
                break
    except Exception:
        pass
    _CANDY_ID_CACHE.append(cid)
    return cid


def _evo_side(M, player, side: str, k: int = 2):
    """(slots_with_edges, min_steps, best_avail, hand_evos, stage2_hand,
    outs, candy_skip_possible_final_ids) for one side."""
    g = M._g
    try:
        ctx = M._evo_position_ctx(player, side)
    except Exception:
        ctx = None
    if ctx is None:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False
    pool, seen, hand_ids, hand_names, unseen, extra, name_outs = ctx
    slots = 0
    min_steps = 0.0
    best_avail = 0.0
    outs_total = 0.0
    stage2_hand = 0.0
    candy_target = False
    steps_seen: list = []
    for zone in ("active", "bench"):
        for mon in g(player, zone, []) or []:
            if mon is None:
                continue
            cid = int(g(mon, "id", 0) or 0)
            if not cid:
                continue
            try:
                edges = M._evo_edges_for(side, cid)
            except Exception:
                edges = ()
            if not edges:
                continue
            slots += 1
            for evo_id, steps, mid in edges:
                steps_seen.append(steps)
                outs_total += max(pool.get(evo_id, 0) - seen.get(evo_id, 0), 0)
                if steps == 2 and hand_ids.get(evo_id, 0) > 0:
                    stage2_hand += 1
                    candy_target = True
                try:
                    a = M._evo_avail(ctx, evo_id, steps, k, mid)
                except Exception:
                    a = 0.0
                if a > best_avail:
                    best_avail = a
    if steps_seen:
        min_steps = float(min(steps_seen))
    hand_evos = 0.0
    if side == "us":
        # evolution cards we are holding that evolve something in play
        in_play_names = set()
        cards, _ = M._tables()
        for zone in ("active", "bench"):
            for mon in g(player, zone, []) or []:
                if mon is None:
                    continue
                nm = getattr(cards.get(int(g(mon, "id", 0) or 0)), "name", None)
                if nm:
                    in_play_names.add(nm)
        for hid, n in hand_ids.items():
            frm = getattr(cards.get(int(hid)), "evolvesFrom", "") or ""
            if frm and frm in in_play_names:
                hand_evos += n
    return (float(slots), min_steps, float(best_avail), hand_evos,
            stage2_hand, float(outs_total), candy_target)


def leaf_features(M, obs, me: int, lag_ctx: dict | None = None) -> list[float]:
    """The full named vector over one (possibly simulated) position."""
    g = M._g
    cur = g(obs, "current")
    players = g(cur, "players", []) or []
    mine, theirs = players[me], players[1 - me]

    hp_m, en_m, bench_m, dmg_m = M._side_totals(mine)
    hp_t, en_t, bench_t, dmg_t = M._side_totals(theirs)
    pz_m = float(len(g(mine, "prize", []) or []))
    pz_t = float(len(g(theirs, "prize", []) or []))
    hand_m = float(M._hand_count(mine))
    hand_t = float(M._hand_count(theirs))
    deck_m = float(g(mine, "deckCount", 0) or 0)
    deck_t = float(g(theirs, "deckCount", 0) or 0)
    act_m = g(mine, "active", []) or []
    act_t = g(theirs, "active", []) or []
    no_act_m = 0.0 if (act_m and act_m[0]) else 1.0
    no_act_t = 0.0 if (act_t and act_t[0]) else 1.0
    turn = float(g(cur, "turn", 1) or 0)

    K = M.TRAJ_K
    thr_m0 = thr_t0 = thr_mk = thr_tk = 0.0
    on_m = on_t = 0.0
    g1_m = g1_t = gk_m = gk_t = 0.0
    e_m2 = e_t2 = 0.0
    exposed = exposed_t = 0.0
    try:
        (sm, gm, e_m2), (st, gt, e_t2) = M._traj_projection(cur, mine, theirs)
        thr_m0 = M._threat_at(sm, gm, 0)
        thr_t0 = M._threat_at(st, gt, 0)
        thr_mk = M._threat_at(sm, gm, K)
        thr_tk = M._threat_at(st, gt, K)
        on_m = float(M._online_turn(sm, gm))
        on_t = float(M._online_turn(st, gt))
        g1_m, g1_t = float(gm(1)), float(gt(1))
        gk_m, gk_t = float(gm(K)), float(gt(K))
        M._posture_specs()
        reach_us = M._GUST_REACH.get(M._TRAJ_ARCH["them"],
                                     M._GUST_REACH_DEFAULT)
        reach_them = M._GUST_REACH.get(M._TRAJ_ARCH["us"],
                                       M._GUST_REACH_DEFAULT)
        exposed = float(M._exposure(mine, M._threat_at(st, gt, 1),
                                    reach_us)[0])
        exposed_t = float(M._exposure(theirs, M._threat_at(sm, gm, 1),
                                      reach_them)[0])
    except Exception:
        pass

    accel_m = accel_t = 0.0
    try:
        accel_m = float(M._visible_accel(mine))
        accel_t = float(M._visible_accel(theirs))
    except Exception:
        pass

    (evs_m, evmin_m, evav_m, evhand_m, evs2h_m, evouts_m,
     candy_target) = _evo_side(M, mine, "us")
    evs_t, evmin_t, evav_t, _, _, _, _ = _evo_side(M, theirs, "them")
    candy_hand = 0.0
    cid = _candy_id(M)
    if cid > 0:
        for card in g(mine, "hand", []) or []:
            if card is not None and int(g(card, "id", 0) or 0) == cid:
                candy_hand += 1
    candy_skip = 1.0 if (candy_hand > 0 and candy_target) else 0.0

    lc = lag_ctx or {}
    depth = float(lc.get("depth", 0) or 0)

    def lags(key: str, cur_val: float) -> list[float]:
        arr = lc.get(key) or []
        out = []
        for i in range(3):
            v = arr[i] if i < len(arr) and arr[i] is not None else None
            out.append(float(v) if v is not None else float(cur_val))
        return out

    pzl_m = lags("prizes_me", pz_m)
    pzl_t = lags("prizes_them", pz_t)
    enl_m = lags("energy_me", en_m)
    enl_t = lags("energy_them", en_t)
    hdl_m = lags("hand_me", hand_m)
    hdl_t = lags("hand_them", hand_t)
    bnl_m = lags("bench_me", bench_m)
    bnl_t = lags("bench_them", bench_t)

    e_delta_m = en_m - enl_m[0]
    e_delta_t = en_t - enl_t[0]
    e_accel_m = (en_m - enl_m[0]) - (enl_m[0] - enl_m[1])
    e_accel_t = (en_t - enl_t[0]) - (enl_t[0] - enl_t[1])

    return [
        pz_t - pz_m, pz_m, pz_t,
        hp_m, hp_t, hp_m - hp_t,
        en_m, en_t, en_m - en_t,
        bench_m, bench_t, bench_m - bench_t,
        dmg_m, dmg_t, dmg_t - dmg_m,
        hand_m, hand_t, hand_m - hand_t,
        deck_m, deck_t,
        no_act_m, no_act_t,
        turn, float((turn + 1) // 2), float(min(pz_m, pz_t)),
        1.0 if min(pz_m, pz_t) <= 4 else 0.0,
        thr_m0, thr_t0, thr_m0 - thr_t0,
        thr_mk, thr_tk, thr_mk - thr_tk,
        on_m, on_t, on_t - on_m,
        g1_m, g1_t,
        gk_m, gk_t, (e_m2 + gk_m) - (e_t2 + gk_t),
        accel_m, accel_t,
        exposed, exposed_t,
        evs_m, evmin_m, evav_m,
        evhand_m, evs2h_m, evouts_m,
        candy_hand, candy_skip,
        evs_t, evmin_t, evav_t,
        pzl_m[0], pzl_m[1], pzl_m[2],
        pzl_t[0], pzl_t[1], pzl_t[2],
        enl_m[0], enl_m[1], enl_m[2],
        enl_t[0], enl_t[1], enl_t[2],
        hdl_m[0], hdl_m[1], hdl_m[2],
        hdl_t[0], hdl_t[1], hdl_t[2],
        bnl_m[0], bnl_m[1], bnl_m[2],
        bnl_t[0], bnl_t[1], bnl_t[2],
        depth,
        e_delta_m, e_delta_t,
        e_accel_m, e_accel_t,
    ]
