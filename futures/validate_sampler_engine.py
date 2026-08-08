"""Phase 0.5: sampler realism with ENGINE legality (futures/README.md, D45).

Phase 0's verdict of record was NO-GO, with the failure decomposed onto the
offline availability surrogate rather than the reply tables (RESULTS.md). This
is the pre-registered follow-up: wire ``sampler.sample_action()`` into the
engine's real legal-menu loop — the exact integration Phase 1 needs — and
re-apply the same Phase 0 tolerances (README A-D, unchanged) with true
availability.

Design
------
Every distinct holdout seat-turn that appears in the positions sample
(``data/mined/2026-08-06/positions.jsonl.gz``, full observations with
``search_begin_input``) becomes a playout root, entered at the EARLIEST
sampled main-menu visit of that turn (170 of 1,101 turns enter at ordinal 0;
the rest enter mid-turn and the real prefix is shared by both sides — the
prefix cannot contain an attack, so the attack-rate criterion is undiluted,
and the action-mix comparison uses suffix visits only on both sides).

For each root, K playouts run inside the engine via ``search_begin`` /
``search_step``:

- the played seat's own hand is real (it is that seat's own observation);
  its deck and prizes are dealt from the replay-derived deck priors
  (``agent/deck_priors.json``) through the agent's own hypergeometric
  posterior, conditioned on everything that seat has revealed INCLUDING its
  hand; the other seat's deck/hand/prizes come from ``_deck_posterior`` —
  the exact machinery the live agent searches with. Decklists are sampled
  per-playout proportional to posterior weight. No dummy fills anywhere
  (data/analysis/PRIOR_ART.md: three teams measured naive determinization
  as harmful).
- at each of the played seat's MAIN menus the sampler picks the action TYPE
  over the types the engine actually offers (``sample_action``); the concrete
  option within the type follows the Phase 1 integration rule the agent
  ships (``_opp_order``): cheapest-KO-else-biggest-hit for attack, Active
  for attach, first option otherwise. Every other prompt (card picks,
  energy, yes/no, either seat) is answered by the agent's rule policy
  ``_policy`` — the same policy every search rollout uses.
- the playout ends at the defending seat's first MAIN menu (the same moment
  the real chip/prize observables are measured at: top of the defender's
  turn), or when the game is decided.

Chip damage and prizes are measured EXACTLY, in-engine: defender board
counters at the end minus the series.parquet top-of-turn baseline (verified
to match the observation board on all 170 ordinal-0 roots), and the seat's
prize-count delta. The real side is the same construction Phase 0 used
(validate_sampler.load_real_damage), on the SAME turns — a paired design.

Verdict: identical tolerances, identical coverage rule (GO iff archetypes
passing A-D cover >= 70% of all 65,287 holdout seat-turns). The
ordinal-0-only subset is reported as a robustness check.

Run (needs python >= 3.10 for engine dataclasses, with pyarrow):

    ~/miniforge3/bin/python3 futures/validate_sampler_engine.py
"""

from __future__ import annotations

import collections
import gzip
import json
import math
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (HERE, os.path.join(ROOT, "engine"), os.path.join(ROOT, "agent")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import fit_policy as fp
import validate_sampler as vs
from sampler import TurnSampler, TYPES

import main as ag                       # agent/main.py: posterior + rule policy
from cg.api import (search_begin, search_end, search_step,
                    to_observation_class)

K = 5                                   # playouts per real turn
SEED = 20260807
POSTERIOR_TOP_K = 3
MAX_MAIN_VISITS = 25                    # matches sampler.MAX_VISITS
MAX_STEPS = 160                         # hard cap on engine steps per playout
HOLDOUT = fp.HOLDOUT_DAY

ARCHS = vs.ARCHS                        # same eight + FIELD reporting set


def day_path(fname):
    return os.path.join(ROOT, "data", "mined", HOLDOUT, fname)


# ---------------------------------------------------------------------------
# holdout loaders
# ---------------------------------------------------------------------------
def load_decision_turns():
    """(ep, seat, turn) -> ordered real visits [(ordinal, type|None, multi)],
    plus band / arch / attacked / attached, and the ordmap for entry lookup."""
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
    turns, ordmap = {}, {}
    for key, idxs in groups.items():
        idxs.sort(key=lambda i: d["step"][i])
        visits = []
        for k, i in enumerate(idxs):
            ordmap[(key[0], key[1], d["step"][i])] = k
            visits.append((k, fp.TYPE_NAME.get(d["chosen_option_type"][i]),
                           d["n_options"][i] > 1))
        names = [t for _, t, _ in visits]
        turns[key] = {
            "visits": visits,
            "arch": d["our_archetype"][idxs[0]],
            "band": fp.band_of(d["agent_rating"][idxs[0]]),
            "turn": key[2],
            "attacked": "attack" in names,
            "attached": "attach" in names,
        }
    return turns, ordmap


def load_series_base():
    """(ep, seat, turn) -> (defender board counters, own prizes remaining)
    at the top of that seat's turn — the chip/prize baseline."""
    import pyarrow.parquet as pq
    cols = ["episode_id", "seat", "turn", "damage_dealt_cumulative",
            "prizes_remaining"]
    d = pq.read_table(day_path("series.parquet"), columns=cols).to_pydict()
    return {(d["episode_id"][i], d["seat"][i], d["turn"][i]):
            (d["damage_dealt_cumulative"][i], d["prizes_remaining"][i])
            for i in range(len(d["turn"]))}


def load_position_roots(ordmap):
    """(ep, seat, turn) -> (entry ordinal, record), earliest sampled visit."""
    best = {}
    with gzip.open(day_path("positions.jsonl.gz"), "rt") as fh:
        for line in fh:
            r = json.loads(line)
            if (r.get("context") != fp.CTX_MAIN or not r.get("turn")
                    or r["turn"] < 1):
                continue
            obs = r.get("observation")
            if isinstance(obs, str):
                obs = json.loads(obs)
            if not isinstance(obs, dict) or not obs.get("search_begin_input"):
                continue
            k = ordmap.get((r["episode_id"], r["seat"], r["step"]))
            if k is None:
                continue
            key = (r["episode_id"], r["seat"], r["turn"])
            if key not in best or k < best[key][0]:
                r["observation"] = obs
                best[key] = (k, r)
    return best


# ---------------------------------------------------------------------------
# determinization (replay-derived priors; agent machinery)
# ---------------------------------------------------------------------------
def seat_posterior(player, top_k=POSTERIOR_TOP_K):
    """Posterior over the PLAYED seat's decklist — same hypergeometric
    likelihood as ag._deck_posterior, but conditioned on the seat's visible
    cards INCLUDING its hand (this is the seat's own observation)."""
    seen = ag._visible(player, include_hand=True)
    scored = []
    for counts, plays in ag._load_priors():
        ll = 0.0
        for cid, k in seen.items():
            ll += ag._log_choose(counts.get(cid, 0), k)
            if ll == float("-inf"):
                break
        if ll == float("-inf"):
            continue
        scored.append((math.log(max(plays, 1)) + ll, counts))
    if not scored:
        pri = ag._load_priors()
        if not pri:
            return []
        return [(max(pri, key=lambda cp: cp[1])[0], 1.0)]
    scored.sort(key=lambda t: -t[0])
    scored = scored[:top_k]
    hi = scored[0][0]
    ws = [math.exp(s - hi) for s, _ in scored]
    tot = sum(ws) or 1.0
    return [(c, w / tot) for (_, c), w in zip(scored, ws)]


def pick_weighted(posterior, rng):
    x = rng.random()
    for counts, w in posterior:
        x -= w
        if x <= 0:
            return counts
    return posterior[-1][0]


_BASIC_IDS = None


def basic_ids():
    global _BASIC_IDS
    if _BASIC_IDS is None:
        cards, _, _ = fp.card_tables()
        _BASIC_IDS = {cid for cid, c in cards.items() if c.get("basic")}
    return _BASIC_IDS


# ---------------------------------------------------------------------------
# one engine playout
# ---------------------------------------------------------------------------
def play_turn(rec, entry_k, sampler, disruption_ids, post_me, post_opp, rng):
    """One sampled completion of the turn. Returns a result dict or None."""
    obs = rec["observation"]
    cur = obs["current"]
    me = cur["yourIndex"]
    players = cur["players"]
    mine, opp = players[me], players[1 - me]
    band = fp.band_of(rec.get("agent_rating"))
    arch = rec["our_archetype"]
    turn = rec["turn"]

    # --- determinize hidden zones from the posteriors -----------------------
    counts_me = pick_weighted(post_me, rng)
    seen_me = ag._visible(mine, include_hand=True)
    my_deck, _, my_prize = ag._hidden_from(
        counts_me, seen_me, mine.get("deckCount", 0) or 0, 0,
        len(mine.get("prize") or []), rng)

    counts_o = pick_weighted(post_opp, rng)
    seen_o, n_deck_o, n_hand_o, n_prize_o = ag._opponent_counts(obs)
    opp_deck, opp_hand, opp_prize = ag._hidden_from(
        counts_o, seen_o, n_deck_o, n_hand_o, n_prize_o, rng)

    opp_active = []
    act0 = (opp.get("active") or [None])
    if act0 and act0[0] is None:                    # facedown Active: rare
        pool = [cid for cid in (opp_hand + opp_deck) if cid in basic_ids()]
        if not pool:
            return None
        opp_active = [pool[0]]

    try:
        root = search_begin(to_observation_class(obs), my_deck, my_prize,
                            opp_deck, opp_prize, opp_hand, opp_active)
    except Exception:
        return None

    visits = []                 # (n_options, sampled type)
    avail_log = []              # (ordinal bucket, offered type) diagnostics
    attacked = attached = False
    n_play = n_disr = 0
    decided = over = forced_end = False
    k = entry_k
    state = root
    try:
        for _ in range(MAX_STEPS):
            ob = state.observation
            curx = ob.current
            if curx is None or ag._decided(curx):
                decided = True
                break
            sel = ob.select
            if sel is None or not sel.option:
                break
            opts = list(sel.option)
            mapped = [fp.TYPE_NAME.get(o.type) for o in opts]
            if sel.context == fp.CTX_MAIN and any(mapped):
                if curx.yourIndex != me:
                    over = True                     # defender's MAIN: done
                    break
                avail = sorted({t for t in mapped if t})
                for t in avail:
                    avail_log.append((min(k, 3), t))
                if k - entry_k >= MAX_MAIN_VISITS and "end_turn" in avail:
                    act, forced_end = "end_turn", True
                else:
                    act = sampler.sample_action(band, arch, turn, k, avail,
                                                rng)
                idx = None
                if act == "attack":
                    idx = ag._choose_attack(opts, ob)
                elif act == "attach":
                    idx = ag._choose_attach(opts, ob)
                if idx is None or fp.TYPE_NAME.get(opts[idx].type) != act:
                    idx = next(i for i, t in enumerate(mapped) if t == act)
                visits.append((len(opts), act, k))
                attacked = attacked or act == "attack"
                attached = attached or act == "attach"
                if act == "play":
                    n_play += 1
                    hand = curx.players[me].hand or []
                    hi = getattr(opts[idx], "index", None)
                    if hi is not None and 0 <= hi < len(hand):
                        card = hand[hi]
                        if card is not None and card.id in disruption_ids:
                            n_disr += 1
                k += 1
                choice = [idx]
            elif sel.context == fp.CTX_MAIN:
                # main menu with no table-typed option: rule policy, but the
                # visit still consumes an ordinal (the corpus counted it too)
                if curx.yourIndex != me:
                    over = True
                    break
                choice = ag._policy(ob)
                k += 1
            else:
                choice = ag._policy(ob)             # any sub-select, any seat
            state = search_step(state.searchId, choice)
        completed = decided or over
        if not completed:
            return {"completed": False}
        end = state.observation.current
        counters = 0
        for zone in (end.players[1 - me].active or [],
                     end.players[1 - me].bench or []):
            for mon in zone:
                if mon is not None:
                    counters += ((getattr(mon, "maxHp", 0) or 0)
                                 - (getattr(mon, "hp", 0) or 0))
        return {"completed": True, "decided": decided,
                "visits": visits, "avail": avail_log,
                "attacked": attacked, "attached": attached,
                "n_play": n_play, "n_disr": n_disr,
                "forced_end": forced_end,
                "end_counters": counters,
                "prizes_left": len(end.players[me].prize or [])}
    except Exception:
        return None
    finally:
        try:
            search_end()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# aggregation — identical metric formulas and tolerances to validate_sampler
# ---------------------------------------------------------------------------
def summarize(rows, real_dmg, disr_real, arch_key):
    """One archetype table row from paired per-turn records."""
    r_types, s_types = [], []
    r_atk = s_atk = r_att = s_att = r_att_sfx = s_att_sfx = 0
    n_r = 0
    n_s = 0
    r_chip, r_pr, s_chip, s_pr = [], [], [], []
    s_play = s_disr = 0
    r_len = s_len = 0                    # suffix MAIN visits per turn
    for row in rows:
        n_r += 1
        r_len += row["real_suffix_len"]
        r_types.append(row["real_suffix_multi"])
        r_atk += row["real_attacked"]
        r_att += row["real_attached"]
        r_att_sfx += row["real_attached_sfx"]
        has_pair = row["key"] in real_dmg
        if has_pair:
            c, p = real_dmg[row["key"]]
            r_chip.append(c)
            r_pr.append(p)
        for po in row["playouts"]:
            n_s += 1
            s_len += len(po["visits"])
            s_types.append([t for n, t, _ in po["visits"] if n > 1])
            s_atk += po["attacked"]
            s_att += po["attached"] or row["prefix_attached"]
            s_att_sfx += po["attached"]
            s_play += po["n_play"]
            s_disr += po["n_disr"]
            if has_pair:
                s_chip.append(po["chip"])
                s_pr.append(po["prizes"])
    if not n_r or not n_s:
        return None
    r_mix, r_nvis = vs.mix(r_types)
    s_mix, s_nvis = vs.mix(s_types)
    d_tv = vs.tv(r_mix, s_mix)
    r_atk /= n_r
    s_atk /= n_s
    r_chip_mean = sum(r_chip) / len(r_chip) if r_chip else None
    s_chip_mean = sum(s_chip) / len(s_chip) if s_chip else None
    r_pr_mean = sum(r_pr) / len(r_pr) if r_pr else None
    s_pr_mean = sum(s_pr) / len(s_pr) if s_pr else None
    w1 = vs.w1_hist(r_chip, s_chip)

    dr = disr_real.get(arch_key, collections.Counter())
    n_pl, k_disr = dr.get("n_play", 0), dr.get("disr", 0)
    r_disr = k_disr / n_pl if n_pl else None
    ci = vs.jeffreys(k_disr, n_pl) if n_pl else None
    s_disr_rate = s_disr / s_play if s_play else None

    rec = {
        "n_real_turns": n_r, "n_sampled_turns": n_s,
        "n_real_visits": r_nvis, "n_sampled_visits": s_nvis,
        "visits_per_turn_real": round(r_len / n_r, 2),
        "visits_per_turn_sampled": round(s_len / n_s, 2),
        "n_damage_pairs": len(r_chip),
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
        "attach_rate_real": round(r_att / n_r, 4),
        "attach_rate_sampled": round(s_att / n_s, 4),
        "attach_rate_real_suffix": round(r_att_sfx / n_r, 4),
        "attach_rate_sampled_suffix": round(s_att_sfx / n_s, 4),
        "disr_per_play_real": None if r_disr is None else round(r_disr, 4),
        "disr_per_play_ci": None if ci is None else [round(x, 4) for x in ci],
        "disr_per_play_n": n_pl,
        "disr_per_play_sampled": (None if s_disr_rate is None
                                  else round(s_disr_rate, 4)),
        "n_sampled_play_visits": s_play,
    }
    # --- pre-registered criteria (README A-D, unchanged from Phase 0)
    if r_chip_mean is not None and s_chip_mean is not None:
        chip_tol = max(10.0, 0.15 * r_chip_mean)
        pass_b = (abs(s_chip_mean - r_chip_mean) <= chip_tol
                  and w1 is not None and w1 <= vs.TOL_CHIP_W1)
        pass_c = abs(s_pr_mean - r_pr_mean) <= vs.TOL_PRIZES
    else:
        pass_b = pass_c = False
    pass_a = d_tv <= vs.TOL_TV
    pass_d = abs(s_atk - r_atk) <= vs.TOL_ATTACK
    rec["criteria"] = {"A_action_mix": pass_a, "B_chip": pass_b,
                       "C_prizes": pass_c, "D_attack_rate": pass_d}
    rec["verdict"] = ("GO" if (pass_a and pass_b and pass_c and pass_d)
                      else "NO-GO")
    rec["flags"] = {
        "attach_within_5pt_suffix":
            abs(s_att_sfx / n_s - r_att_sfx / n_r) <= vs.TOL_ATTACH,
        "attach_within_5pt_fullturn":
            abs(s_att / n_s - r_att / n_r) <= vs.TOL_ATTACH,
        "disruption_ok": (r_disr is None or ci is None
                          or s_disr_rate is None
                          or ci[0] <= s_disr_rate <= ci[1]
                          or abs(s_disr_rate - r_disr) <= vs.TOL_DISR),
    }
    return rec


# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    sampler = TurnSampler(os.path.join(HERE, "policy_train.json"))
    _, _, disruption_ids = fp.card_tables()
    print("loading holdout ...", file=sys.stderr)
    dturns, ordmap = load_decision_turns()
    real_dmg = vs.load_real_damage()
    series_base = load_series_base()
    roots = load_position_roots(ordmap)
    disr_real, _ = vs.load_real_menus()
    disr_real = {("FIELD" if a == "*" else a): c for a, c in disr_real.items()}

    vol = collections.Counter(t["arch"] for t in dturns.values())
    total_turns = sum(vol.values())

    # --- play everything ----------------------------------------------------
    per_turn = []
    stats = collections.Counter()
    entry_hist = collections.Counter()
    keys = sorted(roots)
    limit = int(os.environ.get("CABT_P05_LIMIT") or 0)
    if limit:
        keys = keys[:limit]              # pilot runs only; full run leaves unset
    print("playing %d turns x K=%d ..." % (len(keys), K), file=sys.stderr)
    for n_done, key in enumerate(keys):
        entry_k, rec = roots[key]
        dt = dturns.get(key)
        if dt is None or key not in series_base:
            stats["no_real_row"] += 1
            continue
        obs = rec["observation"]
        cur = obs["current"]
        me = cur["yourIndex"]
        players = cur["players"]
        try:
            post_me = seat_posterior(players[me])
            post_opp = ag._deck_posterior(obs, top_k=POSTERIOR_TOP_K)
        except Exception:
            stats["posterior_fail"] += 1
            continue
        if not post_me or not post_opp:
            stats["posterior_fail"] += 1
            continue
        base_counters, base_prizes = series_base[key]
        playouts = []
        for j in range(K):
            rng = random.Random((key[0] * 1000003) ^ (key[1] << 21)
                                ^ (key[2] << 24) ^ (j * 7919) ^ SEED)
            po = play_turn(rec, entry_k, sampler, disruption_ids,
                           post_me, post_opp, rng)
            if po is None:
                stats["playout_error"] += 1
                continue
            if not po.get("completed"):
                stats["playout_incomplete"] += 1
                continue
            po["chip"] = 10 * round(
                max(0, min(fp.CHIP_CAP, po["end_counters"] - base_counters))
                / 10)
            po["prizes"] = max(0, base_prizes - po["prizes_left"])
            stats["playout_ok"] += 1
            stats["forced_end"] += po["forced_end"]
            playouts.append(po)
        if not playouts:
            stats["turn_dropped"] += 1
            continue
        entry_hist[str(min(entry_k, 5)) + ("+" if entry_k >= 5 else "")] += 1
        prefix = [v for v in dt["visits"] if v[0] < entry_k]
        suffix = [v for v in dt["visits"] if v[0] >= entry_k]
        per_turn.append({
            "key": key, "arch": dt["arch"], "band": dt["band"],
            "turn": dt["turn"], "entry_k": entry_k,
            "real_suffix_multi": [t for _, t, m in suffix if m and t],
            "real_suffix_len": len(suffix),
            "real_attacked": dt["attacked"],
            "real_attached": dt["attached"],
            "real_attached_sfx": any(t == "attach" for _, t, _ in suffix),
            "prefix_attached": any(t == "attach" for _, t, _ in prefix),
            "playouts": playouts,
        })
        if (n_done + 1) % 100 == 0:
            print("  %d/%d turns  (%.0fs)" % (n_done + 1, len(keys),
                                              time.time() - t0),
                  file=sys.stderr)

    # --- aggregate ----------------------------------------------------------
    by_arch = collections.defaultdict(list)
    for row in per_turn:
        by_arch[row["arch"]].append(row)

    results = {
        "holdout_day": HOLDOUT, "K": K, "seed": SEED,
        "design": "engine playouts from mined positions, earliest-ordinal "
                  "entry per turn; priors-determinized hidden zones; "
                  "paired real side",
        "n_roots": len(keys), "n_turns_played": len(per_turn),
        "entry_ordinal_hist": dict(sorted(entry_hist.items())),
        "playout_stats": dict(stats),
        "archetypes": {}, "turnstart_subset": {},
    }
    go_volume = 0
    for arch in ARCHS + ["FIELD"]:
        rows = per_turn if arch == "FIELD" else by_arch.get(arch, [])
        rec = summarize(rows, real_dmg, disr_real, arch) if rows else None
        if rec is None:
            results["archetypes"][arch] = {"n_real_turns": 0,
                                           "verdict": "NO-DATA"}
            print("  %-26s no positions in holdout sample" % arch,
                  file=sys.stderr)
            continue
        rec["other_backoff"] = sampler.arch_label(arch) == "OTHER"
        rec["holdout_turn_share"] = round(vol.get(arch, 0) / total_turns, 4)
        results["archetypes"][arch] = rec
        if arch != "FIELD" and rec["verdict"] == "GO":
            go_volume += vol.get(arch, 0)
        print("  %-26s n=%4d TV=%.3f chipΔ=%s W1=%s atkΔ=%+.3f  %s"
              % (arch, rec["n_real_turns"], rec["tv"],
                 "-" if rec["chip_mean_real"] is None else
                 "%+.1f" % (rec["chip_mean_sampled"] - rec["chip_mean_real"]),
                 "-" if rec["chip_w1"] is None else "%.1f" % rec["chip_w1"],
                 rec["attack_rate_sampled"] - rec["attack_rate_real"],
                 rec["verdict"]), file=sys.stderr)

        # robustness: ordinal-0 entries only (full-turn playouts)
        rows0 = [r for r in rows if r["entry_k"] == 0]
        rec0 = summarize(rows0, real_dmg, disr_real, arch) if rows0 else None
        if rec0 is not None:
            results["turnstart_subset"][arch] = rec0

    # engine-availability diagnostics (FIELD): share of sampled main menus
    # offering each type, by ordinal bucket — the quantity Phase 0's offline
    # surrogate could not recover
    av = collections.defaultdict(collections.Counter)
    menus = collections.Counter()
    for row in per_turn:
        for po in row["playouts"]:
            for ob_k, t in po["avail"]:
                av[ob_k][t] += 1
                if t == "end_turn":      # offered on every main menu
                    menus[ob_k] += 1
    results["engine_availability"] = {
        str(ob_k): {t: round(av[ob_k][t] / menus[ob_k], 4) for t in TYPES
                    if menus[ob_k]}
        for ob_k in sorted(menus)}

    results["coverage"] = {
        "go_turn_share": round(go_volume / total_turns, 4),
        "threshold": vs.COVERAGE_GO,
        "total_holdout_turns": total_turns,
    }
    results["phase05_verdict"] = ("GO" if go_volume / total_turns
                                  >= vs.COVERAGE_GO else "NO-GO")

    dest = os.path.join(HERE, "results_engine.json")
    with open(dest, "w") as fh:
        json.dump(results, fh, indent=1)
    print("phase 0.5 verdict (engine legality): %s (GO archetypes cover "
          "%.1f%% of holdout turns)" % (results["phase05_verdict"],
                                        100 * go_volume / total_turns))
    print("wrote %s  (%.0fs)" % (dest, time.time() - t0))


if __name__ == "__main__":
    main()
