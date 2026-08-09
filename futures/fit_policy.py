"""Refit the opponent reply tables on train days only, plus Phase 0 heads.

Adapted from scripts/build_opponent_policy.py (same cells, same backoff levels,
same availability construction) with three changes for the Path B Phase 0 kill
test (futures/README.md):

1. Holdout discipline: the shipped data/opponent_policy.json was counted over
   all seven mined days, so nothing in the corpus can validate it. This fit
   reads TRAIN_DAYS only (everything but HOLDOUT_DAY) and writes to
   futures/policy_train.json, leaving the shipped file untouched.
2. Damage/prize head: per (archetype, turn bucket), the empirical joint of
   (prizes taken, chip damage) per seat-turn, split by whether the turn
   attacked. Chip damage and prizes come from paired series.parquet snapshots
   (see README: the "cumulative" column is damage-counters-on-board), which
   exist on 2026-08-04/05 in train.
3. Disruption head: P(chosen play is a disruption card | play visit) per
   archetype, off the train positions sample, card ids resolved by name from
   the licensed engine dump at fit time; only rates are stored.

Run with system python3 (pyarrow lives there, not in the venv):

    /usr/bin/python3 futures/fit_policy.py
"""

from __future__ import annotations

import collections
import glob
import gzip
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))

HOLDOUT_DAY = "2026-08-06"

TYPE_NAME = {7: "play", 8: "attach", 9: "evolve", 10: "ability",
             12: "retreat", 13: "attack", 14: "end_turn"}
NAME_TYPE = {v: k for k, v in TYPE_NAME.items()}
TYPES = ["ability", "evolve", "play", "attach", "attack", "retreat", "end_turn"]

CTX_MAIN = 0
AREA_ACTIVE = 4
OPT_ATTACK = 13
OPT_ATTACH = 8
OPT_PLAY = 7

BANDS = [("<900", None, 900.0), ("900-1050", 900.0, 1050.0),
         (">1050", 1050.0, None)]
TURN_BUCKETS = [("1-2", 1, 2), ("3-5", 3, 5), ("6-9", 6, 9), ("10+", 10, 10 ** 9)]
ORD_BUCKETS = ["0", "1", "2", "3+"]

TOP_ARCH = ["Marnie's Grimmsnarl ex", "Fezandipiti ex", "Mega Lopunny ex",
            "Mega Kangaskhan ex", "Teal Mask Ogerpon ex", "Dragapult ex",
            "Cynthia's Garchomp ex", "Team Rocket's Mewtwo ex"]
OTHER = "OTHER"

MIN_CELL = 200          # decision cells, as in the parent script
MIN_HEAD = 100          # damage-head cells back off below this many turns

# Disruption class (README): resolved to card ids by exact name at fit time.
DISRUPTION_NAMES = ("Judge", "Unfair Stamp", "Boss’s Orders",
                    "Prime Catcher", "Eri")

CHIP_CAP = 500          # chip damage histogram cap, multiples of 10


def band_of(rating):
    if rating is None or rating != rating:
        return None
    for name, lo, hi in BANDS:
        if (lo is None or rating >= lo) and (hi is None or rating < hi):
            return name
    return None


def turn_bucket(turn):
    for name, lo, hi in TURN_BUCKETS:
        if lo <= turn <= hi:
            return name
    return None


def ord_bucket(k):
    return ORD_BUCKETS[k] if k < 3 else "3+"


def arch_of(name):
    return name if name in TOP_ARCH else OTHER


def train_day_paths(fname):
    for path in sorted(glob.glob(os.path.join(ROOT, "data", "mined", "*", fname))):
        if os.path.basename(os.path.dirname(path)) != HOLDOUT_DAY:
            yield path


# ---------------------------------------------------------------------------
# pass 1 — the counted cells (train days only)
# ---------------------------------------------------------------------------
def build_cells():
    import pyarrow.parquet as pq
    cols = ["episode_id", "seat", "step", "turn", "context", "n_options",
            "chosen_option_type", "our_archetype", "agent_rating"]
    levels = {"L0": collections.defaultdict(collections.Counter),
              "L1": collections.defaultdict(collections.Counter),
              "L2": collections.defaultdict(collections.Counter),
              "L3": collections.defaultdict(collections.Counter)}
    ordinal = {}
    attacked = set()          # (episode, seat, turn) with an attack chosen
    stats = collections.Counter()

    for path in train_day_paths("decisions.parquet"):
        d = pq.read_table(path, columns=cols).to_pydict()
        n = len(d["context"])
        stats["rows"] += n
        groups = collections.defaultdict(list)
        for i in range(n):
            if d["context"][i] != CTX_MAIN or d["turn"][i] < 1:
                continue
            if d["chosen_option_type"][i] == OPT_ATTACK:
                attacked.add((d["episode_id"][i], d["seat"][i], d["turn"][i]))
            groups[(d["episode_id"][i], d["seat"][i], d["turn"][i])].append(i)
        for key, idxs in groups.items():
            idxs.sort(key=lambda i: d["step"][i])
            for k, i in enumerate(idxs):
                ordinal[(d["episode_id"][i], d["seat"][i], d["step"][i])] = \
                    ord_bucket(k)
                if d["n_options"][i] <= 1:
                    stats["forced"] += 1
                    continue
                t = TYPE_NAME.get(d["chosen_option_type"][i])
                if t is None:
                    stats["off_type"] += 1
                    continue
                tb = turn_bucket(d["turn"][i])
                if tb is None:
                    continue
                ob = ord_bucket(k)
                a = arch_of(d["our_archetype"][i])
                b = band_of(d["agent_rating"][i])
                stats["counted"] += 1
                if b is not None:
                    levels["L0"]["%s|%s|%s|%s" % (b, a, tb, ob)][t] += 1
                else:
                    stats["no_rating"] += 1
                levels["L1"]["%s|%s|%s" % (a, tb, ob)][t] += 1
                levels["L2"]["%s|%s" % (tb, ob)][t] += 1
                levels["L3"][ob][t] += 1
        print("  %s: %d rows" % (os.path.basename(os.path.dirname(path)), n),
              file=sys.stderr)
    return levels, ordinal, attacked, stats


# ---------------------------------------------------------------------------
# pass 2 — availability, sub-choice profiles, disruption head (train days)
# ---------------------------------------------------------------------------
def card_tables():
    with open(os.path.join(ROOT, "data", "engine_dump", "attacks.json")) as fh:
        attacks = {a["attackId"]: a for a in json.load(fh)}
    with open(os.path.join(ROOT, "data", "engine_dump", "cards.json")) as fh:
        cards = {c["cardId"]: c for c in json.load(fh)}
    disruption_ids = {c["cardId"] for c in cards.values()
                      if c.get("name") in DISRUPTION_NAMES}
    return cards, attacks, disruption_ids


def score_attack(obs, opts, ci, cards, attacks, out, out_arch):
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
            if o.get("type") == OPT_ATTACK]
    if len(cand) < 2:
        return
    out["n"] += 1
    out_arch["n"] += 1
    big = max(d for _, d in cand)
    kos = [(i, d) for i, d in cand if thp is not None and d >= thp]
    if dmg(opts[ci]) == big:
        out["highest_damage"] += 1
        out_arch["highest_damage"] += 1
    if kos:
        out["ko_available"] += 1
        out_arch["ko_available"] += 1
        cheap = min(d for _, d in kos)
        if ci in [i for i, d in kos if d == cheap]:
            out["cheapest_ko"] += 1
            out_arch["cheapest_ko"] += 1
        elif ci in [i for i, _ in kos]:
            out["some_ko"] += 1


def build_menus(ordinal):
    avail = collections.defaultdict(collections.Counter)
    avail_n = collections.Counter()
    attack = collections.Counter()
    attach = collections.Counter()
    attack_by_arch = collections.defaultdict(collections.Counter)
    disrupt = collections.defaultdict(collections.Counter)   # arch -> counts

    cards, attacks, disruption_ids = card_tables()
    for path in train_day_paths("positions.jsonl.gz"):
        with gzip.open(path, "rt") as fh:
            for line in fh:
                r = json.loads(line)
                if r.get("context") != CTX_MAIN:
                    continue
                obs = r.get("observation")
                if isinstance(obs, str):
                    obs = json.loads(obs)
                if not isinstance(obs, dict):
                    continue
                sel = obs.get("select") or {}
                opts = sel.get("option") or []
                if len(opts) <= 1:
                    continue
                ob = ordinal.get((r["episode_id"], r["seat"], r["step"]))
                tb = turn_bucket(r.get("turn") or -1)
                if ob is None or tb is None:
                    continue
                types = {o.get("type") for o in opts}
                for key in ("%s|%s" % (tb, ob), ob):
                    avail_n[key] += 1
                    for t in types:
                        if t in TYPE_NAME:
                            avail[key][TYPE_NAME[t]] += 1
                chosen = r.get("chosen") or []
                if not chosen:
                    continue
                ci = chosen[0]
                if not (0 <= ci < len(opts)):
                    continue
                pick = opts[ci]
                arch = arch_of(r.get("our_archetype"))
                if pick.get("type") == OPT_ATTACK:
                    score_attack(obs, opts, ci, cards, attacks, attack,
                                 attack_by_arch[arch])
                if pick.get("type") == OPT_ATTACH:
                    cand = [o for o in opts if o.get("type") == OPT_ATTACH]
                    if len(cand) > 1:
                        attach["n"] += 1
                        attach["active" if pick.get("inPlayArea") == AREA_ACTIVE
                               else "bench"] += 1
                if pick.get("type") == OPT_PLAY:
                    cur = obs.get("current") or {}
                    players = cur.get("players") or []
                    me = cur.get("yourIndex", 0)
                    if len(players) == 2:
                        hand = players[me].get("hand") or []
                        hi = pick.get("index")
                        if isinstance(hi, int) and 0 <= hi < len(hand):
                            cid = (hand[hi] or {}).get("id")
                            disrupt[arch]["n_play"] += 1
                            if cid in disruption_ids:
                                disrupt[arch]["disr"] += 1
    return avail, avail_n, attack, attach, attack_by_arch, disrupt


# ---------------------------------------------------------------------------
# pass 3 — damage/prize head off paired series snapshots (train days with series)
# ---------------------------------------------------------------------------
def build_damage_head(attacked):
    import pyarrow.parquet as pq
    cols = ["episode_id", "seat", "turn", "our_archetype",
            "damage_dealt_cumulative", "damage_taken_cumulative",
            "prizes_remaining", "opp_prizes_remaining"]
    # (episode, seat, turn) -> row dict, per day to bound memory
    head = collections.defaultdict(lambda: {
        "n": 0, "n_attack": 0,
        "A": {"prizes": collections.Counter(),
              "chip0": collections.Counter(), "chip1": collections.Counter()},
        "N": {"prizes": collections.Counter(),
              "chip0": collections.Counter(), "chip1": collections.Counter()},
    })
    n_pairs = 0
    for path in train_day_paths("series.parquet"):
        d = pq.read_table(path, columns=cols).to_pydict()
        rows = {}
        n = len(d["turn"])
        for i in range(n):
            rows[(d["episode_id"][i], d["seat"][i], d["turn"][i])] = i
        for i in range(n):
            ep, s, t = d["episode_id"][i], d["seat"][i], d["turn"][i]
            j = rows.get((ep, 1 - s, t + 1))
            if j is None:
                continue          # terminal turn — censored, see README
            before = d["damage_dealt_cumulative"][i]
            after = d["damage_taken_cumulative"][j]
            chip = max(0, min(CHIP_CAP, after - before))
            chip = 10 * round(chip / 10)
            prizes = max(0, d["prizes_remaining"][i]
                         - d["opp_prizes_remaining"][j])
            atk = (ep, s, t) in attacked
            tb = turn_bucket(t)
            if tb is None:
                continue
            arch = arch_of(d["our_archetype"][i])
            n_pairs += 1
            for key in ("%s|%s" % (arch, tb), "*|%s" % tb, "*"):
                cell = head[key]
                cell["n"] += 1
                cell["n_attack"] += int(atk)
                sub = cell["A" if atk else "N"]
                sub["prizes"][min(prizes, 3)] += 1
                sub["chip1" if prizes >= 1 else "chip0"][chip] += 1
        print("  %s: series paired" % os.path.basename(os.path.dirname(path)),
              file=sys.stderr)
    out = {}
    for key, cell in head.items():
        if cell["n"] < MIN_HEAD and key != "*":
            continue
        out[key] = {
            "n": cell["n"],
            "attack_rate": round(cell["n_attack"] / cell["n"], 4),
            "A": {k: {str(v): c for v, c in sorted(cell["A"][k].items())}
                  for k in ("prizes", "chip0", "chip1")},
            "N": {k: {str(v): c for v, c in sorted(cell["N"][k].items())}
                  for k in ("prizes", "chip0", "chip1")},
        }
    return out, n_pairs


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------
def main():
    train_days = sorted(os.path.basename(os.path.dirname(p))
                        for p in train_day_paths("decisions.parquet"))
    print("train days: %s (holdout %s)" % (", ".join(train_days), HOLDOUT_DAY),
          file=sys.stderr)
    print("counting decisions ...", file=sys.stderr)
    levels, ordinal, attacked, stats = build_cells()
    print("counting menus ...", file=sys.stderr)
    avail, avail_n, attack, attach, attack_by_arch, disrupt = build_menus(ordinal)
    print("pairing series snapshots ...", file=sys.stderr)
    damage_head, n_pairs = build_damage_head(attacked)

    out = {
        "schema": 1,
        "built_by": "futures/fit_policy.py",
        "source": {
            "train_days": train_days,
            "holdout_day": HOLDOUT_DAY,
            "decision_rows": stats["rows"],
            "main_decisions_counted": stats["counted"],
            "forced_menus_skipped": stats["forced"],
            "series_turn_pairs": n_pairs,
        },
        "types": TYPES,
        "bands": [b[0] for b in BANDS],
        "default_band": "900-1050",
        "turn_buckets": [t[0] for t in TURN_BUCKETS],
        "ord_buckets": ORD_BUCKETS,
        "archetypes": TOP_ARCH + [OTHER],
        "min_cell": MIN_CELL,
        "availability": {},
        "cells": {},
        "attack_profile": {},
        "attach_profile": {},
        "damage_head": damage_head,
        "disruption_head": {},
    }

    keys = (["%s|%s" % (t[0], o) for t in TURN_BUCKETS for o in ORD_BUCKETS]
            + ORD_BUCKETS)
    for key in keys:
        n = avail_n.get(key, 0)
        if not n:
            continue
        out["availability"][key] = {
            "n_menus": n,
            "rate": {t: round(avail[key][t] / n, 4) for t in TYPES
                     if avail[key].get(t)},
        }

    for lvl in ("L0", "L1", "L2", "L3"):
        for key, c in sorted(levels[lvl].items()):
            n = sum(c.values())
            if n < MIN_CELL:
                continue
            out["cells"]["%s|%s" % (lvl, key)] = {
                "n": n,
                "p": {t: round(c[t] / n, 4) for t in TYPES if c.get(t)},
            }

    if attack.get("n"):
        out["attack_profile"] = {
            "n_multi_attack_menus": attack["n"],
            "highest_damage_rate": round(attack["highest_damage"] / attack["n"], 4),
            "ko_available_menus": attack["ko_available"],
            "cheapest_ko_rate": (round(attack["cheapest_ko"] / attack["ko_available"], 4)
                                 if attack["ko_available"] else None),
        }
    if attach.get("n"):
        out["attach_profile"] = {
            "n_multi_attach_menus": attach["n"],
            "active_rate": round(attach["active"] / attach["n"], 4),
        }

    pooled = collections.Counter()
    for arch, c in disrupt.items():
        pooled.update(c)
        if c["n_play"] >= 50:
            out["disruption_head"][arch] = {
                "n_play": c["n_play"], "disr": c["disr"],
                "rate": round(c["disr"] / c["n_play"], 4)}
    if pooled.get("n_play"):
        out["disruption_head"]["*"] = {
            "n_play": pooled["n_play"], "disr": pooled["disr"],
            "rate": round(pooled["disr"] / pooled["n_play"], 4)}

    dest = os.path.join(HERE, "policy_train.json")
    with open(dest, "w") as fh:
        json.dump(out, fh, separators=(",", ":"))
    print("wrote %s (%.0f KB, %d cells, %d damage-head cells)"
          % (dest, os.path.getsize(dest) / 1024, len(out["cells"]),
             len(damage_head)))


if __name__ == "__main__":
    main()
