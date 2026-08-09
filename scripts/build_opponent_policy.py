"""Band-conditional opponent reply tables, counted off the mined ladder corpus.

Playbook entry 4. The search rolls the opponent's turn out under *their* measured
behaviour instead of our own priority table, which is the one thing both prior
2-ply rejections named as missing (SEARCH_OPP_BRANCH's comment block: 0.467 and
40.7%, both under our-priority-table fiction).

What is counted
---------------
Every main-menu decision in `data/mined/*/decisions.parquet` where the menu had
more than one option: 2,722,370 of the corpus's 5,345,363 rows. For each one the
cell is

    (rating band, archetype, game-turn bucket, within-turn ordinal bucket)

and the tally is over the seven main-menu option types (ability, evolve, play,
attach, attack, retreat, end_turn). Everything in the output file is a count or
a ratio of counts.

Two conditioners deserve their justification in the file that computes them.

*Within-turn ordinal.* A turn is a sequence of main-menu visits, not one
decision: the field plays its cards, then attaches, then attacks and the turn
ends. A distribution over visits pooled across the turn therefore describes no
position — it would put attack's rate on the first visit of the turn, where
attacking is exactly what the field does not do. The ordinal (this seat's k-th
main decision inside this game turn, bucketed 0 / 1 / 2 / 3+) is recoverable
from the corpus by sorting each (episode, seat, turn) group by step, and it is
what separates "opening the turn" from "closing it".

*Availability.* The decision rows record which option type was chosen and not
which were offered, so a raw rate confounds preference with legality — attack is
chosen on 10.4% of main menus but is only legal on roughly half of them. The
menus themselves survive in `positions.jsonl.gz` (2,000 sampled observations a
day, complete with `select.option`), so availability is counted there, keyed to
the same turn and ordinal buckets by joining back to the decision rows on
(episode, seat, step). The shipped score for a type is

    propensity(type) = P(chosen | cell) / P(available | turn, ordinal)

renormalised at run time over the types the live menu actually offers. Both
factors are printed; a reader can divide them again.

Sub-choices inside a type come from the same positions sample, which is the only
source that carries the option list: how often a seat with two or more attacks
took the hardest hit rather than the cheapest knockout, and how often an attach
went to the Active rather than the Bench.

    /usr/bin/python3 scripts/build_opponent_policy.py

System python3 on purpose: pyarrow lives there (21.0.0) and not in the project
venv, and nothing in this file needs the engine.
"""

from __future__ import annotations

import collections
import glob
import gzip
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The seven main-menu types, in agent/main.py's constant order.
TYPE_NAME = {7: "play", 8: "attach", 9: "evolve", 10: "ability",
             12: "retreat", 13: "attack", 14: "end_turn"}
NAME_TYPE = {v: k for k, v in TYPE_NAME.items()}
TYPES = ["ability", "evolve", "play", "attach", "attack", "retreat", "end_turn"]

CTX_MAIN = 0
AREA_ACTIVE = 4

# Rating bands. Ladder ratings in the corpus run 838 (1st pct) to 1164 (99th),
# median 1026, so these three split the field roughly 10 / 50 / 40.
BANDS = [("<900", None, 900.0), ("900-1050", 900.0, 1050.0),
         (">1050", 1050.0, None)]

TURN_BUCKETS = [("1-2", 1, 2), ("3-5", 3, 5), ("6-9", 6, 9), ("10+", 10, 10 ** 9)]
ORD_BUCKETS = ["0", "1", "2", "3+"]

# Top 8 archetypes by main-menu decision volume over the seven mined days.
# Everything else is OTHER. Both gate decks (Marnie's Grimmsnarl, Cynthia's
# Garchomp) are inside the eight.
TOP_ARCH = ["Marnie's Grimmsnarl ex", "Fezandipiti ex", "Mega Lopunny ex",
            "Mega Kangaskhan ex", "Teal Mask Ogerpon ex", "Dragapult ex",
            "Cynthia's Garchomp ex", "Team Rocket's Mewtwo ex"]
OTHER = "OTHER"

MIN_CELL = 200          # below this a cell backs off instead of being trusted


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


# ---------------------------------------------------------------------------
# pass 1 — the counted cells
# ---------------------------------------------------------------------------
def read_days():
    import pyarrow.parquet as pq
    cols = ["episode_id", "seat", "step", "turn", "context", "n_options",
            "chosen_option_type", "our_archetype", "agent_rating"]
    for path in sorted(glob.glob(os.path.join(ROOT, "data", "mined", "*",
                                              "decisions.parquet"))):
        yield path, pq.read_table(path, columns=cols).to_pydict()


def build_cells():
    """Counts at four backoff levels, plus the (episode, seat, step) ordinal map."""
    levels = {"L0": collections.defaultdict(collections.Counter),  # band|arch|turn|ord
              "L1": collections.defaultdict(collections.Counter),  # arch|turn|ord
              "L2": collections.defaultdict(collections.Counter),  # turn|ord
              "L3": collections.defaultdict(collections.Counter)}  # ord
    ordinal = {}          # (episode_id, seat, step) -> ordinal bucket, for pass 2
    stats = collections.Counter()

    for path, d in read_days():
        n = len(d["context"])
        stats["rows"] += n
        # Ordinal within (episode, seat, game turn), over main menus only, in
        # step order. The parquet is already written in step order per episode,
        # but sorting the group is what makes that irrelevant.
        groups = collections.defaultdict(list)
        for i in range(n):
            if d["context"][i] != CTX_MAIN or d["turn"][i] < 1:
                continue
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
        print("  %s: %d rows, %d main decisions counted"
              % (os.path.basename(os.path.dirname(path)), n, stats["counted"]),
              file=sys.stderr)
    return levels, ordinal, stats


# ---------------------------------------------------------------------------
# pass 2 — availability and the sub-choice profiles, off the sampled menus
# ---------------------------------------------------------------------------
def build_menus(ordinal):
    avail = collections.defaultdict(collections.Counter)   # bucket -> type -> menus
    avail_n = collections.Counter()                        # bucket -> menus
    attack = collections.Counter()
    attach = collections.Counter()
    attack_by_arch = collections.defaultdict(collections.Counter)

    cards, attacks = card_tables()
    for path in sorted(glob.glob(os.path.join(ROOT, "data", "mined", "*",
                                              "positions.jsonl.gz"))):
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
                if pick.get("type") == NAME_TYPE["attack"]:
                    score_attack(obs, opts, ci, cards, attacks, attack,
                                 attack_by_arch[arch])
                if pick.get("type") == NAME_TYPE["attach"]:
                    cand = [o for o in opts
                            if o.get("type") == NAME_TYPE["attach"]]
                    if len(cand) > 1:
                        attach["n"] += 1
                        attach["active" if pick.get("inPlayArea") == AREA_ACTIVE
                               else "bench"] += 1
    return avail, avail_n, attack, attach, attack_by_arch


def card_tables():
    """Attack damage and card HP off the engine dump — no cg import needed."""
    with open(os.path.join(ROOT, "data", "engine_dump", "attacks.json")) as fh:
        attacks = {a["attackId"]: a for a in json.load(fh)}
    with open(os.path.join(ROOT, "data", "engine_dump", "cards.json")) as fh:
        cards = {c["cardId"]: c for c in json.load(fh)}
    return cards, attacks


def score_attack(obs, opts, ci, cards, attacks, out, out_arch):
    """Was the pick the hardest hit, the cheapest knockout, or neither?

    Damage is the printed number doubled on a weakness match, exactly as
    `agent/main.py:_damage_against` reads it, so the rate measured here is the
    rate of the behaviour the agent would reproduce.
    """
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
            if o.get("type") == NAME_TYPE["attack"]]
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


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------
def dist(counter):
    n = sum(counter.values())
    return {t: counter[t] / n for t in TYPES if counter.get(t)}, n


def tv_distance(a, b):
    return 0.5 * sum(abs(a.get(t, 0.0) - b.get(t, 0.0)) for t in TYPES)


def self_check(levels):
    """Most-different archetype pairs by mean table distance.

    Total-variation distance between the two archetypes' type distributions,
    averaged over every (turn bucket, ordinal bucket) cell both of them fill to
    MIN_CELL. Aggro and setup decks should not agree.
    """
    by_arch = collections.defaultdict(dict)
    for key, c in levels["L1"].items():
        a, tb, ob = key.split("|")
        if sum(c.values()) >= MIN_CELL:
            by_arch[a][(tb, ob)] = dist(c)[0]
    names = sorted(by_arch)
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            shared = set(by_arch[names[i]]) & set(by_arch[names[j]])
            if len(shared) < 8:
                continue
            ds = [tv_distance(by_arch[names[i]][k], by_arch[names[j]][k])
                  for k in shared]
            pairs.append({"a": names[i], "b": names[j],
                          "mean_tv": round(sum(ds) / len(ds), 4),
                          "max_tv": round(max(ds), 4), "cells": len(shared)})
    pairs.sort(key=lambda p: -p["mean_tv"])
    return pairs


def main():
    print("counting decisions ...", file=sys.stderr)
    levels, ordinal, stats = build_cells()
    print("counting menus ...", file=sys.stderr)
    avail, avail_n, attack, attach, attack_by_arch = build_menus(ordinal)

    out = {
        "schema": 1,
        "built_by": "scripts/build_opponent_policy.py",
        "source": {
            "days": sorted(os.path.basename(os.path.dirname(p)) for p in
                           glob.glob(os.path.join(ROOT, "data", "mined", "*",
                                                  "decisions.parquet"))),
            "decision_rows": stats["rows"],
            "main_decisions_counted": stats["counted"],
            "forced_menus_skipped": stats["forced"],
            "off_type_skipped": stats["off_type"],
            "unrated_seats_skipped_at_L0": stats["no_rating"],
            "menus_sampled": sum(avail_n.get(o, 0) for o in ORD_BUCKETS),
        },
        "types": TYPES,
        "bands": [b[0] for b in BANDS],
        # The bundled constant the agent reads: which band's behaviour to roll
        # the opponent's turn out under. Ladder ratings in the corpus have
        # median 1026 and quartiles 987 / 1076, so the middle band is where the
        # opponents we actually meet sit. Overridable with CABT_OPP_BAND.
        "default_band": "900-1050",
        "turn_buckets": [t[0] for t in TURN_BUCKETS],
        "ord_buckets": ORD_BUCKETS,
        "archetypes": TOP_ARCH + [OTHER],
        "min_cell": MIN_CELL,
        "availability": {},
        "cells": {},
        "attack_profile": {},
        "attach_profile": {},
        "self_check": {},
    }

    # Keyed "<turn bucket>|<ordinal>" with an "<ordinal>" backoff for the thin
    # corners. A menu counted at the finer key is counted at the coarser one too,
    # so the two levels are the same menus read at two resolutions.
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
            "any_ko_rate": (round((attack["cheapest_ko"] + attack["some_ko"])
                                  / attack["ko_available"], 4)
                            if attack["ko_available"] else None),
            "by_archetype": {a: {"n": c["n"],
                                 "highest_damage_rate": round(c["highest_damage"] / c["n"], 4)}
                             for a, c in sorted(attack_by_arch.items()) if c["n"] >= 30},
        }
    if attach.get("n"):
        out["attach_profile"] = {
            "n_multi_attach_menus": attach["n"],
            "active_rate": round(attach["active"] / attach["n"], 4),
            "bench_rate": round(attach["bench"] / attach["n"], 4),
        }

    out["self_check"]["archetype_pairs_most_different"] = self_check(levels)[:5]
    out["self_check"]["cells_kept"] = len(out["cells"])

    dest = os.path.join(ROOT, "data", "opponent_policy.json")
    with open(dest, "w") as fh:
        json.dump(out, fh, separators=(",", ":"), sort_keys=False)
    print("wrote %s (%.0f KB, %d cells)"
          % (dest, os.path.getsize(dest) / 1024, len(out["cells"])))
    for p in out["self_check"]["archetype_pairs_most_different"]:
        print("  %-26s vs %-26s  mean TV %.3f over %d cells"
              % (p["a"], p["b"], p["mean_tv"], p["cells"]))


if __name__ == "__main__":
    main()
