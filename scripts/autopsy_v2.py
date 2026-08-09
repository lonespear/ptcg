"""Autopsy of the v2 agent's real ladder games: who beat us, and how.

Pulls every completed episode of one of our submissions off the public Kaggle
episode API, replays each game from the recorded observations, and classifies
every loss by proximate cause. The wins run through the same detectors as the
control group, so each detector's alarm rate in games we won anyway is its
noise floor.

Usage (list + replays fetched separately, see --help):

    python scripts/autopsy_v2.py --list list_v2.json --replays replays_v2/ \
        --submission 55311822 --team "Lemmes Yad" \
        --out data/analysis/ladder_autopsy_v2.json

Schema facts this rests on (verified on the 2026-08-07 v2 replays):
  * `steps[1][seat].action` is that seat's 60-card deck list — the opponent's
    archetype is labeled from THAT seat's deck action, not from
    steps[0].visualize (the v1 autopsy's labeling bug);
  * the decision made against `steps[t][seat].observation.select` is recorded
    at `steps[t+1][seat].action` (same convention `ptcg/extract.py` verified);
  * `steps[0][0].visualize` is the full spectator frame stream; its `logs` use
    string type names and include the terminal `Result` log
    (reason 1 = prizes, 2 = deck-out, 3 = no active, 4 = card effect) that
    never appears in either seat's own observation logs;
  * `players[i].prize` is player i's *remaining* face-down prizes, so a drop
    of k means player i just took k prizes;
  * a knockout shows up in the frame logs as the victim's Pokémon cards moving
    ACTIVE/BENCH -> DISCARD alongside the taker's PRIZE -> HAND draws.

Damage model (deliberately conservative, flat-printed-only):
    damage = printed base (variable/multiplier attacks count 0)
             x2 if the attacker card's type matches the defender's weakness
             -30 if it matches the defender's resistance
Attack ids resolve through the engine dump under data/engine_dump/ (local,
uncommitted); card identity/prize denominations come from the competition CSV.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import os
os.environ.setdefault("PTCG_DATA_DIR", str(ROOT / "external" / "grimmsnarl"))

from ptcg import load_cards  # noqa: E402

# option / area / context constants (cg/api.py)
OPT_CARD, OPT_RETREAT, OPT_ATTACK, OPT_END = 3, 12, 13, 14
AREA_ACTIVE, AREA_BENCH, AREA_DISCARD, AREA_PRIZE, AREA_HAND = 4, 5, 3, 6, 2
CTX_MAIN, CTX_TO_ACTIVE = 0, 4
RAINBOW, TEAM_ROCKET = 10, 11

CONDITIONAL_RE = re.compile(r"this attack does nothing|if .*?, this attack",
                            re.IGNORECASE)
COIN_RE = re.compile(r"flip .*?coin", re.IGNORECASE)

# Computable damage riders: printed text -> a function of the visible board.
# Anything not matched contributes nothing beyond the printed base, which
# makes every damage estimate a lower bound (except coin-gated base damage,
# which is excluded from lethal claims outright).
RIDER_PATTERNS = [
    (re.compile(r"(\d+) (?:more )?damage for each Energy attached to both "
                r"Active Pok", re.I),
     lambda n, am, dm, ap, dp: n * (_units(am) + _units(dm))),
    (re.compile(r"(\d+) (?:more )?damage for each Energy attached to this "
                r"Pok", re.I),
     lambda n, am, dm, ap, dp: n * _units(am)),
    (re.compile(r"(\d+) (?:more )?damage for each Energy attached to your "
                r"opponent's Active Pok", re.I),
     lambda n, am, dm, ap, dp: n * _units(dm)),
    (re.compile(r"(\d+) (?:more )?damage for each damage counter on this Pok",
                re.I),
     lambda n, am, dm, ap, dp: n * _counters(am)),
    (re.compile(r"(\d+) (?:more )?damage for each damage counter on your "
                r"opponent's Active Pok", re.I),
     lambda n, am, dm, ap, dp: n * _counters(dm)),
    (re.compile(r"(\d+) (?:more )?damage for each of your Benched Pok", re.I),
     lambda n, am, dm, ap, dp: n * len(ap.get("bench") or [])),
    (re.compile(r"(\d+) (?:more )?damage for each of your opponent's Benched "
                r"Pok", re.I),
     lambda n, am, dm, ap, dp: n * len(dp.get("bench") or [])),
]


def _units(mon):
    return len((mon or {}).get("energies") or [])


def _counters(mon):
    if not mon:
        return 0
    return max(0, ((mon.get("maxHp") or 0) - (mon.get("hp") or 0)) // 10)


# ---------------------------------------------------------------------------
# card tables
# ---------------------------------------------------------------------------

class Tables:
    def __init__(self):
        cards, effects = load_cards()
        self.name = dict(zip(cards.card_id, cards.name))
        self.hp = dict(zip(cards.card_id, cards.hp))
        self.is_pokemon = dict(zip(cards.card_id, cards.is_pokemon))
        prize = {}
        for cid, rule, is_ex in zip(cards.card_id, cards["rule"], cards.is_ex):
            if isinstance(rule, str) and "Mega" in rule:
                prize[cid] = 3
            elif bool(is_ex):
                prize[cid] = 2
            else:
                prize[cid] = 1
        self.prize = prize

        # engine dump: attackId -> damage/cost/text, cardId -> type/weak/resist
        dump = ROOT / "data" / "engine_dump"
        self.attacks = {}
        self.card_engine = {}
        try:
            for a in json.load(open(dump / "attacks.json")):
                self.attacks[a["attackId"]] = a
            for c in json.load(open(dump / "cards.json")):
                self.card_engine[c["cardId"]] = c
        except FileNotFoundError:
            print("WARNING: data/engine_dump/ missing — damage model disabled",
                  file=sys.stderr)

    def prize_value(self, cid):
        return self.prize.get(cid, 1)

    def damage(self, attack_id, attacker_mon, defender_mon,
               attacker_player=None, defender_player=None):
        """Computable damage of an attack against the visible board.

        Returns (damage, unsure) where unsure=True marks coin-gated or
        text-conditional attacks whose printed number is not guaranteed.
        Base + matched riders, then weakness x2 / resistance -30 on the
        defender. Unmatched riders contribute nothing: a lower bound.
        """
        a = self.attacks.get(attack_id)
        if a is None or attacker_mon is None:
            return 0, False
        text = a.get("text") or ""
        dmg = a.get("damage") or 0
        ap = attacker_player or {}
        dp = defender_player or {}
        for pat, fn in RIDER_PATTERNS:
            m = pat.search(text)
            if m:
                dmg += fn(int(m.group(1)), attacker_mon, defender_mon, ap, dp)
                break
        unsure = bool(CONDITIONAL_RE.search(text) or COIN_RE.search(text))
        if not dmg or defender_mon is None:
            return dmg, unsure
        atk_card = self.card_engine.get(attacker_mon.get("id")) or {}
        def_card = self.card_engine.get(defender_mon.get("id")) or {}
        atype = atk_card.get("energyType")
        if atype is not None:
            if def_card.get("weakness") == atype:
                dmg *= 2
            if def_card.get("resistance") == atype:
                dmg = max(0, dmg - 30)
        return dmg, unsure

    def payable(self, attack_id, energies):
        """Can `energies` (list of EnergyType ints on a mon) pay this attack?"""
        a = self.attacks.get(attack_id)
        if a is None:
            return False
        cost = list(a.get("energies") or [])
        pool = list(energies or [])
        for c in sorted(cost, reverse=True):        # typed before colorless
            if c == 0:
                continue
            hit = None
            for i, e in enumerate(pool):
                if e == c or e == RAINBOW or (e == TEAM_ROCKET and c in (5, 7)):
                    hit = i
                    break
            if hit is None:
                return False
            pool.pop(hit)
        n_colorless = sum(1 for c in cost if c == 0)
        return len(pool) >= n_colorless

    def label_archetype(self, deck_ids):
        best, best_hp = "(no Pokémon)", -1.0
        for cid in set(deck_ids):
            if self.is_pokemon.get(cid) and (self.hp.get(cid) or 0) > best_hp:
                best, best_hp = self.name.get(cid, str(cid)), self.hp[cid]
        return best


# ---------------------------------------------------------------------------
# replay parsing
# ---------------------------------------------------------------------------

def board_mons(player):
    out = []
    for zone in ("active", "bench"):
        for m in player.get(zone) or []:
            if isinstance(m, dict):
                out.append(m)
    return out


def parse_events(frames):
    """One pass over the spectator frames -> turn-stamped event ledger.

    Prize draws and the knockout that earned them can land in different
    frames, so discards are matched to the nearest prize event within a
    3-frame window instead of same-frame only.
    """
    turn = 0
    events = []                        # dicts: kind, turn, seat, ...
    ko_moves = []                      # (frame, victim_seat, cardId)
    result = None
    for fi, f in enumerate(frames):
        prize_draws = Counter()
        for lg in f.get("logs") or []:
            t = lg.get("type")
            if t == "TurnStart":
                turn += 1
                events.append({"kind": "turn", "turn": turn,
                               "seat": lg.get("playerIndex", -1)})
            elif t == "Attack":
                events.append({"kind": "attack", "turn": turn,
                               "seat": lg.get("playerIndex"),
                               "cardId": lg.get("cardId"),
                               "serial": lg.get("serial"),
                               "attackId": lg.get("attackId")})
            elif t == "MoveCard":
                pi = lg.get("playerIndex")
                if lg.get("fromArea") == AREA_PRIZE and lg.get("toArea") == AREA_HAND:
                    prize_draws[pi] += 1
                elif (lg.get("fromArea") in (AREA_ACTIVE, AREA_BENCH)
                        and lg.get("toArea") == AREA_DISCARD):
                    ko_moves.append((fi, pi, lg.get("cardId")))
            elif t == "Result":
                result = {"winner": lg.get("result"),
                          "reason": lg.get("reason"), "turn": turn}
        for pi, k in prize_draws.items():
            events.append({"kind": "prizes", "turn": turn, "seat": pi,
                           "count": k, "victim_cards": [], "frame": fi})
    used = set()
    for ev in events:
        if ev["kind"] != "prizes":
            continue
        for j, (fi, pi, cid) in enumerate(ko_moves):
            if (j not in used and pi == 1 - ev["seat"]
                    and abs(fi - ev["frame"]) <= 3):
                ev["victim_cards"].append(cid)
                used.add(j)
    return events, result, turn


def decisions(steps):
    """Yield (t, seat, obs, select, chosen) with the t/t+1 action convention."""
    for t in range(len(steps) - 1):
        for seat, ag in enumerate(steps[t]):
            if ag.get("status") != "ACTIVE":
                continue
            obs = ag.get("observation") or {}
            sel = obs.get("select")
            if not sel or not sel.get("option"):
                continue
            action = steps[t + 1][seat].get("action")
            if not isinstance(action, list):
                action = []
            yield t, seat, obs, sel, [i for i in action if isinstance(i, int)]


# ---------------------------------------------------------------------------
# per-episode analysis
# ---------------------------------------------------------------------------

def analyze(path, tab, team_name):
    d = json.load(open(path))
    info = d.get("info") or {}
    teams = list(info.get("TeamNames") or [])
    if team_name not in teams:
        return None
    me = teams.index(team_name)
    opp = 1 - me
    rewards = d.get("rewards") or [None, None]
    steps = d["steps"]
    frames = steps[0][0].get("visualize") or []

    decks = [steps[1][s].get("action") or [] for s in (0, 1)]
    opp_arch = tab.label_archetype(decks[opp])

    events, result, n_turns = parse_events(frames)
    statuses = d.get("statuses") or []
    if result is not None:
        winner = result["winner"]
        reason = {1: "prizes", 2: "deck-out", 3: "no-active",
                  4: "card-effect"}.get(result["reason"], "unknown")
    else:
        winner = 0 if (rewards[0] or -2) > (rewards[1] or -2) else 1
        reason = "error/forfeit"
    won = winner == me

    # prize race: running (taken_me - taken_opp) after each prize event
    taken = {0: 0, 1: 0}
    race = []                          # (turn, diff after event, taker)
    ko_ledger = []
    for ev in events:
        if ev["kind"] != "prizes":
            continue
        taken[ev["seat"]] += ev["count"]
        race.append((ev["turn"], taken[me] - taken[opp], ev["seat"]))
        victims = [c for c in ev["victim_cards"] if tab.is_pokemon.get(c)]
        top = max(victims, key=lambda c: tab.hp.get(c) or 0, default=None)
        ko_ledger.append({
            "turn": ev["turn"], "taker": "us" if ev["seat"] == me else "them",
            "prizes": ev["count"],
            "victim": tab.name.get(top) if top else None,
            "victim_prize_value": tab.prize_value(top) if top else None,
        })
    final_prizes = {"us": 6 - taken[me], "them": 6 - taken[opp]}

    # --- decision-level detectors -------------------------------------------
    missed_lethal = []                 # (c)
    our_turns_seen = set()
    attackless_turns = set()           # turns where no attack option ever appeared
    turn_had_attack_option = {}
    promotions_bad = []                # (b) promoted multi-prize over 1-prize
    last_main = None                   # our last MAIN decision (for d)

    for t, seat, obs, sel, chosen in decisions(steps):
        if seat != me:
            continue
        cur = obs.get("current") or {}
        players = cur.get("players") or []
        if len(players) != 2:
            continue
        yi = cur.get("yourIndex", seat)
        mine, theirs = players[yi], players[1 - yi]
        turn = cur.get("turn", -1)

        if sel.get("type") == 0 and sel.get("context") == CTX_MAIN:
            our_turns_seen.add(turn)
            atk_opts = [o for o in sel["option"] if o.get("type") == OPT_ATTACK]
            turn_had_attack_option[turn] = (turn_had_attack_option.get(turn, False)
                                            or bool(atk_opts))
            last_main = (t, turn, obs, sel, chosen)

            their_active = (theirs.get("active") or [None])[0]
            my_active = (mine.get("active") or [None])[0]
            need = len(mine.get("prize") or [])
            if (their_active and my_active and atk_opts and need > 0
                    and tab.prize_value(their_active.get("id")) >= need):
                best, unsure = 0, False
                best_id = None
                for o in atk_opts:
                    dmg, u_ = tab.damage(o.get("attackId"), my_active,
                                         their_active, mine, theirs)
                    if dmg > best:
                        best, unsure, best_id = dmg, u_, o.get("attackId")
                if best >= (their_active.get("hp") or 10 ** 9) and not unsure:
                    missed_lethal.append({
                        "turn": turn, "attack_id": best_id, "damage": best,
                        "target": tab.name.get(their_active.get("id")),
                        "target_hp": their_active.get("hp"),
                        "prizes_needed": need,
                        "my_energy": _units(my_active),
                        "their_energy": _units(their_active),
                    })

        # promotion choices: multi-prize promoted while 1-prize listed
        if sel.get("type") == 1 and sel.get("context") == CTX_TO_ACTIVE and chosen:
            opts = sel.get("option") or []
            vals = []
            for o in opts:
                if o.get("type") != OPT_CARD:
                    vals.append(None)
                    continue
                zone = mine.get("bench") if o.get("area") == AREA_BENCH else None
                mon = None
                if zone and 0 <= (o.get("index") or 0) < len(zone):
                    mon = zone[o["index"]]
                vals.append(tab.prize_value(mon["id"]) if mon else None)
            pick = vals[chosen[0]] if chosen[0] < len(vals) else None
            alts = [v for i, v in enumerate(vals)
                    if i != chosen[0] and v is not None]
            if pick and alts and pick >= 2 and min(alts) == 1:
                promotions_bad.append({"turn": turn, "promoted_value": pick,
                                       "alt_value": min(alts)})

    # attackless turns = our turns where no attack option was ever legal
    for turn in our_turns_seen:
        if not turn_had_attack_option.get(turn, False):
            attackless_turns.add(turn)

    # (d) allowed visible lethal: their final attack, seen from our last MAIN
    allowed_visible = None
    if (not won and reason == "prizes" and last_main is not None
            and tab.attacks):
        final_atk = next((ev for ev in reversed(events)
                          if ev["kind"] == "attack" and ev["seat"] == opp), None)
        if final_atk:
            t, turn, obs, sel, chosen = last_main
            cur = obs["current"]
            yi = cur.get("yourIndex", me)
            mine, theirs = cur["players"][yi], cur["players"][1 - yi]
            attacker = next((m for m in board_mons(theirs)
                             if m.get("serial") == final_atk["serial"]), None)
            my_active = (mine.get("active") or [None])[0]
            their_need = len(theirs.get("prize") or [])
            if attacker and my_active:
                dmg, unsure = tab.damage(final_atk["attackId"], attacker,
                                         my_active, theirs, mine)
                payable = tab.payable(final_atk["attackId"],
                                      attacker.get("energies"))
                kills = dmg >= (my_active.get("hp") or 10 ** 9)
                ends = tab.prize_value(my_active.get("id")) >= their_need
                if payable and kills and ends and not unsure:
                    can_retreat = any(o.get("type") == OPT_RETREAT
                                      for o in sel["option"])
                    safe_bench = [
                        m for m in (mine.get("bench") or []) if isinstance(m, dict)
                        and (tab.damage(final_atk["attackId"], attacker,
                                        m, theirs, mine)[0]
                             < (m.get("hp") or 0)
                             or tab.prize_value(m.get("id")) < their_need)
                    ]
                    allowed_visible = {
                        "our_last_turn": turn,
                        "their_attacker": tab.name.get(attacker.get("id")),
                        "attack_damage": dmg,
                        "our_active": tab.name.get(my_active.get("id")),
                        "our_active_hp": my_active.get("hp"),
                        "their_prizes_needed": their_need,
                        "dodge_available": bool(can_retreat and safe_bench),
                    }

    # a lethal seen on a turn the game did not end on is a miss; in a loss,
    # every lethal seen was by definition never converted
    lethal_by_turn = {}
    for m in missed_lethal:
        lethal_by_turn.setdefault(m["turn"], m)
    misses = [m for t_, m in sorted(lethal_by_turn.items())
              if (not won) or t_ < n_turns]

    # --- loss-class flags ----------------------------------------------------
    tags = []
    detail = {}

    # (b) traded badly: held the prize race at some point, finished behind it,
    # with at least one 2-prize concession; or a punished bad promotion
    inverted = False
    if race:
        diffs = [diff for _, diff, _ in race]
        inverted = max(diffs) >= 0 and diffs[-1] < 0
    gave_multi = [k for k in ko_ledger
                  if k["taker"] == "them" and (k["prizes"] or 0) >= 2]
    if not won and ((inverted and gave_multi) or promotions_bad):
        tags.append("b_traded_badly")
        detail["b"] = {"race_inverted": inverted,
                       "multi_prize_concessions": len(gave_multi),
                       "bad_promotions": promotions_bad}

    # (c) missed lethal
    if not won and misses:
        tags.append("c_missed_lethal")
        detail["c"] = misses
    # (d) allowed visible lethal (only actionable if a dodge existed)
    if not won and allowed_visible and allowed_visible["dodge_available"]:
        tags.append("d_allowed_visible_lethal")
        detail["d"] = allowed_visible

    # (e) resource death
    late = sorted(our_turns_seen)[-4:]
    energy_dead = sum(1 for t_ in late if t_ in attackless_turns) >= 2
    if not won and (reason == "deck-out" and winner == opp
                    or (reason == "no-active" and winner == opp)
                    or energy_dead):
        tags.append("e_resource_death")
        detail["e"] = {"reason": reason, "energy_dead_late_turns":
                       sorted(t_ for t_ in late if t_ in attackless_turns)}

    # (a) structural: never ahead on the race, they scored first, nothing else
    if not won and not tags:
        never_ahead = all(diff < 0 or (i == 0 and diff <= 0)
                          for i, (_, diff, _) in enumerate(race)) if race else False
        they_first = race[0][2] == opp if race else False
        if race and never_ahead and they_first:
            tags.append("a_structural_race")
        else:
            tags.append("f_other")
            detail["f"] = "no detector fired and the prize race was not " \
                          "one-sided from the start"

    # Primary = proximate cause. A game that ended with no Pokémon in play or
    # an empty deck died of that, whatever else was also true; games decided
    # on prizes rank decision errors (c, d) over trade economics (b).
    primary = None
    if not won:
        if reason in ("no-active", "deck-out") and "e_resource_death" in tags:
            primary = "e_resource_death"
        else:
            for k in ("c_missed_lethal", "d_allowed_visible_lethal",
                      "b_traded_badly", "e_resource_death",
                      "a_structural_race", "f_other"):
                if k in tags:
                    primary = k
                    break

    return {
        "episode_id": info.get("EpisodeId"),
        "won": won,
        "our_seat": me,
        "opp_team": teams[opp],
        "opp_archetype": opp_arch,
        "our_archetype": tab.label_archetype(decks[me]),
        "first_player": next((ev["seat"] for ev in events
                              if ev["kind"] == "turn"), -1) == me,
        "n_turns": n_turns,
        "termination": reason,
        "final_prizes_left": final_prizes,
        "statuses": statuses,
        "ko_ledger": ko_ledger,
        "race": [{"turn": t_, "diff": diff,
                  "taker": "us" if s == me else "them"} for t_, diff, s in race],
        "lethal_available": missed_lethal,       # every lethal the model saw
        "lethal_missed": misses,                 # lethals not converted that turn
        "allowed_visible": allowed_visible,
        "promotions_bad": promotions_bad,
        "attackless_turns": sorted(attackless_turns),
        "tags": tags if not won else [],
        "primary": primary,
    }


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replays", required=True)
    ap.add_argument("--team", default="Lemmes Yad")
    ap.add_argument("--out", default=str(ROOT / "data" / "analysis"
                                         / "ladder_autopsy_v2.json"))
    args = ap.parse_args()

    tab = Tables()
    records = []
    for path in sorted(Path(args.replays).glob("episode-*-replay.json")):
        try:
            rec = analyze(path, tab, args.team)
        except Exception as e:                    # keep going, report at end
            rec = {"episode_id": path.stem, "error": f"{type(e).__name__}: {e}"}
        if rec:
            records.append(rec)

    losses = [r for r in records if r.get("won") is False]
    wins = [r for r in records if r.get("won") is True]
    dist = Counter(r["primary"] for r in losses if r.get("primary"))
    tagc = Counter(t for r in losses for t in r.get("tags", []))
    by_arch = {}
    for r in records:
        if "error" in r:
            continue
        a = by_arch.setdefault(r["opp_archetype"], {"w": 0, "l": 0})
        a["w" if r["won"] else "l"] += 1
    # control: detector alarm rates inside wins
    control = {
        "wins_with_lethal_seen": sum(
            1 for r in wins if r.get("lethal_available")),
        "wins_with_lethal_missed": sum(
            1 for r in wins if r.get("lethal_missed")),
        "wins_with_race_inversion": sum(
            1 for r in wins if r.get("race") and r["race"][-1]["diff"] < 0),
        "wins_with_bad_promotion": sum(
            1 for r in wins if r.get("promotions_bad")),
    }
    # trade economics: what a knockout was worth to each side
    def econ(rows):
        tk = [k for r in rows for k in r.get("ko_ledger", [])
              if k["taker"] == "us"]
        gv = [k for r in rows for k in r.get("ko_ledger", [])
              if k["taker"] == "them"]
        return {
            "our_kos": len(tk), "our_prizes": sum(k["prizes"] for k in tk),
            "their_kos": len(gv), "their_prizes": sum(k["prizes"] for k in gv),
            "our_prizes_per_ko": round(sum(k["prizes"] for k in tk)
                                       / len(tk), 2) if tk else None,
            "their_prizes_per_ko": round(sum(k["prizes"] for k in gv)
                                         / len(gv), 2) if gv else None,
        }
    denom_signature = sum(
        1 for r in losses
        if sum(1 for k in r["ko_ledger"] if k["taker"] == "us")
        >= sum(1 for k in r["ko_ledger"] if k["taker"] == "them")
        and sum(k["prizes"] for k in r["ko_ledger"] if k["taker"] == "us")
        < sum(k["prizes"] for k in r["ko_ledger"] if k["taker"] == "them"))

    out = {
        "submission": 55311822,
        "team": args.team,
        "n_episodes": len(records),
        "wins": len(wins),
        "losses": len(losses),
        "errors": [r for r in records if "error" in r],
        "primary_distribution": dict(dist),
        "tag_counts": dict(tagc),
        "by_opponent_archetype": by_arch,
        "trade_economics": {"losses": econ(losses), "wins": econ(wins),
                            "losses_with_ko_parity_but_prize_deficit":
                                denom_signature},
        "termination_modes": {
            "losses": dict(Counter(r["termination"] for r in losses)),
            "wins": dict(Counter(r["termination"] for r in wins))},
        "control_group": control,
        "episodes": records,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "episodes"},
                     indent=1))


if __name__ == "__main__":
    main()
