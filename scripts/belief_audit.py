"""The BELIEF-vs-TRUTH differ (D53).

Every ladder gain we have measured came from finding something the agent
believes that is FALSE. Wall blindness (v7) was one instance, found by hand
from one replay: the agent priced an attack at 900 and the engine dealt 0.
This makes that search exhaustive and mechanical.

The instrumented agent is a byte-identical COPY of `agent/main.py` living
outside the repo (default: the session scratchpad). It is never patched. At
every decision the recorder re-derives, from the agent's own functions and
the same observation the agent saw, exactly what the agent believed —
`_damage_against` for the attack it picked, `prize_value` for the defender,
`_attack_profile`'s cost model for what it thinks is payable — and then diffs
that belief against what the ENGINE actually did.

Ground truth is the engine's own log stream (`Observation.logs`), never our
tables: ATTACK (15) declares the attack, HP_CHANGE (16) carries the real
damage, MOVE_CARD (6) ACTIVE->DISCARD is the real knockout, MOVE_CARD
PRIZE->HAND is the real prize. For availability classes the truth is the
menu the engine offered at that instant.

Classes shipped here:

  DAMAGE          expected damage for the chosen attack vs the defender's
                  real HP delta.
  KNOCKOUT        predicted KO vs real KO.
  PRIZE           predicted prizes from that KO vs prizes really taken.
  ATTACK_AVAIL    attacks the agent's cost model counts as payable (the
                  `_threat_at` / `_attack_profile` model: energy COUNT) vs
                  the attacks the engine put on the menu.
  EVOLVE_AVAIL    evolutions the agent's `evolvesFrom`-in-play model counts
                  as legal vs the evolutions the engine offered.
  ABILITY_AVAIL   in-play Pokemon the agent treats as ability-havers vs the
                  abilities the engine offered to use.
  INCOMING        the opponent Active's damage as the agent's own threat
                  pricing reads it vs what their next attack really dealt.

Adding a class is one function: write `p_<name>(rec, ctx)`, append records
with `rec.add(...)` (resolved on the spot) or `rec.pend(...)` (resolved
later against the log stream), and list it in PREDICTORS.

    python scripts/belief_audit.py --games 300 --workers 6 \
        --out experiments_0808/belief_audit.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = Path(os.environ.get(
    "BELIEF_SCRATCH",
    "/private/tmp/claude-501/-Users-austinsemmel-Desktop/"
    "4a2826ba-e7f4-4ad7-9712-4ec201fdff65/scratchpad/belief"))

# option / area / log constants (engine/cg/api.py)
OPT_PLAY, OPT_ATTACH, OPT_EVOLVE, OPT_ABILITY = 7, 8, 9, 10
OPT_DISCARD, OPT_RETREAT, OPT_ATTACK, OPT_END = 11, 12, 13, 14
MAIN_TYPES = {OPT_PLAY, OPT_ATTACH, OPT_EVOLVE, OPT_ABILITY,
              OPT_DISCARD, OPT_RETREAT, OPT_ATTACK, OPT_END}
AREA_DECK, AREA_HAND, AREA_DISCARD = 1, 2, 3
AREA_ACTIVE, AREA_BENCH, AREA_PRIZE = 4, 5, 6
AREA_STADIUM, AREA_LOOKING = 7, 12
LOG_TURN_START, LOG_TURN_END = 2, 3
LOG_MOVE, LOG_MOVE_REV, LOG_ATTACK, LOG_HP = 6, 7, 15, 16
E_COLORLESS, E_RAINBOW, E_TEAM_ROCKET = 0, 10, 11

MAX_STEPS = 20_000
EXAMPLES_PER_KEY = 4


# --- module loading ---------------------------------------------------------
def _load_module(path: Path, name: str):
    """Exec a main.py-style agent with its own directory as cwd.

    The bundle files an agent reads (`trajectory_curves.json`, `postures.json`
    ...) are found relative to the file, so the copy must sit in a directory
    holding the same bundle the repo agent resolves — otherwise the copy is a
    different agent than the one we are auditing.
    """
    cwd = os.getcwd()
    os.chdir(path.parent)
    try:
        spec = importlib.util.spec_from_file_location(name, str(path))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        os.chdir(cwd)


# --- small observation helpers (dict form; the live grader form) ------------
def _mons(player) -> list:
    out = []
    for mon in (player.get("active") or []):
        if mon:
            out.append((AREA_ACTIVE, mon))
    for i, mon in enumerate(player.get("bench") or []):
        if mon:
            out.append((AREA_BENCH, mon))
    return [m for _, m in out]


def _mon_at(player, area: int, index: int):
    """The body an option points at, or None when it points off the board.

    An ABILITY option is not always a Pokemon's: the engine also offers the
    Stadium's ability through the same option type, with area STADIUM. Left
    unguarded, those resolve into the bench list and invent abilities on
    whatever body happens to sit at that index.
    """
    if area not in (AREA_ACTIVE, AREA_BENCH):
        return None
    zone = player.get("active") if area == AREA_ACTIVE else player.get("bench")
    zone = zone or []
    if index is None or index >= len(zone):
        return None
    return zone[index]


def _zone_card(cur, player, area: int, index: int):
    """Any card an option's (area, index) names, in play or not."""
    zone = {AREA_HAND: player.get("hand"),
            AREA_DISCARD: player.get("discard"),
            AREA_ACTIVE: player.get("active"),
            AREA_BENCH: player.get("bench"),
            AREA_STADIUM: (cur or {}).get("stadium")}.get(area) or []
    if index is None or index >= len(zone):
        return None
    return zone[index]


def _pays_typed(cost: list, have: list) -> bool:
    """Type-aware cost check — the model the agent does NOT have.

    Only ever reported as a diagnostic beside a discrepancy, to say whether a
    type-aware cost model would have agreed with the engine.
    """
    pool = Counter(int(e) for e in have or [])
    wild = pool.pop(E_RAINBOW, 0)
    need = Counter(int(e) for e in cost or [])
    colorless = need.pop(E_COLORLESS, 0)
    for t, c in need.items():
        take = min(pool.get(t, 0), c)
        pool[t] -= take
        c -= take
        if c > 0 and t in (5, 7) and pool.get(E_TEAM_ROCKET, 0) > 0:
            take = min(pool[E_TEAM_ROCKET], c)
            pool[E_TEAM_ROCKET] -= take
            c -= take
        if c > 0:
            take = min(wild, c)
            wild -= take
            c -= take
        if c > 0:
            return False
    return sum(v for v in pool.values() if v > 0) + wild >= colorless


# --- the recorder -----------------------------------------------------------
class Recorder:
    """One game's beliefs, and the engine stream they get diffed against.

    The stream is OUR SEAT'S view and nobody else's. `Observation.logs` holds
    everything since *that viewer* last saw the board, so the same attack is
    delivered twice — once to each seat, the opponent's copy with hidden
    moves masked into MOVE_CARD_REVERSE. Concatenating both views double
    counts every public event; taking only the observations addressed to our
    seat yields each event exactly once, and `--validate` proves it by
    reconciling logged prizes against the prize piles on the final board.
    """

    def __init__(self, mod, me: int, cell: str):
        self.mod = mod
        self.me = me
        self.cell = cell
        self.stream: list = []
        self.pending: list = []
        self.rows: list = []
        self.turn_seen: dict = {}     # turn -> serials the engine offered
        self.prize_marks: list = []   # (stream index, our prize count)
        self.errors: Counter = Counter()
        self.final_prizes = [6, 6]
        self.final_serials = [set(), set()]

    # -- record sinks
    def add(self, cls: str, key: str, pred, actual, disc: bool, extra=None):
        extra = extra or {}
        self.rows.append({"cls": cls, "key": key, "pred": pred,
                          "actual": actual, "disc": bool(disc),
                          "extra": extra})
        # A discrepancy carrying a `why` also lands in a companion class
        # whose keys ARE the reasons, so the report counts causes exactly
        # rather than inferring them from a capped example list.
        if disc and extra.get("why"):
            self.rows.append({"cls": cls + "_WHY", "key": extra["why"],
                              "pred": 1, "actual": 0, "disc": True,
                              "extra": {"cell": extra.get("cell")}})

    def pend(self, item: dict):
        item["mark"] = len(self.stream)
        self.pending.append(item)

    def logs(self, obs):
        cur = obs.get("current") or {}
        players = cur.get("players") or []
        if len(players) == 2:
            # Pile sizes and bodies in play are public on either seat's
            # observation, so the final board is readable whoever it was
            # addressed to — which is how the last knockout of a game, the
            # one whose logs we never receive, still gets scored.
            self.final_prizes = [len(p.get("prize") or []) for p in players]
            self.final_serials = [{m.get("serial") for m in _mons(p)}
                                  for p in players]
        if cur.get("yourIndex") == self.me:
            self.stream.extend(obs.get("logs") or [])
            if len(players) == 2:
                self.prize_marks.append(
                    (len(self.stream), len(players[self.me].get("prize")
                                           or [])))

    # -- resolution against the engine stream
    def resolve(self):
        for p in self.pending:
            fn = _RESOLVERS[p["kind"]]
            try:
                fn(self, p)
            except Exception:
                continue


def _window_attack(stream, mark: int, player: int, attack_id=None):
    """Index of the ATTACK log this prediction is about, or None.

    The first ATTACK log at or after the mark decides. If somebody else's
    attack lands first the prediction never resolved (the action was
    superseded), and it is dropped rather than scored against the wrong
    event.
    """
    for i in range(mark, len(stream)):
        lg = stream[i]
        if lg.get("type") != LOG_ATTACK:
            continue
        if lg.get("playerIndex") != player:
            return None
        if attack_id is not None and lg.get("attackId") != attack_id:
            return None
        return i
    return None


def _damage_in(stream, ai: int, serials=None, player=None) -> int:
    """HP lost inside the attack's own resolution — up to the turn end, so
    between-turn poison and burn never land in an attack's column."""
    dmg = 0
    for j in range(ai + 1, len(stream)):
        lg = stream[j]
        t = lg.get("type")
        if t in (LOG_TURN_END, LOG_ATTACK):
            break
        if t != LOG_HP:
            continue
        if serials is not None and lg.get("serial") not in serials:
            continue
        if player is not None and lg.get("playerIndex") != player:
            continue
        v = lg.get("value") or 0
        if v < 0:
            dmg += -v
    return dmg


def _ko_and_prizes(rec: "Recorder", ai: int, serial: int, me: int):
    """Real knockout of that body, and prizes really taken, after the attack.

    Knockout comes off the move log; prizes come off the PILE, because the
    cards leave it a step later than the knockout and the game can end in
    between — our seat never receives the log batch that carries the winning
    prize. Reading the pile at the last observation before our next attack
    (the final board when there is none) is exact either way: our pile only
    shrinks when we take prizes, and only one opponent turn intervenes.
    """
    stream = rec.stream
    ko, starts = False, 0
    nxt = len(stream)
    for j in range(ai + 1, len(stream)):
        lg = stream[j]
        t = lg.get("type")
        if t == LOG_ATTACK and lg.get("playerIndex") == me:
            nxt = j
            break
        if t == LOG_TURN_START:
            starts += 1
            if starts >= 3:
                nxt = j
                break
        if t == LOG_MOVE and lg.get("serial") == serial \
                and lg.get("fromArea") == AREA_ACTIVE \
                and lg.get("toArea") == AREA_DISCARD:
            ko = True
    if nxt >= len(stream):
        # No later attack of ours: the final board is this attack's endpoint,
        # and it is the only endpoint that sees the prizes of a game-winning
        # knockout, whose log batch our seat never receives.
        prize_after = rec.final_prizes[me]
        if not ko:
            ko = serial not in rec.final_serials[1 - me]
    else:
        after = [c for mark, c in rec.prize_marks if ai < mark <= nxt]
        prize_after = after[-1] if after else rec.final_prizes[me]
    return ko, max(0, p_before_minus(rec, ai) - prize_after)


def p_before_minus(rec: "Recorder", ai: int) -> int:
    """Our prize count as of the observation the attack was declared in."""
    before = [c for mark, c in rec.prize_marks if mark <= ai]
    return before[-1] if before else 6


def _resolve_our_attack(rec: Recorder, p: dict):
    ai = _window_attack(rec.stream, p["mark"], rec.me, p["aid"])
    if ai is None:
        return
    actual = _damage_in(rec.stream, ai, serials={p["def_serial"]})
    ko, prizes = _ko_and_prizes(rec, ai, p["def_serial"], rec.me)
    key = "atk%s|att%s|def%s" % (p["aid"], p["att_cid"], p["def_cid"])
    gap = p["pred_dmg"] - actual
    if not gap:
        why = None
    elif p["pred_dmg"] and not actual:
        why = "zeroed|stadium%s" % p["stadium"]
    elif p["unsure"]:
        why = "coin_or_conditional_text|%s" % ("over" if gap > 0 else "under")
    else:
        why = "%s_by_%d" % ("over" if gap > 0 else "under", abs(gap))
    extra = {"hp": p["hp"], "turn": p["turn"], "cell": rec.cell,
             "stadium": p["stadium"], "my_e": p["my_e"],
             "their_e": p["their_e"], "zeroed": bool(p["pred_dmg"] and
                                                     not actual),
             "coin_text": p["unsure"], "why": why}
    rec.add("DAMAGE", key, p["pred_dmg"], actual,
            p["pred_dmg"] != actual, extra)
    rec.add("KNOCKOUT", key, p["pred_ko"], ko, p["pred_ko"] != ko,
            dict(extra, why=("phantom_ko|%s" % ("walled" if not actual
                                                else "short")
                             if p["pred_ko"] else "missed_ko")))
    rec.add("PRIZE", key, p["pred_prizes"], prizes,
            p["pred_prizes"] != prizes,
            dict(extra, ko=ko,
                 why=("no_ko_no_prizes" if p["pred_ko"] and not ko else
                      "clamped_by_prizes_left"
                      if prizes < p["pred_prizes"] and ko else
                      "extra_prize" if prizes > p["pred_prizes"] else
                      "unclaimed")))
    if ko:
        # The clean prize question: the engine knocked this body out, so what
        # is it worth? Free of prizes that came from anywhere else.
        pv = rec.mod.prize_value(int(p["def_cid"] or 0))
        rec.add("PRIZE_ON_KO", "def%s" % p["def_cid"], pv, prizes,
                pv != prizes,
                dict(extra,
                     why=("engine_paid_more_than_prize_value"
                          if prizes > pv else "engine_paid_fewer")))


def _resolve_incoming(rec: Recorder, p: dict):
    them = 1 - rec.me
    ai = _window_attack(rec.stream, p["mark"], them)
    if ai is None:
        return
    actual = _damage_in(rec.stream, ai, player=rec.me)
    aid = rec.stream[ai].get("attackId")
    key = "def%s|threat_from%s" % (p["my_cid"], p["their_cid"])
    # Only "they hit harder than we believed possible" is a model error;
    # under-delivery is their choice of a weaker attack, not our mistake.
    rec.add("INCOMING", key, p["pred"], actual, actual > p["pred"],
            {"turn": p["turn"], "attackId": aid, "cell": rec.cell})


_RESOLVERS = {"our_attack": _resolve_our_attack, "incoming": _resolve_incoming}


# --- predictors -------------------------------------------------------------
# Each takes (rec, ctx) where ctx carries the observation the agent saw, the
# option list, and the indices it returned. Cheap to add: write one, list it.
def p_attack(rec: Recorder, ctx: dict):
    """DAMAGE / KNOCKOUT / PRIZE — the attack the agent actually chose.

    The belief is re-derived through the agent's own `_damage_against` with
    the same scaling context `_choose_attack` builds, so it is the number the
    agent priced, not a reconstruction of it.
    """
    mod, obs = rec.mod, ctx["obs"]
    picked = ctx["picked"]
    if picked is None or picked.get("type") != OPT_ATTACK:
        return
    cur = obs.get("current") or {}
    players = cur.get("players") or []
    if len(players) < 2:
        return
    mine, theirs = players[rec.me], players[1 - rec.me]
    act = (mine.get("active") or [None])[0]
    tgt = (theirs.get("active") or [None])[0]
    if tgt is None:
        return
    sctx, slot_e, slot_dmgc = None, 0.0, 0.0
    if getattr(mod, "SCALED_DAMAGE_ENABLED", False):
        try:
            # An agent whose scaling context also reads the STATE (the
            # Stadium is not on either player) takes a third argument. Ask
            # for it, and fall back so a two-argument agent is audited
            # exactly as before.
            try:
                sctx = mod._scale_ctx(mine, theirs, cur)
            except TypeError:
                sctx = mod._scale_ctx(mine, theirs)
            if act is not None:
                slot_e = float(len(act.get("energies") or []))
                slot_dmgc = mod._dmg_counters(act)
        except Exception:
            sctx = None
    my_cid = act.get("id") if act else None
    aid = picked.get("attackId")
    pred = mod._damage_against(aid, tgt, sctx, slot_e, slot_dmgc,
                               attacker_cid=my_cid)
    hp = tgt.get("hp")
    pred_ko = hp is not None and pred >= hp
    stadium = (cur.get("stadium") or [None])
    rec.pend({"kind": "our_attack", "aid": aid, "att_cid": my_cid,
              "def_cid": tgt.get("id"), "def_serial": tgt.get("serial"),
              "pred_dmg": int(pred), "pred_ko": bool(pred_ko), "hp": hp,
              "pred_prizes": (mod.prize_value(int(tgt.get("id") or 0))
                              if pred_ko else 0),
              "turn": cur.get("turn"),
              # A wall can sit off the defender entirely — the E11 parse
              # only ever reads the defender's own skills, so the Stadium
              # and the attacker's Energy counts ride along as evidence.
              "stadium": (stadium[0] or {}).get("id") if stadium else None,
              "my_e": len(act.get("energies") or []) if act else 0,
              "their_e": len(tgt.get("energies") or []),
              "unsure": bool(mod._attack_unsure(int(aid or 0)))})


def p_attack_avail(rec: Recorder, ctx: dict):
    """ATTACK_AVAIL — the cost model `_threat_at` reaches its verdicts with.

    The agent counts Energy units and compares against `len(attack.energies)`
    (`_attack_profile`). Truth is whether the engine put that attack on the
    menu. The diagnostic says whether a type-aware cost check would have
    matched the engine, which is what tells us the fix.
    """
    if not ctx["is_main"]:
        return
    mod, obs = rec.mod, ctx["obs"]
    cur = obs.get("current") or {}
    players = cur.get("players") or []
    if len(players) < 2:
        return
    mine = players[rec.me]
    act = (mine.get("active") or [None])[0]
    if act is None:
        return
    cards, attacks = mod._tables()
    data = cards.get(act.get("id"))
    if data is None:
        return
    have = [int(e) for e in (act.get("energies") or [])]
    offered = {o.get("attackId") for o in ctx["options"]
               if o.get("type") == OPT_ATTACK}
    cond = any(mine.get(k) for k in ("asleep", "paralyzed", "confused"))
    for aid in (getattr(data, "attacks", None) or []):
        atk = attacks.get(aid)
        if atk is None:
            continue
        cost = [int(e) for e in (getattr(atk, "energies", None) or [])]
        believed = len(cost) <= len(have)
        truth = aid in offered
        typed = _pays_typed(cost, have)
        rec.add("ATTACK_AVAIL", "cid%s|atk%s" % (act.get("id"), aid),
                believed, truth, believed != truth,
                {"cost": len(cost), "have": len(have),
                 "typed_ok": typed == truth, "typed_says": typed,
                 "condition": cond, "turn": cur.get("turn"),
                 "cell": rec.cell,
                 "why": "%s|%s%s" % ("believed_not_offered" if believed
                                     else "offered_not_believed",
                                     "typed_model_fixes_it" if typed == truth
                                     else "typed_model_also_wrong",
                                     "|special_condition" if cond else "")})


def p_evolve_avail(rec: Recorder, ctx: dict):
    """EVOLVE_AVAIL — evolution legality in both directions.

    Belief: the `evolvesFrom` name match the agent uses everywhere it reasons
    about evolutions (`_card_score`'s EVOLUTION_BONUS, `_evo_map`'s edges).
    Truth: the OPT_EVOLVE options the engine actually offered. Believed but
    unoffered catches the just-played and first-turn restrictions; offered
    but unbelieved catches Rare Candy's stage skip.
    """
    if not ctx["is_main"]:
        return
    mod, obs = rec.mod, ctx["obs"]
    cur = obs.get("current") or {}
    players = cur.get("players") or []
    if len(players) < 2:
        return
    mine = players[rec.me]
    hand = mine.get("hand") or []
    board = _mons(mine)
    if not hand or not board:
        return
    cards, _ = mod._tables()

    offered = set()
    for o in ctx["options"]:
        if o.get("type") != OPT_EVOLVE:
            continue
        idx = o.get("index")
        src = hand[idx] if idx is not None and idx < len(hand) else None
        tgt = _mon_at(mine, o.get("inPlayArea"), o.get("inPlayIndex"))
        if src is None or tgt is None:
            continue
        offered.add((src.get("id"), tgt.get("serial")))

    believed = {}
    for card in hand:
        cdata = cards.get(card.get("id"))
        ef = getattr(cdata, "evolvesFrom", None) if cdata else None
        if not ef:
            continue
        for mon in board:
            mdata = cards.get(mon.get("id"))
            if mdata is not None and getattr(mdata, "name", None) == ef:
                believed[(card.get("id"), mon.get("serial"))] = (card, mon)

    for pair in set(believed) | offered:
        card, mon = believed.get(pair, (None, None))
        if mon is None:
            for m in board:
                if m.get("serial") == pair[1]:
                    mon = m
        cdata = cards.get(pair[0])
        mdata = cards.get(mon.get("id")) if mon else None
        b, t = pair in believed, pair in offered
        fresh = bool(mon.get("appearThisTurn")) if mon else None
        s2 = bool(getattr(cdata, "stage2", False)) if cdata else None
        basic = bool(getattr(mdata, "basic", False)) if mdata else None
        turn = cur.get("turn")
        if b and not t:
            why = ("believed_not_offered|target_played_this_turn" if fresh
                   else "believed_not_offered|first_turn" if (turn or 9) <= 2
                   else "believed_not_offered|unexplained")
        elif t and not b:
            why = ("offered_not_believed|stage2_onto_basic" if s2 and basic
                   else "offered_not_believed|unexplained")
        else:
            why = None
        rec.add("EVOLVE_AVAIL", "hand%s|onto%s" % (pair[0],
                                                   mon.get("id") if mon
                                                   else "?"),
                b, t, b != t,
                {"turn": turn, "target_new": fresh, "stage2_hand": s2,
                 "basic_target": basic, "cell": rec.cell, "why": why})


def p_ability_avail(rec: Recorder, ctx: dict):
    """ABILITY_AVAIL — who the agent treats as an ability-haver vs who the
    engine will actually let use one right now.

    The agent has no ability model at all: `_wall_prevents` reads "has an
    Ability" straight off `CardData.skills`, and the rules policy takes
    whatever OPT_ABILITY the menu holds. This measures the gap, and counts
    abilities the engine offered that the agent declined.
    """
    if not ctx["is_main"]:
        return
    mod, obs = rec.mod, ctx["obs"]
    cur = obs.get("current") or {}
    players = cur.get("players") or []
    if len(players) < 2:
        return
    mine = players[rec.me]
    cards, _ = mod._tables()
    offered, offered_idx = set(), set()
    for i, o in enumerate(ctx["options"]):
        if o.get("type") != OPT_ABILITY:
            continue
        mon = _mon_at(mine, o.get("area"), o.get("index"))
        if mon is not None:
            offered.add(mon.get("serial"))
            offered_idx.add(i)
            continue
        # An ability the agent's model has no slot for at all: the engine
        # hands out the Stadium's ability through the same option type.
        src = _zone_card(cur, mine, o.get("area"), o.get("index"))
        rec.add("ABILITY_OFFBOARD", "area%s|card%s"
                % (o.get("area"), (src or {}).get("id")), False, True, True,
                {"turn": cur.get("turn"), "cell": rec.cell,
                 "why": "engine_offers_ability_from_area_%s" % o.get("area")})
    turn = cur.get("turn")
    seen = rec.turn_seen.setdefault(turn, set())
    used = set(seen)
    seen |= offered
    for mon in _mons(mine):
        data = cards.get(mon.get("id"))
        skills = (getattr(data, "skills", None) or []) if data else []
        b = bool(skills)
        t = mon.get("serial") in offered
        if not b and not t:
            continue
        # An ability the engine offered earlier THIS turn and no longer
        # offers is a once-per-turn ability already spent, not a belief the
        # engine contradicts; the flag separates the two readings.
        spent = mon.get("serial") in used
        rec.add("ABILITY_AVAIL", "cid%s" % mon.get("id"), b, t, b != t,
                {"turn": turn, "n_skills": len(skills),
                 "spent_this_turn": spent, "cell": rec.cell,
                 "why": (("believed_not_offered|already_used_this_turn"
                          if spent
                          else "believed_not_offered|never_offered_this_turn")
                         if b else "offered_not_believed")})
    # Declining an ability only counts at the decision that ENDS the turn:
    # earlier in the turn an unused ability is simply one the agent has not
    # reached yet, and it takes them in menu-priority order.
    picked = ctx["picked"]
    if offered_idx and picked is not None \
            and picked.get("type") in (OPT_ATTACK, OPT_END):
        for i in offered_idx:
            mon = _mon_at(mine, ctx["options"][i].get("area"),
                          ctx["options"][i].get("index"))
            rec.add("ABILITY_DECLINED", "cid%s" % (mon.get("id") if mon
                                                   else "?"),
                    True, False, True,
                    {"turn": cur.get("turn"), "cell": rec.cell})


def p_incoming(rec: Recorder, ctx: dict):
    """INCOMING — our price on the opponent Active's next hit.

    Uses the agent's own `_threat_at` on a one-slot list holding their Active,
    with its scaling context and the wall our Active puts up: exactly the k=1
    incoming read the evaluator uses. Budget is grown by one Energy, so a
    flagged row is one where the engine hit harder than the agent believed
    was payable at all.
    """
    picked = ctx["picked"]
    if picked is None or picked.get("type") not in (OPT_ATTACK, OPT_END):
        return
    mod, obs = rec.mod, ctx["obs"]
    cur = obs.get("current") or {}
    players = cur.get("players") or []
    if len(players) < 2:
        return
    mine, theirs = players[rec.me], players[1 - rec.me]
    their_act = (theirs.get("active") or [None])[0]
    my_act = (mine.get("active") or [None])[0]
    if their_act is None or my_act is None:
        return
    # Their side's slots and growth exactly as the evaluator builds them,
    # narrowed to the Active — the only body that can attack next turn.
    try:
        _, (st, gt, _) = mod._traj_projection(cur, mine, theirs)
        growth = gt
    except Exception:
        st, growth = None, (lambda k: float(k))
    slots = mod._SlotList()
    slots.append((float(len(their_act.get("energies") or [])),
                  int(their_act.get("id") or 0),
                  mod._dmg_counters(their_act), False))
    for attr in ("sctx", "wall_def"):
        got = getattr(st, attr, None) if st is not None else None
        if got is not None:
            setattr(slots, attr, got)
    pred = mod._threat_at(slots, growth, 1)
    rec.pend({"kind": "incoming", "pred": float(pred),
              "my_cid": my_act.get("id"), "their_cid": their_act.get("id"),
              "turn": cur.get("turn")})


PREDICTORS = [p_attack, p_attack_avail, p_evolve_avail, p_ability_avail,
              p_incoming]


# --- the instrumented seat --------------------------------------------------
def instrumented(mod, rec_box: dict):
    """The agent, unmodified, with a recorder wrapped around its answers.

    The wrapper runs AFTER `mod.agent` returns, reads only functions with no
    side effects on the agent's state, and is charged no time against the
    agent's own bank — so the games this plays are the games v7 plays.
    """
    def fn(obs_dict):
        choice = mod.agent(obs_dict)
        rec = rec_box.get("rec")
        try:
            if rec is not None and obs_dict.get("select") is not None:
                options = (obs_dict["select"].get("option") or [])
                picked = (options[choice[0]]
                          if len(choice) == 1 and choice
                          and choice[0] < len(options) else None)
                ctx = {"obs": obs_dict, "options": options,
                       "choice": list(choice), "picked": picked,
                       "is_main": any(o.get("type") in MAIN_TYPES
                                      for o in options)}
                for p in PREDICTORS:
                    try:
                        p(rec, ctx)
                    except Exception as exc:
                        # A predictor that throws silently reports a clean
                        # class, which is the one failure mode this whole
                        # instrument cannot afford. Counted and printed.
                        rec.errors["%s:%s" % (p.__name__,
                                              type(exc).__name__)] += 1
        except Exception:
            pass
        return choice
    return fn


def play_audited(agent_a, agent_b, deck_a, deck_b, seed: int, me: int,
                 rec: Recorder):
    """`ptcg.arena.play_game`, with the engine's log stream tapped.

    The loop is a copy rather than a call because the audit needs the
    observation the engine returns after every action — that stream is the
    ground truth the beliefs are diffed against.
    """
    from cg.game import battle_finish, battle_select, battle_start
    random.seed(seed)
    agents = (agent_a, agent_b) if me == 0 else (agent_b, agent_a)
    obs, _start = battle_start(list(deck_a if me == 0 else deck_b),
                               list(deck_b if me == 0 else deck_a))
    if obs is None:
        return None
    rec.logs(obs)
    steps = 0
    try:
        while steps < MAX_STEPS:
            cur = obs.get("current") or {}
            sel = obs.get("select")
            if sel is None or not sel.get("option"):
                break
            who = cur.get("yourIndex", 0)
            try:
                choice = agents[who](obs)
            except Exception:
                return 1 - who
            if not isinstance(choice, list) or not choice:
                return 1 - who
            try:
                obs = battle_select(choice)
            except (ValueError, IndexError):
                return 1 - who
            rec.logs(obs)
            steps += 1
        cur = obs.get("current") or {}
        result = cur.get("result")
        return result if result in (0, 1) else None
    except Exception:
        return None
    finally:
        battle_finish()


# --- worker plumbing --------------------------------------------------------
_STATE: dict = {}


def _init(root: str, agent_dir: str, cells: list):
    sys.path.insert(0, root)
    sys.path.insert(0, str(Path(root) / "engine"))
    import ptcg.creation  # noqa: F401  — engine bootstrap
    _STATE["mod"] = _load_module(Path(agent_dir) / "main.py", "belief_agent")
    _STATE["agent_dir"] = agent_dir
    _STATE["cells"] = {c["name"]: c for c in cells}
    _STATE["opp"] = {}


def _deck(path: str) -> list:
    p = Path(path)
    if p.suffix == ".json":
        return [int(c) for c in json.loads(p.read_text())]
    text = p.read_text()
    return [int(x) for x in text.replace(",", "\n").split() if x.strip()]


def _opponent(cell: dict):
    key = cell["name"]
    got = _STATE["opp"].get(key)
    if got is not None:
        return got
    if cell["opp_kind"] == "external":
        from ptcg.creation.pilots import ExternalPilot
        pilot = ExternalPilot(cell["opp"])
        pilot.bind_deck(_deck(cell["opp_deck"]))
        got = pilot
    else:                                  # our own agent, second instance
        mod = _load_module(Path(cell["opp"]) / "main.py",
                           "belief_opp_%s" % abs(hash(cell["opp"])))
        got = mod.agent
    _STATE["opp"][key] = got
    return got


def _open_episode(agent) -> None:
    try:
        agent({"select": None})
    except Exception:
        pass


def _merge(agg: dict, rows: list):
    for r in rows:
        cls = agg.setdefault(r["cls"], {"n": 0, "disc": 0, "keys": {}})
        cls["n"] += 1
        cls["disc"] += int(r["disc"])
        k = cls["keys"].setdefault(r["key"], {"n": 0, "disc": 0, "ex": [],
                                              "pred_sum": 0.0,
                                              "act_sum": 0.0})
        k["n"] += 1
        k["disc"] += int(r["disc"])
        for fld, val in (("pred_sum", r["pred"]), ("act_sum", r["actual"])):
            try:
                k[fld] += float(val)
            except (TypeError, ValueError):
                pass
        if r["disc"] and len(k["ex"]) < EXAMPLES_PER_KEY:
            k["ex"].append({"pred": r["pred"], "actual": r["actual"],
                            **r["extra"]})


def _play_chunk(job: tuple) -> dict:
    cell_name, seeds = job
    cell = _STATE["cells"][cell_name]
    mod = _STATE["mod"]
    opp = _opponent(cell)
    deck_a = _deck(cell["deck"])
    deck_b = _deck(cell["opp_deck"])
    box: dict = {}
    fn = instrumented(mod, box)
    agg: dict = {}
    perr: Counter = Counter()
    wins, games, errs = 0, 0, 0
    for g in seeds:
        me = g % 2
        rec = Recorder(mod, me, cell_name)
        box["rec"] = rec
        _open_episode(mod.agent)
        _open_episode(opp)
        try:
            winner = play_audited(fn, opp, deck_a, deck_b, g, me, rec)
        except Exception:
            errs += 1
            continue
        rec.resolve()
        _merge(agg, rec.rows)
        perr.update(rec.errors)
        games += 1
        if winner == me:
            wins += 1
    return {"agg": agg, "wins": wins, "games": games, "errors": errs,
            "cell": cell_name, "predictor_errors": dict(perr)}


def _combine(dst: dict, src: dict):
    for cls, blob in src.items():
        d = dst.setdefault(cls, {"n": 0, "disc": 0, "keys": {}})
        d["n"] += blob["n"]
        d["disc"] += blob["disc"]
        for key, k in blob["keys"].items():
            t = d["keys"].setdefault(key, {"n": 0, "disc": 0, "ex": [],
                                           "pred_sum": 0.0, "act_sum": 0.0})
            t["n"] += k["n"]
            t["disc"] += k["disc"]
            t["pred_sum"] += k["pred_sum"]
            t["act_sum"] += k["act_sum"]
            for e in k["ex"]:
                if len(t["ex"]) < EXAMPLES_PER_KEY * 3:
                    t["ex"].append(e)


# --- cells ------------------------------------------------------------------
def default_cells(agent_dir: str) -> list:
    """The audit corpus: our list into the specialists, the mirror, and our
    agent piloting evolution decks — the only way the evolution classes see
    an evolving hand, since our own list is all Basics."""
    ex = ROOT / "external"
    cells = []
    for name, mod_path, deck in (
            ("grimmsnarl", ex / "grimmsnarl_agent.py",
             ex / "grimmsnarl_deck.json"),
            ("lucario", ex / "lucario_agent.py", ex / "lucario_deck.json"),
            ("archaludon", ex / "archaludon_agent.py",
             ex / "archaludon_deck.json"),
            ("alakazam", ex / "codex_alakazam_agent.py",
             ex / "codex_alakazam_deck.json"),
            ("kanga", ex / "kanga_agent.py", ex / "kanga_deck.json"),
            ("garchomp", ex / "garchomp_agent.py", ex / "garchomp_deck.json")):
        cells.append({"name": "vs_" + name, "opp_kind": "external",
                      "opp": str(mod_path), "opp_deck": str(deck),
                      "deck": str(Path(agent_dir) / "deck.csv")})
    cells.append({"name": "mirror", "opp_kind": "self", "opp": agent_dir,
                  "opp_deck": str(Path(agent_dir) / "deck.csv"),
                  "deck": str(Path(agent_dir) / "deck.csv")})
    return cells


def evolution_cells(agent_dir_base: str, decks: list) -> list:
    """Our agent piloting somebody else's evolving list.

    Each needs its own copy directory because the agent reads its 60 cards
    from `deck.csv` beside itself and builds its evolution pool from them.
    """
    out = []
    for name, deck, opp_mod, opp_deck in decks:
        out.append({"name": "as_" + name, "opp_kind": "external",
                    "opp": opp_mod, "opp_deck": opp_deck,
                    "deck": str(Path(agent_dir_base + "_" + name)
                               / "deck.csv"),
                    "agent_dir": agent_dir_base + "_" + name})
    return out


def _card_names(mod=None):
    try:
        sys.path.insert(0, str(ROOT))
        sys.path.insert(0, str(ROOT / "engine"))
        import ptcg.creation  # noqa: F401
        from cg.api import all_attack, all_card_data
        return ({c.cardId: c.name for c in all_card_data()},
                {a.attackId: a.name for a in all_attack()})
    except Exception:
        return {}, {}


def _label(key: str, cards: dict, attacks: dict) -> str:
    """Turn `atk120|att96|def117` into names a human can act on."""
    out = []
    for part in key.split("|"):
        for pre, table in (("atk", attacks), ("att", cards), ("def", cards),
                           ("cid", cards), ("hand", cards), ("onto", cards),
                           ("threat_from", cards)):
            if part.startswith(pre) and part[len(pre):].isdigit():
                num = int(part[len(pre):])
                out.append("%s %d %s" % (pre, num, table.get(num, "?")))
                break
        else:
            out.append(part)
    return " / ".join(out)


def report(paths: list, md_path: str, top: int = 12) -> None:
    """Per-class discrepancy table and the offenders behind it."""
    agg: dict = {}
    games, cells, seconds = 0, {}, 0.0
    for p in paths:
        blob = json.loads(Path(p).read_text())
        _combine(agg, blob["classes"])
        games += blob.get("games", 0)
        seconds += blob.get("seconds", 0.0)
        for name, cs in (blob.get("cells") or {}).items():
            d = cells.setdefault(name, {"wins": 0, "games": 0, "errors": 0})
            for k in d:
                d[k] += cs.get(k, 0)
    cards, attacks = _card_names()
    lines = ["# BELIEF vs TRUTH — discrepancy report", "",
             "%d games, %d cells, %.0f s of engine time." %
             (games, len(cells), seconds), "",
             "| cell | games | win rate | forfeits |",
             "|---|---|---|---|"]
    for name, cs in sorted(cells.items()):
        lines.append("| %s | %d | %.3f | %d |"
                     % (name, cs["games"],
                        cs["wins"] / cs["games"] if cs["games"] else 0.0,
                        cs["errors"]))
    lines += ["", "## Discrepancy by class", "",
              "| class | n | discrepancies | rate | per game |",
              "|---|---|---|---|---|"]
    for cls, blob in sorted(agg.items(), key=lambda kv: -kv[1]["disc"]):
        lines.append("| %s | %d | %d | %.4f | %.2f |"
                     % (cls, blob["n"], blob["disc"],
                        blob["disc"] / blob["n"] if blob["n"] else 0.0,
                        blob["disc"] / games if games else 0.0))
    for cls, blob in sorted(agg.items(), key=lambda kv: -kv[1]["disc"]):
        if not blob["disc"]:
            continue
        lines += ["", "### %s — top offenders" % cls, "",
                  "| key | n | disc | rate | mean believed | mean actual |",
                  "|---|---|---|---|---|---|"]
        rows = sorted(blob["keys"].items(), key=lambda kv: -kv[1]["disc"])
        for key, k in rows[:top]:
            if not k["disc"]:
                continue
            lines.append("| %s | %d | %d | %.3f | %.1f | %.1f |"
                         % (_label(key, cards, attacks), k["n"], k["disc"],
                            k["disc"] / k["n"], k["pred_sum"] / k["n"],
                            k["act_sum"] / k["n"]))
        ex = []
        for key, k in rows[:6]:
            for e in k["ex"][:2]:
                ex.append("- `%s` believed %s, engine %s %s"
                          % (_label(key, cards, attacks), e.get("pred"),
                             e.get("actual"),
                             {a: b for a, b in e.items()
                              if a not in ("pred", "actual")}))
        if ex:
            lines += ["", "Examples:", ""] + ex
    Path(md_path).write_text("\n".join(lines) + "\n")
    print("wrote", md_path)


def validate(agent_dir: str, games: int, seed0: int) -> int:
    """Prove the stream is read correctly before any finding is believed.

    Prizes are conserved: every prize card that left a pile is a prize the
    logs must show moving PRIZE->HAND. Counting the log stream against the
    prize piles on the final board is a check the double-counted stream
    fails loudly (it reports up to twice the prizes actually taken).
    """
    _init(str(ROOT), agent_dir, default_cells(agent_dir))
    mod = _STATE["mod"]
    deck = _deck(str(Path(agent_dir) / "deck.csv"))
    opp = _load_module(Path(agent_dir) / "main.py", "belief_val_opp")
    box: dict = {}
    fn = instrumented(mod, box)
    bad = 0
    for g in range(seed0, seed0 + games):
        me = g % 2
        rec = Recorder(mod, me, "validate")
        box["rec"] = rec
        _open_episode(mod.agent)
        _open_episode(opp.agent)
        play_audited(fn, opp.agent, deck, deck, g, me, rec)
        rec.resolve()
        attributed = sum(r["actual"] for r in rec.rows if r["cls"] == "PRIZE")
        kos = sum(1 for r in rec.rows if r["cls"] == "KNOCKOUT"
                  and r["actual"])
        taken = 6 - (rec.final_prizes[me] or 0)
        ok = attributed == taken
        bad += 0 if ok else 1
        print("game %d seat %d  prizes attributed %d  taken %d  real KOs %d"
              "  %s" % (g, me, attributed, taken, kos,
                        "ok" if ok else "MISMATCH"))
    print("%d/%d games reconcile" % (games - bad, games))
    return bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent-dir", default=str(SCRATCH / "agent_v7"))
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed0", type=int, default=770000)
    ap.add_argument("--cells", default="")
    ap.add_argument("--out", default=str(ROOT / "experiments_0808"
                                         / "belief_audit.json"))
    ap.add_argument("--validate", type=int, default=0,
                    help="reconcile the log stream on N games and exit")
    ap.add_argument("--report", nargs="*", default=None,
                    help="render the markdown report from raw JSONs and exit")
    ap.add_argument("--md", default=str(ROOT / "experiments_0808"
                                        / "BELIEF_AUDIT.md"))
    args = ap.parse_args()

    if args.report is not None:
        report(args.report, args.md)
        return

    agent_dir = str(Path(args.agent_dir).resolve())
    if Path(agent_dir).resolve() == (ROOT / "agent").resolve():
        raise SystemExit("refusing to instrument agent/main.py; use a copy")

    if args.validate:
        raise SystemExit(1 if validate(agent_dir, args.validate,
                                       args.seed0) else 0)

    cells = default_cells(agent_dir)
    if args.cells:
        want = set(args.cells.split(","))
        cells = [c for c in cells if c["name"] in want]
    per = max(1, args.games // len(cells))
    jobs = []
    for c in cells:
        seeds = [args.seed0 + i for i in range(per)]
        for w in range(args.workers):
            chunk = seeds[w::args.workers]
            if chunk:
                jobs.append((c["name"], chunk))

    t0 = time.time()
    agg: dict = {}
    cellstat: dict = {}
    perrors: dict = {}
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=get_context("spawn"),
                             initializer=_init,
                             initargs=(str(ROOT), agent_dir,
                                       cells)) as pool:
        for res in pool.map(_play_chunk, jobs):
            _combine(agg, res["agg"])
            cs = cellstat.setdefault(res["cell"], {"wins": 0, "games": 0,
                                                   "errors": 0})
            cs["wins"] += res["wins"]
            cs["games"] += res["games"]
            cs["errors"] += res["errors"]
            for k, v in (res.get("predictor_errors") or {}).items():
                perrors[k] = perrors.get(k, 0) + v

    out = {"agent_dir": agent_dir, "games": sum(c["games"] for c
                                                in cellstat.values()),
           "seconds": round(time.time() - t0, 1), "cells": cellstat,
           "predictor_errors": perrors, "classes": agg}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    for cls, blob in sorted(agg.items()):
        rate = blob["disc"] / blob["n"] if blob["n"] else 0.0
        print("%-14s n=%-7d disc=%-7d rate=%.4f"
              % (cls, blob["n"], blob["disc"], rate))
    if perrors:
        print("PREDICTOR ERRORS (a class reporting clean may be a lie):",
              perrors)
    print("games", out["games"], "in", out["seconds"], "s ->", args.out)


if __name__ == "__main__":
    main()
