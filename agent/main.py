"""Heuristic Pokémon TCG agent (v4).

Kaggle calls `agent(obs_dict)` and expects a list of option indices. The first
call has `select == None` and must return the 60-card deck.

Two things drive this policy:

1. **Within a turn, take every free action before the one that ends it** —
   abilities, evolutions, plays and attachments all happen before attacking.
2. **Prizes are the win condition, not damage.** A knockout is worth more than
   raw damage, and what you give up when your own Pokémon is knocked out
   (1 prize, 2 for an ex, 3 for a Mega ex) matters as much as what you take.

Everything the engine tells us is used through `cg.api`; card metadata comes
from `all_card_data()`.

NOTE: the grader `exec()`s this file rather than importing it, so `__file__`
does not exist at run time. Never reference it unguarded — that raises
NameError and kills the episode.
"""

from __future__ import annotations

import math
import os
import random
import sys
import time

# --- option / area / context constants (see cg/api.py) ----------------------
OPT_NUMBER, OPT_YES, OPT_NO, OPT_CARD = 0, 1, 2, 3
OPT_TOOL_CARD, OPT_ENERGY_CARD, OPT_ENERGY = 4, 5, 6
OPT_PLAY, OPT_ATTACH, OPT_EVOLVE, OPT_ABILITY = 7, 8, 9, 10
OPT_DISCARD, OPT_RETREAT, OPT_ATTACK, OPT_END = 11, 12, 13, 14

AREA_DECK, AREA_HAND, AREA_DISCARD = 1, 2, 3
AREA_ACTIVE, AREA_BENCH, AREA_PRIZE = 4, 5, 6
AREA_STADIUM, AREA_LOOKING = 7, 12

CTX_SETUP_ACTIVE, CTX_SETUP_BENCH = 1, 2
CTX_TO_ACTIVE, CTX_TO_BENCH, CTX_TO_FIELD, CTX_TO_HAND = 4, 5, 6, 7
CTX_DISCARD = 8
CTX_MAIN = 0
CTX_IS_FIRST, CTX_MULLIGAN, CTX_ACTIVATE = 41, 42, 43

# Attacking ends the turn, so it sorts below everything free.
MAIN_PRIORITY = {
    OPT_ABILITY: 7,
    OPT_EVOLVE: 6,
    OPT_PLAY: 5,
    OPT_ATTACH: 4,
    OPT_ATTACK: 3,
    OPT_RETREAT: 2,
    OPT_END: 1,
}

try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = ""
_KAGGLE_DIR = "/kaggle_simulations/agent"

# ---------------------------------------------------------------------------
# Import the engine API **at module level**, exactly as the official sample
# does — and make sure the agent's own directory is importable first.
#
# This module is exec()'d by the grader with the bundle directory as the working
# directory, but `cg` is NOT on sys.path by the time a function body runs. Doing
# these imports lazily inside _search_main therefore raised
# ModuleNotFoundError("No module named 'cg'") on every single call, and the
# `except Exception: return None` around it turned that into a silent fallback
# to the rule policy.
#
# Measured under kaggle_environments' own `cabt` environment: 77 agent calls,
# 40 search attempts, 0 successful imports. The search we built, measured and
# shipped had never once run on the ladder.
# ---------------------------------------------------------------------------
for _p in (_HERE, _KAGGLE_DIR, os.getcwd()):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from cg.api import (all_attack, all_card_data, search_begin, search_end,
                        search_step, to_observation_class)
    CG_AVAILABLE = True
except Exception:                       # keep playing on rules if it is absent
    CG_AVAILABLE = False

_CARDS: dict | None = None
_ATTACKS: dict | None = None


def _tables() -> tuple[dict, dict]:
    """Card and attack metadata straight from the engine."""
    global _CARDS, _ATTACKS
    if _CARDS is None:
        try:
            if not CG_AVAILABLE:
                raise RuntimeError("cg unavailable")
            _CARDS = {c.cardId: c for c in all_card_data()}
            _ATTACKS = {a.attackId: a for a in all_attack()}
        except Exception:
            _CARDS, _ATTACKS = {}, {}
    return _CARDS, _ATTACKS


def read_deck_csv() -> list[int]:
    paths = ["deck.csv", os.path.join(_KAGGLE_DIR, "deck.csv")]
    if _HERE:
        paths.insert(1, os.path.join(_HERE, "deck.csv"))
    for path in paths:
        try:
            with open(path, "r") as fh:
                text = fh.read()
        except OSError:
            continue
        deck = [int(x) for x in text.replace(",", "\n").split() if x.strip()]
        if len(deck) == 60:
            return deck
        raise ValueError(f"deck.csv must hold 60 card ids, got {len(deck)}")
    raise FileNotFoundError("deck.csv not found next to the agent")


# --- reading the board ------------------------------------------------------
def _g(obj, key, default=None):
    """Read `key` off either observation form.

    The grader hands the live policy a dict; `search_step` hands the rollout
    policy the same observation as dataclasses (cg/api.py), with identical
    field names. One accessor over both is what lets the two policies be the
    same code rather than two copies that drift — the D25 invariant, held
    structurally.
    """
    if obj is None:
        return default
    if isinstance(obj, dict):
        v = obj.get(key, default)
    else:
        v = getattr(obj, key, default)
    return default if v is None else v


def _zone(obs, area: int, player_index: int) -> list:
    """The list an option's (area, index) points into."""
    cur = _g(obs, "current")
    players = _g(cur, "players", []) or []
    if area == AREA_DECK:
        return _g(_g(obs, "select"), "deck", []) or []
    if area == AREA_STADIUM:
        return _g(cur, "stadium", []) or []
    if area == AREA_LOOKING:
        return _g(cur, "looking", []) or []
    if player_index is None or player_index >= len(players):
        return []
    ps = players[player_index]
    return {
        AREA_HAND: _g(ps, "hand"),
        AREA_DISCARD: _g(ps, "discard"),
        AREA_ACTIVE: _g(ps, "active"),
        AREA_BENCH: _g(ps, "bench"),
        AREA_PRIZE: _g(ps, "prize"),
    }.get(area) or []


def _option_card(obs, opt):
    """Resolve a CARD option to the actual card, or None if hidden.

    CARD options carry (area, index, playerIndex) — never a cardId — so
    scoring one means looking it up on the board first.
    """
    zone = _zone(obs, _g(opt, "area"), _g(opt, "playerIndex"))
    idx = _g(opt, "index")
    if idx is None or idx >= len(zone):
        return None
    return zone[idx]


def prize_value(card_id: int) -> int:
    """Prizes the opponent takes when this Pokémon is knocked out."""
    cards, _ = _tables()
    data = cards.get(card_id)
    if data is None:
        return 1
    if getattr(data, "megaEx", False):
        return 3
    if getattr(data, "ex", False):
        return 2
    return 1


def _best_damage(card_id: int) -> int:
    cards, attacks = _tables()
    data = cards.get(card_id)
    if data is None:
        return 0
    best = 0
    for aid in getattr(data, "attacks", None) or []:
        a = attacks.get(aid)
        if a is not None:
            best = max(best, getattr(a, "damage", 0) or 0)
    return best


# Card-selection primitives. All three read card metadata only, so they hold
# for any deck: nothing here names a card.
DUPLICATE_PENALTY = 50.0     # per copy already in hand
EVOLUTION_BONUS = 70.0       # evolves something we already have in play
BENCH_EMERGENCY = 10000.0    # one Pokemon left: a Basic outranks everything


def _board_context(obs) -> dict:
    """Our own side, read once per ranking: hand copies, Pokemon in play, and
    the names an evolution card could be looking for."""
    cur = _g(obs, "current")
    me = _g(cur, "yourIndex", 0)
    players = _g(cur, "players", []) or []
    if me >= len(players):
        return {"hand": {}, "in_play": 0, "names": set()}
    mine = players[me]
    hand: dict = {}
    for c in _g(mine, "hand", []) or []:
        cid = _g(c, "id")
        if cid:
            hand[cid] = hand.get(cid, 0) + 1
    cards, _ = _tables()
    names, in_play = set(), 0
    for zone in ("active", "bench"):
        for mon in _g(mine, zone, []) or []:
            if mon is None:
                continue
            in_play += 1
            data = cards.get(_g(mon, "id"))
            if data is not None:
                names.add(getattr(data, "name", None))
    return {"hand": hand, "in_play": in_play, "names": names}


def _card_score(card, board: dict | None = None,
                in_hand: bool = False) -> float:
    """How much we want this card in play.

    Offence and bulk are good; handing the opponent extra prizes is not, and
    that penalty is what keeps a Mega ex from being promoted on reflex.
    """
    cid = _g(card, "id")
    if cid is None:
        return -1.0
    hp = _g(card, "hp")
    if hp is None:
        cards, _ = _tables()
        data = cards.get(cid)
        hp = getattr(data, "hp", 0) or 0
    score = hp + 2.0 * _best_damage(cid) - 220.0 * (prize_value(cid) - 1)
    if board is None:
        return score
    try:
        copies = board["hand"].get(cid, 0) - (1 if in_hand else 0)
        score -= DUPLICATE_PENALTY * max(copies, 0)
        cards, _ = _tables()
        data = cards.get(cid)
        if data is not None:
            evolves_from = getattr(data, "evolvesFrom", None)
            if evolves_from and evolves_from in board["names"]:
                score += EVOLUTION_BONUS
            if board["in_play"] <= 1 and getattr(data, "basic", False):
                score += BENCH_EMERGENCY
    except Exception:
        pass
    return score


# --- choosing an attack -----------------------------------------------------
def _opponent_active(obs):
    cur = _g(obs, "current")
    me = _g(cur, "yourIndex", 0)
    players = _g(cur, "players", []) or []
    if len(players) < 2:
        return None
    act = _g(players[1 - me], "active", []) or []
    return act[0] if act and act[0] is not None else None


# Measured off the engine (the game logic is a native dylib, so the number
# comes from probing search_step, not from source): a Fighting attack into
# the Abra line (resistance 6) lands 30 under its printed damage, floored at
# zero — 130 printed dealt 100, 30 printed dealt 0 — while an attack whose
# text waives Weakness/Resistance dealt exactly its printed number on the
# same matchup. Weakness x2 confirmed on the same probes (30 printed KO'd a
# 60-HP weak-to-Fighting body). A card never prints the same type in both
# fields, so the two modifiers cannot co-occur and their ordering is moot.
RESISTANCE_DAMAGE = 30


def _damage_against(attack_id: int, target, sctx=None,
                    slot_e: float = 0.0, slot_dmgc: float = 0.0) -> int:
    """Printed damage, doubled if the defender is weak to this attack's type
    and reduced by RESISTANCE_DAMAGE if it resists it.

    With a scaling context (CABT_SCALED_DAMAGE, built by `_choose_attack`),
    an attack the scaler KB names is priced at its observed damage instead —
    our own scaling attacks stop reading as their printed number too. The
    resistance subtraction rides the same flag: it is part of the one
    v3 bundle the gate measures, and the pre-fix baseline ignored it.
    Both modifiers key on the attack's cost energies, the convention the
    weakness check has always used; the engine may key on the attacker
    card's own type instead, but the two coincide on mono-typed attackers,
    which is what the pool plays.
    """
    cards, attacks = _tables()
    a = attacks.get(attack_id)
    if a is None:
        return 0
    dmg = getattr(a, "damage", 0) or 0
    if sctx is not None:
        try:
            sc = _scalers().get(int(attack_id or 0))
            if sc is not None:
                dmg = _scaled_damage(sc, sctx, slot_e, slot_dmgc)
        except Exception:
            TELEMETRY_SCALED["ctx_errors"] += 1
    if not dmg or target is None:
        return dmg
    data = cards.get(_g(target, "id"))
    weakness = getattr(data, "weakness", None) if data else None
    resistance = getattr(data, "resistance", None) if data else None
    energies = [e for e in (getattr(a, "energies", None) or []) if e]
    if weakness is not None and any(e == weakness for e in energies):
        dmg = dmg * 2
    if (SCALED_DAMAGE_ENABLED and resistance is not None
            and any(e == resistance for e in energies)):
        dmg = max(dmg - RESISTANCE_DAMAGE, 0)
    return dmg


def _choose_attack(options, obs) -> int | None:
    """Cheapest knockout if one exists, otherwise the biggest hit."""
    target = _opponent_active(obs)
    target_hp = _g(target, "hp")

    sctx, slot_e, slot_dmgc = None, 0.0, 0.0
    if SCALED_DAMAGE_ENABLED:
        try:
            cur = _g(obs, "current")
            me = _g(cur, "yourIndex", 0)
            players = _g(cur, "players", []) or []
            if len(players) >= 2:
                sctx = _scale_ctx(players[me], players[1 - me])
                act = _g(players[me], "active", []) or []
                if act and act[0] is not None:
                    slot_e = float(len(_g(act[0], "energies", []) or []))
                    slot_dmgc = _dmg_counters(act[0])
        except Exception:
            TELEMETRY_SCALED["ctx_errors"] += 1
            sctx = None

    ko_i, ko_d = None, None
    big_i, big_d = None, -1
    for i, o in enumerate(options):
        if _g(o, "type") != OPT_ATTACK:
            continue
        d = _damage_against(_g(o, "attackId"), target, sctx, slot_e,
                            slot_dmgc)
        if d > big_d:
            big_i, big_d = i, d
        if target_hp is not None and d >= target_hp:
            if ko_d is None or d < ko_d:
                ko_i, ko_d = i, d
    return ko_i if ko_i is not None else big_i


def _choose_attach(options, obs) -> int | None:
    """Energy goes on whatever is going to attack — the Active Pokémon."""
    fallback = None
    for i, o in enumerate(options):
        if _g(o, "type") != OPT_ATTACH:
            continue
        if fallback is None:
            fallback = i
        if _g(o, "inPlayArea") == AREA_ACTIVE:
            return i
    return fallback


def _choose_main(options, obs) -> int:
    # Behind, the fitted comeback posture shifts the menu order (see
    # POSTURE_DELTA); ahead or even it is the flat priority table.
    behind = _behind(obs)
    best_i, best_rank = 0, -1.0
    for i, o in enumerate(options):
        kind = _g(o, "type")
        rank = float(MAIN_PRIORITY.get(kind, 0))
        if behind:
            rank += POSTURE_DELTA.get(kind, 0.0)
        if rank > best_rank:
            best_i, best_rank = i, rank

    kind = _g(options[best_i], "type")
    if kind == OPT_ATTACK:
        alt = _choose_attack(options, obs)
        if alt is not None:
            return alt
    elif kind == OPT_ATTACH:
        alt = _choose_attach(options, obs)
        if alt is not None:
            return alt
    return best_i


def _rank_card_options(options, obs) -> list[int]:
    """Option indices, best card first. Unresolvable cards sort last.

    When a matchup posture is live this is also where its named gust order
    lands. `_card_score` reads a card the way we read our own — bulk and
    damage, minus the prizes it hands over — and applied to the opponent's
    board that ranking picks their biggest attacker, which is the opposite of
    what the deny analysis says to gust. The posture's preference order is the
    correction, and it applies only to cards the option says belong to them,
    so nothing about how we choose among our own cards can move.
    """
    try:
        board = _board_context(obs)
    except Exception:
        board = None
    ranks = _POSTURE_ON["gust_rank"]
    # A discard prompt reads this list backwards (see `_policy`), so the
    # preference would invert there. Nothing in our list makes the opponent
    # discard a Pokemon we choose, but the guard costs a comparison.
    if _g(_g(obs, "select"), "context") == CTX_DISCARD:
        ranks = {}
    me = _g(_g(obs, "current"), "yourIndex", 0)
    scored = []
    targeted = 0
    for i, o in enumerate(options):
        card = _option_card(obs, o)
        if card is not None:
            r = None
            if ranks and _g(o, "playerIndex") not in (None, me):
                r = ranks.get(_g(card, "id"))
            if r is None:
                v = _card_score(card, board, _g(o, "area") == AREA_HAND)
            else:
                v = POSTURE_GUST_BASE - POSTURE_GUST_STEP * r
                targeted += 1
        else:
            v = -1.0
        scored.append((v, -i, i))
    if targeted:
        TELEMETRY_POSTURE["gust_selects"] += 1
        TELEMETRY_POSTURE["gust_targeted"] += targeted
    scored.sort(reverse=True)
    return [i for _, _, i in scored]


def _policy(obs) -> list[int]:
    """The rule policy, over a dict or a dataclass observation alike.

    This is the whole of what the agent plays when the search does not
    override it, and it is also what every rollout inside the search plays.
    Before this was one function it was two, and the rollout half handled only
    the MAIN menu: every other prompt inside a rollout — which card to fetch,
    which to discard, which Pokémon to bench, how many to draw — scored rank 0
    for every option and took index 0. Rollouts therefore valued our candidate
    moves as continued by a policy nobody plays, and the bias fell hardest on
    setup lines, whose payoff arrives through exactly those prompts.
    """
    sel = _g(obs, "select")
    options = _g(sel, "option", []) or []
    if not options:
        return [0]

    ctx = _g(sel, "context")
    min_count = _g(sel, "minCount", 1) or 0
    max_count = _g(sel, "maxCount", 1) or 1
    n = len(options)
    types = {_g(o, "type") for o in options}

    # Yes/no questions.
    if types <= {OPT_YES, OPT_NO}:
        want_yes = ctx != CTX_MULLIGAN   # never redraw a legal opening hand
        for i, o in enumerate(options):
            if (_g(o, "type") == OPT_YES) == want_yes:
                return [i]
        return [0]

    # "How many?" — always take the most.
    if types == {OPT_NUMBER}:
        best_i, best_n = 0, -1
        for i, o in enumerate(options):
            v = _g(o, "number", 0) or 0
            if v > best_n:
                best_i, best_n = i, v
        return [best_i]

    # The main menu.
    if any(_g(o, "type") in MAIN_PRIORITY for o in options):
        return [_choose_main(options, obs)]

    # Card choices: put the most valuable card where we want it, and the least
    # valuable card where we don't.
    if OPT_CARD in types:
        order = _rank_card_options(options, obs)
        if ctx == CTX_DISCARD:
            order = order[::-1]
        k = max(min_count, 1) if max_count >= 1 else min_count
        k = min(k, max_count, n)
        if ctx in (CTX_SETUP_BENCH, CTX_TO_BENCH):
            k = min(max_count, n)      # a full bench is always better
        return sorted(order[:max(k, 1)])

    # Anything else (energy picks, tools): smallest legal set.
    k = min(max(min_count, 1), max_count, n)
    return list(range(max(k, 1)))


# =========================================================================
# Forward search
#
# The engine can roll the game forward from the current position, but only if
# we supply the hidden information: the opponent's deck, hand and prizes. That
# would be hopeless guesswork except the mined replays show the whole metagame
# is ~120 decklists, and the 40 shipped in deck_priors.json cover 94% of it.
# So we hold a posterior over which known list the opponent is playing, fill in
# the rest, and actually simulate our candidate turns instead of guessing.
#
# Which cards of that list sit in the deck rather than the hand or the prizes
# is still unknown, so each candidate is played out over a small budget of
# candidate lists and shuffles and averaged under the posterior weights. Each
# play-out runs our turn to its end and scores the position we hand over.
# =========================================================================

_PRIORS: list[tuple[dict, int]] | None = None
_MY_DECK: list[int] | None = None
SEARCH_ROLLOUT_STEPS = 24     # enough to finish a turn; caps the cost
# How many candidate opponent decklists to average each option over, and when.
#
# Measured against ground truth over 300 replays (scripts/validate_posterior.py),
# the posterior's top pick is right:
#
#   turn 1: 0.63   turn 3: 0.84   turn 5: 0.88   turn 10: 0.94
#
# while the true deck is somewhere in the top 3:
#
#   turn 1: 0.86   turn 3: 0.95   turn 5: 0.97   turn 10: 0.97
#
# So there is a real early-game window where a point estimate is wrong about a
# third of the time and averaging over three candidates recovers most of it —
# and a long later phase where the top pick is already right and the extra two
# determinizations are pure cost.
#
# The posterior is well calibrated (it claims 0.6-0.7 and is right 0.70; claims
# 0.9+ and is right 0.98), so its own confidence is a trustworthy gate.
POSTERIOR_TOP_K = 3
CONFIDENCE_GATE = 0.80
SEARCH_ENABLED = True
# Total determinizations per decision, split across two axes: which decklist,
# and how its unseen cards are dealt. Deck identity turned out to be the easy
# half — 80% of misidentifications are between sister lists that play the same —
# so when the posterior is confident the whole budget goes to shuffles instead.
# Held at 3 because fair sampling at N=3 measured neutral (51.3%) against the
# pre-port agent over 200 games, and the cost is linear in it.
SEARCH_N_DET = 3
# Opponent replies minimized over at ply 2.
#
# History, because this number was 0 for two measured reasons and neither of
# them was the branching. Playing their turn out cost 9 points of win rate
# against the same agent at 0 (40.7% over 200 games) and 14 at one reply
# (36.2%) — a single reply with no minimum taken being the *worse* of the two
# is what said the defect was the model of how they play rather than the depth.
# A separate one-reply test on a single point-estimate determinization landed
# in the same place (0.467). Both of those rolled the opponent's turn out under
# MAIN_PRIORITY, our own hand-set ordering applied to somebody else.
#
# `data/opponent_policy.json` is that missing half — 2,722,350 counted
# main-menu decisions, band- and archetype-conditional (see `_opp_order`) — and
# the retest both rejections asked for has now been run. It fails too. Against
# 5d85b53 on mirror decks with seats swapped, seed block 92000:
#
#   one reply, table-chosen, no minimum   0.4855 over 791 decided
#                                         (Marnie 0.460/491, Garchomp 0.527/300)
#   three replies, minimum taken          0.4758 over 786
#                                         (Marnie 0.471/486, Garchomp 0.483/300)
#   three replies, at TRAJ_K = 3          0.4918 over 791
#                                         (Marnie 0.485/491, Garchomp 0.503/300)
#
# The table is not the part that is broken and this measurement is what says
# so. It covered every position it was asked about — 865,899 cell lookups over
# 40 mirror games, all of them at the finest key (band, archetype, turn bucket,
# within-turn ordinal), zero backoffs and zero misses — and it is anything but
# inert: the search's override rate goes from 0.5038 with the opponent seat
# frozen to 0.6041 at one table reply and 0.5993 at three, over ~3,200 searches
# each. So a better model of their turn moves a tenth of our decisions and
# moves them the wrong way.
#
# That points at the evaluation and not the reply. Handing the position over
# and scoring what comes back is only worth doing if the score at ply 2 is
# better than the score at ply 1, and this evaluator is the same 1-ply linear
# margin either way — the wall D27 hit from the determinization side. Three
# rejections now (0.407, 0.467, 0.476), each having removed the defect the last
# one blamed, and the remaining suspect is the leaf.
#
# What stays: the table, its builder, and every function below it, so the day
# the leaf improves this is one constant.
try:
    SEARCH_OPP_BRANCH = int(os.environ.get("CABT_OPP_BRANCH") or 0)
except Exception:
    SEARCH_OPP_BRANCH = 0
# WEIGHTS["search_margin"] is the override hysteresis. A half-prize (500)
# costs 14 points of win rate here — 35.1% against the unported agent versus
# 48.7% at margin 0 — because this rules policy is the weaker of the two
# (search-on beats rules-only 58.7%). Tunable, defaulted off.

# --- episode time bank ------------------------------------------------------
# The grader gives each agent a 600-second bank per episode (cabt.json
# observation.remainingOverageTime) and sets no per-move deadline (actTimeout
# is 0), so the old flat 0.80s cap was an arbitrary number rather than a
# metered one. Replace it: each decision may spend what is left of the bank
# divided by the decisions still expected, clamped at both ends, and nothing
# at all once the bank is nearly gone.
#
# Measured caveat, so nobody reads more into this than is there: at
# SEARCH_N_DET = 3 a search finishes in 16 ms on average and 176 ms at worst,
# so neither the old 0.80s cap nor the ~1.9s this yields is ever the binding
# constraint. Zero of 7,936 searches over 200 rank-0 Marnie mirror games
# reached either deadline, and a full kaggle_environments episode spends 0.8
# of its 600 seconds. What holds spending down is the fixed determinization
# count, not the clock, and the two agents therefore played to 0.505 over
# those 200 games.
#
# Scaling the work to the meter was the obvious follow-up and it was measured
# and rejected. An adaptive count (n_det = budget divided by an online estimate
# of what one determinization costs, clamped to [3, 64]) raised the mean count
# from 3 to 43 and the mean search from 16 ms to 127 ms, and then won 147 of
# 300 rank-0 Marnie mirror games against this agent at an identical 48-second
# bank: 0.490, 95% CI [0.433, 0.547]. Two other allocations of the same
# enlarged count measured 0.513 and 0.497, and pooled over all three the score
# is 450-450 in 900 games. The override rate says why: at 14x the
# determinizations, how often search overruled the rule policy held flat at
# 53.9% against this agent's 54.2% over the same 8,000 searches, so at three
# determinizations the search has already converged on its answer and the
# sample was never what limited it. Decision invariance over
# K=5 requeries fell too, 45.6% of main decisions against 51.5% for this
# agent. What limits the search is the 1-ply evaluation it converges to, so
# N_DET waits on the opponent policy model SEARCH_OPP_BRANCH is waiting on.
# The meter stays: it is correct, it costs nothing, and it is what the ladder
# would need the day the work is worth scaling.
try:
    BANK_SECONDS = float(os.environ.get("CABT_LOCAL_BANK") or 600.0)
except Exception:
    BANK_SECONDS = 600.0
BANK_SAFETY = 0.80        # fraction of the bank we ever plan against
BANK_MIN_BUDGET = 0.20    # seconds; a search shorter than this buys nothing
BANK_MAX_BUDGET = 5.00    # seconds; ceiling on any one decision
BANK_RESERVE = 5.00       # below this the bank is closed and the rules decide
# How many decisions are still to come. Measured, not assumed: episodes run 85
# agent calls on average and 151 at the longest, so the estimate starts at 150
# and decays with the calls already made. The old 260 was a guess with no
# episode behind it, and it made every early decision plan against a divisor
# 1.7x too large. Over-estimating is the safe direction — it underspends — but
# only up to the point where the whole bank goes unused, which is where 260
# had put us.
BANK_DECISIONS_MAX = 150
BANK_DECISIONS_MIN = 30
BANK_DIVISOR_FLOOR = 20   # never divide by less than this

_bank_remaining = BANK_SECONDS
_bank_decisions = 0


def _reset_bank() -> None:
    """Every episode opens with the deck-selection call, where select is None."""
    global _bank_remaining, _bank_decisions
    _bank_remaining = BANK_SECONDS
    _bank_decisions = 0


def _bank_left(obs=None) -> float:
    """Seconds of thinking time the episode still holds.

    The grader publishes its own accounting on every observation as
    `remainingOverageTime`, and kaggle_environments decrements it by each
    call's full duration because this environment sets no per-move deadline.
    That number is the one the episode is killed on, so it is the one to read;
    our own subtraction is the fallback for harnesses that do not send it.
    Taking the smaller of the two can only underspend.
    """
    try:
        rot = obs.get("remainingOverageTime") if isinstance(obs, dict) else None
        if rot is not None:
            return min(float(rot), _bank_remaining)
    except Exception:
        pass
    return _bank_remaining


def _decision_budget(obs=None) -> float:
    """Seconds this decision may search for; 0.0 means the rules policy only."""
    try:
        left = _bank_left(obs)
        if left < BANK_RESERVE:
            return 0.0
        est = max(BANK_DECISIONS_MIN, BANK_DECISIONS_MAX - _bank_decisions)
        budget = left * BANK_SAFETY / max(est, BANK_DIVISOR_FLOOR)
        return min(max(budget, BANK_MIN_BUDGET), BANK_MAX_BUDGET)
    except Exception:
        return BANK_MIN_BUDGET


def _load_priors() -> list[tuple[dict, int]]:
    global _PRIORS
    if _PRIORS is None:
        _PRIORS = []
        paths = ["deck_priors.json", os.path.join(_KAGGLE_DIR, "deck_priors.json")]
        if _HERE:
            paths.insert(1, os.path.join(_HERE, "deck_priors.json"))
        import json
        for path in paths:
            try:
                with open(path, "r") as fh:
                    raw = json.load(fh)
            except (OSError, ValueError):
                continue
            for e in raw.get("decks", []):
                counts = {int(k): v for k, v in e.get("c", {}).items()}
                _PRIORS.append((counts, int(e.get("p", 1))))
            break
    return _PRIORS


def _counter_from_zone(cards, out: dict) -> None:
    for card in cards or []:
        if not isinstance(card, dict):
            continue
        cid = card.get("id")
        if cid:
            out[cid] = out.get(cid, 0) + 1
        for sub in ("energyCards", "tools", "preEvolution"):
            for c in card.get(sub) or []:
                if isinstance(c, dict) and c.get("id"):
                    out[c["id"]] = out.get(c["id"], 0) + 1


def _visible(player: dict, include_hand: bool) -> dict:
    seen: dict = {}
    for zone in ("active", "bench", "discard"):
        _counter_from_zone(player.get(zone), seen)
    if include_hand:
        _counter_from_zone(player.get("hand"), seen)
    return seen


def _log_choose(n: int, k: int) -> float:
    """log C(n, k), or -inf when the draw is impossible."""
    if k < 0 or n < 0 or k > n:
        return float("-inf")
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1))


def _deck_posterior(obs: dict, top_k: int = 3) -> list[tuple[dict, float]]:
    """Posterior over the opponent's decklist, as (counts, weight) pairs.

    Prior is the empirical play frequency from the mined replays. The likelihood
    of having revealed a multiset S of cards from a 60-card list D is
    multivariate hypergeometric:

        P(S | D)  proportional to  product over cards c of  C(D_c, S_c)

    The C(60, |S|) denominator is identical for every candidate, so it cancels.

    This generalises the hard consistency filter we used before: a deck that
    cannot contain what we have seen gets C(D_c, S_c) = 0 and drops out on its
    own, but now a deck running four copies of a card we have seen once is also
    correctly preferred over one running a single copy, instead of the two being
    treated as equally plausible.
    """
    cur = obs.get("current") or {}
    me = cur.get("yourIndex", 0)
    players = cur.get("players") or []
    if len(players) < 2:
        return []
    seen = _visible(players[1 - me], include_hand=False)

    scored: list[tuple[float, dict]] = []
    for counts, plays in _load_priors():
        loglik = 0.0
        for cid, k in seen.items():
            loglik += _log_choose(counts.get(cid, 0), k)
            if loglik == float("-inf"):
                break
        if loglik == float("-inf"):
            continue
        scored.append((math.log(max(plays, 1)) + loglik, counts))

    if not scored:
        # Nothing explains what we have seen — fall back to the most-played
        # list. Padding with Energy instead risks an opponent deck with no
        # Basic Pokémon, which search_begin rejects outright.
        priors = _load_priors()
        if not priors:
            return []
        return [(max(priors, key=lambda cp: cp[1])[0], 1.0)]

    scored.sort(key=lambda t: -t[0])
    scored = scored[:max(top_k, 1)]
    hi = scored[0][0]
    weights = [math.exp(s - hi) for s, _ in scored]      # softmax, shift-stable
    total = sum(weights) or 1.0
    return [(counts, w / total) for (_, counts), w in zip(scored, weights)]


def _hidden_from(counts: dict, seen: dict, n_deck: int, n_hand: int,
                 n_prize: int, rng=None) -> tuple[list[int], list[int], list[int]]:
    """Split one candidate decklist's unseen cards into (deck, hand, prize).

    The split must be SHUFFLED. Knowing the opponent's 60-card list still leaves
    the harder question of which unseen cards are in hand, deck, or prizes — and
    that is the variable that actually changes game to game.

    This previously took the pool in dict order, which the priors file stores
    sorted by card id, so the simulated opponent hand was deterministically the
    lowest-numbered cards in their list — basic Energy, every rollout, every
    game. Every simulated reply was made by an opponent holding a hand of pure
    Energy and no Pokémon or Trainers.
    """
    pool: list[int] = []
    for cid, n in counts.items():
        pool.extend([cid] * max(n - seen.get(cid, 0), 0))
    need = n_deck + n_hand + n_prize
    if len(pool) < need:
        cycle = [cid for cid, n in counts.items() for _ in range(n)] or [3]
        while len(pool) < need:
            pool.append(cycle[len(pool) % len(cycle)])
    if rng is not None:
        rng.shuffle(pool)
    pool = pool[:need]
    return (pool[n_hand + n_prize:], pool[:n_hand],
            pool[n_hand:n_hand + n_prize])


def _opponent_counts(obs: dict) -> tuple[dict, int, int, int]:
    """(seen counts, deck size, hand size, prize count) for the opponent."""
    cur = obs.get("current") or {}
    me = cur.get("yourIndex", 0)
    players = cur.get("players") or []
    if len(players) < 2:
        return {}, 0, 0, 0
    opp = players[1 - me]
    return (_visible(opp, include_hand=False),
            opp.get("deckCount", 0) or 0,
            opp.get("handCount", 0) or 0,
            len(opp.get("prize") or []))


def _position_rng(obs: dict) -> random.Random:
    """A deterministic RNG keyed to the position.

    Seeding from the position rather than a module-global stream means two runs
    of the same game see the same deals, so both arms of a paired A/B test are
    compared on identical simulated worlds and a rollout is reproducible.

    `step` is a kaggle_environments field and is absent from the raw engine
    observation the local harness passes, so the key also carries
    `turnActionCount` and the option count. Without them the seed is constant
    for every decision inside one of our turns, and all three determinizations
    of every decision replay the same deal.
    """
    cur = obs.get("current") or {}
    step = obs.get("step") or 0
    turn = cur.get("turn") or 0
    action = cur.get("turnActionCount") or 0
    n_opts = len((obs.get("select") or {}).get("option") or [])
    _, n_deck, _, _ = _opponent_counts(obs)
    return random.Random((step * 8191) ^ (turn * 131) ^ (action * 17)
                         ^ (n_opts * 3) ^ n_deck)


def _predict_opponent(
        obs: dict, rng=None) -> tuple[list[int], list[int], list[int]]:
    """(deck, hand, prize) card ids for the opponent's hidden cards.

    The posterior's single most likely decklist, dealt out. `_search_main` goes
    through `_deck_posterior` and `_hidden_from` directly so it can average over
    several candidates; this is the point-estimate entry point for callers that
    want one world.
    """
    seen, n_deck, n_hand, n_prize = _opponent_counts(obs)
    posterior = _deck_posterior(obs, top_k=1)
    if not posterior:
        return [], [], []
    return _hidden_from(posterior[0][0], seen, n_deck, n_hand, n_prize, rng)


def _own_hidden(obs: dict, rng=None) -> tuple[list[int], list[int]]:
    """(deck, prize) for our own side — we know our list, so this is exact."""
    global _MY_DECK
    if _MY_DECK is None:
        try:
            _MY_DECK = read_deck_csv()
        except Exception:
            _MY_DECK = []
    cur = obs.get("current") or {}
    me = cur.get("yourIndex", 0)
    players = cur.get("players") or []
    if me >= len(players):
        return [], []
    mine = players[me]
    n_deck = mine.get("deckCount", 0) or 0
    n_prize = len(mine.get("prize") or [])

    remaining: dict = {}
    for cid in _MY_DECK:
        remaining[cid] = remaining.get(cid, 0) + 1
    for cid, n in _visible(mine, include_hand=True).items():
        remaining[cid] = remaining.get(cid, 0) - n

    hidden = [cid for cid, n in remaining.items() for _ in range(max(n, 0))]
    need = n_deck + n_prize
    if len(hidden) < need:
        hidden.extend([3] * (need - len(hidden)))
    if rng is not None:
        rng.shuffle(hidden)
    hidden = hidden[:need]
    return hidden[n_prize:], hidden[:n_prize]


def _side_totals(player) -> tuple[float, float, float, float]:
    """(board HP, Energy in play, bench width, damage taken) for one side."""
    hp = energy = damage = 0.0
    for zone in ("active", "bench"):
        for mon in _g(player, zone, []) or []:
            if mon is None:
                continue
            cur_hp = _g(mon, "hp", 0) or 0
            hp += cur_hp
            damage += (_g(mon, "maxHp", 0) or 0) - cur_hp
            energy += len(_g(mon, "energies", []) or [])
    bench = len([m for m in (_g(player, "bench", []) or []) if m is not None])
    return hp, energy, bench, damage


# --- trajectory features (playbook C1/C2) -----------------------------------
# The evaluator above scores the board as it stands. C1/C2 of D29 score where
# the board is going: who gets an attacker online first, and how much Energy
# and damage each side is expected to hold two of its own turns from now.
#
# Two shipped tables carry it, and the agent reads both at run time:
#
#   trajectory_curves.json   `ptcg/trajectory.py`'s conditional curves, fitted
#                            on 193,479 (episode, seat, turn) series rows over
#                            14,182 rated ladder games: given an archetype, a
#                            turn bucket and the Energy on its board now, the
#                            mean Energy it holds k of its own turns later.
#                            Held-out bias at k=2 is -0.058 Energy against
#                            +0.926 for the playbook's one-attach-a-turn line
#                            (data/analysis/TRAJECTORY_REPORT.md), which is why
#                            the projection is this table and not arithmetic.
#   energy_mechanics.json    `ptcg/energy_mechanics.py`'s KB: every Energy
#                            accelerator in the pool, classified from its
#                            effect text, with the rate one use is worth
#                            (unbounded text capped at 2, as `energy_plan`
#                            caps it). Our own hand and board are private
#                            information the field curve cannot see, so a
#                            visible accelerator is credited on top of it.
#
# `ptcg/energy_plan.py` is the reference implementation of the same projection
# with exact type matching and a full feed-order search. It cannot ship (it
# imports the card pool and the outs machinery), so what runs here is its lean
# form: cost size instead of exact type matching, board-level growth fed to the
# focal attacker, which is the feed order the planner's greedy would pick.
# `scripts/fit_trajectory_features.py` computes these exact functions over the
# mined positions, so the weights below are fitted on what the agent computes.
TRAJECTORY_FILES = ("trajectory_curves.json",)
MECHANICS_FILES = ("energy_mechanics.json",)

TRAJ_HORIZON = 5          # ptcg.energy_plan.HORIZON — D32's five-turn cap
# The horizon the C2 threat differential is read at, in each side's own turns.
# The fit prices all four and k=3 is the best of them: 1.71 [0.76, 2.65] at
# loglik -5501.91, against 1.42 [0.47, 2.36] at -5503.92 for k=2, 1.03
# [0.05, 2.01] at k=1 and 1.25 [0.15, 2.34] on the present board
# (`data/analysis/trajectory_fit.json`, "horizons"; same 8,792 positions, same
# terms). Two log-likelihood units and a coefficient a fifth larger, for a
# projection that costs the same arithmetic one turn further out — so the
# weight below moves with it.
try:
    TRAJ_K = int(os.environ.get("CABT_TRAJ_K") or 3)
except Exception:
    TRAJ_K = 3
# The fitted coefficient at each horizon, so WEIGHTS below cannot drift out of
# step with TRAJ_K (`data/analysis/trajectory_fit.json`, "horizons").
_THREAT_WEIGHT_BY_K = {0: 1.25, 1: 1.03, 2: 1.42, 3: 1.71}
TRAJ_RATE_CAP = 2         # ptcg.energy_plan.UNBOUNDED_CAP
TRAJ_MAX_BIN = 9          # the curves' top Energy bin is "9+"
# Field-wide realized Energy per own turn, the fallback when no cell matches
# (TRAJECTORY_REPORT.md, "realized rate/turn" for (all)).
TRAJ_FIELD_RATE = 0.541

_CURVES: dict | None = None
_CURVES_SOURCE = ""
_ACCEL: dict | None = None
_ACCEL_SOURCE = ""
_ATTACK_PROFILE: dict = {}
# The archetype label is a per-episode constant, so it is read once a decision
# and held: inside a rollout the opponent's decklist is already fixed by the
# determinization, and rescanning the discard on every leaf buys nothing.
_TRAJ_ARCH = {"us": "(all)", "them": "(all)"}

TELEMETRY_TRAJ = {
    "curves_missing": 0,      # a feature computed with no table loaded
    "feature_errors": 0,      # a guarded failure; the term scored 0.0
    "threat_scored": 0,       # positions the C2 term was computed on
    "threat_nonzero": 0,      # ... of which it moved the margin
    "evo_scored": 0,          # positions the evo integral replaced the term on
}


def _bundle_paths(names, repo_rel=None) -> list[str]:
    """Where a shipped data file can be, bundle first, repo last."""
    paths = []
    for name in names:
        paths.append(name)
        if _HERE:
            paths.append(os.path.join(_HERE, name))
        paths.append(os.path.join(_KAGGLE_DIR, name))
    if _HERE and repo_rel:
        paths.append(os.path.join(_HERE, os.pardir, *repo_rel))
    return paths


def _read_json(paths):
    import json
    for path in paths:
        try:
            with open(path, "r") as fh:
                return json.load(fh), path
        except Exception:
            continue
    return None, ""


def _curves() -> dict:
    """(archetype, k, turn bucket, Energy bin) -> mean Energy at t+k."""
    global _CURVES, _CURVES_SOURCE
    if _CURVES is not None:
        return _CURVES
    _CURVES = {}
    blob, src = _read_json(_bundle_paths(
        TRAJECTORY_FILES, ("data", "analysis", "trajectory_curves.json")))
    for arch, metrics in ((blob or {}).get("curves") or {}).items():
        for kname, cells in ((metrics or {}).get("energy_in_play") or {}).items():
            try:
                k = int(str(kname).lstrip("k"))
            except ValueError:
                continue
            for cell in cells or []:
                try:
                    _CURVES[(arch, k, cell["t_bucket"], int(cell["bin"]))] = \
                        float(cell["mean"])
                except (KeyError, TypeError, ValueError):
                    continue
    if _CURVES:
        _CURVES_SOURCE = src
    return _CURVES


def _accel_rates() -> dict:
    """card id -> Energy one use of it accelerates onto the board.

    Attack riders are excluded: they cost the attack, so they are not free
    ramp on a turn the plan wants damage (`energy_plan.deck_accel_package`
    reports them separately for the same reason).
    """
    global _ACCEL, _ACCEL_SOURCE
    if _ACCEL is not None:
        return _ACCEL
    _ACCEL = {}
    blob, src = _read_json(_bundle_paths(
        MECHANICS_FILES, ("data", "energy_mechanics.json")))
    for rec in ((blob or {}).get("records") or []):
        if rec.get("mechanic") != "acceleration":
            continue
        if rec.get("frequency") == "attack_rider":
            continue
        try:
            cid = int(rec.get("card_id") or 0)
        except (TypeError, ValueError):
            continue
        rate = rec.get("rate")
        rate = TRAJ_RATE_CAP if rate is None else int(rate)
        if cid and rate > _ACCEL.get(cid, 0):
            _ACCEL[cid] = rate
    if _ACCEL:
        _ACCEL_SOURCE = src
    return _ACCEL


def _attack_profile(card_id: int) -> list[tuple[int, float]]:
    """(cost size, damage) for every attack a card prints, cheapest first."""
    prof = _ATTACK_PROFILE.get(card_id)
    if prof is not None:
        return prof
    cards, attacks = _tables()
    prof = []
    card = cards.get(card_id)
    for aid in (getattr(card, "attacks", None) or []):
        atk = attacks.get(aid)
        if atk is None:
            continue
        cost = len(getattr(atk, "energies", None) or [])
        prof.append((cost, float(getattr(atk, "damage", 0) or 0)))
    prof.sort()
    _ATTACK_PROFILE[card_id] = prof
    return prof


# --- scaled damage: honest numbers for scaling attacks (CABT_SCALED_DAMAGE) -
# `_attack_profile` reads printed damage, and a scaling attack prints the
# wrong number: Alakazam's Powerful Hand prints 0 and places 2 damage
# counters per card in its player's hand, so the threat ladder priced a
# 20-card Dudunsparce engine at zero while it assembled a 400-damage
# one-shot. The Energy scalers are the same blindness (attack 339 prints 10
# and does +50 per Energy on our Active).
#
# `attack_scalers.json` is the fix — `ptcg/attack_scalers.py` classifies the
# pool's unambiguously computable scaling attacks (69 of the 364 with
# damage-moving text; coin flips, discard-priced scaling and conditionals
# stay unclassified and keep their printed number) into
# (base, per, unit-kind) records, numbers only, no card text. Damage is
# max(base + per * quantity, 0) with the quantity read off the OBSERVED
# state, both sides. No new evaluator weight: the corrected numbers feed the
# existing screened threat terms (threat_traj at its fitted coefficient, the
# k=1 incoming read, the rules policy's attack pick).
#
# Future-turn projections hold every scaling quantity at its current
# observed value — `_threat_at(k)` grows the Energy BUDGET (can the attack
# be paid) but not the hand/bench/Energy COUNT the damage scales by. A
# hand-growth model is out of scope; the held quantity understates a growing
# engine and never invents damage.
# Default ON: the pre-registered gate passed on 2026-08-07. Targeted cell
# (codex_alakazam specialist, 500 games an arm, paired seed block 110000,
# zero forfeits): 0.4340 off -> 0.4680 on, +3.4 pp; the panel's independent
# alakazam replicate (seed block 321000) read +0.6 pp beside it. Pooled
# specialist panel (8 entries x 500 an arm, play-share weighted, seed block
# 320000): clean 0.7540 -> 0.7382 (-0.81 SE), all-games 0.8047 -> 0.8011
# (-0.27 SE), no per-cell regression at 2 SE — inside the no-regression
# guard rail the gate pre-registered (`scripts/scaled_damage_gate.py`,
# `data/analysis/scaled_damage_gate.json`).
SCALER_FILES = ("attack_scalers.json",)
try:
    SCALED_DAMAGE_ENABLED = bool(
        int(os.environ.get("CABT_SCALED_DAMAGE") or 1))
except Exception:
    SCALED_DAMAGE_ENABLED = True

_SCALERS: dict | None = None
_SCALERS_SOURCE = ""
_ATTACK_PROFILE_SC: dict = {}

TELEMETRY_SCALED = {
    "kb_missing": 0,        # priced a position with no KB loaded
    "ctx_errors": 0,        # a guarded failure; printed damage scored
    "attacks_scaled": 0,    # attack damages the KB re-priced
}


def _scalers() -> dict:
    """attackId -> (base, per, unit kind), from the shipped KB."""
    global _SCALERS, _SCALERS_SOURCE
    if _SCALERS is not None:
        return _SCALERS
    _SCALERS = {}
    blob, src = _read_json(_bundle_paths(SCALER_FILES))
    for aid, rec in ((blob or {}).get("attacks") or {}).items():
        try:
            _SCALERS[int(aid)] = (float(rec["base"]), float(rec["per"]),
                                  str(rec["kind"]))
        except (KeyError, TypeError, ValueError):
            continue
    if _SCALERS:
        _SCALERS_SOURCE = src
    else:
        TELEMETRY_SCALED["kb_missing"] += 1
    return _SCALERS


def _attack_profile_sc(card_id: int) -> list:
    """(cost size, printed damage, scaler-or-None), cheapest first."""
    prof = _ATTACK_PROFILE_SC.get(card_id)
    if prof is not None:
        return prof
    cards, attacks = _tables()
    sc = _scalers()
    prof = []
    card = cards.get(card_id)
    for aid in (getattr(card, "attacks", None) or []):
        atk = attacks.get(aid)
        if atk is None:
            continue
        cost = len(getattr(atk, "energies", None) or [])
        prof.append((cost, float(getattr(atk, "damage", 0) or 0),
                     sc.get(int(aid))))
    prof.sort(key=lambda t: (t[0], t[1]))
    _ATTACK_PROFILE_SC[card_id] = prof
    return prof


def _hand_count(player) -> int:
    """Hand size off either side: the count field, else the visible list."""
    n = int(_g(player, "handCount", 0) or 0)
    if n:
        return n
    return len(_g(player, "hand", []) or [])


def _dmg_counters(mon) -> float:
    """Damage counters on one Pokemon — a counter is 10 damage."""
    return max((float(_g(mon, "maxHp", 0) or 0)
                - float(_g(mon, "hp", 0) or 0)) / 10.0, 0.0)


def _scale_ctx(attacker, defender) -> dict:
    """Every board-level scaling quantity, from the attacking side's seat.

    Per-slot quantities (energy_self, dmg_self) come off the slot inside
    `_threat_at`; everything here is one number per side per evaluation.
    Both sides get a context — their scaling threats and ours price alike.
    """
    _, en_a, bench_a, _ = _side_totals(attacker)
    _, en_d, bench_d, _ = _side_totals(defender)
    act = _g(defender, "active", []) or []
    act = act[0] if act and act[0] is not None else None
    a_act = _g(attacker, "active", []) or []
    a_act = a_act[0] if a_act and a_act[0] is not None else None
    in_play = bench_a + (1 if a_act is not None else 0)
    en_act_a = (float(len(_g(a_act, "energies", []) or []))
                if a_act is not None else 0.0)
    en_act_d = (float(len(_g(act, "energies", []) or []))
                if act is not None else 0.0)
    return {
        "hand_self": float(_hand_count(attacker)),
        "hand_opp": float(_hand_count(defender)),
        "energy_self_all": en_a,
        "energy_opp_all": en_d,
        "energy_opp_active": en_act_d,
        "energy_both_active": en_act_a + en_act_d,
        "bench_self": float(bench_a),
        "bench_opp": float(bench_d),
        "bench_both": float(bench_a + bench_d),
        "in_play_self": float(in_play),
        "dmg_opp_active": _dmg_counters(act) if act is not None else 0.0,
        "prizes_taken_self":
            float(max(6 - len(_g(attacker, "prize", []) or []), 0)),
        "prizes_taken_opp":
            float(max(6 - len(_g(defender, "prize", []) or []), 0)),
        "in_play_self_team_rocket": _named_in_play(attacker, "Team Rocket’s"),
    }


def _named_in_play(player, prefix: str) -> float:
    """In-play Pokemon whose card name carries the given owner prefix — the
    engine's own card table supplies the names at run time, so the shipped
    KB stays numeric."""
    cards, _ = _tables()
    n = 0
    for zone in ("active", "bench"):
        for mon in _g(player, zone, []) or []:
            if mon is None:
                continue
            data = cards.get(int(_g(mon, "id", 0) or 0))
            nm = getattr(data, "name", "") if data is not None else ""
            if nm and nm.replace("'", "’").startswith(prefix):
                n += 1
    return float(n)


def _scaled_damage(scaler, sctx: dict, slot_e: float, slot_dmgc: float,
                   on_bench: bool = False) -> float:
    """max(base + per * observed quantity, 0) for one attack on one slot."""
    base, per, kind = scaler
    if kind == "energy_self":
        q = slot_e
    elif kind == "dmg_self":
        q = slot_dmgc
    elif kind == "if_from_bench":
        # Whether the attacker switched in this turn is not observable, so
        # the rider prices on a benched attacker (it can always switch in on
        # its own turn) and an Active attacker holds at base.
        q = 1.0 if on_bench else 0.0
    else:
        q = sctx.get(kind, 0.0)     # "flat" reads no quantity: base only
    TELEMETRY_SCALED["attacks_scaled"] += 1
    return max(base + per * q, 0.0)


class _SlotList(list):
    """One side's slots plus its scaling context, when the KB is live."""
    __slots__ = ("sctx",)


def _traj_bucket(turn) -> str:
    """The curves' turn bucket, in the seat's own turns.

    The series was indexed by the seat's own turn; an observation carries the
    global turn, and the two seats alternate, so own turn is half of it. The
    buckets are three turns wide, so the half-turn the seats are out of step by
    cannot move a position more than one bucket, and only at a boundary.
    """
    try:
        t = max(1, (int(turn) + 1) // 2)
    except (TypeError, ValueError):
        t = 1
    return "t1-3" if t <= 3 else ("t4-6" if t <= 6 else "t7+")


def _traj_growth(arch: str, bucket: str, e_now: float, k: int) -> float:
    """Expected Energy this side gains over its next k turns.

    k of 1-3 is a table read; beyond that the last fitted step is repeated,
    which is the flattest honest extrapolation (the curves' own steps shrink
    with k, so repeating the k3-k2 step does not run away).
    """
    if k <= 0:
        return 0.0
    curves = _curves()
    if not curves:
        TELEMETRY_TRAJ["curves_missing"] += 1
        return TRAJ_FIELD_RATE * k
    b = min(int(e_now), TRAJ_MAX_BIN)

    def at(kk: int) -> float:
        v = curves.get((arch, kk, bucket, b))
        if v is None:
            v = curves.get(("(all)", kk, bucket, b))
        if v is None:
            return e_now + TRAJ_FIELD_RATE * kk
        return v

    if k <= 3:
        return at(k) - e_now
    g3, g2 = at(3) - e_now, at(2) - e_now
    return g3 + max(g3 - g2, 0.0) * (k - 3)


def _traj_slots(player) -> "_SlotList":
    """(Energy on it, card id, damage counters on it, benched?) for every
    Pokemon this side has in play. The last two elements exist for the
    scaled-damage KB's per-slot quantities and cost a subtraction each."""
    out = _SlotList()
    for zone, benched in (("active", False), ("bench", True)):
        for mon in _g(player, zone, []) or []:
            if mon is None:
                continue
            cid = _g(mon, "id", 0) or 0
            if cid:
                out.append((float(len(_g(mon, "energies", []) or [])),
                            int(cid), _dmg_counters(mon), benched))
    return out


def _visible_accel(player) -> float:
    """The best accelerator this side can see — private information for us.

    Our hand is ours to read; theirs is hidden, so this term is zero on their
    side by construction and the asymmetry is the point. In play it counts
    abilities already on the board; in hand it counts the copy we are holding.
    """
    rates = _accel_rates()
    if not rates:
        return 0.0
    best = 0.0
    for zone in ("active", "bench", "hand"):
        for card in _g(player, zone, []) or []:
            if card is None:
                continue
            cid = _g(card, "id", 0) or 0
            r = rates.get(int(cid), 0)
            if r > best:
                best = float(r)
            for sub in ("tools",):
                for c in _g(card, sub, []) or []:
                    r = rates.get(int(_g(c, "id", 0) or 0), 0)
                    if r > best:
                        best = float(r)
    return best


def _online_turn(slots, growth) -> int:
    """The earliest of our next 5 turns an attacker's cheapest attack is paid.

    Board-level growth is fed to whichever attacker reaches its cost soonest,
    which is the feed order `energy_plan.plan` searches for and finds. Nothing
    online inside the horizon returns HORIZON + 1, so the feature is bounded.
    """
    best = TRAJ_HORIZON + 1
    for e, cid, *_ in slots:
        prof = _attack_profile(cid)
        if not prof:
            continue
        cost = prof[0][0]
        for k in range(0, TRAJ_HORIZON + 1):
            if k >= best:
                break
            if e + growth(k) >= cost:
                best = k
                break
    return best


def _threat_at(slots, growth, k: int) -> float:
    """The hardest attack this side can pay for k of its turns from now.

    With a scaling context on the slots (CABT_SCALED_DAMAGE), an attack the
    KB names is priced at its observed damage instead of its printed number.
    The budget grows with k; the scaling quantities do not — held at their
    current observed values, the stated no-hand-growth-model limitation.
    """
    gain = growth(k)
    best = 0.0
    sctx = getattr(slots, "sctx", None)
    for e, cid, dmgc, on_bench in slots:
        budget = e + gain
        if sctx is None:
            for cost, dmg in _attack_profile(cid):
                if cost <= budget and dmg > best:
                    best = dmg
        else:
            for cost, dmg, sc in _attack_profile_sc(cid):
                if sc is not None:
                    dmg = _scaled_damage(sc, sctx, e, dmgc, on_bench)
                if cost <= budget and dmg > best:
                    best = dmg
    return best


def _label_of_ids(card_ids) -> str:
    """The label the series was cut by: the highest-HP Pokemon in a pile.

    `ptcg.extract.label_archetype_fast` names a decklist that way, and
    `ptcg/trajectory.py` cut its curves by that name, so a curve key is only
    reachable through the same rule.
    """
    cards, _ = _tables()
    best, best_hp = "(all)", -1.0
    for cid in card_ids:
        info = cards.get(int(cid or 0))
        hp = getattr(info, "hp", None) if info is not None else None
        if hp and float(hp) > best_hp:
            best, best_hp = getattr(info, "name", "(all)"), float(hp)
    return best


def _visible_ids(player, zones=("active", "bench", "discard")):
    for zone in zones:
        for card in _g(player, zone, []) or []:
            if card is None:
                continue
            yield _g(card, "id", 0) or 0
            for sub in ("preEvolution",):
                for c in _g(card, sub, []) or []:
                    yield _g(c, "id", 0) or 0


def _refresh_traj_arch(obs) -> None:
    """Re-read both archetype labels. Once a decision, never inside a leaf.

    Their label comes off `_deck_posterior` — the search's own decklist
    inference, graded 0.63 right at turn 1 rising to 0.94 by turn 10 — because
    what is on their board names the deck only 15% of the time at turn 1 and
    the trajectory features are worth most exactly that early. Ours is read
    off our own board, discard and hand, which is private information and
    needs no inference.

    The same read carries the posterior's confidence in that label, over the
    same top-K it would be spending determinizations on, and hands both to
    `_set_posture`: the matchup posture and the trajectory curve are keyed on
    one inference, made once, rather than on two that could disagree.
    """
    try:
        cur = (obs or {}).get("current") or {}
        players = cur.get("players") or []
        if len(players) < 2:
            return
        me = cur.get("yourIndex", 0) or 0
        _TRAJ_ARCH["us"] = _label_of_ids(
            _visible_ids(players[me], ("active", "bench", "discard", "hand")))
        label, confidence, post = "(all)", 0.0, None
        try:
            post = _deck_posterior(obs, top_k=POSTERIOR_TOP_K)
            if post:
                label = _label_of_ids(post[0][0].keys())
                confidence = float(post[0][1])
        except Exception:
            label, confidence = "(all)", 0.0
        if label == "(all)":
            label = _label_of_ids(_visible_ids(players[1 - me]))
            confidence = 0.0        # a board-derived label is not a posterior
        _TRAJ_ARCH["them"] = label
        _set_posture(label, confidence)
        if EVO_INTEGRAL_ENABLED:
            # The evolution pools ride the same once-a-decision read as the
            # archetype labels: theirs is this posterior's top-1, ours the
            # episode's own list, and the leaves below never re-infer either.
            _evo_refresh_pools(post)
    except Exception:
        TELEMETRY_TRAJ["feature_errors"] += 1


def _traj_projection(cur, mine, theirs):
    """Both sides' projected Energy, as (slots, growth) pairs.

    The growth functions are the whole projection: k of our turns from now,
    how much more Energy each side is holding. Ours adds the accelerator we
    can see in our own hand or on our own board — private information the
    field curve averages over and therefore understates for us.
    """
    bucket = _traj_bucket(_g(cur, "turn", 1))
    us, them = _TRAJ_ARCH["us"], _TRAJ_ARCH["them"]
    slots_m, slots_t = _traj_slots(mine), _traj_slots(theirs)
    e_m = sum(s[0] for s in slots_m)
    e_t = sum(s[0] for s in slots_t)
    if SCALED_DAMAGE_ENABLED:
        # Both sides' scaling contexts ride the slots, so every downstream
        # `_threat_at` — the C2 ladder, the k=1 incoming read, the evo
        # integral — prices scaling attacks from the observed state.
        try:
            slots_m.sctx = _scale_ctx(mine, theirs)
            slots_t.sctx = _scale_ctx(theirs, mine)
        except Exception:
            TELEMETRY_SCALED["ctx_errors"] += 1
    accel_m = _visible_accel(mine)

    def g_m(k: int) -> float:
        return _traj_growth(us, bucket, e_m, k) + (accel_m if k else 0.0)

    def g_t(k: int) -> float:
        return _traj_growth(them, bucket, e_t, k)

    return (slots_m, g_m, e_m), (slots_t, g_t, e_t)


def _threat_traj(cur, mine, theirs) -> float:
    """C2's shipped term: the damage-potential differential at t+TRAJ_K.

    The hardest attack each side can pay for TRAJ_K of its own turns from now,
    ours minus theirs. This is the only trajectory term the fit priced (see
    WEIGHTS); the other two are measured-null and are not computed here.
    """
    (sm, gm, _), (st, gt, _) = _traj_projection(cur, mine, theirs)
    return _threat_at(sm, gm, TRAJ_K) - _threat_at(st, gt, TRAJ_K)


def _trajectory_terms(cur, mine, theirs) -> tuple[float, float, float]:
    """(online_lead, energy_traj_diff, threat_traj_diff), all three.

    What `scripts/fit_trajectory_features.py` prices. online_lead is their
    earliest online turn minus ours, so positive means we reach an attack
    first; the other two are our projection at t+TRAJ_K minus theirs. Only the
    third one has a weight — this function stays so the other two can be
    re-measured on new data without being reconstructed from the write-up.
    """
    (sm, gm, e_m), (st, gt, e_t) = _traj_projection(cur, mine, theirs)
    online_lead = float(_online_turn(st, gt) - _online_turn(sm, gm))
    energy_diff = (e_m + gm(TRAJ_K)) - (e_t + gt(TRAJ_K))
    threat_diff = _threat_at(sm, gm, TRAJ_K) - _threat_at(st, gt, TRAJ_K)
    return online_lead, energy_diff, threat_diff


# --- the evolution-aware discounted threat integral (D35 nomination) --------
# `data/analysis/INTERACTION_MINING.md` screened twenty-three candidates and
# nominated exactly one: the composition of two repairs to the C2 ladder.
#
#   coverage   `_attack_profile` reads only a card's own printed attacks, so
#              the shipped ladder never sees a benched Charmander's future
#              Charizard. Here each slot's profile unions in the attacks of
#              evolutions reachable within the horizon — one evolution step
#              per own turn, Energy surviving evolution — with the evolution
#              cards priced by outs arithmetic: a copy in hand is certain, an
#              unseen copy prices hypergeometrically with the draws its step
#              schedule allows, chain steps multiplying under a stated
#              independence approximation. The weighted repair moves the k=3
#              ladder on 31.7% of mined positions; in the side-by-side fit it
#              takes the whole load (+0.171) and the shipped term goes null.
#   shape      the shipped term is a threshold at one horizon (k = TRAJ_K).
#              The integral is the gradient form: sum over k = 0..5 of
#              gamma^k times the threat-at-k differential, so a stoke that
#              closes distance without crossing t+3 payability earns
#              something, and online at t+1 outscores online at t+3.
#
# Each repair alone failed the screen's held-out interval; the composition at
# gamma 0.5-0.8 passes both (training z 2.9-3.6, held-out logloss -0.0012 to
# -0.0014 with the interval excluding zero). Gamma is 0.6 — the middle of
# the passing plateau and the best training z (3.16) — fixed, not fitted at
# run time.
#
# Pools. Their side is the posterior's top-1 decklist, the same inference the
# curve and the postures already key on. Our side is the episode's own list
# read from deck.csv — the agent knows its 60 cards, where the screen's fit
# had to stand in a posterior for the unknown focal seat; the nomination
# names this substitution as the Goodhart risk to close. The shipped
# competition list carries zero evolution cards, so on our own deck the
# evo profile equals the shipped profile exactly and the repair's whole
# effect arrives through the opponent's side; a side with no pool loaded
# falls back to the shipped profile and is the same degrade.
#
# When CABT_EVO_INTEGRAL is on this REPLACES the threat_traj feature value in
# `_margin` — same slot, same one projection — and the weight swaps with it
# (see WEIGHTS). Off, nothing here is computed and nothing costs time.
try:
    EVO_INTEGRAL_ENABLED = bool(int(os.environ.get("CABT_EVO_INTEGRAL") or 0))
except Exception:
    EVO_INTEGRAL_ENABLED = False
EVO_GAMMA = 0.6
# The fitted coefficient for the replacement term, in evaluator units per raw
# unit of the discounted integral. Provenance, screen fit reproduced exactly
# (`scripts/mine_interactions.py screen`, integral_evo_g0.6 in
# `data/analysis/interaction_screen.json`; 7,506 training episodes
# 07-31..08-05, logistic MLE on the base terms with threat_traj replaced,
# turn fixed effects): beta +0.091037 per training SD [0.034608, 0.147465],
# z 3.16; the raw integral's training SD is 182.5935 and the same fit's
# prize_diff coefficient is 0.60211, so the prize-1000 anchor — the exact
# rule that produced the shipped 1.71 (`scripts/fit_trajectory_features.py`)
# — gives 1000 x (0.091037 / 182.5935) / 0.60211 = 0.828, 95% CI
# [0.31, 1.34]. Held out (08-06, 1,286 episodes): logloss -0.00132
# [-0.00256, -0.00018], AUC +0.0024 [-0.0000, +0.0049].
THREAT_INTEGRAL_WEIGHT = 0.828

_EVO_MAP: dict | None = None
_EVO_POOLS: dict = {"us": None, "them": None}
_EVO_EDGES: dict = {"us": {}, "them": {}}


def _evo_map() -> dict:
    """name -> [(evolution card id, steps)] within two stages.

    `scripts/mine_interactions.py::_evo_map`, verbatim: forward over the card
    table's own `evolvesFrom` names, capped at two steps because the game
    rules cap evolution lines at basic / stage 1 / stage 2.
    """
    global _EVO_MAP
    if _EVO_MAP is not None:
        return _EVO_MAP
    cards, _ = _tables()
    fwd: dict = {}
    names: dict = {}
    for cid, card in cards.items():
        nm = getattr(card, "name", None)
        if nm:
            names[cid] = nm
        ef = getattr(card, "evolvesFrom", None)
        if ef:
            fwd.setdefault(ef, []).append(cid)
    out: dict = {}
    for nm in set(fwd) | set(names.values()):
        steps = []
        for e1 in fwd.get(nm, ()):
            steps.append((e1, 1))
            for e2 in fwd.get(names.get(e1, ""), ()):
                steps.append((e2, 2))
        if steps:
            out[nm] = steps
    _EVO_MAP = out
    return _EVO_MAP


def _p_at_least_one(outs: int, unseen: int, draws: int) -> float:
    """`ptcg.creation.outs.p_at_least_one`, inlined — the bundle cannot
    import ptcg, and the arithmetic is four lines of math.comb."""
    if outs <= 0 or unseen <= 0 or draws <= 0:
        return 0.0
    outs = min(outs, unseen)
    draws = min(draws, unseen)
    blanks = unseen - outs
    if blanks < draws:
        return 1.0
    return 1.0 - math.comb(blanks, draws) / math.comb(unseen, draws)


def _evo_refresh_pools(post) -> None:
    """Re-point both sides' evolution pools; once a decision, never in a leaf.

    Our pool is the episode's own 60-card list and never changes; theirs is
    the posterior top-1 the caller already computed. A pool change resets
    that side's edge cache, so a re-identified opponent cannot keep the old
    list's evolution lines.
    """
    global _MY_DECK
    if _EVO_POOLS["us"] is None:
        if _MY_DECK is None:
            try:
                _MY_DECK = read_deck_csv()
            except Exception:
                _MY_DECK = []
        pool: dict = {}
        for cid in _MY_DECK:
            pool[cid] = pool.get(cid, 0) + 1
        _EVO_POOLS["us"] = pool
        _EVO_EDGES["us"] = {}
    theirs = dict(post[0][0]) if post else {}
    if theirs and theirs != _EVO_POOLS["them"]:
        _EVO_POOLS["them"] = theirs
        _EVO_EDGES["them"] = {}


def _evo_edges_for(side: str, cid: int):
    """The (evolution id, steps, mid-stage name) edges a slot can reach,
    filtered to evolutions the side's pool actually contains — the cached
    attack-profile-union structure; only the availability price varies by
    position."""
    cache = _EVO_EDGES[side]
    hit = cache.get(cid)
    if hit is not None:
        return hit
    pool = _EVO_POOLS[side] or {}
    cards, _ = _tables()
    nm = getattr(cards.get(cid), "name", None)
    edges = []
    for evo_id, steps in (_evo_map().get(nm or "", ()) or ()):
        if pool.get(evo_id, 0) <= 0:
            continue
        mid = ""
        if steps == 2:
            mid = getattr(cards.get(evo_id), "evolvesFrom", "") or ""
        edges.append((evo_id, steps, mid))
    cache[cid] = tuple(edges)
    return cache[cid]


def _evo_position_ctx(player, side: str):
    """One side's availability inputs, read once per evaluation.

    The exact counting the screen fitted (`stage_evo`): seen is active,
    bench, discard plus preEvolution cards underneath — and our own hand,
    which is private information; theirs is hidden, so their unseen set is
    deck plus hand and their hand size is extra draws already taken.
    """
    pool = _EVO_POOLS[side] or {}
    if not pool:
        return None
    cards, _ = _tables()
    seen: dict = {}
    zones = (("active", "bench", "discard", "hand") if side == "us"
             else ("active", "bench", "discard"))
    for zone in zones:
        for card in _g(player, zone, []) or []:
            if card is None:
                continue
            cid = int(_g(card, "id", 0) or 0)
            if cid:
                seen[cid] = seen.get(cid, 0) + 1
            for c in _g(card, "preEvolution", []) or []:
                pid = int(_g(c, "id", 0) or 0)
                if pid:
                    seen[pid] = seen.get(pid, 0) + 1
    hand_ids: dict = {}
    hand_names: dict = {}
    if side == "us":
        for card in _g(player, "hand", []) or []:
            if card is None:
                continue
            cid = int(_g(card, "id", 0) or 0)
            if not cid:
                continue
            hand_ids[cid] = hand_ids.get(cid, 0) + 1
            nm = getattr(cards.get(cid), "name", None)
            if nm:
                hand_names[nm] = hand_names.get(nm, 0) + 1
        unseen = int(_g(player, "deckCount", 0) or 0)
        extra = 0
    else:
        n_hand = int(_g(player, "handCount", 0) or 0)
        unseen = int(_g(player, "deckCount", 0) or 0) + n_hand
        extra = n_hand
    name_outs: dict = {}
    for i, c in pool.items():
        nm = getattr(cards.get(int(i)), "name", None)
        if nm:
            name_outs[nm] = name_outs.get(nm, 0) \
                + max(c - seen.get(int(i), 0), 0)
    return (pool, seen, hand_ids, hand_names, max(unseen, 1), extra,
            name_outs)


def _evo_avail(ctx, evo_id: int, steps: int, k: int, mid: str) -> float:
    """P(the chain is playable on schedule): a copy in hand is certain, an
    unseen copy prices by outs with the draws its step schedule allows (the
    step-s card of an s-step chain must come down by own turn k - (steps -
    s)), chain steps multiplying — the screen's `make_avail`, verbatim."""
    pool, seen, hand_ids, hand_names, unseen, extra, name_outs = ctx
    p = 1.0
    for s in range(1, steps + 1):
        if s < steps:                    # the mid-stage card, priced by name
            if hand_names.get(mid, 0) > 0:
                continue
            outs = name_outs.get(mid, 0)
        else:
            if hand_ids.get(evo_id, 0) > 0:
                continue
            outs = pool.get(evo_id, 0) - seen.get(evo_id, 0)
        ps = _p_at_least_one(max(outs, 0), unseen, k - (steps - s) + extra)
        if ps <= 0.0:
            return 0.0
        p *= ps
    return p


def _evo_threat_at(slots, growth, k: int, side: str, ctx) -> float:
    """`_threat_at` with evolution coverage: everything the shipped ladder
    does is kept — full board growth to every slot, hardest payable attack,
    max over slots — and each slot's profile unions in the attacks of
    evolutions reachable within k own turns, damage weighted by the chain's
    availability (`scripts/mine_interactions.py::_threat_at_evo`, the
    weighted variant that won the screen)."""
    gain = growth(k)
    best = 0.0
    sctx = getattr(slots, "sctx", None)
    for e, cid, dmgc, on_bench in slots:
        budget = e + gain
        if sctx is None:
            for cost, dmg in _attack_profile(int(cid)):
                if cost <= budget and dmg > best:
                    best = dmg
        else:
            for cost, dmg, sc in _attack_profile_sc(int(cid)):
                if sc is not None:
                    dmg = _scaled_damage(sc, sctx, e, dmgc, on_bench)
                if cost <= budget and dmg > best:
                    best = dmg
        for evo_id, steps, mid in _evo_edges_for(side, int(cid)):
            if steps > k:
                continue
            p = _evo_avail(ctx, evo_id, steps, k, mid)
            if p <= 0.0:
                continue
            # Damage counters survive evolution and the Energy stays, so the
            # evolution's scaling attacks price off the same slot quantities.
            if sctx is None:
                for cost, dmg in _attack_profile(evo_id):
                    if cost <= budget and dmg * p > best:
                        best = dmg * p
            else:
                for cost, dmg, sc in _attack_profile_sc(evo_id):
                    if sc is not None:
                        dmg = _scaled_damage(sc, sctx, e, dmgc, on_bench)
                    if cost <= budget and dmg * p > best:
                        best = dmg * p
    return best


def _threat_integral(mine, theirs, sm, gm, st, gt) -> float:
    """Sum over k = 0..TRAJ_HORIZON of gamma^k times the evolution-aware
    threat differential. A side with no pool loaded scores the shipped
    profile, which for a pool with no evolution cards is the identical
    number anyway."""
    ctx_m = _evo_position_ctx(mine, "us")
    ctx_t = _evo_position_ctx(theirs, "them")
    total = 0.0
    for k in range(TRAJ_HORIZON + 1):
        um = (_evo_threat_at(sm, gm, k, "us", ctx_m) if ctx_m is not None
              else _threat_at(sm, gm, k))
        ut = (_evo_threat_at(st, gt, k, "them", ctx_t) if ctx_t is not None
              else _threat_at(st, gt, k))
        total += (EVO_GAMMA ** k) * (um - ut)
    TELEMETRY_TRAJ["evo_scored"] += 1
    return total


# --- attacker protection (the bench-out loss mode) --------------------------
# `data/analysis/DECK_REFINEMENT.md`: 68% of this deck's losses are "no active
# Pokemon" — we run out of promotable bodies and the game ends with prizes
# still on both sides. The list carries four Teal Mask Ogerpon ex and nothing
# else that can attack, so every Pokemon we lose is a quarter of the deck's
# whole capacity to play the game, and losing the last one loses the game
# outright regardless of the prize count.
#
# The evaluator scored HP and bench width, both of which are averages over
# bodies. Neither says which of our Pokemon dies on their next turn. This
# does: `attackers_exposed` counts our attack-capable Pokemon whose remaining
# HP is at or under the damage the opponent can afford one of their own turns
# from now — the same `_threat_at` projection C2 already runs on their side,
# read at k=1 instead of k=TRAJ_K, so the feature costs nothing new.
#
# Exposure needs reach as well as damage. Our Active is always reachable.
# A benched Pokemon is reachable only if their deck can pull it into the
# Active Spot, which is a property of their archetype and is read off
# `postures.json`'s `gust_reach` — the play-weighted majority over the mined
# lists of whether the archetype runs a card whose text switches in one of the
# opponent's Benched Pokemon (`ptcg/matchup_postures.py`). The measurement
# came back unanimous: all eleven archetypes in the prior run a gust, Boss's
# Orders being a field staple, so in the current metagame the bench is always
# reachable. The table ships anyway because that is a fact about this
# metagame and not an assumption, and an archetype the table does not name
# falls back to the same majority.
#
# Both of this section's behaviours carry an off switch, read from the
# environment the way SEARCH_OPP_BRANCH is, so a gate can run either one
# without the other against the same baseline binary. Both are 0, which is
# where the gate left them.
#
# Against 89995b7 on mirror decks with seats swapped, seed block 94000 —
# 500 Marnie's Grimmsnarl plus 300 Cynthia's Garchomp an arm:
#
#   postures alone     0.4923 over 784 decided (Marnie 0.483/484,
#                                               Garchomp 0.507/300)
#   protection alone   0.4918 over 791         (0.491/491, 0.493/300)
#   both               0.4893 over 791         (0.499/491, 0.473/300)
#
# The mirror is the wrong instrument for a matchup feature and we knew it
# going in — both seats play one deck, so a deny-posture fires against an
# opponent playing our own archetype, which is one of the two archetypes
# POSTURES.md says has no lever at all. So the arm that was supposed to
# decide this was the specialist: our own Ogerpon list against the harvested
# Grimmsnarl control agent playing the list it was written for
# (`scripts/specialist_gate.py`, 900 games an arm, seats swapped, seed block
# 94000). Reported over games neither side forfeited, because that agent's
# own last-resort fallback returns an empty selection whenever the prompt
# carries maxCount 0 and it hands over about a third of its games that way:
#
#   baseline 89995b7   0.7850 over 572 clean of 900   (0.8633 counting all)
#   postures alone     0.7986 over 576                 (0.8711)
#   protection alone   0.7757 over 602                 (0.8500)
#   both               0.7500 over 592                 (0.8356)
#
# Every one of those sits inside 1.5 standard errors of the control on a
# difference SE of about 2.4 points, and the sign of the best of them is not
# stable: at 300 games the both-arm read +8.1 points and at 900 it reads
# -3.5. That swing is the honest headline of this entry. Postures is the only
# arm whose point estimate is on the right side of the control in both the
# clean and the all-games counts, and +1.4 points is not a result.
#
# So neither behaviour ships on. The machinery stays in the tree exactly as
# playbook entry 4's reply tables did, and turning either back on is one
# environment variable — the day the leaf improves, or the day there is a
# specialist panel wide enough to resolve a two-point matchup effect.
try:
    PROTECTION_ENABLED = bool(int(os.environ.get("CABT_PROTECT") or 0))
except Exception:
    PROTECTION_ENABLED = False
try:
    POSTURE_MATCHUP_ENABLED = bool(int(os.environ.get("CABT_POSTURES") or 0))
except Exception:
    POSTURE_MATCHUP_ENABLED = False

POSTURE_FILES = ("postures.json",)
_POSTURES: dict | None = None
_POSTURES_SOURCE = ""
_GUST_REACH: dict = {}
_GUST_REACH_DEFAULT = True

# A last-attacker exposure is not a linear feature and is not priced as one.
# When our only remaining attack-capable Pokemon is inside their next-turn
# knockout range, the position is one attack from the loss mode that takes
# 68% of our games, and the alternatives available on the same turn — bench
# another body, retreat it out of reach — are worth taking almost regardless
# of what else is on the board. So it enters the margin as a named constant
# rather than a fitted coefficient, and it only ever changes a decision when
# a line exists that leaves it false: every candidate the search scores gets
# the same penalty when every candidate leads to the same exposure.
#
# Half a prize. Large enough to outrank playing a Basic to the bench (153) or
# any HP swing the term competes with, small enough that it never outbids the
# prize the exposure would cost us. Hand-set and screened in the gate arms
# below, never fitted — the fit sample cannot see the counterfactual.
LAST_ATTACKER_PENALTY = 500.0 if PROTECTION_ENABLED else 0.0

# The exposure COUNT was fitted and refused, and the refusal is the
# interesting half of this entry (`scripts/fit_protection_feature.py`,
# `data/analysis/protection_fit.json`, 8,792 episodes, one position each,
# rating >= 1000, turn fixed effects, on top of the six terms this evaluator
# already scores including C2).
#
# Pooled over the field the interval excludes zero — and does so with the
# wrong sign: +136 [+70, +202] entered alone, +166 [+91, +241] jointly, and
# +139 [+71, +207] with the seat's archetype held fixed, so it is not a deck
# confound. Positions where more of our attackers sit inside their next-turn
# knockout range are positions the winner is more often in. Shipping that
# number as a decision weight would pay the search to walk its Pokemon into
# range, which is the hand-differential mistake with a sign flip: a fitted
# advantage component is a description of positions winners reach, and it is
# only a decision weight if spending the resource produces the advantage it
# measures.
#
# On the one cell that describes the deck we actually play the interval
# covers zero and the point estimate turns over: Teal Mask Ogerpon ex seats,
# -213 [-721, +295] over 577 episodes. Narrow boards, the ones the rule below
# is about, are the same story: +175 [-403, +752] over 1,747. So the term
# ships at 0.0 as measured-null on our cell, the feature stays computed so
# new data can re-price it, and CABT_EXPOSED_W is how the pooled number gets
# measured on the board instead of argued about.
try:
    ATTACKERS_EXPOSED_WEIGHT = float(os.environ.get("CABT_EXPOSED_W") or 0.0)
except Exception:
    ATTACKERS_EXPOSED_WEIGHT = 0.0
if not PROTECTION_ENABLED:
    ATTACKERS_EXPOSED_WEIGHT = 0.0

TELEMETRY_PROT = {
    "scored": 0,            # positions the exposure count was computed on
    "exposed": 0,           # ... of which at least one attacker was exposed
    "last_attacker": 0,     # ... of which it was our last attack-capable one
}


def _posture_specs() -> dict:
    """{archetype: {"delta": weight deltas, "gust_rank": {card id: rank}}}."""
    global _POSTURES, _POSTURES_SOURCE, _GUST_REACH, _GUST_REACH_DEFAULT
    if _POSTURES is not None:
        return _POSTURES
    _POSTURES = {}
    blob, src = _read_json(_bundle_paths(
        POSTURE_FILES, ("data", "analysis", "postures.json")))
    for arch, spec in ((blob or {}).get("postures") or {}).items():
        try:
            delta = {k: float(v)
                     for k, v in (spec.get("weight_delta") or {}).items()}
            order = [int(c) for c in (spec.get("gust_order") or [])]
        except (TypeError, ValueError):
            continue
        if not delta and not order:
            continue          # a spec that says "no lever here" is not a rule
        _POSTURES[arch] = {"delta": delta,
                           "gust_rank": {c: i for i, c in enumerate(order)}}
    _GUST_REACH = {a: bool(v)
                   for a, v in ((blob or {}).get("gust_reach") or {}).items()}
    if blob is not None and blob.get("gust_reach_default") is not None:
        _GUST_REACH_DEFAULT = bool(blob["gust_reach_default"])
    if _POSTURES:
        _POSTURES_SOURCE = src
    else:
        TELEMETRY_POSTURE["specs_missing"] += 1
    return _POSTURES


def _attack_capable(card_id) -> bool:
    """Does this Pokemon print an attack at all?

    The bench-out mode is about bodies that could take over the Active Spot,
    not about who can pay for an attack this instant, so the test is what the
    card prints. On our own list every Pokemon passes it, which is the point:
    the four Ogerpon are the whole deck's capacity to play.
    """
    try:
        return bool(_attack_profile(int(card_id or 0)))
    except (TypeError, ValueError):
        return False


def _exposure(mine, incoming: float, bench_reachable: bool) -> tuple[int, int]:
    """(attackers inside their next-turn knockout range, attackers in play)."""
    exposed = capable = 0
    for zone, reachable in (("active", True), ("bench", bench_reachable)):
        for mon in _g(mine, zone, []) or []:
            if mon is None:
                continue
            if not _attack_capable(_g(mon, "id", 0)):
                continue
            capable += 1
            if reachable and float(_g(mon, "hp", 0) or 0) <= incoming:
                exposed += 1
    return exposed, capable


def _threat_and_exposure(cur, mine, theirs) -> tuple[float, int, int]:
    """(C2 differential, exposed attackers, attackers in play), one projection.

    Both features read the same `_traj_projection`, so scoring them together
    costs one projection rather than two — the C2 term at TRAJ_K of each
    side's own turns, the exposure at one of theirs.
    """
    (sm, gm, _), (st, gt, _) = _traj_projection(cur, mine, theirs)
    if EVO_INTEGRAL_ENABLED:
        # D35: the discounted evolution-aware integral replaces the point
        # term — same slot, same projection, and the weight swapped with it.
        thr = _threat_integral(mine, theirs, sm, gm, st, gt)
    else:
        thr = _threat_at(sm, gm, TRAJ_K) - _threat_at(st, gt, TRAJ_K)
    if not PROTECTION_ENABLED:
        return thr, 0, 0     # a refused feature costs no time
    incoming = _threat_at(st, gt, 1)
    _posture_specs()
    reach = _GUST_REACH.get(_TRAJ_ARCH["them"], _GUST_REACH_DEFAULT)
    exposed, capable = _exposure(mine, incoming, reach)
    return thr, exposed, capable


# --- matchup deny-postures (playbook: the D30 exploit half) -----------------
# `data/analysis/POSTURES.md` fitted, per top-8 archetype, which advantage
# component they convert to wins and which Pokemon on their board embody it.
# Our list can deny exactly one way — Boss's Orders on a cheap engine body,
# then knock it out — so each posture is two things: a raise to `bench`, which
# pays the search for any line that removes a body from their side, and the
# spec's named preference order over which body to gust.
#
# The weight raise is `20 x deny_pp_per_sd x lever_factor`
# (`ptcg/matchup_postures.py`), between +98 and +178 against a default bench
# weight of 153, and it is the only term any posture moves. The named order is
# the one thing a weight cannot express, because the evaluator has no
# per-target term: bench width scores their board getting smaller and cannot
# prefer Drakloak over Dreepy. It enters as a bonus in the card ranking, on
# opponent-owned cards only, so it can only ever reorder a choice among THEIR
# Pokemon — which is where gust selection happens and nowhere else.
#
# Activation is the posterior's own confidence gate, the same 0.80 the search
# uses to stop spending determinizations on alternative decklists: the top
# pick is right 84% of the time by turn 3 and 98% of the time when it claims
# 0.9, so the gate is what keeps a posture off a misidentified opponent.
#
# The named order REPLACES the generic score on a named target rather than
# adding to it, and the first version of this did add to it, which is the
# reason the distinction is written down. `_card_score` reads a Pokemon as
# hull plus offence, so among Grimmsnarl's four engine bodies it ranks
# Munkidori (110 HP, a real attack) about 120 clear of Impidimp (70 HP,
# almost none) — more than any per-rank step small enough to be called a
# tie-break could close. An additive bonus therefore preferred named targets
# over unnamed ones and then re-sorted the named ones by our own generic
# score, which is precisely the ordering the spec exists to overrule:
# Impidimp is the better gust BECAUSE it is the cheap body the 320 HP line
# is built on, not despite it. Replacing the score makes the spec's order the
# order, and the base sits above `BENCH_EMERGENCY` so no unnamed card can
# outbid it. It applies to opponent-owned cards outside a discard prompt,
# which in this pool means gust selection and the odd damage-move target.
POSTURE_GUST_BASE = 20000.0  # above every score _card_score can produce
POSTURE_GUST_STEP = 100.0    # one rank down the spec's preference order
_POSTURE_ON = {"arch": None, "delta": {}, "gust_rank": {}}

TELEMETRY_POSTURE = {
    "decisions": 0,          # agent calls the posture was re-evaluated on
    "active": 0,             # ... of which a matchup posture was on
    "by_archetype": {},      # ... broken out
    "gust_selects": 0,       # card selects holding a named target of theirs
    "gust_targeted": 0,      # options a named target's bonus was applied to
    "specs_missing": 0,      # postures.json did not load
}
_W_EFF: dict | None = None


def _refresh_weights() -> None:
    """Fold the live posture's deltas into the vectors `_margin` reads.

    The phase vectors compose the same way: phase override first (it replaces
    a cell), posture delta on top (it shifts one), so a posture means the same
    thing whichever phase the game is in.
    """
    global _W_EFF, _PHASE_W_EFF
    delta = _POSTURE_ON["delta"]

    def folded(base: dict) -> dict:
        if not delta:
            return base
        w = dict(base)
        for key, dv in delta.items():
            if key in w:
                w[key] = w[key] + dv
        return w

    _W_EFF = folded(WEIGHTS)
    _PHASE_W_EFF = tuple(folded(dict(WEIGHTS, **d)) for d in PHASE_DELTAS)


def _set_posture(label: str, confidence: float) -> None:
    """Turn the matchup posture for `label` on, off, or over to another one."""
    spec = None
    if POSTURE_MATCHUP_ENABLED and label and confidence >= CONFIDENCE_GATE:
        spec = _posture_specs().get(label)
    TELEMETRY_POSTURE["decisions"] += 1
    if spec is None:
        _POSTURE_ON.update(arch=None, delta={}, gust_rank={})
    else:
        _POSTURE_ON.update(arch=label, delta=spec["delta"],
                           gust_rank=spec["gust_rank"])
        TELEMETRY_POSTURE["active"] += 1
        by = TELEMETRY_POSTURE["by_archetype"]
        by[label] = by.get(label, 0) + 1
    _refresh_weights()


# Every number the position evaluator and the search override use, in one
# place so a tuner can inject a vector instead of editing code.
#
# These are fitted, not chosen. `ptcg/advantage.py` fits a logistic on the
# eventual win over 2,823 mined positions drawn one per episode from 14,172
# decided ladder games at rating >= 1000 (turns 3-11, turn fixed effects),
# and `data/analysis/REPORT.md` §A2 rescales the coefficients into these units
# by anchoring prize_diff at its hand-set 1000. Two fits are quoted below: the
# five-term fit over exactly what this evaluator used to score ("pooled"), and
# the nine-term fit that adds what it did not score ("richer"). Where they
# disagree the source is named per term.
#
#   prize   prizes are the win condition, so they set the scale — and the
#           anchor, so the rest of the vector is read against it. Fitted
#           1000 [819, 1181] by construction of the anchor.
#   hp      2.6 [2.0, 3.2] (pooled fit; was hand-set 1.0). The richer fit puts
#           it at 1.17 [0.52, 1.82] because damage_diff below takes part of
#           the same signal; 2.6 is what the evaluator's own five terms
#           support and is the number ratified for this build.
#   energy  kept at the hand-set 30. Pooled fit 37.2 [-25.4, 99.9], richer fit
#           70.0 [8.7, 131.4] — 30 sits inside both intervals, so the data
#           declines to move it and D30 discipline says leave it alone.
#   (no hand term)  The fit is unambiguous that holding more cards than they
#           do predicts winning — hand_diff 97 [40, 155] in the richer fit,
#           while the old own-hand coefficient turns over to -43.8
#           [-111.8, +24.1] once the differential is in the model. Shipping it
#           as a decision weight is what failed: at 97 a card is a tenth of a
#           prize to play, so the agent hoards its hand and stops converting
#           it. Measured against the 8bef47a agent on the rank-0 Marnie mirror,
#           posture off, hand_diff alone on top of hp 2.6: **0.360 over 494
#           decided games** [0.318, 0.403], and the whole richer vector with it
#           in scores 0.396 over 495. Every other new term wins the same test
#           (hp 2.6 alone 0.547, bench 0.537, damage 0.578; the old vector
#           minus its own-hand term 0.539), so the defect is this term and not
#           the fitting. The old +5-per-card own-hand term goes with it: it is
#           the same mistake at a smaller magnitude, and the fit says its sign
#           is wrong anyway. A fitted advantage component is a description of
#           positions winners reach; it is only a decision weight if spending
#           the resource does not produce the advantage it measures.
#   bench   153 [43, 263] (richer fit). Bench width was unscored; it carries
#           advantage the HP term does not, because a wide bench is what
#           survives a knockout.
#   damage  4.2 [2.5, 6.0] (richer fit), on damage already dealt (theirs minus
#           ours). Unscored before: HP alone cannot tell a fresh 60-HP Basic
#           from a 300-HP ex two hits from falling.
#   no_active  an empty Active Spot loses the game outright, so it keeps its
#           hand-set 4000 — the fit (354 [-674, +1382], richer 430
#           [-460, +1319]) is measuring mid-turn promote prompts, not a lost
#           game, and its interval is wide enough to say so. Every other term
#           is bounded: 6 prizes = 6000, six Pokemon at the pool's 380 max HP
#           = 5928 of HP and at most 9576 of damage, Energy, bench and hand
#           differentials under 2000 each, so the ceiling stays four orders of
#           magnitude clear of the ±1e6 a decided game scores.
#   search_margin  how far search must beat the rules pick before it overrides
#   threat_traj  the damage-potential differential TRAJ_K of each side's own
#           turns from now (playbook C2). Fitted by
#           `scripts/fit_trajectory_features.py` on the same sample and by the
#           same machinery as the vector above — 8,792 positions, one an
#           episode, rating >= 1000, turn fixed effects — entered on top of
#           exactly the terms this evaluator scores
#           (`data/analysis/trajectory_fit.json`). The weight is read out of
#           the horizon table below, so moving TRAJ_K moves the coefficient
#           with it and the evaluator never scores one horizon at another's
#           price. Shipped at k=3: 1.71 [0.76, 2.65], the best-fitting of the
#           four (k=2 1.42 [0.47, 2.36], k=1 1.03 [0.05, 2.01], present board
#           1.25 [0.15, 2.34]).
#           The projection is what carries it, not the board: the same
#           differential taken on the present board fits 1.25 alone and goes
#           null (0.48 [-0.82, 1.79]) once the t+2 version is in the model
#           beside it, while the t+2 term survives at 1.18 [0.04, 2.32]. That
#           contrast is the C2 result.
#   two trajectory features are measured-null and carry no weight, by D30:
#           online_lead (their earliest attack turn minus ours) 31 [-16, +78]
#           in the joint fit and 40 [-7, +86] alone; energy_traj_t2 (projected
#           Energy differential at t+2) -27 [-58, +3], correlated 0.68 with
#           the Energy term already in the vector. Both stay computable in
#           `_trajectory_terms` so they can be re-priced on new data.
#   attackers_exposed  how many of our attack-capable Pokemon sit inside the
#           damage the opponent can afford on their next turn, reach included
#           (`_exposure`). Fitted by `scripts/fit_protection_feature.py` on the
#           same sample and the same machinery as everything above it — one
#           position an episode, rating >= 1000, turn fixed effects, entered on
#           top of exactly the terms this evaluator scores including the
#           shipped C2 term. Measured-null on our own list and refused with
#           its sign inverted on the field; the argument is written out at
#           ATTACKERS_EXPOSED_WEIGHT above. The protection this entry does
#           ship is LAST_ATTACKER_PENALTY, which is not a slope.
WEIGHTS = {
    "prize": 1000.0,
    "hp": 2.6,
    "energy": 30.0,
    "bench": 153.0,
    "damage": 4.2,
    "no_active": 4000.0,
    # With CABT_EVO_INTEGRAL on, the slot holds the D35 replacement — the
    # evolution-aware discounted threat integral at its own fitted, anchored
    # coefficient (0.828; provenance at THREAT_INTEGRAL_WEIGHT) — because the
    # feature value `_threat_and_exposure` returns swaps with it and a
    # horizon-point coefficient priced at another feature's scale would be
    # exactly the drift the _THREAT_WEIGHT_BY_K table exists to prevent.
    "threat_traj": (THREAT_INTEGRAL_WEIGHT if EVO_INTEGRAL_ENABLED
                    else _THREAT_WEIGHT_BY_K.get(TRAJ_K, 1.42)),
    "attackers_exposed": ATTACKERS_EXPOSED_WEIGHT,
    "search_margin": 0.0,
}
_DEFAULT_WEIGHTS = dict(WEIGHTS)

# --- phase-conditional vectors (CABT_PHASE_WEIGHTS=1; playbook entry 8) -----
# The turn-checkpoint fits always said the coefficients drift across the game;
# the tree leaf (entry 7) was refused for capturing that drift with hundreds
# of unscreened local weights. This is the sanctioned form: a mixture of
# linear experts with a NAMED gate, two phases, each phase the same vector
# shape as WEIGHTS and every cell under the same citation discipline.
#
# The gate is the leading side's prize pile — min(ours, theirs) remaining —
# because that is the candidate that transferred: fitted per-phase on
# 07-31..08-05 and scored on the held-out day 08-06
# (`scripts/fit_phase_weights.py`, `data/analysis/phase_weights_fit.json`),
# min-prizes >= 5 vs <= 4 beats the global fit by +4.9 held-out log-lik over
# 1,286 episodes, ahead of every prizes-total band, while every turn band
# transfers at or below the global fit. Three phases lost to two on the same
# instrument (the 3-band min gate reads -38.9 against global).
#
# Per phase, the exact v2 rule the WEIGHTS block above documents: the
# incumbent value stands unless the phase's fitted 95% interval excludes it.
# Three cells moved, each cited (n = 6,621 / 2,171 episodes, prize anchored
# at 1000 per phase):
#   development (both piles >= 5):  hp 2.6 -> 1.13 [0.49, 1.78];
#     damage 4.2 -> 7.10 [5.00, 9.20]. Chip damage and the threat of the
#     first knockout are worth more before anyone has converted; raw HP
#     totals are worth less.
#   the race (leader <= 4):  bench 153 -> 11.66 [-100.30, 123.62] — the one
#     null that still moves, because the incumbent 153 sits OUTSIDE the
#     interval: once half a pile is gone, bench width has stopped predicting
#     the winner and the fit rejects paying 153 for it.
# Everything else — energy 30, threat_traj 1.71, no_active 4000 — stays: the
# phase intervals all contain the incumbent (threat fitted 3.18 [1.54, 4.81]
# early, null late; both contain 1.71, so the C2 term stays global).
try:
    PHASE_WEIGHTS_ENABLED = bool(int(os.environ.get("CABT_PHASE_WEIGHTS") or 0))
except ValueError:
    PHASE_WEIGHTS_ENABLED = False
PHASE_MIN_PRIZES = 5                    # phase 0 while min(piles) >= this
PHASE_DELTAS = (
    {"hp": 1.13, "damage": 7.1},        # development
    {"bench": 11.66},                   # the race
)
TELEMETRY_PHASE = {"development": 0, "race": 0}

_W_EFF = WEIGHTS
_PHASE_W_EFF = tuple(dict(WEIGHTS, **d) for d in PHASE_DELTAS)


def set_weights(weights: dict | None = None) -> None:
    """Re-point the eval weights; None restores the defaults.

    Unknown keys are ignored and missing keys keep their default, so a partial
    vector is legal and a bad one degrades rather than raises.
    """
    global WEIGHTS
    fresh = dict(_DEFAULT_WEIGHTS)
    for key, value in (weights or {}).items():
        if key in fresh:
            try:
                fresh[key] = float(value)
            except (TypeError, ValueError):
                pass
    WEIGHTS = fresh
    _refresh_weights()          # an injected vector is still the posture's base


def _margin(cur, me: int) -> float:
    """The evaluator's margin over a state, from seat `me`.

    One implementation over both observation forms (D25): the search scores
    dataclasses through `_evaluate`, the posture below reads the grader's dict,
    and every named term is the same arithmetic in both.

    The vector is `_W_EFF` rather than `WEIGHTS` because a matchup posture is
    a weight delta: `_set_posture` folds the live one in once a decision, so
    the leaf reads a plain dict and pays nothing for the mechanism.
    """
    w = _W_EFF or WEIGHTS
    players = _g(cur, "players", []) or []
    mine, theirs = players[me], players[1 - me]
    pz_m = len(_g(mine, "prize", []) or [])
    pz_t = len(_g(theirs, "prize", []) or [])
    # Phase-conditional vector: the named gate is the leading side's pile.
    # Both observation forms answer the same two length reads, so the rollout
    # and live policies see the same phase — the D25 invariant holds by the
    # same construction as every other term here.
    if PHASE_WEIGHTS_ENABLED:
        if min(pz_m, pz_t) >= PHASE_MIN_PRIZES:
            w = _PHASE_W_EFF[0]
            TELEMETRY_PHASE["development"] += 1
        else:
            w = _PHASE_W_EFF[1]
            TELEMETRY_PHASE["race"] += 1
    hp_m, en_m, bench_m, dmg_m = _side_totals(mine)
    hp_t, en_t, bench_t, dmg_t = _side_totals(theirs)
    # Our prize pile shrinking means we have been taking prizes.
    score = w["prize"] * (pz_t - pz_m)
    score += w["hp"] * (hp_m - hp_t)
    # Energy in play is board progress the rollout horizon usually cannot see.
    # Without this term two different attachment targets evaluate identically
    # unless one happens to enable a knockout this turn — which is why search
    # alone never fixed our worst decision. It matters most for an attacker
    # whose damage is a linear function of Energy, which ours is.
    score += w["energy"] * (en_m - en_t)
    score += w["bench"] * (bench_m - bench_t)
    # Damage already dealt: theirs minus ours, so a board we have chipped is
    # worth more than the same board at full HP.
    score += w["damage"] * (dmg_t - dmg_m)
    active = _g(mine, "active", []) or []
    if not (active and active[0]):
        score -= w["no_active"]
    # C2 (where the board is going) and the protection feature (which of our
    # bodies does not survive their next turn) read one projection between
    # them. Zero both weights with the last-attacker rule off and nothing is
    # projected at all, so a struck feature costs no time.
    if w["threat_traj"] or w["attackers_exposed"] or LAST_ATTACKER_PENALTY:
        try:
            thr, exposed, capable = _threat_and_exposure(cur, mine, theirs)
            TELEMETRY_TRAJ["threat_scored"] += 1
            if thr:
                TELEMETRY_TRAJ["threat_nonzero"] += 1
            score += w["threat_traj"] * thr
            if PROTECTION_ENABLED:
                TELEMETRY_PROT["scored"] += 1
                if exposed:
                    TELEMETRY_PROT["exposed"] += 1
                score += w["attackers_exposed"] * exposed
                # The whole deck's remaining capacity to play, one attack
                # from gone: the bench-out mode, priced as a posture and not
                # a slope.
                if capable == 1 and exposed == 1:
                    TELEMETRY_PROT["last_attacker"] += 1
                    score -= LAST_ATTACKER_PENALTY
        except Exception:
            TELEMETRY_TRAJ["feature_errors"] += 1
    return score


def _evaluate(observation, me: int) -> float:
    """Score a simulated position from our seat.

    Prizes dominate because prizes are the win condition; every other term is
    a fitted advantage component that points toward taking the next one.
    """
    cur = observation.current
    result = getattr(cur, "result", -1)
    if result is not None and result != -1:
        return 1e6 if result == me else -1e6
    if TREE_LEAF_ENABLED:
        try:
            raw = _tree_raw(cur, me)      # loads the forest, and TREE_SCALE
            return TREE_SCALE * raw
        except Exception:
            TELEMETRY_TREE["errors"] += 1
    return _margin(cur, me)


# --- the tree leaf (playbook: the D34 entry) --------------------------------
# D33 closed the 2-ply axis and named the defendant: three refusals of a deeper
# search, each one removing the excuse the last had used, with the leaf the
# only thing they had in common. The leaf was a linear margin over six terms.
# This is the sanctioned escalation — the SAME named features, scored by a
# gradient-boosted forest fitted on real ladder outcomes instead of by a dot
# product, so the leaf can hold "a bench body is worth more when you are two
# prizes down" without a hand-written interaction.
#
# Three constraints shaped what ships here.
#
#   * No new dependency on the grader. The forest is data — a JSON of nested
#     thresholds and leaf values — and the evaluator below is a tree walk in
#     the standard library, the same discipline the calibration table ships
#     under (C3).
#   * The linear evaluator stays the spine. `_margin` is untouched, and every
#     policy path that reads it — `_behind` through `_pwin`, the comeback
#     posture, the C3 verdict — still reads the linear number whether this
#     leaf is on or off. Only `_evaluate`, the thing the search calls on a
#     simulated position, swaps. That keeps the D25 invariant a mechanical
#     fact rather than a claim: turning the leaf on cannot move a rules
#     decision.
#   * The forest scores the same features the linear model was refused on.
#     `hand_diff`, `attackers_exposed`, `online_lead` and `energy_traj` were
#     all measured and all refused as WEIGHTS — but every one of those
#     refusals was a LINEAR pathology (a sign that inverts, a coefficient that
#     describes positions winners reach rather than resources worth spending).
#     A tree can hold a feature whose sign depends on the rest of the board,
#     which is exactly what those measurements said was happening. They enter
#     here on that argument and the gate decides.
#
# Output units. The forest predicts P(win); `_tree_raw` returns its log-odds,
# and TREE_SCALE puts that on the linear margin's scale. With
# WEIGHTS["search_margin"] at 0 the override test is invariant to any monotone
# rescaling, so the scale changes nothing about which candidate wins a
# comparison between two live positions. What it does govern is the trade
# against the +/-1e6 a decided determinization scores: a forest whose spread
# were 4 log-odds wide would let a single terminal determinization outvote
# every other one far more easily than the linear leaf does. TREE_SCALE is
# therefore set so the leaf's standard deviation over the fitted corpus
# matches the linear margin's, and that trade is left where the shipped
# evaluator had it. It is not a fitted quantity and does not pretend to be.
TREE_LEAF_NAME = os.environ.get("CABT_TREE_LEAF_FILE") or "tree_leaf.json"
TREE_LEAF_FILES = (TREE_LEAF_NAME,)
try:
    TREE_LEAF_ENABLED = bool(int(os.environ.get("CABT_TREE_LEAF") or 0))
except Exception:
    TREE_LEAF_ENABLED = False

TELEMETRY_TREE = {
    "scored": 0,          # positions the forest scored
    "errors": 0,          # guarded failures; the linear margin scored instead
    "missing": 0,         # the forest asked for and not found in the bundle
}

# The feature vector, in the order the forest's `feature` indices refer to.
# Every one is a named quantity this agent already computes or already has:
# the six shipped margin terms as differentials, the two prize piles apart so
# the forest can see "two down" and not only "two behind", the C2 projection,
# and the four named features the linear fits refused.
TREE_FEATURES = (
    "prize_diff",              # their prizes remaining - ours
    "prizes_left_me",
    "prizes_left_them",
    "hp_diff",
    "energy_diff",
    "bench_diff",
    "damage_diff",             # damage on their board - damage on ours
    "no_active_me",
    "threat_traj",             # C2, at TRAJ_K of each side's own turns
    "attackers_exposed",       # ours inside their next-turn knockout range
    "attackers_exposed_them",  # theirs inside ours, the mirror
    "hand_diff",
    "hand_me",
    "online_lead",             # their earliest attack turn minus ours
    "energy_traj",             # projected Energy differential at TRAJ_K
    "turn",
)


def _tree_features(cur, me: int) -> list[float]:
    """The forest's input vector over a position, from seat `me`.

    One implementation for both observation forms, exactly as `_margin` is:
    the search hands it dataclasses and the offline fit hands it the mined
    dicts, and `_g` makes those the same code. The offline fitter imports THIS
    function rather than reimplementing it, so a training-time feature and a
    run-time feature cannot drift apart.
    """
    players = _g(cur, "players", []) or []
    mine, theirs = players[me], players[1 - me]
    hp_m, en_m, bench_m, dmg_m = _side_totals(mine)
    hp_t, en_t, bench_t, dmg_t = _side_totals(theirs)
    prize_m = len(_g(mine, "prize", []) or [])
    prize_t = len(_g(theirs, "prize", []) or [])
    active = _g(mine, "active", []) or []

    thr = lead = e_traj = 0.0
    exposed = exposed_t = 0
    try:
        (sm, gm, e_m2), (st, gt, e_t2) = _traj_projection(cur, mine, theirs)
        thr = _threat_at(sm, gm, TRAJ_K) - _threat_at(st, gt, TRAJ_K)
        lead = float(_online_turn(st, gt) - _online_turn(sm, gm))
        e_traj = (e_m2 + gm(TRAJ_K)) - (e_t2 + gt(TRAJ_K))
        _posture_specs()
        reach_us = _GUST_REACH.get(_TRAJ_ARCH["them"], _GUST_REACH_DEFAULT)
        reach_them = _GUST_REACH.get(_TRAJ_ARCH["us"], _GUST_REACH_DEFAULT)
        exposed, _ = _exposure(mine, _threat_at(st, gt, 1), reach_us)
        exposed_t, _ = _exposure(theirs, _threat_at(sm, gm, 1), reach_them)
    except Exception:
        TELEMETRY_TRAJ["feature_errors"] += 1

    return [
        float(prize_t - prize_m),
        float(prize_m),
        float(prize_t),
        float(hp_m - hp_t),
        float(en_m - en_t),
        float(bench_m - bench_t),
        float(dmg_t - dmg_m),
        0.0 if (active and active[0]) else 1.0,
        float(thr),
        float(exposed),
        float(exposed_t),
        float((_g(mine, "handCount", 0) or 0) - (_g(theirs, "handCount", 0) or 0)),
        float(_g(mine, "handCount", 0) or 0),
        float(lead),
        float(e_traj),
        float(_g(cur, "turn", 1) or 0),
    ]


_TREE: dict | None = None
_TREE_SOURCE = ""
TREE_SCALE = 1.0


def _tree() -> dict:
    """The bundled forest: {"bias", "learning_rate", "scale", "trees": [...]}.

    Each tree is four parallel arrays over its nodes — `feature`, `threshold`,
    `left`, `right` — with a leaf marked by feature -1 and its value in
    `threshold`. Parallel arrays rather than nested dicts because the walk
    below is the hot path and an index is cheaper than an attribute.
    """
    global _TREE, _TREE_SOURCE, TREE_SCALE
    if _TREE is not None:
        return _TREE
    blob, src = _read_json(_bundle_paths(
        TREE_LEAF_FILES, ("data", "analysis", TREE_LEAF_NAME)))
    if not blob or not (blob.get("trees") or []):
        TELEMETRY_TREE["missing"] += 1
        _TREE = {}
        return _TREE
    # The forest's split indices point into `_tree_features`' vector by
    # position, so a forest fitted against a different feature order is not a
    # degraded leaf, it is a wrong one. Refuse it rather than score it.
    if list(blob.get("features") or ()) != list(TREE_FEATURES):
        TELEMETRY_TREE["missing"] += 1
        _TREE = {}
        return _TREE
    _TREE = blob
    _TREE_SOURCE = src
    try:
        TREE_SCALE = float(blob.get("scale") or 1.0)
    except Exception:
        TREE_SCALE = 1.0
    return _TREE


def _tree_raw(cur, me: int) -> float:
    """The forest's log-odds of winning this position, from seat `me`.

    Bias plus the learning rate times every tree's leaf, which is sklearn's
    own decision function for a log-loss gradient boosting ensemble. A missing
    forest returns 0.0, which `_evaluate` cannot tell from a dead-even
    position — so the falsification gate asserts the file loaded rather than
    trusting the value.
    """
    forest = _tree()
    trees = forest.get("trees") or ()
    if not trees:
        raise RuntimeError("no tree leaf bundled")
    x = _tree_features(cur, me)
    total = 0.0
    for feat, thr, left, right in trees:
        node = 0
        f = feat[node]
        while f >= 0:
            node = left[node] if x[f] <= thr[node] else right[node]
            f = feat[node]
        total += thr[node]
    TELEMETRY_TREE["scored"] += 1
    return forest["bias"] + forest["learning_rate"] * total


# --- calibrated win probability (playbook C3) -------------------------------
# The margin above is in the evaluator's own units, where "+1400" means
# nothing. `ptcg/creation/calibration.py` fits the monotone (isotonic) map from
# that margin to P(win) over self-play games and scores it on held-out games;
# the fitted table ships with the bundle so the agent can read its own verdict
# at run time instead of a judge reading it afterwards.
#
# The table is data, not code: rebuilt by
#   python -m ptcg.creation.calibration --games 3000 --out data/calibration_v2.json
# whenever WEIGHTS move, because a curve fitted to one weight vector says
# nothing about another.
CALIBRATION_FILES = ("calibration.json",)
_CALIB: list[tuple[float, float]] | None = None
_CALIB_SOURCE = ""

# Counted events, so nothing about this behaviour has to be taken on trust
# (D25: every fallback firing is a counted telemetry event).
TELEMETRY = {
    "main_decisions": 0,      # main-menu picks the posture was consulted on
    "posture_behind": 0,      # ... of which fired the behind-posture
    "calibration_missing": 0,  # _pwin asked for with no table loaded
}


def _calibration_paths() -> list[str]:
    paths = []
    for name in CALIBRATION_FILES:
        paths.append(name)
        if _HERE:
            paths.append(os.path.join(_HERE, name))
        paths.append(os.path.join(_KAGGLE_DIR, name))
    if _HERE:                       # in-repo runs read the fit where it lives
        paths.append(os.path.join(_HERE, os.pardir, "data",
                                  "calibration_v2.json"))
    return paths


def _load_calibration() -> list[tuple[float, float]]:
    """The isotonic breakpoints, as (margin, p) sorted by margin."""
    global _CALIB, _CALIB_SOURCE
    if _CALIB is not None:
        return _CALIB
    import json
    _CALIB = []
    for path in _calibration_paths():
        try:
            with open(path, "r") as fh:
                blob = json.load(fh)
        except Exception:
            continue
        curve = blob.get("curve") or []
        points = []
        for row in curve:
            try:
                points.append((float(row["margin"]), float(row["p"])))
            except Exception:
                continue
        if points:
            points.sort()
            _CALIB, _CALIB_SOURCE = points, path
            break
    return _CALIB


def _pwin(margin: float) -> float:
    """P(win) at this margin off the fitted table; 0.5 when it is missing.

    A step lookup, matching the isotonic fit exactly: the probability of the
    last breakpoint at or below the margin. Returning 0.5 on a missing table
    is the neutral answer — every posture keyed on it then stays off — and it
    is counted rather than silent.
    """
    curve = _load_calibration()
    if not curve:
        TELEMETRY["calibration_missing"] += 1
        return 0.5
    if margin < curve[0][0]:
        return curve[0][1]
    lo, hi = 0, len(curve) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if curve[mid][0] <= margin:
            lo = mid
        else:
            hi = mid - 1
    return curve[lo][1]


# --- comeback posture (playbook C4) -----------------------------------------
# What losing seats that came back actually did differently, from 14k rated
# ladder games (`data/analysis/comeback_B.json`, REPORT.md §B). Among seats
# whose fitted win probability was under 0.45 at turn 5 or 7, entering every
# behaviour at once on top of the position at t, the fitted deficit and the
# window length (n = 3,190 at t5 / 3,372 at t7, McFadden R² 0.12):
#
#   attack per turn   +1.21 [+0.88, +1.53] t5   +1.41 [+1.10, +1.71] t7
#   attach per turn   +1.09 [+0.81, +1.38] t5   +0.80 [+0.55, +1.05] t7
#   play per turn     -0.40 [-0.51, -0.29] t5   -0.59 [-0.69, -0.48] t7
#   ability per turn  -0.46 [-0.65, -0.26] t5   -0.51 [-0.70, -0.32] t7
#   retreat per turn  -1.15 [-1.54, -0.75] t5   -0.61 [-0.96, -0.26] t7
#
# The stratified rows hold deficit quintile and archetype fixed and the
# next-turn rates cut the window to the immediate reply; none of that
# identifies a treatment effect, and the causal caveat in REPORT.md §B stands.
# So this ships as a modest reordering of the main menu, not as a rewrite of
# it: close the game out (attack, attach) ahead of churning through it (play,
# ability), and retreat last of all. The two magnitudes hold the fitted 2:1
# ratio between the closing terms and the churn terms, small enough that only
# adjacent priorities trade places.
#
# Behind (p < 0.45) the menu order becomes
#   evolve 6.00 > ability 5.75 > attach 5.50 > attack 4.50 > play 3.75
#   > end 1.00 > retreat 0.75
# against the ahead-or-even order
#   ability 7 > evolve 6 > play 5 > attach 4 > attack 3 > retreat 2 > end 1.
POSTURE_ENABLED = True
BEHIND_PWIN = 0.45          # the deficit bin the comeback analysis conditions on
BEHIND_CLOSE_BONUS = 1.5    # attack and attach, the two positive terms
BEHIND_CHURN_PENALTY = 1.25  # play, ability, retreat, the three negative ones
POSTURE_DELTA = {
    OPT_ATTACK: +BEHIND_CLOSE_BONUS,
    OPT_ATTACH: +BEHIND_CLOSE_BONUS,
    OPT_PLAY: -BEHIND_CHURN_PENALTY,
    OPT_ABILITY: -BEHIND_CHURN_PENALTY,
    OPT_RETREAT: -BEHIND_CHURN_PENALTY,
}


def _behind(obs) -> bool:
    """Is the calibrated verdict on this position under the deficit bar?"""
    if not POSTURE_ENABLED:
        return False
    try:
        cur = _g(obs, "current")
        me = _g(cur, "yourIndex", 0)
        p = _pwin(_margin(cur, me))
    except Exception:
        return False
    TELEMETRY["main_decisions"] += 1
    if p < BEHIND_PWIN:
        TELEMETRY["posture_behind"] += 1
        return True
    return False


def _decided(cur) -> bool:
    res = getattr(cur, "result", -1)
    return res is not None and res != -1


def _greedy_complete(state, owner: int, rules_choice, deadline=None):
    """Step the rule policy for as long as it is `owner`'s turn."""
    from cg.api import search_step
    steps = 0
    while steps < SEARCH_ROLLOUT_STEPS:
        if deadline is not None and time.monotonic() > deadline:
            break                      # out of time: score where we stand
        o = state.observation
        cur = o.current
        if cur is None or _decided(cur):
            break
        sel = o.select
        if sel is None or not sel.option:
            break
        if cur.yourIndex != owner:
            break                      # the turn has handed over; stop here
        try:
            state = search_step(state.searchId, rules_choice(o))
        except Exception:
            break
        steps += 1
    return state


def _advance_forced(state, owner: int, rules_choice, deadline=None, limit=8):
    """Resolve `owner`'s forced sub-selects (promote after a knockout, set up
    an Active) so the position reaches their MAIN menu rather than stopping on
    a prompt that tells us nothing about how they will play."""
    from cg.api import search_step
    for _ in range(limit):
        if deadline is not None and time.monotonic() > deadline:
            break
        o = state.observation
        cur = o.current
        if cur is None or o.select is None or _decided(cur):
            break
        if cur.yourIndex != owner or o.select.context == CTX_MAIN:
            break
        try:
            state = search_step(state.searchId, rules_choice(o))
        except Exception:
            break
    return state


def _rules_order_for(observation) -> list[int]:
    """Option indices under the rule policy's priority, best first."""
    opts = _g(_g(observation, "select"), "option", []) or []

    def rank(i):
        return (-MAIN_PRIORITY.get(_g(opts[i], "type"), 0), i)

    return sorted(range(len(opts)), key=rank)


# --- the opponent reply model (playbook entry 4) ----------------------------
#
# Inside a rollout the opponent's turn used to be played by MAIN_PRIORITY —
# our own hand-set ordering, applied to a player who is not us. Both prior
# 2-ply rejections named that as the defect they could not separate from the
# branching itself (see SEARCH_OPP_BRANCH). This replaces it with the field's
# own ordering, counted.
#
# `data/opponent_policy.json` (scripts/build_opponent_policy.py) holds, for
# each (rating band, archetype, game-turn bucket, within-turn ordinal bucket),
# the frequency of each main-menu option type over 2,722,350 main-menu
# decisions from seven mined days, together with the rate at which each type
# was legal at all, counted on 7,366 complete menus sampled from the same
# games. The score a type gets here is the first divided by the second and
# renormalised over what the live menu offers, which is the only way to read a
# chosen-rate as a preference: attack is picked on 10.4% of main menus and is
# legal on roughly half of them, so its rate understates it by a factor of two.
#
# Temperature 0. The reply is the modal type of the cell, and the 2-ply branch
# takes the modal type and its runners-up in the same order — deterministic, so
# two searches of the same position agree and the decision-stability
# certificate still means something.
#
# Dormant at SEARCH_OPP_BRANCH = 0, which is where the gate left it: read the
# measurement block up at that constant before turning any of this back on.
OPPONENT_POLICY_FILES = ("opponent_policy.json",)
_OPP: dict | None = None
_OPP_SOURCE = ""
_OPP_AVAIL_FLOOR = 0.02   # a type legal in under 2% of menus divides by 0.02
OPP_TURN_BUCKETS = ((2, "1-2"), (5, "3-5"), (9, "6-9"), (10 ** 9, "10+"))
OPP_TYPE_NAME = {OPT_PLAY: "play", OPT_ATTACH: "attach", OPT_EVOLVE: "evolve",
                 OPT_ABILITY: "ability", OPT_RETREAT: "retreat",
                 OPT_ATTACK: "attack", OPT_END: "end_turn"}

TELEMETRY_OPP = {
    "policy_missing": 0,    # the table was asked for and is not loaded
    "consulted": 0,         # opponent main menus the table answered
    "hit_L0": 0,            # ... at band|archetype|turn|ordinal
    "hit_L1": 0,            # ... backed off to archetype|turn|ordinal
    "hit_L2": 0,            # ... backed off to turn|ordinal
    "hit_L3": 0,            # ... backed off to ordinal
    "miss": 0,              # no cell at any level: MAIN_PRIORITY decided
    "branch_replies": 0,    # 2-ply replies drawn from the table's ordering
    "searches": 0,          # completed searches over a deduped candidate set
    "overrides": 0,         # ... of which moved the pick off the rules answer
}

# Which band's behaviour to roll the opponent out under. One bundled constant,
# not a per-game inference: we cannot see an opponent's rating from inside an
# episode, and the ladder pairs us near our own. The file's `default_band` is
# the band holding the corpus median (1026) and its interquartile range
# (987-1076) — the opponents we actually meet.
OPP_BAND_ENV = os.environ.get("CABT_OPP_BAND") or ""


def _opp_policy() -> dict:
    global _OPP, _OPP_SOURCE
    if _OPP is not None:
        return _OPP
    blob, src = _read_json(_bundle_paths(
        OPPONENT_POLICY_FILES, ("data", "opponent_policy.json")))
    _OPP = blob if isinstance(blob, dict) and blob.get("cells") else {}
    if _OPP:
        _OPP_SOURCE = src
    return _OPP


def _opp_band() -> str:
    return OPP_BAND_ENV or _opp_policy().get("default_band", "900-1050")


def _opp_turn_bucket(turn) -> str:
    try:
        t = int(turn or 1)
    except (TypeError, ValueError):
        t = 1
    for hi, name in OPP_TURN_BUCKETS:
        if t <= hi:
            return name
    return "10+"


def _opp_arch() -> str:
    """The opponent's archetype as the table names it, or OTHER."""
    label = _TRAJ_ARCH.get("them", "(all)")
    return label if label in (_opp_policy().get("archetypes") or ()) else "OTHER"


def _opp_cell(tb: str, ob: str) -> dict:
    """The finest cell that exists, backing off band then archetype then turn."""
    pol = _opp_policy()
    cells = pol.get("cells") or {}
    arch = _opp_arch()
    for level, key in (("L0", "L0|%s|%s|%s|%s" % (_opp_band(), arch, tb, ob)),
                       ("L1", "L1|%s|%s|%s" % (arch, tb, ob)),
                       ("L2", "L2|%s|%s" % (tb, ob)),
                       ("L3", "L3|%s" % ob)):
        cell = cells.get(key)
        if cell:
            TELEMETRY_OPP["hit_" + level] += 1
            return cell.get("p") or {}
    TELEMETRY_OPP["miss"] += 1
    return {}


def _opp_avail(tb: str, ob: str) -> dict:
    av = _opp_policy().get("availability") or {}
    for key in ("%s|%s" % (tb, ob), ob):
        rec = av.get(key)
        if rec and rec.get("rate"):
            return rec["rate"]
    return {}


def _opp_type_scores(turn, ordinal: int) -> dict:
    """type -> propensity, for every type the corpus ever saw chosen here."""
    tb = _opp_turn_bucket(turn)
    ob = str(ordinal) if ordinal < 3 else "3+"
    chosen = _opp_cell(tb, ob)
    if not chosen:
        return {}
    avail = _opp_avail(tb, ob)
    out = {}
    for name, p in chosen.items():
        out[name] = p / max(avail.get(name, 1.0), _OPP_AVAIL_FLOOR)
    return out


def _opp_order(observation, turn, ordinal: int) -> list[int]:
    """One option index per available type, best-supported type first.

    Within a type the sub-choice is the rule policy's: the cheapest knockout or
    the biggest hit for an attack, the Active for an attach, otherwise the
    first option of that type — the same tie-break `_choose_main` applies, so
    what the table changes is which *kind* of move the opponent makes.

    Both sub-choices were measured on the same corpus and neither moved. The
    field takes the cheapest available knockout on 22 of 26 menus that offered
    one (0.846) and some knockout on 25 of 26, which is the rule already here;
    its 0.583 rate of taking the hardest hit is over 84 menus, too few to ship
    against. The attach target is the honest split: 905 menus with more than
    one attach went to the Bench 54.1% of the time against the Active's 45.9%,
    and a modal read of a 46/54 count is a coin flip dressed as a behaviour, so
    the Active rule stays. Both numbers are in
    `data/opponent_policy.json` under `attack_profile` and `attach_profile`.
    """
    opts = _g(_g(observation, "select"), "option", []) or []
    scores = _opp_type_scores(turn, ordinal)
    if not scores:
        return []
    reps: dict = {}
    for i, o in enumerate(opts):
        t = _g(o, "type")
        name = OPP_TYPE_NAME.get(t)
        if name is None or name in reps:
            continue
        if t == OPT_ATTACK:
            alt = _choose_attack(opts, observation)
        elif t == OPT_ATTACH:
            alt = _choose_attach(opts, observation)
        else:
            alt = i
        reps[name] = i if alt is None else alt
    if not reps:
        return []
    order = sorted(reps, key=lambda n: (-scores.get(n, 0.0), n))
    return [reps[n] for n in order]


# The opponent's turn is a sequence of main-menu visits and the table is cut by
# where in that sequence we are, so the rollout counts them. Reset explicitly
# at each branch rather than inferred from the turn number: the same turn is
# replayed once per candidate and once per determinization.
_OPP_ORDINAL = 0
_SEARCH_ME: int | None = None


def _opp_reset(k: int = 0) -> None:
    global _OPP_ORDINAL
    _OPP_ORDINAL = k


def _opp_choice(observation) -> list[int] | None:
    """The measured field reply, or None to let the rule policy answer."""
    global _OPP_ORDINAL
    if not _opp_policy():
        TELEMETRY_OPP["policy_missing"] += 1
        return None
    sel = _g(observation, "select")
    if _g(sel, "context") != CTX_MAIN:
        return None
    cur = _g(observation, "current")
    order = _opp_order(observation, _g(cur, "turn", 1), _OPP_ORDINAL)
    _OPP_ORDINAL += 1
    if not order:
        return None
    TELEMETRY_OPP["consulted"] += 1
    return [order[0]]


def _opp_seat(observation) -> bool:
    """Is this position the opponent's move inside our own search?"""
    if _SEARCH_ME is None:
        return False
    cur = _g(observation, "current")
    if cur is None:
        return False
    return _g(cur, "yourIndex", _SEARCH_ME) != _SEARCH_ME


def _rollout_value(state, me: int, rules_choice, deadline=None) -> float:
    """Finish our turn, then score the worst of the opponent's likely replies.

    Scoring the position we hand over rewards moves that look good only until
    the opponent answers them; taking the minimum over their top replies is
    what makes the search value a move rather than a snapshot.
    """
    state = _greedy_complete(state, me, rules_choice, deadline)
    o = state.observation
    cur = o.current
    if (cur is None or _decided(cur) or o.select is None
            or cur.yourIndex == me or SEARCH_OPP_BRANCH < 1):
        return _evaluate(o, me)

    state = _advance_forced(state, 1 - me, rules_choice, deadline)
    o = state.observation
    cur = o.current
    if (cur is None or _decided(cur) or o.select is None
            or cur.yourIndex == me or o.select.context != CTX_MAIN):
        return _evaluate(o, me)

    from cg.api import search_step
    branch_id = state.searchId
    # The replies are the field's, in the field's order: the modal main-menu
    # type of the cell first, its runners-up behind it. `_rules_order_for` —
    # our own priority table standing in for theirs — is the fallback for a
    # position no cell covers, and every use of it is counted as a miss.
    order = _opp_order(o, _g(_g(o, "current"), "turn", 1), 0)
    if order:
        TELEMETRY_OPP["branch_replies"] += 1
    else:
        order = _rules_order_for(o)
    worst = None
    for k in range(min(SEARCH_OPP_BRANCH, len(order))):
        if deadline is not None and time.monotonic() > deadline:
            break
        try:
            reply = search_step(branch_id, [order[k]])
        except Exception:
            continue
        # That reply was the turn's first main decision; the rest of their turn
        # continues from ordinal 1.
        _opp_reset(1)
        reply = _greedy_complete(reply, 1 - me, rules_choice, deadline)
        # our own forced replacement after their attack
        reply = _advance_forced(reply, me, rules_choice, deadline, limit=6)
        v = _evaluate(reply.observation, me)
        worst = v if worst is None else min(worst, v)
    return _evaluate(o, me) if worst is None else worst


# Options that differ only in which copy of a card they spend are the same
# move. Attaching any of three identical Energy to the Active is one candidate,
# not three: searching them separately spends three shares of a fixed
# determinization budget on one line, and the fair-sample rule then compares a
# line sampled three ways against one sampled once.
#
# The key is deliberately narrow. It collapses only moves that name both the
# card being spent and the target it goes to, so two same-named Pokemon already
# in play — which differ in damage taken, Energy attached and tools held — are
# never treated as interchangeable.
_DEDUP_TYPES = (OPT_ATTACH, OPT_EVOLVE, OPT_PLAY)


def _dedup_key(obs, opt):
    """A key equal for interchangeable options, or None to never collapse."""
    t = _g(opt, "type")
    if t not in _DEDUP_TYPES:
        return None
    cid = _g(_option_card(obs, opt), "id")
    if cid is None and t == OPT_PLAY:
        # PLAY carries an index into the hand and no area (cg/api.py).
        me = _g(_g(obs, "current"), "yourIndex", 0)
        hand = _zone(obs, AREA_HAND, me)
        idx = _g(opt, "index")
        if idx is not None and idx < len(hand):
            cid = _g(hand[idx], "id")
    if cid is None:
        cid = _g(opt, "cardId")
    if cid is None:
        return None          # unresolvable card: keep it as its own candidate
    return (t, cid, _g(opt, "inPlayArea"), _g(opt, "inPlayIndex"))


def _dedup_options(obs, options) -> tuple[list[int], dict]:
    """(candidate indices, option index -> the candidate that stands for it)."""
    groups: dict = {}
    rep: dict = {}
    cand: list[int] = []
    for i, o in enumerate(options):
        try:
            key = _dedup_key(obs, o)
        except Exception:
            key = None
        j = groups.get(key) if key is not None else None
        if j is not None:
            rep[i] = j
            continue
        if key is not None:
            groups[key] = i
        rep[i] = i
        cand.append(i)
    return cand, rep


def _search_main(obs: dict, options: list[dict]) -> int | None:
    """1-ply search over N determinizations; None means "rules policy decides".

    Each option's value is averaged over the determinization plan, weighted by
    posterior mass, instead of committing to the single most likely opponent
    list. Searching a point estimate is Perfect Information Monte Carlo, and it
    is over-confident exactly when the inference is weakest — early, before they
    have shown much. Weighting over the surviving decklists is the
    determinization step of Information Set MCTS.

    Search only overrides the rule policy when it can show the configured
    margin, and only against candidates it sampled as often as the rules pick —
    an unfair sample is how a noisy average steals a decision.
    """
    if not SEARCH_ENABLED or not obs.get("search_begin_input"):
        return None
    if not CG_AVAILABLE:
        return None
    budget = _decision_budget(obs)
    if budget <= 0.0:
        return None            # bank spent; the rules policy plays it out

    try:
        o = to_observation_class(obs)
        me = o.current.yourIndex
        rules_i = _choose_main(options, obs)
        seen, n_deck, n_hand, n_prize = _opponent_counts(obs)
        posterior = _deck_posterior(obs, top_k=POSTERIOR_TOP_K)
        # Once the posterior is confident it is right ~98% of the time, so the
        # extra candidate decklists buy nothing and cost 3x. Spend them only in
        # the early window where the top pick is genuinely unreliable.
        if posterior and posterior[0][1] >= CONFIDENCE_GATE:
            posterior = posterior[:1]
    except Exception:
        return None

    if not posterior:
        return None

    # Fixed budget, reallocated rather than multiplied: confident about the
    # decklist means spending all of it on shuffles of that one list; uncertain
    # means splitting it across the candidate lists. Cost stays flat either way.
    # Deck identity is the easy half — 80% of misidentifications are between
    # sister lists that play identically — while how the unseen cards split into
    # hand / deck / prizes is what genuinely differs game to game.
    per_deck = max(1, SEARCH_N_DET // max(len(posterior), 1))
    plan: list[tuple[dict, float]] = []
    for counts, weight in posterior:
        for _ in range(per_deck):
            plan.append((counts, weight / per_deck))

    # One deadline governs every loop below, so running short of time costs
    # determinizations or candidates rather than producing a garbage answer.
    deadline = time.monotonic() + budget
    rng = _position_rng(obs)
    # Which seat is ours, for as long as this search runs: it is what tells
    # `_rules_choice_for` whether the position in front of it is ours to play
    # under the live policy or theirs to play under the field table.
    global _SEARCH_ME
    _SEARCH_ME = me
    try:
        cand, rep = _dedup_options(obs, options)
        rules_i = rep.get(rules_i, rules_i)
    except Exception:
        cand = list(range(len(options)))
    acc = {i: 0.0 for i in cand}       # posterior-weighted value
    mass = {i: 0.0 for i in cand}      # weight actually spent on i
    n_eval = {i: 0 for i in cand}      # determinizations i survived
    try:
        for counts, weight in plan:
            if time.monotonic() > deadline:
                break
            try:
                my_deck, my_prize = _own_hidden(obs, rng)
                opp_deck, opp_hand, opp_prize = _hidden_from(
                    counts, seen, n_deck, n_hand, n_prize, rng)
                root = search_begin(o, my_deck, my_prize, opp_deck, opp_prize,
                                    opp_hand, [])
            except Exception:
                continue   # one bad decklist, not a dead search
            try:
                for i in cand:
                    if time.monotonic() > deadline:
                        break
                    try:
                        child = search_step(root.searchId, [i])
                        v = _rollout_value(child, me, _rules_choice_for,
                                           deadline)
                    except Exception:
                        continue   # one bad candidate, not a dead search
                    acc[i] += weight * v
                    mass[i] += weight
                    n_eval[i] += 1
            finally:
                try:
                    search_end()
                except Exception:
                    pass

        n_top = n_eval.get(rules_i, 0)
        if not n_top:
            return None
        evaluated = [i for i in cand if n_eval[i] == n_top and mass[i] > 0.0]
        if rules_i not in evaluated:
            return None
        avg = {i: acc[i] / mass[i] for i in evaluated}
        best = max(evaluated, key=lambda i: avg[i])
        # The override rate is the diagnostic that read D27: a change that
        # leaves it where it was has not changed what the search believes.
        TELEMETRY_OPP["searches"] += 1
        if best == rules_i:
            return None
        if avg[best] < avg[rules_i] + WEIGHTS["search_margin"]:
            return None
        TELEMETRY_OPP["overrides"] += 1
        return best
    except Exception:
        return None
    finally:
        _SEARCH_ME = None


def _rules_choice_for(observation) -> list[int]:
    """The rollout policy, one seat each.

    Our seat plays the live policy — D25's invariant, unchanged and still
    asserted by `scripts/validate_invariant.py`. The opponent's seat plays the
    counted field table (playbook entry 4); when no cell covers the position it
    falls back to the same live policy, and the fallback is counted.
    """
    try:
        if _opp_seat(observation):
            picked = _opp_choice(observation)
            if picked is not None:
                return picked
        return _policy(observation)
    except Exception:
        return [0]            # a rollout must never raise; scoring goes on


def _agent(obs_dict: dict) -> list[int]:
    obs = obs_dict
    select = obs.get("select")
    if select is None:
        return read_deck_csv()

    # Both archetype labels, re-read once here and held for every leaf the
    # search scores below (they are per-episode constants).
    _refresh_traj_arch(obs)

    # The main menu: simulate the candidates when we can, fall back to rules.
    options = select.get("option") or []
    if len(options) > 1 and any(o.get("type") in MAIN_PRIORITY
                                for o in options):
        picked = _search_main(obs, options)
        if picked is not None:
            return [picked]
    return _policy(obs)


def agent(obs_dict: dict) -> list[int]:
    """Never raise on a play decision — an exception forfeits the game.

    Defined last on purpose: kaggle_environments binds the *last* callable in
    the exec'd file as the entrypoint, so anything below this line would be
    called instead and this guard would never run.

    Also the one place the episode time bank is kept: reset on the
    deck-selection call that opens every episode, and charged the measured
    wall time of every call, so what `_decision_budget` divides up is what the
    grader will actually still be holding.
    """
    global _bank_remaining, _bank_decisions
    started = time.monotonic()
    try:
        try:
            if (obs_dict or {}).get("select") is None:
                _reset_bank()
        except Exception:
            pass
        return _agent(obs_dict)
    except Exception:
        sel = (obs_dict or {}).get("select")
        if sel is None:
            raise  # the deck must load; failing loudly here is correct
        n = len(sel.get("option") or [1])
        k = max(1, min(sel.get("minCount", 1) or 1, n))
        return list(range(k))
    finally:
        try:
            _bank_remaining -= time.monotonic() - started
            _bank_decisions += 1
        except Exception:
            pass
