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


def _damage_against(attack_id: int, target) -> int:
    """Printed damage, doubled if the defender is weak to this attack's type."""
    cards, attacks = _tables()
    a = attacks.get(attack_id)
    if a is None:
        return 0
    dmg = getattr(a, "damage", 0) or 0
    if not dmg or target is None:
        return dmg
    data = cards.get(_g(target, "id"))
    weakness = getattr(data, "weakness", None) if data else None
    if weakness is None:
        return dmg
    energies = [e for e in (getattr(a, "energies", None) or []) if e]
    if any(e == weakness for e in energies):
        return dmg * 2
    return dmg


def _choose_attack(options, obs) -> int | None:
    """Cheapest knockout if one exists, otherwise the biggest hit."""
    target = _opponent_active(obs)
    target_hp = _g(target, "hp")

    ko_i, ko_d = None, None
    big_i, big_d = None, -1
    for i, o in enumerate(options):
        if _g(o, "type") != OPT_ATTACK:
            continue
        d = _damage_against(_g(o, "attackId"), target)
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
    """Option indices, best card first. Unresolvable cards sort last."""
    try:
        board = _board_context(obs)
    except Exception:
        board = None
    scored = []
    for i, o in enumerate(options):
        card = _option_card(obs, o)
        if card is not None:
            v = _card_score(card, board, _g(o, "area") == AREA_HAND)
        else:
            v = -1.0
        scored.append((v, -i, i))
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


def _traj_slots(player) -> list[tuple[float, int]]:
    """(Energy on it, card id) for every Pokemon this side has in play."""
    out = []
    for zone in ("active", "bench"):
        for mon in _g(player, zone, []) or []:
            if mon is None:
                continue
            cid = _g(mon, "id", 0) or 0
            if cid:
                out.append((float(len(_g(mon, "energies", []) or [])), int(cid)))
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
    for e, cid in slots:
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
    """The hardest attack this side can pay for k of its turns from now."""
    gain = growth(k)
    best = 0.0
    for e, cid in slots:
        budget = e + gain
        for cost, dmg in _attack_profile(cid):
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
    """
    try:
        cur = (obs or {}).get("current") or {}
        players = cur.get("players") or []
        if len(players) < 2:
            return
        me = cur.get("yourIndex", 0) or 0
        _TRAJ_ARCH["us"] = _label_of_ids(
            _visible_ids(players[me], ("active", "bench", "discard", "hand")))
        label = "(all)"
        try:
            post = _deck_posterior(obs, top_k=1)
            if post:
                label = _label_of_ids(post[0][0].keys())
        except Exception:
            label = "(all)"
        if label == "(all)":
            label = _label_of_ids(_visible_ids(players[1 - me]))
        _TRAJ_ARCH["them"] = label
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
    e_m = sum(e for e, _ in slots_m)
    e_t = sum(e for e, _ in slots_t)
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
WEIGHTS = {
    "prize": 1000.0,
    "hp": 2.6,
    "energy": 30.0,
    "bench": 153.0,
    "damage": 4.2,
    "no_active": 4000.0,
    "threat_traj": _THREAT_WEIGHT_BY_K.get(TRAJ_K, 1.42),
    "search_margin": 0.0,
}
_DEFAULT_WEIGHTS = dict(WEIGHTS)


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


def _margin(cur, me: int) -> float:
    """The evaluator's margin over a state, from seat `me`.

    One implementation over both observation forms (D25): the search scores
    dataclasses through `_evaluate`, the posture below reads the grader's dict,
    and every named term is the same arithmetic in both.
    """
    w = WEIGHTS
    players = _g(cur, "players", []) or []
    mine, theirs = players[me], players[1 - me]
    hp_m, en_m, bench_m, dmg_m = _side_totals(mine)
    hp_t, en_t, bench_t, dmg_t = _side_totals(theirs)
    # Our prize pile shrinking means we have been taking prizes.
    score = w["prize"] * (len(_g(theirs, "prize", []) or [])
                          - len(_g(mine, "prize", []) or []))
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
    # C2: where the board is going. Zero the weight and the projection is not
    # computed at all, so a struck feature costs no time.
    if w["threat_traj"]:
        try:
            thr = _threat_traj(cur, mine, theirs)
            TELEMETRY_TRAJ["threat_scored"] += 1
            if thr:
                TELEMETRY_TRAJ["threat_nonzero"] += 1
            score += w["threat_traj"] * thr
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
    return _margin(cur, me)


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
