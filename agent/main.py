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
    best_i, best_rank = 0, -1
    for i, o in enumerate(options):
        rank = MAIN_PRIORITY.get(_g(o, "type"), 0)
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
# Opponent replies minimized over at ply 2. Off, because playing their turn
# out costs 9 points of win rate against the same agent at 0 (40.7% over 200
# games) and 14 at one reply (36.2%) — a single reply with no minimum taken is
# the worse of the two, so what fails is the model of how they play, not the
# branching or the pessimism. A separate one-reply test on a single
# point-estimate determinization landed in the same place (0.467). The shuffled
# hand model has since made determinization honest without moving the number,
# so what is left to fix is the policy the opponent's turn is rolled out under —
# ours, on a guessed decklist. Turn this on once that is better.
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


def _side_hp(player) -> float:
    total = 0.0
    for zone in ("active", "bench"):
        for mon in getattr(player, zone, None) or []:
            if mon is not None:
                total += getattr(mon, "hp", 0) or 0
    return total


def _side_energy(player) -> float:
    total = 0.0
    for zone in ("active", "bench"):
        for mon in getattr(player, zone, None) or []:
            if mon is not None:
                total += len(getattr(mon, "energies", None) or [])
    return total


# Every number the position evaluator and the search override use, in one
# place so a tuner can inject a vector instead of editing code.
#
#   prize   prizes are the win condition, so they set the scale
#   hp      board HP, the tie-breaker that points at the next prize
#   energy  on the scale of HP rather than prizes: board progress the rollout
#           horizon cannot see, but never enough to outweigh a knockout
#   hand    cards in hand
#   no_active  an empty Active Spot loses the game outright. Every other term
#           is bounded — 6 prizes = 6000, six Pokemon at the pool's 380 max HP
#           = 2280, Energy and hand under 2000 each — so the 14380 ceiling
#           keeps ±1e6 for a decided game out of reach of any live position.
#   search_margin  how far search must beat the rules pick before it overrides
WEIGHTS = {
    "prize": 1000.0,
    "hp": 1.0,
    "energy": 30.0,
    "hand": 5.0,
    "no_active": 4000.0,
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


def _evaluate(observation, me: int) -> float:
    """Score a simulated position from our seat.

    Prizes dominate because prizes are the win condition; board HP is the
    tie-breaker that points toward taking the next one.
    """
    cur = observation.current
    result = getattr(cur, "result", -1)
    if result is not None and result != -1:
        return 1e6 if result == me else -1e6
    w = WEIGHTS
    mine = cur.players[me]
    theirs = cur.players[1 - me]
    # Our prize pile shrinking means we have been taking prizes.
    score = (len(theirs.prize or []) - len(mine.prize or [])) * w["prize"]
    score += w["hp"] * (_side_hp(mine) - _side_hp(theirs))
    # Energy in play is board progress the rollout horizon usually cannot see.
    # Without this term two different attachment targets evaluate identically
    # unless one happens to enable a knockout this turn — which is why search
    # alone never fixed our worst decision. It matters most for an attacker
    # whose damage is a linear function of Energy, which ours is.
    score += w["energy"] * (_side_energy(mine) - _side_energy(theirs))
    score += (getattr(mine, "handCount", 0) or 0) * w["hand"]
    active = getattr(mine, "active", None) or []
    if not (active and active[0]):
        score -= w["no_active"]
    return score


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
    order = _rules_order_for(o)
    worst = None
    for k in range(min(SEARCH_OPP_BRANCH, len(order))):
        if deadline is not None and time.monotonic() > deadline:
            break
        try:
            reply = search_step(branch_id, [order[k]])
        except Exception:
            continue
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
        if best == rules_i:
            return None
        if avg[best] < avg[rules_i] + WEIGHTS["search_margin"]:
            return None
        return best
    except Exception:
        return None


def _rules_choice_for(observation) -> list[int]:
    """Rule policy inside a rollout. The same function live play uses."""
    try:
        return _policy(observation)
    except Exception:
        return [0]            # a rollout must never raise; scoring goes on


def _agent(obs_dict: dict) -> list[int]:
    obs = obs_dict
    select = obs.get("select")
    if select is None:
        return read_deck_csv()

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
