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

import os

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

_CARDS: dict | None = None
_ATTACKS: dict | None = None


def _tables() -> tuple[dict, dict]:
    """Card and attack metadata straight from the engine."""
    global _CARDS, _ATTACKS
    if _CARDS is None:
        try:
            from cg.api import all_attack, all_card_data
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
def _zone(obs: dict, area: int, player_index: int) -> list:
    """The list an option's (area, index) points into."""
    cur = obs.get("current") or {}
    players = cur.get("players") or []
    if area == AREA_DECK:
        return (obs.get("select") or {}).get("deck") or []
    if area == AREA_STADIUM:
        return cur.get("stadium") or []
    if area == AREA_LOOKING:
        return cur.get("looking") or []
    if player_index is None or player_index >= len(players):
        return []
    ps = players[player_index]
    return {
        AREA_HAND: ps.get("hand"),
        AREA_DISCARD: ps.get("discard"),
        AREA_ACTIVE: ps.get("active"),
        AREA_BENCH: ps.get("bench"),
        AREA_PRIZE: ps.get("prize"),
    }.get(area) or []


def _option_card(obs: dict, opt: dict) -> dict | None:
    """Resolve a CARD option to the actual card dict, or None if hidden.

    CARD options carry (area, index, playerIndex) — never a cardId — so
    scoring one means looking it up on the board first.
    """
    zone = _zone(obs, opt.get("area"), opt.get("playerIndex"))
    idx = opt.get("index")
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


def _card_score(card: dict) -> float:
    """How much we want this card in play.

    Offence and bulk are good; handing the opponent extra prizes is not, and
    that penalty is what keeps a Mega ex from being promoted on reflex.
    """
    if not isinstance(card, dict):
        return -1.0
    cid = card.get("id")
    if cid is None:
        return -1.0
    hp = card.get("hp")
    if hp is None:
        cards, _ = _tables()
        data = cards.get(cid)
        hp = getattr(data, "hp", 0) or 0
    return hp + 2.0 * _best_damage(cid) - 220.0 * (prize_value(cid) - 1)


# --- choosing an attack -----------------------------------------------------
def _opponent_active(obs: dict) -> dict | None:
    cur = obs.get("current") or {}
    me = cur.get("yourIndex", 0)
    players = cur.get("players") or []
    if len(players) < 2:
        return None
    act = players[1 - me].get("active") or []
    return act[0] if act and isinstance(act[0], dict) else None


def _damage_against(attack_id: int, target: dict | None) -> int:
    """Printed damage, doubled if the defender is weak to this attack's type."""
    cards, attacks = _tables()
    a = attacks.get(attack_id)
    if a is None:
        return 0
    dmg = getattr(a, "damage", 0) or 0
    if not dmg or not target:
        return dmg
    data = cards.get(target.get("id"))
    weakness = getattr(data, "weakness", None) if data else None
    if weakness is None:
        return dmg
    energies = [e for e in (getattr(a, "energies", None) or []) if e]
    if any(e == weakness for e in energies):
        return dmg * 2
    return dmg


def _choose_attack(options: list[dict], obs: dict) -> int | None:
    """Cheapest knockout if one exists, otherwise the biggest hit."""
    target = _opponent_active(obs)
    target_hp = target.get("hp") if target else None

    ko_i, ko_d = None, None
    big_i, big_d = None, -1
    for i, o in enumerate(options):
        if o.get("type") != OPT_ATTACK:
            continue
        d = _damage_against(o.get("attackId"), target)
        if d > big_d:
            big_i, big_d = i, d
        if target_hp is not None and d >= target_hp:
            if ko_d is None or d < ko_d:
                ko_i, ko_d = i, d
    return ko_i if ko_i is not None else big_i


def _choose_attach(options: list[dict], obs: dict) -> int | None:
    """Energy goes on whatever is going to attack — the Active Pokémon."""
    fallback = None
    for i, o in enumerate(options):
        if o.get("type") != OPT_ATTACH:
            continue
        if fallback is None:
            fallback = i
        if o.get("inPlayArea") == AREA_ACTIVE:
            return i
    return fallback


def _choose_main(options: list[dict], obs: dict) -> int:
    best_i, best_rank = 0, -1
    for i, o in enumerate(options):
        rank = MAIN_PRIORITY.get(o.get("type"), 0)
        if rank > best_rank:
            best_i, best_rank = i, rank

    kind = options[best_i].get("type")
    if kind == OPT_ATTACK:
        alt = _choose_attack(options, obs)
        if alt is not None:
            return alt
    elif kind == OPT_ATTACH:
        alt = _choose_attach(options, obs)
        if alt is not None:
            return alt
    return best_i


def _rank_card_options(options: list[dict], obs: dict) -> list[int]:
    """Option indices, best card first. Unresolvable cards sort last."""
    scored = []
    for i, o in enumerate(options):
        card = _option_card(obs, o)
        scored.append((_card_score(card) if card else -1.0, -i, i))
    scored.sort(reverse=True)
    return [i for _, _, i in scored]


# =========================================================================
# Forward search
#
# The engine can roll the game forward from the current position, but only if
# we supply the hidden information: the opponent's deck, hand and prizes. That
# would be hopeless guesswork except the mined replays show the whole metagame
# is ~120 decklists, and the 40 shipped in deck_priors.json cover 94% of it.
# So we identify which known list the opponent is playing, fill in the rest,
# and actually simulate our candidate turns instead of guessing at them.
# =========================================================================

_PRIORS: list[tuple[dict, int]] | None = None
_MY_DECK: list[int] | None = None
SEARCH_ROLLOUT_STEPS = 24     # enough to finish a turn; caps the cost
SEARCH_ENABLED = True


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


def _predict_opponent(obs: dict) -> tuple[list[int], list[int], list[int]]:
    """(deck, hand, prize) card ids for the opponent's hidden cards."""
    cur = obs.get("current") or {}
    me = cur.get("yourIndex", 0)
    players = cur.get("players") or []
    if len(players) < 2:
        return [], [], []
    opp = players[1 - me]
    n_deck = opp.get("deckCount", 0) or 0
    n_hand = opp.get("handCount", 0) or 0
    n_prize = len(opp.get("prize") or [])

    seen = _visible(opp, include_hand=False)
    best, best_plays = None, -1
    for counts, plays in _load_priors():
        if all(counts.get(cid, 0) >= n for cid, n in seen.items()):
            if plays > best_plays:
                best, best_plays = counts, plays

    hidden: list[int] = []
    if best is not None:
        for cid, n in best.items():
            hidden.extend([cid] * max(n - seen.get(cid, 0), 0))

    need = n_deck + n_hand + n_prize
    if len(hidden) < need:
        hidden.extend([3] * (need - len(hidden)))   # Basic Energy is safe filler
    hidden = hidden[:need]
    return (hidden[n_hand + n_prize:], hidden[:n_hand],
            hidden[n_hand:n_hand + n_prize])


def _own_hidden(obs: dict) -> tuple[list[int], list[int]]:
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
    hidden = hidden[:need]
    return hidden[n_prize:], hidden[:n_prize]


def _side_hp(player) -> float:
    total = 0.0
    for zone in ("active", "bench"):
        for mon in getattr(player, zone, None) or []:
            if mon is not None:
                total += getattr(mon, "hp", 0) or 0
    return total


def _evaluate(observation, me: int) -> float:
    """Score a simulated position from our seat.

    Prizes dominate because prizes are the win condition; board HP is the
    tie-breaker that points toward taking the next one.
    """
    cur = observation.current
    result = getattr(cur, "result", -1)
    if result is not None and result != -1:
        return 1e6 if result == me else -1e6
    mine = cur.players[me]
    theirs = cur.players[1 - me]
    # Our prize pile shrinking means we have been taking prizes.
    score = (len(theirs.prize or []) - len(mine.prize or [])) * 1000.0
    score += _side_hp(mine) - _side_hp(theirs)
    score += (getattr(mine, "handCount", 0) or 0) * 5.0
    return score


def _rollout_value(state, me: int, rules_choice) -> float:
    """Play our own turn out with the rule policy, then score the position."""
    from cg.api import search_step
    steps = 0
    while steps < SEARCH_ROLLOUT_STEPS:
        o = state.observation
        sel = o.select
        if sel is None or not sel.option:
            break
        if o.current.yourIndex != me:
            break                      # our turn is over; stop here
        try:
            state = search_step(state.searchId, rules_choice(o))
        except Exception:
            break
        steps += 1
    return _evaluate(state.observation, me)


def _search_main(obs: dict, options: list[dict]) -> int | None:
    """1-ply search: simulate each option, finish the turn, keep the best."""
    if not SEARCH_ENABLED or not obs.get("search_begin_input"):
        return None
    try:
        from cg.api import (search_begin, search_end, search_step,
                            to_observation_class)
    except Exception:
        return None

    try:
        o = to_observation_class(obs)
        me = o.current.yourIndex
        my_deck, my_prize = _own_hidden(obs)
        opp_deck, opp_hand, opp_prize = _predict_opponent(obs)

        root = search_begin(o, my_deck, my_prize, opp_deck, opp_prize,
                            opp_hand, [])
    except Exception:
        return None

    best_i, best_v = None, None
    try:
        for i in range(len(options)):
            try:
                child = search_step(root.searchId, [i])
            except Exception:
                continue
            v = _rollout_value(child, me, _rules_choice_for)
            if best_v is None or v > best_v:
                best_i, best_v = i, v
    finally:
        try:
            search_end()
        except Exception:
            pass
    return best_i


def _rules_choice_for(observation) -> list[int]:
    """Rule policy over a simulated Observation (dataclass, not dict)."""
    sel = observation.select
    opts = sel.option or []
    if not opts:
        return [0]
    types = {getattr(o, "type", None) for o in opts}
    if types <= {OPT_YES, OPT_NO}:
        for i, o in enumerate(opts):
            if getattr(o, "type", None) == OPT_YES:
                return [i]
        return [0]
    best_i, best_rank = 0, -1
    for i, o in enumerate(opts):
        rank = MAIN_PRIORITY.get(getattr(o, "type", None), 0)
        if rank > best_rank:
            best_i, best_rank = i, rank
    lo = max(getattr(sel, "minCount", 1) or 1, 1)
    hi = min(getattr(sel, "maxCount", 1) or 1, len(opts))
    if lo > 1:
        return list(range(min(lo, hi)))
    return [best_i]


def agent(obs_dict: dict) -> list[int]:
    """Never raise on a play decision — an exception forfeits the game."""
    try:
        return _agent(obs_dict)
    except Exception:
        sel = (obs_dict or {}).get("select")
        if sel is None:
            raise  # the deck must load; failing loudly here is correct
        n = len(sel.get("option") or [1])
        k = max(1, min(sel.get("minCount", 1) or 1, n))
        return list(range(k))


def _agent(obs_dict: dict) -> list[int]:
    obs = obs_dict
    select = obs.get("select")
    if select is None:
        return read_deck_csv()

    options = select.get("option") or []
    if not options:
        return [0]

    ctx = select.get("context")
    min_count = select.get("minCount", 1) or 0
    max_count = select.get("maxCount", 1) or 1
    n = len(options)
    types = {o.get("type") for o in options}

    # Yes/no questions.
    if types <= {OPT_YES, OPT_NO}:
        want_yes = ctx != CTX_MULLIGAN   # never redraw a legal opening hand
        for i, o in enumerate(options):
            if (o.get("type") == OPT_YES) == want_yes:
                return [i]
        return [0]

    # "How many?" — always take the most.
    if types == {OPT_NUMBER}:
        best_i, best_n = 0, -1
        for i, o in enumerate(options):
            v = o.get("number", 0) or 0
            if v > best_n:
                best_i, best_n = i, v
        return [best_i]

    # The main menu: simulate the candidates when we can, fall back to rules.
    if any(o.get("type") in MAIN_PRIORITY for o in options):
        if len(options) > 1:
            picked = _search_main(obs, options)
            if picked is not None:
                return [picked]
        return [_choose_main(options, obs)]

    # Card choices: put the most valuable card where we want it, and the least
    # valuable card where we don't.
    if OPT_CARD in types:
        order = _rank_card_options(options, obs)
        worst_first = ctx == CTX_DISCARD
        if worst_first:
            order = order[::-1]
        k = max(min_count, 1) if max_count >= 1 else min_count
        k = min(k, max_count, n)
        if ctx in (CTX_SETUP_BENCH, CTX_TO_BENCH):
            k = min(max_count, n)      # a full bench is always better
        return sorted(order[:max(k, 1)])

    # Anything else (energy picks, tools): smallest legal set.
    k = min(max(min_count, 1), max_count, n)
    return list(range(max(k, 1)))
