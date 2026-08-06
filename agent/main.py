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

    # The main menu.
    if any(o.get("type") in MAIN_PRIORITY for o in options):
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
