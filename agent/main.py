"""Heuristic Pokémon TCG agent.

Submission entry point: Kaggle calls `agent(obs_dict)` and expects a list of
option indices. The very first call has `obs.select == None` and must return the
60-card deck.

The policy is deliberately simple and readable, because the Strategy writeup has
to explain it: within a turn, take every free action before the one that ends
the turn. Concretely — use Abilities, evolve, play cards, attach Energy, and
only then attack, choosing the attack that does the most damage.
"""

from __future__ import annotations

import json
import os
import random

# OptionType values (see cg/api.py). Inlined so the agent has no import cost
# and no dependency on the dataclass layer.
OPT_NUMBER, OPT_YES, OPT_NO, OPT_CARD = 0, 1, 2, 3
OPT_TOOL_CARD, OPT_ENERGY_CARD, OPT_ENERGY = 4, 5, 6
OPT_PLAY, OPT_ATTACH, OPT_EVOLVE, OPT_ABILITY = 7, 8, 9, 10
OPT_DISCARD, OPT_RETREAT, OPT_ATTACK, OPT_END = 11, 12, 13, 14
OPT_SKILL, OPT_SPECIAL_CONDITION = 15, 16

# SelectContext values used below.
CTX_SETUP_ACTIVE, CTX_SETUP_BENCH = 1, 2
CTX_IS_FIRST, CTX_MULLIGAN, CTX_ACTIVATE = 41, 42, 43

# Attacking ends the turn, so everything free happens first. Higher acts sooner.
MAIN_PRIORITY = {
    OPT_ABILITY: 7,
    OPT_EVOLVE: 6,
    OPT_PLAY: 5,
    OPT_ATTACH: 4,
    OPT_ATTACK: 3,
    OPT_RETREAT: 2,
    OPT_END: 1,
}

_ATTACK_DAMAGE: dict[int, int] | None = None


def attack_damage() -> dict[int, int]:
    """attackId -> printed damage, read from the engine's own table."""
    global _ATTACK_DAMAGE
    if _ATTACK_DAMAGE is None:
        try:
            from cg.sim import lib
            rows = json.loads(lib.AllAttack().decode())
            _ATTACK_DAMAGE = {r["attackId"]: r.get("damage") or 0 for r in rows}
        except Exception:
            _ATTACK_DAMAGE = {}
    return _ATTACK_DAMAGE


def read_deck_csv() -> list[int]:
    path = "deck.csv"
    if not os.path.exists(path):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deck.csv")
    if not os.path.exists(path):
        path = "/kaggle_simulations/agent/deck.csv"
    with open(path, "r") as fh:
        text = fh.read()
    deck = [int(x) for x in text.replace(",", "\n").split() if x.strip()]
    if len(deck) != 60:
        raise ValueError(f"deck.csv must hold 60 card ids, got {len(deck)}")
    return deck


def _pokemon_in_play(obs: dict) -> list[dict]:
    cur = obs.get("current") or {}
    me = cur.get("yourIndex", 0)
    players = cur.get("players") or []
    if me >= len(players):
        return []
    p = players[me]
    return (p.get("active") or []) + (p.get("bench") or [])


def _best_attack(options: list[dict]) -> int | None:
    """Index of the highest-damage attack among MAIN options."""
    dmg = attack_damage()
    best, best_d = None, -1
    for i, o in enumerate(options):
        if o.get("type") != OPT_ATTACK:
            continue
        d = dmg.get(o.get("attackId"), 0)
        if d > best_d:
            best, best_d = i, d
    return best


def _choose_main(options: list[dict]) -> int:
    """One action from the main menu, by the free-actions-first rule."""
    best_i, best_rank = 0, -1
    for i, o in enumerate(options):
        rank = MAIN_PRIORITY.get(o.get("type"), 0)
        if rank > best_rank:
            best_i, best_rank = i, rank
    # Among equally-ranked attacks, take the hardest hitting one.
    if options[best_i].get("type") == OPT_ATTACK:
        alt = _best_attack(options)
        if alt is not None:
            return alt
    return best_i


def _choose_setup(options: list[dict], obs: dict) -> int:
    """Opening Active: the sturdiest Basic we can lead with."""
    try:
        from cg.sim import lib
        cards = {c["cardId"]: c for c in json.loads(lib.AllCard().decode())}
    except Exception:
        return 0
    best_i, best_hp = 0, -1
    for i, o in enumerate(options):
        cid = o.get("cardId")
        hp = (cards.get(cid) or {}).get("hp", 0) or 0
        if hp > best_hp:
            best_i, best_hp = i, hp
    return best_i


def agent(obs_dict: dict) -> list[int]:
    obs = obs_dict
    select = obs.get("select")
    if select is None:
        return read_deck_csv()

    options = select.get("option") or []
    if not options:
        return [0]

    ctx = select.get("context")
    min_count = select.get("minCount", 1) or 1
    max_count = select.get("maxCount", 1) or 1
    n = len(options)

    # Single-pick decisions we have an opinion about.
    if max_count == 1:
        types = {o.get("type") for o in options}
        if types <= {OPT_YES, OPT_NO}:
            want_yes = ctx != CTX_MULLIGAN  # never redraw a legal opening hand
            for i, o in enumerate(options):
                if (o.get("type") == OPT_YES) == want_yes:
                    return [i]
            return [0]
        if ctx == CTX_SETUP_ACTIVE:
            return [_choose_setup(options, obs)]
        if any(o.get("type") in MAIN_PRIORITY for o in options):
            return [_choose_main(options)]
        return [0]

    # Multi-pick: take the smallest legal set, lowest indices, no duplicates.
    k = max(min_count, 1)
    k = min(k, max_count, n)
    return list(range(k))


if __name__ == "__main__":
    random.seed(0)
    print(f"deck loads: {len(read_deck_csv())} cards")
    print(f"attack table: {len(attack_damage())} attacks")
