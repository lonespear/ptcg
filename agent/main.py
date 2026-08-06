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

# The grader exec()s this file rather than importing it, so `__file__` does not
# exist at run time. Touching it raises NameError and the episode dies — this is
# exactly what failed the first two submissions, and it cannot reproduce locally
# because a normal import always defines `__file__`.
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = ""

# Where the competition mounts the agent bundle.
_KAGGLE_DIR = "/kaggle_simulations/agent"


def _candidate_paths(name: str) -> list[str]:
    paths = [name, os.path.join(_KAGGLE_DIR, name)]
    if _HERE:
        paths.insert(1, os.path.join(_HERE, name))
    return paths


_TABLES: dict | None = None


def _load_tables() -> dict:
    """Attack damage and card HP, baked in at build time.

    This deliberately does NOT import `cg`. The grader already has the engine
    loaded, and importing our own copy runs `lib.GameInitialize()` a second
    time against a live battle — which is what killed the first submission
    ("Validation Episode failed"). A local test cannot reproduce it, because
    there the module is already imported and the re-import is a no-op.
    """
    global _TABLES
    if _TABLES is None:
        for path in _candidate_paths("agent_data.json"):
            try:
                with open(path, "r") as fh:
                    raw = json.load(fh)
                _TABLES = {
                    "attack_damage": {int(k): v for k, v in
                                      raw.get("attack_damage", {}).items()},
                    "card_hp": {int(k): v for k, v in
                                raw.get("card_hp", {}).items()},
                }
                break
            except (OSError, ValueError):
                continue
        else:
            _TABLES = {"attack_damage": {}, "card_hp": {}}
    return _TABLES


def attack_damage() -> dict[int, int]:
    """attackId -> printed damage."""
    return _load_tables()["attack_damage"]


def card_hp() -> dict[int, int]:
    """cardId -> printed HP."""
    return _load_tables()["card_hp"]


def read_deck_csv() -> list[int]:
    text = None
    for path in _candidate_paths("deck.csv"):
        try:
            with open(path, "r") as fh:
                text = fh.read()
            break
        except OSError:
            continue
    if text is None:
        raise FileNotFoundError("deck.csv not found next to the agent")
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


def _opponent_active_hp(obs: dict) -> int | None:
    """Remaining HP of the Pokémon we would be attacking."""
    cur = obs.get("current") or {}
    me = cur.get("yourIndex", 0)
    players = cur.get("players") or []
    if len(players) < 2:
        return None
    active = players[1 - me].get("active") or []
    if not active:
        return None
    return active[0].get("hp")


def _best_attack(options: list[dict], obs: dict) -> int | None:
    """Pick the attack to use.

    A knockout is worth more than raw damage — it takes a prize and removes the
    threat — so the cheapest attack that knocks the target out wins. With no
    knockout available, fall back to the biggest hit.
    """
    dmg = attack_damage()
    target_hp = _opponent_active_hp(obs)

    ko_i, ko_d = None, None
    big_i, big_d = None, -1
    for i, o in enumerate(options):
        if o.get("type") != OPT_ATTACK:
            continue
        d = dmg.get(o.get("attackId"), 0) or 0
        if d > big_d:
            big_i, big_d = i, d
        if target_hp is not None and d >= target_hp:
            # Cheapest knockout = smallest damage that still finishes the job,
            # which keeps the harder-hitting attack available for later.
            if ko_d is None or d < ko_d:
                ko_i, ko_d = i, d
    return ko_i if ko_i is not None else big_i


def _best_attach(options: list[dict]) -> int | None:
    """Attach Energy to the Active Pokémon — that is the one that attacks."""
    fallback = None
    for i, o in enumerate(options):
        if o.get("type") != OPT_ATTACH:
            continue
        if fallback is None:
            fallback = i
        if o.get("inPlayArea") == 4:  # AreaType.ACTIVE
            return i
    return fallback


def _choose_main(options: list[dict], obs: dict) -> int:
    """One action from the main menu, by the free-actions-first rule."""
    best_i, best_rank = 0, -1
    for i, o in enumerate(options):
        rank = MAIN_PRIORITY.get(o.get("type"), 0)
        if rank > best_rank:
            best_i, best_rank = i, rank

    kind = options[best_i].get("type")
    if kind == OPT_ATTACK:
        alt = _best_attack(options, obs)
        if alt is not None:
            return alt
    elif kind == OPT_ATTACH:
        alt = _best_attach(options)
        if alt is not None:
            return alt
    return best_i


def _choose_setup(options: list[dict], obs: dict) -> int:
    """Opening Active: the sturdiest Basic we can lead with."""
    hp_table = card_hp()
    if not hp_table:
        return 0
    best_i, best_hp = 0, -1
    for i, o in enumerate(options):
        hp = hp_table.get(o.get("cardId"), 0) or 0
        if hp > best_hp:
            best_i, best_hp = i, hp
    return best_i


def agent(obs_dict: dict) -> list[int]:
    """Never raise — an exception forfeits the game, so fall back to a legal pick."""
    try:
        return _agent(obs_dict)
    except Exception:
        sel = (obs_dict or {}).get("select")
        if sel is None:
            raise  # the deck must load; failing loudly here is correct
        k = max(1, sel.get("minCount", 1) or 1)
        return list(range(min(k, len(sel.get("option") or [1]))))


def _agent(obs_dict: dict) -> list[int]:
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
            return [_choose_main(options, obs)]
        return [0]

    # Multi-pick: take the smallest legal set, lowest indices, no duplicates.
    k = max(min_count, 1)
    k = min(k, max_count, n)
    return list(range(k))


if __name__ == "__main__":
    random.seed(0)
    print(f"deck loads: {len(read_deck_csv())} cards")
    print(f"attack table: {len(attack_damage())} attacks")
