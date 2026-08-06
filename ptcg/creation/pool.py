"""Card pool loaded from the engine dump (decisions.md D2)."""

import json
from collections import Counter
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DUMP = ROOT / "data" / "engine_dump"

COLORLESS = 0

TYPE_NAMES = {
    0: "Colorless", 1: "Grass", 2: "Fire", 3: "Water", 4: "Lightning",
    5: "Psychic", 6: "Fighting", 7: "Darkness", 8: "Metal", 9: "Dragon",
    10: "Rainbow", 11: "TeamRocket",
}

# CardType enum values (cg/api.py)
POKEMON, ITEM, TOOL, SUPPORTER, STADIUM, BASIC_ENERGY, SPECIAL_ENERGY = range(7)

# What each special energy card provides, from the EN CSV effect text.
# "wild" = every type (with an attachment condition noted in `gate`).
SPECIAL_ENERGY_PROVIDES = {
    9: {"types": set()},                                    # Boomerang: {C} only
    10: {"types": "wild", "gate": "stage2"},                # Neo Upper
    11: {"types": set()},                                   # Mist: {C} only
    12: {"types": "wild", "gate": None},                    # Legacy (ACE SPEC)
    13: {"types": set()},                                   # Enriching: {C} only
    14: {"types": set()},                                   # Spiky: {C} only
    15: {"types": {5, 7}, "gate": "team_rocket"},           # Team Rocket's
    16: {"types": "wild", "gate": "basic"},                 # Prism
    17: {"types": set()},                                   # Ignition: {C} only
    18: {"types": {1}, "gate": None},                       # Grow Grass
    19: {"types": {5}, "gate": None},                       # Telepath Psychic
    20: {"types": {6}, "gate": None},                       # Rock Fighting
}


def _ensure_dump() -> None:
    """Card/attack tables are engine-derived (licensed): generated locally
    into the gitignored data/ area, never committed."""
    if (DUMP / "cards.json").exists() and (DUMP / "attacks.json").exists():
        return
    from dataclasses import asdict
    from cg.api import all_card_data, all_attack
    DUMP.mkdir(parents=True, exist_ok=True)
    (DUMP / "cards.json").write_text(
        json.dumps([asdict(c) for c in all_card_data()]))
    (DUMP / "attacks.json").write_text(
        json.dumps([asdict(a) for a in all_attack()]))


class CardPool:
    def __init__(self) -> None:
        _ensure_dump()
        cards = json.loads((DUMP / "cards.json").read_text())
        attacks = json.loads((DUMP / "attacks.json").read_text())
        self.by_id: dict[int, dict] = {c["cardId"]: c for c in cards}
        self.attack_by_id: dict[int, dict] = {a["attackId"]: a for a in attacks}
        self.ids_by_name: dict[str, list[int]] = {}
        for c in cards:
            self.ids_by_name.setdefault(c["name"], []).append(c["cardId"])

    def card(self, card_id: int) -> dict | None:
        return self.by_id.get(card_id)

    def name(self, card_id: int) -> str:
        return self.by_id[card_id]["name"]

    def typed_cost(self, attack_id: int) -> Counter:
        """Non-colorless energy requirement of an attack, by type."""
        return Counter(
            e for e in self.attack_by_id[attack_id]["energies"] if e != COLORLESS
        )

    def evolves_from_name(self, name: str) -> str | None:
        """evolvesFrom for any pool card with this name (prints agree)."""
        for cid in self.ids_by_name.get(name, []):
            ef = self.by_id[cid]["evolvesFrom"]
            if ef:
                return ef
        return None


@lru_cache(maxsize=1)
def pool() -> CardPool:
    return CardPool()
