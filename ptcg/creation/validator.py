"""Deck validator: hard legality plus coherence heuristics (decisions.md D5/D6).

Hard legality (errors -> deck is rejected):
  - exactly 60 cards, all IDs in the pool
  - max 4 copies per card name, basic energy exempt
  - at most 1 ACE SPEC card
  - at least 1 Basic Pokemon

Coherence (warnings -> deck is legal but structurally suspect):
  - evolution closure: every evolution card can actually be reached
    (pre-evolutions present; Rare Candy covers a Basic -> Stage 2 skip)
  - line ratios: no stage with more copies than the stage below it
  - energy consistency: every attacker has at least one attack whose typed
    cost the deck's energy base can produce (stage gates on special energy
    respected)
  - mulligan risk: opening-hand no-Basic probability

Usage: .venv/bin/python -m ptcg.validator reference_deck.csv
"""

from collections import Counter
from dataclasses import dataclass, field
from math import comb

from .pool import (
    BASIC_ENERGY, POKEMON, SPECIAL_ENERGY, SPECIAL_ENERGY_PROVIDES,
    TYPE_NAMES, CardPool, pool,
)


@dataclass
class DeckReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    @property
    def legal(self) -> bool:
        return not self.errors

    @property
    def coherent(self) -> bool:
        return self.legal and not self.warnings

    def __str__(self) -> str:
        lines = [f"legal={self.legal} coherent={self.coherent}"]
        for e in self.errors:
            lines.append(f"  ERROR: {e}")
        for w in self.warnings:
            lines.append(f"  warn:  {w}")
        for k, v in self.stats.items():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)


def _energy_supply(deck_cards: list[dict]) -> tuple[set[int], bool, set[int]]:
    """(typed supply, any-wildcard present, gated-wildcard stages) of the deck."""
    typed: set[int] = set()
    wildcard = False
    gated: set[str] = set()
    for c in deck_cards:
        if c["cardType"] == BASIC_ENERGY:
            typed.add(c["energyType"])
        elif c["cardType"] == SPECIAL_ENERGY:
            provides = SPECIAL_ENERGY_PROVIDES.get(c["cardId"], {"types": set()})
            if provides["types"] == "wild":
                if provides.get("gate") is None:
                    wildcard = True
                else:
                    gated.add(provides["gate"])
            else:
                typed |= provides["types"]
    return typed, wildcard, gated


def _attack_coverable(typed_cost: Counter, supply: set[int], wildcard: bool,
                      gated: set[str], card: dict) -> bool:
    for t in typed_cost:
        if t in supply or wildcard:
            continue
        if "basic" in gated and card["basic"]:
            continue
        if "stage2" in gated and card["stage2"]:
            continue
        return False
    return True


def validate(deck: list[int], p: CardPool | None = None) -> DeckReport:
    p = p or pool()
    r = DeckReport()

    if len(deck) != 60:
        r.errors.append(f"deck has {len(deck)} cards, must be exactly 60")
    unknown = [cid for cid in deck if p.card(cid) is None]
    if unknown:
        r.errors.append(f"unknown card IDs: {sorted(set(unknown))}")
        return r
    cards = [p.card(cid) for cid in deck]

    # copy limits (by name; basic energy exempt)
    name_counts = Counter(c["name"] for c in cards if c["cardType"] != BASIC_ENERGY)
    for name, n in name_counts.items():
        if n > 4:
            r.errors.append(f"{n} copies of '{name}' (max 4)")

    ace_specs = [c["name"] for c in cards if c["aceSpec"]]
    if len(ace_specs) > 1:
        r.errors.append(f"{len(ace_specs)} ACE SPEC cards (max 1): {ace_specs}")

    basics = [c for c in cards if c["cardType"] == POKEMON and c["basic"]]
    if not basics:
        r.errors.append("no Basic Pokemon")

    # --- coherence ---
    deck_names = set(name_counts) | {c["name"] for c in cards}
    has_rare_candy = "Rare Candy" in deck_names

    for name in sorted({c["name"] for c in cards
                        if c["cardType"] == POKEMON and not c["basic"]}):
        pre = p.evolves_from_name(name)
        if pre is None:
            continue
        card = next(c for c in cards if c["name"] == name)
        if pre in deck_names:
            pass
        elif card["stage2"]:
            basic_name = p.evolves_from_name(pre)
            if basic_name in deck_names and has_rare_candy:
                pass  # Basic -> Stage 2 via Rare Candy
            else:
                r.warnings.append(
                    f"'{name}' (Stage 2) unreachable: no '{pre}'"
                    + ("" if has_rare_candy else " and no Rare Candy path")
                )
        else:
            r.warnings.append(f"'{name}' (Stage 1) unreachable: no '{pre}'")

    # line ratios: a stage with more copies than the stage below it
    for name in sorted({c["name"] for c in cards
                        if c["cardType"] == POKEMON and not c["basic"]}):
        pre = p.evolves_from_name(name)
        if pre in name_counts:
            card = next(c for c in cards if c["name"] == name)
            slack = name_counts.get("Rare Candy", 0) if card["stage2"] else 0
            if name_counts[name] > name_counts[pre] + slack:
                r.warnings.append(
                    f"line ratio: {name_counts[name]}x '{name}' over "
                    f"{name_counts[pre]}x '{pre}'"
                )

    # energy consistency: every attacker must have one coverable attack
    supply, wildcard, gated = _energy_supply(cards)
    for name in sorted({c["name"] for c in cards if c["cardType"] == POKEMON}):
        card = next(c for c in cards if c["name"] == name)
        if not card["attacks"]:
            continue
        costs = [p.typed_cost(aid) for aid in card["attacks"]]
        if not any(_attack_coverable(tc, supply, wildcard, gated, card)
                   for tc in costs):
            needed = sorted({TYPE_NAMES[t] for tc in costs for t in tc})
            r.warnings.append(
                f"'{name}' has no usable attack: needs {needed}, "
                f"supply is {sorted(TYPE_NAMES[t] for t in supply)}"
            )

    # --- stats ---
    n_basic = sum(1 for c in cards if c["cardType"] == POKEMON and c["basic"])
    mulligan_p = comb(60 - n_basic, 7) / comb(60, 7) if n_basic <= 53 else 0.0
    type_counts = Counter(c["cardType"] for c in cards)
    r.stats = {
        "pokemon/item/tool/supporter/stadium/energy":
            f"{type_counts.get(0,0)}/{type_counts.get(1,0)}/{type_counts.get(2,0)}"
            f"/{type_counts.get(3,0)}/{type_counts.get(4,0)}"
            f"/{type_counts.get(5,0)+type_counts.get(6,0)}",
        "basic_pokemon": n_basic,
        "mulligan_probability": round(mulligan_p, 4),
        "energy_supply": sorted(TYPE_NAMES[t] for t in supply)
            + (["<any>"] if wildcard else [])
            + [f"<any:{g}>" for g in sorted(gated)],
    }
    if n_basic and mulligan_p > 0.30:
        r.warnings.append(
            f"high mulligan risk: {mulligan_p:.1%} with {n_basic} Basics"
        )
    return r


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "reference_deck.csv"
    with open(path) as f:
        deck = [int(line) for line in f.read().split("\n")[:60]]
    print(f"validating {path}:")
    print(validate(deck))
