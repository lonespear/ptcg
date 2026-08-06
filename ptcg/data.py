"""Load and normalize the competition card data.

The shipped CSV is one row per *effect* (an attack or an ability), with the card's
own attributes repeated on every row. This module splits that into two tidy tables:

  cards   — one row per Card ID (identity, HP, type, weakness, retreat, ...)
  effects — one row per attack/ability, keyed by Card ID, with cost and damage parsed

Usage:
    from ptcg import load_cards
    cards, effects = load_cards()
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd

# kagglehub extracts here; override with PTCG_DATA_DIR if you move it.
COMP_DIR = Path(
    os.environ.get(
        "PTCG_DATA_DIR",
        Path.home()
        / ".cache"
        / "kagglehub"
        / "competitions"
        / "pokemon-tcg-ai-battle-challenge-strategy",
    )
)

STAGE_COL = "Stage (Pokémon)/Type (Energy and Trainer)"

# Energy symbols as they appear in the Cost / Type columns.
ENERGY_NAMES = {
    "G": "Grass",
    "R": "Fire",
    "W": "Water",
    "L": "Lightning",
    "P": "Psychic",
    "F": "Fighting",
    "D": "Darkness",
    "M": "Metal",
    "N": "Dragon",
    "Y": "Fairy",
    "C": "Colorless",
}

# A colorless energy requirement is drawn as a filled circle rather than {C}.
COLORLESS_GLYPH = "●"

_BRACE_RE = re.compile(r"\{([A-Z])\}")
_ABILITY_TAG_RE = re.compile(r"^\s*\[([^\]]+)\]\s*")
# Damage is like "120", "30×", "50+", or "" — capture the number and the modifier.
_DAMAGE_RE = re.compile(r"^\s*(\d+)\s*([×x+\-])?\s*$")

# Ways an attack pays a cost that never shows up in its energy requirement.
# Order matters — the first match wins, so the harshest costs are listed first.
APOS = r"[’']"
DRAWBACK_PATTERNS = {
    # Discarding the attacker outright is the harshest cost in the pool.
    "Self-KO / discard": rf"discard this pok[eé]mon",
    # Covers both "can't use attacks" and the named form "can't use Mega Brave".
    "Cannot attack next turn": rf"during your next turn.*can{APOS}t"
                               rf"|this pok[eé]mon can{APOS}t (?:use|attack)",
    "Self-discard energy": r"discard (?:an?|\d+|all) .*energy .*from this pok[eé]mon"
                           r"|move an energy from this pok[eé]mon",
    "Requires hand resource": rf"if you can{APOS}t.*this attack does nothing"
                              r"|discard .*from your hand",
    "Conditional / fails": r"this attack does nothing"
                           rf"|this attack{APOS}s base damage is",
    "Self-status condition": r"this pok[eé]mon is now (?:confused|asleep|paralyzed"
                             r"|burned|poisoned)",
    "Self-damage": r"damage to (?:itself|this pok[eé]mon)"
                   r"|damage to each of your benched",
    "Self-mill": r"discard the top \d+ cards? of your deck",
    "Coin flip": r"flip \d* ?coins?",
}


def _blank_to_na(df: pd.DataFrame) -> pd.DataFrame:
    """The CSV uses the literal string 'n/a' as its missing marker."""
    return df.replace({"n/a": pd.NA, "": pd.NA})


def load_raw(region: str = "EN") -> pd.DataFrame:
    """Return the competition CSV exactly as shipped (one row per effect)."""
    path = COMP_DIR / f"{region.upper()}_Card_Data.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/download_data.py` first, "
            f"or set PTCG_DATA_DIR to the extracted competition folder."
        )
    # The file is UTF-8; reading it as cp1252 is what produces the 'PokÃ©mon' mojibake.
    df = pd.read_csv(path, encoding="utf-8", keep_default_na=False, na_values=[""])
    return _blank_to_na(df)


def parse_cost(cost: object) -> dict:
    """Parse an energy cost string like '{P}●●' into a structured requirement.

    Returns total symbol count, the colorless (wildcard) portion, and a per-type
    breakdown of the colored requirements.
    """
    out = {"cost_total": 0, "cost_colorless": 0, "cost_colored": 0, "cost_types": ()}
    if not isinstance(cost, str):
        return out
    colored = _BRACE_RE.findall(cost)
    colorless = cost.count(COLORLESS_GLYPH)
    out["cost_colored"] = len(colored)
    out["cost_colorless"] = colorless
    out["cost_total"] = len(colored) + colorless
    out["cost_types"] = tuple(sorted(set(colored)))
    return out


def parse_damage(damage: object) -> dict:
    """Parse a damage string into a base number and its modifier.

    '120' -> (120, flat); '30×' -> (30, multiplier); '50+' -> (50, plus).
    Variable-damage attacks keep the base so it can be treated as a floor.
    """
    out = {"damage_base": pd.NA, "damage_mod": pd.NA}
    if not isinstance(damage, str):
        return out
    m = _DAMAGE_RE.match(damage)
    if not m:
        return out
    out["damage_base"] = int(m.group(1))
    mod = m.group(2)
    out["damage_mod"] = {None: "flat", "×": "multiplier", "x": "multiplier",
                         "+": "plus", "-": "minus"}[mod]
    return out


def _split_effect_name(name: object) -> tuple[str | None, str | None]:
    """Split '[Ability] Snack Seek' into ('Ability', 'Snack Seek').

    Attacks carry no tag. '[Tera]' and similar tags have no trailing name.
    """
    if not isinstance(name, str):
        return (None, None)
    name = name.strip()
    m = _ABILITY_TAG_RE.match(name)
    if not m:
        return (None, name or None)
    return (m.group(1), (name[m.end():].strip() or None))


def load_cards(region: str = "EN") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (cards, effects) — the normalized two-table view of the card pool."""
    raw = load_raw(region)
    raw = raw.copy()
    raw["Move Name"] = raw["Move Name"].astype("string").str.strip()

    # ---- cards: card-level attributes, deduplicated -------------------------
    card_cols = [
        "Card ID", "Card Name", "Expansion", "Collection No.", STAGE_COL,
        "Rule", "Category", "Previous stage", "HP", "Type",
        "Weakness", "Resistance (Type)", "Retreat",
    ]
    cards = raw[card_cols].drop_duplicates("Card ID").reset_index(drop=True)
    cards = cards.rename(columns={
        "Card ID": "card_id",
        "Card Name": "name",
        "Expansion": "expansion",
        "Collection No.": "collection_no",
        STAGE_COL: "stage",
        "Rule": "rule",
        "Category": "category",
        "Previous stage": "evolves_from",
        "HP": "hp",
        "Type": "type",
        "Weakness": "weakness",
        "Resistance (Type)": "resistance",
        "Retreat": "retreat",
    })

    cards["hp"] = pd.to_numeric(cards["hp"], errors="coerce").astype("Int64")
    cards["retreat"] = pd.to_numeric(cards["retreat"], errors="coerce").astype("Int64")

    # 'Pokémon Tool' contains the word Pokémon but is a Trainer subtype.
    cards["is_pokemon"] = (
        cards["stage"].str.contains("Pokémon", na=False)
        & ~cards["stage"].str.contains("Tool", na=False)
    )
    cards["supertype"] = "Trainer"
    cards.loc[cards["is_pokemon"], "supertype"] = "Pokémon"
    cards.loc[cards["stage"].str.contains("Energy", na=False), "supertype"] = "Energy"

    # A Pokémon printed with no retreat symbols shows 'n/a' here — that is a
    # free retreat (cost 0), not a missing value. Verified against known
    # free-retreat cards (Shaymin, Emolga, Tynamo, Jolteon ex).
    free_retreat = cards["is_pokemon"] & cards["retreat"].isna()
    cards.loc[free_retreat, "retreat"] = 0

    # A Rule Box (ex / Mega ex / ACE SPEC) is the main risk lever in deckbuilding:
    # ex Pokémon give up two prizes when knocked out.
    cards["has_rule_box"] = cards["rule"].notna()
    cards["is_ex"] = cards["rule"].fillna("").str.contains("ex", case=False)

    for col in ("type", "weakness", "resistance"):
        cards[f"{col}_name"] = (
            cards[col].str.extract(r"\{([A-Z])\}", expand=False).map(ENERGY_NAMES)
        )

    # ---- effects: one row per attack / ability ------------------------------
    # Trainer and Special Energy cards carry their rules text in Effect
    # Explanation with no Move Name, so keying on Move Name alone loses them.
    eff = raw[["Card ID", "Card Name", "Move Name", "Cost", "Damage",
               "Effect Explanation"]].copy()
    eff = eff[eff["Move Name"].notna() | eff["Effect Explanation"].notna()]
    eff = eff.reset_index(drop=True)
    eff = eff.rename(columns={
        "Card ID": "card_id",
        "Card Name": "name",
        "Move Name": "effect_name_raw",
        "Cost": "cost_raw",
        "Damage": "damage_raw",
        "Effect Explanation": "text",
    })

    tags = eff["effect_name_raw"].map(_split_effect_name)
    eff["effect_tag"] = [t[0] for t in tags]
    eff["effect_name"] = [t[1] for t in tags]
    eff["effect_kind"] = eff["effect_tag"].where(eff["effect_tag"].isna(), "Ability")
    eff["effect_kind"] = eff["effect_kind"].fillna("Attack")
    # A row with no move name at all is the card's own rules text.
    eff.loc[eff["effect_name_raw"].isna(), "effect_kind"] = "Card text"

    eff = pd.concat([eff, eff["cost_raw"].map(parse_cost).apply(pd.Series)], axis=1)
    eff = pd.concat([eff, eff["damage_raw"].map(parse_damage).apply(pd.Series)], axis=1)
    eff["damage_base"] = pd.to_numeric(eff["damage_base"], errors="coerce").astype("Int64")

    # Damage per energy is the core efficiency metric for attacker evaluation.
    dmg = pd.to_numeric(eff["damage_base"], errors="coerce").astype(float)
    cost = pd.to_numeric(eff["cost_total"], errors="coerce").astype(float)
    eff["damage_per_energy"] = (dmg / cost.where(cost > 0)).astype(float)

    # The printed energy cost is not the real cost. Every high-damage-per-energy
    # attack in this pool pays somewhere else — a lost turn, a discarded hand, a
    # coin flip — so damage/energy alone ranks glass cannons above real attackers.
    txt = eff["text"].fillna("").str.lower()
    eff["drawback"] = pd.NA
    for label, pattern in DRAWBACK_PATTERNS.items():
        hit = txt.str.contains(pattern, regex=True, na=False) & eff["drawback"].isna()
        eff.loc[hit, "drawback"] = label
    eff["is_conditional"] = eff["drawback"].notna()

    return cards, eff
