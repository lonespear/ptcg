"""The energy-mechanics knowledge base: which cards break the 1-attach-per-turn rule.

Energy trajectory (D31) is arithmetic only if you know what the pool can do to
an energy count besides the one manual attach a turn. This module reads the
effect text of every card in the 1,267-card pool and classifies the ones that
move, make, recycle, or burn Energy, into seven mechanic classes:

    acceleration  something puts Energy onto YOUR board from hand/deck/discard
    movement      Energy already on YOUR board changes Pokemon
    retrieval     Energy leaves the discard pile for your hand (or your deck)
    tutor         Energy leaves the deck for your hand (enables the manual attach)
    discard_cost  YOUR attached Energy is spent - the negative trajectory event
    disruption    the OPPONENT's attached Energy is removed or moved
    scaling       damage (or HP) that scales with an Energy count

    threshold     damage (or HP) that unlocks at an Energy count - a second
                  online date beyond the printed attack cost

`tutor`, `disruption` and `threshold` are additions beyond the five classes in
the tasking: a deck->hand energy search changes whether the base attach can even
happen, opponent-side removal is the mirror of our own acceleration when
projecting their trajectory, and a threshold is an online date the printed cost
does not show. All three are labelled so they can be ignored.

Precision over recall. A sentence that does not match a rule lands in the
`unclassified` bucket and is reported, never forced into a class.

Corpus: the engine dump (`ptcg.creation.pool`) - card skills and attack text -
unioned with the Effect Explanation column of the competition EN CSV
(`ptcg.data.load_cards`). The CSV is, empirically, a subset for Energy text:
its 35 rows absent from the engine dump mention no Energy at all.

    python -m ptcg.energy_mechanics          # self-check + spot-checks
    python -m ptcg.energy_mechanics --write  # regenerate data/energy_mechanics.json
"""

from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

from ptcg.creation.pool import (BASIC_ENERGY, ITEM, POKEMON, SPECIAL_ENERGY,
                                STADIUM, SUPPORTER, TOOL, TYPE_NAMES, pool)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "energy_mechanics.json"

CARD_TYPE_NAME = {
    POKEMON: "Pokemon", ITEM: "Item", TOOL: "Tool", SUPPORTER: "Supporter",
    STADIUM: "Stadium", BASIC_ENERGY: "Basic Energy",
    SPECIAL_ENERGY: "Special Energy",
}

# The CSV and the engine dump both use curly apostrophes; some card names use
# the straight one. Every ownership/possessive rule has to match either.
APO = "[’']"

CLASSES = ("acceleration", "movement", "retrieval", "tutor", "discard_cost",
           "disruption", "scaling", "threshold")

# --- text plumbing ----------------------------------------------------------

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def split_sentences(text: str) -> list[str]:
    """Sentences, whitespace-normalized. Reminder text in parens is kept."""
    if not text:
        return []
    return [re.sub(r"\s+", " ", s).strip()
            for s in _SENT_SPLIT.split(text) if s.strip()]


def _norm(text: str | None) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


# A destination or source phrase belongs to the opponent if it says so. The
# Defending / Attacking Pokemon are the opponent's by definition of the rules
# text they appear in (an attack's own rider names its user "this Pokemon").
_OPP = re.compile(
    rf"opponent{APO}s|\btheir\b|\bthey\b|Defending Pok|Attacking Pok"
    rf"|each player|that player",
    re.I)


def _is_opponent(phrase: str) -> bool:
    return bool(_OPP.search(phrase or ""))


# --- field parsers ----------------------------------------------------------

_WORD_NUM = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4,
             "five": 5, "six": 6, "seven": 7}


def parse_rate(obj: str) -> tuple[int | None, str]:
    """(energies moved per use, how the count is written).

    `None` means the text does not bound it: "any number of", "any amount of",
    or a count that depends on a coin flip or a board census.
    """
    o = obj.lower()
    if re.search(r"\bany (?:number|amount) of\b", o):
        return None, "unbounded"
    if re.search(r"\bup to the (?:number|amount) of\b", o):
        return None, "board_dependent"
    if re.search(r"\bnumber of heads\b", o):
        return None, "coin"
    m = re.search(r"\bup to (\d+|a|an|one|two|three|four|five|six|seven)\b", o)
    if m:
        g = m.group(1)
        return (int(g) if g.isdigit() else _WORD_NUM[g]), "up_to"
    m = re.match(r"\s*(\d+)\b", o)
    if m:
        return int(m.group(1)), "fixed"
    if re.match(rf"\s*(?:a|an|one|this card|it|them|all)\b", o):
        # "all" is every Energy on one Pokemon: unbounded in principle, but as
        # a cost it is always "whatever is there". Leave it unbounded.
        if re.match(r"\s*all\b", o):
            return None, "all"
        return 1, "fixed"
    return None, "unparsed"


_ZONE_MARK = re.compile(
    r"from your hand|from your discard pile|in your discard pile"
    r"|search your deck for|from your deck"
    r"|look at the top \d+ cards? of your deck", re.I)


def _zone_of(marker: str) -> str:
    m = marker.lower()
    if "top" in m:
        return "deck_top"
    if "deck" in m:
        return "deck"
    if "discard" in m:
        return "discard"
    return "hand"


def last_zone(window: str) -> tuple[str, int]:
    """(zone, end offset) of the LAST zone marker in `window`.

    A card can name two zones in one sentence - "when you play this Pokemon
    from your hand ... search your deck for Energy" - and the one that governs
    the Energy is always the one nearest the verb, i.e. the last.
    """
    marks = list(_ZONE_MARK.finditer(window or ""))
    if not marks:
        return "unspecified", 0
    return _zone_of(marks[-1].group(0)), marks[-1].end()


def parse_source_zone(sentence: str, obj: str) -> str:
    """Where the Energy comes from. The object phrase wins over the sentence."""
    for scope in (obj, sentence):
        zone, _ = last_zone(scope or "")
        if zone != "unspecified":
            return zone
    return "unspecified"


_ENERGY_TYPE = re.compile(r"\{([A-Z])\}")
_TYPE_BY_LETTER = {"G": 1, "R": 2, "W": 3, "L": 4, "P": 5, "F": 6, "D": 7,
                   "M": 8, "N": 9, "Y": 10, "C": 0}


def parse_energy_filter(obj: str) -> dict:
    """Which Energy cards the effect can touch."""
    letters = sorted(set(_ENERGY_TYPE.findall(obj or "")))
    return {
        "basic_only": bool(re.search(r"\bbasic\b", obj or "", re.I)),
        "types": letters,                       # [] means any type
        "type_names": [TYPE_NAMES.get(_TYPE_BY_LETTER.get(c, -1), c)
                       for c in letters],
        "different_types": bool(re.search(r"of different types", obj or "",
                                          re.I)),
    }


_TAG = re.compile(rf"([A-Z][A-Za-z]*){APO}s Pok", re.U)
_GROUP = re.compile(r"\b(Future|Ancient|Tera|Fusion Strike|Rapid Strike"
                    r"|Single Strike)\b")


def parse_target(dest: str) -> dict:
    """The constraint on which of our Pokemon can receive the Energy."""
    d = dest or ""
    low = d.lower()
    if re.search(r"\bthis pok|\bit\b|\bitself\b", low):
        scope = "self"
    elif "each of" in low:
        scope = "each"
    elif re.search(r"another of your|your other pok|1 of your|your pok", low):
        scope = "choice"
    elif re.search(r"that pok|the new|them\b", low):
        scope = "referent"
    else:
        scope = "unparsed"
    position = ("bench" if re.search(r"bench", low)
                else "active" if re.search(r"active", low) else "any")
    letters = sorted(set(_ENERGY_TYPE.findall(d)))
    tag = _TAG.search(d)
    group = _GROUP.search(d)
    stage = re.search(r"stage (\d)", low)
    return {
        "scope": scope,
        "position": position,
        "types": letters,
        "owner_tag": tag.group(1) if tag else None,       # Marnie's, Iono's...
        "group": group.group(1) if group else None,       # Future, Tera...
        "stage": int(stage.group(1)) if stage else None,
        "free_distribution": "in any way you like" in low,
        "raw": _norm(d),
    }


def parse_frequency(card: dict, kind: str, text: str) -> str:
    """How often the effect can fire, in trajectory terms.

    An attack rider costs the turn's attack; a Supporter costs the turn's
    Supporter; an Item costs nothing but a card. Those are different budgets
    and the planner spends them separately.
    """
    if kind == "attack":
        return "attack_rider"
    t = (text or "").lower()
    ct = card["cardType"]
    if ct == ITEM:
        return "item_play"
    if ct == SUPPORTER:
        return "supporter_play"
    if ct == STADIUM:
        return "stadium_in_play"
    if ct == TOOL:
        return "tool_attached"
    if ct in (BASIC_ENERGY, SPECIAL_ENERGY):
        return "energy_card"
    # Pokemon abilities.
    if re.search(rf"as often as you like during your turn", t):
        return "unlimited_per_turn"
    if re.search(r"when you play this pok.mon from your hand to evolve", t):
        return "on_evolve"
    if re.search(r"when you play this pok.mon from your hand onto your bench",
                 t):
        return "on_bench_play"
    if re.search(r"once during your turn, when this pok.mon moves from your "
                 r"bench", t):
        return "on_promote"
    if re.search(r"once during your turn", t):
        return "once_per_turn"
    if re.search(r"at the end of your turn", t):
        return "end_of_turn"
    if re.search(r"when .{0,60}knocked out|is damaged by an attack", t):
        return "reactive"
    return "ability_other"


# --- the record -------------------------------------------------------------

@dataclass
class Mechanic:
    card_id: int
    name: str
    card_type: str
    effect_kind: str            # ability | attack | card_text
    effect_name: str | None
    mechanic: str
    source_zone: str
    dest_zone: str
    target: dict
    energy_filter: dict
    rate: int | None
    rate_kind: str
    frequency: str
    optional: bool
    attack_id: int | None = None
    attack_cost: list[int] | None = None
    scale: dict | None = None
    symmetric: bool = False
    sentence: str = ""
    text: str = ""


# --- extractors -------------------------------------------------------------

# " to " that is not the " to " of "up to".
_TO = r"(?<!\bup)\s+to\s+"

# The verb and its clause are matched separately: a single combined pattern
# swallows the real clause when an earlier "Energy attached to ..." appears in
# the same sentence (Dedenne counts the opponent's board, then attaches).
_ATTACH_VERB = re.compile(r"attach(?:es|ed|ing)?\b", re.I)
_ATTACH_TAIL = re.compile(
    rf"\s*(?P<obj>.{{0,140}}?){_TO}(?P<dest>[^.,;]{{0,90}})", re.I)
# A trigger clause ("whenever you attach ... , heal") is not acceleration.
_ATTACH_TRIGGER = re.compile(
    r"when(?:ever)?\s+(?:you|they|a player)\s+attach", re.I)
# Referents that stand in for an Energy noun phrase named earlier in the clause.
_PRONOUN = re.compile(r"(?:them|it|these|those|1 of them|\d+ of them)", re.I)
# A rider that only *reports* the attach the same effect already made.
_ATTACH_BACKREF = re.compile(r"if you attached .{0,60}in this way", re.I)

_MOVE = re.compile(
    rf"\bmoves?\s+(?P<obj>[^.]{{0,90}}?)\s+from\s+(?P<src>[^.]{{0,70}}?)"
    rf"{_TO}(?P<dest>[^.,;]{{0,90}})", re.I)

_TO_HAND = re.compile(
    r"put\s+(?P<obj>[^.]{0,140}?)\s+(?:in|into)\s+your\s+hand", re.I)
# The same move, phrased from the other seat ("into their hand").
_TO_OPP_HAND = re.compile(
    r"put\s+(?P<obj>[^.]{0,140}?)\s+(?:in|into)\s+their\s+hand", re.I)
# A card both players may use reads as the opponent's; it is also ours.
_SYMMETRIC = re.compile(r"each player|that player", re.I)
_SEARCH_HAND = re.compile(
    r"search your deck for\s+(?P<obj>[^.]{0,160}?),?\s*(?:reveal[^.,]*,\s*)?"
    r"and put (?:it|them|\d+ of them)[^.]{0,20}?into your hand", re.I)
_RECYCLE = re.compile(
    r"shuffle\s+(?P<obj>[^.]{0,90}?)\s+from your discard pile into your deck",
    re.I)
# Paying an attack by shuffling your own attached Energy back into the deck.
_SHUFFLE_COST = re.compile(
    r"shuffle\s+(?P<obj>(?:all|\d+|up to \d+|any amount of)[^.]{0,60}?Energy"
    r"\s+attached to\s+(?P<src>[^.,;]{0,50}?))\s+into your deck", re.I)

_DISCARD = re.compile(
    r"discard\s+(?P<obj>[^.]{0,90}?)\s+from\s+(?P<src>[^.,;]{0,70})", re.I)
# "Discard all Energy from this Pokemon" is caught above; this catches the
# form with no `from` clause at all, which never occurs for Energy in this
# pool but is cheap insurance against a reprint.
_DISCARD_BARE = re.compile(
    r"discard\s+(?P<obj>(?:all|\d+|up to \d+|a|an|any amount of)\s+"
    r"[^.,;]{0,40}?Energy[^.,;]{0,30})(?:[.,;]|$)", re.I)

_SCALE_DMG = re.compile(
    r"(?:does|do|doing)\s+(?P<n>\d+)\s*(?P<mod>more|less)?\s*damage"
    r"[^.]{0,60}?for each\s+(?P<what>[^.]{0,110}?Energy[^.]{0,70}?)(?=[.]|$)",
    re.I)
_SCALE_COUNTERS = re.compile(
    r"(?:put|place)\s+(?P<n>\d+)\s+damage counters?"
    r"[^.]{0,70}?for each\s+(?P<what>[^.]{0,110}?Energy[^.]{0,70}?)(?=[.]|$)",
    re.I)
_SCALE_HP = re.compile(
    r"gets?\s*\+(?P<n>\d+)\s*HP\s*for each\s+"
    r"(?P<what>[^.]{0,110}?Energy[^.]{0,70}?)(?=[.]|$)", re.I)

_HAS_ENERGY = re.compile(r"\bEnerg(?:y|ies)\b", re.I)

# "If this Pokemon has 2 or more {G} Energy attached, this attack does 120 more
# damage." - an online date the printed cost never shows.
_THRESHOLD = re.compile(
    r"\bif (?P<who>this pok.mon has|you have)\s+"
    r"(?P<n>at least \d+|\d+ or more|\d+|any|no)\s+"
    r"(?P<what>[^.,]{0,45}?Energy)(?:\s+attached|\s+in play)?[^.,]{0,20},\s*"
    r"(?:this attack does|it gets \+)\s*(?P<bonus>\d+)\s*"
    r"(?P<unit>more damage|HP)", re.I)

# "Flip a coin for each Energy attached to this Pokemon." + "This attack does
# 70 damage for each heads." - scaling on Energy, paid through a coin.
_COIN_PER_ENERGY = re.compile(
    r"flip a coin for each\s+(?P<what>[^.]{0,60}?Energy[^.]{0,40}?)(?=[.]|$)",
    re.I)
_PER_HEADS = re.compile(
    r"does\s+(?P<n>\d+)\s*(?P<mod>more|less)?\s*damage[^.]{0,40}?for each heads",
    re.I)


def _optional(sentence: str) -> bool:
    return bool(re.search(r"\byou may\b|\bmay\b", sentence, re.I))


def _scale_ref(what: str) -> str:
    w = (what or "").lower()
    if "discarded in this way" in w or "you discarded" in w:
        return "discarded_this_attack"
    if re.search(rf"in your opponent{APO}s discard", w):
        return "opponent_discard"
    if "in your discard pile" in w:
        return "own_discard"
    if "both active" in w:
        return "both_active"
    if _is_opponent(w):
        return "opponent_board"
    if "attached to this pok" in w or re.search(r"attached to it\b", w):
        return "self"
    if re.search(r"attached to (all of )?your", w):
        return "own_board"
    if "in their hand" in w or "find there" in w:
        return "revealed"
    return "other"


def classify_effect(card: dict, kind: str, effect_name: str | None, text: str,
                    attack_id: int | None = None,
                    attack_cost: list[int] | None = None) -> list[Mechanic]:
    """Every mechanic record one attack or ability produces. May be empty."""
    out: list[Mechanic] = []

    def rec(**kw) -> None:
        out.append(Mechanic(
            card_id=card["cardId"], name=card["name"],
            card_type=CARD_TYPE_NAME.get(card["cardType"], "?"),
            effect_kind=kind, effect_name=effect_name,
            frequency=parse_frequency(card, kind, text),
            attack_id=attack_id, attack_cost=attack_cost,
            text=_norm(text), **kw))

    is_energy_card = card["cardType"] in (BASIC_ENERGY, SPECIAL_ENERGY)
    sents = split_sentences(text)
    for i, sent in enumerate(sents):
        if not _HAS_ENERGY.search(sent):
            # On an Energy card, "this card" IS the Energy (Boomerang Energy
            # re-attaches itself without ever using the word).
            if not (is_energy_card and "this card" in sent.lower()):
                continue

        # --- acceleration: Energy onto our board -----------------------------
        for vm in _ATTACH_VERB.finditer(sent):
            m = _ATTACH_TAIL.match(sent, vm.end())
            if not m:
                continue
            obj, dest = m.group("obj"), m.group("dest")
            if _ATTACH_TRIGGER.search(sent[:vm.end()]):
                continue                       # "whenever you attach ..."
            if _ATTACH_BACKREF.search(sent):
                continue                       # "if you attached ... this way"
            if _is_opponent(dest) or _is_opponent(obj):
                continue
            # "search your deck for N Energy ... and attach THEM to ..." - the
            # object is a pronoun and the real noun phrase sits in the clause
            # before the verb, after the last zone marker.
            window = sent[:vm.start()] + " " + obj
            zone, cut = last_zone(window)
            if _HAS_ENERGY.search(obj) or "this card" in obj.lower():
                obj_phrase = obj
            elif _PRONOUN.fullmatch(obj.strip()):
                obj_phrase = window[cut:]
            else:
                continue
            if not (_HAS_ENERGY.search(obj_phrase)
                    or "this card" in obj_phrase.lower()):
                continue
            obj = obj_phrase
            rate, kind_ = parse_rate(obj)
            if rate is None and kind_ == "unparsed" \
                    and "this card" in obj.lower():
                rate, kind_ = 1, "fixed"       # the Energy card is itself the one
            tgt = parse_target(dest)
            # "attach X to each of your Y" multiplies by the target count.
            if tgt["scope"] == "each" and rate is not None:
                n = re.search(r"choose up to (\d+)", sent, re.I)
                rate = rate * int(n.group(1)) if n else None
                kind_ = "per_target"
            rec(mechanic="acceleration",
                source_zone=zone, dest_zone="in_play",
                target=tgt, energy_filter=parse_energy_filter(obj),
                rate=rate, rate_kind=kind_, optional=_optional(sent),
                sentence=sent)
            break                              # one accel clause per sentence

        # --- movement / disruption-by-move -----------------------------------
        for m in _MOVE.finditer(sent):
            obj, src, dest = m.group("obj"), m.group("src"), m.group("dest")
            if not _HAS_ENERGY.search(obj):
                continue                       # damage counters, not Energy
            opp = _is_opponent(src) or _is_opponent(dest)
            rate, kind_ = parse_rate(obj)
            rec(mechanic="disruption" if opp else "movement",
                source_zone="opponent_in_play" if opp else "in_play",
                dest_zone="opponent_in_play" if opp else "in_play",
                target=parse_target(dest),
                energy_filter=parse_energy_filter(obj),
                rate=rate, rate_kind=kind_, optional=_optional(sent),
                sentence=sent)
            break

        # --- retrieval / tutor: Energy into our hand or back into the deck ----
        m = _SEARCH_HAND.search(sent)
        if m and _HAS_ENERGY.search(m.group("obj")):
            rate, kind_ = parse_rate(m.group("obj"))
            rec(mechanic="tutor", source_zone="deck", dest_zone="hand",
                target={}, energy_filter=parse_energy_filter(m.group("obj")),
                rate=rate, rate_kind=kind_, optional=_optional(sent),
                sentence=sent)
        else:
            m = _TO_HAND.search(sent) or _TO_OPP_HAND.search(sent)
            if m and _HAS_ENERGY.search(m.group("obj")):
                obj = m.group("obj")
                sym = bool(_SYMMETRIC.search(sent))
                if _is_opponent(obj) and not sym:
                    # Bouncing the opponent's attached Energy to their hand is
                    # removal by another name.
                    if "attached" in obj.lower():
                        rate, kind_ = parse_rate(obj)
                        rec(mechanic="disruption",
                            source_zone="opponent_in_play", dest_zone="hand",
                            target=parse_target(obj),
                            energy_filter=parse_energy_filter(obj),
                            rate=rate, rate_kind=kind_,
                            optional=_optional(sent), sentence=sent)
                else:
                    zone = ("discard" if "discard pile" in obj.lower()
                            else "in_play" if "attached" in obj.lower()
                            else parse_source_zone(sent, obj))
                    rate, kind_ = parse_rate(obj)
                    rec(mechanic="retrieval", source_zone=zone,
                        dest_zone="hand", target={},
                        energy_filter=parse_energy_filter(obj),
                        rate=rate, rate_kind=kind_, optional=_optional(sent),
                        symmetric=sym, sentence=sent)
                    if zone == "in_play":
                        # Off the board is a trajectory loss as well.
                        rec(mechanic="discard_cost", source_zone="in_play",
                            dest_zone="hand", target=parse_target(obj),
                            energy_filter=parse_energy_filter(obj),
                            rate=rate, rate_kind=kind_,
                            optional=_optional(sent), symmetric=sym,
                            sentence=sent)
        m = _SHUFFLE_COST.search(sent)
        if m and not _is_opponent(m.group("src")):
            rate, kind_ = parse_rate(m.group("obj"))
            rec(mechanic="discard_cost", source_zone="in_play",
                dest_zone="deck", target=parse_target(m.group("src")),
                energy_filter=parse_energy_filter(m.group("obj")),
                rate=rate, rate_kind=kind_, optional=_optional(sent),
                sentence=sent)
        m = _RECYCLE.search(sent)
        if m and _HAS_ENERGY.search(m.group("obj")):
            rate, kind_ = parse_rate(m.group("obj"))
            rec(mechanic="retrieval", source_zone="discard", dest_zone="deck",
                target={}, energy_filter=parse_energy_filter(m.group("obj")),
                rate=rate, rate_kind=kind_, optional=_optional(sent),
                sentence=sent)

        # --- discard cost / disruption-by-discard ----------------------------
        m = _DISCARD.search(sent) or _DISCARD_BARE.search(sent)
        if m and _HAS_ENERGY.search(m.group("obj")):
            obj = m.group("obj")
            src = m.groupdict().get("src") or "this Pokemon"
            if "discard pile" not in src.lower():          # not an accel echo
                rate, kind_ = parse_rate(obj)
                opp = _is_opponent(src)
                zone = ("opponent_in_play" if opp
                        else "hand" if "your hand" in src.lower()
                        else "deck" if "your deck" in src.lower()
                        else "in_play")
                rec(mechanic="disruption" if opp else "discard_cost",
                    source_zone=zone, dest_zone="discard",
                    target=parse_target(src),
                    energy_filter=parse_energy_filter(obj),
                    rate=rate, rate_kind=kind_, optional=_optional(sent),
                    sentence=sent)

        # --- scaling ---------------------------------------------------------
        for rx, unit, mult in ((_SCALE_DMG, "damage", 1),
                               (_SCALE_COUNTERS, "damage", 10),
                               (_SCALE_HP, "hp", 1)):
            m = rx.search(sent)
            if not m:
                continue
            n = int(m.group("n")) * mult
            sign = -1 if (m.groupdict().get("mod") or "").lower() == "less" \
                else 1
            what = m.group("what")
            rec(mechanic="scaling", source_zone="n/a", dest_zone="n/a",
                target={}, energy_filter=parse_energy_filter(what),
                rate=None, rate_kind="n/a", optional=False,
                scale={"per_energy": n * sign, "unit": unit,
                       "reference": _scale_ref(what), "what": _norm(what)},
                sentence=sent)
            break

        # A coin per Energy, then damage per head: scaling at half rate.
        cm = _COIN_PER_ENERGY.search(sent)
        if cm:
            nxt = " ".join(sents[i:i + 2])
            hm = _PER_HEADS.search(nxt)
            if hm:
                what = cm.group("what")
                rec(mechanic="scaling", source_zone="n/a", dest_zone="n/a",
                    target={}, energy_filter=parse_energy_filter(what),
                    rate=None, rate_kind="n/a", optional=False,
                    scale={"per_energy": int(hm.group("n")) / 2,
                           "unit": "damage", "coin": True,
                           "reference": _scale_ref(what),
                           "what": _norm(what)},
                    sentence=_norm(nxt))

        # --- threshold: a damage step that unlocks at an Energy count ---------
        tm = _THRESHOLD.search(sent)
        if tm and not _is_opponent(sent[:tm.start()] + tm.group("what")):
            n = tm.group("n").lower()
            need = (1 if n == "any" else 0 if n == "no"
                    else int(re.search(r"\d+", n).group(0)))
            what = tm.group("what")
            rec(mechanic="threshold", source_zone="n/a", dest_zone="n/a",
                target={}, energy_filter=parse_energy_filter(what),
                rate=None, rate_kind="n/a", optional=False,
                scale={"needs": need,
                       "scope": ("self" if "pok" in tm.group("who").lower()
                                 else "own_board"),
                       "bonus": int(tm.group("bonus")),
                       "unit": ("damage" if "damage" in tm.group("unit").lower()
                                else "hp"),
                       "what": _norm(what)},
                sentence=sent)
    return out


# --- corpus -----------------------------------------------------------------

def csv_effect_texts() -> dict[int, set[str]]:
    """Effect Explanation text from the competition CSV, keyed by card id.

    Returns an empty mapping if the CSV is not reachable; the engine dump is
    the primary source and the KB is complete without it.
    """
    try:
        from ptcg.data import load_cards
        _, eff = load_cards()
    except Exception:
        return {}
    out: dict[int, set[str]] = {}
    for r in eff.itertuples():
        t = _norm(r.text if isinstance(r.text, str) else "")
        if t:
            out.setdefault(int(r.card_id), set()).add(t)
    return out


def effect_corpus() -> list[dict]:
    """Every (card, effect, text) in the pool, engine dump unioned with the CSV."""
    p = pool()
    csv = csv_effect_texts()
    rows: list[dict] = []
    for card in p.by_id.values():
        seen: set[str] = set()
        for s in card["skills"]:
            t = _norm(s.get("text"))
            seen.add(t)
            rows.append({"card": card, "kind": "ability" if
                         card["cardType"] == POKEMON else "card_text",
                         "effect_name": s.get("name"), "text": t,
                         "attack_id": None, "attack_cost": None})
        for aid in card["attacks"]:
            a = p.attack_by_id.get(aid)
            if not a:
                continue
            t = _norm(a.get("text"))
            seen.add(t)
            rows.append({"card": card, "kind": "attack",
                         "effect_name": a.get("name"), "text": t,
                         "attack_id": aid, "attack_cost": a.get("energies")})
        for t in sorted(csv.get(card["cardId"], set()) - seen):
            rows.append({"card": card, "kind": "card_text",
                         "effect_name": None, "text": t,
                         "attack_id": None, "attack_cost": None})
    return rows


@lru_cache(maxsize=1)
def build_kb() -> dict:
    """Classify the whole pool. Cached; pure over the dump and the CSV."""
    records: list[dict] = []
    unclassified: list[dict] = []
    for row in effect_corpus():
        got = classify_effect(row["card"], row["kind"], row["effect_name"],
                              row["text"], row["attack_id"],
                              row["attack_cost"])
        if got:
            records.extend(asdict(m) for m in got)
        elif _HAS_ENERGY.search(row["text"] or ""):
            unclassified.append({
                "card_id": row["card"]["cardId"], "name": row["card"]["name"],
                "effect_kind": row["kind"], "effect_name": row["effect_name"],
                "text": row["text"]})
    counts = Counter(r["mechanic"] for r in records)
    return {
        "records": records,
        "unclassified": unclassified,
        "counts": {c: counts.get(c, 0) for c in CLASSES},
        "cards_touched": len({r["card_id"] for r in records}),
        "pool_size": len(pool().by_id),
        "energy_mentioning_effects": sum(
            1 for r in effect_corpus() if _HAS_ENERGY.search(r["text"] or "")),
    }


# --- indexes the planner uses ----------------------------------------------

def by_card(mechanic: str | None = None) -> dict[int, list[dict]]:
    """card_id -> its mechanic records, optionally filtered to one class."""
    out: dict[int, list[dict]] = {}
    for r in build_kb()["records"]:
        if mechanic and r["mechanic"] != mechanic:
            continue
        out.setdefault(r["card_id"], []).append(r)
    return out


def accelerators() -> dict[int, list[dict]]:
    """card_id -> acceleration records. The planner's main lookup."""
    return by_card("acceleration")


def is_accelerator(card_id: int) -> bool:
    return card_id in accelerators()


def write_json(path: Path = OUT) -> Path:
    kb = build_kb()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(kb, indent=1, ensure_ascii=False))
    return path


# --- self-check -------------------------------------------------------------

# Cards whose classification is asserted rather than eyeballed, one per class.
_EXPECT = [
    ("Teal Mask Ogerpon ex", "acceleration", dict(
        source_zone="hand", rate=1, frequency="once_per_turn")),
    ("Marnie's Grimmsnarl ex", "acceleration", dict(
        source_zone="deck", rate=5, frequency="on_evolve")),
    ("Eelektrik", "acceleration", dict(
        source_zone="discard", rate=1, frequency="once_per_turn")),
    ("Energy Switch", "movement", dict(rate=1, frequency="item_play")),
    ("Energy Retrieval", "retrieval", dict(
        source_zone="discard", rate=2, frequency="item_play")),
    ("Energy Search", "tutor", dict(source_zone="deck", rate=1)),
    ("Pikachu ex", "discard_cost", dict(source_zone="in_play", rate=3)),
    ("Crushing Hammer", "disruption", dict(frequency="item_play")),
]


def _self_check(seed: int = 20260806) -> None:
    kb = build_kb()
    recs = kb["records"]
    idx: dict[tuple[str, str], list[dict]] = {}
    for r in recs:
        idx.setdefault((r["name"], r["mechanic"]), []).append(r)

    for name, mech, want in _EXPECT:
        got = idx.get((name, mech))
        assert got, f"{name}: no {mech} record"
        assert any(all(g[k] == v for k, v in want.items()) for g in got), \
            f"{name}/{mech}: wanted {want}, got {[{k: g[k] for k in want} for g in got]}"

    # Ownership rules: nothing on our side may be credited to the opponent.
    for r in recs:
        if r["mechanic"] in ("acceleration", "movement"):
            assert not _is_opponent(r["target"].get("raw", "")), r
        if r["mechanic"] == "disruption":
            assert "opponent" in r["source_zone"] or _is_opponent(
                r["sentence"]), r

    # Known false-positive traps stay out.
    assert ("Magearna", "acceleration") not in idx, \
        "Magearna's heal trigger is not acceleration"
    assert ("Pachirisu", "acceleration") not in idx, \
        "Pachirisu punishes the opponent's attach"

    # Every acceleration record names a source zone we can actually draw from.
    zones = Counter(r["source_zone"] for r in recs
                    if r["mechanic"] == "acceleration")
    assert set(zones) <= {"hand", "deck", "discard", "deck_top"}, zones

    print(f"energy_mechanics self-check: {len(_EXPECT)} asserted cards, "
          f"{len(recs)} records over {kb['cards_touched']} cards")
    print("counts:", kb["counts"])
    print(f"unclassified energy-mentioning effects: {len(kb['unclassified'])}"
          f" of {kb['energy_mentioning_effects']}")
    print("acceleration by source:", dict(zones))
    print("acceleration by frequency:",
          dict(Counter(r["frequency"] for r in recs
                       if r["mechanic"] == "acceleration")))

    rng = random.Random(seed)
    print("\n--- 10 randomized spot-checks (text vs classification) ---")
    for r in rng.sample(recs, 10):
        tgt = r["target"].get("raw") or "-"
        print(f"\n[{r['name']} / {r['effect_name']} / {r['effect_kind']}]")
        print(f"  text : {r['sentence']}")
        print(f"  class: {r['mechanic']}  {r['source_zone']}->{r['dest_zone']}"
              f"  rate={r['rate']}({r['rate_kind']})  freq={r['frequency']}"
              f"  target={tgt}"
              + (f"  scale={r['scale']}" if r["scale"] else ""))

    print("\n--- 5 randomized unclassified (the honest bucket) ---")
    for u in rng.sample(kb["unclassified"], min(5, len(kb["unclassified"]))):
        print(f"[{u['name']} / {u['effect_name']}] {u['text'][:160]}")


if __name__ == "__main__":
    _self_check()
    if "--write" in sys.argv:
        print("\nwrote", write_json())
