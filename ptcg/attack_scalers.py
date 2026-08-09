"""The attack-scaler knowledge base: honest damage numbers for scaling attacks.

The agent's threat machinery (`agent/main.py::_attack_profile`) reads an
attack's printed damage, and a scaling attack prints the wrong number:
Alakazam's Powerful Hand prints 0 and deals 20 per card in its player's hand,
so a 20-card Dudunsparce engine registers as zero threat while it assembles a
one-shot. This module reads the effect text of every attack in the engine dump
and classifies the ones whose damage is a computable function of the OBSERVED
board into scaler records the agent can price at run time:

    hand_self          x per card in the attacker's hand
    hand_opp           x per card in the defender's hand
    energy_self        x per Energy attached to the attacking Pokemon
    energy_self_all    x per Energy attached to all of the attacker's Pokemon
    energy_opp_active  x per Energy attached to the defender's Active
    energy_opp_all     x per Energy attached to all of the defender's Pokemon
    energy_both_active x per Energy attached to the two Active Pokemon
                       (our own Ogerpon list's attack 120 is this kind)
    bench_self         x per Pokemon on the attacker's Bench
    bench_opp          x per Pokemon on the defender's Bench
    bench_both         x per Benched Pokemon, both sides
    in_play_self       x per Pokemon the attacker has in play
    dmg_self           x per damage counter on the attacking Pokemon
    dmg_opp_active     x per damage counter on the defender's Active
    prizes_taken_self  x per Prize card the attacker has taken
    prizes_taken_opp   x per Prize card the defender has taken
    in_play_self_team_rocket
                       x per in-play Pokemon named "Team Rocket's ..."
    if_from_bench      +x when the attacker switched in this turn (priced
                       when the attacker is benched; see _FROM_BENCH)
    flat               no scaling: fixed effect damage the printed number
                       omits — a damage-counter placement, or a zero-print
                       "does N damage to <opponent target>" attack such as
                       Fezandipiti ex's Cruel Arrow (same blindness class,
                       constant rather than scaling)

"Place N damage counters ... for each X" classifies at 10*N per unit — a
damage counter is 10 damage. Damage is `max(base + per * quantity, 0)`; a
"less damage" scaler stores a negative `per`.

Precision over recall, exactly as `ptcg/energy_mechanics.py` holds it: a
damage-scaling sentence that does not match a rule lands the whole attack in
the `unclassified` bucket and is reported, never forced into a class. Coin
flips, discard-priced scaling ("for each card you discarded in this way"),
named-card counts and typed-qualifier board counts are unclassified by
design — their quantity is not an observable the agent already reads.
A typed Energy count ("{W} Energy attached to this Pokemon") IS classified,
priced by the slot's total Energy count, which upper-bounds it; mono-typed
attachers, which is what plays these attacks, make the bound tight.

Corpus: `data/engine_dump/attacks.json` (attackId, printed damage, effect
text). The committed KB (`agent/attack_scalers.json`) holds attackId and
numeric parameters ONLY — the engine's card text is licensed for competition
use and never enters the repo.

    python -m ptcg.attack_scalers          # self-check + classification counts
    python -m ptcg.attack_scalers --write  # regenerate agent/attack_scalers.json
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DUMP = ROOT / "data" / "engine_dump" / "attacks.json"
OUT = ROOT / "agent" / "attack_scalers.json"

# The dump uses curly apostrophes throughout; tolerate the straight one.
APO = "[’']"

# --- text plumbing ----------------------------------------------------------

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def split_sentences(text: str) -> list[str]:
    if not text:
        return []
    return [re.sub(r"\s+", " ", s).strip()
            for s in _SENT_SPLIT.split(text) if s.strip()]


# --- unit rules -------------------------------------------------------------
# Each rule must match the WHOLE unit phrase (the text after "for each") up to
# the sentence's closing period or a trailing parenthetical, so a qualified
# phrase ("Benched {F} Pokémon", "Pokémon in play that has the Round attack")
# falls through to unclassified instead of matching its unqualified prefix.

_TYPE = r"(?:\{\w+\} )?"          # an optional {W}/{P}/... type qualifier
_TAIL = r"\s*(?:\(.*\))?\s*$"     # optional reminder text, end of sentence

UNIT_RULES: list[tuple[str, re.Pattern]] = [
    ("hand_self", re.compile(r"^card in your hand" + _TAIL, re.I)),
    ("hand_opp", re.compile(
        rf"^card in your opponent{APO}s hand" + _TAIL, re.I)),
    ("energy_self", re.compile(
        rf"^{_TYPE}Energy attached to this Pok\S+" + _TAIL, re.I)),
    ("energy_self_all", re.compile(
        rf"^{_TYPE}Energy attached to all of your Pok\S+" + _TAIL, re.I)),
    ("energy_opp_active", re.compile(
        rf"^{_TYPE}Energy attached to your opponent{APO}s Active Pok\S+"
        + _TAIL, re.I)),
    ("energy_opp_all", re.compile(
        rf"^{_TYPE}Energy attached to all of your opponent{APO}s Pok\S+"
        + _TAIL, re.I)),
    ("energy_both_active", re.compile(
        rf"^{_TYPE}Energy attached to both Active Pok\S+" + _TAIL, re.I)),
    ("bench_self", re.compile(
        r"^of your Benched Pok\S+" + _TAIL, re.I)),
    ("bench_opp", re.compile(
        rf"^of your opponent{APO}s Benched Pok\S+" + _TAIL, re.I)),
    ("bench_both", re.compile(
        rf"^Benched Pok\S+ \(both yours and your opponent{APO}s\)" + _TAIL,
        re.I)),
    ("in_play_self", re.compile(
        r"^of your Pok\S+ in play" + _TAIL, re.I)),
    # Name-filtered board count (audit addition: Rocket Rush realizes ~180
    # from a printed 0). The runtime counts in-play names with the
    # "Team Rocket's" prefix off the engine's own card table; other owner
    # prefixes (Erika's, Cynthia's, ...) stay unclassified.
    ("in_play_self_team_rocket", re.compile(
        rf"^of your Team Rocket{APO}s Pok\S+ in play" + _TAIL, re.I)),
    ("dmg_self", re.compile(
        r"^damage counter on this Pok\S+" + _TAIL, re.I)),
    ("dmg_opp_active", re.compile(
        rf"^damage counter on your opponent{APO}s Active Pok\S+" + _TAIL,
        re.I)),
    ("prizes_taken_self", re.compile(
        r"^Prize card you have taken" + _TAIL, re.I)),
    ("prizes_taken_opp", re.compile(
        r"^Prize card your opponent has taken" + _TAIL, re.I)),
]

KINDS = tuple(k for k, _ in UNIT_RULES) + ("flat", "if_from_bench")


def _classify_unit(phrase: str) -> str | None:
    for kind, rule in UNIT_RULES:
        if rule.match(phrase):
            return kind
    return None


# --- sentence rules ---------------------------------------------------------
# A damage-scaling sentence, in the four shapes the dump prints. The unit
# phrase is everything after "for each "; targeted forms ("to 1 of your
# opponent's Pokémon") still deal the number, so the threat model keeps them.

_SCALE = re.compile(
    rf"^This attack (?:also )?does (\d+) (more |less )?damage"
    rf"(?: to (?:1 of )?your opponent{APO}s(?: Benched)? Pok\S+"
    rf"(?: \S+)*?)? for each (.+?)\.?$",
    re.I)
_COUNTERS_EACH = re.compile(
    rf"^(?:Place|Put) (\d+) damage counters? on (?:1 of |each of )?"
    rf"your opponent{APO}s(?: Active| Benched)? Pok\S+ for each (.+?)\.?$",
    re.I)
_COUNTERS_FLAT = re.compile(
    rf"^(?:Place|Put) (\d+) damage counters? on (?:1 of |each of )?"
    rf"your opponent{APO}s(?: Active| Benched)? Pok\S+\.?$",
    re.I)
# Flat effect damage: "This attack does N damage to <opponent target>", no
# scaling — Fezandipiti ex's Cruel Arrow prints 0 and does 100 to any
# Pokemon. The leading alternation admits a self-cost clause ("Discard all
# Energy from this Pokemon, and this attack does ..."), which the printed-
# damage model has always ignored on printed attacks and stays ignored here.
# A qualified target ("Pokemon {ex}") fails the tail and stays unclassified.
_FLAT_DMG = re.compile(
    rf"^(?:\S[^.]*?, and this|This) attack does (\d+) damage to "
    rf"(?:(?:(?:1|2|3|each|each of 2) of )?your opponent{APO}s"
    rf"(?: Active| Benched)? Pok\S+|the new Active Pok\S+)\.?" + _TAIL,
    re.I)
# The Gale Thrust rider (audit addition): +N when the attacker switched in
# this turn. Whether it switched in is not in the observation, so the
# runtime prices the rider when the attacker is on the BENCH — a benched
# body can always switch in on its own turn — and holds an Active attacker
# at base, the documented precision-first approximation.
_FROM_BENCH = re.compile(
    r"^If this Pok\S+ moved from your Bench to the Active Spot this turn, "
    r"this attack does (\d+) more damage\.?$", re.I)
# Any sentence that talks about this attack's damage moving, places damage
# counters, or deals effect damage — the trigger for "this attack is not its
# printed number". A trigger sentence that no rule above parses sends the
# attack to unclassified.
_TRIGGER = re.compile(
    r"for each|damage counters? on|damage times|more damage|less damage"
    r"|does \d+ damage to (?!itself)",
    re.I)
# Damage-counter sentences that are NOT opponent-side damage (own-side setup,
# healing, movement) are riders, not damage — never a trigger on their own.
_COUNTER_RIDER = re.compile(
    rf"counters? on (?:this|1 of your(?! opponent)|each of your(?! opponent)"
    rf"|your(?! opponent)|all of your(?! opponent))",
    re.I)


def classify_attack(attack: dict) -> tuple[dict | None, str | None]:
    """(record, None) for a classified attack, (None, reason) otherwise.

    Record: {"id", "base", "per", "kind"} — numbers only, no text. An attack
    with no damage-moving text at all returns (None, None): its printed
    number is already honest.
    """
    text = re.sub(r"\s+", " ", attack.get("text") or "").strip()
    printed = int(attack.get("damage") or 0)
    if not text:
        return None, None
    if "does nothing" in text.lower():
        # A number gated on a condition ("if you don't have 10 or more ...,
        # this attack does nothing") is not an honest constant.
        return None, "conditional: attack can do nothing"
    sentences = split_sentences(text)
    triggers = [s for s in sentences
                if _TRIGGER.search(s) and not (
                    "damage counter" in s.lower() and _COUNTER_RIDER.search(s)
                    and "for each" not in s.lower())]
    if not triggers:
        return None, None

    parsed: list[tuple[int, int, str]] = []   # (base, per, kind)
    for s in triggers:
        m = _SCALE.match(s)
        if m:
            n, mode, unit = int(m.group(1)), (m.group(2) or "").strip(), \
                m.group(3)
            kind = _classify_unit(unit)
            if kind is None:
                return None, f"unit: {unit[:60]}"
            if mode == "more":
                parsed.append((printed, n, kind))
            elif mode == "less":
                parsed.append((printed, -n, kind))
            else:
                # A total that replaces the printed number is only unambiguous
                # when the printed number is 0.
                if printed:
                    return None, "printed>0 with a replacing scaler"
                parsed.append((0, n, kind))
            continue
        m = _COUNTERS_EACH.match(s)
        if m:
            unit = m.group(2)
            kind = _classify_unit(unit)
            if kind is None:
                return None, f"unit: {unit[:60]}"
            parsed.append((printed, 10 * int(m.group(1)), kind))
            continue
        m = _COUNTERS_FLAT.match(s)
        if m and "up to" not in s.lower():
            parsed.append((printed + 10 * int(m.group(1)), 0, "flat"))
            continue
        m = _FLAT_DMG.match(s)
        if m:
            if printed:
                return None, "printed>0 with flat effect damage"
            parsed.append((int(m.group(1)), 0, "flat"))
            continue
        m = _FROM_BENCH.match(s)
        if m:
            parsed.append((printed, int(m.group(1)), "if_from_bench"))
            continue
        return None, f"sentence: {s[:60]}"

    if len(parsed) != 1:
        return None, "multiple scaling sentences"
    base, per, kind = parsed[0]
    return {"id": int(attack["attackId"]), "base": base, "per": per,
            "kind": kind}, None


def build() -> tuple[list[dict], list[tuple[int, str]]]:
    attacks = json.loads(DUMP.read_text())
    records, unclassified = [], []
    for a in attacks:
        rec, reason = classify_attack(a)
        if rec is not None:
            records.append(rec)
        elif reason is not None:
            unclassified.append((int(a["attackId"]), reason))
    return records, unclassified


def _selfcheck(records: list[dict]) -> None:
    by_id = {r["id"]: r for r in records}
    # Powerful Hand: printed 0, 2 counters per card in own hand.
    assert by_id[1072] == {"id": 1072, "base": 0, "per": 20,
                           "kind": "hand_self"}, by_id.get(1072)
    # Cruel Arrow: printed 0, flat 100 to any of their Pokemon.
    assert by_id[183] == {"id": 183, "base": 100, "per": 0,
                          "kind": "flat"}, by_id.get(183)
    # Rocket Rush: printed 0, 30 per Team Rocket's Pokemon in play.
    assert by_id[560] == {"id": 560, "base": 0, "per": 30,
                          "kind": "in_play_self_team_rocket"}, by_id.get(560)
    # Gale Thrust (Mega Lopunny ex): printed 60, +170 on the switch-in.
    assert by_id[1225] == {"id": 1225, "base": 60, "per": 170,
                           "kind": "if_from_bench"}, by_id.get(1225)
    for r in records:
        assert r["kind"] in KINDS
        assert r["per"] != 0 or r["kind"] == "flat"
        assert r["kind"] != "flat" or (r["per"] == 0 and r["base"] > 0)
    print(f"self-check OK ({len(records)} records)")


def main() -> None:
    records, unclassified = build()
    _selfcheck(records)
    print(f"classified {len(records)} attacks, "
          f"{len(unclassified)} unclassified")
    for kind, n in Counter(r["kind"] for r in records).most_common():
        print(f"  {kind:18s} {n}")
    reasons = Counter(reason.split(":")[0] for _, reason in unclassified)
    print("unclassified by reason:", dict(reasons))
    if "--verbose" in sys.argv:
        for aid, reason in unclassified:
            print(f"  {aid:5d}  {reason}")
    if "--write" in sys.argv:
        blob = {"generator": "ptcg/attack_scalers.py",
                "source": "data/engine_dump/attacks.json (effect text; "
                          "text itself never enters this file)",
                "attacks": {str(r["id"]): {"base": r["base"], "per": r["per"],
                                           "kind": r["kind"]}
                            for r in sorted(records, key=lambda r: r["id"])}}
        OUT.write_text(json.dumps(blob, indent=1) + "\n")
        print(f"wrote {OUT} ({len(records)} attacks)")


if __name__ == "__main__":
    main()
