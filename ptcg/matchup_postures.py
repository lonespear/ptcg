"""Matchup deny-postures: the C-analysis, operationalized (decisions.md D30).

REPORT.md section C fitted, per top-8 archetype, which advantage components
THEY convert to wins (turn-7 logit AMEs, pp per SD) and how they lose (raced:
median 4 prizes still on their side). This module translates each finding
into a POSTURE SPEC: their win condition, the component to deny, the engine
Pokemon that embody it on their board, what OUR list (agent/deck.csv — Teal
Mask Ogerpon ex) can actually do about it, and a weight-delta over
agent.main's WEIGHTS that biases the forward search toward the denial.

Honesty constraints, stated once here and reflected in every spec:

- We cannot touch their draws, searches, or abilities directly. Our list has
  no ability lock, no item lock, and a single active-only attacker (Myriad
  Leaf Shower, attack 120: 30 + 30 x Energy on both Actives — no bench
  damage). Denial therefore means tempo pressure on their setup turns and
  KO-priority on their engine bodies, reachable only via Boss's Orders (x3,
  gust) followed by a KO.
- agent.main's evaluator has no per-target term: a weight-delta can prefer
  "their bench got smaller" (the `bench` term) but cannot name Munkidori
  over Impidimp. Named targeting is written below as `scoring_rule` text
  for the main.py integration pass (not ours to make — see task boundary).
- Judge x4 is our only draw-volume lever and hand size is not in the
  evaluator at all, so Judge prioritization is likewise a scoring_rule for
  integration, never a weight-delta.
- Where no lever exists in the current list (Mega Kangaskhan ex, the
  Ogerpon mirror), the spec says so; those rows feed deck refinement, not
  weights.

The one expressible lever
-------------------------
`WEIGHTS["bench"]` scores (our bench count - theirs), so raising it pays the
search extra for any line that removes a body from their bench — which, with
our action space, is exactly the Boss's Orders gust-KO on a 30-110 HP engine
Pokemon. Under the default vector a bench-body KO already scores about
prize 1000 + bench 153 + hp ~180 vs ~756 for a full 180-damage face hit on a
320 HP tank; what stays unpriced is the engine value the body would have
generated on their later turns (draws, evolutions, acceleration). That
future value is what section C measured, so the delta is scaled to it:

    bench_delta = BENCH_PER_PP_SD * deny_pp_per_sd * lever_factor

with lever_factor 1.0 when their engine sits on cheap gustable bodies, 0.5
when the component lives partly in trainers we cannot reach (Mega Lucario
ex), and 0.0 where no lever exists. BENCH_PER_PP_SD = 20 puts the largest
deltas at roughly +1.2x the default bench weight of 153 — big enough to
reorder gust lines, far too small to outbid a prize.

Weight changes only act through the forward search (`search=True`); the
rules-only policy consults WEIGHTS solely for the behind/ahead pwin trigger.
Screening probes must run search-on.

Provenance: data/analysis/archetypes_C.json (14,172 episodes, 2026-08-04..06,
rating cut 1000; regressions at t=7), REPORT.md section C one-liners, engine
Pokemon named from each archetype's most-played list in agent/deck_priors.json
with names from data/engine_dump/cards.json. Posterior identification rates
from scripts/validate_posterior.py (quoted in agent/main.py): top-1 correct
0.84 by turn 3, 0.88 by turn 5. Gust availability 0.711 = hypergeometric
P(>=1 of 3 Boss's Orders in the first 20 cards seen), before Pokegear x3.

Usage:
    from ptcg.matchup_postures import POSTURES, apply
    w = apply("Marnie's Grimmsnarl ex")          # full WEIGHTS dict
    pilot = JonDayPilot(search=True, weights=w)  # drop-in, main.py untouched

    python -m ptcg.matchup_postures   # regenerates data/analysis/postures.json
"""

from __future__ import annotations

import json
from pathlib import Path

BENCH_PER_PP_SD = 20.0

# Field share denominator: rated seats at the cut in archetypes_C.json meta.
_RATED_SEATS = 17203

# Measured posterior identification (top-1 correct), by our turn count.
POSTERIOR_ID = {"t1": 0.63, "t3": 0.84, "t5": 0.88, "t10": 0.94}

# P(>=1 of 3 Boss's Orders among the first 20 cards we see), hypergeometric;
# Pokegear 3.0 x3 (top-7 supporter search) raises the effective rate.
GUST_AVAILABILITY_20 = 0.711


def _spec(archetype, n_seats, win_condition, lose_condition, deny_component,
          deny_pp_per_sd, engine_pokemon, our_levers, lever_factor,
          scoring_rule, note=None):
    delta = ({"bench": round(BENCH_PER_PP_SD * deny_pp_per_sd * lever_factor, 1)}
             if lever_factor > 0 else None)
    return {
        "archetype": archetype,
        "win_condition": win_condition,
        "lose_condition": lose_condition,
        "deny_component": deny_component,
        "deny_pp_per_sd": deny_pp_per_sd,
        "engine_pokemon": engine_pokemon,
        "our_levers": our_levers,
        "lever_factor": lever_factor,
        "weight_delta": delta,
        "scoring_rule": scoring_rule,
        "expected_trigger": {
            "encounter_share": round(n_seats / _RATED_SEATS, 3),
            "posterior_id_by_t3": POSTERIOR_ID["t3"],
            "gust_availability": (GUST_AVAILABILITY_20
                                  if lever_factor > 0 else None),
            "activation": ("posterior top-1 == archetype at confidence >= "
                           "0.80 (agent/main.py CONFIDENCE_GATE); the delta "
                           "then rides every searched decision"),
        },
        "note": note,
    }


POSTURES = {
    "Marnie's Grimmsnarl ex": _spec(
        "Marnie's Grimmsnarl ex", 5134,
        "close the prize race (89% of wins) by turn 12 [10-14]",
        "raced: median 4 prizes still on their side at turn 12",
        "draw and search volume (cardsel_diff)", 8.6,
        [
            {"card_id": 646, "name": "Marnie's Impidimp", "copies": 4,
             "hp": 70, "role": "base of the 320 HP Grimmsnarl ex line"},
            {"card_id": 647, "name": "Marnie's Morgrem", "copies": 3,
             "hp": 100, "role": "stage 1 of the line"},
            {"card_id": 112, "name": "Munkidori", "copies": 4, "hp": 110,
             "role": "Adrena-Brain damage mover"},
            {"card_id": 860, "name": "Snorunt", "copies": 2, "hp": 70,
             "role": "base of Froslass (Freezing Shroud)"},
        ],
        {"real": ["Boss's Orders x3: gust-KO Impidimp/Morgrem before Rare "
                  "Candy resolves; Myriad Leaf Shower at 2+ Energy clears "
                  "any of these bodies",
                  "Judge x4: reset their hand during setup turns (their "
                  "draw volume is trainer-heavy: Poke Pad, Petrel, Lillie's)"],
         "unavailable": ["no direct draw denial; no ability lock for "
                         "Punk Up / Adrena-Brain"]},
        1.0,
        "gust preference order Impidimp > Morgrem > Munkidori > Snorunt when "
        "posterior says Grimmsnarl; bump Judge's card score on their setup "
        "turns (t<=4) — both need main.py card/target scoring, not weights",
        note="ladder already favors us 0.86 (n=527); the posture defends the "
             "edge rather than creating it"),

    "Mega Lopunny ex": _spec(
        "Mega Lopunny ex", 2758,
        "close the prize race (85% of wins) by turn 14 [12-16]",
        "raced: median 4 prizes still on their side at turn 12",
        "evolution line (evolve_diff)", 5.0,
        [
            {"card_id": 848, "name": "Buneary", "copies": 4, "hp": 70,
             "role": "base of the 330 HP Mega Lopunny ex line"},
            {"card_id": 305, "name": "Dunsparce", "copies": 4, "hp": 70,
             "role": "base of Dudunsparce"},
            {"card_id": 66, "name": "Dudunsparce", "copies": 4, "hp": 140,
             "role": "Run Away Draw engine"},
            {"card_id": 174, "name": "Fan Rotom", "copies": 1, "hp": 70,
             "role": "Fan Call opener"},
        ],
        {"real": ["Boss's Orders x3: gust-KO Buneary pre-evolution — every "
                  "Buneary down is a Mega Lopunny that never lands; "
                  "Dunsparce equally cheap",
                  "Judge x4: strips held Wally's Compassion / Hilda"],
         "unavailable": ["nothing stops Run Away Draw once Dudunsparce "
                         "stands"]},
        1.0,
        "gust preference Buneary > Dunsparce when posterior says Lopunny; "
        "value denial highest on turns 2-4 before Mega Lopunny lands"),

    "Fezandipiti ex": _spec(
        "Fezandipiti ex", 2658,
        "close the prize race (84% of wins) by turn 12 [9-14]",
        "raced: median 4 prizes still on their side at turn 11",
        "draw and search volume (cardsel_diff)", 4.9,
        [
            {"card_id": 741, "name": "Abra", "copies": 4, "hp": 50,
             "role": "base of the Psychic Draw line"},
            {"card_id": 742, "name": "Kadabra", "copies": 4, "hp": 80,
             "role": "Psychic Draw stage 1"},
            {"card_id": 743, "name": "Alakazam", "copies": 4, "hp": 140,
             "role": "Psychic Draw stage 2"},
            {"card_id": 66, "name": "Dudunsparce", "copies": 2, "hp": 140,
             "role": "Run Away Draw"},
            {"card_id": 140, "name": "Fezandipiti ex", "copies": 1, "hp": 210,
             "role": "Flip the Script"},
        ],
        {"real": ["Boss's Orders x3: 50 HP Abra is the cheapest gust-KO in "
                  "the format; Kadabra at 80 still one-shot territory",
                  "Judge x4 against a hand-sculpting deck"],
         "caution": ["their Enhanced Hammer x4 strips our Grow Grass Energy "
                     "x2 — energy denial runs the other way too"]},
        1.0,
        "gust preference Abra > Kadabra when posterior says Fezandipiti; "
        "their Flip the Script makes our own KOs feed their draw — prefer "
        "KO sequencing that closes prizes fast over chip"),

    "Mega Kangaskhan ex": _spec(
        "Mega Kangaskhan ex", 2111,
        "close the prize race (69% of wins) by turn 16, and 18% of wins by "
        "grinding past the prizes entirely (attrition)",
        "raced: median 4 prizes still on their side at turn 12",
        "evolution line / board development (evolve_diff)", 8.3,
        [
            {"card_id": 756, "name": "Mega Kangaskhan ex", "copies": 3,
             "hp": 300, "role": "Run Errand draw engine ON the attacker"},
            {"card_id": 96, "name": "Teal Mask Ogerpon ex", "copies": 3,
             "hp": 210, "role": "Teal Dance acceleration"},
            {"card_id": 1071, "name": "Meowth ex", "copies": 3, "hp": 170,
             "role": "Last-Ditch Catch"},
            {"card_id": 184, "name": "Latias ex", "copies": 2, "hp": 210,
             "role": "Skyliner mobility"},
        ],
        {"unavailable": ["every engine body is a 170-300 HP multi-prize ex: "
                         "gust-KO costs us 2+ turns of attacks and hands "
                         "them 2-prize trades on OUR 210 HP attacker; no "
                         "cheap denial target exists",
                         "their 18% attrition mode threatens our 18-energy "
                         "list with deck-out in long games"]},
        0.0,
        "none expressible: race them — their median winning turn (16) is the "
        "slowest in the top 8 and our t11 close beats it; deck refinement "
        "ask: bench damage or a cheap-prize secondary attacker",
        note="DENY LEVER ABSENT IN OUR LIST — feeds deck-refinement "
             "workstream; ladder says they beat us 0.78 (n=145)"),

    "Teal Mask Ogerpon ex": _spec(
        "Teal Mask Ogerpon ex", 1475,
        "close the prize race (92% of wins) by turn 11 [9-12] — the fastest "
        "close in the top 8",
        "raced: median 4 prizes still on their side at turn 10",
        "ability activations (Teal Dance)", 7.0,
        [
            {"card_id": 96, "name": "Teal Mask Ogerpon ex", "copies": 4,
             "hp": 210, "role": "Teal Dance accel + only attacker"},
        ],
        {"unavailable": ["no ability lock; every KO target is a 210 HP "
                         "2-prizer — denial IS the prize race, nothing "
                         "cheaper exists"],
         "real": ["Judge x4 still taxes their Lillie's/Judge-fueled setup; "
                  "went_first is worth +4.7 pp/SD to them, symmetric"]},
        0.0,
        "none expressible: the mirror is a pure race decided by energy "
        "tempo; C4's risk posture (buy variance when behind) is the correct "
        "instrument here, not a deny weight",
        note="DENY LEVER ABSENT — mirror; deck refinement ask: any card "
             "that breaks attachment symmetry"),

    "Dragapult ex": _spec(
        "Dragapult ex", 1455,
        "close the prize race (90% of wins) by turn 13 [11-15]",
        "raced: median 4 prizes still on their side at turn 12",
        "draw and search volume (cardsel_diff)", 8.9,
        [
            {"card_id": 119, "name": "Dreepy", "copies": 4, "hp": 70,
             "role": "base of the 320 HP Dragapult ex line"},
            {"card_id": 120, "name": "Drakloak", "copies": 4, "hp": 90,
             "role": "Recon Directive draw engine"},
            {"card_id": 112, "name": "Munkidori", "copies": 2, "hp": 110,
             "role": "Adrena-Brain"},
            {"card_id": 235, "name": "Budew", "copies": 2, "hp": 30,
             "role": "item-lock nuisance, free prize"},
        ],
        {"real": ["Boss's Orders x3: Drakloak is both their draw engine AND "
                  "their evolution step — the single highest-value gust-KO "
                  "in any matchup; Dreepy/Budew nearly free prizes",
                  "Judge x4"],
         "caution": ["their Crushing Hammer x4 + Jamming Tower pressure our "
                     "energy and tools; our worst ladder matchup (0.19, "
                     "n=98)"]},
        1.0,
        "gust preference Drakloak > Dreepy > Budew > Munkidori when "
        "posterior says Dragapult; this is the flagship posture for the "
        "matchup we most need to move"),

    "Mega Lucario ex": _spec(
        "Mega Lucario ex", 585,
        "close the prize race (85% of wins) by turn 11 [10-13]",
        "raced: median 4 prizes still on their side at turn 11",
        "draw and search volume (cardsel_diff)", 11.8,
        [
            {"card_id": 677, "name": "Riolu", "copies": 3, "hp": 80,
             "role": "base of the 340 HP Mega Lucario ex line"},
            {"card_id": 675, "name": "Lunatone", "copies": 2, "hp": 110,
             "role": "Lunar Cycle draw"},
            {"card_id": 674, "name": "Hariyama", "copies": 2, "hp": 150,
             "role": "Heave-Ho Catcher (their gust)"},
        ],
        {"partial": ["Boss's Orders x3: Riolu and Lunatone are gustable, "
                     "but the 11.8 pp/SD cardsel edge lives mostly in "
                     "Judge x4 / Lillie's x4 / Poke Pad x4 — trainers we "
                     "cannot reach",
                     "Judge x4: our best counter to their trainer draw, "
                     "and not a weight"]},
        0.5,
        "gust preference Riolu > Lunatone when posterior says Lucario; "
        "counter-Judge on their sculpted hands is the real lever and needs "
        "card-score integration"),

    "Cynthia's Garchomp ex": _spec(
        "Cynthia's Garchomp ex", 321,
        "close the prize race (89% of wins) by turn 13 [11-15]",
        "raced: median 4 prizes still on their side at turn 11",
        "ability activations (Champion's Call / Cheer On to Glory)", 8.0,
        [
            {"card_id": 379, "name": "Cynthia's Gible", "copies": 4,
             "hp": 70, "role": "base of the 330 HP Garchomp ex line"},
            {"card_id": 380, "name": "Cynthia's Gabite", "copies": 4,
             "hp": 100, "role": "Champion's Call search engine"},
            {"card_id": 341, "name": "Cynthia's Roselia", "copies": 4,
             "hp": 70, "role": "base of Roserade"},
            {"card_id": 342, "name": "Cynthia's Roserade", "copies": 3,
             "hp": 130, "role": "Cheer On to Glory"},
        ],
        {"real": ["Boss's Orders x3: Gabite IS the ability engine and dies "
                  "to any mid-game Myriad Leaf Shower; Gible/Roselia at 70 "
                  "are free",
                  "Judge x4"]},
        1.0,
        "gust preference Gabite > Gible > Roselia > Roserade when posterior "
        "says Garchomp",
        note="smallest sample in the top 8 (n=321 seats; only prize_diff "
             "and ability_diff clear significance)"),
}


def default_weights() -> dict:
    """agent.main's default vector, read from the module so main.py stays
    the single source of truth. Requires the engine bootstrap to have run
    (import ptcg.creation first) when agent.main is not yet loaded."""
    import agent.main as _jon
    return dict(_jon._DEFAULT_WEIGHTS)


def apply(archetype: str, base: dict | None = None) -> dict:
    """Full WEIGHTS dict for the posture against `archetype`.

    `base` defaults to agent.main's defaults. Unknown archetypes and
    no-lever postures return the base unchanged, so callers can key this
    straight off the posterior's top pick.
    """
    w = dict(base) if base is not None else default_weights()
    spec = POSTURES.get(archetype)
    if spec and spec["weight_delta"]:
        for k, dv in spec["weight_delta"].items():
            w[k] = w.get(k, 0.0) + dv
    return w


def all_deltas() -> dict:
    """{archetype: weight_delta or None} — the integration surface."""
    return {a: s["weight_delta"] for a, s in POSTURES.items()}


def write_json(path: str | Path = None) -> Path:
    root = Path(__file__).resolve().parents[1]
    path = Path(path) if path else root / "data" / "analysis" / "postures.json"
    payload = {
        "spec": "matchup deny-postures v1 (D30 component 1, exploit half)",
        "source": "ptcg/matchup_postures.py",
        "provenance": {
            "analysis": "data/analysis/archetypes_C.json + REPORT.md #C",
            "lists": "agent/deck_priors.json most-played per archetype",
            "names": "data/engine_dump/cards.json",
            "posterior_id": "scripts/validate_posterior.py via agent/main.py",
        },
        "bench_per_pp_sd": BENCH_PER_PP_SD,
        "postures": POSTURES,
    }
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n")
    return path


if __name__ == "__main__":
    out = write_json()
    print(f"wrote {out}")
    for a, s in POSTURES.items():
        d = s["weight_delta"]
        print(f"  {a}: {d if d else 'no expressible delta — ' + ('deck refinement' if s['lever_factor'] == 0 else '')}")
