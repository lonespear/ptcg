"""Build the FIELD panel: real ladder decks, piloted by a frozen copy of us.

Motivation (decisions.md D58). The six `external/*_deck.json` specialist cells
misrepresent the ladder three measured ways — 24.9% opponent forfeit rate
against 0.0% on the ladder, zero damage-wall cards against 3 of 79 real
opponent lists, and a lethal-decline frequency that made a +13pt panel gain a
0pt ladder gain. This panel replaces the decks and the drivers at once:

  * DECKS come from `data/analysis/field_decks.json` — 60-card lists that real
    opponents actually brought, weighted by how often we met them in the
    700-1000 rating band where our submissions play.
  * DRIVERS are a frozen copy of OUR OWN agent (default: the v6 bundle), one
    process per cell, deck injected through `ExternalPilot._MY_DECK`. Our agent
    has never forfeited once in 38,000 recorded gate games, so cell forfeit
    rates go to ~0 and D54's exclusion stops eating a quarter of the sample.

WHAT THIS PANEL MEASURES, AND WHAT IT DOES NOT.
  Measures: deck and mechanic matchups. Whether our 60 cards and our rules
  model hold up against the 60 cards the field actually plays, including the
  damage-wall lists our old panel could not express at all.
  Does NOT measure: opponent skill. Every cell is driven by the same frozen
  policy, so a cell's win rate is our deck against their deck under one fixed
  pilot, not against the human-tuned agent that brought it. Real opponents in
  this corpus rated 461 to 982 and their policies differ wildly; the panel
  flattens that to a single competent driver. A cell therefore reads matchup
  advantage, not ladder outcome, and the aggregate is only expected to track
  the ladder to the extent the field's average policy resembles our own.

    python scripts/build_field_panel.py --out data/analysis/field_panel.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

WALLS = {83, 117, 158, 207, 330, 345}
# Stadiums our damage model never reads (D60): Full Metal Lab is -30 against
# Metal defenders, Neutralization Zone zeroes an all-ex board's damage against
# non-Rule-Box defenders. All 31 over-predicted attacks in the ladder corpus
# trace to one of them.
DAMAGE_STADIUMS = {1244: "Full Metal Lab", 1247: "Neutralization Zone"}

# Cells, in field-frequency order within the 700-1000 band, plus MECHANIC
# cells for effects the old panel could not express at all: damage walls, and
# the two Stadiums that break our damage model. `deck_id` keys into
# field_decks.json.
CELLS = [
    ("grimmsnarl", "field00", "frequency"),      # 26.3% of 700-1000 episodes
    ("archaludon", "field05", "frequency"),      # 19.3%; 4x Full Metal Lab
    ("fezandipiti", "field01", "frequency"),     # 17.5%
    ("dragapult", "field16", "frequency"),       #  7.0%
    ("lucario", "field02", "frequency"),         #  7.0%
    ("dudunsparce", "field04", "frequency"),     #  7.0%; 1x Neutralization Zone
    ("ogerpon_mirror", "field06", "frequency"),  #  3.5%
    ("starmie", "field10", "frequency"),         #  3.5%
    ("wall_lopunny", "field24", "mechanic"),     # 117 Cornerstone Mask
    ("wall_kangaskhan", "field21", "mechanic"),  # 345 Crustle, 4 copies
    ("stadium_nz", "field14", "mechanic"),       # 1247 Neutralization Zone
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(ROOT / "data/analysis/field_decks.json"))
    ap.add_argument("--pilot-src",
                    default="/private/tmp/claude-501/-Users-austinsemmel-Desktop/"
                            "4a2826ba-e7f4-4ad7-9712-4ec201fdff65/scratchpad/"
                            "v6_main_backup.py",
                    help="frozen agent copy that drives every field deck")
    ap.add_argument("--rig", default="/private/tmp/claude-501/"
                    "-Users-austinsemmel-Desktop/"
                    "4a2826ba-e7f4-4ad7-9712-4ec201fdff65/scratchpad/fieldpanel")
    ap.add_argument("--out", default=str(ROOT / "data/analysis/field_panel.json"))
    ap.add_argument("--manifest", default=str(ROOT / "data/analysis/FIELD_PANEL.md"))
    args = ap.parse_args()

    corpus = json.loads(Path(args.corpus).read_text())
    by_id = {d["deck_id"]: d for d in corpus["decks"]}
    eps = corpus["episodes"]

    hi = [e for e in eps if e["opp_score_pre"] and 700 <= e["opp_score_pre"] < 1000]
    arch_hi = Counter(e["opp_archetype"] for e in hi)
    arch_all = Counter(e["opp_archetype"] for e in eps)

    rig = Path(args.rig)
    rig.mkdir(parents=True, exist_ok=True)
    src = Path(args.pilot_src)
    if not src.exists():
        raise SystemExit(f"frozen pilot not found: {src}")

    # Observed carriage of the two damage Stadiums, for weighting and for the
    # report the Stadium-fix workstream needs.
    stadium_sigs = {d["sig"] for d in corpus["decks"]
                    if any(int(c) in DAMAGE_STADIUMS for c in d["deck"])}
    stad_eps = [e for e in eps if e["opp_sig"] in stadium_sigs]
    stad_hi = [e for e in hi if e["opp_sig"] in stadium_sigs]
    nz_sigs = {d["sig"] for d in corpus["decks"] if "1247" in d["deck"]}
    nz_hi = [e for e in hi if e["opp_sig"] in nz_sigs]

    cells = []
    for name, did, kind in CELLS:
        d = by_id[did]
        deck = sorted(int(c) for c, n in d["deck"].items() for _ in range(n))
        assert len(deck) == 60, (name, len(deck))
        pdir = rig / f"pilot_{name}"
        pdir.mkdir(exist_ok=True)
        shutil.copy(src, pdir / "main.py")
        for aux in ("deck_priors.json", "attack_scalers.json"):
            shutil.copy(ROOT / "agent" / aux, pdir / aux)
        (pdir / "deck.csv").write_text("\n".join(str(c) for c in deck) + "\n")
        dj = rig / f"{name}_deck.json"
        dj.write_text(json.dumps(deck))

        a = d["archetype"]
        dk = {int(c): n for c, n in d["deck"].items()}
        stadiums = {DAMAGE_STADIUMS[c]: dk[c] for c in DAMAGE_STADIUMS if c in dk}
        # Frequency cells carry their archetype's band share. Mechanic cells
        # carry the observed rate of the mechanic itself, which is what makes
        # a rare-but-real effect measurable without pretending it is common.
        if kind == "frequency":
            weight = arch_hi[a] / len(hi)
        elif name == "stadium_nz":
            weight = len(nz_hi) / len(hi)
        else:
            weight = arch_hi[a] / len(hi)
        cells.append({
            "cell": name, "deck_id": did, "archetype": a, "kind": kind,
            "pilot": str(pdir / "main.py"),
            "deck_json": str(dj),
            "damage_stadiums": stadiums,
            "field_episodes_all": arch_all[a],
            "field_episodes_700_1000": arch_hi[a],
            "weight": round(weight, 4),
            "list_episodes": d["n_episodes"],
            "opp_rating_mean": d["rating_mean"],
            "opp_rating_range": [d["rating_min"], d["rating_max"]],
            "opp_win_rate_vs_us_live": d["opp_win_rate_vs_us"],
            "walls": d["walls"],
            "teams": d["teams"],
        })

    covered = sum(c["field_episodes_700_1000"] for c in
                  {c["archetype"]: c for c in cells}.values())
    payload = {
        "built_from": args.corpus,
        "pilot_src": str(src),
        "pilot_note": ("every cell is driven by ONE frozen copy of our own "
                       "agent; cells measure deck/mechanic matchup, not "
                       "opponent skill"),
        "band": "700-1000 (weights), corpus spans 461-982",
        "n_field_episodes": len(eps),
        "n_band_episodes": len(hi),
        "archetype_coverage_700_1000": round(covered / len(hi), 4),
        "damage_stadium_carriage": {
            "lists": len(stadium_sigs), "of_lists": len(corpus["decks"]),
            "episodes": len(stad_eps), "of_episodes": len(eps),
            "rate_all": round(len(stad_eps) / len(eps), 4),
            "rate_700_1000": round(len(stad_hi) / len(hi), 4),
            "live_win_rate_vs": round(
                sum(1 for e in stad_eps if e["won"]) / len(stad_eps), 4),
            "live_win_rate_elsewhere": round(
                sum(1 for e in eps if e["opp_sig"] not in stadium_sigs
                    and e["won"])
                / max(1, sum(1 for e in eps if e["opp_sig"] not in stadium_sigs)), 4),
            "nz_rate_700_1000": round(len(nz_hi) / len(hi), 4),
        },
        "cells": cells,
    }
    Path(args.out).write_text(json.dumps(payload, indent=1))
    print(f"wrote {args.out} ({len(cells)} cells, "
          f"{payload['archetype_coverage_700_1000']:.1%} archetype coverage)")

    L = ["# Field panel — cells, weights and honest limits", "",
         f"Built from `{Path(args.corpus).name}`: {len(eps)} real ladder "
         f"episodes, {len(hi)} of them against opponents rated 700-1000. "
         f"Frequency cells are weighted by their archetype's share of the "
         f"700-1000 band and cover "
         f"{payload['archetype_coverage_700_1000']:.1%} of it; mechanic cells "
         f"are weighted by the observed rate of the mechanic itself.",
         "",
         "Every cell is driven by a FROZEN copy of our own agent "
         f"(`{src.name}`), deck injected per call, so a cell's opponent plays "
         "at a known and constant strength. Note that the 24.9% opponent "
         "forfeit rate D54 attributed to the external specialists was our own "
         "`ptcg/arena.py` bug — an empty selection is legal whenever a "
         "prompt's minCount is 0 — and is fixed as of D60. Every number here "
         "was produced with the fixed harness.",
         "",
         "## Cells", "",
         "| cell | kind | archetype | weight | field eps (band / all) | "
         "list eps | opp rating mean | opp WR live | walls | damage stadiums |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for c in cells:
        st = ", ".join(f"{k} x{v}" for k, v in c["damage_stadiums"].items())
        L.append(f"| {c['cell']} | {c['kind']} | {c['archetype']} | "
                 f"{c['weight']:.3f} | "
                 f"{c['field_episodes_700_1000']} / {c['field_episodes_all']} | "
                 f"{c['list_episodes']} | {c['opp_rating_mean']} | "
                 f"{c['opp_win_rate_vs_us_live']:.2f} | "
                 f"{', '.join(str(w) for w in c['walls']) or '-'} | "
                 f"{st or '-'} |")
    ds = payload["damage_stadium_carriage"]
    L += ["", "## Damage-Stadium carriage (D60)", "",
          f"Full Metal Lab (1244) and Neutralization Zone (1247) appear in "
          f"{ds['lists']} of {ds['of_lists']} distinct field lists and "
          f"{ds['episodes']} of {ds['of_episodes']} episodes "
          f"({ds['rate_all']:.1%}; {ds['rate_700_1000']:.1%} of the 700-1000 "
          f"band). Our live record against those lists is "
          f"{ds['live_win_rate_vs']:.3f} against "
          f"{ds['live_win_rate_elsewhere']:.3f} everywhere else.",
          "",
          "Cells that carry them: `archaludon` (4x Full Metal Lab), "
          "`dudunsparce` (1x Neutralization Zone), `stadium_nz` (1x "
          "Neutralization Zone, Fezandipiti shell). Full Metal Lab is the "
          "reliably exercised one — nine of the eleven carrying lists are "
          "Archaludon builds running 3-4 copies. Neutralization Zone is never "
          "run above one copy anywhere in the field, so an NZ cell exercises "
          "the effect in only a minority of its games and needs more games, "
          "not more weight, to resolve a fix.", ""]

    thin = [c for c in cells if c["list_episodes"] <= 2]
    L += ["", "## What is thin, stated honestly", "",
          "* Cells built from one or two observed episodes are a real list "
          "played by a real opponent, but they are one sample of that "
          "archetype's list space, not its modal build: "
          + ", ".join(f"`{c['cell']}` ({c['list_episodes']} ep)" for c in thin)
          + ".",
          "* Wall lists are 3 of 79 field episodes (3.8%) and all three were "
          "seen only in the 900-1000 band. They are carried here at their "
          "band weight (1.8% each) so the mechanic is testable at all; a "
          "wall-cell result is a mechanic check, not a field-share claim.",
          "* Archetypes below the cut (Thwackey, Cynthia's Garchomp ex, "
          "Team Rocket's Mewtwo ex, Hop's Snorlax, Mega Abomasnow ex, "
          "Alakazam, Ceruledge ex) are 1-2 episodes each and are not "
          "represented.",
          "* Opponent SKILL is not modelled. One frozen policy drives every "
          "cell. Field opponents rated 461-982 and played their own agents; "
          "a cell reads our deck against their deck, not our agent against "
          "their agent.",
          "* OUR list is held fixed in every cell, so the panel cannot test "
          "the deck-axis mismatches at all: the field averages 13.1 one-prize "
          "attackers per episode and 5.7 distinct Pokemon species, while our "
          "60 holds four Pokemon cards of a single species, every one of them "
          "an ex. Any candidate that changes the 60 needs its own deck gate.",
          ""]
    Path(args.manifest).write_text("\n".join(L))
    print(f"wrote {args.manifest}")


if __name__ == "__main__":
    main()
