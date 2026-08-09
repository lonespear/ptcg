"""Build the ladder-representative fitness panel v3 (D46).

Source: data/analysis/ladder_autopsy.json — 187 classified ladder episodes
with full opponent deck lists and score bands. Cells are REAL field lists
(deduped by deck signature); cell weights are the archetype's encounter
share in the 700+ bands (700-800 / 800-900 / 900+, the scoring
neighborhood we actually play in), split across the archetype's variants
by their high-band game counts. Sub-1% archetypes that earn a cell get a
1% weight floor. Two non-autopsy cells are added deliberately:

  - AlphaStarmie's exact Dudunsparce-Alakazam list (recovered from its
    2026-08-06 replays) — the strongest stall list on the ladder;
  - our own shipped v4 Ogerpon list (agent/deck.csv) as an opponent —
    we get netdecked, so the mirror cell plays our exact 60.

Cell counts per archetype: every distinct variant for the big four
(Grimmsnarl capped at ~20% of cells), floor of one cell for each named
sub-5% archetype (mono-Kangaskhan, Ogerpon mirror, Mega Lopunny,
Dragapult, Crustle, the recurring Cynthia-Fighting and Spidops "other"
decks).

    python scripts/build_panel_v3.py \
        --out data/panel_ladder_v3.json \
        --manifest data/analysis/PANEL_V3.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ptcg.creation.pool import pool  # noqa: E402
from ptcg.creation.validator import validate  # noqa: E402

HI_BANDS = {"700-800", "800-900", "900+"}

# archetype -> max cells (None = all distinct variants)
PLAN = [
    ("Grimmsnarl/Munkidori", 5),
    ("Dudunsparce-Alakazam", 7),          # AlphaStarmie exact dedupes in
    ("Archaludon-Duraludon-Cinderace", 6),
    ("Mega Lucario", 5),
    ("Kangaskhan (mono)", 3),
    ("Ogerpon (mirror)", 3),              # + our shipped v4 = 4
    ("Mega Lopunny", 2),
    ("Dragapult", 2),
    ("Crustle", 1),
]
# recurring "other" decks that earn a floor cell (matched by prefix)
OTHER_FLOORS = [
    ("Cynthia-Fighting", "other [Cynthia's Roselia x4, Cynthia's Gible x4"),
    ("Spidops", "other [Team Rocket's Tarountula x4"),
]
WEIGHT_FLOOR = 0.01
ALPHASTARMIE = {
    "5": 2, "13": 1, "19": 4, "66": 2, "140": 1, "305": 3, "343": 1,
    "741": 4, "742": 4, "743": 4, "1079": 3, "1081": 4, "1086": 4,
    "1097": 1, "1129": 1, "1152": 4, "1182": 3, "1184": 1, "1197": 3,
    "1225": 4, "1231": 4, "1266": 2,
}


def deck_list(c: dict) -> list[int]:
    return [int(k) for k, n in c.items() for _ in range(int(n))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--autopsy", default="data/analysis/ladder_autopsy.json")
    ap.add_argument("--out", default="data/panel_ladder_v3.json")
    ap.add_argument("--manifest", default="data/analysis/PANEL_V3.md")
    args = ap.parse_args()

    autopsy = json.loads((ROOT / args.autopsy).read_text())
    recs = [r for r in autopsy["records"] if r.get("opp_deck")]
    hi_total = sum(1 for r in recs if r["band"] in HI_BANDS)

    # variants per archetype
    var = defaultdict(dict)          # arch -> sig -> info
    arch_hi = Counter()
    for r in recs:
        s = var[r["archetype"]].setdefault(
            r["opp_deck_sig"],
            {"hi": 0, "lo": 0, "team": r["opp_team"], "c": r["opp_deck"]})
        hi = r["band"] in HI_BANDS
        s["hi" if hi else "lo"] += 1
        if hi:
            arch_hi[r["archetype"]] += 1

    p = pool()
    cells = []                       # (name, team, c-dict, arch, hi, lo)

    def add(name, team, c, arch, hi, lo):
        d = deck_list(c)
        v = validate(d)
        if len(d) != 60 or not v.legal:
            print(f"  SKIP illegal: {name} ({v.problems if hasattr(v, 'problems') else 'illegal'})")
            return
        cells.append({"name": name, "team": team, "c": dict(c),
                      "arch": arch, "hi": hi, "lo": lo})

    for arch, cap in PLAN:
        sigs = sorted(var[arch].values(),
                      key=lambda s: (-s["hi"], -s["lo"]))[:cap]
        alpha = tuple(sorted(deck_list(ALPHASTARMIE)))
        for k, s in enumerate(sigs, 1):
            name = f"{arch} #{k} ({s['team']})"
            if (arch == "Dudunsparce-Alakazam"
                    and tuple(sorted(deck_list(s["c"]))) == alpha):
                name = f"{arch} #{k} ({s['team']} = AlphaStarmie exact)"
            add(name, s["team"], s["c"], arch, s["hi"], s["lo"])
        if arch == "Dudunsparce-Alakazam":
            taken = {tuple(sorted(deck_list(s["c"]))) for s in sigs}
            if alpha not in taken:
                add(f"{arch} #X (AlphaStarmie exact)", "AlphaStarmie",
                    ALPHASTARMIE, arch, 1, 0)
        if arch == "Ogerpon (mirror)":
            ours = Counter(int(x) for x in
                           (ROOT / "agent" / "deck.csv").read_text().split())
            add(f"{arch} #V4 (our shipped list)", "us",
                {str(k): v for k, v in ours.items()}, arch, 1, 0)

    for label, prefix in OTHER_FLOORS:
        matches = [a for a in var if a.startswith(prefix)]
        if not matches:
            continue
        sigs = sorted((s for a in matches for s in var[a].values()),
                      key=lambda s: (-s["hi"], -s["lo"]))
        s = sigs[0]
        add(f"{label} #1 ({s['team']})", s["team"], s["c"],
            label, s["hi"], s["lo"])

    # weights: archetype share of hi-band encounters, floored; split
    # across the archetype's cells by (hi games + 0.5)
    arch_cells = defaultdict(list)
    for cell in cells:
        arch_cells[cell["arch"]].append(cell)
    for arch, cs in arch_cells.items():
        share = max(sum(c["hi"] for c in cs) / hi_total, WEIGHT_FLOOR)
        z = sum(c["hi"] + 0.5 for c in cs)
        for c in cs:
            c["w"] = share * (c["hi"] + 0.5) / z
    z = sum(c["w"] for c in cells)
    for c in cells:
        c["w"] /= z

    out = {"total": len(cells),
           "source": ("ladder_autopsy.json 700+ bands "
                      f"({hi_total}/{len(recs)} episodes) "
                      "+ AlphaStarmie exact + our shipped v4"),
           "snapshot": "2026-08-07",
           "decks": [{"c": c["c"], "p": round(c["w"], 5), "a": c["name"],
                      "team": c["team"], "rank": None} for c in cells]}
    (ROOT / args.out).write_text(json.dumps(out, indent=1))

    lines = ["# Panel v3 manifest (D46) — ladder-representative, weighted",
             "",
             f"{len(cells)} cells. Source: {out['source']}.",
             "Weights = archetype encounter share in the 700+ score bands,",
             f"floor {WEIGHT_FLOOR:.0%} per archetype, split over variants "
             "by high-band game count.",
             "",
             "| # | cell | weight | hi-band games | low-band |",
             "|---|---|---|---|---|"]
    for i, c in enumerate(cells):
        lines.append(f"| {i} | {c['name']} | {c['w']:.4f} | "
                     f"{c['hi']} | {c['lo']} |")
    by_arch = [(a, sum(c['w'] for c in cs), len(cs))
               for a, cs in arch_cells.items()]
    lines += ["", "| archetype | weight | cells |", "|---|---|---|"]
    for a, w, n in sorted(by_arch, key=lambda x: -x[1]):
        lines.append(f"| {a} | {w:.3f} | {n} |")
    (ROOT / args.manifest).write_text("\n".join(lines) + "\n")
    print(f"panel: {len(cells)} cells -> {args.out}")
    for a, w, n in sorted(by_arch, key=lambda x: -x[1]):
        print(f"  {w:.3f} x{n:2d} {a}")


if __name__ == "__main__":
    main()
