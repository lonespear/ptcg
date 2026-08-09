"""Assemble the D46 founder bank (runs/seeded_d46/founders.json).

One file serves both machines; the machines differ by --chains subset
and RNG --seed. Sources (all committed or regenerated in-repo except
the Sebastian live-stop pop, snapshotted to
runs/seeded_d46/seb_v3_rainbow_pops.json):

  spec-Ogerpon       shipped v4 list (agent/deck.csv) + both v3 harvest
                     elites (raw 0.654 sebastian / 0.648 laptop) + the
                     two D17 sister lists from the v3 founder bank.
  grimmsnarl-mirror  v3 mono-Darkness harvest elites (raw 0.691 laptop,
                     0.673 sebastian era-0 refine — the "0.673
                     Sebastian elite"; it is a Darkness/Grimmsnarl
                     deck, so it seeds this chain, not spec-Ogerpon) +
                     the 5 autopsy Grimmsnarl field variants (via
                     panel v3).
  engine             AlphaStarmie exact stall list + the two energyless
                     Mega Lopunny prior lists + the two 2-energy
                     Fezandipiti prior lists + the 3-energy field
                     Lopunny (panel v3).
  archaludon         the top-5 autopsy Archaludon-Duraludon-Cinderace
                     field lists (via panel v3).
  counter-900        v3 kanga-counter harvest elites (raw 0.333 laptop
                     Cornerstone-Metal, 0.265 sebastian Lightning) +
                     3 of the v3 Fighting founder classes (reseed
                     material; the retype mutation can leave).
  rainbow-Kanga      v3 rainbow harvest elite (raw 0.364 laptop) + the
                     heads of Sebastian's live-stop rainbow pops
                     (refine[:2] + explore[:2], era 31).

    python scripts/build_founders_d46.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ptcg.creation.validator import validate  # noqa: E402

OUT = ROOT / "runs" / "seeded_d46" / "founders.json"


def deck_list(c: dict) -> list[int]:
    return [int(k) for k, n in c.items() for _ in range(int(n))]


def main() -> None:
    harvest = json.loads(
        (ROOT / "runs/seeded_overnight/harvest_best.json").read_text())
    v3f = json.loads(
        (ROOT / "runs/seeded_overnight/founders.json").read_text())
    panel = json.loads((ROOT / "data/panel_ladder_v3.json").read_text())
    priors = json.loads((ROOT / "agent/deck_priors.json").read_text())
    seb_rainbow = json.loads(
        (ROOT / "runs/seeded_d46/seb_v3_rainbow_pops.json").read_text())

    def hv(chain: str) -> list[list[int]]:
        return [e["deck"] for e in
                sorted(harvest["chains"][chain], key=lambda e: -e["raw"])]

    def panel_decks(prefix: str, n: int = 99) -> list[list[int]]:
        return [deck_list(e["c"]) for e in panel["decks"]
                if e["a"].startswith(prefix)][:n]

    shipped = [int(x) for x in
               (ROOT / "agent/deck.csv").read_text().split()]

    # Sebastian mono-Darkness era-0 refine elite, raw 0.6728
    # (runs/seeded_v3_seb/era_000.json on the Mini).
    seb_dark_0673 = [
        7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 104, 104, 112, 112, 112, 112,
        646, 646, 646, 646, 647, 647, 647, 648, 648, 648, 860, 860,
        1079, 1079, 1079, 1080, 1086, 1086, 1086, 1086, 1097, 1097,
        1097, 1122, 1137, 1152, 1152, 1152, 1152, 689, 1182, 1219,
        1219, 1219, 1219, 1227, 1227, 1227, 1227, 1231, 1259, 1259,
        1259, 1259]

    fez = sorted((d for d in priors["decks"] if d["a"] == "Fezandipiti ex"),
                 key=lambda d: -d["p"])
    lop = sorted((d for d in priors["decks"] if "Lopunny" in d["a"]),
                 key=lambda d: -d["p"])
    fez_low = [deck_list(d["c"]) for d in fez
               if sum(d["c"].values()) == 60][:3]
    lop_zero = [deck_list(d["c"]) for d in lop][:3]

    founders = {
        "spec-Ogerpon": ([shipped] + hv("spec-Ogerpon")
                         + v3f.get("spec-Ogerpon", [])),
        "grimmsnarl-mirror": (hv("mono-Darkness")[:1] + [seb_dark_0673]
                              + panel_decks("Grimmsnarl")),
        "engine": ([deck_list(e["c"]) for e in panel["decks"]
                    if "AlphaStarmie exact" in e["a"]]
                   + lop_zero + fez_low
                   + panel_decks("Mega Lopunny #2", 1)),
        "archaludon": panel_decks("Archaludon", 5),
        "counter-900": hv("kanga-counter") + v3f.get("kanga-counter", [])[:3],
        "rainbow-Kanga": (hv("rainbow-Kanga")[:1]
                          + seb_rainbow["rainbow-Kanga/refine"][:2]
                          + seb_rainbow["rainbow-Kanga/explore"][:2]),
    }

    for sk, decks in founders.items():
        clean = []
        for d in decks:
            d = [int(c) for c in d]
            ok = len(d) == 60 and validate(d).legal
            if ok:
                clean.append(d)
            else:
                print(f"  DROP illegal founder in {sk}")
        # dedupe
        seen, uniq = set(), []
        for d in clean:
            k = tuple(sorted(d))
            if k not in seen:
                seen.add(k)
                uniq.append(d)
        founders[sk] = uniq
        print(f"{sk}: {len(uniq)} founders")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(founders))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
