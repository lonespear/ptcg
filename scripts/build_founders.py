"""Build founders.json for the seeded overnight archipelago (D38).

Founding population classes per island (head of each population; randoms
fill the remainder inside the GA):
  1. sprint/deep-eval elites of the island's type (runs/mono_sprint/
     final_eval.json), top 1-2 by deep win rate;
  2. the field's best list(s) of that type from agent/deck_priors.json
     (a priors list belongs to the type of its modal basic energy);
  3. one constructed template: 4x the type's core attacker (the sprint
     elite's archetype card, else the priors list's) + 19 basic energy +
     the field's consensus trainer engine (copies weighted by play share);
  spec-Ogerpon additionally founds from Jon's exact submitted list
  (agent/deck.csv) and its D17 sister lists (the priors' Teal Mask
  Ogerpon ex entries).

    python scripts/build_founders.py --out runs/seeded_overnight/founders.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ptcg.creation.pool import (  # noqa: E402
    BASIC_ENERGY, ITEM, POKEMON, STADIUM, SUPPORTER, TOOL, TYPE_NAMES, pool)
from ptcg.creation.validator import validate  # noqa: E402

TRAINER = {ITEM, TOOL, SUPPORTER, STADIUM}
SETS = {"mono-Grass": 1, "mono-Fire": 2, "mono-Fighting": 6,
        "mono-Darkness": 7}


def deck_of(entry: dict) -> list[int]:
    return [int(c) for c, n in entry["c"].items() for _ in range(n)]


def modal_energy(deck: list[int], p) -> int | None:
    et = Counter(p.by_id[c]["energyType"] for c in deck
                 if p.by_id[c]["cardType"] == BASIC_ENERGY)
    return et.most_common(1)[0][0] if et else None


def archetype_card(deck: list[int], p) -> int | None:
    """Card id of the deck's highest-HP Pokemon (the archetype rule)."""
    best, hp = None, -1
    for c in deck:
        card = p.by_id[c]
        if card["cardType"] == POKEMON and (card.get("hp") or 0) > hp:
            best, hp = c, card["hp"]
    return best


def consensus_trainers(priors: dict, p, slots: int) -> list[int]:
    """The field's trainer engine: for each trainer name, play-weighted
    mean copies; fill `slots` by weighted presence, 4-copy capped."""
    presence: dict[str, float] = defaultdict(float)
    copies: dict[str, float] = defaultdict(float)
    ids: dict[str, int] = {}
    total = sum(e["p"] for e in priors["decks"])
    for e in priors["decks"]:
        w = e["p"] / total
        counts = Counter()
        for c, n in e["c"].items():
            card = p.by_id[int(c)]
            if card["cardType"] in TRAINER:
                counts[card["name"]] += n
                ids.setdefault(card["name"], int(c))
        for name, n in counts.items():
            presence[name] += w
            copies[name] += w * n
    out: list[int] = []
    for name in sorted(presence, key=lambda n: -presence[n]):
        want = max(1, min(4, round(copies[name] / presence[name])))
        take = min(want, slots - len(out))
        out.extend([ids[name]] * take)
        if len(out) >= slots:
            break
    return out


def evolution_line(core: int, p) -> list[int]:
    """The core attacker plus its pre-evolutions (a Stage 2 alone has no
    Basic and the engine refuses the deck — errorType 3)."""
    line = [core]
    cur = p.by_id[core]
    while cur.get("evolvesFrom"):
        prev = next((cid for cid, c in p.by_id.items()
                     if c["name"] == cur["evolvesFrom"]
                     and c["cardType"] == POKEMON), None)
        if prev is None:
            break
        line.append(prev)
        cur = p.by_id[prev]
    return line


def constructed_template(core: int, etype: int, priors: dict, p
                         ) -> list[int] | None:
    energy_id = next((cid for cid, c in p.by_id.items()
                      if c["cardType"] == BASIC_ENERGY
                      and c["energyType"] == etype), None)
    if energy_id is None:
        return None
    deck = [cid for cid in evolution_line(core, p) for _ in range(4)]
    deck += [energy_id] * 19
    deck += consensus_trainers(priors, p, 60 - len(deck))
    if len(deck) != 60 or not validate(deck).legal:
        return None
    return deck


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final-eval",
                    default=str(ROOT / "runs/mono_sprint/final_eval.json"))
    ap.add_argument("--out",
                    default=str(ROOT / "runs/seeded_overnight/founders.json"))
    ap.add_argument("--elites-per-set", type=int, default=2)
    args = ap.parse_args()

    p = pool()
    priors = json.loads((ROOT / "agent/deck_priors.json").read_text())
    fe = json.loads(Path(args.final_eval).read_text())

    def deep_wr(e: dict) -> float:
        num = den = 0.0
        for o in e["per_opp"]:
            n = o["w"] + o["l"]
            if n:
                num += o["weight"] * (o["w"] / n)
                den += o["weight"]
        return num / den if den else e["ga_fitness"]

    founders: dict[str, list[dict]] = defaultdict(list)

    # 1. sprint elites by set (Metal/Water excluded by the island roster)
    for sk in SETS:
        recs = sorted((e for e in fe["elites"] if e["set"] == sk),
                      key=deep_wr, reverse=True)[:args.elites_per_set]
        for e in recs:
            founders[sk].append({"class": "sprint-elite",
                                 "label": f"{e['island']} deep {deep_wr(e):.3f}",
                                 "deck": e["deck"]})

    # 2. field's best lists per type (modal basic energy assigns the type)
    best_field: dict[str, dict] = {}
    for e in sorted(priors["decks"], key=lambda e: -e["p"]):
        d = deck_of(e)
        if len(d) != 60 or not validate(d).legal:
            continue
        et = modal_energy(d, p)
        for sk, t in SETS.items():
            if et == t and len([f for f in founders[sk]
                                if f["class"] == "field-list"]) < 2:
                founders[sk].append({"class": "field-list",
                                     "label": f"{e['a']} (p={e['p']})",
                                     "deck": d})
                best_field.setdefault(sk, e)

    # 2b. D38: the Darkness island founds from the field's Grimmsnarl AND
    # Fezandipiti lists explicitly (the Fez lists' modal energy is not
    # Darkness, so the type rule alone misses them; purity floors will
    # convert their off-type cores while keeping the trainer engine).
    n_fez = 0
    for e in sorted(priors["decks"], key=lambda e: -e["p"]):
        if e["a"] != "Fezandipiti ex" or n_fez >= 2:
            continue
        d = deck_of(e)
        if len(d) == 60 and validate(d).legal:
            founders["mono-Darkness"].append(
                {"class": "field-list-fez", "label": f"{e['a']} (p={e['p']})",
                 "deck": d})
            n_fez += 1

    # 3. one constructed template per set
    for sk, t in SETS.items():
        src = next((f["deck"] for f in founders[sk]
                    if f["class"] == "sprint-elite"), None)
        if src is None:
            src = next((f["deck"] for f in founders[sk]
                        if f["class"] == "field-list"), None)
        core = archetype_card(src, p) if src else None
        if core is None:
            continue
        deck = constructed_template(core, t, priors, p)
        if deck:
            founders[sk].append({"class": "constructed",
                                 "label": f"4x {p.by_id[core]['name']} "
                                          "+ 19 energy + consensus engine",
                                 "deck": deck})

    # spec-Ogerpon: Jon's exact list + D17 sisters (priors Ogerpon entries)
    jon = [int(x) for x in
           (ROOT / "agent/deck.csv").read_text().split()]
    if len(jon) == 60:
        founders["spec-Ogerpon"].append({"class": "jon-submitted",
                                         "label": "agent/deck.csv (shipped)",
                                         "deck": jon})
    for e in sorted(priors["decks"], key=lambda e: -e["p"]):
        if e["a"] != "Teal Mask Ogerpon ex":
            continue
        d = deck_of(e)
        if len(d) == 60 and validate(d).legal:
            founders["spec-Ogerpon"].append(
                {"class": "d17-sister", "label": f"priors p={e['p']}",
                 "deck": d})
    founders["spec-Ogerpon"] = founders["spec-Ogerpon"][:4]

    # rainbow-Kanga (D39): the field's rainbow Mega Kangaskhan toolbox
    # lists — >=4 basic energy types — plus one deterministic sister with
    # the energy base rebalanced toward uniform across its types. Mutated
    # variants and randoms are generated at seeding time.
    for e in sorted(priors["decks"], key=lambda e: -e["p"]):
        if e["a"] != "Mega Kangaskhan ex":
            continue
        d = deck_of(e)
        if len(d) != 60 or not validate(d).legal:
            continue
        ets = sorted({p.by_id[c]["energyType"] for c in d
                      if p.by_id[c]["cardType"] == BASIC_ENERGY})
        if len(ets) < 4:
            continue
        founders["rainbow-Kanga"].append(
            {"class": "field-rainbow",
             "label": f"{e['a']} (p={e['p']}, "
                      f"types {[TYPE_NAMES[t] for t in ets]})",
             "deck": d})
        eid = {t: next(cid for cid, c in p.by_id.items()
                       if c["cardType"] == BASIC_ENERGY
                       and c["energyType"] == t) for t in ets}
        idxs = [i for i, c in enumerate(d)
                if p.by_id[c]["cardType"] == BASIC_ENERGY]
        sister = list(d)
        for j, i in enumerate(idxs):
            sister[i] = eid[ets[j % len(ets)]]
        if validate(sister).legal:
            founders["rainbow-Kanga"].append(
                {"class": "sister-rebalance",
                 "label": "energy base rebalanced uniform across types",
                 "deck": sister})

    # kanga-counter (D39): founds from the Fighting classes; the island's
    # retype mutation is free to leave Fighting for another mechanism.
    founders["kanga-counter"] = [dict(f) for f in founders["mono-Fighting"]]

    for sk in list(founders):
        kept = []
        for f in founders[sk]:
            if len(f["deck"]) == 60 and validate(f["deck"]).legal:
                kept.append(f)
            else:
                print(f"DROPPED illegal founder {sk}: [{f['class']}] "
                      f"{f['label']}")
        founders[sk] = kept

    out = {sk: [f["deck"] for f in fl] for sk, fl in founders.items()}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out))
    manifest = {sk: [{"class": f["class"], "label": f["label"]} for f in fl]
                for sk, fl in founders.items()}
    mpath = Path(args.out).with_name("founders_manifest.json")
    mpath.write_text(json.dumps(manifest, indent=1))
    for sk, fl in founders.items():
        print(f"{sk}: " + "; ".join(f"[{f['class']}] {f['label']}"
                                    for f in fl))
    print(f"wrote {args.out} and {mpath}")


if __name__ == "__main__":
    main()
