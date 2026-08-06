"""Island-model genetic algorithm over decks (decisions.md D5-D7).

Era-0 scope: template-seeded populations per island, validator-driven
repair, fitness = seat-swapped win rate vs a fixed panel under the frozen
greedy pilot, ring migration, JSON checkpoints per era.

Island kinds (D6):
  mono:  one island per basic energy type with real attacker support
  dual:  islands for the supported dragon-bridge pairs
  tri:   gem island — each individual is anchored on one gemstone-ex
         package with its trio energy + stage-matched special energy
"""

import json
import random
import subprocess
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .harness import play_match
from .pilots import GreedyPilot
from .pool import (
    BASIC_ENERGY, POKEMON, SPECIAL_ENERGY, TYPE_NAMES, CardPool, pool,
)
from .validator import validate

DUAL_PAIRS = [(2, 3), (2, 5), (6, 8), (2, 4), (3, 4)]  # supported bridges (D6)

PRISM = 16       # any type, Basic only
NEO_UPPER = 10   # any type x2, Stage 2 only


# ---------------------------------------------------------------------------
# pool-derived gene bank
# ---------------------------------------------------------------------------

class GeneBank:
    def __init__(self, p: CardPool):
        self.pool = p
        self._chain_cache: dict = {}
        self._lines_cache: dict = {}
        cards = list(p.by_id.values())
        self.trainer_names = sorted({c["name"] for c in cards
                                     if c["cardType"] in (1, 2, 3, 4)})
        self.energy_id_by_type = {
            c["energyType"]: c["cardId"] for c in cards
            if c["cardType"] == BASIC_ENERGY
        }
        self.pokemon_names = sorted({c["name"] for c in cards
                                     if c["cardType"] == POKEMON})
        # gem anchors: tri-typed footprint Pokemon (analysis/energy_typing.py)
        self.gem_anchors = []
        for name in self.pokemon_names:
            for cid in p.ids_by_name[name]:
                c = p.by_id[cid]
                if c["cardType"] != POKEMON:
                    continue
                fp = set()
                for aid in c["attacks"]:
                    fp |= set(p.typed_cost(aid))
                if len(fp) >= 3:
                    self.gem_anchors.append((name, tuple(sorted(fp))[:3]))
                    break

    def chain(self, name: str) -> list[str] | None:
        """Evolution line from basic up to `name`; None if broken."""
        if name in self._chain_cache:
            return self._chain_cache[name]
        line = [name]
        seen = {name}
        while True:
            pre = self.pool.evolves_from_name(line[0])
            if pre is None:
                self._chain_cache[name] = line
                return line
            if pre not in self.pool.ids_by_name or pre in seen:
                self._chain_cache[name] = None
                return None
            line.insert(0, pre)
            seen.add(pre)

    def name_playable(self, name: str, allowed: set[int]) -> bool:
        """Some print of `name` has no attacks or an attack payable in
        `allowed` types (colorless always payable)."""
        for cid in self.pool.ids_by_name.get(name, []):
            c = self.pool.by_id[cid]
            if c["cardType"] != POKEMON:
                return False
            if not c["attacks"]:
                return True
            for aid in c["attacks"]:
                if set(self.pool.typed_cost(aid)) <= allowed:
                    return True
        return False

    def eligible_lines(self, allowed: set[int]) -> list[list[str]]:
        key = frozenset(allowed)
        if key in self._lines_cache:
            return self._lines_cache[key]
        lines = []
        for name in self.pokemon_names:
            ch = self.chain(name)
            if ch and all(self.name_playable(n, allowed) for n in ch):
                lines.append(ch)
        self._lines_cache[key] = lines
        return lines

    def pick_print(self, name: str, rng: random.Random) -> int:
        ids = [cid for cid in self.pool.ids_by_name[name]
               if self.pool.by_id[cid]["cardType"] == POKEMON]
        return rng.choice(ids or self.pool.ids_by_name[name])


# ---------------------------------------------------------------------------
# islands
# ---------------------------------------------------------------------------

@dataclass
class Island:
    label: str
    allowed: frozenset  # energy types this island builds on
    anchor: str | None = None  # gem island: forced anchor name


def make_islands(mode: str, bank: GeneBank) -> list[Island]:
    if mode == "mono":
        islands = []
        for t in range(1, 9):
            if len(bank.eligible_lines({t})) >= 6:
                islands.append(Island(f"mono-{TYPE_NAMES[t]}", frozenset({t})))
        return islands
    if mode == "multi":
        islands = [
            Island(f"dual-{TYPE_NAMES[a]}+{TYPE_NAMES[b]}", frozenset({a, b}))
            for a, b in DUAL_PAIRS
        ]
        islands.append(Island("tri-gem", frozenset()))  # anchor set per individual
        return islands
    raise ValueError(f"unknown mode {mode}")


# ---------------------------------------------------------------------------
# deck construction / repair
# ---------------------------------------------------------------------------

def build_template(island: Island, bank: GeneBank, rng: random.Random) -> list[int]:
    p = bank.pool
    if island.label == "tri-gem":
        anchor_name, trio = rng.choice(bank.gem_anchors)
        allowed = set(trio)
    else:
        anchor_name = None
        allowed = set(island.allowed)

    lines = bank.eligible_lines(allowed)
    rng.shuffle(lines)
    deck_names: Counter = Counter()

    def add_line(line: list[str]) -> None:
        counts = {0: rng.choice([3, 4]), 1: rng.choice([2, 3]), 2: rng.choice([2, 3])}
        for stage, n in enumerate(line):
            deck_names[n] += counts.get(stage, 2)

    if anchor_name:
        ch = bank.chain(anchor_name)
        if ch:
            add_line(ch)
    for line in lines:
        if sum(deck_names.values()) >= rng.choice([12, 14, 16]):
            break
        add_line(line)

    trainers = rng.sample(bank.trainer_names, k=min(9, len(bank.trainer_names)))
    for t in trainers:
        deck_names[t] += rng.choice([2, 3])

    deck = []
    for name, n in deck_names.items():
        n = min(n, 4)
        if p.by_id[p.ids_by_name[name][0]]["cardType"] == POKEMON:
            deck += [bank.pick_print(name, rng)] * n
        else:
            deck += [p.ids_by_name[name][0]] * n

    # special energy for gem anchors, stage-gated (D6)
    if anchor_name:
        anchor_card = p.by_id[bank.pick_print(anchor_name, rng)]
        special = PRISM if anchor_card["basic"] else (
            NEO_UPPER if anchor_card["stage2"] else None)
        if special:
            deck += [special] * 4

    energy_types = sorted(allowed) or [rng.randint(1, 8)]
    while len(deck) < 60:
        deck.append(bank.energy_id_by_type[rng.choice(energy_types)])
    return repair(deck[:60], island, bank, rng)


def repair(deck: list[int], island: Island, bank: GeneBank,
           rng: random.Random) -> list[int]:
    """Force a genome through the validator's hard rules + closure/basics."""
    p = bank.pool
    if island.label == "tri-gem" or not island.allowed:
        fill_types = sorted({p.by_id[c]["energyType"] for c in deck
                             if p.by_id[c]["cardType"] == BASIC_ENERGY}) or [3]
    else:
        fill_types = sorted(island.allowed)

    def filler() -> int:
        return bank.energy_id_by_type[rng.choice(fill_types)]

    # copy cap by name (basic energy exempt) + single ACE SPEC
    out, name_counts, ace_seen = [], Counter(), False
    for cid in deck:
        c = p.by_id.get(cid)
        if c is None:
            out.append(filler())
            continue
        if c["aceSpec"]:
            if ace_seen:
                out.append(filler())
                continue
            ace_seen = True
        if c["cardType"] != BASIC_ENERGY:
            if name_counts[c["name"]] >= 4:
                out.append(filler())
                continue
            name_counts[c["name"]] += 1
        out.append(cid)

    # evolution closure: drop unreachable evolutions
    names = {p.by_id[c]["name"] for c in out}
    def reachable(cid: int) -> bool:
        c = p.by_id[cid]
        if c["cardType"] != POKEMON or c["basic"]:
            return True
        pre = p.evolves_from_name(c["name"])
        if pre in names:
            return True
        if c["stage2"] and "Rare Candy" in names:
            basic = p.evolves_from_name(pre) if pre else None
            return basic in names
        return False
    changed = True
    while changed:
        changed = False
        for i, cid in enumerate(out):
            if not reachable(cid):
                out[i] = filler()
                changed = True
        names = {p.by_id[c]["name"] for c in out}

    # basics floor (mulligan risk)
    n_basic = sum(1 for c in out if p.by_id[c]["cardType"] == POKEMON
                  and p.by_id[c]["basic"])
    if n_basic < 9:
        have = [c for c in out if p.by_id[c]["cardType"] == POKEMON
                and p.by_id[c]["basic"]]
        if have:
            energies = [i for i, c in enumerate(out)
                        if p.by_id[c]["cardType"] == BASIC_ENERGY]
            for i in energies[:9 - n_basic]:
                candidate = rng.choice(have)
                if sum(1 for c in out
                       if p.by_id[c]["name"] == p.by_id[candidate]["name"]) < 4:
                    out[i] = candidate

    while len(out) < 60:
        out.append(filler())
    return out[:60]


# ---------------------------------------------------------------------------
# breeding
# ---------------------------------------------------------------------------

def mutate(deck: list[int], island: Island, bank: GeneBank,
           rng: random.Random) -> list[int]:
    p = bank.pool
    out = list(deck)
    for _ in range(rng.randint(1, 3)):
        op = rng.random()
        if op < 0.35:  # swap one trainer name for a fresh one
            slots = [i for i, c in enumerate(out)
                     if p.by_id[c]["cardType"] in (1, 2, 3, 4)]
            if slots:
                i = rng.choice(slots)
                new = p.ids_by_name[rng.choice(bank.trainer_names)][0]
                out[i] = new
        elif op < 0.65:  # duplicate a random present name (count +1)
            i = rng.randrange(60)
            out[rng.randrange(60)] = out[i]
        elif op < 0.85:  # inject a fresh eligible attacker line member
            allowed = set(island.allowed) or {
                t for c in out if p.by_id[c]["cardType"] == BASIC_ENERGY
                for t in [p.by_id[c]["energyType"]]}
            lines = bank.eligible_lines(allowed)
            if lines:
                line = rng.choice(lines)
                out[rng.randrange(60)] = bank.pick_print(rng.choice(line), rng)
        else:  # energy count tweak
            out[rng.randrange(60)] = bank.energy_id_by_type[
                rng.choice(sorted(island.allowed) or [3])] \
                if island.allowed else out[rng.randrange(60)]
    return repair(out, island, bank, rng)


def crossover(a: list[int], b: list[int], island: Island, bank: GeneBank,
              rng: random.Random) -> list[int]:
    p = bank.pool
    pokemon = [c for c in a if p.by_id[c]["cardType"] == POKEMON]
    rest = [c for c in b if p.by_id[c]["cardType"] != POKEMON]
    child = (pokemon + rest)[:60]
    return repair(child, island, bank, rng)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def evaluate(deck: list[int], panel: list[list[int]], pilot_a, pilot_b,
             games_per_opponent: int) -> float:
    rates = []
    for opp_deck in panel:
        m = play_match(pilot_a, pilot_b, deck, opp_deck, games_per_opponent)
        rates.append(m.win_rate(0))
    return sum(rates) / len(rates)


def run(mode: str, run_dir: Path, pop_size: int = 12,
        games_per_opponent: int = 24, max_eras: int = 10_000,
        hours: float = 8.0, migrate_every: int = 5, seed: int = 0,
        panel_seed: int = 42, git_commit: bool = False,
        panel_from: list | None = None, pilot_factory=None) -> None:
    pf = pilot_factory or (lambda s: GreedyPilot(seed=s))
    rng = random.Random(seed)
    bank = GeneBank(pool())
    islands = make_islands(mode, bank)
    run_dir.mkdir(parents=True, exist_ok=True)

    # shared fixed panel (same panel_seed on every machine -> comparable)
    prng = random.Random(panel_seed)
    panel_islands = [Island(f"panel-{TYPE_NAMES[t]}", frozenset({t}))
                     for t in (3, 4, 6)]
    panel = [build_template(isl, bank, prng) for isl in panel_islands]
    # optionally harden the panel with elites from earlier checkpoints
    for path in (panel_from or []):
        try:
            ck = json.loads(Path(path).read_text())
            elites = sorted(ck.get("islands", {}).values(),
                            key=lambda i: -i.get("best", 0))
            for isl_report in elites[:2]:
                d = isl_report.get("best_deck")
                if d and len(d) == 60:
                    panel.append(d)
        except Exception as e:
            print(f"panel-from {path} skipped: {e}", flush=True)
    panel = panel[:6]
    (run_dir / "panel.json").write_text(json.dumps(panel))

    pilot_a, pilot_b = pf(101), pf(202)
    pops = {isl.label: [build_template(isl, bank, rng) for _ in range(pop_size)]
            for isl in islands}
    fitness_cache: dict[tuple, float] = {}
    prev_elite: list[int] | None = None
    t_start = time.time()

    def fit(deck: list[int]) -> float:
        key = tuple(sorted(deck))
        if key not in fitness_cache:
            fitness_cache[key] = evaluate(deck, panel, pilot_a, pilot_b,
                                          games_per_opponent)
        return fitness_cache[key]

    for era in range(max_eras):
        if (time.time() - t_start) / 3600 >= hours:
            break
        era_report = {}
        for isl in islands:
            popn = pops[isl.label]
            scored = sorted(((fit(d), d) for d in popn),
                            key=lambda s: -s[0])
            elites = [d for _, d in scored[:2]]
            children = list(elites)
            while len(children) < pop_size:
                pa = max(rng.sample(scored, k=min(3, len(scored))),
                         key=lambda s: s[0])[1]
                if rng.random() < 0.7:
                    pb = max(rng.sample(scored, k=min(3, len(scored))),
                             key=lambda s: s[0])[1]
                    child = crossover(pa, pb, isl, bank, rng)
                else:
                    child = list(pa)
                children.append(mutate(child, isl, bank, rng))
            pops[isl.label] = children
            era_report[isl.label] = {
                "best": round(scored[0][0], 4),
                "mean": round(sum(s for s, _ in scored) / len(scored), 4),
                "best_deck": scored[0][1],
            }

        # ring migration
        if era % migrate_every == migrate_every - 1 and len(islands) > 1:
            labels = [i.label for i in islands]
            bests = {l: max(pops[l], key=fit) for l in labels}
            for i, l in enumerate(labels):
                nxt = labels[(i + 1) % len(labels)]
                worst = min(range(pop_size), key=lambda j: fit(pops[nxt][j]))
                pops[nxt][worst] = repair(list(bests[l]),
                                          islands[(i + 1) % len(islands)],
                                          bank, rng)

        # global elite + D7 progress gate
        all_best = max(
            ((era_report[l]["best"], era_report[l]["best_deck"], l)
             for l in era_report), key=lambda s: s[0])
        gate = None
        if prev_elite is not None and prev_elite != all_best[1]:
            m = play_match(pilot_a, pilot_b, all_best[1], prev_elite, 60)
            gate = round(m.win_rate(0), 3)
        prev_elite = all_best[1]

        ckpt = {
            "era": era,
            "elapsed_h": round((time.time() - t_start) / 3600, 3),
            "islands": era_report,
            "elite": {"island": all_best[2], "fitness": all_best[0],
                      "deck": all_best[1]},
            "gate_vs_prev_elite": gate,
            "evals_cached": len(fitness_cache),
        }
        (run_dir / f"era_{era:03d}.json").write_text(json.dumps(ckpt))
        (run_dir / "latest.json").write_text(json.dumps(ckpt))
        report = validate(all_best[1])
        line = (f"era {era:3d} [{ckpt['elapsed_h']:.2f}h] "
                f"elite={all_best[0]:.3f} ({all_best[2]}) "
                f"gate={gate} coherent={report.coherent}")
        print(line, flush=True)
        if git_commit:
            subprocess.run(["git", "add", str(run_dir)],
                           cwd=run_dir.parent.parent, capture_output=True)
            subprocess.run(["git", "commit", "-m", f"GA {run_dir.name}: {line}"],
                           cwd=run_dir.parent.parent, capture_output=True)
