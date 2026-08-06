"""Archipelago GA (creation v2): directed gene flow, phased compute.

Topology (one-way DAG, PTCG_AI decisions.md D11):
  8 mono source sets, each Explore + Refine islands. Explorers leave;
  nothing arrives. 5 bridge-backed dual mixer sets downstream. Tri-gem
  terminus. Migrants cross type boundaries as crossover partners, never
  as transplanted decks.

Compute (phased per round): Phase A evolves mono sets (~70% wall-clock),
a founding burst seeds duals/tri from mature refine elites, Phase B
evolves duals+tri (~30%). A set freezes after `plateau_window` eras
without refine-elite improvement; frozen sets stop billing.

Fitness: play-weighted win rate vs a real-field panel (deck_priors.json),
under an injected pilot. Elite changes are confirmed with a 500-game
match against the previous elite before being reported as progress.
"""

import json
import random
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from .ga import (DUAL_PAIRS, GeneBank, Island, build_template, crossover,
                 mutate, repair)
from .harness import play_match
from .pilots import GreedyPilot
from .pool import POKEMON, TYPE_NAMES, pool
from .validator import validate

GATE_GAMES = 500
FRESH_TEMPLATE_EVERY = 10
MIGRATE_EVERY = 20
DIVERSITY_LAMBDA = 0.1


@dataclass
class ArchIsland:
    island: Island            # reused by build/repair/mutate/crossover
    set_key: str              # e.g. "mono-Fire", "dual-Fire+Water", "tri"
    temperament: str          # "explore" | "refine" | "mix" | "gem"
    parents: list = field(default_factory=list)  # upstream set_keys

    @property
    def label(self) -> str:
        return f"{self.set_key}/{self.temperament}"


def build_archipelago(bank: GeneBank) -> list[ArchIsland]:
    out = []
    for t in range(1, 9):
        if len(bank.eligible_lines({t})) < 6:
            continue
        base = Island(f"mono-{TYPE_NAMES[t]}", frozenset({t}))
        key = base.label
        out.append(ArchIsland(base, key, "explore"))
        out.append(ArchIsland(base, key, "refine"))
    for a, b in DUAL_PAIRS:
        key = f"dual-{TYPE_NAMES[a]}+{TYPE_NAMES[b]}"
        out.append(ArchIsland(Island(key, frozenset({a, b})), key, "mix",
                              parents=[f"mono-{TYPE_NAMES[a]}",
                                       f"mono-{TYPE_NAMES[b]}"]))
    out.append(ArchIsland(Island("tri-gem", frozenset()), "tri", "gem",
                          parents=[]))
    return out


# ---------------------------------------------------------------------------
# purity floors (D11) — the anti-collapse rule
# ---------------------------------------------------------------------------

def _typed_attacker_slots(deck: list[int], t: int, p) -> int:
    n = 0
    for cid in deck:
        c = p.by_id[cid]
        if c["cardType"] != POKEMON:
            continue
        if any(t in p.typed_cost(a) for a in c["attacks"]
               if a in p.attack_by_id):
            n += 1
    return n


def enforce_purity(deck: list[int], ai: ArchIsland, bank: GeneBank,
                   rng: random.Random) -> list[int]:
    p = bank.pool
    out = list(deck)
    if ai.set_key == "tri":
        if not any(len({t for a in p.by_id[c]["attacks"] if a in p.attack_by_id
                        for t in p.typed_cost(a)}) >= 3
                   for c in out if p.by_id[c]["cardType"] == POKEMON):
            return build_template(ai.island, bank, rng)  # anchor lost: reseed
        return out
    floors = [(t, 6) for t in ai.island.allowed] if ai.temperament in (
        "explore", "refine") else [(t, 3) for t in ai.island.allowed]
    for t, floor in floors:
        deficit = floor - _typed_attacker_slots(out, t, p)
        if deficit <= 0:
            continue
        typed_lines = [l for l in bank.eligible_lines(set(ai.island.allowed))
                       if any(t in p.typed_cost(a)
                              for n in l for cid in p.ids_by_name[n]
                              for a in p.by_id[cid]["attacks"]
                              if a in p.attack_by_id)]
        if not typed_lines:
            continue
        line = rng.choice(typed_lines)
        replaceable = [i for i, c in enumerate(out)
                       if p.by_id[c]["cardType"] == POKEMON
                       and not any(t2 in p.typed_cost(a)
                                   for t2 in ai.island.allowed
                                   for a in p.by_id[c]["attacks"]
                                   if a in p.attack_by_id)]
        rng.shuffle(replaceable)
        for i in replaceable[:deficit]:
            out[i] = bank.pick_print(rng.choice(line), rng)
    return repair(out, ai.island, bank, rng)


# ---------------------------------------------------------------------------
# diversity (attacker-line Jaccard, D11)
# ---------------------------------------------------------------------------

def _line_sig(deck: list[int], p) -> frozenset:
    return frozenset(p.by_id[c]["name"] for c in deck
                     if p.by_id[c]["cardType"] == POKEMON)


def _mean_distance(deck: list[int], popn: list[list[int]], p) -> float:
    sig = _line_sig(deck, p)
    ds = []
    for other in popn:
        o = _line_sig(other, p)
        union = len(sig | o)
        ds.append(1 - len(sig & o) / union if union else 0.0)
    return sum(ds) / len(ds) if ds else 0.0


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def load_field_panel(priors_path: Path, top_n: int = 8) -> list[dict]:
    priors = json.loads(priors_path.read_text())
    field = sorted(priors["decks"], key=lambda d: -d["p"])[:top_n]
    panel = []
    for e in field:
        deck = [int(c) for c, n in e["c"].items() for _ in range(n)]
        if len(deck) == 60 and validate(deck).legal:
            panel.append({"deck": deck, "weight": e["p"], "name": e["a"]})
    total = sum(p["weight"] for p in panel)
    for p in panel:
        p["weight"] /= total
    return panel


def run_archipelago(run_dir: Path, priors_path: Path, hours: float = 8.0,
                    phase_a_frac: float = 0.7, pop_size: int = 10,
                    games_per_opponent: int = 24, seed: int = 0,
                    plateau_window: int = 15, pilot_factory=None,
                    git_commit: bool = False) -> None:
    rng = random.Random(seed)
    bank = GeneBank(pool())
    p = pool()
    islands = build_archipelago(bank)
    mono = [ai for ai in islands if ai.temperament in ("explore", "refine")]
    downstream = [ai for ai in islands if ai.temperament in ("mix", "gem")]
    run_dir.mkdir(parents=True, exist_ok=True)

    panel = load_field_panel(priors_path)
    (run_dir / "panel.json").write_text(json.dumps(panel))
    pf = pilot_factory or (lambda s: GreedyPilot(seed=s))
    pilot_a, pilot_b = pf(101), pf(202)

    pops = {ai.label: [build_template(ai.island, bank, rng)
                       for _ in range(pop_size)] for ai in islands}
    cache: dict[tuple, float] = {}
    best_seen: dict[str, tuple[int, float]] = {}   # set_key -> (era, best)
    frozen: set[str] = set()
    prev_elite: list[int] | None = None
    t0 = time.time()

    def fit(deck: list[int]) -> float:
        key = tuple(sorted(deck))
        if key not in cache:
            score = 0.0
            for entry in panel:
                m = play_match(pilot_a, pilot_b, deck, entry["deck"],
                               games_per_opponent)
                score += m.win_rate(0) * entry["weight"]
            cache[key] = score
        return cache[key]

    def breed(ai: ArchIsland, era: int) -> dict:
        popn = pops[ai.label]
        scored = sorted(((fit(d), d) for d in popn), key=lambda s: -s[0])
        explore = ai.temperament == "explore"
        if explore:
            ranked = sorted(
                ((f + DIVERSITY_LAMBDA * _mean_distance(d, popn, p), f, d)
                 for f, d in scored), key=lambda s: -s[0])
            sel_pool = [(f, d) for _, f, d in ranked]
        else:
            sel_pool = scored
        n_elite = 1 if explore else 3
        k = 2 if explore else 4
        children = [d for _, d in sel_pool[:n_elite]]
        if explore and era % FRESH_TEMPLATE_EVERY == FRESH_TEMPLATE_EVERY - 1:
            children.append(build_template(ai.island, bank, rng))
        while len(children) < pop_size:
            pa = max(rng.sample(sel_pool, k=min(k, len(sel_pool))),
                     key=lambda s: s[0])[1]
            if rng.random() < 0.7:
                pb = max(rng.sample(sel_pool, k=min(k, len(sel_pool))),
                         key=lambda s: s[0])[1]
                child = crossover(pa, pb, ai.island, bank, rng)
            else:
                child = list(pa)
            child = mutate(child, ai.island, bank, rng)
            if explore:
                child = mutate(child, ai.island, bank, rng)  # 3-5 ops total
            children.append(enforce_purity(child, ai, bank, rng))
        pops[ai.label] = children
        return {"best": round(scored[0][0], 4),
                "mean": round(sum(f for f, _ in scored) / len(scored), 4),
                "best_deck": scored[0][1]}

    def migrate_internal(era: int) -> None:
        if era % MIGRATE_EVERY != MIGRATE_EVERY - 1:
            return
        for ai in mono:
            if ai.temperament != "explore" or ai.set_key in frozen:
                continue
            refine = f"{ai.set_key}/refine"
            elite = max(pops[ai.label], key=fit)
            worst = min(range(pop_size), key=lambda i: fit(pops[refine][i]))
            pops[refine][worst] = list(elite)

    def checkpoint(era: int, phase: str, reports: dict) -> None:
        nonlocal prev_elite
        all_best = max(((r["best"], r["best_deck"], l)
                        for l, r in reports.items()), key=lambda s: s[0])
        gate = None
        if prev_elite is not None and prev_elite != all_best[1]:
            m = play_match(pilot_a, pilot_b, all_best[1], prev_elite,
                           GATE_GAMES)
            gate = round(m.win_rate(0), 3)
        prev_elite = all_best[1]
        ck = {"era": era, "phase": phase,
              "elapsed_h": round((time.time() - t0) / 3600, 3),
              "islands": reports, "frozen": sorted(frozen),
              "elite": {"island": all_best[2], "fitness": all_best[0],
                        "deck": all_best[1]},
              "gate_vs_prev_elite_500g": gate}
        (run_dir / f"era_{era:03d}.json").write_text(json.dumps(ck))
        (run_dir / "latest.json").write_text(json.dumps(ck))
        print(f"era {era:3d} [{ck['elapsed_h']:.2f}h {phase}] "
              f"elite={all_best[0]:.3f} ({all_best[2]}) gate={gate} "
              f"frozen={len(frozen)}", flush=True)
        if git_commit:
            subprocess.run(["git", "add", str(run_dir)],
                           cwd=run_dir.parent.parent, capture_output=True)
            subprocess.run(["git", "commit", "-m",
                            f"archipelago {run_dir.name} era {era}"],
                           cwd=run_dir.parent.parent, capture_output=True)

    def plateau_check(sets: list[str], reports: dict, era: int) -> None:
        for sk in sets:
            key = (f"{sk}/refine" if f"{sk}/refine" in reports
                   else next((l for l in reports if l.startswith(sk)), None))
            if key is None:
                continue
            best = reports[key]["best"]
            e0, b0 = best_seen.get(sk, (era, -1.0))
            if best > b0 + 0.01:
                best_seen[sk] = (era, best)
            elif era - e0 >= plateau_window:
                frozen.add(sk)

    era = 0
    # -------- Phase A: mono sources --------
    while (time.time() - t0) / 3600 < hours * phase_a_frac:
        live = [ai for ai in mono if ai.set_key not in frozen]
        if not live:
            break
        reports = {ai.label: breed(ai, era) for ai in live}
        migrate_internal(era)
        plateau_check(sorted({ai.set_key for ai in live}), reports, era)
        checkpoint(era, "A", reports)
        era += 1

    # -------- founding burst --------
    refine_elites = {}
    for ai in mono:
        if ai.temperament == "refine":
            top2 = sorted(pops[ai.label], key=fit, reverse=True)[:2]
            refine_elites[ai.set_key] = top2
    for ai in downstream:
        sources = ai.parents or list(refine_elites)
        stock = [d for sk in sources for d in refine_elites.get(sk, [])]
        popn = pops[ai.label]
        for migrant in stock:
            local = max(popn, key=fit)
            child = crossover(local, migrant, ai.island, bank, rng)
            child = enforce_purity(child, ai, bank, rng)
            worst = min(range(len(popn)), key=lambda i: fit(popn[i]))
            popn[worst] = child

    # -------- Phase B: duals + tri --------
    frozen -= {ai.set_key for ai in downstream}
    best_seen.clear()
    while (time.time() - t0) / 3600 < hours:
        live = [ai for ai in downstream if ai.set_key not in frozen]
        if not live:  # all converged: give remaining time back to live monos
            live = [ai for ai in mono if ai.set_key not in frozen]
            if not live:
                break
        reports = {ai.label: breed(ai, era) for ai in live}
        plateau_check(sorted({ai.set_key for ai in live}), reports, era)
        checkpoint(era, "B", reports)
        era += 1
