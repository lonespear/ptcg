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

Fitness: play-weighted win rate vs a real-field panel (deck_priors.json).
The candidate is played by the injected pilot; each field deck is played by
its own harvested specialist where one exists (see specialist_panel.py), so
opponents execute their plans instead of being flattened by one generalist.
Elite changes are confirmed with a 500-game match against the previous
elite before being reported as progress.
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
from .pool import BASIC_ENERGY, POKEMON, SPECIAL_ENERGY, TYPE_NAMES, pool
from .validator import validate

GATE_GAMES = 500
FRESH_TEMPLATE_EVERY = 10
MIGRATE_EVERY = 20
DIVERSITY_LAMBDA = 0.1

# spec-Ogerpon island (D38 correction): every genome must keep at least
# this many copies of the anchor card — enforced in the repair path, so
# crossover/mutation/templates cannot drift the card out.
SPEC_ANCHORS = {"spec-Ogerpon": ("Teal Mask Ogerpon ex", 2)}


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
    anchor = SPEC_ANCHORS.get(ai.set_key)
    if anchor:
        name, min_n = anchor
        have = sum(1 for c in out if p.by_id[c]["name"] == name)
        ids = p.ids_by_name.get(name, [])
        if have < min_n and ids:
            # replace non-anchor Pokemon first, then trainers — never the
            # energy base; repair() below re-legalizes whatever this does
            mons = [i for i, c in enumerate(out)
                    if p.by_id[c]["cardType"] == POKEMON
                    and p.by_id[c]["name"] != name]
            trainers = [i for i, c in enumerate(out)
                        if p.by_id[c]["cardType"] not in
                        (POKEMON, BASIC_ENERGY, SPECIAL_ENERGY)]
            rng.shuffle(mons)
            rng.shuffle(trainers)
            for i in (mons + trainers)[:min_n - have]:
                out[i] = ids[0]
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


def _jsonable_rng(state) -> list:
    return [state[0], list(state[1]), state[2]]


def run_archipelago(run_dir: Path, priors_path: Path, hours: float = 8.0,
                    phase_a_frac: float = 0.7, pop_size: int = 10,
                    games_per_opponent: int = 24, seed: int = 0,
                    plateau_window: int = 15, pilot_factory=None,
                    git_commit: bool = False, workers: int = 1,
                    generalist_name: str = "jon",
                    seed_deck: list | None = None,
                    resume: bool = False,
                    mono_types: list[str] | None = None,
                    mono_only: bool = False,
                    wall_hours: float | None = None,
                    deep_top: int = 3,
                    preload_archives: list[Path] | None = None,
                    seed_elites: list[dict] | None = None,
                    v2: bool = False,
                    explore_pop: int | None = None,
                    refine_pop: int | None = None,
                    migrate_every: int | None = None,
                    migrate_top: int = 1,
                    floor_wr: float = 0.35,
                    founders: dict | None = None,
                    panel_top_n: int = 8) -> None:
    rng = random.Random(seed)
    bank = GeneBank(pool())
    p = pool()
    islands = build_archipelago(bank)
    if mono_types:
        keep = {f"mono-{t}" for t in mono_types}
        have = {ai.set_key for ai in islands}
        missing = keep - have
        if missing:
            raise ValueError(f"unknown/ineligible mono sets: {sorted(missing)}")
        islands = [ai for ai in islands
                   if ai.set_key in keep
                   or (not mono_only and ai.temperament in ("mix", "gem"))]
    if founders and "spec-Ogerpon" in founders:
        # D38 specialty island: refinement of a proven Grass deck (Jon's
        # list + sisters), not discovery. Grass purity floors apply.
        spec = Island("spec-Ogerpon", frozenset({1}))
        islands.append(ArchIsland(spec, "spec-Ogerpon", "explore"))
        islands.append(ArchIsland(spec, "spec-Ogerpon", "refine"))
    mono = [ai for ai in islands if ai.temperament in ("explore", "refine")]
    downstream = [ai for ai in islands if ai.temperament in ("mix", "gem")]
    run_dir.mkdir(parents=True, exist_ok=True)

    # local import: specialist_panel reuses load_field_panel from this module
    from .specialist_panel import (build_specialist_panel, make_panel_pilots,
                                   panel_report)

    pf = pilot_factory or (lambda s: GreedyPilot(seed=s))
    pilot_a, pilot_b = pf(101), pf(202)
    # Each field deck is played by its own specialist when one was harvested
    # for that archetype; the rest keep the injected generalist.
    panel = build_specialist_panel(priors_path, top_n=panel_top_n)
    panel_pilots = make_panel_pilots(panel, pf)
    pfit = None
    if workers > 1:
        from .parallel import ParallelFitness
        pfit = ParallelFitness(panel, generalist_name, games_per_opponent,
                               workers, str(Path.cwd()))
    (run_dir / "panel.json").write_text(json.dumps(panel))
    print("panel:\n" + panel_report(panel), flush=True)

    def island_pop(ai: ArchIsland) -> int:
        if ai.temperament == "explore" and explore_pop:
            return explore_pop
        if ai.temperament == "refine" and refine_pop:
            return refine_pop
        return pop_size

    def fresh_member(ai: ArchIsland) -> list[int]:
        """Random template; anchored islands get their structural
        constraint applied from birth, not just at breeding."""
        d = build_template(ai.island, bank, rng)
        if ai.set_key in SPEC_ANCHORS:
            d = enforce_purity(d, ai, bank, rng)
        return d

    pops = {ai.label: [fresh_member(ai)
                       for _ in range(island_pop(ai))] for ai in islands}

    def apply_founders(ai: ArchIsland) -> int:
        """Head of the population becomes the founder decks (exact where
        purity/legality allow — enforce_purity only swaps off-type
        Pokemon); the tail stays random templates. Returns founders used."""
        fl = (founders or {}).get(ai.set_key)
        if not fl:
            return 0
        popn = pops[ai.label]
        k = min(len(fl), len(popn))
        for i in range(k):
            popn[i] = enforce_purity([int(c) for c in fl[i]], ai, bank, rng)
        return k

    if founders:
        for ai in islands:
            n = apply_founders(ai)
            if n:
                print(f"founders: {ai.label} seeded {n}/{len(pops[ai.label])}",
                      flush=True)

    def reseed_set(sk: str) -> None:
        """Below-floor plateau (D38): back to the founder classes —
        founders at the head, fresh random templates behind them."""
        for ai in islands:
            if ai.set_key != sk or ai.temperament not in ("explore",
                                                          "refine"):
                continue
            popn = pops[ai.label]
            for i in range(len(popn)):
                popn[i] = fresh_member(ai)
            apply_founders(ai)
    if seed_deck:
        # rebuild mode: half the matching mono populations start as mutated
        # variants of the seed list (purity floors add the backup attackers)
        from collections import Counter as _Counter
        from .pool import BASIC_ENERGY as _BE
        et = _Counter(p.by_id[c]["energyType"] for c in seed_deck
                      if p.by_id[c]["cardType"] == _BE)
        dom = et.most_common(1)[0][0] if et else 1
        for ai in islands:
            if ai.temperament in ("explore", "refine") and \
                    dom in ai.island.allowed:
                popn = pops[ai.label]
                for i in range(len(popn) // 2):
                    popn[i] = enforce_purity(
                        mutate(list(seed_deck), ai.island, bank, rng),
                        ai, bank, rng)
    if seed_elites:
        # sprint handoff: for each elite record ({"set": "mono-X", "deck":
        # [...]}), half the matching mono populations start as mutated,
        # purity-repaired variants of that elite.
        by_set: dict[str, list[list[int]]] = {}
        for rec in seed_elites:
            by_set.setdefault(rec["set"], []).append(
                [int(c) for c in rec["deck"]])
        for ai in islands:
            if ai.temperament in ("explore", "refine") and ai.set_key in by_set:
                popn = pops[ai.label]
                stock = by_set[ai.set_key]
                for i in range(len(popn) // 2):
                    popn[i] = enforce_purity(
                        mutate(list(stock[i % len(stock)]), ai.island, bank,
                               rng), ai, bank, rng)
    cache: dict[tuple, tuple] = {}
    best_seen: dict[str, tuple[int, float]] = {}   # set_key -> (era, best)
    waves: dict[str, int] = {}                     # set_key -> migrations in
    frozen: set[str] = set()
    prev_elite: list[int] | None = None
    t0 = time.time()

    # ---- resumable state (a kill loses at most the in-progress era) ------
    state_path = run_dir / "state.json"
    archive_path = run_dir / "archive.jsonl"
    archived: set[tuple] = set()
    start_era = 0
    phase_b_entered = False
    if resume and archive_path.exists():
        with archive_path.open() as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue        # torn tail from a mid-write kill
                key = tuple(rec["k"])
                cache[key] = (rec["f"], rec["s"], rec["x"], rec.get("p"))
                archived.add(key)
    if resume and state_path.exists():
        st = json.loads(state_path.read_text())
        for label, popn in st["pops"].items():
            if label in pops:
                pops[label] = [list(d) for d in popn]
        frozen = set(st["frozen"])
        best_seen = {k: tuple(v) for k, v in st["best_seen"].items()}
        waves = dict(st.get("waves", {}))
        prev_elite = st["prev_elite"]
        start_era = st["era"] + 1
        phase_b_entered = st["phase"] == "B"
        rs = st.get("rng_state")
        if rs:
            rng.setstate((rs[0], tuple(rs[1]), rs[2]))
        t0 = time.time() - st["elapsed_h"] * 3600
        print(f"resumed: era {start_era}, phase {st['phase']}, "
              f"{st['elapsed_h']:.2f}h consumed, {len(cache)} cached evals",
              flush=True)
    for pre in preload_archives or []:
        # warm-start from another run's fitness ledger: identical genomes
        # cost nothing to re-score. Entries are NOT marked archived, so the
        # first save_state copies them into this run's own archive.
        n0 = len(cache)
        with Path(pre).open() as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = tuple(rec["k"])
                if key not in cache:
                    cache[key] = (rec["f"], rec["s"], rec["x"], rec.get("p"))
        print(f"preloaded {len(cache) - n0} evals from {pre}", flush=True)

    def save_state(era: int, phase: str) -> None:
        st = {"era": era, "phase": phase,
              "elapsed_h": round((time.time() - t0) / 3600, 4),
              "pops": pops, "frozen": sorted(frozen),
              "best_seen": {k: list(v) for k, v in best_seen.items()},
              "waves": waves,
              "prev_elite": prev_elite,
              "rng_state": _jsonable_rng(rng.getstate())}
        tmp = state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(st))
        tmp.replace(state_path)
        with archive_path.open("a") as fh:   # append-only fitness ledger
            for key, v in cache.items():
                if key in archived:
                    continue
                fh.write(json.dumps(
                    {"k": list(key), "f": v[0], "s": v[1], "x": v[2],
                     "p": v[3] if len(v) > 3 else None}) + "\n")
                archived.add(key)

    # Expensive specialists (codex ~0.9 s/game vs 16-107 ms for the rest)
    # run at a quarter of the game count during GA fitness; gates and
    # tournaments use full games (decisions.md 2026-08-07 operational call).
    slow = {i for i, e in enumerate(panel)
            if e["pilot"].get("specialist") == "codex_alakazam"}

    TERM_LAMBDA = 0.15  # D18
    EXHAUSTION = ("deck-out", "no active Pokemon")

    def fit(deck: list[int]) -> float:
        key = tuple(sorted(deck))
        if key not in cache:
            score, losses, exh = 0.0, 0, 0
            profile = []       # per-panel-entry win rate: the matchup vector
            for i, (entry, opp_pilot) in enumerate(zip(panel, panel_pilots)):
                n = max(6, games_per_opponent // 4) if i in slow \
                    else games_per_opponent
                m = play_match(pilot_a, opp_pilot, deck, entry["deck"], n)
                wr = m.win_rate(0)
                profile.append(round(wr, 4))
                score += wr * entry["weight"]
                for g in m.games:
                    if g.winner == 1:
                        losses += 1
                        exh += g.reason in EXHAUSTION
            frag = exh / losses if losses else 0.0
            cache[key] = (score - TERM_LAMBDA * frag, score, frag, profile)
        return cache[key][0]

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
        # v2 (D38): explore = tournament size 3 over the full pop;
        # refine = truncation top-4 parent pool with 2-elite preservation.
        n_elite = 1 if explore else (2 if v2 else 3)
        k = (3 if v2 else 2) if explore else 4
        trunc = v2 and not explore

        def pick_parent():
            if trunc:
                return rng.choice(sel_pool[:min(4, len(sel_pool))])[1]
            return max(rng.sample(sel_pool, k=min(k, len(sel_pool))),
                       key=lambda s: s[0])[1]

        children = [d for _, d in sel_pool[:n_elite]]
        if explore and era % FRESH_TEMPLATE_EVERY == FRESH_TEMPLATE_EVERY - 1:
            children.append(fresh_member(ai))
        while len(children) < island_pop(ai):
            pa = pick_parent()
            if rng.random() < 0.7:
                pb = pick_parent()
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
        every = migrate_every or MIGRATE_EVERY
        if era % every != every - 1:
            return
        if pfit:
            # prefetch: post-breed populations are mostly unscored children;
            # without this, elite/worst selection below evaluates them one
            # at a time on a single core (archi_r0p lost 34 min to it once).
            fresh = [d for ai in mono
                     if ai.temperament == "explore" and ai.set_key not in frozen
                     for d in pops[ai.label] + pops[f"{ai.set_key}/refine"]
                     if tuple(sorted(d)) not in cache]
            cache.update(pfit.evaluate_many(fresh))
        for ai in mono:
            if ai.temperament != "explore" or ai.set_key in frozen:
                continue
            refine = f"{ai.set_key}/refine"
            top = sorted(pops[ai.label], key=fit,
                         reverse=True)[:max(1, migrate_top)]
            order = sorted(range(len(pops[refine])),
                           key=lambda i: fit(pops[refine][i]))
            for slot, elite in zip(order, top):
                pops[refine][slot] = list(elite)
            waves[ai.set_key] = waves.get(ai.set_key, 0) + 1

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
                        "deck": all_best[1],
                        "components": cache.get(tuple(sorted(all_best[1])))},
              "gate_vs_prev_elite_500g": gate}
        (run_dir / f"era_{era:03d}.json").write_text(json.dumps(ck))
        (run_dir / "latest.json").write_text(json.dumps(ck))
        save_state(era, phase)
        print(f"era {era:3d} [{ck['elapsed_h']:.2f}h {phase}] "
              f"elite={all_best[0]:.3f} ({all_best[2]}) gate={gate} "
              f"frozen={len(frozen)}", flush=True)
        if git_commit:
            subprocess.run(["git", "add", str(run_dir)],
                           cwd=run_dir.parent.parent, capture_output=True)
            subprocess.run(["git", "commit", "-m",
                            f"archipelago {run_dir.name} era {era}"],
                           cwd=run_dir.parent.parent, capture_output=True)

    def set_raw_score(sk: str) -> float | None:
        """Raw weighted win rate (pre-D18-penalty) of the set's elite."""
        best = None
        for temp in ("explore", "refine"):
            for d in pops.get(f"{sk}/{temp}", []):
                v = cache.get(tuple(sorted(d)))
                if v and (best is None or v[0] > best[0]):
                    best = v
        return best[1] if best else None

    def plateau_check(sets: list[str], reports: dict, era: int) -> None:
        for sk in sets:
            vals = [r["best"] for l, r in reports.items()
                    if l.startswith(sk + "/")]
            if not vals:
                continue
            best = max(vals)   # set elite = best across explore+refine
            e0, b0 = best_seen.get(sk, (era, -1.0))
            if best > b0 + 0.01:
                best_seen[sk] = (era, best)
            elif era - e0 >= plateau_window:
                if v2:
                    # D38: patience is not armed until the island has
                    # received a migration wave, and a below-floor elite
                    # can never freeze its island — it reseeds instead.
                    if waves.get(sk, 0) < 1:
                        continue
                    raw = set_raw_score(sk)
                    if raw is not None and raw < floor_wr:
                        reseed_set(sk)
                        best_seen.pop(sk, None)
                        print(f"era {era}: {sk} plateaued below floor "
                              f"(raw {raw:.3f} < {floor_wr:.2f}) — reseeded "
                              f"from founder classes", flush=True)
                        continue
                frozen.add(sk)

    era = start_era
    # -------- Phase A: mono sources --------
    while (not phase_b_entered
           and (time.time() - t0) / 3600 < hours * phase_a_frac):
        live = [ai for ai in mono if ai.set_key not in frozen]
        if not live:
            break
        if pfit:
            fresh = [d for ai in live for d in pops[ai.label]
                     if tuple(sorted(d)) not in cache]
            cache.update(pfit.evaluate_many(fresh))
        reports = {ai.label: breed(ai, era) for ai in live}
        migrate_internal(era)
        plateau_check(sorted({ai.set_key for ai in live}), reports, era)
        checkpoint(era, "A", reports)
        era += 1

    if not phase_b_entered and not mono_only:
        # -------- founding burst --------
        if pfit:      # prefetch: downstream pops have never been scored
            fresh = [d for ai in downstream for d in pops[ai.label]
                     if tuple(sorted(d)) not in cache]
            cache.update(pfit.evaluate_many(fresh))
        refine_elites = {}
        for ai in mono:
            if ai.temperament == "refine":
                top2 = sorted(pops[ai.label], key=fit, reverse=True)[:2]
                refine_elites[ai.set_key] = top2
        for ai in downstream:
            sources = ai.parents or list(refine_elites)
            stock = [d for sk in sources for d in refine_elites.get(sk, [])]
            popn = pops[ai.label]
            # Batched: children are crossovers of the local elite with each
            # migrant, replacing the worst locals. Their fitness is deferred
            # to Phase B's parallel prefetch — the serial per-insertion
            # re-evaluation here used to cost more wall-clock than an era.
            local = max(popn, key=fit)
            children = [enforce_purity(
                crossover(local, migrant, ai.island, bank, rng),
                ai, bank, rng) for migrant in stock]
            order = sorted(range(len(popn)), key=lambda i: fit(popn[i]))
            keep = max(len(popn) - len(children), 1)   # elite always survives
            for slot, child in zip(order[:len(popn) - keep], children):
                popn[slot] = child

        frozen -= {ai.set_key for ai in downstream}
        best_seen.clear()

    # -------- mono sprint: deep final evaluation on reclaimed time --------
    if mono_only:
        wall = wall_hours if wall_hours is not None else hours + 0.5
        term = {}
        for sk in sorted({ai.set_key for ai in mono}):
            e0, b0 = best_seen.get(sk, (None, None))
            term[sk] = {"path": "plateau" if sk in frozen else "time-cap",
                        "last_improve_era": e0, "best": b0}
        finals = []
        for sk in sorted({ai.set_key for ai in mono}):
            seen, cands = set(), []
            for temp in ("refine", "explore"):
                for d in pops.get(f"{sk}/{temp}", []):
                    key = tuple(sorted(d))
                    if key in cache and key not in seen:
                        seen.add(key)
                        cands.append((cache[key][0], d, f"{sk}/{temp}"))
            cands.sort(key=lambda s: -s[0])
            for f, d, label in cands[:deep_top]:
                comp = cache[tuple(sorted(d))]
                finals.append({
                    "set": sk, "island": label, "ga_fitness": round(f, 4),
                    "ga_score": comp[1], "ga_frag": comp[2],
                    "ga_profile": comp[3], "deck": d, "deep_games": 0,
                    "per_opp": [{"name": e["name"], "weight": e["weight"],
                                 "w": 0, "l": 0, "draw": 0, "capped": 0,
                                 "win_reasons": {}, "loss_reasons": {}}
                                for e in panel]})

        def _flush_final() -> None:
            out = {"termination": term, "plateau_window": plateau_window,
                   "elites": finals,
                   "elapsed_h": round((time.time() - t0) / 3600, 3)}
            tmp = run_dir / "final_eval.tmp"
            tmp.write_text(json.dumps(out))
            tmp.replace(run_dir / "final_eval.json")

        _flush_final()

        def _merge(po: dict, add: dict) -> None:
            for k in ("w", "l", "draw", "capped"):
                po[k] += add[k]
            for k in ("win_reasons", "loss_reasons"):
                for r, n in add[k].items():
                    po[k][r] = po[k].get(r, 0) + n

        BLOCK, MAXG = 100, 400
        last_round_h = None
        for rnd in range(MAXG // BLOCK):
            remaining = wall - (time.time() - t0) / 3600
            need = (last_round_h * 1.3 if last_round_h is not None
                    else 0.08)
            if remaining < need:
                print(f"deep-eval stopping before round {rnd + 1}: "
                      f"{remaining:.2f}h left < {need:.2f}h needed",
                      flush=True)
                break
            rt = time.time()
            if pfit:
                # the worker pool from Phase A: one deck per worker, whole
                # panel per task — serial here once cost 12 min/round
                batch = pfit.evaluate_reasons(
                    [rec["deck"] for rec in finals], BLOCK)
                for rec, per_opp in zip(finals, batch):
                    for po, add in zip(rec["per_opp"], per_opp):
                        _merge(po, add)
                    rec["deep_games"] += BLOCK
            else:
                for rec in finals:
                    if (wall - (time.time() - t0) / 3600) < 0.06:
                        break
                    for i, entry in enumerate(panel):
                        n = max(6, BLOCK // 4) if i in slow else BLOCK
                        m = play_match(pilot_a, panel_pilots[i],
                                       rec["deck"], entry["deck"], n)
                        po = rec["per_opp"][i]
                        for g in m.games:
                            if g.winner == 0:
                                po["w"] += 1
                                po["win_reasons"][g.reason] = \
                                    po["win_reasons"].get(g.reason, 0) + 1
                            elif g.winner == 1:
                                po["l"] += 1
                                po["loss_reasons"][g.reason] = \
                                    po["loss_reasons"].get(g.reason, 0) + 1
                            elif g.winner == 2:
                                po["draw"] += 1
                            else:
                                po["capped"] += 1
                    rec["deep_games"] += BLOCK
            last_round_h = (time.time() - rt) / 3600
            _flush_final()
            print(f"deep-eval round {rnd + 1} "
                  f"[{(time.time() - t0) / 3600:.2f}h, "
                  f"{last_round_h * 60:.1f} min]", flush=True)
        if pfit:
            pfit.close()
        _flush_final()
        print("mono sprint complete: final_eval.json written", flush=True)
        return

    # -------- Phase B: duals + tri --------
    while (time.time() - t0) / 3600 < hours:
        live = [ai for ai in downstream if ai.set_key not in frozen]
        if not live:  # all converged: give remaining time back to live monos
            live = [ai for ai in mono if ai.set_key not in frozen]
            if not live:
                break
        if pfit:
            fresh = [d for ai in live for d in pops[ai.label]
                     if tuple(sorted(d)) not in cache]
            cache.update(pfit.evaluate_many(fresh))
        reports = {ai.label: breed(ai, era) for ai in live}
        plateau_check(sorted({ai.set_key for ai in live}), reports, era)
        checkpoint(era, "B", reports)
        era += 1
