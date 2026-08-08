"""Seeded archipelago v4 — D46 chains x (explore + refine), D38-D41 rules.

Chains and objectives (cell weights computed at runtime from the
ladder-representative panel, data/panel_ladder_v3.json, whose cell
weights are 700+-band encounter shares — so "panel weights" IS the
ladder objective):

  spec-Ogerpon       Grass purity; hard anchor >=2x Teal Mask Ogerpon
                     ex in the repair path (D38a). Fitness: panel
                     weights (v3 panels are representative; the old
                     hand-set 55% Grimmsnarl mass is superseded).
  grimmsnarl-mirror  Darkness purity, the field's own weapon. Fitness:
                     2/3 mass on the Grimmsnarl cells (the mirror),
                     1/3 spread over the rest by panel weight.
  engine             low-energy draw-engine discovery chain: total
                     basic energy capped at 10 in the repair path;
                     energy type free (the deck's own base). Fitness:
                     uniform cell weights (discovery, not the ladder
                     objective).
  archaludon         harvest-then-refine of the ladder's
                     Archaludon-Duraludon-Cinderace shell (our worst
                     hole). Allowed energy = union of its founders'
                     bases. Fitness: panel weights.
  counter-900        energy type unconstrained: a retype mutation
                     (p=0.2) swaps the whole energy base to a random
                     type and pulls in that type's attacker lines.
                     Fitness: 50% mass on the 900+-band archetype
                     cells (Grimmsnarl + Dudunsparce-Alakazam +
                     mono-Kangaskhan), 50% over the rest by panel
                     weight, plus the neutral floor.
  rainbow-Kanga      the field's multi-energy Mega Kangaskhan toolbox
                     chain; allowed energy = union of its founders'
                     bases. Fitness: panel weights.

Neutral-matchup floor (D39 "sensible version, documented"): floored
chains pay NEUTRAL_SLOPE * max(0, NEUTRAL_FLOOR - mean win rate over
the chain's NON-target cells). At floor 0.15 / slope 2.0 a candidate
that zeroes its neutral game pays -0.30 fitness — more than any
target-cell gain can buy back — while any candidate holding >=0.15 on
neutral cells pays nothing. Target mass already prices the target
cells; the floor only vetoes neutral collapse.

Pilot split (D40): the greedy pilot screens every member cheaply; each
chain's real pilot re-scores the screen-top members (top 6 explore /
top 4 refine) and ONLY those real-tier scores decide elites, refine
parents, migrants, plateau, the below-floor rule, and the deep final
eval. Selection noise beats selection blindness, so real games per
opponent are fewer than screen games. Purity/anchor repairs, D18
termination pricing (fitness -= 0.15 * exhaustion share of losses,
per tier), and resumable state carry over from the v2 archipelago.

Cross-island reporting: island fitnesses are weighted differently, so
the checkpoint's global "elite" is ranked by uniform panel win rate
(real tier), not island fitness.
"""

import json
import random
import time
from pathlib import Path

from .archipelago import (GATE_GAMES, ArchIsland, _jsonable_rng,
                          _mean_distance, enforce_purity)
from .ga import GeneBank, Island, build_template, crossover, mutate, repair
from .harness import play_match
from .pool import BASIC_ENERGY, POKEMON, pool

TERM_LAMBDA = 0.15          # D18
EXHAUSTION = ("deck-out", "no active Pokemon")
NEUTRAL_FLOOR = 0.15        # D39 neutral floor (see module docstring)
NEUTRAL_SLOPE = 2.0
DIVERSITY_LAMBDA = 0.1      # D11; screen tier only
FRESH_TEMPLATE_EVERY = 10
PAYABLE_FLOOR = 6           # attacker slots payable from the deck's base

GRIMM = "Grimmsnarl"
C900 = ("Grimmsnarl", "Dudunsparce-Alakazam", "Kangaskhan (mono)")
RAINBOW_FALLBACK = frozenset({1, 3, 4, 5, 6})
ENGINE_ENERGY_CAP = 10
UNIFORM_CHAINS = {"engine"}     # fitness ignores panel weights
# chains whose allowed-energy set is the union of their founders' bases
FOUNDER_TYPED = ("rainbow-Kanga", "archaludon")

# set_key -> (default real pilot kind, target matcher, target mass, floored)
CHAINS = {
    "spec-Ogerpon": ("jon", None, 0.0, False),
    "grimmsnarl-mirror": ("grimmsnarl", lambda n: GRIMM in n, 2 / 3, False),
    "engine": ("jon", None, 0.0, False),
    # archaludon external: deck-specialized harvest agent on its own
    # archetype — same tradeoff as the grimmsnarl external (D40).
    "archaludon": ("archaludon", None, 0.0, False),
    "counter-900": ("jon", lambda n: any(t in n for t in C900), 0.5, True),
    # Rainbow real pilot: the 2026-08-07 audition on the rainbow founder
    # measured jon 0.069 / greedy 0.208 / kanga external 0.181 vs three
    # panel cells — no pilot is competent on the toolbox; the kanga
    # external at least knows the Mega Kangaskhan + Crispin core, so it
    # takes the real tier (see the run README's pilot-split table).
    "rainbow-Kanga": ("kanga", None, 0.0, False),
}


def build_islands(bank: GeneBank, founders: dict | None,
                  chains: list[str]) -> list[ArchIsland]:
    ft: dict[str, set[int]] = {}
    for sk in FOUNDER_TYPED:
        rt: set[int] = set()
        for d in (founders or {}).get(sk, []):
            for c in d:
                card = bank.pool.by_id.get(int(c))
                if card and card["cardType"] == BASIC_ENERGY:
                    rt.add(card["energyType"])
        ft[sk] = rt
    specs = {"spec-Ogerpon": frozenset({1}),
             "grimmsnarl-mirror": frozenset({7}),
             "engine": frozenset(),
             "archaludon": frozenset(ft["archaludon"]) or frozenset({4, 8}),
             "counter-900": frozenset(),
             "rainbow-Kanga": (frozenset(ft["rainbow-Kanga"])
                               or RAINBOW_FALLBACK)}
    out = []
    for sk in chains:
        isl = Island(sk, specs[sk])
        out.append(ArchIsland(isl, sk, "explore"))
        out.append(ArchIsland(isl, sk, "refine"))
    return out


def build_weights(panel: list[dict], chains: list[str]) -> dict:
    """set_key -> (per-cell weights, neutral cell indices, floored).

    Base weights are the panel's own (representative) cell weights;
    UNIFORM_CHAINS use uniform cells; target chains concentrate `mass`
    on their target cells and spread the rest by panel weight."""
    n = len(panel)
    base = [e["weight"] for e in panel]
    out = {}
    for sk in chains:
        _pilot, match, mass, floored = CHAINS[sk]
        if sk in UNIFORM_CHAINS:
            out[sk] = ([1.0 / n] * n, list(range(n)), floored)
            continue
        tgt = ([i for i, e in enumerate(panel) if match(e["name"])]
               if match else [])
        if not tgt:
            out[sk] = (list(base), list(range(n)), floored)
            continue
        rest = [i for i in range(n) if i not in tgt]
        tw = sum(base[i] for i in tgt)
        rw = sum(base[i] for i in rest)
        w = [0.0] * n
        for i in tgt:
            w[i] = mass * base[i] / tw
        for i in rest:
            w[i] = (1.0 - mass) * base[i] / rw
        out[sk] = (w, rest, floored)
    return out


def run_seeded(run_dir: Path, panel_path: Path, hours: float = 10.5,
               wall_hours: float = 11.5, screen_games: int = 8,
               real_games: int = 6, workers: int = 7, seed: int = 74,
               plateau_window: int = 12, migrate_every: int = 10,
               migrate_top: int = 2, floor_wr: float = 0.35,
               explore_pop: int = 24, refine_pop: int = 10,
               real_top_explore: int = 6, real_top_refine: int = 4,
               founders: dict | None = None, resume: bool = False,
               deep_top: int = 3, deep_block: int = 100,
               deep_max: int = 400,
               pilot_overrides: dict | None = None,
               chains: list[str] | None = None,
               improve_eps: float = 0.005) -> None:
    rng = random.Random(seed)
    bank = GeneBank(pool())
    p = bank.pool
    chains = list(chains) if chains else list(CHAINS)
    islands = build_islands(bank, founders, chains)
    run_dir.mkdir(parents=True, exist_ok=True)

    real_kind = {sk: CHAINS[sk][0] for sk in chains}
    real_kind.update(pilot_overrides or {})

    from .specialist_panel import build_specialist_panel, panel_report
    n_cells = len(json.loads(panel_path.read_text())["decks"])
    panel = build_specialist_panel(panel_path, top_n=n_cells)
    wspec = build_weights(panel, chains)
    (run_dir / "panel.json").write_text(json.dumps(panel))
    print("panel:\n" + panel_report(panel), flush=True)
    for sk, (w, neutral, floored) in wspec.items():
        n_t = sum(1 for x in w if x > 1.0 / len(w) + 1e-9)
        print(f"weights {sk}: target cells {n_t}, "
              f"target mass {sum(x for x in w if x > 1.0/len(w)+1e-9):.2f}, "
              f"floor {'on' if floored else 'off'}, "
              f"real pilot {real_kind[sk]}", flush=True)

    pfit = None
    if workers > 1:
        from .parallel import ParallelFitness
        pfit = ParallelFitness(panel, "jon", screen_games, workers,
                               str(Path.cwd()))

    # ---- serial fallback pilots (workers == 1, and the era gate) ---------
    from .pilots import ExternalPilot, GreedyPilot, JonDayPilot
    from .specialist_panel import make_panel_pilots
    factory = {"greedy": lambda s: GreedyPilot(seed=s),
               "jon": lambda s: JonDayPilot(seed=s, search=False)}
    pilot_a, pilot_b = factory["jon"](101), factory["jon"](202)
    panel_pilots = None
    cand_pilots: dict = {}

    def parent_cand(kind: str):
        if kind not in cand_pilots:
            if kind in factory:
                cand_pilots[kind] = factory[kind](301)
            else:
                try:
                    cand_pilots[kind] = ExternalPilot(
                        str(Path("external") / f"{kind}_agent.py"))
                except Exception:  # noqa: BLE001 — degraded, not dead
                    cand_pilots[kind] = factory["jon"](301)
        return cand_pilots[kind]

    slow_cells = {i for i, e in enumerate(panel)
                  if e["pilot"].get("specialist") == "codex_alakazam"}
    screen_pilots = None

    def serial_profiles(items):
        nonlocal panel_pilots, screen_pilots
        if panel_pilots is None:
            panel_pilots = make_panel_pilots(panel, factory["jon"])
            screen_pilots = [factory["jon"](700 + i) if i in slow_cells
                             else pl for i, pl in enumerate(panel_pilots)]
        out = {}
        for d, kind, games in items:
            cand = parent_cand(kind)
            screen = kind == "greedy"
            opps = screen_pilots if screen else panel_pilots
            losses = exh = 0
            profile = []
            for i, (entry, op) in enumerate(zip(panel, opps)):
                n = games if screen else (
                    max(3, games // 4) if i in slow_cells else games)
                m = play_match(cand, op, d, entry["deck"], n)
                profile.append(round(m.win_rate(0), 4))
                for g in m.games:
                    if g.winner == 1:
                        losses += 1
                        exh += g.reason in EXHAUSTION
            out[(kind, tuple(sorted(d)))] = (
                profile, exh / losses if losses else 0.0)
        return out

    # ---- profile cache: (pilot_kind, deck_key) -> (profile, frag) --------
    pcache: dict[tuple, tuple] = {}

    def key(d: list[int]) -> tuple:
        return tuple(sorted(d))

    def fetch(items: list[tuple]) -> None:
        todo, seen = [], set()
        for d, kind, games in items:
            k2 = (kind, key(d))
            if k2 not in pcache and k2 not in seen:
                seen.add(k2)
                todo.append((d, kind, games))
        if not todo:
            return
        if pfit:
            pcache.update(pfit.evaluate_profiles(todo))
        else:
            pcache.update(serial_profiles(todo))

    def island_fit(sk: str, profile: list[float], frag: float) -> float:
        w, neutral, floored = wspec[sk]
        f = sum(wi * pi for wi, pi in zip(w, profile)) - TERM_LAMBDA * frag
        if floored and neutral:
            nm = sum(profile[i] for i in neutral) / len(neutral)
            f -= NEUTRAL_SLOPE * max(0.0, NEUTRAL_FLOOR - nm)
        return f

    def raw_wr(profile: list[float]) -> float:
        return sum(profile) / len(profile)

    def screen_fit(sk: str, d: list[int]) -> float:
        v = pcache.get(("greedy", key(d)))
        return island_fit(sk, v[0], v[1]) if v else -9.0

    def real_val(sk: str, d: list[int]) -> tuple | None:
        return pcache.get((real_kind[sk], key(d)))

    def real_fit(sk: str, d: list[int]) -> float:
        v = real_val(sk, d)
        return island_fit(sk, v[0], v[1]) if v else -9.0

    # ---- structural repair per chain -------------------------------------
    def cap_energy(out: list[int]) -> list[int]:
        """Engine chain: cap total basic energy at ENGINE_ENERGY_CAP by
        duplicating the deck's own non-energy cards (copy-cap and ACE
        SPEC respected; duplication is closure-safe)."""
        e_idx = [i for i, c in enumerate(out)
                 if p.by_id[c]["cardType"] == BASIC_ENERGY]
        excess = len(e_idx) - ENGINE_ENERGY_CAP
        if excess <= 0:
            return out
        rng.shuffle(e_idx)
        from collections import Counter as _C
        counts = _C(p.by_id[c]["name"] for c in out
                    if p.by_id[c]["cardType"] != BASIC_ENERGY)
        for i in e_idx[:excess]:
            cands = [c for c in out
                     if p.by_id[c]["cardType"] != BASIC_ENERGY
                     and not p.by_id[c]["aceSpec"]
                     and counts[p.by_id[c]["name"]] < 4]
            if not cands:
                break
            pick = rng.choice(cands)
            out[i] = pick
            counts[p.by_id[pick]["name"]] += 1
        return out

    def repair_member(deck: list[int], ai: ArchIsland) -> list[int]:
        if ai.set_key in FOUNDER_TYPED:
            allowed = set(ai.island.allowed)
        elif ai.set_key in ("counter-900", "engine"):
            allowed = {p.by_id[c]["energyType"] for c in deck
                       if p.by_id[c]["cardType"] == BASIC_ENERGY}
        else:
            return enforce_purity(deck, ai, bank, rng)
        # Payable floor: >=PAYABLE_FLOOR Pokemon with some attack payable
        # from the deck's energy base as a whole. Per-type floors would
        # demand 6 attackers of each of 5 types — impossible for a toolbox.
        out = list(deck)
        have, replaceable = 0, []
        for i, cid in enumerate(out):
            c = p.by_id[cid]
            if c["cardType"] != POKEMON:
                continue
            ok = not c["attacks"] or any(
                set(p.typed_cost(a)) <= allowed for a in c["attacks"]
                if a in p.attack_by_id)
            if ok:
                have += 1
            else:
                replaceable.append(i)
        if have < PAYABLE_FLOOR and allowed:
            lines = bank.eligible_lines(allowed)
            if lines:
                rng.shuffle(replaceable)
                for i in replaceable[:PAYABLE_FLOOR - have]:
                    out[i] = bank.pick_print(
                        rng.choice(rng.choice(lines)), rng)
        out = repair(out, ai.island, bank, rng)
        if ai.set_key == "engine":
            out = cap_energy(out)
        return out

    def mutate_member(deck: list[int], ai: ArchIsland,
                      explore: bool) -> list[int]:
        out = list(deck)
        work = ai.island
        if ai.set_key == "engine":
            types = {p.by_id[c]["energyType"] for c in out
                     if p.by_id[c]["cardType"] == BASIC_ENERGY}
            work = Island(ai.set_key,
                          frozenset(types) or frozenset({rng.randint(1, 8)}))
        elif ai.set_key == "counter-900":
            if rng.random() < 0.2:
                # retype: the counter island picks its weapon (D39)
                t = rng.randint(1, 8)
                eid = bank.energy_id_by_type[t]
                for i, c in enumerate(out):
                    if (p.by_id[c]["cardType"] == BASIC_ENERGY
                            and rng.random() < 0.85):
                        out[i] = eid
                lines = bank.eligible_lines({t})
                for line in rng.sample(lines, k=min(2, len(lines))):
                    for name in line:
                        out[rng.randrange(60)] = bank.pick_print(name, rng)
                work = Island(ai.set_key, frozenset({t}))
            else:
                types = {p.by_id[c]["energyType"] for c in out
                         if p.by_id[c]["cardType"] == BASIC_ENERGY}
                work = Island(ai.set_key, frozenset(types)
                              or frozenset({rng.randint(1, 8)}))
        out = mutate(out, work, bank, rng)
        if explore:
            out = mutate(out, work, bank, rng)
        return out

    def fresh_member(ai: ArchIsland) -> list[int]:
        isl = ai.island
        if ai.set_key in ("counter-900", "engine"):
            isl = Island(ai.set_key, frozenset({rng.randint(1, 8)}))
        return repair_member(build_template(isl, bank, rng), ai)

    def island_pop(ai: ArchIsland) -> int:
        return explore_pop if ai.temperament == "explore" else refine_pop

    pops = {ai.label: [fresh_member(ai) for _ in range(island_pop(ai))]
            for ai in islands}

    def apply_founders(ai: ArchIsland) -> int:
        """Exact founders at the head (structural repairs only), mutated
        founder variants over half the remaining slots, randoms behind."""
        fl = (founders or {}).get(ai.set_key) or []
        popn = pops[ai.label]
        k = min(len(fl), len(popn))
        for i in range(k):
            popn[i] = repair_member([int(c) for c in fl[i]], ai)
        if k:
            for j in range((len(popn) - k) // 2):
                popn[k + j] = repair_member(
                    mutate_member(list(popn[j % k]), ai, explore=False), ai)
        return k

    if founders:
        for ai in islands:
            n = apply_founders(ai)
            if n:
                print(f"founders: {ai.label} seeded {n} exact + "
                      f"{(len(pops[ai.label]) - n) // 2} variants "
                      f"/ {len(pops[ai.label])}", flush=True)

    def reseed_set(sk: str) -> None:
        for ai in islands:
            if ai.set_key == sk:
                pops[ai.label] = [fresh_member(ai)
                                  for _ in range(island_pop(ai))]
                apply_founders(ai)

    # ---- resumable state -------------------------------------------------
    best_seen: dict[str, tuple[int, float]] = {}
    waves: dict[str, int] = {}
    frozen: set[str] = set()
    prev_elite: list[int] | None = None
    t0 = time.time()
    state_path = run_dir / "state.json"
    archive_path = run_dir / "archive.jsonl"
    archived: set[tuple] = set()
    start_era = 0
    if resume and archive_path.exists():
        with archive_path.open() as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue        # torn tail from a mid-write kill
                if "pk" not in rec:
                    continue        # v2-format record: different panel/pilots
                k2 = (rec["pk"], tuple(rec["k"]))
                pcache[k2] = (rec["pr"], rec["x"])
                archived.add(k2)
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
        rs = st.get("rng_state")
        if rs:
            rng.setstate((rs[0], tuple(rs[1]), rs[2]))
        t0 = time.time() - st["elapsed_h"] * 3600
        print(f"resumed: era {start_era}, {st['elapsed_h']:.2f}h consumed, "
              f"{len(pcache)} cached profiles", flush=True)

    def save_state(era: int) -> None:
        st = {"era": era,
              "elapsed_h": round((time.time() - t0) / 3600, 4),
              "pops": pops, "frozen": sorted(frozen),
              "best_seen": {k: list(v) for k, v in best_seen.items()},
              "waves": waves, "prev_elite": prev_elite,
              "rng_state": _jsonable_rng(rng.getstate())}
        tmp = state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(st))
        tmp.replace(state_path)
        with archive_path.open("a") as fh:
            for k2, v in pcache.items():
                if k2 in archived:
                    continue
                fh.write(json.dumps({"pk": k2[0], "k": list(k2[1]),
                                     "pr": v[0], "x": v[1]}) + "\n")
                archived.add(k2)

    # ---- era machinery ---------------------------------------------------
    real_idx: dict[str, list[int]] = {}   # label -> real-tier indices

    def pick_real(ai: ArchIsland) -> list[int]:
        """Screen-top indices for the real tier; explore adds the D11
        diversity bonus so the real tier sees distinct bodies."""
        popn = pops[ai.label]
        sk = ai.set_key
        kmax = (real_top_explore if ai.temperament == "explore"
                else real_top_refine)
        if ai.temperament == "explore":
            scored = sorted(
                range(len(popn)),
                key=lambda i: -(screen_fit(sk, popn[i])
                                + DIVERSITY_LAMBDA
                                * _mean_distance(popn[i], popn, p)))
        else:
            scored = sorted(range(len(popn)),
                            key=lambda i: -screen_fit(sk, popn[i]))
        out, seen = [], set()
        for i in scored:
            k2 = key(popn[i])
            if k2 in seen:
                continue
            seen.add(k2)
            out.append(i)
            if len(out) >= kmax:
                break
        return out

    def breed(ai: ArchIsland, era: int) -> dict:
        popn = pops[ai.label]
        sk = ai.set_key
        explore = ai.temperament == "explore"
        idxs = real_idx[ai.label]
        scored_real = sorted(((real_fit(sk, popn[i]), i) for i in idxs),
                             key=lambda s: -s[0])
        taken = {key(popn[i]) for _, i in scored_real}
        tail = sorted((i for i in range(len(popn))
                       if key(popn[i]) not in taken),
                      key=lambda i: -screen_fit(sk, popn[i]))
        composite = [popn[i] for _, i in scored_real] + \
                    [popn[i] for i in tail]
        n_elite = 1 if explore else 2
        children = [list(d) for d in composite[:n_elite]]
        if explore and era % FRESH_TEMPLATE_EVERY == FRESH_TEMPLATE_EVERY - 1:
            children.append(fresh_member(ai))

        def pick_parent() -> list[int]:
            if explore:   # rank tournament, size 3, over the full pop
                return composite[min(rng.sample(range(len(composite)),
                                                k=min(3, len(composite))))]
            return rng.choice(composite[:min(4, len(composite))])

        while len(children) < island_pop(ai):
            pa = pick_parent()
            if rng.random() < 0.7:
                child = crossover(pa, pick_parent(), ai.island, bank, rng)
            else:
                child = list(pa)
            children.append(repair_member(
                mutate_member(child, ai, explore), ai))
        pops[ai.label] = children
        best_f, best_i = scored_real[0]
        bp = real_val(sk, popn[best_i])
        return {"best": round(best_f, 4), "best_deck": popn[best_i],
                "raw": round(raw_wr(bp[0]), 4), "frag": round(bp[1], 4),
                "profile": bp[0],
                "screen_best": round(
                    max(screen_fit(sk, d) for d in popn), 4)}

    def checkpoint(era: int, reports: dict) -> None:
        nonlocal prev_elite
        label, r = max(reports.items(), key=lambda kv: kv[1]["raw"])
        gate = None
        if prev_elite is not None and prev_elite != r["best_deck"]:
            m = play_match(pilot_a, pilot_b, r["best_deck"], prev_elite,
                           GATE_GAMES)
            gate = round(m.win_rate(0), 3)
        prev_elite = r["best_deck"]
        slim = {l: {k: v for k, v in rep.items() if k != "profile"}
                for l, rep in reports.items()}
        ck = {"era": era,
              "elapsed_h": round((time.time() - t0) / 3600, 3),
              "islands": slim, "frozen": sorted(frozen), "waves": waves,
              "elite": {"island": label, "raw": r["raw"],
                        "fitness": r["best"], "deck": r["best_deck"],
                        "profile": r["profile"]},
              "gate_vs_prev_elite_500g": gate}
        (run_dir / f"era_{era:03d}.json").write_text(json.dumps(ck))
        (run_dir / "latest.json").write_text(json.dumps(ck))
        save_state(era)
        print(f"era {era:3d} [{ck['elapsed_h']:.2f}h] "
              f"elite raw={r['raw']:.3f} fit={r['best']:.3f} ({label}) "
              f"gate={gate} frozen={len(frozen)} "
              f"waves={sum(waves.values())}", flush=True)

    def plateau_check(reports: dict, era: int) -> None:
        for sk in sorted({ai.set_key for ai in islands
                          if ai.set_key not in frozen}):
            recs = [r for l, r in reports.items()
                    if l.startswith(sk + "/")]
            if not recs:
                continue
            best = max(r["best"] for r in recs)
            e0, b0 = best_seen.get(sk, (era, -9.0))
            if best > b0 + improve_eps:
                best_seen[sk] = (era, best)
            elif era - e0 >= plateau_window:
                if waves.get(sk, 0) < 1:
                    continue    # patience arms after the first wave (D38)
                raw = max(r["raw"] for r in recs)
                if raw < floor_wr:
                    reseed_set(sk)
                    best_seen.pop(sk, None)
                    print(f"era {era}: {sk} plateaued below floor "
                          f"(raw {raw:.3f} < {floor_wr:.2f}) — reseeded "
                          f"from founder classes", flush=True)
                else:
                    frozen.add(sk)
                    print(f"era {era}: {sk} frozen "
                          f"(raw {raw:.3f})", flush=True)

    # ---- GA loop ---------------------------------------------------------
    era = start_era
    while (time.time() - t0) / 3600 < hours:
        live = [ai for ai in islands if ai.set_key not in frozen]
        if not live:
            print("all chains frozen — GA loop done", flush=True)
            break
        fetch([(d, "greedy", screen_games)
               for ai in live for d in pops[ai.label]])
        # explore real tier
        explorers = [ai for ai in live if ai.temperament == "explore"]
        for ai in explorers:
            real_idx[ai.label] = pick_real(ai)
        fetch([(pops[ai.label][i], real_kind[ai.set_key], real_games)
               for ai in explorers for i in real_idx[ai.label]])
        # migration: explore's real-top elites replace refine's screen-worst
        if era % migrate_every == migrate_every - 1:
            for ai in explorers:
                sk = ai.set_key
                ref = f"{sk}/refine"
                top = sorted(real_idx[ai.label],
                             key=lambda i: -real_fit(sk, pops[ai.label][i])
                             )[:migrate_top]
                worst = sorted(range(len(pops[ref])),
                               key=lambda i: screen_fit(sk, pops[ref][i]))
                for slot, i in zip(worst, top):
                    pops[ref][slot] = list(pops[ai.label][i])
                waves[sk] = waves.get(sk, 0) + 1
            print(f"era {era}: migration wave "
                  f"(top {migrate_top} explore -> refine)", flush=True)
        # refine real tier (migrants already cached from the explore side)
        refiners = [ai for ai in live if ai.temperament == "refine"]
        for ai in refiners:
            real_idx[ai.label] = pick_real(ai)
        fetch([(pops[ai.label][i], real_kind[ai.set_key], real_games)
               for ai in refiners for i in real_idx[ai.label]])
        reports = {ai.label: breed(ai, era) for ai in live}
        plateau_check(reports, era)
        checkpoint(era, reports)
        era += 1

    # ---- deep final evaluation (real pilots, reclaimed time) -------------
    term = {}
    for sk in chains:
        e0, b0 = best_seen.get(sk, (None, None))
        term[sk] = {"path": "plateau" if sk in frozen else "time-cap",
                    "last_improve_era": e0, "best": b0,
                    "waves": waves.get(sk, 0)}
    finals = []
    for sk in chains:
        kind = real_kind[sk]
        seen, cands = set(), []
        for temp in ("refine", "explore"):
            for d in pops.get(f"{sk}/{temp}", []):
                k2 = key(d)
                if k2 in seen or (kind, k2) not in pcache:
                    continue
                seen.add(k2)
                cands.append((real_fit(sk, d), d, f"{sk}/{temp}"))
        cands.sort(key=lambda s: -s[0])
        for f, d, label in cands[:deep_top]:
            pr, frag = pcache[(kind, key(d))]
            finals.append({
                "set": sk, "island": label, "pilot": kind,
                "ga_fitness": round(f, 4), "ga_raw": round(raw_wr(pr), 4),
                "ga_profile": pr, "ga_frag": frag, "deck": d,
                "deep_games": 0,
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

    def _merge(po: dict, add: dict) -> None:
        for k in ("w", "l", "draw", "capped"):
            po[k] += add[k]
        for k in ("win_reasons", "loss_reasons"):
            for r, n in add[k].items():
                po[k][r] = po[k].get(r, 0) + n

    _flush_final()
    last_round_h = None
    for rnd in range(max(0, deep_max // deep_block)):
        remaining = wall_hours - (time.time() - t0) / 3600
        need = last_round_h * 1.3 if last_round_h is not None else 0.15
        if remaining < need:
            print(f"deep-eval stopping before round {rnd + 1}: "
                  f"{remaining:.2f}h left < {need:.2f}h needed", flush=True)
            break
        rt = time.time()
        decks = [rec["deck"] for rec in finals]
        kinds = [rec["pilot"] for rec in finals]
        if pfit:
            batch = pfit.evaluate_reasons(decks, deep_block,
                                          pilot_kinds=kinds)
        else:
            batch = []
            for d, kd in zip(decks, kinds):
                if panel_pilots is None:
                    panel_pilots = make_panel_pilots(panel, factory["jon"])
                out = []
                cand = parent_cand(kd)
                for entry, op in zip(panel, panel_pilots):
                    m = play_match(cand, op, d, entry["deck"], deep_block)
                    po = {"w": 0, "l": 0, "draw": 0, "capped": 0,
                          "win_reasons": {}, "loss_reasons": {}}
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
                    out.append(po)
                batch.append(out)
        for rec, per_opp in zip(finals, batch):
            for po, add in zip(rec["per_opp"], per_opp):
                _merge(po, add)
            rec["deep_games"] += deep_block
        last_round_h = (time.time() - rt) / 3600
        _flush_final()
        print(f"deep-eval round {rnd + 1} "
              f"[{(time.time() - t0) / 3600:.2f}h, "
              f"{last_round_h * 60:.1f} min]", flush=True)
    if pfit:
        pfit.close()
    _flush_final()
    print("seeded v3 complete: final_eval.json written", flush=True)
