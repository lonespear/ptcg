"""Local search around the shipped Ogerpon list (FEEDBACK_FOR_AUSTIN §4).

The live deck (agent/deck.csv) is the field's best response — mono-Grass into
a ~67%-weight Grimmsnarl metagame — but runs 4 Pokemon in 60 cards and, under
our own pilot, loses by exhaustion (deck-out / bench-out) far above the real
field's rate. This is NOT a GA rebuild (decisions.md Q1 rejected that): it is
a 1-4 card-swap neighborhood search around the proven list, scored under the
strong pilot (JonDayPilot -> agent.main v2 evaluator, current WEIGHTS) with
the D18 termination-mode fitness:

    fitness = play-weighted win rate vs the specialist panel
              - 0.15 * (exhaustion losses / losses)

Candidate additions are justified, not free-form:
  * Grass backup attackers: basics whose main attack the deck's 18 Grass
    energy can pay and whose online date (cost, at 1 attach/turn) is <= t3.
    Non-ex bodies are Bug Catching Set-searchable and Lively Stadium-boosted.
  * Recovery / consistency trainers: discard-pile recursion attacks both
    exhaustion modes directly (Pokemon back = bench-out; cards to deck =
    deck-out); Pokemon search makes a 5-7 Pokemon list find its bodies.
Removals come from a flex list only; the 4 Ogerpon ex core, Grow Grass,
Bug Catching Set, Boss's Orders and Lillie's Determination are untouchable,
and basic Grass energy never drops below 14 (Myriad Leaf Shower scales with
attached energy).

Every neighbor is validator-legal. Each evaluation reports per-opponent
cells (the Grimmsnarl edge must hold), the exhaustion share, and a goldfish
speed profile (first-damage quantiles must not regress materially).

Noise calibration (measured on Sebastian before the run): the same base deck
screened four times at 24 games/opponent scored weighted wr 0.63/0.64/0.47/
0.46 — the 24-game screen is a coarse filter only. The protocol is therefore
three-stage: screen (24 g/opp) -> stage-2 re-rank of the top-10 plus base at
2x100 g/opp pooled (200 games/opponent) -> top-3 by pooled stage-2 fitness.

Usage (from repo root, engine venv):
    .venv/bin/python scripts/refine_deck.py [--neighbors 84] [--games 24]
        [--confirm-games 100] [--workers 5] [--seed 20260807]
        [--out data/analysis/deck_refinement.json]
"""

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# candidate pool (card IDs from data/engine_dump; texts quoted in DECK_REFINEMENT.md)
# ---------------------------------------------------------------------------

# Backup attackers: {card_id: max copies to add}. All basics, all Grass-payable,
# all online <= t3 at one attach per turn.
ATTACKERS = {
    95: 3,    # Teal Mask Ogerpon (non-ex) 110HP: Ogre Comeback [GC] 20+20/opp bench; Mountain Stroll free (fetch 2 energy)
    349: 3,   # Teal Mask Ogerpon (non-ex) 110HP: Ogre's Hammer [GGC] 120; Grass Kagura free (search+ATTACH energy)
    27: 3,    # Iron Leaves 120HP: Avenging Edge [GCC] 100(+ if avenging); Recovery Net [G]: 2 Pokemon discard->hand
    25: 2,    # Pinsir 110HP: Superpowered Horns [GCC] 100
    785: 2,   # Genesect 120HP: Speed Attack [GGC] 110; Bug's Cannon [G] snipes 20/G-energy
    178: 2,   # Zarude 120HP: Jungle Whip [GGC] 80(+80, energy to hand); Leaf Drain [G] 20+heal
    75: 2,    # Iron Leaves ex 220HP (2 prizes): Prism Edge [GGC] 180
    198: 2,   # Durant ex 190HP (2 prizes): Vengeful Crush [GCC] 120+30/their prize taken
    255: 2,   # Maractus 110HP: Corner [C] 20 + retreat lock; 6 counters on KO
}
# names 95/349 share the "Teal Mask Ogerpon" 4-copy cap; validator enforces it.

# Recovery / consistency trainers: {card_id: max copies to add}
TRAINERS = {
    1097: 2,  # Night Stretcher (I): Pokemon or Basic Energy, discard -> hand
    1110: 1,  # Max Rod (I): up to 5 Pokemon+Basic Energy, discard -> hand
    1129: 1,  # Sacred Ash (I): up to 5 Pokemon, discard -> DECK (anti deck-out too)
    1139: 1,  # Energy Recycler (I): up to 5 Basic Energy, discard -> DECK
    1184: 2,  # Lana's Aid (S): up to 3 non-RuleBox Pokemon + Basic Energy -> hand
    1152: 2,  # Poke Pad (I): search a non-RuleBox Pokemon -> hand
    1125: 2,  # Master Ball (I): search any Pokemon -> hand
    1126: 1,  # Precious Trolley (I): search any number of Basics onto Bench
    1123: 2,  # Switch (I): protect a damaged attacker
    1203: 1,  # Surfer (S): switch + draw to 5
}

# Flex removals: {card_id: max copies removable}. Base counts in comments.
FLEX = {
    1: 4,     # Basic Grass Energy (18) — floor 14 enforced below
    1119: 3,  # Energy Search (4)  — redundant beside 18 energy + Mountain Stroll
    1122: 2,  # Pokegear 3.0 (3)
    1137: 1,  # Tool Scrapper (1)
    1147: 2,  # Jumbo Ice Cream (2)
    1223: 2,  # Harlequin (2) — coin-flip hand reset
    1251: 1,  # Lively Stadium (2) — keep >=1: +30HP on our basics
    1221: 1,  # N's Plan (1)
    1201: 1,  # Briar (1)
    1213: 2,  # Judge (4) — flex down to 2
    1127: 1,  # Tera Orb (2) — keep >=1 to find the ex
    1118: 1,  # Energy Retrieval (2)
}

ENERGY_ID = 1
ENERGY_FLOOR = 14

# Hand-built package neighbors: (label, removals list, additions list)
DESIGNED = [
    ("ogre_comeback_pkg", [1119, 1119, 1147], [95, 95, 1097]),
    ("ogre_hammer_pkg", [1119, 1119, 1122], [349, 349, 1097]),
    ("ogre_mixed_pkg", [1119, 1119, 1147, 1223], [95, 349, 1097, 1152]),
    ("iron_leaves_pkg", [1119, 1119, 1147], [27, 27, 1097]),
    ("iron_leaves_recovery", [1119, 1122, 1223, 1147], [27, 27, 1129, 1152]),
    ("iron_leaves_ex_pkg", [1119, 1119, 1213], [75, 75, 1097]),
    ("durant_pkg", [1119, 1119, 1137], [198, 198, 1097]),
    ("genesect_pkg", [1119, 1119, 1147], [785, 785, 1097]),
    ("zarude_pkg", [1119, 1119], [178, 1097]),
    ("maractus_wall", [1119, 1122], [255, 255]),
    ("recovery_only", [1119, 1147], [1097, 1129]),
    ("recovery_deep", [1119, 1119, 1147, 1223], [1097, 1097, 1129, 1139]),
    ("trolley_bodies", [1119, 1119, 1122, 1147], [95, 27, 1126, 1097]),
    ("one_stretcher", [1147], [1097]),
    ("one_ogre", [1119], [95]),
    ("one_iron_leaves", [1119], [27]),
]


def read_base() -> list[int]:
    return [int(x) for x in (ROOT / "agent" / "deck.csv").read_text().split()]


def apply_swap(base: list[int], removals: list[int], additions: list[int]):
    deck = list(base)
    for cid in removals:
        if cid in deck:
            deck.remove(cid)
        else:
            return None
    deck.extend(additions)
    if len(deck) != 60 or deck.count(ENERGY_ID) < ENERGY_FLOOR:
        return None
    return sorted(deck)


def generate_neighbors(base: list[int], n_target: int, seed: int):
    """Designed packages + seeded random 1-4 swaps; unique, validator-legal."""
    from ptcg.creation.validator import validate

    rng = random.Random(seed)
    base_key = tuple(sorted(base))
    out: dict[tuple, dict] = {}

    def admit(label, removals, additions):
        deck = apply_swap(base, removals, additions)
        if deck is None:
            return False
        key = tuple(deck)
        if key == base_key or key in out:
            return False
        if not validate(deck).legal:
            return False
        out[key] = {"label": label, "removals": sorted(removals),
                    "additions": sorted(additions), "deck": deck}
        return True

    for label, rem, add in DESIGNED:
        admit(label, rem, add)

    add_pool = list(ATTACKERS.items()) + list(TRAINERS.items())
    tries = 0
    while len(out) < n_target and tries < 20000:
        tries += 1
        k = rng.randint(1, 4)
        additions: list[int] = []
        add_counts: Counter = Counter()
        for _ in range(k):
            cid, cap = rng.choice(add_pool)
            if add_counts[cid] < cap:
                additions.append(cid)
                add_counts[cid] += 1
        if not additions:
            continue
        k = len(additions)
        removals: list[int] = []
        rem_counts: Counter = Counter()
        flex_ids = list(FLEX)
        rng.shuffle(flex_ids)
        for cid in flex_ids * 2:
            if len(removals) == k:
                break
            if rem_counts[cid] < FLEX[cid] and rng.random() < 0.5:
                removals.append(cid)
                rem_counts[cid] += 1
        if len(removals) != k:
            continue
        admit(f"rand_{len(out):03d}", removals, additions)
    return list(out.values())


# ---------------------------------------------------------------------------
# parallel evaluation (one engine per process; modeled on creation/parallel.py)
# ---------------------------------------------------------------------------

_STATE: dict = {}
TERM_LAMBDA = 0.15                       # D18
EXHAUSTION = ("deck-out", "no active Pokemon")


def _init_worker(root: str, nice: int) -> None:
    os.nice(nice)
    os.chdir(root)
    sys.path.insert(0, root)
    import ptcg.creation  # noqa: F401 — engine bootstrap
    from ptcg.creation.pilots import JonDayPilot
    from ptcg.creation.specialist_panel import (build_specialist_panel,
                                                make_panel_pilots)
    factory = lambda s: JonDayPilot(seed=s, search=False)  # noqa: E731
    panel = build_specialist_panel(Path(root) / "agent" / "deck_priors.json")
    _STATE["panel"] = panel
    _STATE["pilots"] = make_panel_pilots(panel, factory)
    _STATE["candidate"] = factory(101 + os.getpid() % 97)
    _STATE["slow"] = {i for i, e in enumerate(panel)
                     if e["pilot"].get("specialist") == "codex_alakazam"}


def _eval_deck(task: tuple) -> tuple:
    """(deck, games, goldfish_games) -> (key, result dict with cells)."""
    from ptcg.creation.harness import play_match
    from ptcg.creation.goldfish import profile

    deck, games, goldfish_games = task
    st = _STATE
    cells, score, wins, decided, losses, exh = [], 0.0, 0, 0, 0, 0
    for i, (entry, pilot) in enumerate(zip(st["panel"], st["pilots"])):
        n = max(6, games // 4) if i in st["slow"] else games
        m = play_match(st["candidate"], pilot, deck, entry["deck"], n)
        wr = m.win_rate(0)
        score += wr * entry["weight"]
        reasons: Counter = Counter()
        for g in m.games:
            if g.winner in (0, 1):
                decided += 1
                wins += g.winner == 0
            if g.winner == 1:
                losses += 1
                exh += g.reason in EXHAUSTION
                reasons[g.reason or "?"] += 1
        cells.append({"opponent": entry["name"], "weight": entry["weight"],
                      "n": n, "wr": round(wr, 4),
                      "loss_reasons": dict(reasons),
                      "median_turns": sorted(g.turns for g in m.games)[len(m.games) // 2]})
    frag = exh / losses if losses else 0.0
    gf = profile(deck, n_games=goldfish_games)
    gf.pop("first_damage_turns", None)
    result = {
        "weighted_wr": round(score, 4),
        "exhaustion_share": round(frag, 4),
        "fitness": round(score - TERM_LAMBDA * frag, 4),
        "raw_wr": round(wins / decided, 4) if decided else None,
        "losses": losses, "exhaustion_losses": exh,
        "cells": cells,
        "goldfish": {
            "median_first_damage_turn": gf["median_first_damage_turn"],
            "first_damage_turn_quantiles": gf["first_damage_turn_quantiles"],
            "mean_damage_by_turn5": gf["mean_damage_by_turn5"],
            "games_with_no_damage": gf["games_with_no_damage"],
            "n_games": gf["n_games"],
        },
    }
    return tuple(sorted(deck)), result


def pool_results(a: dict, b: dict) -> dict:
    """Pool two independent evaluations of the same deck (cells summed;
    wr pooled by decided-game weight; goldfish kept from the first)."""
    cells = []
    score = 0.0
    for ca, cb in zip(a["cells"], b["cells"]):
        n = ca["n"] + cb["n"]
        wr = (ca["wr"] * ca["n"] + cb["wr"] * cb["n"]) / n
        reasons = Counter(ca["loss_reasons"]) + Counter(cb["loss_reasons"])
        cells.append({"opponent": ca["opponent"], "weight": ca["weight"],
                      "n": n, "wr": round(wr, 4),
                      "loss_reasons": dict(reasons),
                      "median_turns": round((ca["median_turns"]
                                             + cb["median_turns"]) / 2, 1)})
        score += wr * ca["weight"]
    losses = a["losses"] + b["losses"]
    exh = a["exhaustion_losses"] + b["exhaustion_losses"]
    frag = exh / losses if losses else 0.0
    return {"weighted_wr": round(score, 4),
            "exhaustion_share": round(frag, 4),
            "fitness": round(score - TERM_LAMBDA * frag, 4),
            "raw_wr": None, "losses": losses, "exhaustion_losses": exh,
            "cells": cells, "goldfish": a["goldfish"],
            "replicates": [{"weighted_wr": r["weighted_wr"],
                            "exhaustion_share": r["exhaustion_share"],
                            "fitness": r["fitness"]} for r in (a, b)]}


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def card_name(cid: int, names: dict) -> str:
    return f"{names.get(cid, '?')} [{cid}]"


def diff_str(entry: dict, names: dict) -> str:
    rem = Counter(entry["removals"])
    add = Counter(entry["additions"])
    parts = [f"-{n} {card_name(c, names)}" for c, n in sorted(rem.items())]
    parts += [f"+{n} {card_name(c, names)}" for c, n in sorted(add.items())]
    return ", ".join(parts)


def deck_lines(deck: list[int], names: dict) -> list[str]:
    return [f"  {n}x {card_name(c, names)}"
            for c, n in sorted(Counter(deck).items())]


def grimm_cell(res: dict) -> str:
    cells = [c for c in res["cells"] if "Grimmsnarl" in c["opponent"]]
    tot_w = sum(c["weight"] for c in cells)
    wr = sum(c["wr"] * c["weight"] for c in cells) / tot_w if tot_w else 0.0
    return f"{wr:.3f}"


def kang_cell(res: dict) -> str:
    cells = [c for c in res["cells"] if "Kangaskhan" in c["opponent"]]
    return f"{cells[0]['wr']:.3f}" if cells else "n/a"


def write_markdown(path: Path, payload: dict, names: dict) -> None:
    base = payload["base"]
    lines = ["# Deck refinement: local search around the shipped Ogerpon list",
             "",
             f"Generated {payload['generated']} on {payload['host']} — "
             f"{payload['n_neighbors']} validator-legal neighbors (1-4 card "
             f"swaps), {payload['games_screen']} games/opponent screen under "
             f"JonDayPilot(search=False, v2 WEIGHTS) vs the specialist panel; "
             f"top-10 re-ranked at 2x{payload['games_confirm']} games/opponent "
             f"pooled (the 'confirm' rows below; the 24-game screen was "
             f"noise-calibrated at ~±0.09 weighted wr, so only the pooled "
             f"read ranks). D18 fitness = weighted wr − 0.15 × "
             f"exhaustion-loss share.",
             "", "## Baseline (agent/deck.csv, the 52.8% field list)", ""]
    for tag in ("screen", "confirm"):
        r = base.get(tag)
        if not r:
            continue
        lines.append(
            f"- {tag}: fitness {r['fitness']}, weighted wr {r['weighted_wr']}, "
            f"exhaustion share {r['exhaustion_share']} "
            f"({r['exhaustion_losses']}/{r['losses']} losses), "
            f"vs-Grimmsnarl {grimm_cell(r)}, vs-Kangaskhan {kang_cell(r)}, "
            f"goldfish median first damage t{r['goldfish']['median_first_damage_turn']}, "
            f"dmg-by-t5 {r['goldfish']['mean_damage_by_turn5']}")
    lines += ["", "## Top 3 neighbors", ""]
    for i, entry in enumerate(payload["top3"], 1):
        lines += [f"### #{i}: {entry['label']}", "",
                  f"Diff vs base: {diff_str(entry, names)}", ""]
        for tag in ("screen", "confirm"):
            r = entry.get(tag)
            if not r:
                continue
            lines.append(
                f"- {tag}: fitness {r['fitness']}, weighted wr "
                f"{r['weighted_wr']}, exhaustion share {r['exhaustion_share']} "
                f"({r['exhaustion_losses']}/{r['losses']} losses), "
                f"vs-Grimmsnarl {grimm_cell(r)}, vs-Kangaskhan {kang_cell(r)}, "
                f"goldfish median first damage "
                f"t{r['goldfish']['median_first_damage_turn']}, "
                f"dmg-by-t5 {r['goldfish']['mean_damage_by_turn5']}")
        lines += ["", "Full 60:", ""] + deck_lines(entry["deck"], names) + [""]
        conf = entry.get("confirm")
        if conf:
            lines += ["Per-opponent (confirmation):", ""]
            lines += [f"  w={c['weight']:.3f} {c['opponent'][:32]:34s} "
                      f"wr {c['wr']:.3f} (n={c['n']}) losses {c['loss_reasons']}"
                      for c in conf["cells"]]
            lines.append("")
    lines += ["## Stage-2 field (top-10 by screen, re-ranked at "
              f"2x{payload['games_confirm']} g/opp pooled)", "",
              "| rank | label | fitness | wr | exh share | vs-Grimm | "
              "vs-Kang | diff |", "|---|---|---|---|---|---|---|---|"]
    for i, e in enumerate(payload["stage2"], 1):
        r = e["confirm"]
        lines.append(f"| {i} | {e['label']} | {r['fitness']} | "
                     f"{r['weighted_wr']} | {r['exhaustion_share']} | "
                     f"{grimm_cell(r)} | {kang_cell(r)} | "
                     f"{diff_str(e, names)} |")
    lines += ["", "## All neighbors (screen ranking)", "",
              "| rank | label | fitness | wr | exh share | vs-Grimm | diff |",
              "|---|---|---|---|---|---|---|"]
    for i, e in enumerate(payload["ranking"], 1):
        r = e["screen"]
        lines.append(f"| {i} | {e['label']} | {r['fitness']} | "
                     f"{r['weighted_wr']} | {r['exhaustion_share']} | "
                     f"{grimm_cell(r)} | {diff_str(e, names)} |")
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--neighbors", type=int, default=84)
    ap.add_argument("--games", type=int, default=24)
    ap.add_argument("--confirm-games", type=int, default=100)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260807)
    ap.add_argument("--goldfish-games", type=int, default=40)
    ap.add_argument("--nice", type=int, default=10)
    ap.add_argument("--out", default="data/analysis/deck_refinement.json")
    args = ap.parse_args()

    os.nice(args.nice)
    os.chdir(ROOT)
    import ptcg.creation  # noqa: F401 — engine bootstrap (main proc: names only)
    from ptcg.creation.pool import pool

    names = {cid: c["name"] for cid, c in pool().by_id.items()}
    base = read_base()
    neighbors = generate_neighbors(base, args.neighbors, args.seed)
    print(f"{len(neighbors)} neighbors generated (target {args.neighbors})",
          flush=True)

    exe = ProcessPoolExecutor(
        max_workers=args.workers, mp_context=get_context("spawn"),
        initializer=_init_worker, initargs=(str(ROOT), args.nice))

    # ---- screen: base + every neighbor, same protocol, same batch ----
    t0 = time.time()
    tasks = [(base, args.games, args.goldfish_games)] + \
            [(e["deck"], args.games, args.goldfish_games) for e in neighbors]
    results: dict[tuple, dict] = {}
    for i, (key, res) in enumerate(exe.map(_eval_deck, tasks, chunksize=1)):
        results[key] = res
        print(f"  screen {i + 1}/{len(tasks)} fitness={res['fitness']} "
              f"wr={res['weighted_wr']} exh={res['exhaustion_share']} "
              f"[{(time.time() - t0) / 60:.1f}m]", flush=True)

    base_entry = {"label": "BASE", "deck": sorted(base), "removals": [],
                  "additions": [], "screen": results[tuple(sorted(base))]}
    for e in neighbors:
        e["screen"] = results[tuple(e["deck"])]
    ranking = sorted(neighbors, key=lambda e: -e["screen"]["fitness"])

    # ---- stage 2: top-10 + base, two independent 100 g/opp replicates ----
    # (the 24-game screen is noise-calibrated at ~±0.09 on weighted wr; the
    # pooled 200 games/opponent read is what ranks candidates)
    stage2 = ranking[:10]
    entries = [base_entry] + stage2
    reps: dict[tuple, list] = {tuple(e["deck"]): [] for e in entries}
    s2_tasks = [(e["deck"], args.confirm_games,
                 args.goldfish_games * 2 if rep == 0 else 4)
                for rep in range(2) for e in entries]
    for i, (key, res) in enumerate(exe.map(_eval_deck, s2_tasks,
                                           chunksize=1)):
        reps[key].append(res)
        print(f"  stage2 {i + 1}/{len(s2_tasks)} fitness={res['fitness']} "
              f"wr={res['weighted_wr']} exh={res['exhaustion_share']} "
              f"[{(time.time() - t0) / 60:.1f}m]", flush=True)
    exe.shutdown()
    for e in entries:
        a, b = reps[tuple(e["deck"])]
        e["confirm"] = pool_results(a, b)
    stage2.sort(key=lambda e: -e["confirm"]["fitness"])
    top3 = stage2[:3]

    payload = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "host": os.uname().nodename,
        "protocol": {
            "pilot": "JonDayPilot(search=False), agent.main v2 WEIGHTS",
            "panel": "specialist_panel top-8 (codex slot at games//4, floor 6)",
            "fitness": f"weighted_wr - {TERM_LAMBDA} * exhaustion_share (D18)",
            "seed": args.seed,
        },
        "games_screen": args.games, "games_confirm": args.confirm_games,
        "n_neighbors": len(neighbors),
        "base": base_entry,
        "top3": top3,
        "stage2": [{k: e[k] for k in
                    ("label", "removals", "additions", "deck", "screen",
                     "confirm")} for e in stage2],
        "ranking": [{k: e[k] for k in
                     ("label", "removals", "additions", "deck", "screen")}
                    for e in ranking],
        "elapsed_min": round((time.time() - t0) / 60, 1),
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))
    write_markdown(out.parent / "DECK_REFINEMENT.md", payload, names)
    print(f"wrote {out} and DECK_REFINEMENT.md "
          f"({payload['elapsed_min']} min)", flush=True)


if __name__ == "__main__":
    main()
