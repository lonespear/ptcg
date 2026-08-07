"""Postures decided on the full specialist panel, not one opponent.

Playbook entry 5's specialist arm was a single opponent (the Grimmsnarl
agent on its own list) and 576 clean games, which resolves nothing smaller
than ~5 points. This harness is the same measurement widened to the whole
fitness panel: our Ogerpon list piloted by agent/main.py as shipped, against
all eight field entries, each played by its matched harvested specialist
where one exists (make_panel_pilots' matching) and by the fitness generalist
(JonDayPilot, search off) where none does. Two arms, CABT_POSTURES=0 and 1,
identical seed blocks, seats swapped every other game.

The flag is read at agent/main.py import time, so each arm sets the env in
the worker initializer before loading our module — and clears it again
before the opponent pilot is built, so a generalist opponent (which imports
agent.main as its own module object) never receives the treatment.

Forfeit accounting is specialist_gate.py's: the harvested specialists'
last-resort fallback returns an empty selection on maxCount-0 prompts and
forfeits, so the headline is the clean rate (games nobody forfeited) with
the all-games rate reported beside it. Both aggregates are weighted by the
panel's play share, exactly as fitness weighs them.

    python scripts/posture_panel_gate.py --games 500 --workers 8
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_STATE: dict = {}


def _load_main(path: str, name: str):
    cwd = os.getcwd()
    os.chdir(Path(path).parent)
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        os.chdir(cwd)


def _init(root: str, a_path: str, deck_a: str, arm: str,
          pilot_spec: dict, deck_b: list[int]):
    # The treatment: our module reads CABT_POSTURES at import.
    os.environ["CABT_POSTURES"] = arm
    os.environ["CABT_PROTECT"] = "0"
    sys.path.insert(0, root)
    sys.path.insert(0, str(Path(root) / "engine"))
    import ptcg.creation  # noqa: F401  — engine bootstrap
    from ptcg.arena import load_deck

    mod = _load_main(a_path, "ppg_a")
    if mod.POSTURE_MATCHUP_ENABLED != (arm == "1"):
        raise RuntimeError(f"arm {arm}: flag did not land in our module")
    if arm == "1" and not mod._posture_specs():
        raise RuntimeError("arm 1: postures.json resolved empty")
    _STATE["a"] = mod.agent
    _STATE["deck_a"] = load_deck(Path(deck_a))
    _STATE["deck_b"] = list(deck_b)

    # The opponent never gets the treatment, whichever arm this is.
    os.environ["CABT_POSTURES"] = "0"
    if pilot_spec["kind"] == "external":
        from ptcg.creation.pilots import ExternalPilot
        pilot = ExternalPilot(pilot_spec["path"])
    else:
        from ptcg.creation.pilots import JonDayPilot
        pilot = JonDayPilot(seed=pilot_spec["seed"], search=False)
        if pilot._jon.POSTURE_MATCHUP_ENABLED:
            raise RuntimeError("generalist opponent imported with postures on")
    pilot.bind_deck(_STATE["deck_b"])
    _STATE["b"] = pilot


def _open_episode(agent) -> None:
    """The deck-selection call every Kaggle episode opens with — resets the
    harvested specialists' per-game state and our agent's time bank; see
    specialist_gate.py for the forfeit cascade it prevents."""
    try:
        agent({"select": None})
    except Exception:
        pass


def _play(seeds: list[int]) -> dict:
    from ptcg.arena import play_game
    out = {"a": 0, "b": 0, "draws": 0, "turns": [], "errors": [],
           "a_error_forfeits": 0, "b_error_forfeits": 0,
           "a_clean": 0, "b_clean": 0}
    for g in seeds:
        _open_episode(_STATE["a"])
        _open_episode(_STATE["b"])
        flip = g % 2 == 1
        a0, a1 = ((_STATE["b"], _STATE["a"]) if flip
                  else (_STATE["a"], _STATE["b"]))
        d0, d1 = ((_STATE["deck_b"], _STATE["deck_a"]) if flip
                  else (_STATE["deck_a"], _STATE["deck_b"]))
        r = play_game(a0, a1, d0, d1, seed=g)
        if r.error:
            out["errors"].append(r.error)
            who = 0 if r.error.startswith("agent 0") else 1
            ours = (who == 1) if flip else (who == 0)
            out["a_error_forfeits" if ours else "b_error_forfeits"] += 1
        if r.winner is None:
            out["draws"] += 1
            continue
        actual = (1 - r.winner) if flip else r.winner
        out["a" if actual == 0 else "b"] += 1
        if not r.error:
            out["a_clean" if actual == 0 else "b_clean"] += 1
        out["turns"].append(r.turns)
    return out


def _run_cell(arm: str, entry: dict, entry_idx: int, args) -> dict:
    seeds = [args.seed + 1000 * entry_idx + g for g in range(args.games)]
    chunks = [seeds[i::args.workers] for i in range(args.workers)]
    spec = entry["pilot"]
    pilot_spec = ({"kind": "external", "path": spec["path"]}
                  if spec["kind"] == "external"
                  else {"kind": "generalist", "seed": 900 + entry_idx})
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=get_context("spawn"),
                             initializer=_init,
                             initargs=(str(ROOT), str(ROOT / args.a),
                                       str(ROOT / args.deck_a), arm,
                                       pilot_spec, entry["deck"])) as pool:
        parts = list(pool.map(_play, chunks))

    res = {"a": 0, "b": 0, "draws": 0, "turns": [], "errors": [],
           "a_error_forfeits": 0, "b_error_forfeits": 0,
           "a_clean": 0, "b_clean": 0}
    for p in parts:
        for k in ("a", "b", "draws", "a_error_forfeits", "b_error_forfeits",
                  "a_clean", "b_clean"):
            res[k] += p[k]
        res["turns"] += p["turns"]
        res["errors"] += p["errors"]
    decided = res["a"] + res["b"]
    clean = res["a_clean"] + res["b_clean"]
    cell = {"entry": entry_idx, "name": entry["name"],
            "weight": entry["weight"],
            "pilot": (spec.get("specialist") if spec["kind"] == "external"
                      else "generalist"),
            "arm": arm, "games": args.games, "seed0": seeds[0],
            "decided": decided, "a_wins": res["a"], "b_wins": res["b"],
            "draws": res["draws"],
            "wr_all": round(res["a"] / decided, 4) if decided else None,
            "clean": clean, "a_clean": res["a_clean"],
            "b_clean": res["b_clean"],
            "wr_clean": round(res["a_clean"] / clean, 4) if clean else None,
            "a_forfeits": res["a_error_forfeits"],
            "b_forfeits": res["b_error_forfeits"],
            "errors": [e for e, _ in Counter(res["errors"]).most_common(3)],
            "median_turns": (sorted(res["turns"])[len(res["turns"]) // 2]
                             if res["turns"] else None),
            "seconds": round(time.time() - t0, 1)}
    print(f"  arm={arm} [{entry_idx}] {entry['name'][:28]:30s} "
          f"({cell['pilot']:14s}) clean {res['a_clean']}/{clean} "
          f"= {cell['wr_clean']}   all {res['a']}/{decided} "
          f"= {cell['wr_all']}   forfeits ours {res['a_error_forfeits']} "
          f"theirs {res['b_error_forfeits']}   [{cell['seconds']}s]",
          flush=True)
    return cell


def _weighted(cells: list[dict], key_wins: str, key_n: str):
    """Play-share-weighted rate and its SE over one arm's cells."""
    rate = se2 = 0.0
    for c in cells:
        n = c[key_n]
        if not n:
            continue
        p = c[key_wins] / n
        rate += c["weight"] * p
        se2 += (c["weight"] ** 2) * p * (1 - p) / n
    return rate, math.sqrt(se2)


def summarize(cells: list[dict]) -> dict:
    arms = {a: sorted([c for c in cells if c["arm"] == a],
                      key=lambda c: c["entry"]) for a in ("0", "1")}
    out = {"per_matchup": [], "regressions": []}
    for c0, c1 in zip(arms["0"], arms["1"]):
        row = {"entry": c0["entry"], "name": c0["name"],
               "pilot": c0["pilot"], "weight": c0["weight"]}
        for tag, wk, nk in (("clean", "a_clean", "clean"),
                            ("all", "a_wins", "decided")):
            p0, p1 = c0[wk] / c0[nk], c1[wk] / c1[nk]
            se = math.sqrt(p0 * (1 - p0) / c0[nk] + p1 * (1 - p1) / c1[nk])
            row[tag] = {"off": round(p0, 4), "on": round(p1, 4),
                        "delta": round(p1 - p0, 4), "se": round(se, 4)}
        out["per_matchup"].append(row)
        d = row["clean"]
        if d["delta"] < -2 * d["se"]:
            out["regressions"].append(row["entry"])
    for tag, wk, nk in (("clean", "a_clean", "clean"),
                        ("all", "a_wins", "decided")):
        r0, s0 = _weighted(arms["0"], wk, nk)
        r1, s1 = _weighted(arms["1"], wk, nk)
        sed = math.sqrt(s0 ** 2 + s1 ** 2)
        out[f"weighted_{tag}"] = {
            "off": round(r0, 4), "on": round(r1, 4),
            "delta": round(r1 - r0, 4), "se_delta": round(sed, 4),
            "delta_over_se": round((r1 - r0) / sed, 2) if sed else None}
    w = out["weighted_clean"]
    out["ship"] = (w["delta"] >= 1.5 * w["se_delta"]
                   and not out["regressions"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="agent/main.py")
    ap.add_argument("--deck-a", default="agent/deck.csv")
    ap.add_argument("--priors", default="agent/deck_priors.json")
    ap.add_argument("--games", type=int, default=500)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=300000)
    ap.add_argument("--out", default="data/analysis/posture_panel_gate.json")
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "engine"))
    import ptcg.creation  # noqa: F401
    from ptcg.creation.specialist_panel import (build_specialist_panel,
                                                panel_report)

    panel = build_specialist_panel(ROOT / args.priors)
    print(panel_report(panel), flush=True)

    cells: list[dict] = []
    blob = {"a": args.a, "deck_a": args.deck_a, "games": args.games,
            "seed": args.seed, "cells": cells}
    out_path = ROOT / args.out
    for i, entry in enumerate(panel):
        for arm in ("0", "1"):
            cells.append(_run_cell(arm, entry, i, args))
            out_path.write_text(json.dumps(blob, indent=1))
    blob["summary"] = summarize(cells)
    out_path.write_text(json.dumps(blob, indent=1))
    s = blob["summary"]
    print(json.dumps({k: s[k] for k in
                      ("weighted_clean", "weighted_all", "regressions",
                       "ship")}, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
