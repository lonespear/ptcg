"""The A/B ship gate, on as many cores as the laptop has.

`scripts/compare_agents.py` plays the same match serially; this is that match
spread over worker processes, because a 500-game mirror is the gate every
evaluator change has to clear and the wall time of it decides how often we can
afford to run one.

Seats swap every other game exactly as `ptcg.arena.match` swaps them, and the
seed of a game is its index, so a chunk boundary cannot change a result.

SCORING (D54, 2026-08-08). `a_win_rate_clean` over `clean_games` — games
neither seat forfeited on an engine error — is the headline. The all-games
rate is kept beside it as a diagnostic. In a mirror both seats run our own
code and every mirror gate on record has zero errors, so the two numbers
coincide there; they stop coinciding the moment this harness is pointed at
an arm that can crash, and the gate should say so rather than pay the
survivor a silent bonus. Power floor and its arithmetic: see
`scripts/specialist_gate.py`, which owns the same rule.

    python scripts/ab_gate.py --a agent/main.py --b build/agents/base_ef84786/main.py \
        --deck external/grimmsnarl/deck.csv --games 600 --workers 8
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CLEAN_FLOOR = 600
CLEAN_HARD_FLOOR = 300

_STATE: dict = {}


def _se(wins: int, n: int) -> float:
    if not n:
        return float("nan")
    p = wins / n
    return math.sqrt(p * (1 - p) / n)


def _load_agent(path: str, name: str):
    cwd = os.getcwd()
    os.chdir(Path(path).parent)
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod.agent
    finally:
        os.chdir(cwd)


def _init(root: str, a_path: str, b_path: str, deck_a: str, deck_b: str):
    sys.path.insert(0, root)
    sys.path.insert(0, str(Path(root) / "engine"))
    from ptcg.arena import load_deck
    _STATE["a"] = _load_agent(a_path, "ab_agent_a")
    _STATE["b"] = _load_agent(b_path, "ab_agent_b")
    _STATE["deck_a"] = load_deck(Path(deck_a))
    _STATE["deck_b"] = load_deck(Path(deck_b))


def _play(seeds: list[int]) -> dict:
    from ptcg.arena import play_game
    out = {"a": 0, "b": 0, "draws": 0, "turns": [], "errors": [],
           "a_error_forfeits": 0, "b_error_forfeits": 0,
           "a_clean": 0, "b_clean": 0}
    for g in seeds:
        flip = g % 2 == 1
        a0, a1 = ((_STATE["b"], _STATE["a"]) if flip
                  else (_STATE["a"], _STATE["b"]))
        d0, d1 = ((_STATE["deck_b"], _STATE["deck_a"]) if flip
                  else (_STATE["deck_a"], _STATE["deck_b"]))
        r = play_game(a0, a1, d0, d1, seed=g)
        if r.error:
            out["errors"].append(r.error)
            # Whose fault, in our own seat numbering (the specialist_gate
            # attribution, so the two harnesses report the same thing).
            who = 0 if r.error.startswith("agent 0") else 1
            ours = (who == 1) if flip else (who == 0)
            out["a_error_forfeits" if ours else "b_error_forfeits"] += 1
        if r.winner is None:
            out["draws"] += 1
            continue
        actual = (1 - r.winner) if flip else r.winner
        out["a" if actual == 0 else "b"] += 1
        # D54: a forfeited game measures the forfeiter's defect, not the
        # matchup, so it does not enter the primary rate.
        if not r.error:
            out["a_clean" if actual == 0 else "b_clean"] += 1
        out["turns"].append(r.turns)
    return out


def run(a_path: Path, b_path: Path, deck_a: Path, deck_b: Path, games: int,
        workers: int, seed0: int = 0) -> dict:
    seeds = [seed0 + g for g in range(games)]
    chunks = [seeds[i::workers] for i in range(workers)]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers,
                             mp_context=get_context("spawn"),
                             initializer=_init,
                             initargs=(str(ROOT), str(a_path), str(b_path),
                                       str(deck_a), str(deck_b))) as pool:
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
    return {
        "games": games,
        # PRIMARY (D54); the all-games block below is diagnostic.
        "primary_metric": "a_win_rate_clean",
        "a_win_rate_clean": (res["a_clean"] / clean if clean
                             else float("nan")),
        "clean_games": clean, "a_clean": res["a_clean"],
        "b_clean": res["b_clean"],
        "clean_se": _se(res["a_clean"], clean),
        "a_error_forfeits": res["a_error_forfeits"],
        "b_error_forfeits": res["b_error_forfeits"],
        "forfeit_rate": ((decided - clean) / decided if decided
                         else float("nan")),
        "decided": decided, "draws": res["draws"],
        "a_wins": res["a"], "b_wins": res["b"],
        "a_win_rate": res["a"] / decided if decided else float("nan"),
        "median_turns": (sorted(res["turns"])[len(res["turns"]) // 2]
                         if res["turns"] else None),
        "n_errors": len(res["errors"]), "errors": res["errors"][:5],
        "seconds": round(time.time() - t0, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="agent/main.py")
    ap.add_argument("--b", default="build/agents/base_ef84786/main.py")
    ap.add_argument("--deck", action="append", required=True,
                    help="mirror deck; repeatable, paired with --games")
    ap.add_argument("--deck-b", default=None,
                    help="B's decklist, when the arm is not a mirror — the "
                         "specialist test, where each agent plays its own "
                         "list and the seats still swap every other game")
    ap.add_argument("--games", action="append", type=int, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clean-floor", type=int, default=CLEAN_FLOOR,
                    help="warn when games surviving forfeit exclusion fall "
                         f"below this (default {CLEAN_FLOOR})")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if len(args.deck) != len(args.games):
        sys.exit("--deck and --games must come in pairs")

    arms, pooled_a, pooled_n = [], 0, 0
    pooled_ac, pooled_nc = 0, 0
    for deck, games in zip(args.deck, args.games):
        deck_b = args.deck_b or deck
        r = run(ROOT / args.a, ROOT / args.b, ROOT / deck, ROOT / deck_b,
                games, args.workers, seed0=args.seed)
        r["deck"] = deck
        r["deck_b"] = deck_b
        arms.append(r)
        pooled_a += r["a_wins"]
        pooled_n += r["decided"]
        pooled_ac += r["a_clean"]
        pooled_nc += r["clean_games"]
        print(f"{deck}: PRIMARY clean A {r['a_clean']} / B {r['b_clean']} of "
              f"{r['clean_games']} -> {r['a_win_rate_clean']:.4f} "
              f"+- {r['clean_se']:.4f} SE   [{r['seconds']}s]")
        print(f"    diagnostic all-games: A {r['a_wins']} / B {r['b_wins']} "
              f"of {r['decided']} decided ({r['draws']} draws) -> "
              f"{r['a_win_rate']:.4f}; forfeits ours "
              f"{r['a_error_forfeits']} theirs {r['b_error_forfeits']}")
        if r["clean_games"] < CLEAN_HARD_FLOOR:
            print(f"    !! UNDERPOWERED: {r['clean_games']} clean games is "
                  f"below the {CLEAN_HARD_FLOOR} hard floor — this arm "
                  f"decides nothing. Extend it.")
        elif r["clean_games"] < args.clean_floor:
            print(f"    !! WARN: {r['clean_games']} clean games is below the "
                  f"{args.clean_floor} floor; minimum resolvable two-arm "
                  f"effect ~"
                  f"{2.8 * math.sqrt(2 * 0.25 / r['clean_games']):.1%}.")
        for e in r["errors"]:
            print(f"    - {e}")

    pooled = pooled_a / pooled_n if pooled_n else float("nan")
    pooled_c = pooled_ac / pooled_nc if pooled_nc else float("nan")
    print(f"\npooled PRIMARY (clean): {pooled_ac} / {pooled_nc} = "
          f"{pooled_c:.4f} +- {_se(pooled_ac, pooled_nc):.4f} SE")
    print(f"pooled diagnostic (all games): {pooled_a} / {pooled_n} decided "
          f"= {pooled:.4f}")
    blob = {"a": args.a, "b": args.b, "deck_b": args.deck_b,
            "env": {k: os.environ[k] for k in sorted(os.environ)
                    if k.startswith("CABT_")},
            "seed0": args.seed,
            "primary_metric": "pooled_win_rate_clean",
            "pooled_win_rate_clean": round(pooled_c, 4),
            "pooled_clean_games": pooled_nc, "pooled_a_clean": pooled_ac,
            "pooled_clean_se": round(_se(pooled_ac, pooled_nc), 4),
            "clean_floor": args.clean_floor,
            "clean_hard_floor": CLEAN_HARD_FLOOR,
            "underpowered": bool(pooled_nc < args.clean_floor),
            "arms": arms, "pooled_a_wins": pooled_a,
            "pooled_decided": pooled_n, "pooled_win_rate": round(pooled, 4)}
    if args.out:
        Path(args.out).write_text(json.dumps(blob, indent=1))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
