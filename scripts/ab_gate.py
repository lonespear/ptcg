"""The A/B ship gate, on as many cores as the laptop has.

`scripts/compare_agents.py` plays the same match serially; this is that match
spread over worker processes, because a 500-game mirror is the gate every
evaluator change has to clear and the wall time of it decides how often we can
afford to run one.

Seats swap every other game exactly as `ptcg.arena.match` swaps them, and the
seed of a game is its index, so a chunk boundary cannot change a result.

    python scripts/ab_gate.py --a agent/main.py --b build/agents/base_ef84786/main.py \
        --deck external/grimmsnarl/deck.csv --games 500 --workers 8
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_STATE: dict = {}


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
    out = {"a": 0, "b": 0, "draws": 0, "turns": [], "errors": []}
    for g in seeds:
        flip = g % 2 == 1
        a0, a1 = ((_STATE["b"], _STATE["a"]) if flip
                  else (_STATE["a"], _STATE["b"]))
        d0, d1 = ((_STATE["deck_b"], _STATE["deck_a"]) if flip
                  else (_STATE["deck_a"], _STATE["deck_b"]))
        r = play_game(a0, a1, d0, d1, seed=g)
        if r.error:
            out["errors"].append(r.error)
        if r.winner is None:
            out["draws"] += 1
            continue
        actual = (1 - r.winner) if flip else r.winner
        out["a" if actual == 0 else "b"] += 1
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
    res = {"a": 0, "b": 0, "draws": 0, "turns": [], "errors": []}
    for p in parts:
        for k in ("a", "b", "draws"):
            res[k] += p[k]
        res["turns"] += p["turns"]
        res["errors"] += p["errors"]
    decided = res["a"] + res["b"]
    return {
        "games": games, "decided": decided, "draws": res["draws"],
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
    ap.add_argument("--games", action="append", type=int, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if len(args.deck) != len(args.games):
        sys.exit("--deck and --games must come in pairs")

    arms, pooled_a, pooled_n = [], 0, 0
    for deck, games in zip(args.deck, args.games):
        r = run(ROOT / args.a, ROOT / args.b, ROOT / deck, ROOT / deck,
                games, args.workers, seed0=args.seed)
        r["deck"] = deck
        arms.append(r)
        pooled_a += r["a_wins"]
        pooled_n += r["decided"]
        print(f"{deck}: A {r['a_wins']} / B {r['b_wins']} of {r['decided']} "
              f"decided ({r['draws']} draws) -> {r['a_win_rate']:.3f}   "
              f"[{r['seconds']}s, {r['n_errors']} errors]")
        for e in r["errors"]:
            print(f"    - {e}")

    pooled = pooled_a / pooled_n if pooled_n else float("nan")
    print(f"\npooled: {pooled_a} / {pooled_n} decided = {pooled:.4f}")
    blob = {"a": args.a, "b": args.b, "seed0": args.seed,
            "arms": arms, "pooled_a_wins": pooled_a,
            "pooled_decided": pooled_n, "pooled_win_rate": round(pooled, 4)}
    if args.out:
        Path(args.out).write_text(json.dumps(blob, indent=1))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
