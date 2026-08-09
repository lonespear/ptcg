"""What the C2 trajectory feature costs per decision, measured both ways.

A feature that fits and plays is still not shippable if it eats the time bank.
This plays the same games with the feature's weight on and with it set to
zero — same agent, same decks, same seeds, so the only difference is whether
`_margin` computes the projection — and reports the wall time of an `agent()`
call in each condition, split by whether the search fired on that call.

    python scripts/timing_check.py --games 20
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "engine"))

from ptcg.arena import load_deck, play_game  # noqa: E402

MAIN = ROOT / "agent" / "main.py"


def load_agent(path: Path = MAIN, name: str = "timed_agent"):
    cwd = os.getcwd()
    os.chdir(path.parent)
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        os.chdir(cwd)


def measure(mod, deck, games: int, seed0: int, weight: float | None) -> dict:
    if weight is not None:
        mod.set_weights({"threat_traj": weight})
    calls: list[tuple[float, bool]] = []

    def timed(obs):
        started = time.perf_counter()
        out = mod.agent(obs)
        sel = (obs or {}).get("select") or {}
        options = sel.get("option") or []
        main_menu = (len(options) > 1
                     and any(o.get("type") in mod.MAIN_PRIORITY
                             for o in options))
        calls.append((time.perf_counter() - started, main_menu))
        return out

    for g in range(games):
        play_game(timed, mod.agent, list(deck), list(deck), seed=seed0 + g)

    allw = [c for c, _ in calls]
    rules = [c for c, m in calls if not m]
    main = [c for c, m in calls if m]

    def stat(xs):
        if not xs:
            return {"n": 0}
        s = sorted(xs)
        return {"n": len(s), "mean_ms": round(1000 * sum(s) / len(s), 3),
                "p95_ms": round(1000 * s[int(0.95 * (len(s) - 1))], 3),
                "max_ms": round(1000 * s[-1], 3)}

    return {"weight": weight, "all": stat(allw), "rules_path": stat(rules),
            "main_menu": stat(main),
            "trajectory_telemetry": dict(mod.TELEMETRY_TRAJ)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--deck", default="external/grimmsnarl/deck.csv")
    ap.add_argument("--seed", type=int, default=7000)
    ap.add_argument("--against", default=None,
                    help="a second agent file to time instead of toggling "
                         "the C2 weight — each side at its own shipped "
                         "vector, which is what a ship gate needs")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    mod = load_agent()
    deck = load_deck(ROOT / args.deck)
    if args.against:
        on = measure(mod, deck, args.games, args.seed, None)
        base = load_agent(ROOT / args.against, "timed_baseline")
        off = measure(base, deck, args.games, args.seed, None)
        labels = (f"this build", f"{args.against}")
    else:
        on = measure(mod, deck, args.games, args.seed, 1.42)
        for k in mod.TELEMETRY_TRAJ:
            mod.TELEMETRY_TRAJ[k] = 0
        off = measure(mod, deck, args.games, args.seed, 0.0)
        labels = ("feature on", "feature off")

    print(f"{args.games} games on {args.deck}, both sides the same agent\n")
    print(f"{'condition':12s} {'calls':>7s} {'mean ms':>9s} {'p95 ms':>9s} "
          f"{'max ms':>9s}")
    for label, res in zip(labels, (on, off)):
        for path in ("all", "rules_path", "main_menu"):
            s = res[path]
            if not s.get("n"):
                continue
            print(f"{label + ' ' + path:34s} {s['n']:7d} {s['mean_ms']:9.3f} "
                  f"{s['p95_ms']:9.3f} {s['max_ms']:9.3f}")
    for path in ("rules_path", "main_menu"):
        if on[path].get("n") and off[path].get("n"):
            d = on[path]["mean_ms"] - off[path]["mean_ms"]
            print(f"\n{path}: {d:+.3f} ms a decision for {labels[0]}")
    print(f"telemetry (on): {on['trajectory_telemetry']}")
    if args.out:
        Path(args.out).write_text(json.dumps({"on": on, "off": off}, indent=1))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
