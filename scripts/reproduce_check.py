"""Does a frozen agent file reproduce its own win rate? (E18)

D66 ran one frozen v7 file three times on the grimmsnarl cell at 600 clean
games with identical seeds, env and `--workers 3`, and got 0.7800 / 0.7500 /
0.7367. This script is that measurement, made repeatable and made to cover the
conditions the original could not separate.

An ARM is a way of running the same cell:

    unpinned_idle   as the project has always run it
    unpinned_load   the same, with CPU burners running
    pinned_idle     under scripts/run_pinned.sh, quiet machine
    pinned_load     under scripts/run_pinned.sh, with CPU burners running
    pinned_w4       pinned, at a different --workers, to price D60

Each arm runs `--reps` times and the report is the SPREAD: max minus min win
rate over the repetitions. A reproducible arm reads 0.0000 and every rep is
the same number; anything else is measurement noise that no experiment on this
project can tell apart from an effect.

Burners are this process's own children and are killed by PID. Nothing here
signals a process it did not start.

    python scripts/reproduce_check.py --games 600 --reps 3 --workers 2
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

ARMS = ("unpinned_idle", "unpinned_load", "pinned_idle", "pinned_load",
        "pinned_w4")


def _burners(n: int) -> list[subprocess.Popen]:
    """CPU hogs, held by handle so only these get killed."""
    return [subprocess.Popen([PY, "-c", "while True: pass"],
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
            for _ in range(n)]


def _stop(procs: list[subprocess.Popen]) -> None:
    for p in procs:
        try:
            p.send_signal(signal.SIGTERM)
        except Exception:
            pass
    for p in procs:
        try:
            p.wait(timeout=10)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass


def _run(arm: str, args, rep: int, tmp: Path) -> dict:
    out = tmp / f"{arm}_{rep}.json"
    cmd = [PY, str(ROOT / "scripts" / "specialist_gate.py"),
           "--a", args.a, "--deck-a", args.deck_a,
           "--specialist", args.specialist,
           "--deck-specialist", args.deck_specialist,
           "--games", str(args.games), "--seed", str(args.seed),
           "--workers", str(4 if arm == "pinned_w4" else args.workers),
           "--out", str(out)]
    if not arm.startswith("unpinned"):
        cmd = ["sh", str(ROOT / "scripts" / "run_pinned.sh")] + cmd
    env = dict(os.environ)
    env["CABT_BUDGET_MODE"] = args.budget_mode
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env,
                          capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout[-2000:])
        print(proc.stderr[-2000:], file=sys.stderr)
        raise SystemExit(f"{arm} rep {rep} failed")
    blob = json.loads(out.read_text())
    blob["wall_seconds"] = round(time.time() - t0, 1)
    return blob


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="agent/main.py")
    ap.add_argument("--deck-a", default="agent/deck.csv")
    ap.add_argument("--specialist", default="external/grimmsnarl_agent.py")
    ap.add_argument("--deck-specialist",
                    default="external/grimmsnarl_deck.json")
    ap.add_argument("--games", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--burners", type=int, default=12)
    ap.add_argument("--budget-mode", default="count",
                    choices=("count", "clock"))
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        if a not in ARMS:
            raise SystemExit(f"unknown arm {a}; pick from {ARMS}")

    tmp = Path(tempfile.mkdtemp(prefix="reprocheck_"))
    report: dict = {"a": args.a, "specialist": args.specialist,
                    "games": args.games, "seed0": args.seed,
                    "reps": args.reps, "workers": args.workers,
                    "budget_mode": args.budget_mode, "arms": {}}
    print(f"{args.a} vs {args.specialist}: {args.games} games x {args.reps} "
          f"reps per arm, budget mode {args.budget_mode}\n")

    for arm in arms:
        hogs = _burners(args.burners) if arm.endswith("_load") else []
        if hogs:
            time.sleep(2)
        try:
            runs = [_run(arm, args, r, tmp) for r in range(args.reps)]
        finally:
            _stop(hogs)
        rates = [r["a_win_rate_clean"] for r in runs]
        cleans = [r["clean_games"] for r in runs]
        spread = max(rates) - min(rates)
        report["arms"][arm] = {
            "win_rates": rates, "clean_games": cleans,
            "spread": round(spread, 4),
            "identical": len(set(rates)) == 1 and len(set(cleans)) == 1,
            "seconds": [r["wall_seconds"] for r in runs]}
        flag = "REPRODUCES" if report["arms"][arm]["identical"] else "VARIES"
        print(f"  {arm:15s} {'  '.join(f'{x:.4f}' for x in rates)}   "
              f"spread {spread*100:+.2f} pt   clean {cleans}   {flag}")

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
