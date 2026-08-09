"""Audit the measurement harness itself. Run before trusting any result.

Three bugs invalidated most of this project's evidence, and every one of them
lived in an instrument rather than in the agent:

  * the arena scored legal empty selections as forfeits — a penalty only
    opponents could incur, worth ~25 points of phantom win rate to us;
  * the engine was never seeded, so "paired" comparisons were never paired;
  * three probes measured something other than what they claimed.

None was found by looking at win rates, because a biased instrument reports
plausible numbers. Each was found by asking the instrument a question whose
answer is known in advance. That is what this does.

    python scripts/harness_selftest.py

Checks:
  1. SYMMETRY   — the same matchup, seats swapped, must score symmetrically.
                  A one-sided scoring rule shows up here and nowhere else.
  2. FORFEITS   — neither side should be losing games to bookkeeping.
  3. DETERMINISM— the same seed must reproduce the same game.
  4. REALISM    — game length must resemble real replays, not a private world.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "engine"))

from ptcg.arena import load_deck, play_game  # noqa: E402

# Real-replay reference, from 150 mined episodes (scripts/validate_rollout.py).
REAL_MEDIAN_TURNS = 13
REAL_MEDIAN_STEPS = 158


def load_agent(path: Path, name: str):
    cwd = os.getcwd()
    os.chdir(path.parent)
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod.agent
    finally:
        os.chdir(cwd)


def run_block(agent, deck_a, deck_b, n, seed0, swap):
    """Play n games. `swap` puts deck_b in seat 0."""
    wins_a = forfeits = errors = 0
    turns, steps = [], []
    for g in range(n):
        if swap:
            r = play_game(agent, agent, deck_b, deck_a, seed=seed0 + g)
            a_won = r.winner == 1
        else:
            r = play_game(agent, agent, deck_a, deck_b, seed=seed0 + g)
            a_won = r.winner == 0
        if r.error:
            errors += 1
            if "returned" in (r.error or ""):
                forfeits += 1
        wins_a += bool(a_won)
        turns.append(r.turns)
        steps.append(r.steps)
    return {"wins_a": wins_a, "n": n, "forfeits": forfeits, "errors": errors,
            "turns": turns, "steps": steps}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="agent/main.py")
    ap.add_argument("--deck", default="agent/deck.csv")
    ap.add_argument("--deck-b", default="build/archaludon_deck.csv")
    ap.add_argument("--games", type=int, default=60)
    args = ap.parse_args()

    agent = load_agent(ROOT / args.agent, "selftest_agent")
    deck_a = load_deck(ROOT / args.deck)
    deck_b_path = ROOT / args.deck_b
    deck_b = load_deck(deck_b_path) if deck_b_path.exists() else list(deck_a)
    failures = []

    # ---- 1. symmetry ------------------------------------------------------
    print("1. SYMMETRY — same matchup, seats swapped")
    fwd = run_block(agent, deck_a, deck_b, args.games, 5000, swap=False)
    rev = run_block(agent, deck_a, deck_b, args.games, 5000, swap=True)
    wr_f = fwd["wins_a"] / fwd["n"]
    wr_r = rev["wins_a"] / rev["n"]
    gap = abs(wr_f - wr_r)
    # Two independent binomials; ~2 sigma of their difference.
    tol = 2.0 * (2 * 0.25 / args.games) ** 0.5
    print(f"   deck A in seat 0 : {wr_f:.3f}")
    print(f"   deck A in seat 1 : {wr_r:.3f}")
    print(f"   gap {gap:.3f} vs tolerance {tol:.3f}")
    if gap > tol:
        failures.append(
            f"asymmetric scoring: {gap:.3f} > {tol:.3f}. A rule that can only "
            f"penalise one seat (or a real first-player edge) is in play.")
        print("   FAIL")
    else:
        print("   pass")

    # ---- 2. forfeits ------------------------------------------------------
    print("\n2. FORFEITS — neither side should lose games to bookkeeping")
    tot_f = fwd["forfeits"] + rev["forfeits"]
    tot_e = fwd["errors"] + rev["errors"]
    tot_n = fwd["n"] + rev["n"]
    print(f"   forfeits {tot_f}/{tot_n} ({tot_f/tot_n:.1%}), "
          f"all errors {tot_e}/{tot_n} ({tot_e/tot_n:.1%})")
    if tot_f / tot_n > 0.02:
        failures.append(f"{tot_f/tot_n:.1%} of games ended on a forfeit — "
                        f"the arena is scoring legal play as a crash.")
        print("   FAIL")
    else:
        print("   pass")

    # ---- 3. determinism ---------------------------------------------------
    print("\n3. DETERMINISM — the same seed must reproduce the same game")
    a = run_block(agent, deck_a, deck_b, 8, 9100, swap=False)
    b = run_block(agent, deck_a, deck_b, 8, 9100, swap=False)
    same = a["turns"] == b["turns"] and a["steps"] == b["steps"]
    print(f"   run 1 turns {a['turns']}")
    print(f"   run 2 turns {b['turns']}")
    if not same:
        failures.append(
            "same seed produced different games. `libcg` draws from "
            "std::random_device and ignores random.seed(), so nothing here is "
            "reproducible or paired. Use Austin's tools/engine_seed preload.")
        print("   FAIL — seeding does not reach the engine")
    else:
        print("   pass")

    # ---- 4. realism -------------------------------------------------------
    print("\n4. REALISM — our games should resemble real replays")
    med_t = statistics.median(fwd["turns"] + rev["turns"])
    med_s = statistics.median(fwd["steps"] + rev["steps"])
    rt, rs = med_t / REAL_MEDIAN_TURNS, med_s / REAL_MEDIAN_STEPS
    print(f"   median turns {med_t:.0f} vs real {REAL_MEDIAN_TURNS} "
          f"(ratio {rt:.2f})")
    print(f"   median steps {med_s:.0f} vs real {REAL_MEDIAN_STEPS} "
          f"(ratio {rs:.2f})")
    if not (0.5 <= rt <= 1.8):
        failures.append(f"games run {rt:.2f}x real length — the simulated "
                        f"world is not the one we are scored in.")
        print("   FAIL")
    else:
        print("   pass (note: a mirror or a lopsided matchup shifts this)")

    print("\n" + "=" * 66)
    if failures:
        print(f"{len(failures)} PROBLEM(S) — results from this harness are not "
              f"trustworthy:\n")
        for f in failures:
            print(f"  * {f}")
        sys.exit(1)
    print("harness clean on all four checks")


if __name__ == "__main__":
    main()
