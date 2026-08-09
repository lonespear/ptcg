"""D25 invariant: our seat's rollout policy is the live policy.

The agent sees the same game in two shapes — a dict from the grader, and
dataclasses from `search_step` inside its own rollouts — and it must play them
the same way. This walks real positions of both kinds through the policy and
asserts the choice is identical:

  live     positions the arena hands the agent, checked dict vs the dataclass
           form `to_observation_class` builds from it
  rollout  positions of *our own* seat that the search's rollouts hand
           `_rules_choice_for`, checked dataclass vs the dict form
           `dataclasses.asdict` builds

The opponent's seat is deliberately outside the invariant since playbook entry
4: their turn is rolled out under the counted field table
(`data/opponent_policy.json`), not under our policy, which is the whole point
of the entry. Those positions are counted separately and the run fails if none
of them was answered by the table while 2-ply is on — an invariant that passes
because the new machinery never fired would be worth nothing.

A failure prints the context, the two answers and the option types.

    python scripts/validate_invariant.py --games 3
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "engine"))

from ptcg.arena import load_deck, play_game  # noqa: E402


def load_agent_module(path: Path, name: str):
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=3)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--max-rollout-checks", type=int, default=4000,
                    help="asdict() is not cheap; cap the rollout sample")
    args = ap.parse_args()

    M = load_agent_module(ROOT / "agent" / "main.py", "invariant_main")
    from cg.api import to_observation_class

    stats = {"live": Counter(), "rollout": Counter()}
    fails: list[dict] = []

    def record(kind, ctx, a, b, options):
        stats[kind][ctx] += 1
        if a != b:
            stats[kind]["MISMATCH"] += 1
            if len(fails) < 20:
                fails.append({"kind": kind, "context": ctx, "dict": a,
                              "dataclass": b,
                              "types": [M._g(o, "type") for o in options]})

    # rollout side: every position the rollout policy is asked about
    _rules_choice_for = M._rules_choice_for
    checked = {"n": 0}

    seats = Counter()

    def checking_rules_choice(observation):
        theirs = M._opp_seat(observation)
        out = _rules_choice_for(observation)
        seats["opponent" if theirs else "ours"] += 1
        if theirs:
            return out                    # the field table's, not ours to check
        if checked["n"] < args.max_rollout_checks:
            checked["n"] += 1
            try:
                as_dict = dataclasses.asdict(observation)
                other = M._policy(as_dict)
                sel = getattr(observation, "select", None)
                ctx = getattr(sel, "context", None)
                opts = (getattr(sel, "option", None) or []) if sel else []
                record("rollout", ctx, other, out, opts)
            except Exception as exc:      # a broken check is a failed check
                fails.append({"kind": "rollout", "error": repr(exc)})
        return out

    M._rules_choice_for = checking_rules_choice

    # live side: every position the arena hands the agent
    _agent = M.agent

    def checking_agent(obs):
        out = _agent(obs)
        sel = obs.get("select")
        if sel is not None:
            try:
                dc = to_observation_class(obs)
                record("live", sel.get("context"), M._policy(obs),
                       M._policy(dc), sel.get("option") or [])
            except Exception as exc:
                fails.append({"kind": "live", "error": repr(exc)})
        return out

    deck = load_deck(ROOT / "agent" / "deck.csv")
    for g in range(args.games):
        M.agent({"select": None})
        r = play_game(checking_agent, checking_agent, deck, list(deck),
                      seed=args.seed0 + g)
        print(f"  game {g + 1}: winner {r.winner} turns {r.turns} "
              f"steps {r.steps} {r.error or ''}")

    for kind in ("live", "rollout"):
        c = stats[kind]
        n = sum(v for k, v in c.items() if k != "MISMATCH")
        by = ", ".join(f"ctx{k}:{v}" for k, v in sorted(
            ((k, v) for k, v in c.items() if k != "MISMATCH"),
            key=lambda t: -t[1]))
        print(f"{kind:8s} positions {n:6d}  mismatches {c['MISMATCH']}")
        print(f"          contexts: {by}")

    opp = M.TELEMETRY_OPP
    print(f"rollout seats: {seats['ours']} ours, {seats['opponent']} theirs")
    print(f"opponent table: {opp}")

    if fails:
        print("\nFAILURES")
        for f in fails:
            print(" ", f)
        sys.exit(1)
    if M.SEARCH_OPP_BRANCH >= 1:
        if seats["opponent"] == 0:
            sys.exit("FAIL: 2-ply is on and no opponent position was rolled "
                     "out — the table under test never ran")
        if opp["consulted"] == 0:
            sys.exit("FAIL: the opponent table answered nothing")
        if opp["policy_missing"]:
            sys.exit(f"FAIL: the table was asked for {opp['policy_missing']} "
                     f"times with nothing loaded")
    print("\nPASS: on our own seat the rollout policy == the live policy on "
          "every position checked")


if __name__ == "__main__":
    main()
