"""Does one game replay identically? A decision-level determinism probe.

D66 measured the same frozen file scoring 0.7800 / 0.7500 / 0.7367 on the
grimmsnarl cell at 600 clean games with identical seeds, env and worker count.
A win-rate spread is a slow and indirect way to see that: 600 games is an hour
and the answer comes back as a number that could also be sampling.

This probe asks the question directly and per decision. It plays a handful of
seeds and rolls a SHA-256 over every option list our seat was shown and every
selection it returned, so two runs that played the same games produce the same
64 hex characters and two runs that diverged on one decision out of thousands
produce different ones. A digest mismatch is proof of nondeterminism; a match
over a few dozen games is strong evidence against it.

`--budget inf` monkey-patches `_decision_budget` to a value the search can
never spend, which removes the wall-clock deadline without touching the agent
file. Comparing `normal` against `inf` on an idle machine says whether the
clock meter is binding at all; comparing either against itself under CPU load
says whether it is the source of the divergence.

    python scripts/determinism_probe.py --games 12 --budget normal
    python scripts/determinism_probe.py --games 12 --budget inf
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


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


def _digest_wrap(agent, h, tag: bytes):
    """Fold every prompt and every answer into the running digest."""
    def wrapped(obs):
        sel = (obs or {}).get("select")
        if sel is not None:
            opts = sel.get("option") or []
            h.update(tag)
            h.update(str(len(opts)).encode())
            h.update(str([o.get("type") for o in opts]).encode())
        out = agent(obs)
        h.update(b"->")
        h.update(str(out).encode())
        return out
    return wrapped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="agent/main.py")
    ap.add_argument("--deck-a", default="agent/deck.csv")
    ap.add_argument("--specialist", default="external/grimmsnarl_agent.py")
    ap.add_argument("--deck-specialist",
                    default="external/grimmsnarl_deck.json")
    ap.add_argument("--games", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", choices=("normal", "inf"), default="normal")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "engine"))
    import ptcg.creation  # noqa: F401
    from ptcg.arena import load_deck, play_game
    from ptcg.creation.pilots import ExternalPilot
    from ptcg import engine_seed

    mod = _load_main(str(ROOT / args.a), "probe_a")
    deck_a = load_deck(ROOT / args.deck_a)
    spec_deck = str(ROOT / args.deck_specialist)
    if spec_deck.endswith(".json"):
        deck_b = [int(c) for c in json.loads(Path(spec_deck).read_text())]
    else:
        deck_b = load_deck(spec_deck)
    pilot = ExternalPilot(str(ROOT / args.specialist))
    pilot.bind_deck(deck_b)

    if args.budget == "inf":
        mod._decision_budget = lambda obs=None: 1e9

    h = hashlib.sha256()
    per_game = []
    wins = 0
    t0 = time.time()
    for g in range(args.seed, args.seed + args.games):
        for a in (mod.agent, pilot):
            try:
                a({"select": None})
            except Exception:
                pass
        gh = hashlib.sha256()
        flip = g % 2 == 1
        wa = _digest_wrap(mod.agent, gh, b"A")
        wb = _digest_wrap(pilot, gh, b"B")
        a0, a1 = (wb, wa) if flip else (wa, wb)
        d0, d1 = ((deck_b, deck_a) if flip else (deck_a, deck_b))
        r = play_game(a0, a1, d0, d1, seed=g)
        actual = None if r.winner is None else (
            (1 - r.winner) if flip else r.winner)
        if actual == 0:
            wins += 1
        gd = gh.hexdigest()[:16]
        h.update(gd.encode())
        per_game.append({"seed": g, "digest": gd, "winner": actual,
                         "turns": r.turns, "error": r.error})

    secs = round(time.time() - t0, 1)
    blob = {"label": args.label or args.budget, "a": args.a,
            "budget_mode": args.budget,
            "engine_pinned": engine_seed.available(),
            "env": {k: os.environ[k] for k in sorted(os.environ)
                    if k.startswith("CABT_")},
            "games": args.games, "seed0": args.seed,
            "overall_digest": h.hexdigest(),
            "a_wins": wins, "seconds": secs, "per_game": per_game}
    print(f"[{blob['label']}] {args.games} games  digest "
          f"{blob['overall_digest'][:24]}  A wins {wins}  [{secs}s]"
          f"{'' if blob['engine_pinned'] else '  UNPINNED — not replayable'}")
    if args.out:
        Path(args.out).write_text(json.dumps(blob, indent=1))


if __name__ == "__main__":
    main()
