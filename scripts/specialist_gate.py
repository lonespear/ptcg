"""Our agent on our list against a harvested specialist on its own list.

A mirror gate hands both agents the same deck, which is the one condition
under which a matchup feature has least to say: the postures are about what
THEIR archetype puts on the board, and in a mirror we are that archetype. The
specialist arm is the natural habitat — our Ogerpon list against the harvested
Grimmsnarl control agent playing the Grimmsnarl list it was written for, which
is the matchup the deny analysis fitted and the screening probe in POSTURES.md
measured.

The specialist is loaded through `ptcg.creation.pilots.ExternalPilot`, not
exec'd like `ab_gate.py` does: it is a package rather than a single file, and
it needs its own decklist injected into its module globals every call. Loaded
the other way it returns an empty selection on more than half the games and
forfeits them, which reads on the scoreboard as a 0.89 win rate for whoever
it is playing.

Seats swap every other game and a game's seed is its index, exactly as
`ab_gate.py` does it, so the two harnesses report comparable numbers.

    python scripts/specialist_gate.py --a agent/main.py \
        --specialist external/grimmsnarl_agent.py --games 300 --workers 8
"""

from __future__ import annotations

import argparse
import importlib.util
import json
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


def _init(root: str, a_path: str, spec_path: str, deck_a: str,
          deck_spec: str):
    sys.path.insert(0, root)
    sys.path.insert(0, str(Path(root) / "engine"))
    import ptcg.creation  # noqa: F401  — engine bootstrap
    from ptcg.arena import load_deck
    from ptcg.creation.pilots import ExternalPilot

    mod = _load_main(a_path, "spec_gate_a")
    _STATE["a_mod"] = mod
    _STATE["a"] = mod.agent
    _STATE["deck_a"] = load_deck(Path(deck_a))
    if deck_spec.endswith(".json"):
        _STATE["deck_b"] = [int(c) for c in
                            json.loads(Path(deck_spec).read_text())]
    else:
        _STATE["deck_b"] = load_deck(Path(deck_spec))
    pilot = ExternalPilot(spec_path)
    pilot.bind_deck(_STATE["deck_b"])
    _STATE["b"] = pilot


def _open_episode(agent) -> None:
    """The deck-selection call every Kaggle episode opens with.

    `ptcg.arena.play_game` starts the battle itself and never makes it, which
    is harmless for a stateless agent and not harmless for this one: the
    harvested specialist resets its move history and its opponent-profile
    memory inside that branch, so without it one worker's games all share one
    accumulating state and its expert stack starts returning selections its
    own legality check rejects. Its last-resort fallback is
    `range(min(option_count, maxCount))`, which is empty whenever maxCount is
    0, so the failure surfaces as an empty selection and a forfeit — 174 of
    300 games before this call was added. Our own agent uses the same call to
    reset its time bank, so both seats get it.
    """
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
            # Whose fault, in our own seat numbering.
            who = 0 if r.error.startswith("agent 0") else 1
            ours = (who == 1) if flip else (who == 0)
            out["a_error_forfeits" if ours else "b_error_forfeits"] += 1
        if r.winner is None:
            out["draws"] += 1
            continue
        actual = (1 - r.winner) if flip else r.winner
        out["a" if actual == 0 else "b"] += 1
        # The headline number is over games nobody forfeited. The harvested
        # specialist's own last-resort fallback returns an empty selection
        # whenever the prompt carries maxCount 0, and a game it hands over
        # that way measures its defect rather than this matchup.
        if not r.error:
            out["a_clean" if actual == 0 else "b_clean"] += 1
        out["turns"].append(r.turns)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="agent/main.py")
    ap.add_argument("--deck-a", default="agent/deck.csv")
    ap.add_argument("--specialist", default="external/grimmsnarl_agent.py")
    ap.add_argument("--deck-specialist",
                    default="external/grimmsnarl_deck.json")
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    seeds = [args.seed + g for g in range(args.games)]
    chunks = [seeds[i::args.workers] for i in range(args.workers)]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=get_context("spawn"),
                             initializer=_init,
                             initargs=(str(ROOT), str(ROOT / args.a),
                                       str(ROOT / args.specialist),
                                       str(ROOT / args.deck_a),
                                       str(ROOT / args.deck_specialist))
                             ) as pool:
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
    wr = res["a"] / decided if decided else float("nan")
    clean = res["a_clean"] + res["b_clean"]
    wr_clean = res["a_clean"] / clean if clean else float("nan")
    secs = round(time.time() - t0, 1)

    print(f"{args.a} on {args.deck_a} vs {args.specialist} on its own list")
    print(f"  A {res['a']} / B {res['b']} of {decided} decided "
          f"({res['draws']} draws) -> {wr:.4f}   [{secs}s]")
    print(f"  no forfeit either side: A {res['a_clean']} / B {res['b_clean']} "
          f"of {clean} -> {wr_clean:.4f}")
    print(f"  forfeits: ours {res['a_error_forfeits']}, "
          f"theirs {res['b_error_forfeits']}")
    for e, n in Counter(res["errors"]).most_common(5):
        print(f"    {n:4d}  {e[:110]}")
    if res["turns"]:
        print(f"  median turns {sorted(res['turns'])[len(res['turns']) // 2]}")

    blob = {"a": args.a, "deck_a": args.deck_a,
            "specialist": args.specialist,
            "deck_specialist": args.deck_specialist,
            "env": {k: os.environ[k] for k in sorted(os.environ)
                    if k.startswith("CABT_")},
            "games": args.games, "seed0": args.seed, "decided": decided,
            "a_wins": res["a"], "b_wins": res["b"], "draws": res["draws"],
            "a_win_rate": round(wr, 4),
            "clean_games": clean, "a_clean": res["a_clean"],
            "b_clean": res["b_clean"], "a_win_rate_clean": round(wr_clean, 4),
            "a_error_forfeits": res["a_error_forfeits"],
            "b_error_forfeits": res["b_error_forfeits"],
            "n_errors": len(res["errors"]),
            "errors": [e for e, _ in Counter(res["errors"]).most_common(5)],
            "seconds": secs}
    if args.out:
        Path(args.out).write_text(json.dumps(blob, indent=1))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
