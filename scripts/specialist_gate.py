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

SCORING (D54, 2026-08-08). The headline number is `a_win_rate_clean` over
`clean_games` — games neither seat forfeited on an engine error. The
all-games rate is printed and stored beside it as a DIAGNOSTIC only, because
it is not a measurement of this matchup: a forfeit hands the game to whoever
did not forfeit, our seat has never forfeited once in 38,000 recorded gate
games, and the specialists forfeit at rates from 0% (lucario,
codex_alakazam) to 58% (archaludon). All-games scoring therefore pays us a
different bonus in every cell. Across 39 real ladder episodes there were
zero forfeits, so the bonus does not exist where it counts.

POWER (D54 + D51). Exclusions cost sample, and the cost is the whole point:
600 games in the archaludon cell is ~250 clean games. `--clean-floor`
(default 600, matching D51's 600-game rule read on the CLEAN sample) makes
the tool say so. At p=0.5 the SE of a two-arm difference is 0.0408 at
n=300 and 0.0289 at n=600, so the minimum effect resolvable at 80% power
(two-sided 0.05) is 11.4 pt and 8.1 pt respectively. We work in the 3-12 pt
range, so 600 clean is the bar for a ship or kill decision and 300 clean is
the bar below which the cell decides nothing at all. Extend the run; never
score the contaminated number instead.

    python scripts/specialist_gate.py --a agent/main.py \
        --specialist external/grimmsnarl_agent.py --games 600 --workers 8
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

# D51's 600-game rule, read on the clean sample (see module docstring).
CLEAN_FLOOR = 600
# Below this a cell resolves nothing we have ever measured; it is not a
# weak reading, it is not a reading.
CLEAN_HARD_FLOOR = 300

_STATE: dict = {}


def _se(wins: int, n: int) -> float:
    """Standard error of a win rate, the honest width on `n` clean games."""
    if not n:
        return float("nan")
    p = wins / n
    return math.sqrt(p * (1 - p) / n)


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
    from ptcg import engine_seed

    # Whether this worker's games are reproducible, recorded per worker rather
    # than read off the parent: the preload is inherited through the spawn and
    # a run that lost it would otherwise report a number nothing can replay.
    _STATE["pinned"] = engine_seed.available()

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
           "a_clean": 0, "b_clean": 0,
           "pinned": bool(_STATE.get("pinned"))}
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
    ap.add_argument("--clean-floor", type=int, default=CLEAN_FLOOR,
                    help="warn when games surviving forfeit exclusion fall "
                         f"below this (default {CLEAN_FLOOR}, D51's rule "
                         "read on the clean sample)")
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
    pinned = all(p.get("pinned") for p in parts)
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
    se_clean = _se(res["a_clean"], clean)
    forfeit_rate = ((decided - clean) / decided) if decided else float("nan")
    secs = round(time.time() - t0, 1)

    print(f"{args.a} on {args.deck_a} vs {args.specialist} on its own list")
    print(f"  PRIMARY (D54, forfeits excluded): A {res['a_clean']} / "
          f"B {res['b_clean']} of {clean} clean -> {wr_clean:.4f} "
          f"+- {se_clean:.4f} SE   [{secs}s]")
    print(f"  diagnostic only, all games: A {res['a']} / B {res['b']} of "
          f"{decided} decided ({res['draws']} draws) -> {wr:.4f}")
    print(f"  engine RNG: {'pinned, this run replays' if pinned else 'UNPINNED — these games cannot be replayed (D66); run under scripts/run_pinned.sh'}")
    print(f"  forfeits: theirs {res['b_error_forfeits']}, "
          f"ours {res['a_error_forfeits']} "
          f"({forfeit_rate:.1%} of decided games excluded)")
    warn = None
    if clean < CLEAN_HARD_FLOOR:
        warn = (f"UNDERPOWERED: {clean} clean games is below the "
                f"{CLEAN_HARD_FLOOR} hard floor. A two-arm difference here "
                f"has SE ~{math.sqrt(2 * 0.25 / max(clean, 1)):.4f}; nothing "
                f"under ~{2.8 * math.sqrt(2 * 0.25 / max(clean, 1)):.1%} is "
                f"resolvable. This cell decides nothing — extend it.")
    elif clean < args.clean_floor:
        warn = (f"WARN: {clean} clean games is below the {args.clean_floor} "
                f"floor (D51's 600-game rule read on the clean sample). "
                f"Minimum resolvable two-arm effect ~"
                f"{2.8 * math.sqrt(2 * 0.25 / clean):.1%} at 80% power. "
                f"Extend with more games; do not score the all-games number "
                f"instead.")
    if warn:
        print(f"  !! {warn}")
    for e, n in Counter(res["errors"]).most_common(5):
        print(f"    {n:4d}  {e[:110]}")
    if res["turns"]:
        print(f"  median turns {sorted(res['turns'])[len(res['turns']) // 2]}")

    blob = {"a": args.a, "deck_a": args.deck_a,
            "specialist": args.specialist,
            "deck_specialist": args.deck_specialist,
            "env": {k: os.environ[k] for k in sorted(os.environ)
                    if k.startswith("CABT_")},
            "games": args.games, "seed0": args.seed,
            # PRIMARY metric (D54). Everything below `a_win_rate` is a
            # diagnostic. Field names are unchanged from the pre-D54 tool so
            # every result file already on disk stays readable; what changed
            # is which of them is the answer.
            "engine_pinned": pinned,
            "primary_metric": "a_win_rate_clean",
            "a_win_rate_clean": round(wr_clean, 4),
            "clean_games": clean, "a_clean": res["a_clean"],
            "b_clean": res["b_clean"],
            "clean_se": round(se_clean, 4),
            "clean_floor": args.clean_floor,
            "clean_hard_floor": CLEAN_HARD_FLOOR,
            "underpowered": bool(clean < args.clean_floor),
            "power_warning": warn,
            "forfeit_rate": round(forfeit_rate, 4),
            "a_error_forfeits": res["a_error_forfeits"],
            "b_error_forfeits": res["b_error_forfeits"],
            # Diagnostic: inflated by opponent forfeits, never a ship
            # criterion (D54).
            "a_win_rate": round(wr, 4),
            "decided": decided,
            "a_wins": res["a"], "b_wins": res["b"], "draws": res["draws"],
            "n_errors": len(res["errors"]),
            "errors": [e for e, _ in Counter(res["errors"]).most_common(5)],
            "seconds": secs}
    if args.out:
        Path(args.out).write_text(json.dumps(blob, indent=1))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
