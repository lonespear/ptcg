"""D47 — fresh specialist decision corpora for thin archetype cells.

`e7_corpus.py`'s geometry (specialist on its own list vs the shipped pilot,
every specialist-seat decision with >= 2 options recorded with its full
observation), plus the game OUTCOME on every record — records are buffered
per game and flushed with `won` filled once the result is known, so the
value head gets labels the e7 corpora lack.

    python scripts/leaf_selfplay.py <cell> [games=150] [seed0=97000] [tag=a]

Output: data/leaf_train/selfplay_<cell>_<tag>.jsonl.gz (header, then one
record per decision in game order). Feed to
`python -m ptcg.leaf.build_table --corpus <path>`.
"""

from __future__ import annotations

import gzip
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "engine"))
import ptcg.creation  # noqa: F401, E402
from ptcg.arena import play_game, load_deck  # noqa: E402
from ptcg.creation.pilots import ExternalPilot  # noqa: E402

CELL = sys.argv[1]
GAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 150
SEED0 = int(sys.argv[3]) if len(sys.argv) > 3 else 97000
TAG = sys.argv[4] if len(sys.argv) > 4 else "a"
DECK_OVERRIDE = sys.argv[5] if len(sys.argv) > 5 else ""   # csv path: the
# specialist drives THIS list instead of its native one (gate-deck corpora)


def load_main(path, name):
    cwd = os.getcwd()
    os.chdir(os.path.dirname(path))
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        m = importlib.util.module_from_spec(spec)
        sys.modules[name] = m
        spec.loader.exec_module(m)
        return m
    finally:
        os.chdir(cwd)


M = load_main(str(ROOT / "agent" / "main.py"), "leaf_selfplay_opp")
deck_us = load_deck(str(ROOT / "agent" / "deck.csv"))
if DECK_OVERRIDE:
    deck_spec = [int(x) for x in
                 Path(DECK_OVERRIDE).read_text().split() if x.strip()]
else:
    deck_spec = [int(c) for c in
                 json.load(open(ROOT / "external" / f"{CELL}_deck.json"))]
assert len(deck_spec) == 60, f"deck has {len(deck_spec)} cards"

out_path = ROOT / "data" / "leaf_train" / f"selfplay_{CELL}_{TAG}.jsonl.gz"
out_path.parent.mkdir(parents=True, exist_ok=True)
out = gzip.open(out_path, "wt")
out.write(json.dumps({"header": True, "cell": CELL, "deck": deck_spec,
                      "games": GAMES, "seed0": SEED0, "tag": TAG,
                      "agent": f"external/{CELL}_agent.py"}) + "\n")

n_rec = 0
t0 = time.time()
for g in range(GAMES):
    seed = SEED0 + g
    sp = ExternalPilot(str(ROOT / "external" / f"{CELL}_agent.py"))
    sp.bind_deck(deck_spec)
    for a in (sp, M.agent):
        try:
            a({"select": None})
        except Exception:
            pass
    idx = [0]
    buffered: list[dict] = []

    def spy(obs):
        obs_js = None
        try:
            sel = obs.get("select") or {}
            opts = sel.get("option") or []
            if len(opts) >= 2:
                obs_js = json.dumps(obs)      # snapshot BEFORE the call
        except Exception:
            pass
        choice = sp(obs)
        if obs_js is not None:
            try:
                sel = obs.get("select") or {}
                cur = obs.get("current") or {}
                rec = {"seed": seed, "i": idx[0], "turn": cur.get("turn"),
                       "ctx": sel.get("context"),
                       "min": sel.get("minCount"),
                       "max": sel.get("maxCount"),
                       "n_opts": len(sel.get("option") or []),
                       "chosen": list(choice) if isinstance(choice, list)
                       else [choice]}
                buffered.append((obs_js, rec))
                idx[0] += 1
            except Exception:
                pass
        return choice

    flip = g % 2 == 1
    a0, a1 = (M.agent, spy) if flip else (spy, M.agent)
    d0, d1 = (deck_us, deck_spec) if flip else (deck_spec, deck_us)
    r = play_game(a0, a1, d0, d1, seed=seed)
    spec_seat = 1 if flip else 0
    won = -1 if r.winner is None else int(r.winner == spec_seat)
    for obs_js, rec in buffered:
        rec["won"] = won
        out.write('{"observation": ' + obs_js + ", "
                  + json.dumps(rec)[1:] + "\n")
    n_rec += idx[0]
    print(f"{CELL}/{TAG} g{g} seed={seed} turns={r.turns} "
          f"winner={r.winner} won={won} recs={idx[0]} tot={n_rec} "
          f"{time.time() - t0:.0f}s", flush=True)
out.close()
print(f"done: {n_rec} records -> {out_path}")
