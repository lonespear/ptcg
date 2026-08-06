"""When a game ends with no Pokemon left — is it US or the opponent?

termination_modes.py measured self-play on a MIRROR: both sides ran the same
4-Pokemon deck, so "70% of games end with someone out of Pokemon" was
guaranteed by construction and says nothing about whether *we* are fragile
against the field. This checks against real field decks and attributes the
empty board to a seat.
"""
import importlib.util
import os
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(r"C:\Users\jonathan.day\Documents\ptcg")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "engine"))

from ptcg.arena import load_deck, play_game  # noqa
from ptcg.meta import signature_to_deck  # noqa
from cg.game import battle_finish, battle_select, battle_start  # noqa


def load_agent(path, name):
    c = os.getcwd()
    os.chdir(os.path.dirname(path))
    try:
        s = importlib.util.spec_from_file_location(name, os.path.basename(path))
        m = importlib.util.module_from_spec(s)
        sys.modules[name] = m
        s.loader.exec_module(m)
        return m.agent
    finally:
        os.chdir(c)


agent = load_agent(str(ROOT / "agent" / "main.py"), "probe_ro")
ours = load_deck(ROOT / "agent" / "deck.csv")

df = pd.read_csv(ROOT / "data" / "history_decklists.csv")
g = (df.groupby("signature", as_index=False)
     .agg(plays=("decks", "sum"), archetype=("archetype", "first"))
     .nlargest(3, "plays"))

import random
for r in g.itertuples():
    opp = signature_to_deck(r.signature)
    stat = Counter()
    for i in range(24):
        random.seed(9000 + i)
        obs, _ = battle_start(list(ours), list(opp))   # we are seat 0
        steps = 0
        while steps < 4000:
            sel = obs.get("select")
            if sel is None or not sel.get("option"):
                break
            obs = battle_select(agent(obs))
            steps += 1
        cur = obs.get("current") or {}
        players = cur.get("players") or []
        res = cur.get("result")
        if len(players) == 2:
            for seat in (0, 1):
                board = (players[seat].get("active") or []) + \
                        (players[seat].get("bench") or [])
                if not [m for m in board if m]:
                    stat["WE ran out" if seat == 0 else "THEY ran out"] += 1
        stat["we won" if res == 0 else "we lost"] += 1
        battle_finish()
    tot = 24
    print(f"\nvs {r.archetype[:26]} (n={tot})")
    for k, v in stat.most_common():
        print(f"   {k:<14} {v:>3} ({v/tot:.0%})")
