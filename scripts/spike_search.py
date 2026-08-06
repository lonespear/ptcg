"""Does the engine's forward search actually work with a predicted opponent?

`cg.api.search_begin` can roll the game forward from the current position, but
only if we hand it the hidden information — the opponent's deck, hand and
prizes. This checks two things before any agent is built on top of it:

  1. that our mined-decklist prediction is accepted at all, and
  2. how often the prediction is even consistent with what we have seen.

    python scripts/spike_search.py --games 10
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "engine"))
sys.path.insert(0, str(ROOT / "agent"))

from ptcg.arena import load_deck  # noqa: E402
from ptcg.opponent import DeckPredictor  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--deck", default="agent/deck.csv")
    args = ap.parse_args()

    from cg.api import search_begin, search_end, search_step, to_observation_class
    from cg.game import battle_finish, battle_select, battle_start
    from main import agent as heuristic

    my_deck = load_deck(ROOT / args.deck)
    pred = DeckPredictor.from_history()
    print(f"loaded {len(pred.candidates)} candidate decklists "
          f"({pred.total_plays} observed plays)\n")

    stats = Counter()
    conf_sum = 0.0
    conf_n = 0

    for game in range(args.games):
        random.seed(game)
        obs, _ = battle_start(list(my_deck), list(my_deck))
        steps = 0
        tried = 0
        while steps < 2000 and tried < 3:
            sel = obs.get("select")
            if sel is None or not sel.get("option"):
                break
            # Only try search once the board has something to predict from.
            cur = obs.get("current") or {}
            if cur.get("turn", 0) >= 3 and obs.get("search_begin_input"):
                guess = pred.predict(obs)
                stats["attempts"] += 1
                conf_sum += guess.confidence
                conf_n += 1
                if guess.consistent:
                    stats["had_match"] += 1
                try:
                    o = to_observation_class(obs)
                    n_prize = len(o.current.players[o.current.yourIndex].prize)
                    state = search_begin(
                        o,
                        your_deck=[3] * o.current.players[
                            o.current.yourIndex].deckCount,
                        your_prize=[3] * n_prize,
                        opponent_deck=guess.deck,
                        opponent_prize=guess.prize,
                        opponent_hand=guess.hand,
                        opponent_active=[],
                    )
                    stats["search_begin_ok"] += 1
                    # Can we step it forward?
                    s2 = state.observation.select
                    if s2 is not None and s2.option:
                        search_step(state.searchId, [0])
                        stats["search_step_ok"] += 1
                    if state.observation.current.result != -1:
                        stats["already_finished"] += 1
                    search_end()
                except Exception as e:
                    stats[f"ERR {type(e).__name__}: {str(e)[:60]}"] += 1
                tried += 1
            obs = battle_select(heuristic(obs))
            steps += 1
        battle_finish()

    print("results over", args.games, "games:")
    for k, v in stats.most_common():
        print(f"  {k:<52} {v}")
    if conf_n:
        print(f"\nmean prediction confidence: {conf_sum / conf_n:.2f}")


if __name__ == "__main__":
    main()
