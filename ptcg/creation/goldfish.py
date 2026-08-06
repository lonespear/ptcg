"""Goldfish speed profiler: how fast does a deck start hurting?

Solitaire. The deck is piloted by the baseline GreedyPilot against a pass
agent that ends its turn as soon as it legally can and otherwise makes the
smallest legal selection, so the only clock in the game is the deck's own
setup. Two numbers come out: the turn of the first damaging attack (median
over games) and the damage dealt through turn 5 (mean over games).

The harness's play_game discards logs, and speed lives in the logs, so the
battle loop is repeated here rather than widened there. Seats alternate,
so the profile averages the play and the draw.

Usage: python -m ptcg.creation.goldfish <deck.json|deck.csv> [n_games]
       python -m ptcg.creation.goldfish priors:<rank> [n_games]
"""

import json
import statistics
import sys
from pathlib import Path

from cg.game import battle_finish, battle_select, battle_start

from .pilots import GreedyPilot

HP_CHANGE, RESULT = 16, 23        # LogType
OPT_ATTACK, OPT_END = 13, 14      # OptionType
DAMAGE_HORIZON = 5


class PassAgent:
    """Ends the turn when it can; otherwise takes the fewest legal options."""

    def __call__(self, obs: dict) -> list[int]:
        sel = obs.get("select") or {}
        options = sel.get("option") or []
        for i, o in enumerate(options):
            if o.get("type") == OPT_END:
                return [i]
        lo = max(int(sel.get("minCount", 0) or 0), 0)
        return list(range(min(lo, len(options))))


def _my_turn(current: dict, me: int) -> int:
    """The goldfish player's own turn index (1 = its first turn)."""
    turn = int(current.get("turn", 0) or 0)
    return (turn + 1) // 2 if int(current.get("firstPlayer", 0)) == me \
        else turn // 2


def goldfish_game(pilot, deck: list[int], seat: int = 0,
                  max_selects: int = 20000) -> dict:
    """One solitaire game. Returns first damaging-attack turn and the
    per-turn damage the deck put on the passive opponent."""
    passer = PassAgent()
    agents = (pilot, passer) if seat == 0 else (passer, pilot)
    obs, start = battle_start(deck, deck)
    if obs is None:
        raise RuntimeError(f"battle_start failed: errorType={start.errorType}")
    if hasattr(pilot, "bind_deck"):
        pilot.bind_deck(deck)

    first_damage_turn = None
    last_attack_turn = None
    damage: dict[int, int] = {}
    turn = 0
    selects = 0
    try:
        while selects < max_selects:
            done = False
            for log in obs["logs"]:
                t = log["type"]
                if t == RESULT:
                    done = True
                elif t == HP_CHANGE and log.get("playerIndex") != seat \
                        and (log.get("value") or 0) < 0:
                    # the passive seat takes damage only from this deck
                    at = last_attack_turn if last_attack_turn else turn
                    damage[at] = damage.get(at, 0) - int(log["value"])
                    if first_damage_turn is None and last_attack_turn:
                        first_damage_turn = last_attack_turn
            if done:
                break
            actor = obs["current"]["yourIndex"]
            action = agents[actor](obs)
            if actor == seat:
                turn = _my_turn(obs["current"], seat)
                opts = obs["select"]["option"]
                if any(opts[i].get("type") == OPT_ATTACK for i in action
                       if 0 <= i < len(opts)):
                    last_attack_turn = turn
            obs = battle_select(action)
            selects += 1
        return {"first_damage_turn": first_damage_turn,
                "damage": damage,
                "damage_by_turn5": sum(v for t, v in damage.items()
                                       if t <= DAMAGE_HORIZON),
                "turns": turn}
    finally:
        battle_finish()


def profile(deck: list[int], n_games: int = 50, seed: int = 7) -> dict:
    """Speed profile of one deck: median first-damage turn, mean damage
    by turn 5, and the per-turn damage curve behind them."""
    pilot = GreedyPilot(seed=seed)
    games = [goldfish_game(pilot, deck, seat=i % 2) for i in range(n_games)]
    firsts = [g["first_damage_turn"] for g in games
              if g["first_damage_turn"] is not None]
    curve = {}
    for t in range(1, DAMAGE_HORIZON + 1):
        curve[t] = round(sum(g["damage"].get(t, 0) for g in games)
                         / len(games), 1)
    return {
        "n_games": n_games,
        "median_first_damage_turn":
            statistics.median(firsts) if firsts else None,
        "mean_damage_by_turn5":
            round(sum(g["damage_by_turn5"] for g in games) / len(games), 1),
        "mean_first_damage_turn":
            round(statistics.mean(firsts), 2) if firsts else None,
        "games_with_no_damage": n_games - len(firsts),
        "mean_damage_per_turn": curve,
    }


# ---------------------------------------------------------------------------

def read_deck(source: str) -> list[int]:
    """A deck from JSON (list of 60 ids), CSV (one id per line), or
    `priors:<rank>` for the nth-most-played field deck."""
    if source.startswith("priors:"):
        rank = int(source.split(":", 1)[1])
        root = Path(__file__).resolve().parents[2]
        priors = json.loads((root / "agent" / "deck_priors.json").read_text())
        entry = sorted(priors["decks"], key=lambda d: -d["p"])[rank]
        print(f"deck: {entry['a']} (field rank {rank})")
        return [int(c) for c, n in entry["c"].items() for _ in range(n)]
    text = Path(source).read_text()
    if source.endswith(".json"):
        return [int(c) for c in json.loads(text)]
    return [int(x) for x in text.split() if x.strip()]


if __name__ == "__main__":
    src = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    print(json.dumps(profile(read_deck(src), n), indent=2))
