"""Goldfish speed profiler: how fast does a deck start hurting?

Solitaire. The deck is piloted by the baseline GreedyPilot against a pass
agent that ends its turn as soon as it legally can and otherwise makes the
smallest legal selection, so the only clock in the game is the deck's own
setup. Two numbers come out: the turn of the first damaging attack (median
over games) and the damage dealt through turn 5 (mean over games).

A mean hides the shape of a deck's draw. The profile therefore also reports
the 10/25/50/75/90th percentiles of *cumulative* damage at each turn and the
full distribution of first-damage turns, so a fast deck that bricks a fifth
of the time is distinguishable from a merely average one.

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


QUANTILES = (10, 25, 50, 75, 90)


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile (the numpy default), pure Python."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    pos = (q / 100.0) * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return float(s[lo]) * (1.0 - frac) + float(s[hi]) * frac


def profile(deck: list[int], n_games: int = 50, seed: int = 7) -> dict:
    """Speed profile of one deck: median first-damage turn, mean damage
    by turn 5, the per-turn damage curve behind them, per-turn quantile
    bands on cumulative damage, and the first-damage-turn distribution."""
    pilot = GreedyPilot(seed=seed)
    games = [goldfish_game(pilot, deck, seat=i % 2) for i in range(n_games)]
    firsts = [g["first_damage_turn"] for g in games
              if g["first_damage_turn"] is not None]
    curve = {}
    for t in range(1, DAMAGE_HORIZON + 1):
        curve[t] = round(sum(g["damage"].get(t, 0) for g in games)
                         / len(games), 1)

    # cumulative damage per game per turn, then the bands across games
    cumulative = {t: [] for t in range(1, DAMAGE_HORIZON + 1)}
    for g in games:
        run = 0
        for t in range(1, DAMAGE_HORIZON + 1):
            run += g["damage"].get(t, 0)
            cumulative[t].append(run)
    bands = {t: {f"p{q}": round(percentile(cumulative[t], q), 1)
                 for q in QUANTILES}
             for t in range(1, DAMAGE_HORIZON + 1)}
    for t in range(1, DAMAGE_HORIZON + 1):
        bands[t]["mean"] = round(sum(cumulative[t]) / len(games), 1)

    # first-damage turn as a distribution, not just a median
    hist: dict[str, int] = {}
    for g in games:
        key = (str(g["first_damage_turn"])
               if g["first_damage_turn"] is not None else "none")
        hist[key] = hist.get(key, 0) + 1

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
        "cumulative_damage_bands": bands,
        "first_damage_turn_quantiles":
            {f"p{q}": round(percentile(firsts, q), 1) for q in QUANTILES}
            if firsts else None,
        "first_damage_turn_counts": dict(
            sorted(hist.items(), key=lambda kv: (kv[0] == "none", kv[0]))),
        "first_damage_turns": [g["first_damage_turn"] for g in games],
    }


def format_profile(prof: dict) -> str:
    """The bands as a table a judge can read without parsing JSON."""
    lines = [f"games: {prof['n_games']}   "
             f"no damage in {prof['games_with_no_damage']}",
             f"first damage turn: median {prof['median_first_damage_turn']}"
             f"  mean {prof['mean_first_damage_turn']}"]
    fq = prof.get("first_damage_turn_quantiles")
    if fq:
        lines.append("  quantiles " + "  ".join(
            f"{k} {v}" for k, v in fq.items()))
    lines.append("  distribution " + "  ".join(
        f"T{k}x{v}" if k != "none" else f"none x{v}"
        for k, v in prof["first_damage_turn_counts"].items()))
    lines.append("")
    lines.append("cumulative damage by turn (percentiles over games)")
    head = "turn " + "".join(f"{f'p{q}':>8}" for q in QUANTILES) + f"{'mean':>9}"
    lines.append(head)
    for t in range(1, DAMAGE_HORIZON + 1):
        b = prof["cumulative_damage_bands"][t]
        lines.append(f"{t:>4} " + "".join(f"{b[f'p{q}']:>8.0f}"
                                          for q in QUANTILES)
                     + f"{b['mean']:>9.1f}")
    return "\n".join(lines)


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
    args = [a for a in sys.argv[1:] if a != "--json"]
    json_only = "--json" in sys.argv[1:]
    src = args[0]
    n = int(args[1]) if len(args) > 1 else 50
    prof = profile(read_deck(src), n)
    if json_only:
        print(json.dumps(prof, indent=2))
    else:
        print(format_profile(prof))
        print()
        print(json.dumps(prof, indent=2))
