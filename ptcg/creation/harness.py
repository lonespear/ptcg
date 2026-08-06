"""Gauntlet harness: agent-vs-agent matches on the cabt engine (decisions.md D5).

An agent is any callable(obs_dict) -> list[int] over the engine's legal
options; the harness owns deck submission and seat assignment. Seats are
swapped halfway through a match so first-player advantage cancels out.

Usage: .venv/bin/python -m ptcg.harness [n_games]
"""

import random
import time
from dataclasses import dataclass, field

from cg.game import battle_start, battle_select, battle_finish

TURN_START = 2
RESULT = 23

RESULT_REASONS = {
    1: "prizes taken",
    2: "deck-out",
    3: "no active Pokemon",
    4: "card effect",
}


class RandomAgent:
    """Uniform choice among legal selections."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def __call__(self, obs: dict) -> list[int]:
        select = obs["select"]
        count = self.rng.randint(select["minCount"], select["maxCount"])
        return self.rng.sample(range(len(select["option"])), count)


@dataclass
class GameResult:
    winner: int | None  # agent index (0/1), 2 for draw, None if capped
    reason: str | None
    turns: int
    selects: int


@dataclass
class MatchResult:
    games: list[GameResult] = field(default_factory=list)
    elapsed: float = 0.0

    @property
    def wins(self) -> tuple[int, int, int]:
        w0 = sum(1 for g in self.games if g.winner == 0)
        w1 = sum(1 for g in self.games if g.winner == 1)
        draws = sum(1 for g in self.games if g.winner == 2)
        return w0, w1, draws

    def win_rate(self, agent_index: int = 0) -> float:
        decided = [g for g in self.games if g.winner in (0, 1)]
        if not decided:
            return 0.0
        return sum(1 for g in decided if g.winner == agent_index) / len(decided)

    def __str__(self) -> str:
        w0, w1, draws = self.wins
        capped = sum(1 for g in self.games if g.winner is None)
        turns = sorted(g.turns for g in self.games)
        reasons = {}
        for g in self.games:
            if g.reason:
                reasons[g.reason] = reasons.get(g.reason, 0) + 1
        return (
            f"{len(self.games)} games in {self.elapsed:.1f}s: "
            f"A0={w0} A1={w1} draws={draws} capped={capped} | "
            f"A0 win rate {self.win_rate(0):.1%} | "
            f"median turns {turns[len(turns) // 2]} | {reasons}"
        )


def play_game(agent_p0, agent_p1, deck_p0: list[int], deck_p1: list[int],
              max_selects: int = 20000) -> GameResult:
    """One game; agents/decks are given in seat order (P0 acts on index 0)."""
    obs, start = battle_start(deck_p0, deck_p1)
    if obs is None:
        raise RuntimeError(
            f"battle_start failed: errorPlayer={start.errorPlayer} "
            f"errorType={start.errorType}"
        )
    agents = (agent_p0, agent_p1)
    for agent, deck in ((agent_p0, deck_p0), (agent_p1, deck_p1)):
        if hasattr(agent, "bind_deck"):
            agent.bind_deck(deck)  # deck-aware pilots get their seat's list
    turns = selects = 0
    winner = reason = None
    try:
        while selects < max_selects:
            for log in obs["logs"]:
                if log["type"] == TURN_START:
                    turns += 1
                elif log["type"] == RESULT:
                    winner = log["result"]
                    reason = RESULT_REASONS.get(log["reason"], str(log["reason"]))
            if winner is not None:
                break
            selector = obs["current"]["yourIndex"]
            obs = battle_select(agents[selector](obs))
            selects += 1
        return GameResult(winner, reason, turns, selects)
    finally:
        battle_finish()


def play_match(agent0, agent1, deck0: list[int], deck1: list[int],
               n_games: int = 100, swap_seats: bool = True) -> MatchResult:
    """n_games between agent0(deck0) and agent1(deck1); winners are reported
    as agent indices regardless of seat."""
    result = MatchResult()
    t0 = time.time()
    for i in range(n_games):
        swapped = swap_seats and i % 2 == 1
        if swapped:
            g = play_game(agent1, agent0, deck1, deck0)
            if g.winner in (0, 1):
                g = GameResult(1 - g.winner, g.reason, g.turns, g.selects)
        else:
            g = play_game(agent0, agent1, deck0, deck1)
        result.games.append(g)
    result.elapsed = time.time() - t0
    return result


def read_deck(path: str) -> list[int]:
    with open(path) as f:
        return [int(line) for line in f.read().split("\n")[:60]]


if __name__ == "__main__":
    import sys

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    deck = read_deck("reference_deck.csv")
    match = play_match(RandomAgent(seed=1), RandomAgent(seed=2), deck, deck, n)
    print(match)
