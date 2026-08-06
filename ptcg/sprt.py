"""Sequential Probability Ratio Test for agent comparisons.

We lost real rating today to noisy 100-game reads: a 0.33 that was really 0.276,
a "regression" that was inside the noise, and a search agent nearly discarded
because its edge was measured through the wrong opponent. Fixed-N testing is the
wrong tool — it either stops too early (no power) or wastes thousands of games
proving something already obvious.

Wald's SPRT (1945) stops exactly when the evidence is sufficient. This is the
same machinery Stockfish's fishtest uses to accept or reject engine patches.

We test the log-likelihood ratio of two hypotheses about the win rate p:

    H0: p = p0   (no improvement)
    H1: p = p1   (improvement worth adopting)

accepting H1 when the LLR crosses log((1-beta)/alpha) and H0 when it crosses
log(beta/(1-alpha)). Draws are excluded, so p is the win rate among decided
games, which is what "is A better than B" actually means.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class SPRTResult:
    decision: str          # "H1" (better), "H0" (not better), "continue"
    llr: float
    lower: float
    upper: float
    wins: int
    losses: int

    @property
    def n(self) -> int:
        return self.wins + self.losses

    @property
    def win_rate(self) -> float:
        return self.wins / self.n if self.n else float("nan")


class SPRT:
    """Sequential test of H0: p = p0 against H1: p = p1.

    Defaults test "is A's win rate 50% (no better) or 55% (worth shipping)"
    at 5% error rates in both directions.
    """

    def __init__(self, p0: float = 0.50, p1: float = 0.55,
                 alpha: float = 0.05, beta: float = 0.05):
        if not 0 < p0 < 1 or not 0 < p1 < 1:
            raise ValueError("p0 and p1 must be strictly between 0 and 1")
        if p1 <= p0:
            raise ValueError("p1 must exceed p0")
        self.p0, self.p1 = p0, p1
        self.upper = math.log((1 - beta) / alpha)
        self.lower = math.log(beta / (1 - alpha))
        self.wins = 0
        self.losses = 0
        self._llr = 0.0
        # Per-observation log-likelihood contributions.
        self._w = math.log(p1 / p0)
        self._l = math.log((1 - p1) / (1 - p0))

    def update(self, won: bool) -> SPRTResult:
        if won:
            self.wins += 1
            self._llr += self._w
        else:
            self.losses += 1
            self._llr += self._l
        return self.result()

    def result(self) -> SPRTResult:
        if self._llr >= self.upper:
            decision = "H1"
        elif self._llr <= self.lower:
            decision = "H0"
        else:
            decision = "continue"
        return SPRTResult(decision, self._llr, self.lower, self.upper,
                          self.wins, self.losses)

    def describe(self) -> str:
        r = self.result()
        return (f"n={r.n} wr={r.win_rate:.3f} llr={r.llr:+.2f} "
                f"[{self.lower:.2f}, {self.upper:.2f}] -> {r.decision}")
