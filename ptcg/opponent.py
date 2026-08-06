"""Guess the opponent's decklist from what we have seen them play.

The engine's search (`cg.api.search_begin`) can simulate forward from the
current position, but it needs the hidden information filled in: the opponent's
deck, hand and prizes. Guessing 60 cards blind would be hopeless — except the
mined replays say the whole metagame is only ~120 distinct decklists, and a
third of it is one deck. So this is a small ranking problem, not a search.

    pred = DeckPredictor.from_history()
    guess = pred.predict(obs)     # -> PredictedHidden(deck, hand, prize)
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_HISTORY = Path(__file__).resolve().parents[1] / "data" / "history_decklists.csv"

# Zones where a card we can see is definitely not in the hidden pool any more.
_VISIBLE_ZONES = ("active", "bench", "discard")


@dataclass
class Candidate:
    counts: Counter
    plays: int
    wins: int
    archetype: str

    @property
    def win_rate(self) -> float:
        return self.wins / self.plays if self.plays else 0.0


@dataclass
class PredictedHidden:
    deck: list[int] = field(default_factory=list)
    hand: list[int] = field(default_factory=list)
    prize: list[int] = field(default_factory=list)
    archetype: str = "?"
    confidence: float = 0.0
    consistent: int = 0


def _signature_to_counts(sig: str) -> Counter:
    out: Counter = Counter()
    for part in sig.split(","):
        cid, n = part.split(":")
        out[int(cid)] += int(n)
    return out


class DeckPredictor:
    """Ranks known decklists by how well they explain the cards we have seen."""

    def __init__(self, candidates: list[Candidate]):
        self.candidates = candidates
        self.total_plays = sum(c.plays for c in candidates) or 1
        # Fallback when nothing matches: the pooled card frequency across the
        # whole metagame, which is still far better than uniform.
        pool: Counter = Counter()
        for c in candidates:
            for cid, n in c.counts.items():
                pool[cid] += n * c.plays
        self.pool = pool

    @classmethod
    def from_history(cls, path: Path | str = DEFAULT_HISTORY,
                     min_plays: int = 3) -> "DeckPredictor":
        import pandas as pd
        p = Path(path)
        if not p.exists():
            return cls([])
        df = pd.read_csv(p)
        g = (df.groupby("signature", as_index=False)
             .agg(plays=("decks", "sum"), wins=("wins", "sum"),
                  archetype=("archetype", "first")))
        g = g[g["plays"] >= min_plays]
        return cls([
            Candidate(_signature_to_counts(r.signature), int(r.plays),
                      int(r.wins), str(r.archetype))
            for r in g.itertuples()
        ])

    # ---- observation -----------------------------------------------------
    @staticmethod
    def observed_cards(obs: dict) -> Counter:
        """Opponent card ids we can currently see, with multiplicity."""
        seen: Counter = Counter()
        cur = obs.get("current") or {}
        me = cur.get("yourIndex", 0)
        players = cur.get("players") or []
        if len(players) < 2:
            return seen
        opp = players[1 - me]
        for zone in _VISIBLE_ZONES:
            for card in opp.get(zone) or []:
                if isinstance(card, dict) and card.get("id"):
                    seen[card["id"]] += 1
                    # A Pokémon in play carries its own history with it.
                    for sub in ("energyCards", "tools", "preEvolution"):
                        for c in card.get(sub) or []:
                            if isinstance(c, dict) and c.get("id"):
                                seen[c["id"]] += 1
        stadium = cur.get("stadium") or []
        for card in stadium:
            if isinstance(card, dict) and card.get("id"):
                seen[card["id"]] += 1
        return seen

    def _consistent(self, seen: Counter) -> list[Candidate]:
        out = []
        for c in self.candidates:
            if all(c.counts.get(cid, 0) >= n for cid, n in seen.items()):
                out.append(c)
        return out

    # ---- prediction ------------------------------------------------------
    def predict(self, obs: dict) -> PredictedHidden:
        cur = obs.get("current") or {}
        me = cur.get("yourIndex", 0)
        players = cur.get("players") or []
        if len(players) < 2:
            return PredictedHidden()
        opp = players[1 - me]
        n_deck = opp.get("deckCount", 0) or 0
        n_hand = opp.get("handCount", 0) or 0
        n_prize = len(opp.get("prize") or [])

        seen = self.observed_cards(obs)
        matches = self._consistent(seen)

        if matches:
            best = max(matches, key=lambda c: c.plays)
            remaining: Counter = Counter(best.counts)
            remaining.subtract(seen)
            hidden = [cid for cid, n in remaining.items() for _ in range(max(n, 0))]
            confidence = best.plays / max(sum(m.plays for m in matches), 1)
            archetype, n_ok = best.archetype, len(matches)
        else:
            hidden = []
            confidence, archetype, n_ok = 0.0, "?", 0

        need = n_deck + n_hand + n_prize
        if len(hidden) < need:
            hidden.extend(self._pad(need - len(hidden), seen))
        hidden = hidden[:need]

        # Order matters only in that each bucket must be the right size.
        return PredictedHidden(
            hand=hidden[:n_hand],
            prize=hidden[n_hand:n_hand + n_prize],
            deck=hidden[n_hand + n_prize:],
            archetype=archetype,
            confidence=confidence,
            consistent=n_ok,
        )

    def _pad(self, k: int, seen: Counter) -> list[int]:
        """Fill from the metagame-wide card frequency when the list is short."""
        if k <= 0:
            return []
        if not self.pool:
            return [3] * k          # a Basic Energy is the safest filler
        ranked = [cid for cid, _ in self.pool.most_common()]
        out: list[int] = []
        i = 0
        while len(out) < k and ranked:
            out.append(ranked[i % len(ranked)])
            i += 1
        return out
