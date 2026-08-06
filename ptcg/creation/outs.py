"""The outs ledger: hypergeometric draw odds for a fetch decision (D16.5).

A fetch decision is a bet on the top of the deck, and the bet has an exact
price. Given our 60-card list, the cards we have already seen, and the cards
that would solve the turn, this computes P(at least one hit) in the next d
draws for d = 1..5 and prints the arithmetic a judge can recompute by hand:

    6 outs / 42 unseen -> 56% by turn 5

Pure functions over card ids: no engine, no observation, no globals. The
pilot supplies the predicate; this file supplies the number.

Self-check: python -m ptcg.creation.outs
"""

from collections import Counter
from math import comb
from typing import Callable, Iterable

MAX_DRAWS = 5


# --- the arithmetic ---------------------------------------------------------

def p_at_least_one(outs: int, unseen: int, draws: int) -> float:
    """P(at least one of `outs` hits among `unseen` cards in `draws` draws).

    The complement is the only closed form worth writing: all `draws` cards
    come from the `unseen - outs` blanks.
    """
    if outs <= 0 or unseen <= 0 or draws <= 0:
        return 0.0
    outs = min(outs, unseen)
    draws = min(draws, unseen)
    blanks = unseen - outs
    if blanks < draws:
        return 1.0
    return 1.0 - comb(blanks, draws) / comb(unseen, draws)


def outs_curve(outs: int, unseen: int, max_draws: int = MAX_DRAWS
               ) -> dict[int, float]:
    """P(hit) for every draw count 1..max_draws."""
    return {d: p_at_least_one(outs, unseen, d)
            for d in range(1, max_draws + 1)}


# --- counting outs against a real board -------------------------------------

def unseen_counts(decklist: Iterable[int], seen: Iterable[int]) -> Counter:
    """Cards still hidden (deck + prizes), as a multiset of card ids.

    `seen` is a multiset of our own cards already revealed anywhere — hand,
    discard, in play, a revealed prize. Copies seen beyond the list's count
    are clamped rather than raising: an engine log can double-report a card
    that moved zones, and a fetch decision should not die over it.
    """
    remaining = Counter(decklist)
    for cid, n in Counter(seen).items():
        if cid in remaining:
            remaining[cid] = max(0, remaining[cid] - n)
    return +remaining          # drop zero and negative entries


def count_outs(decklist: Iterable[int], seen: Iterable[int],
               wanted: Iterable[int] | Callable[[int], bool]) -> tuple[int, int]:
    """(outs, unseen) for a target set of card ids or a predicate over ids."""
    remaining = unseen_counts(decklist, seen)
    if callable(wanted):
        hit = wanted
    else:
        want = set(wanted)
        def hit(cid: int) -> bool:
            return cid in want
    outs = sum(n for cid, n in remaining.items() if hit(cid))
    return outs, sum(remaining.values())


# --- the ledger entry -------------------------------------------------------

def ledger(decklist: Iterable[int], seen: Iterable[int],
           wanted: Iterable[int] | Callable[[int], bool],
           label: str = "", max_draws: int = MAX_DRAWS) -> dict:
    """A whole explanation object for one fetch decision."""
    outs, unseen = count_outs(decklist, seen, wanted)
    curve = outs_curve(outs, unseen, max_draws)
    return {
        "label": label,
        "outs": outs,
        "unseen": unseen,
        "p_by_draws": {d: round(p, 4) for d, p in curve.items()},
        "text": format_outs(outs, unseen, max_draws, label),
    }


def format_outs(outs: int, unseen: int, draws: int = MAX_DRAWS,
                label: str = "") -> str:
    """`6 outs / 42 unseen -> 56% by turn 5`, optionally named."""
    p = p_at_least_one(outs, unseen, draws)
    head = f"{label}: " if label else ""
    return (f"{head}{outs} outs / {unseen} unseen -> "
            f"{p * 100:.0f}% by turn {draws}")


def format_curve(outs: int, unseen: int, max_draws: int = MAX_DRAWS,
                 label: str = "") -> str:
    """The whole curve, one draw count per line."""
    lines = [format_outs(outs, unseen, max_draws, label)]
    for d, p in outs_curve(outs, unseen, max_draws).items():
        lines.append(f"  draw {d}: {p * 100:5.1f}%")
    return "\n".join(lines)


# --- self-check -------------------------------------------------------------

def _self_check() -> None:
    def eq(got, want, what, tol=1e-12):
        assert abs(got - want) <= tol, f"{what}: got {got!r}, want {want!r}"

    # 1. One draw off a fresh 60 with 4 copies: the naive fraction.
    eq(p_at_least_one(4, 60, 1), 4 / 60, "4/60 in one draw")

    # 2. Five draws, hand-computed with math.comb.
    want = 1 - comb(56, 5) / comb(60, 5)
    eq(p_at_least_one(4, 60, 5), want, "4/60 in five draws")
    assert 0.30 < want < 0.31, f"sanity: 4-of by draw 5 is ~30.1%, got {want}"

    # 3. The docstring example, recomputed rather than quoted.
    eq(p_at_least_one(6, 42, 5), 1 - comb(36, 5) / comb(42, 5), "6/42 by 5")
    assert format_outs(6, 42, 5) == "6 outs / 42 unseen -> 56% by turn 5", \
        format_outs(6, 42, 5)

    # 4. Degenerate ends.
    eq(p_at_least_one(0, 40, 5), 0.0, "no outs is never")
    eq(p_at_least_one(40, 40, 1), 1.0, "all outs is always")
    eq(p_at_least_one(3, 3, 1), 1.0, "outs fill the unseen")
    eq(p_at_least_one(1, 5, 5), 1.0, "drawing the whole unseen finds it")
    eq(p_at_least_one(4, 60, 0), 0.0, "zero draws is never")

    # 5. Monotone in draws.
    curve = outs_curve(3, 47, 5)
    assert all(curve[d] <= curve[d + 1] for d in range(1, 5)), curve

    # 6. Two outs beat one at every draw count.
    for d in range(1, 6):
        assert p_at_least_one(2, 50, d) > p_at_least_one(1, 50, d), d

    # 7. Seen-card accounting: 4-of with two already visible.
    deck = [7] * 4 + [11] * 4 + [99] * 52
    outs, unseen = count_outs(deck, seen=[7, 7], wanted={7})
    assert (outs, unseen) == (2, 58), (outs, unseen)
    outs, unseen = count_outs(deck, seen=[7, 7, 11, 99, 99], wanted={7, 11})
    assert (outs, unseen) == (5, 55), (outs, unseen)

    # 8. A predicate is interchangeable with an id set.
    a = count_outs(deck, [], wanted={7, 11})
    b = count_outs(deck, [], wanted=lambda cid: cid in (7, 11))
    assert a == b == (8, 60), (a, b)

    # 9. Over-reported cards clamp instead of going negative.
    outs, unseen = count_outs([7] * 4 + [99] * 56, seen=[7] * 9, wanted={7})
    assert (outs, unseen) == (0, 56), (outs, unseen)

    # 10. The ledger object agrees with its own text.
    led = ledger(deck, [7, 7], {7}, label="Alakazam line")
    assert led["outs"] == 2 and led["unseen"] == 58, led
    eq(led["p_by_draws"][5], round(p_at_least_one(2, 58, 5), 4), "ledger p5")
    assert led["text"].startswith("Alakazam line: 2 outs / 58 unseen"), led

    print("outs self-check: 10 cases pass")
    print(format_curve(6, 42, 5, label="example"))
    print(format_curve(2, 58, 5, label="2 outs left of a 4-of"))


if __name__ == "__main__":
    _self_check()
