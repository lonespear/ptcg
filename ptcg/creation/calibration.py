"""Calibrated win probability (D16.3): what the evaluator's margin is worth.

The agent scores a position with a linear evaluator whose units are arbitrary
— "+1400" means nothing to a judge. This fits the map from that margin to the
only unit that matters, P(win), over self-play games, and then checks the map
against held-out games: when the agent says 70%, does it win 70%?

The evaluator is recomputed here from the observation dict rather than
imported, so the fit is an independent reimplementation of the arithmetic and
a drift check on it:

    margin = 1000 * (their prizes left - our prizes left)
           +        (our board HP     - their board HP)
           +   30 * (our Energy in play - their Energy in play)
           +    5 *  our hand size

The fit is isotonic (pool-adjacent-violators): monotone by construction, so a
bigger margin can never map to a smaller win probability, and free of any
shape assumption otherwise. Training and reliability use disjoint halves of
the games — an isotonic fit scored on its own training data is perfect by
construction and tells a judge nothing.

Usage: python -m ptcg.creation.calibration --games 2000
       python -m ptcg.creation.calibration --games 4000 --out data/calibration.json
"""

import argparse
import json
import random
import time
from pathlib import Path

from cg.game import battle_finish, battle_select, battle_start

from .goldfish import read_deck
from .pilots import JonDayPilot

RESULT = 23
ROOT = Path(__file__).resolve().parents[2]

PRIZE_WEIGHT = 1000.0
ENERGY_WEIGHT = 30.0
HAND_WEIGHT = 5.0


# --- the evaluator, recomputed from the observation dict --------------------

def _side_hp(player: dict) -> float:
    total = 0.0
    for zone in ("active", "bench"):
        for mon in player.get(zone) or []:
            if mon:
                total += mon.get("hp", 0) or 0
    return total


def _side_energy(player: dict) -> float:
    total = 0.0
    for zone in ("active", "bench"):
        for mon in player.get(zone) or []:
            if mon:
                total += len(mon.get("energies") or [])
    return total


def margin(obs: dict, me: int) -> float:
    """Evaluator margin from seat `me`, in the agent's own units."""
    cur = obs["current"]
    mine = cur["players"][me]
    theirs = cur["players"][1 - me]
    score = (len(theirs.get("prize") or [])
             - len(mine.get("prize") or [])) * PRIZE_WEIGHT
    score += _side_hp(mine) - _side_hp(theirs)
    score += ENERGY_WEIGHT * (_side_energy(mine) - _side_energy(theirs))
    score += (mine.get("handCount", 0) or 0) * HAND_WEIGHT
    return score


# --- self-play data collection ----------------------------------------------

def collect_game(agent0, agent1, deck0: list[int], deck1: list[int],
                 max_selects: int = 20000) -> dict:
    """One game; returns (turn, seat, margin) at every turn boundary plus the
    winner. The margin is taken from the seat on move, which is the seat the
    agent would be evaluating for."""
    obs, start = battle_start(deck0, deck1)
    if obs is None:
        raise RuntimeError(f"battle_start failed: errorType={start.errorType}")
    agents = (agent0, agent1)
    for a, d in ((agent0, deck0), (agent1, deck1)):
        if hasattr(a, "bind_deck"):
            a.bind_deck(d)
    points: list[tuple[int, int, float]] = []
    winner = None
    last_turn = -1
    selects = 0
    try:
        while selects < max_selects:
            for log in obs["logs"]:
                if log["type"] == RESULT:
                    winner = log["result"]
            if winner is not None:
                break
            me = obs["current"]["yourIndex"]
            turn = int(obs["current"].get("turn", 0) or 0)
            if turn != last_turn:
                points.append((turn, me, margin(obs, me)))
                last_turn = turn
            obs = battle_select(agents[me](obs))
            selects += 1
        return {"winner": winner, "points": points}
    finally:
        battle_finish()


def collect(decks: list[tuple[str, list[int]]], n_games: int,
            seed: int = 3) -> list[dict]:
    """Self-play across every ordered deck pairing, round-robin."""
    a0 = JonDayPilot(seed=seed, search=False)
    a1 = JonDayPilot(seed=seed + 1, search=False)
    pairs = [(i, j) for i in range(len(decks)) for j in range(len(decks))
             if i != j]
    rng = random.Random(seed)
    rng.shuffle(pairs)
    games = []
    for g in range(n_games):
        i, j = pairs[g % len(pairs)]
        out = collect_game(a0, a1, decks[i][1], decks[j][1])
        if out["winner"] not in (0, 1):
            continue                       # draws and caps carry no label
        games.append({
            "game": g,
            "matchup": f"{decks[i][0]} vs {decks[j][0]}",
            "samples": [{"turn": t, "y": 1 if w == out["winner"] else 0,
                         "margin": m} for t, w, m in out["points"]],
        })
    return games


# --- isotonic regression (pool adjacent violators) ---------------------------

def isotonic_fit(x: list[float], y: list[int]) -> list[dict]:
    """Monotone non-decreasing least-squares fit of P(win) on margin.

    Returns the step function as breakpoints: `p` holds from `margin` up to
    the next breakpoint. Samples sharing a margin are pooled first, so the
    result is a genuine function of the margin. sklearn's IsotonicRegression
    when the venv has it, pool-adjacent-violators otherwise — the same
    estimator either way.
    """
    agg: dict[float, list[float]] = {}
    for xi, yi in zip(x, y):
        cell = agg.setdefault(xi, [0.0, 0.0])
        cell[0] += yi
        cell[1] += 1.0
    xs = sorted(agg)
    means = [agg[v][0] / agg[v][1] for v in xs]
    weights = [agg[v][1] for v in xs]

    fitted: list[float] | None = None
    try:
        from sklearn.isotonic import IsotonicRegression
        ir = IsotonicRegression(out_of_bounds="clip")
        fitted = [float(v) for v in
                  ir.fit_transform(xs, means, sample_weight=weights)]
    except ImportError:
        blocks: list[list[float]] = []        # [start_index, value, weight]
        for i, (m, w) in enumerate(zip(means, weights)):
            blocks.append([i, m, w])
            while len(blocks) > 1 and blocks[-2][1] > blocks[-1][1]:
                _, v2, w2 = blocks.pop()
                i1, v1, w1 = blocks.pop()
                blocks.append([i1, (v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2])
        fitted = [0.0] * len(xs)
        for b, block in enumerate(blocks):
            end = blocks[b + 1][0] if b + 1 < len(blocks) else len(xs)
            for i in range(int(block[0]), end):
                fitted[i] = block[1]

    curve = []
    for xv, fv in zip(xs, fitted):
        if not curve or abs(curve[-1]["p"] - fv) > 1e-12:
            curve.append({"margin": float(xv), "p": float(fv)})
    return curve


def predict(curve: list[dict], m: float) -> float:
    """Step lookup: the probability of the last breakpoint at or below m."""
    if not curve:
        return 0.5
    lo, hi = 0, len(curve) - 1
    if m < curve[0]["margin"]:
        return curve[0]["p"]
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if curve[mid]["margin"] <= m:
            lo = mid
        else:
            hi = mid - 1
    return curve[lo]["p"]


# --- reliability -------------------------------------------------------------

BINS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def reliability(preds: list[float], ys: list[int],
                edges: list[float] = BINS) -> list[dict]:
    """Predicted-vs-actual by probability bin."""
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        last = hi >= edges[-1]
        idx = [i for i, p in enumerate(preds)
               if (lo <= p < hi) or (last and p == hi)]
        if not idx:
            rows.append({"bin": f"[{lo:.1f},{hi:.1f})", "n": 0,
                         "predicted": None, "actual": None, "gap": None})
            continue
        pm = sum(preds[i] for i in idx) / len(idx)
        am = sum(ys[i] for i in idx) / len(idx)
        rows.append({"bin": f"[{lo:.1f},{hi:.1f})", "n": len(idx),
                     "predicted": round(pm, 4), "actual": round(am, 4),
                     "gap": round(am - pm, 4)})
    return rows


def brier(preds: list[float], ys: list[int]) -> float:
    return sum((p - y) ** 2 for p, y in zip(preds, ys)) / len(preds)


def ece(rows: list[dict], n_total: int) -> float:
    """Expected calibration error: n-weighted mean |actual - predicted|."""
    return sum(r["n"] * abs(r["gap"]) for r in rows if r["n"]) / n_total


# --- the whole run -----------------------------------------------------------

def run(decks: list[tuple[str, list[int]]], n_games: int = 2000,
        seed: int = 3) -> dict:
    t0 = time.time()
    games = collect(decks, n_games, seed=seed)
    rng = random.Random(seed + 7)
    split = {g["game"]: rng.random() < 0.5 for g in games}   # by game, not row

    def rows(train: bool):
        xs, ys, turns = [], [], []
        for g in games:
            if split[g["game"]] != train:
                continue
            for s in g["samples"]:
                xs.append(s["margin"])
                ys.append(s["y"])
                turns.append(s["turn"])
        return xs, ys, turns

    tr_x, tr_y, _ = rows(True)
    te_x, te_y, te_t = rows(False)
    if not tr_x or not te_x:
        raise RuntimeError("not enough games to split")

    curve = isotonic_fit(tr_x, tr_y)

    te_p = [predict(curve, m) for m in te_x]
    tr_p = [predict(curve, m) for m in tr_x]
    rel = reliability(te_p, te_y)
    base = sum(te_y) / len(te_y)

    # what the curve says at margins a reader can picture
    landmarks = [-3000, -2000, -1000, -500, -200, 0, 200, 500, 1000,
                 2000, 3000]
    by_turn = {}
    for t in sorted({min(t, 20) for t in te_t}):
        idx = [i for i, tt in enumerate(te_t) if min(tt, 20) == t]
        if len(idx) < 30:
            continue
        by_turn[str(t)] = {
            "n": len(idx),
            "predicted": round(sum(te_p[i] for i in idx) / len(idx), 4),
            "actual": round(sum(te_y[i] for i in idx) / len(idx), 4),
        }

    return {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "decks": [d[0] for d in decks],
        "n_games_requested": n_games,
        "n_games_decided": len(games),
        "n_samples_train": len(tr_x),
        "n_samples_test": len(te_x),
        "elapsed_s": round(time.time() - t0, 1),
        "weights": {"prize": PRIZE_WEIGHT, "hp": 1.0,
                    "energy": ENERGY_WEIGHT, "hand": HAND_WEIGHT},
        "curve_breakpoints": len(curve),
        "curve": curve,
        "landmarks": {str(m): round(predict(curve, m), 4) for m in landmarks},
        "reliability_test": rel,
        "reliability_train": reliability(tr_p, tr_y),
        "brier_test": round(brier(te_p, te_y), 4),
        "brier_baserate": round(brier([base] * len(te_y), te_y), 4),
        "brier_train": round(brier(tr_p, tr_y), 4),
        "ece_test": round(ece(rel, len(te_y)), 4),
        "base_rate_test": round(base, 4),
        "by_turn_test": by_turn,
    }


def format_report(rep: dict) -> str:
    lines = [
        f"calibration fit  {rep['generated']}",
        f"decks: {', '.join(rep['decks'])}",
        f"games {rep['n_games_decided']} decided of {rep['n_games_requested']}"
        f"   samples {rep['n_samples_train']} train / "
        f"{rep['n_samples_test']} test   ({rep['elapsed_s']}s)",
        f"isotonic curve: {rep['curve_breakpoints']} breakpoints",
        "",
        "reliability on held-out games",
        f"{'bin':>12}{'n':>8}{'predicted':>12}{'actual':>9}{'gap':>9}",
    ]
    for r in rep["reliability_test"]:
        if not r["n"]:
            lines.append(f"{r['bin']:>12}{0:>8}{'-':>12}{'-':>9}{'-':>9}")
            continue
        lines.append(f"{r['bin']:>12}{r['n']:>8}{r['predicted']:>12.3f}"
                     f"{r['actual']:>9.3f}{r['gap']:>+9.3f}")
    lines += [
        "",
        f"Brier  test {rep['brier_test']}   base rate "
        f"{rep['brier_baserate']}   train {rep['brier_train']}",
        f"expected calibration error (test) {rep['ece_test']}",
        f"test base rate {rep['base_rate_test']}",
        "",
        "P(win) at landmark margins",
    ]
    for m, p in rep["landmarks"].items():
        lines.append(f"  margin {int(m):>+6} -> {p:.3f}")
    if rep["by_turn_test"]:
        lines += ["", "held-out predicted vs actual by turn",
                  f"{'turn':>6}{'n':>8}{'predicted':>12}{'actual':>9}"]
        for t, r in rep["by_turn_test"].items():
            lines.append(f"{t:>6}{r['n']:>8}{r['predicted']:>12.3f}"
                         f"{r['actual']:>9.3f}")
    return "\n".join(lines)


def priors_decks(n: int) -> list[tuple[str, list[int]]]:
    priors = json.loads((ROOT / "agent" / "deck_priors.json").read_text())
    top = sorted(priors["decks"], key=lambda d: -d["p"])
    out, seen = [], set()
    for entry in top:
        if entry["a"] in seen:
            continue                       # one list per archetype
        seen.add(entry["a"])
        out.append((entry["a"],
                    [int(c) for c, k in entry["c"].items() for _ in range(k)]))
        if len(out) == n:
            break
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--decks", type=int, default=4,
                    help="how many top field archetypes to play")
    ap.add_argument("--deck", action="append", default=None,
                    help="explicit deck source; repeatable")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default=str(ROOT / "data" / "calibration.json"))
    a = ap.parse_args()

    if a.deck:
        decks = [(d, read_deck(d)) for d in a.deck]
    else:
        decks = priors_decks(a.decks)
    rep = run(decks, n_games=a.games, seed=a.seed)
    print(format_report(rep))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2))
    print(f"\nwrote {out}")
