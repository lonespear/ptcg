"""R1 weight tuner: two optimizers over one weight vector (decisions.md D7/D12).

D7 puts learning into the pilot as feature-weight tuning, never as an opaque
policy, and D16 keeps every decision an arithmetic identity over named terms.
So the object being optimized here is `agent.main.WEIGHTS` — six named eval
constants — and nothing else about the pilot changes.

Two optimizers see the same vector, which makes them comparable:

  ES   (mu=4, lambda=12) over log-space perturbations. Black box: it only
       ever asks "does this vector win more games?". Prize stays pinned at
       1000 as the scale anchor, so the other five dims are tuned relative
       to a prize.

  TD(lambda)  linear value function over the same named terms, learned from
       self-play by temporal-difference credit assignment with eligibility
       traces, then mapped back to eval-weight ratios by normalizing the
       learned coefficients so prize = 1000. TD never plays a candidate
       vector: it watches the default pilot play and infers what the terms
       are worth.

Fitness of a vector w: JonDayPilot(search=False, weights=w) against
JonDayPilot(search=False, weights=defaults), mirror matches (same deck both
seats) on three decks, seats swapped, win rate averaged over decks.

Usage:
    python -m ptcg.creation.tuning --mode both --games 200 --workers 8 \
        --out data/tuned_weights.json --stamp r1
    python -m ptcg.creation.tuning --mode duel --games 300 --workers 8 \
        --out data/tuned_weights.json --stamp r1

Modes accumulate into --out: es and td write under "methods", duel reads
whatever methods are already there and writes the head-to-head table.
"""

import argparse
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# --- the vector -------------------------------------------------------------
# prize is the scale anchor and is never tuned: a weight vector only has
# meaning up to a positive scale, so one dim must be pinned or the optimizer
# wanders along the ray.
ANCHOR = "prize"
ANCHOR_VALUE = 1000.0
TUNED_DIMS = ("hp", "energy", "hand", "no_active", "search_margin")

# ES works in log space so a step means "x% bigger", not "+0.3", which is what
# ratio-valued weights want. search_margin's default is 0 and log 0 is not a
# number, so it is coded as log(x + 1) and decoded as exp(theta) - 1.
LOG_SHIFT = {"search_margin": 1.0}

MIN_GAMES_PER_DECK = 2


def defaults() -> dict:
    """The incumbent weight vector, straight from the live agent."""
    import agent.main as jon
    return dict(jon._DEFAULT_WEIGHTS)


def _encode(weights: dict) -> list[float]:
    return [math.log(weights[d] + LOG_SHIFT.get(d, 0.0)) for d in TUNED_DIMS]


def _decode(theta: list[float]) -> dict:
    w = {ANCHOR: ANCHOR_VALUE}
    for d, t in zip(TUNED_DIMS, theta):
        # exp() keeps hp/energy/hand/no_active positive; the shifted dims are
        # floored at 0 because a negative search_margin would invert the
        # override test rather than loosen it.
        w[d] = max(0.0, math.exp(min(t, 30.0)) - LOG_SHIFT.get(d, 0.0))
    return w


def _round(weights: dict, places: int = 4) -> dict:
    return {k: round(float(v), places) for k, v in weights.items()}


# --- the deck set -----------------------------------------------------------
def _expand(counts: dict) -> list[int]:
    return [int(cid) for cid, n in counts.items() for _ in range(n)]


def load_decks(root: Path = ROOT) -> list[dict]:
    """Three mirror decks: two top field lists plus today's GA elite.

    Field lists come from agent/deck_priors.json (mined play counts, sorted by
    weight), the elite from the archipelago run's archive. If the archive is
    missing the slot falls back to the next-most-played archetype, so the
    tuner never depends on a run that may still be writing.
    """
    priors = json.loads((root / "agent" / "deck_priors.json").read_text())
    entries = priors["decks"]

    def top(archetype: str) -> dict | None:
        hits = [e for e in entries if e.get("a") == archetype]
        return max(hits, key=lambda e: e["w"]) if hits else None

    decks: list[dict] = []
    for archetype in ("Marnie's Grimmsnarl ex", "Cynthia's Garchomp ex"):
        entry = top(archetype)
        if entry is not None:
            decks.append({"name": archetype, "source": "deck_priors.json",
                          "play_weight": entry["w"], "deck": _expand(entry["c"])})

    elite_path = root / "runs" / "archi_r0" / "latest.json"
    elite = None
    if elite_path.exists():
        try:
            record = json.loads(elite_path.read_text())
            cards = (record.get("elite") or {}).get("deck")
            if cards and len(cards) == 60:
                elite = {"name": "GA elite (archi_r0)",
                         "source": "runs/archi_r0/latest.json",
                         "island": (record.get("elite") or {}).get("island"),
                         "ga_fitness": (record.get("elite") or {}).get("fitness"),
                         "deck": [int(c) for c in cards]}
        except (OSError, ValueError, KeyError):
            elite = None
    if elite is None:
        used = {d["name"] for d in decks}
        for entry in entries:
            if entry.get("a") not in used:
                elite = {"name": entry["a"], "source": "deck_priors.json (elite fallback)",
                         "play_weight": entry["w"], "deck": _expand(entry["c"])}
                break
    if elite is not None:
        decks.append(elite)

    bad = [d["name"] for d in decks if len(d["deck"]) != 60]
    if bad:
        raise ValueError(f"decks are not 60 cards: {bad}")
    return decks


# --- worker side ------------------------------------------------------------
# One engine per process (the cabt engine plays one battle at a time per
# process), same shape as ptcg/creation/parallel.py. Workers hold the deck
# list; jobs carry weight vectors, which are small.
_STATE: dict = {}


def _init_worker(root: str, decks_json: str) -> None:
    os.chdir(root)
    if root not in sys.path:
        sys.path.insert(0, root)
    import ptcg.creation  # noqa: F401 — engine bootstrap
    _STATE["decks"] = json.loads(decks_json)


def _pilot(weights: dict | None):
    from ptcg.creation.pilots import JonDayPilot
    return JonDayPilot(search=False, weights=weights)


def _match_job(job: tuple) -> tuple:
    """(tag, deck_index, weights_a, weights_b, n_games) -> win rate for A.

    weights None means the shipped defaults. Seats are swapped inside
    play_match, so the returned rate is seat-neutral.
    """
    from ptcg.creation.harness import play_match
    tag, deck_i, wa, wb, n_games = job
    deck = _STATE["decks"][deck_i]
    match = play_match(_pilot(wa), _pilot(wb), deck, deck, n_games)
    w0, w1, draws = match.wins
    return tag, deck_i, match.win_rate(0), w0, w1, draws


# --- TD(lambda) features ----------------------------------------------------
# The same named terms agent.main._evaluate uses, recomputed from the raw
# observation dict instead of a search state. Scales are fixed constants that
# put every feature in roughly [-1, 1] so one learning rate fits all of them;
# they are divided back out when the learned vector is mapped to eval weights.
FEATURES = ("prize", "hp", "energy", "hand", "no_active")
FEATURE_SCALE = {"prize": 6.0, "hp": 1000.0, "energy": 10.0,
                 "hand": 10.0, "no_active": 1.0}


def _mons(side: dict) -> list:
    out = []
    for zone in ("active", "bench"):
        for mon in side.get(zone) or []:
            if mon:
                out.append(mon)
    return out


def _obs_features(obs: dict, me: int) -> list[float]:
    cur = obs["current"]
    mine = cur["players"][me]
    theirs = cur["players"][1 - me]
    prize = len(theirs.get("prize") or []) - len(mine.get("prize") or [])
    hp = (sum(m.get("hp") or 0 for m in _mons(mine))
          - sum(m.get("hp") or 0 for m in _mons(theirs)))
    energy = (sum(len(m.get("energies") or []) for m in _mons(mine))
              - sum(len(m.get("energies") or []) for m in _mons(theirs)))
    hand = mine.get("handCount") or 0
    active = mine.get("active") or []
    no_active = 0.0 if (active and active[0]) else 1.0
    raw = {"prize": prize, "hp": hp, "energy": energy,
           "hand": hand, "no_active": no_active}
    return [raw[f] / FEATURE_SCALE[f] for f in FEATURES]


def _selfplay_job(job: tuple) -> list:
    """(deck_index, n_games, first_game_index) -> [(features, outcome), ...].

    Self-play by the default rules pilot on both seats. The behaviour policy
    is fixed, so trajectories do not depend on the value function being
    learned and can be generated in parallel without changing the result.
    The learner's seat alternates by game so first-player advantage does not
    leak into the outcome signal. One sample per turn, taken at the first
    decision of that turn.
    """
    from cg.game import battle_start, battle_select, battle_finish
    from ptcg.creation.harness import TURN_START, RESULT
    deck_i, n_games, first = job
    deck = _STATE["decks"][deck_i]
    out = []
    for g in range(n_games):
        me = (first + g) % 2
        agents = (_pilot(None), _pilot(None))
        obs, start = battle_start(deck, deck)
        if obs is None:
            raise RuntimeError(f"battle_start failed: {start.errorType}")
        for agent in agents:
            agent.bind_deck(deck)
        feats: list[list[float]] = []
        seen_turn = -1
        winner = None
        try:
            for _ in range(20000):
                for log in obs["logs"]:
                    if log["type"] == RESULT:
                        winner = log["result"]
                if winner is not None:
                    break
                cur = obs["current"]
                seat = cur["yourIndex"]
                if seat == me and cur.get("turn") != seen_turn:
                    seen_turn = cur.get("turn")
                    feats.append(_obs_features(obs, me))
                obs = battle_select(agents[seat](obs))
        finally:
            battle_finish()
        if winner is None or not feats:
            continue                      # capped or empty: no usable signal
        outcome = 0.0 if winner == 2 else (1.0 if winner == me else -1.0)
        out.append((feats, outcome))
    return out


# --- runner -----------------------------------------------------------------
class Runner:
    """Map jobs over a process pool, or in-process when workers == 1."""

    def __init__(self, workers: int, decks: list[list[int]], root: Path = ROOT):
        self.workers = max(1, workers)
        payload = (str(root), json.dumps(decks))
        if self.workers == 1:
            _init_worker(*payload)
            self.pool = None
        else:
            self.pool = ProcessPoolExecutor(
                max_workers=self.workers, mp_context=get_context("spawn"),
                initializer=_init_worker, initargs=payload)

    def map(self, fn, jobs: list) -> list:
        jobs = list(jobs)
        if self.pool is None:
            return [fn(j) for j in jobs]
        return list(self.pool.map(fn, jobs, chunksize=1))

    def close(self) -> None:
        if self.pool is not None:
            self.pool.shutdown(wait=True)


# --- fitness ----------------------------------------------------------------
def _split_games(games: int, n_decks: int) -> int:
    return max(MIN_GAMES_PER_DECK, games // n_decks)


def evaluate_batch(runner: Runner, candidates: list[dict], n_decks: int,
                   games: int) -> list[dict]:
    """Fitness for a batch of vectors: win rate vs defaults, mean over decks.

    The whole batch goes out as one map so a generation fills the pool.
    """
    per_deck = _split_games(games, n_decks)
    jobs = [(i, d, w, None, per_deck)
            for i, w in enumerate(candidates) for d in range(n_decks)]
    scores: list[list[float]] = [[] for _ in candidates]
    for tag, _deck_i, rate, *_ in runner.map(_match_job, jobs):
        scores[tag].append(rate)
    return [{"fitness": sum(s) / len(s), "per_deck": s} for s in scores]


# --- optimizer 1: evolution strategies --------------------------------------
def run_es(runner: Runner, decks: list[dict], games: int, gens: int = 10,
           mu: int = 4, lam: int = 12, sigma0: float = 0.5,
           decay: float = 0.9, seed: int = 0) -> dict:
    """(mu, lambda) ES over log-space perturbations of the tuned dims.

    Each generation samples lambda children around the current mean, keeps the
    best mu, and recentres on their mean; sigma decays geometrically so early
    generations range and late ones polish. The mean itself is re-evaluated
    each generation, which is what the trajectory reports — a child's score is
    a lucky draw as often as a good vector, at these sample sizes.

    The answer returned is the final distribution mean, not the highest score
    seen. Taking the maximum over noisy evaluations returns whichever vector
    got the friendliest shuffles; the mean is what the population actually
    learned. The high-water mark is recorded as best_observed for reference.
    """
    rng = random.Random(seed)
    n_decks = len(decks)
    theta = _encode(defaults())
    sigma = sigma0
    baseline = evaluate_batch(runner, [None], n_decks, games)[0]["fitness"]
    parent = evaluate_batch(runner, [_decode(theta)], n_decks, games)[0]
    best = {"weights": _round(_decode(theta)), "fitness": round(parent["fitness"], 4),
            "generation": 0}
    final = {"weights": _decode(theta), "fitness": parent["fitness"]}
    trajectory = [{"generation": 0, "sigma": round(sigma, 4),
                   "mean_fitness": round(parent["fitness"], 4),
                   "best_child": None, "median_child": None,
                   "weights": _round(_decode(theta))}]
    print(f"  defaults-vs-defaults control {baseline:.3f} | "
          f"gen 0 mean {parent['fitness']:.3f}", flush=True)

    for gen in range(1, gens + 1):
        children = [[t + rng.gauss(0.0, sigma) for t in theta]
                    for _ in range(lam)]
        results = evaluate_batch(runner, [_decode(c) for c in children],
                                 n_decks, games)
        ranked = sorted(range(lam), key=lambda i: results[i]["fitness"],
                        reverse=True)
        elite = ranked[:mu]
        theta = [sum(children[i][d] for i in elite) / mu
                 for d in range(len(TUNED_DIMS))]
        weights = _decode(theta)
        mean_score = evaluate_batch(runner, [weights], n_decks, games)[0]
        fits = sorted(r["fitness"] for r in results)
        trajectory.append({
            "generation": gen, "sigma": round(sigma, 4),
            "mean_fitness": round(mean_score["fitness"], 4),
            "best_child": round(fits[-1], 4),
            "median_child": round(fits[len(fits) // 2], 4),
            "weights": _round(weights)})
        if mean_score["fitness"] > best["fitness"]:
            best = {"weights": _round(weights),
                    "fitness": round(mean_score["fitness"], 4),
                    "generation": gen}
        final = {"weights": weights, "fitness": mean_score["fitness"]}
        print(f"  gen {gen:>2} sigma {sigma:.3f} mean {mean_score['fitness']:.3f} "
              f"best-child {fits[-1]:.3f} | "
              + " ".join(f"{d}={weights[d]:.3g}" for d in TUNED_DIMS),
              flush=True)
        sigma *= decay

    per_eval = _split_games(games, n_decks) * n_decks
    return {"method": "es", "weights": _round(final["weights"]),
            "fitness": round(final["fitness"], 4),
            "from_generation": gens,
            "best_observed": best,
            "control_defaults_vs_defaults": round(baseline, 4),
            "trajectory": trajectory,
            "settings": {"mu": mu, "lambda": lam, "generations": gens,
                         "sigma0": sigma0, "sigma_decay": decay, "seed": seed},
            "games_per_eval": per_eval,
            "total_games": per_eval * (lam + 1) * gens + 2 * per_eval}


# --- optimizer 2: TD(lambda) ------------------------------------------------
def td_update(theta: list[float], feats: list[list[float]], outcome: float,
              lam: float, gamma: float, alpha: float) -> float:
    """One episode of online TD(lambda) with accumulating eligibility traces.

    Non-terminal target is the next state's own value; the last state's target
    is the game result (+1 win, -1 loss, 0 draw). Returns mean |delta|, which
    is the convergence trace worth watching.
    """
    trace = [0.0] * len(theta)
    total = 0.0
    for t, phi in enumerate(feats):
        value = sum(w * x for w, x in zip(theta, phi))
        if t + 1 < len(feats):
            nxt = feats[t + 1]
            target = gamma * sum(w * x for w, x in zip(theta, nxt))
        else:
            target = outcome
        delta = target - value
        total += abs(delta)
        for i, x in enumerate(phi):
            trace[i] = gamma * lam * trace[i] + x
            theta[i] += alpha * delta * trace[i]
    return total / len(feats)


def theta_to_weights(theta: list[float]) -> tuple[dict, dict]:
    """Map learned value-function coefficients to eval weights.

    Undo the feature scaling, then normalize so prize = 1000 — the same anchor
    ES uses, which is what makes the two vectors comparable. no_active enters
    the evaluator as a penalty (score -= w * flag), so its learned coefficient
    flips sign on the way out. TD sees no signal for search_margin, which is a
    search-override threshold rather than a position term, so it keeps the
    default.

    Read the hand coefficient with care: the value function has no intercept,
    and hand count is the one feature that is never near zero, so it absorbs
    whatever constant offset the outcome scale carries. Its mapped weight is
    therefore an upper bound on what TD actually attributes to holding cards.
    """
    raw = {f: theta[i] / FEATURE_SCALE[f] for i, f in enumerate(FEATURES)}
    anchor = raw["prize"]
    notes = {"raw_coefficients": {k: round(v, 6) for k, v in raw.items()}}
    if anchor <= 0:
        notes["degenerate"] = ("learned prize coefficient is not positive "
                               f"({anchor:.6g}); defaults returned")
        return defaults(), notes
    scale = ANCHOR_VALUE / anchor
    weights = {ANCHOR: ANCHOR_VALUE}
    for f in ("hp", "energy", "hand"):
        weights[f] = raw[f] * scale
    weights["no_active"] = -raw["no_active"] * scale
    if weights["no_active"] < 0:
        notes["no_active_sign"] = ("TD learned an empty Active Spot as good; "
                                   "clamped to 0")
        weights["no_active"] = 0.0
    weights["search_margin"] = defaults()["search_margin"]
    notes["search_margin"] = "not a position feature; default kept"
    return weights, notes


def run_td(runner: Runner, decks: list[dict], games: int, lam: float = 0.8,
           gamma: float = 1.0, alpha0: float = 0.05,
           batch: int = 64) -> dict:
    """TD(lambda) over self-play games by the default rules pilot.

    Games are generated in parallel batches and the updates applied in order,
    which is identical to sequential play because the behaviour policy is
    fixed. alpha anneals as alpha0 / (1 + g / (games/4)): a quarter of the run
    at close to full rate, then a long decay.
    """
    n_decks = len(decks)
    theta = [0.0] * len(FEATURES)
    trajectory = []
    played = kept = 0
    chunk = 0
    anneal = max(1.0, games / 4.0)
    while played < games:
        take = min(batch * runner.workers, games - played)
        per_job = max(1, take // runner.workers)
        jobs, issued = [], 0
        while issued < take:
            n = min(per_job, take - issued)
            # decks rotate job by job, so the value function is fit across the
            # whole deck set rather than to one archetype's board shapes
            jobs.append((chunk % n_decks, n, played + issued))
            chunk += 1
            issued += n
        deltas = []
        for episodes in runner.map(_selfplay_job, jobs):
            for feats, outcome in episodes:
                alpha = alpha0 / (1.0 + kept / anneal)
                deltas.append(td_update(theta, feats, outcome, lam, gamma, alpha))
                kept += 1
        played += take
        weights, _ = theta_to_weights(theta)
        trajectory.append({
            "games": played, "episodes_used": kept,
            "alpha": round(alpha0 / (1.0 + kept / anneal), 5),
            "mean_abs_td_error": round(sum(deltas) / len(deltas), 4) if deltas else None,
            "theta": {f: round(theta[i], 5) for i, f in enumerate(FEATURES)},
            "weights": _round(weights)})
        print(f"  {played:>5} games | mean |TD error| "
              f"{trajectory[-1]['mean_abs_td_error']} | "
              + " ".join(f"{f}={theta[i]:+.4f}" for i, f in enumerate(FEATURES)),
              flush=True)

    weights, notes = theta_to_weights(theta)
    return {"method": "td", "weights": _round(weights),
            "theta": {f: round(theta[i], 6) for i, f in enumerate(FEATURES)},
            "feature_scale": FEATURE_SCALE, "mapping_notes": notes,
            "trajectory": trajectory,
            "settings": {"lambda": lam, "gamma": gamma, "alpha0": alpha0,
                         "alpha_anneal_games": anneal, "batch": batch},
            "games": played, "episodes_used": kept, "total_games": played}


# --- head to head -----------------------------------------------------------
def run_duel(runner: Runner, decks: list[dict], entries: list[tuple[str, dict | None]],
             games: int) -> dict:
    """Round robin over named weight vectors: games per pairing per deck.

    Only the upper triangle is played; the mirrored cell is 1 - rate, since
    win_rate already drops draws. Diagonals are same-vector mirrors and read
    as a noise floor on the whole table.
    """
    labels = [name for name, _ in entries]
    n = len(entries)
    jobs = []
    for i in range(n):
        for j in range(i, n):
            for d in range(len(decks)):
                jobs.append(((i, j, d), d, entries[i][1], entries[j][1], games))
    cells: dict[tuple, list] = {}
    for (i, j, _d), deck_i, rate, w0, w1, draws in runner.map(_match_job, jobs):
        cells.setdefault((i, j), []).append(
            {"deck": decks[deck_i]["name"], "rate": round(rate, 4),
             "wins": w0, "losses": w1, "draws": draws})
    table = [[None] * n for _ in range(n)]
    per_deck = {}
    for (i, j), rows in cells.items():
        mean = sum(r["rate"] for r in rows) / len(rows)
        table[i][j] = round(mean, 4)
        table[j][i] = round(1.0 - mean, 4)
        per_deck[f"{labels[i]} vs {labels[j]}"] = rows
    return {"labels": labels, "table": table, "per_deck": per_deck,
            "games_per_pairing_per_deck": games,
            "total_games": len(jobs) * games,
            "note": "cell [row][col] = row's win rate against col, "
                    "averaged over decks, seats swapped"}


def format_duel(duel: dict) -> str:
    labels = duel["labels"]
    width = max(len(x) for x in labels) + 2
    head = " " * width + "".join(f"{x:>10}" for x in labels)
    lines = [head]
    for i, row in enumerate(duel["table"]):
        cells = "".join(f"{v:>10.3f}" if v is not None else f"{'-':>10}"
                        for v in row)
        lines.append(f"{labels[i]:<{width}}{cells}")
    return "\n".join(lines)


# --- CLI --------------------------------------------------------------------
def _load_out(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except ValueError:
            pass
    return {}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", choices=("es", "td", "both", "duel"),
                    default="both")
    ap.add_argument("--games", type=int, default=None,
                    help="ES: games per fitness evaluation (split over decks); "
                         "TD: self-play games; duel: games per pairing per deck")
    ap.add_argument("--es-games", type=int, default=None)
    ap.add_argument("--td-games", type=int, default=None)
    ap.add_argument("--duel-games", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--gens", type=int, default=10)
    ap.add_argument("--mu", type=int, default=4)
    ap.add_argument("--lam", type=int, default=12, help="ES lambda (children)")
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--sigma-decay", type=float, default=0.9)
    ap.add_argument("--td-lambda", type=float, default=0.8)
    ap.add_argument("--td-alpha", type=float, default=0.05)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/tuned_weights.json")
    ap.add_argument("--stamp", default=None,
                    help="run label recorded in the output JSON")
    args = ap.parse_args(argv)

    es_games = args.es_games or args.games or 200
    td_games = args.td_games or args.games or 3000
    duel_games = args.duel_games or args.games or 300

    decks = load_decks()
    print(f"decks: " + ", ".join(f"{d['name']} [{d['source']}]" for d in decks),
          flush=True)
    runner = Runner(args.workers, [d["deck"] for d in decks])
    out_path = Path(args.out)
    record = _load_out(out_path)
    record.setdefault("methods", {})
    record["stamp"] = args.stamp
    record["timestamp"] = time.time()
    record["defaults"] = _round(defaults())
    record["decks"] = [{k: v for k, v in d.items() if k != "deck"} for d in decks]
    started = time.time()

    try:
        if args.mode in ("es", "both"):
            print(f"ES (mu={args.mu}, lambda={args.lam}) "
                  f"{args.gens} gens, {es_games} games/eval", flush=True)
            record["methods"]["es"] = run_es(
                runner, decks, es_games, gens=args.gens, mu=args.mu,
                lam=args.lam, sigma0=args.sigma, decay=args.sigma_decay,
                seed=args.seed)
        if args.mode in ("td", "both"):
            print(f"TD(lambda={args.td_lambda}) over {td_games} self-play games",
                  flush=True)
            record["methods"]["td"] = run_td(
                runner, decks, td_games, lam=args.td_lambda, gamma=args.gamma,
                alpha0=args.td_alpha)
        if args.mode == "duel":
            entries: list[tuple[str, dict | None]] = []
            for name in ("es", "td"):
                got = record["methods"].get(name)
                if got and got.get("weights"):
                    entries.append((name, got["weights"]))
                else:
                    print(f"  no {name} weights in {out_path}; skipping",
                          flush=True)
            entries.append(("defaults", None))
            print(f"duel: {[n for n, _ in entries]} at {duel_games} "
                  f"games/pairing/deck", flush=True)
            record["duel"] = run_duel(runner, decks, entries, duel_games)
            print(format_duel(record["duel"]), flush=True)
    finally:
        runner.close()

    record["elapsed_s"] = round(time.time() - started, 1)
    record["workers"] = args.workers
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"wrote {out_path} in {record['elapsed_s']}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    import ptcg.creation  # noqa: F401 — engine bootstrap
    raise SystemExit(main())
