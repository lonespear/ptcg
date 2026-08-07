"""Portfolio selection: a stable of decks with decorrelated weaknesses (D30).

The GA's per-deck fitness (D18: play-weighted win rate vs the specialist
panel minus the termination-mode penalty) says nothing about how the stable
fails as a whole. D30's objective is a ~100-deck stable whose members lose
to DIFFERENT parts of the field, so a ladder opponent that beats one member
has learned little about the next random draw.

Weakness vector: per panel entry, the loss rate 1 - win_rate — the matchup
profile every archive row already stores. Redundancy between two decks is
the Pearson correlation of their weakness vectors: 1.0 means they fold to
exactly the same opponents.

Selection is greedy max-diversity (the marginal-gain rule): seed with the
fittest deck, then repeatedly add

    argmax over remaining d of  fitness(d) - gamma * max corr(d, selected)

The max (not mean) over the selected set is deliberate: a candidate is
penalised by its closest twin already in the stable, so pockets of clones
cannot amortise each other away.
"""

from __future__ import annotations

import json
import math
from pathlib import Path


def weakness(profile: list[float]) -> list[float]:
    return [1.0 - w for w in profile]


def pearson(a: list[float], b: list[float]) -> float:
    """Correlation of two vectors; 0.0 when either has no variance
    (a deck that never loses shares weaknesses with nobody)."""
    n = len(a)
    if n == 0 or n != len(b):
        return 0.0
    ma, mb = sum(a) / n, sum(b) / n
    da = [x - ma for x in a]
    db = [x - mb for x in b]
    sa = math.sqrt(sum(x * x for x in da))
    sb = math.sqrt(sum(x * x for x in db))
    if sa < 1e-9 or sb < 1e-9:
        return 0.0
    return sum(x * y for x, y in zip(da, db)) / (sa * sb)


def load_archive(paths: list[Path]) -> list[dict]:
    """Merge archive.jsonl files (multi-machine) into unique candidates.

    Rows: {"k": sorted deck, "f": fitness, "s": raw score, "x": exhaustion
    fraction, "p": per-panel win rates}. Rows without a profile (pre-profile
    checkpoints) are dropped — no weakness vector, no seat in the stable.
    On a duplicate deck the higher-fitness row wins (different machines run
    different game counts; more optimistic ≈ more games at these sizes is
    not guaranteed, but the disagreement is noise either way).
    """
    best: dict[tuple, dict] = {}
    for path in paths:
        with Path(path).open() as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not rec.get("p"):
                    continue
                key = tuple(rec["k"])
                if key not in best or rec["f"] > best[key]["f"]:
                    best[key] = rec
    return [{"deck": list(k), "fitness": r["f"], "score": r["s"],
             "exhaustion": r["x"], "profile": r["p"]}
            for k, r in best.items()]


def select_stable(candidates: list[dict], size: int = 100,
                  fitness_floor: float | None = None,
                  gamma: float = 0.1) -> dict:
    """Greedy max-diversity stable from an evaluated-deck archive.

    candidates: dicts with deck/fitness/profile (load_archive's shape).
    fitness_floor: drop candidates below it; None keeps everything and
    lets gamma do the work.
    Returns {members, summary} where each member carries its selection-time
    max correlation to the rest of the stable.
    """
    pool = [c for c in candidates
            if fitness_floor is None or c["fitness"] >= fitness_floor]
    pool.sort(key=lambda c: -c["fitness"])
    if not pool:
        return {"members": [], "summary": {"n": 0}}

    weak = [weakness(c["profile"]) for c in pool]
    selected: list[int] = [0]                    # fittest seeds the stable
    max_corr = [pearson(w, weak[0]) for w in weak]
    max_corr[0] = 1.0
    picked = {0}
    order_stats = [(pool[0]["fitness"], 0.0)]

    while len(selected) < min(size, len(pool)):
        best_i, best_gain = None, -float("inf")
        for i in range(len(pool)):
            if i in picked:
                continue
            gain = pool[i]["fitness"] - gamma * max_corr[i]
            if gain > best_gain:
                best_i, best_gain = i, gain
        if best_i is None:
            break
        selected.append(best_i)
        picked.add(best_i)
        order_stats.append((pool[best_i]["fitness"], max_corr[best_i]))
        for i in range(len(pool)):               # one pass keeps max updated
            if i not in picked:
                c = pearson(weak[i], weak[best_i])
                if c > max_corr[i]:
                    max_corr[i] = c

    members = []
    for rank, (i, (f, corr_at_pick)) in enumerate(zip(selected, order_stats)):
        m = dict(pool[i])
        m["rank"] = rank
        m["max_corr_at_selection"] = round(corr_at_pick, 4)
        members.append(m)

    sel_weak = [weak[i] for i in selected]
    pair_corrs = [pearson(sel_weak[i], sel_weak[j])
                  for i in range(len(sel_weak))
                  for j in range(i + 1, len(sel_weak))]
    summary = {
        "n": len(members),
        "gamma": gamma,
        "fitness_floor": fitness_floor,
        "candidates": len(pool),
        "fitness_mean": round(sum(m["fitness"] for m in members)
                              / len(members), 4),
        "fitness_min": round(min(m["fitness"] for m in members), 4),
        "fitness_max": round(max(m["fitness"] for m in members), 4),
        "pairwise_corr_mean": round(sum(pair_corrs) / len(pair_corrs), 4)
        if pair_corrs else None,
        "pairwise_corr_max": round(max(pair_corrs), 4) if pair_corrs else None,
    }
    return {"members": members, "summary": summary}
