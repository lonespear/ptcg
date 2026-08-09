"""Advantage inference from mined ladder games (D30, component 1).

Three analyses over `data/mined/<date>/`, all deterministic, all with Wald CIs:

  A. ADVANTAGE DECOMPOSITION. At turn checkpoints t = 3, 5, 7, 9, reconstruct
     each seat's state from the decision rows and regress the eventual result on
     the interpretable components. Two fits per checkpoint:
       A1  full corpus, proxy components rebuilt from the decision stream
           (decisions.parquet carries prizes but no board state, so Energy,
           bench and hand arrive as cumulative-action proxies);
       A2  the positions sample, where the full observation gives the exact
           components the agent's evaluator scores -- prize, board HP, Energy,
           hand, empty Active -- so the fitted vector is directly comparable to
           the hand-set WEIGHTS in agent/main.py.
     A2 pooled over turns is the mapping onto the evaluator scale, because the
     evaluator itself is turn-independent; per-checkpoint fits are reported
     beside it and never averaged into it.

  B. COMEBACK. Among seats behind the fitted index at t = 5 and t = 7, compare
     the behaviour of eventual winners against eventual losers over the next
     four turns, inside matched deficit bins.

  C. OPPONENT WIN / LOSE CONDITIONS. Per top-8 archetype: how their wins end
     against how their losses end, turns to each, and which components their
     own advantage rides on (per-archetype refit of A1).

Sampling rules that hold throughout: the focal seat must be rated >= 1000 (the
one-sided filter), and exactly one seat per episode enters any regression --
the two seats of one game carry complementary outcomes and mirrored
differentials, so keeping both would halve the standard errors for free.

    python -m ptcg.advantage --out data/analysis
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MINED = ROOT / "data" / "mined"

CHECKPOINTS = (3, 5, 7, 9)
# At an odd global turn the player who went first is the one mid-turn, so the
# seats are one part-turn out of step and `went_first` absorbs that offset.
# The even checkpoints put both seats on the same number of completed turns and
# are the honest read on seat order.
EVEN_CHECKPOINTS = (4, 6, 8, 10)
RATING_CUT = 1000.0
Z95 = 1.959964

# The hand-set evaluator vector this analysis is measured against
# (agent/main.py WEIGHTS).
HAND_SET = {"prize": 1000.0, "hp": 1.0, "energy": 30.0, "hand": 5.0,
            "no_active": 4000.0}

ACTION_TYPES = ["attach", "play", "evolve", "ability", "attack", "retreat",
                "card", "end_turn", "energy", "discard", "yes", "no", "number"]


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------

def norm_sf(z: float) -> float:
    """Two-sided normal tail, without scipy."""
    return math.erfc(abs(z) / math.sqrt(2.0))


def logit_fit(X: np.ndarray, y: np.ndarray, l2: float = 1e-6,
              max_iter: int = 200) -> dict:
    """Newton-Raphson logistic MLE with Wald standard errors.

    A whisper of ridge (1e-6) keeps the Hessian invertible under near-separation;
    it is four orders below the smallest information any live column carries, so
    the estimates are the MLE to reporting precision.
    """
    n, k = X.shape
    beta = np.zeros(k)
    for _ in range(max_iter):
        eta = np.clip(X @ beta, -35, 35)
        p = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(p * (1 - p), 1e-9, None)
        grad = X.T @ (y - p) - l2 * beta
        H = (X.T * w) @ X + l2 * np.eye(k)
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, grad, rcond=None)[0]
        # step halving keeps a wild first Newton step from overshooting
        for _ in range(20):
            cand = beta + step
            eta_c = np.clip(X @ cand, -35, 35)
            ll_c = float(np.sum(y * eta_c - np.log1p(np.exp(eta_c))))
            eta_b = np.clip(X @ beta, -35, 35)
            ll_b = float(np.sum(y * eta_b - np.log1p(np.exp(eta_b))))
            if ll_c >= ll_b - 1e-12:
                break
            step = step / 2.0
        beta = beta + step
        if np.max(np.abs(step)) < 1e-9:
            break
    eta = np.clip(X @ beta, -35, 35)
    p = 1.0 / (1.0 + np.exp(-eta))
    w = np.clip(p * (1 - p), 1e-9, None)
    H = (X.T * w) @ X + l2 * np.eye(k)
    cov = np.linalg.inv(H)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    ll = float(np.sum(y * eta - np.log1p(np.exp(eta))))
    ybar = float(y.mean())
    ll0 = float(n * (ybar * math.log(ybar) + (1 - ybar) * math.log(1 - ybar)))
    return {"beta": beta, "se": se, "p_hat": p, "loglik": ll, "loglik_null": ll0,
            "mcfadden_r2": 1.0 - ll / ll0 if ll0 else float("nan"),
            "mean_pq": float(np.mean(p * (1 - p))), "n": int(n)}


def coef_table(fit: dict, names: list[str], X: np.ndarray) -> list[dict]:
    """Coefficients with 95% CIs and average marginal effects in win-probability."""
    out = []
    mpq = fit["mean_pq"]
    for i, nm in enumerate(names):
        b, s = float(fit["beta"][i]), float(fit["se"][i])
        sd = float(np.std(X[:, i])) if nm != "const" else 0.0
        z = b / s if s > 0 else float("nan")
        out.append({
            "term": nm,
            "beta": round(b, 6), "se": round(s, 6),
            "ci_lo": round(b - Z95 * s, 6), "ci_hi": round(b + Z95 * s, 6),
            "z": round(z, 3), "p_value": round(norm_sf(z), 6) if s > 0 else None,
            "odds_ratio": round(math.exp(b), 4) if abs(b) < 30 else None,
            "ame_pp_per_unit": round(100 * b * mpq, 4),
            "ame_pp_per_sd": round(100 * b * mpq * sd, 4),
            "sd_x": round(sd, 4),
            "sig": bool(s > 0 and abs(z) > Z95),
        })
    return out


def prop_diff_ci(k1: int, n1: int, k2: int, n2: int) -> dict:
    """Winner-minus-loser difference in a rate, Wald 95%."""
    if n1 == 0 or n2 == 0:
        return {"diff": None, "ci_lo": None, "ci_hi": None}
    p1, p2 = k1 / n1, k2 / n2
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    return {"p_win": round(p1, 4), "p_lose": round(p2, 4),
            "diff": round(p1 - p2, 4),
            "ci_lo": round(p1 - p2 - Z95 * se, 4),
            "ci_hi": round(p1 - p2 + Z95 * se, 4),
            "sig": bool(abs(p1 - p2) > Z95 * se)}


def mean_diff_ci(a: np.ndarray, b: np.ndarray) -> dict:
    """Welch difference in means, winners minus losers."""
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return {"diff": None}
    se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    d = float(a.mean() - b.mean())
    return {"mean_win": round(float(a.mean()), 4),
            "mean_lose": round(float(b.mean()), 4),
            "n_win": int(len(a)), "n_lose": int(len(b)),
            "diff": round(d, 4),
            "ci_lo": round(d - Z95 * se, 4), "ci_hi": round(d + Z95 * se, 4),
            "sig": bool(se > 0 and abs(d) > Z95 * se)}


# --------------------------------------------------------------------------
# corpus
# --------------------------------------------------------------------------

def load_decisions(dates: list[str] | None) -> pd.DataFrame:
    paths = sorted(MINED.glob("*/decisions.parquet"))
    if dates:
        paths = [p for p in paths if p.parent.name in dates]
    if not paths:
        raise SystemExit(f"no decisions.parquet under {MINED}")
    cols = ["episode_id", "date", "seat", "agent_rating", "turn", "step",
            "select_type_name", "n_options", "n_chosen", "max_count",
            "chosen_option_type_name", "our_archetype", "opp_archetype",
            "prizes_mine", "prizes_theirs", "went_first", "won"]
    frames = []
    for p in paths:
        d = pd.read_parquet(p, columns=cols)
        d["episode_id"] = d["episode_id"].astype("int64")
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    # A decided game and a known first player are preconditions for every fit.
    df = df[(df.won >= 0) & (df.went_first >= 0)]
    return df


def seat_base(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (episode, seat): outcome, rating, archetypes, seat order."""
    g = df.groupby(["episode_id", "seat"], sort=False)
    base = g.agg(won=("won", "first"), rating=("agent_rating", "first"),
                 went_first=("went_first", "first"),
                 our_archetype=("our_archetype", "first"),
                 opp_archetype=("opp_archetype", "first"),
                 date=("date", "first"),
                 max_turn=("turn", "max")).reset_index()
    return base


def checkpoint_features(df: pd.DataFrame, base: pd.DataFrame, t: int) -> pd.DataFrame:
    """Per-seat state at the START of global turn t, rebuilt from the stream.

    Everything is counted over completed turns strictly before t. At an odd t
    that is (t-1)/2 turns for each seat, so the two sides are turn-matched and
    the differentials mean what they say; had the in-progress turn been included
    the seat acting at t would carry a part-turn of extra actions and
    `went_first` would absorb the offset.

    Components available: prize differential (recorded on every row) plus the
    cumulative action counts through turn t, which stand in for board state --
    Energy attachments made, cards played, evolutions, ability activations,
    attacks launched, retreats -- and the mean option count on the seat's most
    recent MAIN menu, which moves with hand size and board width.
    """
    sub = df[(df.turn > 0) & (df.turn < t)]
    if sub.empty:
        return pd.DataFrame()

    # cumulative action counts per seat
    counts = (sub.groupby(["episode_id", "seat", "chosen_option_type_name"])
                 .size().unstack(fill_value=0))
    for c in ACTION_TYPES:
        if c not in counts.columns:
            counts[c] = 0
    counts = counts[ACTION_TYPES].astype("float64")
    counts["n_decisions"] = counts.sum(axis=1)
    counts = counts.reset_index()

    # own turns taken by turn t
    turns_taken = (sub.groupby(["episode_id", "seat"]).turn.nunique()
                      .rename("turns_taken").reset_index())

    # menu breadth on the seat's most recent MAIN menu at or before t
    m = sub[sub.select_type_name == "main"]
    if len(m):
        last_turn = m.groupby(["episode_id", "seat"]).turn.transform("max")
        m = m[m.turn == last_turn]
        breadth = (m.groupby(["episode_id", "seat"]).n_options.mean()
                    .rename("breadth").reset_index())
    else:
        breadth = pd.DataFrame(columns=["episode_id", "seat", "breadth"])

    # prize state: the latest row at or before t, mapped onto both seats
    last = sub.sort_values("step").groupby("episode_id").tail(1)
    pz = []
    for ep, s, pm, pt_ in zip(last.episode_id, last.seat,
                              last.prizes_mine, last.prizes_theirs):
        pz.append((ep, int(s), int(pm)))
        pz.append((ep, 1 - int(s), int(pt_)))
    prizes = pd.DataFrame(pz, columns=["episode_id", "seat", "prizes_left"])

    reached = base[base.max_turn >= t][["episode_id", "seat"]]
    f = (reached.merge(counts, on=["episode_id", "seat"], how="left")
                .merge(turns_taken, on=["episode_id", "seat"], how="left")
                .merge(breadth, on=["episode_id", "seat"], how="left")
                .merge(prizes, on=["episode_id", "seat"], how="left"))
    num = [c for c in f.columns if c not in ("episode_id", "seat", "breadth",
                                             "prizes_left")]
    f[num] = f[num].fillna(0.0)
    f = f.merge(base, on=["episode_id", "seat"], how="left")

    # mirror onto the opponent
    opp = f.copy()
    opp["seat"] = 1 - opp["seat"]
    keep = ["episode_id", "seat", "prizes_left", "breadth", "turns_taken",
            "n_decisions"] + ACTION_TYPES
    opp = opp[keep].rename(columns={c: c + "_opp" for c in keep
                                    if c not in ("episode_id", "seat")})
    f = f.merge(opp, on=["episode_id", "seat"], how="inner")

    f["prize_diff"] = f["prizes_left_opp"] - f["prizes_left"]
    # attachment tempo, not Energy in play: the proxy counts attachments made,
    # and a knockout takes the Energy with it (see proxy_validation)
    f["attach_diff"] = f["attach"] - f["attach_opp"]
    f["board_diff"] = f["play"] - f["play_opp"]
    f["evolve_diff"] = f["evolve"] - f["evolve_opp"]
    f["ability_diff"] = f["ability"] - f["ability_opp"]
    f["attack_diff"] = f["attack"] - f["attack_opp"]
    f["retreat_diff"] = f["retreat"] - f["retreat_opp"]
    f["breadth_diff"] = f["breadth"].fillna(f["breadth"].median()) - \
                        f["breadth_opp"].fillna(f["breadth"].median())
    f["cardsel_diff"] = f["card"] - f["card_opp"]
    f["turn_checkpoint"] = t
    return f


def focal_rows(f: pd.DataFrame) -> pd.DataFrame:
    """One rated seat per episode: the one-sided >= 1000 filter, de-duplicated.

    Where both seats clear the cut, episode-id parity picks which one enters --
    deterministic, and uncorrelated with anything in the model.
    """
    q = f[f.rating >= RATING_CUT].copy()
    both = q.groupby("episode_id").seat.transform("size") > 1
    pick = (~both) | (q.seat == (q.episode_id % 2))
    return q[pick].copy()


def design(f: pd.DataFrame, terms: list[str], arch_levels: list[str],
           opp_fe: bool = True) -> tuple[np.ndarray, list[str]]:
    """Model matrix: intercept, components, seat order, archetype fixed effects."""
    cols = [np.ones(len(f))]
    names = ["const"]
    dropped = []
    for t in terms:
        v = f[t].to_numpy(dtype=float)
        if np.std(v) == 0:          # no variation at this checkpoint
            dropped.append(t)
            continue
        cols.append(v)
        names.append(t)
    base_arch = arch_levels[0]
    for a in arch_levels[1:]:
        cols.append((f.our_archetype == a).to_numpy(dtype=float))
        names.append(f"ours={a}")
    if opp_fe:
        for a in arch_levels[1:]:
            cols.append((f.opp_archetype == a).to_numpy(dtype=float))
            names.append(f"theirs={a}")
    X = np.column_stack(cols)
    return X, names, dropped


# --------------------------------------------------------------------------
# A2 -- exact components from the positions sample
# --------------------------------------------------------------------------

def load_positions(dates: list[str] | None) -> pd.DataFrame:
    paths = sorted(MINED.glob("*/positions.jsonl.gz"))
    if dates:
        paths = [p for p in paths if p.parent.name in dates]
    rows = []
    for p in paths:
        with gzip.open(p, "rt", encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                obs = r.get("observation") or {}
                cur = obs.get("current") or {}
                pl = cur.get("players") or []
                if len(pl) != 2 or r.get("won", -1) < 0:
                    continue
                me = cur.get("yourIndex")
                if me not in (0, 1):
                    me = r.get("seat", 0)
                mine, theirs = pl[me], pl[1 - me]

                def side(pp):
                    hp = ener = mons = dmg = 0.0
                    for z in ("active", "bench"):
                        for mon in pp.get(z) or []:
                            if not mon:
                                continue
                            mons += 1
                            hp += float(mon.get("hp") or 0)
                            dmg += float((mon.get("maxHp") or 0) - (mon.get("hp") or 0))
                            ener += len(mon.get("energies") or [])
                    return hp, ener, mons, dmg

                hp_m, en_m, mon_m, dm_m = side(mine)
                hp_t, en_t, mon_t, dm_t = side(theirs)
                rows.append({
                    "episode_id": int(r["episode_id"]), "seat": int(r["seat"]),
                    "date": r.get("date"), "turn": int(r.get("turn", -1)),
                    "rating": r.get("agent_rating"),
                    "our_archetype": r.get("our_archetype"),
                    "opp_archetype": r.get("opp_archetype"),
                    "went_first": int(r.get("went_first", -1)),
                    "won": int(r["won"]),
                    "prize_diff": len(theirs.get("prize") or []) - len(mine.get("prize") or []),
                    "hp_diff": hp_m - hp_t,
                    "energy_diff": en_m - en_t,
                    "hand_me": float(mine.get("handCount") or 0),
                    "hand_diff": float(mine.get("handCount") or 0) - float(theirs.get("handCount") or 0),
                    "bench_diff": len(mine.get("bench") or []) - len(theirs.get("bench") or []),
                    "deck_diff": float(mine.get("deckCount") or 0) - float(theirs.get("deckCount") or 0),
                    "damage_diff": dm_t - dm_m,
                    "no_active_me": 0.0 if (mine.get("active") or []) else 1.0,
                    "no_active_opp": 0.0 if (theirs.get("active") or []) else 1.0,
                })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# analyses
# --------------------------------------------------------------------------

A1_TERMS = ["prize_diff", "attach_diff", "board_diff", "evolve_diff",
            "ability_diff", "attack_diff", "retreat_diff", "breadth_diff",
            "cardsel_diff", "went_first"]


def analysis_A(df: pd.DataFrame, base: pd.DataFrame, pos: pd.DataFrame,
               arch_levels: list[str]) -> tuple[dict, dict]:
    terms = A1_TERMS
    out = {"spec": {"terms": terms, "rating_cut": RATING_CUT,
                    "one_seat_per_episode": True,
                    "archetype_fixed_effects": arch_levels,
                    "checkpoint": ("state at the start of turn t: every count is "
                                   "over completed turns strictly before t"),
                    "parity_note": (
                        "odd checkpoints turn-match the seats ((t-1)/2 completed "
                        "turns each) and are the reported fits; at an even "
                        "checkpoint the first player has one completed turn more, "
                        "so went_first there absorbs the offset and flips sign -- "
                        "that contrast is the diagnostic, not a finding")},
           "checkpoints": {}, "even_checkpoints": {}}
    feats: dict[int, pd.DataFrame] = {}
    for t in tuple(CHECKPOINTS) + tuple(EVEN_CHECKPOINTS):
        f = checkpoint_features(df, base, t)
        feats[t] = f
        q = focal_rows(f)
        X, names, dropped = design(q, terms, arch_levels)
        y = q.won.to_numpy(dtype=float)
        fit = logit_fit(X, y)
        q = q.assign(p_hat=fit["p_hat"])
        feats[t] = f.merge(q[["episode_id", "seat", "p_hat"]],
                           on=["episode_id", "seat"], how="left")
        corr = q[terms].corr().round(3)
        slot = "checkpoints" if t in CHECKPOINTS else "even_checkpoints"
        out[slot][str(t)] = {
            "n_seats_modeled": int(len(q)),
            "n_episodes_reaching_turn": int(f.episode_id.nunique()),
            "n_seats_rated_at_cut": int((f.rating >= RATING_CUT).sum()),
            "base_rate_win": round(float(y.mean()), 4),
            "first_player_win_rate": round(float(q[q.went_first == 1].won.mean()), 4),
            "n_went_first": int((q.went_first == 1).sum()),
            "mcfadden_r2": round(fit["mcfadden_r2"], 4),
            "dropped_no_variation": dropped,
            "coefficients": coef_table(fit, names, X),
            "component_correlations": corr.to_dict(),
            "component_means": {k: round(float(q[k].mean()), 3) for k in terms},
        }
    out["_feats"] = feats

    # A2: exact evaluator components on the positions sample
    p = pos[(pos.rating >= RATING_CUT) & (pos.turn > 0) & (pos.went_first >= 0)].copy()
    p = p.sort_values(["episode_id", "turn"]).drop_duplicates("episode_id")
    ev_terms = ["prize_diff", "hp_diff", "energy_diff", "hand_me"]
    rich_terms = ["prize_diff", "hp_diff", "energy_diff", "hand_me",
                  "bench_diff", "damage_diff", "deck_diff", "hand_diff"]
    a2 = {"spec": {"terms": ev_terms + ["no_active_me (dropped where constant)"],
                   "note": "exact evaluator components; one position per episode",
                   "rating_cut": RATING_CUT},
          "checkpoints": {}, "pooled": {}, "pooled_beyond_evaluator": {}}

    def fit_positions(sel: pd.DataFrame, turn_fe: bool,
                      ev_terms: list[str] = ev_terms) -> dict | None:
        if len(sel) < 60:
            return None
        cols = [np.ones(len(sel))]
        names = ["const"]
        for tm in ev_terms:
            cols.append(sel[tm].to_numpy(dtype=float))
            names.append(tm)
        if sel.no_active_me.std() > 0 and sel.no_active_me.sum() >= 10:
            cols.append(sel.no_active_me.to_numpy(dtype=float))
            names.append("no_active_me")
        cols.append(sel.went_first.to_numpy(dtype=float))
        names.append("went_first")
        if turn_fe:
            for tv in sorted(sel.turn.unique())[1:]:
                cols.append((sel.turn == tv).to_numpy(dtype=float))
                names.append(f"turn={tv}")
        X = np.column_stack(cols)
        y = sel.won.to_numpy(dtype=float)
        fit = logit_fit(X, y)
        tab = coef_table(fit, names, X)
        bp = next(r for r in tab if r["term"] == "prize_diff")
        scaled = {}
        for r in tab:
            if r["term"] in ("prize_diff", "hp_diff", "energy_diff", "hand_me",
                             "hand_diff", "bench_diff", "damage_diff",
                             "deck_diff", "no_active_me"):
                if bp["beta"] != 0:
                    lo = 1000.0 * r["ci_lo"] / bp["beta"]
                    hi = 1000.0 * r["ci_hi"] / bp["beta"]
                    scaled[r["term"]] = {
                        "inferred_weight": round(1000.0 * r["beta"] / bp["beta"], 2),
                        "ci_lo": round(min(lo, hi), 2), "ci_hi": round(max(lo, hi), 2),
                    }
        return {"n": int(len(sel)), "base_rate_win": round(float(y.mean()), 4),
                "mcfadden_r2": round(fit["mcfadden_r2"], 4),
                "coefficients": tab,
                "inferred_weights_prize_1000": scaled,
                "hand_set_weights": HAND_SET}

    for t in CHECKPOINTS:
        r = fit_positions(p[p.turn == t], turn_fe=False)
        a2["checkpoints"][str(t)] = r or {"n": int((p.turn == t).sum()),
                                          "note": "too few positions to fit"}
    pooled = p[(p.turn >= 3) & (p.turn <= 11)]
    a2["pooled"] = fit_positions(pooled, turn_fe=True) or {}
    a2["pooled"]["turn_window"] = "3-11, turn fixed effects"
    a2["pooled_beyond_evaluator"] = fit_positions(
        pooled, turn_fe=True, ev_terms=rich_terms) or {}
    a2["pooled_beyond_evaluator"]["note"] = (
        "adds the components the evaluator does not score: bench width, damage "
        "already on the opponent's board, deck remaining, and the symmetric "
        "hand differential")
    a2["turn_counts"] = {str(k): int(v) for k, v in
                         p.turn.value_counts().sort_index().items()}

    # proxy validation: cumulative attachments against Energy actually in play
    return out, a2


def validate_proxy(feats: dict, pos: pd.DataFrame) -> dict:
    """Do the decision-stream proxies track the true board state?"""
    out = {}
    for t in CHECKPOINTS:
        f = feats[t]
        p = pos[pos.turn == t][["episode_id", "seat", "energy_diff", "hp_diff",
                                "hand_me", "bench_diff", "prize_diff"]]
        j = f.merge(p, on=["episode_id", "seat"], how="inner",
                    suffixes=("_proxy", "_true"))
        if len(j) < 30:
            out[str(t)] = {"n": int(len(j)), "note": "too few joined positions"}
            continue

        def r(a, b):
            a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
            if a.std() == 0 or b.std() == 0:
                return None
            return round(float(np.corrcoef(a, b)[0, 1]), 3)

        out[str(t)] = {
            "n": int(len(j)),
            "r_attach_vs_energy_in_play": r(j.attach_diff, j.energy_diff),
            "r_play_vs_bench": r(j.board_diff, j.bench_diff),
            "r_breadth_vs_hand": r(j.breadth_diff.fillna(0), j.hand_me),
            "r_prize_check": r(j.prize_diff_proxy, j.prize_diff_true),
        }
    return out


BEHAVIOURS = ["attack", "attach", "play", "evolve", "ability", "retreat", "card"]


def window_behaviour(df: pd.DataFrame, t: int, span: int = 4) -> pd.DataFrame:
    """Per-seat action mix over turns t+1 .. t+span."""
    w = df[(df.turn > t) & (df.turn <= t + span)]
    if w.empty:
        return pd.DataFrame()
    counts = (w.groupby(["episode_id", "seat", "chosen_option_type_name"])
                .size().unstack(fill_value=0))
    for c in BEHAVIOURS:
        if c not in counts.columns:
            counts[c] = 0
    counts = counts[BEHAVIOURS].astype(float)
    counts["n_dec"] = (w.groupby(["episode_id", "seat"]).size()
                        .reindex(counts.index).astype(float))
    turns = w.groupby(["episode_id", "seat"]).turn.nunique().rename("own_turns")
    counts = counts.join(turns)
    multi = w[w.max_count > 1]
    if len(multi):
        ms = (multi.groupby(["episode_id", "seat"])
                   .apply(lambda g: float((g.n_chosen > 1).mean()),
                          include_groups=False).rename("multi_rate"))
        counts = counts.join(ms)
    else:
        counts["multi_rate"] = np.nan
    b = counts.reset_index()
    for c in BEHAVIOURS:
        b[f"{c}_per_turn"] = b[c] / b["own_turns"].replace(0, np.nan)
        b[f"{c}_share"] = b[c] / b["n_dec"].replace(0, np.nan)
    b["decisions_per_turn"] = b["n_dec"] / b["own_turns"].replace(0, np.nan)
    return b


def analysis_B(df: pd.DataFrame, feats: dict, arch_levels: list[str],
               checkpoints=(5, 7)) -> dict:
    out = {"spec": {
        "behind": "fitted P(win) from the A1 model at the checkpoint",
        "bins": "deep 0.05-0.30, shallow 0.30-0.45",
        "window": "turns t+1 .. t+4, rates per own turn and as decision shares",
        "next_turn_window": "the seat's single next turn, t+1 .. t+2",
        "rating_cut": RATING_CUT, "one_seat_per_episode": True},
        "checkpoints": {}}
    for t in checkpoints:
        f = feats[t]
        q = focal_rows(f)
        q = q[q.p_hat.notna()]
        beh = window_behaviour(df, t)
        if beh.empty:
            continue
        nxt = window_behaviour(df, t, span=2)
        if not nxt.empty:
            nxt = nxt[["episode_id", "seat"] +
                      [c for c in nxt.columns if c.endswith("_per_turn")]]
            nxt = nxt.rename(columns={c: "next_" + c for c in nxt.columns
                                      if c.endswith("_per_turn")})
            beh = beh.merge(nxt, on=["episode_id", "seat"], how="left")
        q = q.merge(beh, on=["episode_id", "seat"], how="inner")
        # survivorship control: the game must reach t+2 for either side to have
        # had a chance to act on the deficit
        q = q[q.max_turn >= t + 2]
        bins = {"deep_0.05_0.30": (0.05, 0.30), "shallow_0.30_0.45": (0.30, 0.45)}
        res = {}
        for bname, (lo, hi) in bins.items():
            b = q[(q.p_hat >= lo) & (q.p_hat < hi)]
            if len(b) < 100:
                res[bname] = {"n": int(len(b)), "note": "too few seats"}
                continue
            win, lose = b[b.won == 1], b[b.won == 0]
            metrics = {}
            for c in BEHAVIOURS:
                metrics[f"{c}_per_turn"] = mean_diff_ci(
                    win[f"{c}_per_turn"].to_numpy(dtype=float),
                    lose[f"{c}_per_turn"].to_numpy(dtype=float))
                metrics[f"{c}_share"] = mean_diff_ci(
                    win[f"{c}_share"].to_numpy(dtype=float),
                    lose[f"{c}_share"].to_numpy(dtype=float))
            for c in ("multi_rate", "decisions_per_turn", "own_turns"):
                metrics[c] = mean_diff_ci(win[c].to_numpy(dtype=float),
                                          lose[c].to_numpy(dtype=float))
            for c in [c for c in b.columns if c.startswith("next_")]:
                metrics[c.replace("_per_turn", "_rate")] = mean_diff_ci(
                    win[c].to_numpy(dtype=float), lose[c].to_numpy(dtype=float))
            # tighter conditioning: same deficit quintile, then quintile x archetype
            tight = {}
            b2 = b.copy()
            b2["dec"] = pd.qcut(b2.p_hat, 5, labels=False, duplicates="drop")
            for keys, tag in ((["dec"], "by_deficit"),
                              (["dec", "our_archetype"], "by_deficit_archetype")):
                for c in ("attack_per_turn", "attach_per_turn", "ability_per_turn",
                          "play_per_turn", "retreat_per_turn", "multi_rate",
                          "next_attack_per_turn", "next_attach_per_turn"):
                    if c not in b2.columns:
                        continue
                    diffs, ws = [], []
                    for _, g in b2.groupby(keys):
                        gw, gl = g[g.won == 1][c].dropna(), g[g.won == 0][c].dropna()
                        if len(gw) < 10 or len(gl) < 10:
                            continue
                        d = float(gw.mean() - gl.mean())
                        v = gw.var(ddof=1) / len(gw) + gl.var(ddof=1) / len(gl)
                        if v <= 0:
                            continue
                        diffs.append(d)
                        ws.append(1.0 / v)
                    if diffs:
                        ws = np.array(ws)
                        d = float(np.sum(np.array(diffs) * ws) / ws.sum())
                        se = float(math.sqrt(1.0 / ws.sum()))
                        tight[f"{c}::{tag}"] = {
                            "stratified_diff": round(d, 4),
                            "ci_lo": round(d - Z95 * se, 4),
                            "ci_hi": round(d + Z95 * se, 4),
                            "strata": len(diffs),
                            "sig": bool(abs(d) > Z95 * se)}
            per_arch = {}
            for a in arch_levels:
                ba = b[b.our_archetype == a]
                if (ba.won == 1).sum() < 30 or (ba.won == 0).sum() < 30:
                    continue
                per_arch[a] = {
                    "n": int(len(ba)),
                    "comeback_rate": round(float(ba.won.mean()), 4),
                    "attack_per_turn": mean_diff_ci(
                        ba[ba.won == 1].attack_per_turn.to_numpy(dtype=float),
                        ba[ba.won == 0].attack_per_turn.to_numpy(dtype=float)),
                    "attach_per_turn": mean_diff_ci(
                        ba[ba.won == 1].attach_per_turn.to_numpy(dtype=float),
                        ba[ba.won == 0].attach_per_turn.to_numpy(dtype=float)),
                    "ability_per_turn": mean_diff_ci(
                        ba[ba.won == 1].ability_per_turn.to_numpy(dtype=float),
                        ba[ba.won == 0].ability_per_turn.to_numpy(dtype=float)),
                }
            res[bname] = {"n": int(len(b)), "n_win": int((b.won == 1).sum()),
                          "n_lose": int((b.won == 0).sum()),
                          "comeback_rate": round(float(b.won.mean()), 4),
                          "metrics": metrics,
                          "stratified": tight,
                          "per_archetype": per_arch}

        # the tightest conditioning available: every behaviour entered at once,
        # on top of the position that produced the deficit
        behind = q[(q.p_hat >= 0.05) & (q.p_hat < 0.45)].copy()
        beh_terms = ["attack_per_turn", "attach_per_turn", "play_per_turn",
                     "evolve_per_turn", "ability_per_turn", "retreat_per_turn",
                     "decisions_per_turn", "own_turns"]
        state_terms = ["prize_diff", "attach_diff", "board_diff", "breadth_diff",
                       "cardsel_diff", "went_first"]
        sel = behind.dropna(subset=beh_terms + state_terms + ["p_hat"])
        if len(sel) >= 200:
            logit_p = np.log(np.clip(sel.p_hat, 1e-6, 1 - 1e-6) /
                             (1 - np.clip(sel.p_hat, 1e-6, 1 - 1e-6)))
            cols = [np.ones(len(sel))] + \
                   [sel[c].to_numpy(dtype=float) for c in beh_terms + state_terms] + \
                   [logit_p.to_numpy(dtype=float)]
            names = ["const"] + beh_terms + state_terms + ["logit_p_hat"]
            X = np.column_stack(cols)
            fit = logit_fit(X, sel.won.to_numpy(dtype=float))
            res["controlled_regression"] = {
                "n": int(len(sel)),
                "comeback_rate": round(float(sel.won.mean()), 4),
                "mcfadden_r2": round(fit["mcfadden_r2"], 4),
                "note": ("win on behaviour over turns t+1..t+4, controlling for "
                         "the position at t and for how many turns the window "
                         "actually contained"),
                "coefficients": coef_table(fit, names, X)}
        out["checkpoints"][str(t)] = res
    return out


def analysis_C(df: pd.DataFrame, base: pd.DataFrame, feats: dict,
               arch_levels: list[str]) -> dict:
    """Per-archetype termination modes and the components their wins ride on."""
    d = df[df.turn > 0]
    last = d.sort_values("step").groupby(["episode_id", "seat"]).tail(1)
    last = last[["episode_id", "seat", "turn", "prizes_mine", "prizes_theirs",
                 "our_archetype", "opp_archetype", "won", "agent_rating",
                 "went_first"]]
    rated = last[last.agent_rating >= RATING_CUT]

    def modes(sel: pd.DataFrame) -> dict:
        n = len(sel)
        if n == 0:
            return {"n": 0}
        pm = sel.prizes_mine
        return {
            "n": int(n),
            "prize_race_le2": round(float((pm <= 2).mean()), 4),
            "prizes_left_3": round(float((pm == 3).mean()), 4),
            "attrition_ge4": round(float((pm >= 4).mean()), 4),
            "untouched_6": round(float((pm == 6).mean()), 4),
            "median_prizes_left": float(pm.median()),
            "median_turn": float(sel.turn.median()),
            "p25_turn": float(sel.turn.quantile(0.25)),
            "p75_turn": float(sel.turn.quantile(0.75)),
        }

    out = {"spec": {
        "termination": ("decisions.parquet stops one decision short of the final "
                        "action, so the mode is read off the prize counts at the "
                        "last captured decision: <=2 prizes left for the winner is "
                        "a prize race closed by a one- or two-prize knockout, "
                        ">=4 left means the game ended some other way (deck-out, "
                        "no Pokemon to promote, or a timeout)"),
        "rating_cut": RATING_CUT},
        "archetypes": {}}

    terms = A1_TERMS
    for a in arch_levels:
        sel = rated[rated.our_archetype == a]
        wins, losses = sel[sel.won == 1], sel[sel.won == 0]
        entry = {
            "n_seats": int(len(sel)),
            "win_rate": round(float(sel.won.mean()), 4) if len(sel) else None,
            "wins": modes(wins), "losses": modes(losses),
            "first_player_rate": round(float(sel.went_first.mean()), 4),
        }
        # what their advantage rides on, at t=7 (widest checkpoint with depth)
        for t in (5, 7):
            f = feats[t]
            q = focal_rows(f)
            q = q[q.our_archetype == a]
            if len(q) < 200 or q.won.nunique() < 2:
                entry[f"regression_t{t}"] = {"n": int(len(q)),
                                             "note": "too few seats to fit"}
                continue
            cols = [np.ones(len(q))] + [q[tm].to_numpy(dtype=float) for tm in terms]
            names = ["const"] + terms
            X = np.column_stack(cols)
            fit = logit_fit(X, q.won.to_numpy(dtype=float))
            entry[f"regression_t{t}"] = {
                "n": int(len(q)),
                "mcfadden_r2": round(fit["mcfadden_r2"], 4),
                "coefficients": coef_table(fit, names, X),
            }
        # matchup profile: win rate against each opposing archetype
        mm = {}
        for o in arch_levels:
            s2 = sel[sel.opp_archetype == o]
            if len(s2) >= 40:
                p = float(s2.won.mean())
                se = math.sqrt(max(p * (1 - p), 1e-9) / len(s2))
                mm[o] = {"n": int(len(s2)), "win_rate": round(p, 4),
                         "ci_lo": round(p - Z95 * se, 4),
                         "ci_hi": round(p + Z95 * se, 4)}
        entry["matchups"] = mm
        entry["one_liner"] = one_liner(a, entry)
        out["archetypes"][a] = entry
    return out


PLAIN = {
    "prize_diff": "the early prize lead",
    "attach_diff": "their Energy attachment tempo",
    "board_diff": "their bench development",
    "evolve_diff": "their evolution line",
    "ability_diff": "their ability activations",
    "attack_diff": "their attack count",
    "retreat_diff": "their retreats",
    "breadth_diff": "their option breadth",
    "cardsel_diff": "their draw and search volume",
}


def one_liner(name: str, e: dict) -> str:
    """They win by X, they lose by Y, deny Z -- with the numbers attached."""
    wn, ls = e["wins"], e["losses"]
    if not wn.get("n"):
        return ""
    race = wn["prize_race_le2"]
    win_mode = (f"closing the prize race ({race:.0%} of wins) by turn "
                f"{wn['median_turn']:.0f}")
    if wn["attrition_ge4"] >= 0.15:
        win_mode += (f", and {wn['attrition_ge4']:.0%} of the time by grinding the "
                     f"game past the prizes entirely")
    lose_mode = (f"getting raced: {ls['median_prizes_left']:.0f} prizes still on "
                 f"their side at turn {ls['median_turn']:.0f}")
    deny = None
    reg = e.get("regression_t7") or {}
    for r in sorted(reg.get("coefficients", []),
                    key=lambda r: -abs(r["ame_pp_per_sd"])):
        if (r["sig"] and r["beta"] > 0 and r["term"] in PLAIN
                and r["term"] != "prize_diff"):
            deny = f"{PLAIN[r['term']]} ({r['ame_pp_per_sd']:+.1f} pp per SD to them)"
            break
    if deny is None:
        deny = "the prize lead itself -- nothing else clears 95% for this deck"
    return (f"{name}: they win by {win_mode}; they lose by {lose_mode}; "
            f"deny {deny}.")


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def fmt_ci(r: dict, key="beta") -> str:
    return f"{r[key]:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]"


def write_report(path: Path, A: dict, A2: dict, val: dict, B: dict, C: dict,
                 meta: dict) -> None:
    L = []
    w = L.append
    w("# Advantage inference from mined ladder games")
    w("")
    w(f"Corpus: {meta['dates']} — {meta['episodes']:,} decided episodes, "
      f"{meta['rows']:,} decision rows, {meta['positions']:,} full positions. "
      f"Focal seat rated >= {int(RATING_CUT)}; one seat per episode in every fit.")
    w("")
    w("## A. Advantage decomposition")
    w("")
    w("### A1 — full corpus, decision-stream components")
    w("")
    w("A checkpoint is the state at the start of turn t. The prize differential "
      "is recorded on every decision row; the rest are cumulative actions over "
      "the completed turns before t (Energy attachments, cards played, "
      "evolutions, abilities, attacks, retreats, card selections) plus the mean "
      "option count on the seat's most recent MAIN menu. At odd t the two seats "
      "have completed the same number of turns. Every term is a differential, ours minus "
      "theirs. Coefficients are log-odds of the eventual win; AME is the average "
      "marginal effect in percentage points of win probability per unit. "
      "Archetype fixed effects for both seats are fitted and suppressed here.")
    w("")

    def cp_table(c):
        w(f"**t = {c['_t']}** — n = {c['n_seats_modeled']:,} seats "
          f"({c['n_episodes_reaching_turn']:,} episodes reached the turn), "
          f"win base rate {c['base_rate_win']:.3f}, McFadden R² {c['mcfadden_r2']:.3f}")
        w("")
        w("| component | beta [95% CI] | z | AME pp/unit | AME pp/SD |")
        w("|---|---|---|---|---|")
        for r in c["coefficients"]:
            if r["term"] == "const" or r["term"].startswith(("ours=", "theirs=")):
                continue
            w(f"| {r['term']} | {fmt_ci(r)} | {r['z']:+.1f} | "
              f"{r['ame_pp_per_unit']:+.2f} | {r['ame_pp_per_sd']:+.2f} |")
        w("")

    for t in CHECKPOINTS:
        c = dict(A["checkpoints"][str(t)], _t=t)
        cp_table(c)
    w("**Seat order.** The first player wins "
      f"{meta['first_player_win_rate']:.3f} of decided games; among seats rated "
      f"at the cut it is {meta['rated_seat_win_rate_going_first']:.3f} going "
      f"first against {meta['rated_seat_win_rate_going_second']:.3f} going "
      "second. Those are the numbers to quote. The went_first coefficient in the tables above is "
      "not that quantity: the action counts it is conditioned on are themselves "
      "downstream of seat order, and the first player's opening turn is the "
      "restricted one, so matching on counts hands the first player credit for "
      "reaching them short-handed.")
    w("")
    w("**Parity diagnostic.** " + A["spec"]["parity_note"] + ". The even "
      "checkpoints, where the first player has completed one turn more:")
    w("")
    for t in EVEN_CHECKPOINTS:
        c = dict(A["even_checkpoints"][str(t)], _t=t)
        cp_table(c)
    w("### A2 — positions sample, exact evaluator components")
    w("")
    w("The 2,000-positions-per-day reservoir carries the full observation, so "
      "these are the quantities the agent's evaluator actually scores. The "
      "pooled fit (turns 3-11, turn fixed effects) is the mapping onto the "
      "evaluator scale, because the evaluator is turn-independent by "
      "construction; the per-checkpoint fits sit beside it, not inside it.")
    w("")
    pooled = A2.get("pooled") or {}
    if pooled.get("coefficients"):
        w(f"Pooled: n = {pooled['n']:,}, McFadden R² {pooled['mcfadden_r2']:.3f}")
        w("")
        w("| component | beta [95% CI] | z | AME pp/unit |")
        w("|---|---|---|---|")
        for r in pooled["coefficients"]:
            if r["term"] == "const" or r["term"].startswith("turn="):
                continue
            w(f"| {r['term']} | {fmt_ci(r)} | {r['z']:+.1f} | {r['ame_pp_per_unit']:+.3f} |")
        w("")
        w("**Inferred weight vector against the hand-set one** (prize anchored at 1000):")
        w("")
        w("| component | inferred [95% CI] | hand-set |")
        w("|---|---|---|")
        for k, v in pooled["inferred_weights_prize_1000"].items():
            hs = HAND_SET.get(k.replace("_me", "").replace("_diff", ""), "")
            w(f"| {k} | {v['inferred_weight']:.1f} [{v['ci_lo']:.1f}, {v['ci_hi']:.1f}] | {hs} |")
        w("")
        w("The empty-Active row is the one not to act on. The hand-set 4,000 "
          "encodes a lost game; an empty Active Spot in a sampled position is "
          "almost always a promote prompt mid-turn, which is transient, so the "
          "fitted coefficient measures something else entirely and its interval "
          "is wide enough to say so.")
        w("")
    for t in CHECKPOINTS:
        c = A2["checkpoints"][str(t)]
        if not c.get("coefficients"):
            w(f"- t = {t}: n = {c.get('n', 0)} positions — {c.get('note', '')}")
            continue
        iw = c["inferred_weights_prize_1000"]
        w(f"- t = {t}: n = {c['n']} — " + ", ".join(
            f"{k} {v['inferred_weight']:.0f} [{v['ci_lo']:.0f}, {v['ci_hi']:.0f}]"
            for k, v in iw.items()))
    w("")
    rich = A2.get("pooled_beyond_evaluator") or {}
    if rich.get("coefficients"):
        w("**What the evaluator does not score.** The same pooled sample with "
          "bench width, damage already dealt, deck remaining and the symmetric "
          f"hand differential added (n = {rich['n']:,}, "
          f"McFadden R² {rich['mcfadden_r2']:.3f}):")
        w("")
        w("| component | beta [95% CI] | z | AME pp/unit |")
        w("|---|---|---|---|")
        for r in rich["coefficients"]:
            if r["term"] == "const" or r["term"].startswith("turn="):
                continue
            w(f"| {r['term']} | {fmt_ci(r)} | {r['z']:+.1f} | {r['ame_pp_per_unit']:+.3f} |")
        w("")
        w("Bench width, damage already on their board, and the hand differential "
          "each carry advantage the evaluator does not score, and the own-hand "
          "term turns over once the hand differential is in the model -- what "
          "pays is holding more cards than they do, not holding cards.")
        w("")
    w("### Proxy validation")
    w("")
    w("Where a mined position and a reconstructed checkpoint describe the same "
      "seat at the same turn, the proxies can be scored against the truth.")
    w("")
    for t, v in val.items():
        if "r_attach_vs_energy_in_play" in v:
            w(f"- t = {t} (n = {v['n']}): attachments vs Energy in play "
              f"r = {v['r_attach_vs_energy_in_play']}, cards played vs bench "
              f"r = {v['r_play_vs_bench']}, menu breadth vs hand "
              f"r = {v['r_breadth_vs_hand']}, prize differential (identity check) "
              f"r = {v['r_prize_check']}")
        else:
            w(f"- t = {t}: n = {v['n']} — {v.get('note', '')}")
    w("")
    w("Read the attachment row carefully: cumulative attachments do not measure "
      "Energy in play, because a knockout takes the attached Energy with it. "
      "The A1 term is attachment tempo, and the Energy weight in the evaluator "
      "must be read off A2, where Energy on the board is observed directly.")
    w("")
    w("## B. Comeback analysis")
    w("")
    w("Behind means a fitted A1 win probability below 0.45 at the checkpoint. "
      "Behaviour is measured over the next four turns as rates per own turn and "
      "as shares of decisions; the game must reach t+2. Differences are "
      "winners minus losers inside a deficit bin, Welch 95%.")
    w("")
    for t, res in B["checkpoints"].items():
        for bname, r in res.items():
            if bname == "controlled_regression":
                w(f"**t = {t}, every behaviour entered at once** — n = {r['n']:,} "
                  f"behind seats, comeback rate {r['comeback_rate']:.3f}, "
                  f"McFadden R² {r['mcfadden_r2']:.3f}. Win on behaviour, "
                  "controlling for the position at t, the fitted deficit, and the "
                  "turns the window contained.")
                w("")
                w("| term | beta [95% CI] | z | AME pp/unit |")
                w("|---|---|---|---|")
                for c in r["coefficients"]:
                    if c["term"] == "const":
                        continue
                    w(f"| {c['term']} | {fmt_ci(c)} | {c['z']:+.1f} | "
                      f"{c['ame_pp_per_unit']:+.2f} |")
                w("")
                continue
            if "metrics" not in r:
                w(f"**t = {t}, {bname}** — n = {r['n']}, {r.get('note', '')}")
                continue
            w(f"**t = {t}, {bname}** — n = {r['n']:,} "
              f"({r['n_win']:,} came back, {r['n_lose']:,} did not; "
              f"comeback rate {r['comeback_rate']:.3f})")
            w("")
            w("| behaviour | winners | losers | diff [95% CI] |")
            w("|---|---|---|---|")
            for k, m in r["metrics"].items():
                if m.get("diff") is None or not k.endswith(("_per_turn", "_rate", "_turns")):
                    continue
                w(f"| {k} | {m['mean_win']:.3f} | {m['mean_lose']:.3f} | "
                  f"{m['diff']:+.3f} [{m['ci_lo']:+.3f}, {m['ci_hi']:+.3f}]"
                  f"{' *' if m['sig'] else ''} |")
            w("")
            if r["stratified"]:
                w("Stratified, inverse-variance pooled — `by_deficit` holds the "
                  "deficit quintile fixed, `by_deficit_archetype` holds quintile "
                  "and own archetype fixed:")
                w("")
                for k, s in r["stratified"].items():
                    w(f"- {k}: {s['stratified_diff']:+.3f} "
                      f"[{s['ci_lo']:+.3f}, {s['ci_hi']:+.3f}] over {s['strata']} strata"
                      f"{' *' if s['sig'] else ''}")
                w("")
    w("Causal caveat: a seat that attacks more, or activates more abilities, may "
      "simply hold the board and hand that permit it, so these differences read "
      "as much on hidden state as on choice, and a seat that is winning by turn "
      "t+3 finds attacks easier to make. Three tightenings run against that and "
      "none of them identifies a treatment effect: the stratified rows hold "
      "deficit depth and archetype fixed, the pooled fit enters every behaviour "
      "at once on top of the position at t, and the next-turn rates cut the "
      "window to the seat's immediate reply, before the outcome can feed back.")
    w("")
    w("## C. Opponent win and lose conditions")
    w("")
    w(C["spec"]["termination"])
    w("")
    w("### The one line per deck")
    w("")
    for a, e in C["archetypes"].items():
        if e.get("one_liner"):
            w(f"- **{e['one_liner']}**")
    w("")
    for a, e in C["archetypes"].items():
        wn, ls = e["wins"], e["losses"]
        if not wn.get("n"):
            continue
        w(f"### {a} — n = {e['n_seats']:,} rated seats, win rate {e['win_rate']:.3f}")
        w("")
        w(f"- Wins (n = {wn['n']:,}): prize race {wn['prize_race_le2']:.0%}, "
          f"attrition {wn['attrition_ge4']:.0%}, median turn {wn['median_turn']:.0f} "
          f"[{wn['p25_turn']:.0f}–{wn['p75_turn']:.0f}]")
        w(f"- Losses (n = {ls['n']:,}): prizes still on their side "
          f"median {ls['median_prizes_left']:.0f}, median turn {ls['median_turn']:.0f} "
          f"[{ls['p25_turn']:.0f}–{ls['p75_turn']:.0f}]")
        reg = e.get("regression_t7") or {}
        if reg.get("coefficients"):
            sig = [r for r in reg["coefficients"]
                   if r["sig"] and r["term"] != "const"]
            sig.sort(key=lambda r: -abs(r["ame_pp_per_sd"]))
            if sig:
                w(f"- Components at t = 7 (n = {reg['n']:,}): " + ", ".join(
                    f"{r['term']} {r['ame_pp_per_sd']:+.1f} pp/SD" for r in sig[:5]))
            else:
                w(f"- Components at t = 7 (n = {reg['n']:,}): none clears 95%")
        elif reg:
            w(f"- No per-archetype fit at t = 7: n = {reg.get('n', 0)}")
        mm = e.get("matchups") or {}
        if mm:
            worst = sorted(mm.items(), key=lambda kv: kv[1]["win_rate"])[:2]
            best = sorted(mm.items(), key=lambda kv: -kv[1]["win_rate"])[:2]
            w("- Matchups: best " + ", ".join(
                f"{k} {v['win_rate']:.2f} (n={v['n']})" for k, v in best) +
              "; worst " + ", ".join(
                f"{k} {v['win_rate']:.2f} (n={v['n']})" for k, v in worst))
        w("")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def strip_private(obj):
    if isinstance(obj, dict):
        return {k: strip_private(v) for k, v in obj.items() if not k.startswith("_")}
    return obj


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dates", nargs="*", default=None)
    ap.add_argument("--out", default="data/analysis")
    ap.add_argument("--top-archetypes", type=int, default=8)
    args = ap.parse_args()

    outdir = ROOT / args.out
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_decisions(args.dates)
    base = seat_base(df)
    pos = load_positions(args.dates)

    top = (base[base.rating >= RATING_CUT].groupby("our_archetype").size()
               .sort_values(ascending=False))
    arch_levels = list(top.head(args.top_archetypes).index)

    meta = {"dates": sorted(df.date.unique().tolist()),
            "episodes": int(df.episode_id.nunique()),
            "rows": int(len(df)),
            "positions": int(len(pos)),
            "seats_rated_at_cut": int((base.rating >= RATING_CUT).sum()),
            "seats_total": int(len(base)),
            "first_player_win_rate": round(
                float(base[base.went_first == 1].won.mean()), 4),
            "rated_seat_win_rate_going_first": round(
                float(base[(base.went_first == 1) &
                           (base.rating >= RATING_CUT)].won.mean()), 4),
            "rated_seat_win_rate_going_second": round(
                float(base[(base.went_first == 0) &
                           (base.rating >= RATING_CUT)].won.mean()), 4),
            "top_archetypes": {k: int(v) for k, v in top.head(args.top_archetypes).items()}}
    print("corpus:", json.dumps(meta)[:400], flush=True)

    A, A2 = analysis_A(df, base, pos, arch_levels)
    feats = A.pop("_feats")
    val = validate_proxy(feats, pos)
    print("A done", flush=True)
    B = analysis_B(df, feats, arch_levels)
    print("B done", flush=True)
    C = analysis_C(df, base, feats, arch_levels)
    print("C done", flush=True)

    (outdir / "advantage_A.json").write_text(
        json.dumps({"meta": meta, "A1_full_corpus": strip_private(A),
                    "A2_positions_evaluator_scale": A2,
                    "proxy_validation": val}, indent=1), encoding="utf-8")
    (outdir / "comeback_B.json").write_text(
        json.dumps({"meta": meta, **B}, indent=1), encoding="utf-8")
    (outdir / "archetypes_C.json").write_text(
        json.dumps({"meta": meta, **C}, indent=1), encoding="utf-8")
    write_report(outdir / "REPORT.md", A, A2, val, B, C, meta)
    print("wrote", outdir, flush=True)


if __name__ == "__main__":
    sys.exit(main())
