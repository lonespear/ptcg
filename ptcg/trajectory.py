"""Empirical turn-to-turn trajectories, fitted from the per-turn series.

`ptcg.extract`'s `series` extractor writes one row per (episode, seat, turn):
the board as that seat's turn opened, read off the replay's true state. This
module turns those rows into the conditional curves a projection needs -- given
what a deck has in play on turn t, what it actually has on turns t+1, t+2, t+3 --
and then checks the projection rule the playbook's C1 entry assumes against what
the games did.

    python -m ptcg.trajectory                     # every mined day with a series
    python -m ptcg.trajectory --dates 2026-08-05 --min-rating 1000

Writes `data/analysis/trajectory_curves.json` and `TRAJECTORY_REPORT.md`.

Two facts about the data constrain every number below.

  * Energy in play is Energy sitting on Pokemon, not Energy attached over the
    game. A knocked-out Pokemon takes its Energy with it, so the series can and
    does fall, which is exactly what a +1-per-turn projection cannot represent.
  * A pair (t, t+k) only exists when the game lasted k more of that seat's
    turns. Every conditional expectation here is therefore conditioned on
    survival to t+k, and late-turn cells are drawn from longer games.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
MINED = ROOT / "data" / "mined"
ANALYSIS = ROOT / "data" / "analysis"

METRICS = ("energy_in_play", "board_hp", "damage_dealt_cumulative")
HORIZONS = (1, 2, 3)
TURN_BUCKETS = ((1, 3, "t1-3"), (4, 6, "t4-6"), (7, 99, "t7+"))
MIN_CELL = 30
TOP_N = 8

# How the current-state axis is cut, per metric. Energy is small enough to bin
# on the integer itself; HP and damage go in coarse bands.
BINNERS = {
    "energy_in_play": (1, 9),
    "board_hp": (150, 900),
    "damage_dealt_cumulative": (100, 500),
}


def load_series(dates: list[str] | None = None) -> pd.DataFrame:
    """Every mined day that carries a series table, concatenated."""
    frames, found = [], []
    for d in sorted(MINED.iterdir()):
        if not d.is_dir() or (dates and d.name not in dates):
            continue
        pq, csv = d / "series.parquet", d / "series.csv.gz"
        path = pq if pq.exists() else (csv if csv.exists() else None)
        if path is None:
            continue
        df = pd.read_parquet(path) if path.suffix == ".parquet" \
            else pd.read_csv(path)
        frames.append(df)
        found.append((d.name, len(df), int(df.episode_id.nunique())))
    if not frames:
        raise SystemExit(
            "no series tables found -- mine a day with the v2 extractor first: "
            "python scripts/mine_day.py <date> --extract")
    df = pd.concat(frames, ignore_index=True)
    df.attrs["days"] = found
    return df


def add_leads(df: pd.DataFrame, horizons=HORIZONS) -> pd.DataFrame:
    """Attach x_{t+k} for each metric, within (episode, seat), k turns ahead.

    The shift is taken on the seat's own turn index, and kept only where the
    row k positions later really is turn t+k -- a missing turn (a seat with no
    decision inside a turn) breaks the pair rather than silently shortening it.
    """
    df = df.sort_values(["episode_id", "seat", "seat_turn"]).reset_index(drop=True)
    g = df.groupby(["episode_id", "seat"], sort=False)
    for k in horizons:
        ahead = g["seat_turn"].shift(-k)
        ok = ahead == df["seat_turn"] + k
        for m in METRICS:
            lead = g[m].shift(-k)
            df[f"{m}_t{k}"] = lead.where(ok)
        df[f"valid_t{k}"] = ok
    return df


def bin_values(s: pd.Series, metric: str) -> tuple[pd.Series, pd.Series]:
    """Bin the current-state axis; returns (bin key, label)."""
    width, cap = BINNERS[metric]
    v = pd.to_numeric(s, errors="coerce").fillna(-1)
    b = (np.minimum(v, cap) // width * width).astype(int)
    if width == 1:
        lab = b.astype(str).where(b < cap, f"{cap}+")
    else:
        lab = (b.astype(str) + "-" + (b + width - 1).astype(str)).where(
            b < cap, f"{cap}+")
    return b, lab


def turn_bucket(s: pd.Series) -> pd.Series:
    out = pd.Series(pd.NA, index=s.index, dtype="object")
    for lo, hi, name in TURN_BUCKETS:
        out[(s >= lo) & (s <= hi)] = name
    return out


def cells(df: pd.DataFrame, metric: str, k: int) -> list[dict]:
    """E[x_{t+k} | bin(x_t), bucket(t)] with 10/50/90 bands, per cell."""
    col = f"{metric}_t{k}"
    sub = df[df[f"valid_t{k}"] & df[col].notna()]
    if sub.empty:
        return []
    key, lab = bin_values(sub[metric], metric)
    work = pd.DataFrame({
        "bin": key, "bin_label": lab, "t_bucket": turn_bucket(sub["seat_turn"]),
        "now": pd.to_numeric(sub[metric], errors="coerce"),
        "then": pd.to_numeric(sub[col], errors="coerce"),
    }).dropna(subset=["t_bucket"])
    out = []
    for (tb, b), grp in work.groupby(["t_bucket", "bin"], sort=True):
        n = len(grp)
        if n < MIN_CELL:
            continue
        q = np.percentile(grp["then"], [10, 50, 90])
        out.append({
            "t_bucket": tb, "bin": int(b), "bin_label": grp["bin_label"].iloc[0],
            "n": int(n),
            "mean_now": round(float(grp["now"].mean()), 2),
            "mean": round(float(grp["then"].mean()), 2),
            "p10": round(float(q[0]), 1), "p50": round(float(q[1]), 1),
            "p90": round(float(q[2]), 1),
            "delta_mean": round(float((grp["then"] - grp["now"]).mean()), 2),
            "p_falls": round(float((grp["then"] < grp["now"]).mean()), 3),
        })
    return sorted(out, key=lambda r: (r["t_bucket"], r["bin"]))


def validate_projection(df: pd.DataFrame, curves: dict, arch: str | None,
                        k: int = 2, rate: float = 1.0) -> dict:
    """Score the C1 projection rule for OUR side: Energy grows `rate` a turn.

    Bias is projected minus realized, so a positive bias is a projection that
    promises Energy the games did not deliver. The fitted conditional mean --
    the same cells this module writes out -- is scored on the identical rows so
    the two rules are comparable rather than merely both reported.
    """
    col = f"energy_in_play_t{k}"
    sub = df[df[f"valid_t{k}"] & df[col].notna()]
    if arch is not None:
        sub = sub[sub["our_archetype"] == arch]
    if len(sub) < MIN_CELL:
        return {}
    now = pd.to_numeric(sub["energy_in_play"], errors="coerce").to_numpy(float)
    real = pd.to_numeric(sub[col], errors="coerce").to_numpy(float)
    proj = now + rate * k

    curve = (curves.get(arch or "(all)", {}).get("energy_in_play", {})
             .get(f"k{k}", []))
    key, _ = bin_values(sub["energy_in_play"], "energy_in_play")
    tb = turn_bucket(sub["seat_turn"])
    means = {(r["t_bucket"], r["bin"]): r["mean"] for r in curve}
    fitted = np.array([means.get((t, int(b)), np.nan)
                       for b, t in zip(key.to_numpy(), tb.to_numpy())])
    have = ~np.isnan(fitted)

    def score(p, r):
        e = p - r
        return {"bias": round(float(e.mean()), 3),
                "mae": round(float(np.abs(e).mean()), 3),
                "rmse": round(float(np.sqrt((e ** 2).mean())), 3)}

    out = {"archetype": arch or "(all)", "n": int(len(sub)), "k": k,
           "mean_energy_now": round(float(now.mean()), 2),
           "mean_energy_realized": round(float(real.mean()), 2),
           "naive": score(proj, real),
           "implied_rate_mean": round(float(((real - now) / k).mean()), 3),
           "implied_rate_median": round(float(np.median((real - now) / k)), 3),
           "share_realized_below_projection":
               round(float((real < proj).mean()), 3),
           "share_energy_fell": round(float((real < now).mean()), 3)}
    if have.sum() >= MIN_CELL:
        out["fitted_conditional_mean"] = score(fitted[have], real[have])
        out["fitted_n"] = int(have.sum())
    return out


def energy_curves(df: pd.DataFrame, arches: list[str], k: int = 2) -> dict:
    """Just the Energy cells, for scoring a fit on days it was not fitted on."""
    out = {"(all)": {"energy_in_play": {f"k{k}": cells(df, "energy_in_play", k)}}}
    for a in arches:
        sub = df[df["our_archetype"] == a]
        out[a] = {"energy_in_play": {f"k{k}": cells(sub, "energy_in_play", k)}}
    return out


def build(df: pd.DataFrame, top_n: int = TOP_N) -> dict:
    counts = df.groupby("our_archetype").size().sort_values(ascending=False)
    tops = [a for a in counts.index[:top_n]]
    curves: dict[str, dict] = {}
    for arch in ["(all)"] + tops:
        sub = df if arch == "(all)" else df[df["our_archetype"] == arch]
        block: dict[str, dict] = {}
        for m in METRICS:
            block[m] = {f"k{k}": cells(sub, m, k) for k in HORIZONS}
        block["rows"] = int(len(sub))
        block["episodes"] = int(sub.episode_id.nunique())
        block["mean_seat_turns"] = round(
            float(sub.groupby(["episode_id", "seat"]).size().mean()), 2)
        curves[arch] = block
    val = [validate_projection(df, curves, None)]
    val += [v for a in tops if (v := validate_projection(df, curves, a))]

    # Same check on a day the cells were not fitted on, so the fitted column is
    # a forecast rather than a description of its own training rows.
    holdout: dict = {}
    days = sorted(df["date"].astype(str).unique())
    if len(days) > 1:
        test_day = days[-1]
        train = df[df["date"].astype(str) != test_day]
        test = df[df["date"].astype(str) == test_day]
        fit = energy_curves(train, tops)
        rows = [validate_projection(test, fit, None)]
        rows += [v for a in tops if (v := validate_projection(test, fit, a))]
        holdout = {"fitted_on": days[:-1], "scored_on": test_day, "rows": rows}

    return {"curves": curves, "archetypes": tops,
            "validation_energy_k2": val,
            "validation_energy_k2_holdout": holdout,
            "days": [{"date": d, "rows": r, "episodes": e}
                     for d, r, e in df.attrs.get("days", [])]}


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def fmt_curve(curve: list[dict], unit: str = "") -> list[str]:
    lines = ["| t | now | n | mean t+k | p10 | p50 | p90 | delta mean | falls |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in curve:
        lines.append(
            f"| {r['t_bucket']} | {r['bin_label']}{unit} | {r['n']} | "
            f"{r['mean']} | {r['p10']} | {r['p50']} | {r['p90']} | "
            f"{r['delta_mean']:+} | {r['p_falls']:.0%} |")
    return lines


def write_report(doc: dict, path: Path) -> None:
    c = doc["curves"]
    L: list[str] = []
    w = L.append
    w("# Turn-to-turn trajectories, fitted from replays")
    w("")
    w("Question this answers: given what a deck has on the board at the top of "
      "turn t, what does it actually have one, two and three of its own turns "
      "later, and does the playbook's one-Energy-a-turn projection match what "
      "the games did?")
    w("")
    w("## Where the numbers come from")
    w("")
    w("`ptcg.extract`'s `series` extractor writes one row per (episode, seat, "
      "turn), taken at the first decision the seat is asked for inside a turn "
      "that belongs to it -- the board as its turn opened. Turns alternate from "
      "the first player, and a seat is also asked to decide during the "
      "opponent's turn (choosing a new Active after a knockout, for one); those "
      "mid-opponent-turn decisions are excluded, and including them shifts the "
      "Energy curve down by about half an Energy from turn 4 on.")
    w("")
    w("Energy, HP, bench width, hand size and prizes are read from the "
      "observation's true state (`current.players`), so Energy in play is "
      "Energy sitting on Pokemon rather than a running count of attachments.")
    w("")
    w("Turn index `t` is the seat's own turn number, so t+1 is that seat's next "
      "turn, one full round of play later. A pair (t, t+k) exists only when the "
      "game ran k more of the seat's turns, so every cell is conditioned on "
      "surviving that long.")
    w("")
    days = doc.get("days") or []
    if days:
        w("| day | series rows | episodes |")
        w("|---|---|---|")
        for d in days:
            w(f"| {d['date']} | {d['rows']:,} | {d['episodes']:,} |")
        w(f"| **total** | **{sum(d['rows'] for d in days):,}** | "
          f"**{sum(d['episodes'] for d in days):,}** |")
        w("")
    w(f"Cells with fewer than {MIN_CELL} observations are dropped. Bands are "
      "the 10th, 50th and 90th percentiles of the realized value at t+k; "
      "\"falls\" is the share of cells where the value at t+k is below the "
      "value at t.")
    w("")

    w("## The projection check")
    w("")
    w("C1 projects our own Energy forward at one attachment a turn. Scored "
      "against the realized Energy in play two of our turns later, over every "
      "(t, t+2) pair in the series. Bias is projected minus realized, so a "
      "positive bias is Energy the projection promised and the board did not "
      "hold. The fitted column is this module's own conditional mean scored on "
      "the same rows.")
    w("")
    w("| archetype | pairs | Energy now | Energy at t+2 | +1/turn bias | +1/turn "
      "MAE | fitted bias | fitted MAE | realized rate/turn | Energy fell |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for v in doc["validation_energy_k2"]:
        f = v.get("fitted_conditional_mean") or {}
        w(f"| {v['archetype']} | {v['n']:,} | {v['mean_energy_now']} | "
          f"{v['mean_energy_realized']} | {v['naive']['bias']:+} | "
          f"{v['naive']['mae']} | "
          f"{f.get('bias', float('nan')):+} | {f.get('mae', '-')} | "
          f"{v['implied_rate_mean']} | "
          f"{v['share_energy_fell']:.0%} |")
    w("")
    w("The fitted column above is scored on the rows it was fitted on, so its "
      "zero bias is arithmetic rather than evidence. The same table below is "
      "fitted on the earlier days and scored on the last one.")
    w("")
    ho = doc.get("validation_energy_k2_holdout") or {}
    if ho:
        w(f"Fitted on {', '.join(ho['fitted_on'])}, scored on "
          f"{ho['scored_on']}.")
        w("")
        w("| archetype | pairs | +1/turn bias | +1/turn MAE | fitted bias | "
          "fitted MAE |")
        w("|---|---|---|---|---|---|")
        for v in ho["rows"]:
            f = v.get("fitted_conditional_mean") or {}
            w(f"| {v['archetype']} | {v['n']:,} | {v['naive']['bias']:+} | "
              f"{v['naive']['mae']} | {f.get('bias', float('nan')):+} | "
              f"{f.get('mae', '-')} |")
        w("")

    w("## Conditional curves")
    w("")
    for arch in ["(all)"] + doc["archetypes"]:
        b = c[arch]
        w(f"### {arch}")
        w("")
        w(f"{b['rows']:,} turn rows over {b['episodes']:,} episodes, "
          f"{b['mean_seat_turns']} turns a seat on average.")
        w("")
        for m, unit, ks in (("energy_in_play", "", (1, 2, 3)),
                            ("board_hp", " HP", (2,)),
                            ("damage_dealt_cumulative", " dmg", (2,))):
            for k in ks:
                curve = b[m][f"k{k}"]
                if not curve:
                    continue
                w(f"**{m}, k={k}**")
                w("")
                L.extend(fmt_curve(curve, unit))
                w("")
    w("## What limits these numbers")
    w("")
    w("Knockouts remove a Pokemon and its Energy from the board, which is the "
      "mechanism behind every falling cell and the reason a monotone projection "
      "runs high. Damage dealt is damage standing on the opponent's Pokemon "
      "right now; a knockout resets it to zero for that Pokemon, so the damage "
      "curves understate output in exactly the games where output was best. "
      "Seat ratings come from a current leaderboard snapshot joined on team "
      "name, not the rating at game time.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dates", nargs="*", default=None)
    ap.add_argument("--min-rating", type=float, default=0.0,
                    help="keep only seats rated at or above this")
    ap.add_argument("--top", type=int, default=TOP_N)
    ap.add_argument("--out", default=str(ANALYSIS))
    args = ap.parse_args()

    df = load_series(args.dates)
    days = df.attrs.get("days", [])
    print(f"loaded {len(df):,} series rows from {len(days)} day(s)")
    for d, r, e in days:
        print(f"  {d}  {r:,} rows  {e:,} episodes  "
              f"({r / max(e, 1) / 2:.1f} turns per seat)")
    if args.min_rating:
        before = len(df)
        df = df[df["agent_rating"].fillna(-1) >= args.min_rating]
        print(f"  rating filter >= {args.min_rating}: {len(df):,} of {before:,}")
    keep = df.attrs.get("days")
    df = add_leads(df)
    df.attrs["days"] = keep

    doc = build(df, top_n=args.top)
    doc["min_rating"] = args.min_rating
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "trajectory_curves.json").write_text(
        json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(doc, out / "TRAJECTORY_REPORT.md")

    print("\nprojection check, Energy at t+2 (bias = projected - realized):")
    print(f"  {'archetype':<28}{'n':>8}{'bias':>8}{'MAE':>7}{'fit bias':>10}"
          f"{'fit MAE':>9}{'rate':>7}{'fell':>7}")
    for v in doc["validation_energy_k2"]:
        f = v.get("fitted_conditional_mean") or {}
        print(f"  {v['archetype'][:27]:<28}{v['n']:>8,}"
              f"{v['naive']['bias']:>8.2f}{v['naive']['mae']:>7.2f}"
              f"{f.get('bias', float('nan')):>10.2f}"
              f"{f.get('mae', float('nan')):>9.2f}"
              f"{v['implied_rate_mean']:>7.2f}"
              f"{v['share_energy_fell']:>7.0%}")
    ho = doc.get("validation_energy_k2_holdout") or {}
    if ho:
        print(f"\nheld out {ho['scored_on']}, fitted on "
              f"{', '.join(ho['fitted_on'])}:")
        for v in ho["rows"]:
            f = v.get("fitted_conditional_mean") or {}
            print(f"  {v['archetype'][:27]:<28}{v['n']:>8,}"
                  f"{v['naive']['bias']:>8.2f}{v['naive']['mae']:>7.2f}"
                  f"{f.get('bias', float('nan')):>10.2f}"
                  f"{f.get('mae', float('nan')):>9.2f}")
    print(f"\nwrote {out / 'trajectory_curves.json'}")
    print(f"wrote {out / 'TRAJECTORY_REPORT.md'}")


if __name__ == "__main__":
    main()
