"""Full matchup matrix over the top-8 field decks (agent/deck_priors.json).

Every unordered pairing of the field panel plays one seat-swapped match
(default 500 games) under ptcg.creation.harness.play_match: 28 pairings +
8 mirrors = 36 cells. Each deck is piloted by its harvested specialist
where one matches (specialist_panel), else the generalist
JonDayPilot(search=False).

Results land incrementally in <out>/matrix.json — one record per cell,
written after the cell finishes — so a killed run resumes by skipping
cells already recorded at the requested game count. A human-readable
win-rate table is rewritten to <out>/matrix.txt after every cell.

    .venv/bin/python scripts/matchup_matrix.py --games 500
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ptcg.creation.harness import play_match  # noqa: E402
from ptcg.creation.pilots import JonDayPilot  # noqa: E402
from ptcg.creation.specialist_panel import (  # noqa: E402
    build_specialist_panel, make_panel_pilots, panel_report)

PRIORS = ROOT / "agent" / "deck_priors.json"


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _pilot_name(entry: dict) -> str:
    spec = entry.get("pilot") or {"kind": "generalist"}
    return spec.get("specialist", "generalist") if spec["kind"] == "external" \
        else "generalist"


def _cell_record(panel: list[dict], i: int, j: int, match, games: int) -> dict:
    w0, w1, draws = match.wins
    capped = sum(1 for g in match.games if g.winner is None)
    reasons: dict[str, int] = {}
    for g in match.games:
        if g.reason:
            reasons[g.reason] = reasons.get(g.reason, 0) + 1
    turns = sorted(g.turns for g in match.games)
    return {
        "decks": [panel[i]["name"], panel[j]["name"]],
        "pilots": [_pilot_name(panel[i]), _pilot_name(panel[j])],
        "games": games,
        "wins": [w0, w1],
        "draws": draws,
        "capped": capped,
        "win_rate": match.win_rate(0),  # deck i's rate over decided games
        "end_reasons": reasons,
        "median_turns": turns[len(turns) // 2] if turns else 0,
        "elapsed_s": round(match.elapsed, 1),
    }


def _table(panel: list[dict], cells: dict) -> str:
    """Win-rate grid, row deck's rate vs column deck (decided games only)."""
    n = len(panel)
    rate = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            c = cells.get(f"{i}v{j}")
            if not c:
                continue
            w0, w1 = c["wins"]
            if w0 + w1 == 0:
                continue
            rate[i][j] = w0 / (w0 + w1)
            if i != j:
                rate[j][i] = w1 / (w0 + w1)
    lines = ["Matchup matrix: row deck's win rate vs column deck "
             "(decided games; draws/caps excluded)", ""]
    for k, e in enumerate(panel):
        lines.append(f"  [{k}] w={e['weight']:.3f} {e['name'][:44]:46s} "
                     f"pilot={_pilot_name(e)}")
    lines.append("")
    head = "      " + "".join(f"  [{j}]  " for j in range(n))
    lines.append(head)
    for i in range(n):
        row = f"  [{i}] "
        for j in range(n):
            row += "   --  " if rate[i][j] is None else f" {rate[i][j]:5.1%} "
        lines.append(row)
    done = sum(1 for i in range(n) for j in range(i, n)
               if cells.get(f"{i}v{j}"))
    total = n * (n + 1) // 2
    lines.append("")
    lines.append(f"cells complete: {done}/{total}")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--games", type=int, default=500,
                    help="seat-swapped games per cell (default 500)")
    ap.add_argument("--out", default=str(ROOT / "runs" / "matrix"),
                    help="output directory (default runs/matrix)")
    ap.add_argument("--priors", default=str(PRIORS))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    matrix_path = out / "matrix.json"
    table_path = out / "matrix.txt"

    def generalist(seed: int):
        return JonDayPilot(seed=seed, search=False)

    panel = build_specialist_panel(Path(args.priors))
    # Two independent pilot lists so a mirror cell never hands the same
    # pilot object to both seats.
    row_pilots = make_panel_pilots(panel, generalist, seed=900)
    col_pilots = make_panel_pilots(panel, generalist, seed=950)
    print("panel:\n" + panel_report(panel), flush=True)

    state = (json.loads(matrix_path.read_text()) if matrix_path.exists()
             else {"games_per_cell": args.games, "cells": {}})
    cells = state["cells"]

    n = len(panel)
    pairs = [(i, j) for i in range(n) for j in range(i, n)]
    print(f"{len(pairs)} cells x {args.games} games -> {out}", flush=True)

    for k, (i, j) in enumerate(pairs, 1):
        key = f"{i}v{j}"
        prior = cells.get(key)
        if prior and prior.get("games", 0) >= args.games:
            print(f"[{k}/{len(pairs)}] {key} already done, skipping",
                  flush=True)
            continue
        print(f"[{k}/{len(pairs)}] {key}: {panel[i]['name'][:30]} vs "
              f"{panel[j]['name'][:30]} ({args.games} games)", flush=True)
        t0 = time.time()
        match = play_match(row_pilots[i], col_pilots[j],
                           panel[i]["deck"], panel[j]["deck"],
                           n_games=args.games)
        cells[key] = _cell_record(panel, i, j, match, args.games)
        _atomic_write(matrix_path, json.dumps(state, indent=1))
        _atomic_write(table_path, _table(panel, cells))
        c = cells[key]
        print(f"    {c['wins'][0]}-{c['wins'][1]} "
              f"(draws {c['draws']}, capped {c['capped']}) "
              f"wr={c['win_rate']:.1%} in {time.time() - t0:.0f}s",
              flush=True)

    print("matrix complete:\n" + _table(panel, cells), flush=True)


if __name__ == "__main__":
    main()
