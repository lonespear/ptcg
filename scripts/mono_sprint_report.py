"""Mono-sprint report generator: typing-robustness readout from a
mono-only archipelago run (scripts/run_archipelago.py --mono-only).

Reads runs/<run-id>/{era_*.json, final_eval.json, panel.json} and writes
data/analysis/MONO_SPRINT.md: per-island elite trajectories, the
typing-robustness table (deep win rate vs the specialist panel + CI,
matchup-profile spread, termination-mode distribution, prize-denomination
structure), and the top decks per island summarized by card name.

    python scripts/mono_sprint_report.py --run-id mono_sprint
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ptcg.creation.pool import (  # noqa: E402
    BASIC_ENERGY, ITEM, POKEMON, SPECIAL_ENERGY, STADIUM, SUPPORTER, TOOL,
    pool)

KIND = {POKEMON: "Pokemon", ITEM: "Item", TOOL: "Tool",
        SUPPORTER: "Supporter", STADIUM: "Stadium",
        BASIC_ENERGY: "Energy", SPECIAL_ENERGY: "Energy*"}


def prizes_of(card: dict) -> int:
    if card.get("megaEx"):
        return 3
    if card.get("ex"):
        return 2
    return 1


def denomination(deck: list[int], p) -> dict:
    mons = [p.by_id[c] for c in deck if p.by_id[c]["cardType"] == POKEMON]
    pr = Counter(prizes_of(m) for m in mons)
    return {"bodies": len(mons), "p1": pr.get(1, 0), "p2": pr.get(2, 0),
            "p3": pr.get(3, 0),
            "prizes_per_body": round(sum(prizes_of(m) for m in mons)
                                     / len(mons), 2) if mons else 0.0}


def deck_summary(deck: list[int], p) -> list[str]:
    by_kind: dict[str, Counter] = {}
    for c in deck:
        card = p.by_id[c]
        k = KIND.get(card["cardType"], "?")
        by_kind.setdefault(k, Counter())[card["name"]] += 1
    lines = []
    for k in ("Pokemon", "Item", "Tool", "Supporter", "Stadium", "Energy",
              "Energy*"):
        if k not in by_kind:
            continue
        parts = [f"{n}x {name}" for name, n in by_kind[k].most_common()]
        lines.append(f"{k}: " + ", ".join(parts))
    return lines


def weighted_deep_wr(per_opp: list[dict]) -> tuple[float, float, int]:
    """Weighted win rate over decided games, normal-approx 95% half-width."""
    num = den = var = 0.0
    n_total = 0
    for o in per_opp:
        n = o["w"] + o["l"]
        if n == 0:
            continue
        pr = o["w"] / n
        num += o["weight"] * pr
        den += o["weight"]
        var += (o["weight"] ** 2) * pr * (1 - pr) / n
        n_total += n
    if den == 0:
        return 0.0, 0.0, 0
    return num / den, 1.96 * math.sqrt(var) / den, n_total


def reason_mix(per_opp: list[dict], key: str) -> dict:
    tot: Counter = Counter()
    for o in per_opp:
        tot.update(o[key])
    n = sum(tot.values())
    return {k: round(v / n, 3) for k, v in tot.most_common()} if n else {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="mono_sprint")
    ap.add_argument("--out", default=str(ROOT / "data/analysis/MONO_SPRINT.md"))
    args = ap.parse_args()
    run_dir = ROOT / "runs" / args.run_id
    p = pool()

    fe = json.loads((run_dir / "final_eval.json").read_text())
    eras = sorted(run_dir.glob("era_*.json"))
    trajectories: dict[str, list[tuple[int, float]]] = {}
    phase_a_end = None
    for ef in eras:
        ck = json.loads(ef.read_text())
        phase_a_end = ck["era"]
        era_best: dict[str, float] = {}
        for label, r in ck["islands"].items():
            sk = label.rsplit("/", 1)[0]
            era_best[sk] = max(era_best.get(sk, -9.9), r["best"])
        for sk, b in era_best.items():
            cur = trajectories.setdefault(sk, [])
            best = max(b, cur[-1][1]) if cur else b
            cur.append((ck["era"], best))

    sets = sorted({e["set"] for e in fe["elites"]})
    lead = {sk: max((e for e in fe["elites"] if e["set"] == sk),
                    key=lambda e: e["ga_fitness"]) for sk in sets}

    out = []
    out.append("# Mono sprint — energy-typing robustness vs the field\n")
    out.append(f"Run `runs/{args.run_id}/` (seed 73, 7 workers, pop 10, "
               "24 games/opponent GA fitness, plateau patience "
               f"{fe['plateau_window']} eras at +0.01). "
               f"{(phase_a_end or 0) + 1} GA eras completed, "
               f"{fe['elapsed_h']:.2f}h total. Fitness: play-weighted win "
               "rate vs the specialist-piloted top-8 field panel minus the "
               "D18 termination penalty (0.15 x exhaustion share of "
               "losses). Deep eval: elites replayed vs the full panel for "
               "tighter CIs; games per opponent as listed (codex-piloted "
               "Fezandipiti at a quarter rate, as in GA fitness).\n")

    out.append("## Termination paths\n")
    out.append("| island | path | last improvement (era) | set best |")
    out.append("|---|---|---|---|")
    for sk, t in sorted(fe["termination"].items()):
        out.append(f"| {sk} | {t['path']} | {t['last_improve_era']} | "
                   f"{t['best']:.3f} |")
    out.append("")

    out.append("## Typing-robustness table (island elites, deep eval)\n")
    out.append("| island | GA fit | deep WR (95% CI) | decided g | profile "
               "min/median/max | exh. loss share | bodies | 1-prize | "
               "2-prize | 3-prize | prizes/body |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for sk in sets:
        e = lead[sk]
        wr, hw, n = weighted_deep_wr(e["per_opp"])
        prof = sorted(o["w"] / (o["w"] + o["l"])
                      for o in e["per_opp"] if o["w"] + o["l"])
        if not prof:               # no deep games: fall back to GA profile
            prof = sorted(e["ga_profile"])
        med = prof[len(prof) // 2] if prof else 0.0
        lm = reason_mix(e["per_opp"], "loss_reasons")
        exh = lm.get("deck-out", 0) + lm.get("no active Pokemon", 0)
        d = denomination(e["deck"], p)
        out.append(
            f"| {sk} | {e['ga_fitness']:.3f} | {wr:.3f} +/- {hw:.3f} | {n} "
            f"| {prof[0]:.2f}/{med:.2f}/{prof[-1]:.2f} | {exh:.2f} "
            f"| {d['bodies']} | {d['p1']} | {d['p2']} | {d['p3']} "
            f"| {d['prizes_per_body']} |")
    out.append("")

    out.append("### Per-opponent deep win rates (island elites)\n")
    names = [o["name"] for o in fe["elites"][0]["per_opp"]]
    out.append("| island | " + " | ".join(
        f"{n} ({o['weight']:.2f})" for n, o in
        zip(names, fe["elites"][0]["per_opp"])) + " |")
    out.append("|---|" + "---|" * len(names))
    for sk in sets:
        e = lead[sk]
        cells = []
        for o in e["per_opp"]:
            n = o["w"] + o["l"]
            cells.append(f"{o['w'] / n:.2f}" if n else "-")
        out.append(f"| {sk} | " + " | ".join(cells) + " |")
    out.append("")

    out.append("### Termination-mode distribution (island elites, deep eval)\n")
    out.append("| island | wins by | losses by |")
    out.append("|---|---|---|")
    for sk in sets:
        e = lead[sk]
        wm = reason_mix(e["per_opp"], "win_reasons")
        lm = reason_mix(e["per_opp"], "loss_reasons")
        fmt = lambda d: ", ".join(f"{k} {v:.0%}" for k, v in d.items())
        out.append(f"| {sk} | {fmt(wm)} | {fmt(lm)} |")
    out.append("")

    out.append("## Elite trajectories (set best-so-far by era)\n")
    for sk in sets:
        tr = trajectories.get(sk, [])
        pts = tr[:: max(1, len(tr) // 12)]
        if tr and pts[-1][0] != tr[-1][0]:
            pts.append(tr[-1])
        line = " -> ".join(f"e{e}:{b:.3f}" for e, b in pts)
        out.append(f"- **{sk}**: {line}")
    out.append("")

    out.append("## Top decks per island\n")
    for sk in sets:
        recs = sorted((e for e in fe["elites"] if e["set"] == sk),
                      key=lambda e: -e["ga_fitness"])
        out.append(f"### {sk}\n")
        for i, e in enumerate(recs, 1):
            wr, hw, n = weighted_deep_wr(e["per_opp"])
            d = denomination(e["deck"], p)
            out.append(f"**#{i}** ({e['island']}) GA {e['ga_fitness']:.3f}, "
                       f"deep {wr:.3f} +/- {hw:.3f} ({n} decided); "
                       f"{d['bodies']} bodies: {d['p1']}x1-prize, "
                       f"{d['p2']}x2-prize, {d['p3']}x3-prize\n")
            for line in deck_summary(e["deck"], p):
                out.append(f"- {line}")
            out.append("")

    Path(args.out).write_text("\n".join(out) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
