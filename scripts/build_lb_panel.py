"""Build the stratified leaderboard fitness panel (decisions.md D38 amendment).

One deck per 10-rank stratum of the current Kaggle leaderboard (ranks 1-10,
11-20, ... until --strata strata), each reconstructed from that team's most
recent public episode replays — the same mining path that built deck_priors.
Within a stratum the team with the most locally mined games is preferred;
teams whose replays cannot be fetched/parsed fall through to the next
candidate, and a stratum with no reachable team falls back to the field
priors' best list of the stratum leader's archetype (labeled in the
manifest — never silently).

After building, panel archetype shares are compared to the field priors'
shares; if any >=5%-share archetype is off by more than --gap-pp points the
panel is augmented toward --max-decks with unused teams of the missing
archetype.

Cell weights are UNIFORM (p=1 for every entry): proportional representation
lives in the panel composition itself. No Kaggle token is used — the episode
endpoints are public.

    python scripts/build_lb_panel.py
        --out data/panel_lb.json --manifest data/analysis/PANEL_LB.md
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ptcg.creation.pool import POKEMON, pool  # noqa: E402
from ptcg.creation.validator import validate  # noqa: E402

BASE = "https://www.kaggle.com/api/i/competitions.EpisodeService"
HEADERS = {"Content-Type": "application/json",
           "User-Agent": "ptcg-panel-builder/1.0"}


PACE_S = 2.0        # politeness gap after every successful call


def api(method: str, body: dict, retries: int = 4) -> dict | None:
    for i in range(retries):
        try:
            r = requests.post(f"{BASE}/{method}", json=body,
                              headers=HEADERS, timeout=30)
            if r.status_code == 200:
                time.sleep(PACE_S)
                return r.json()
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After") or 0) or 30 * (i + 1)
                print(f"  429 on {method}, backing off {min(wait, 120)}s",
                      flush=True)
                time.sleep(min(wait, 120))
                continue
        except requests.RequestException:
            pass
        time.sleep(2 * (i + 1))
    return None


def replay_decks(episode_id: int) -> tuple[list[str], list[Counter]] | None:
    """(team_names, [deck_p0, deck_p1]) from one replay, or None.

    Replays are public static content on kaggleusercontent.com — no auth,
    and none of the EpisodeService rate limiting."""
    rep = None
    for i in range(3):
        try:
            r = requests.get(
                f"https://www.kaggleusercontent.com/episodes/"
                f"{int(episode_id)}.json", headers=HEADERS, timeout=60)
            if r.status_code == 200:
                rep = r.json()
                time.sleep(PACE_S)
                break
            if r.status_code == 429:
                time.sleep(30 * (i + 1))
                continue
            return None          # 404 etc: episode not served
        except (requests.RequestException, json.JSONDecodeError):
            time.sleep(2 * (i + 1))
    if rep is None:
        return None
    info = rep.get("info") or {}
    steps = rep.get("steps") or []
    if not steps:
        return None
    try:
        vis = steps[0][0]["visualize"][0]["action"]
        decks = [Counter(x) for x in vis
                 if isinstance(x, list)
                 and all(isinstance(i, int) for i in x)]
    except (KeyError, IndexError, TypeError):
        return None
    if len(decks) != 2:
        return None
    return list(info.get("TeamNames") or []), decks


def label_archetype(deck: Counter, p) -> str:
    """Highest-HP Pokemon in the deck — the same rule as ptcg.meta."""
    best, best_hp = None, -1
    for cid in deck:
        c = p.by_id.get(cid)
        if not c or c["cardType"] != POKEMON or not c.get("hp"):
            continue
        if c["hp"] > best_hp:
            best, best_hp = c["name"], c["hp"]
    return best or "(no Pokemon)"


def team_deck_from_api(episode_ids: list[int], team_name: str,
                       max_replays: int = 2) -> tuple[Counter, int] | None:
    """(modal recent deck, n_replays_used) for a team.

    Episode ids come from the local mined history (meta_decks.csv), newest
    first, so only the public GetEpisodeReplay endpoint is hit — the
    ListEpisodes endpoint rate-limits far more aggressively."""
    got: list[Counter] = []
    for eid in episode_ids:
        if len(got) >= max_replays:
            break
        out = replay_decks(eid)
        if out is None:
            continue
        names, decks = out
        if team_name not in names:
            continue
        got.append(decks[names.index(team_name)])
    if not got:
        return None
    sigs = Counter(tuple(sorted(d.elements())) for d in got)
    modal = sigs.most_common(1)[0][0]
    return Counter(modal), len(got)


def priors_fallback(archetype: str, priors: dict) -> Counter | None:
    cands = [e for e in priors["decks"] if e["a"] == archetype]
    if not cands:
        return None
    e = max(cands, key=lambda e: e["p"])
    return Counter({int(c): n for c, n in e["c"].items()})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strata", type=int, default=25)
    ap.add_argument("--stratum-size", type=int, default=10)
    ap.add_argument("--max-decks", type=int, default=30)
    ap.add_argument("--gap-pp", type=float, default=10.0)
    ap.add_argument("--out", default=str(ROOT / "data/panel_lb.json"))
    ap.add_argument("--manifest",
                    default=str(ROOT / "data/analysis/PANEL_LB.md"))
    args = ap.parse_args()

    p = pool()
    lb = pd.read_csv(ROOT / "data/leaderboard_snapshot.csv")
    md = pd.read_csv(ROOT / "data/meta_decks.csv")
    games = md["agent"].value_counts()
    priors = json.loads((ROOT / "agent/deck_priors.json").read_text())

    rows, used_teams = [], set()

    def try_team(r, stratum) -> bool:
        eps = (md[md["agent"] == r.TeamName]
               .sort_values("episode_id", ascending=False)
               ["episode_id"].unique().tolist()[:6])
        if not eps:
            return False
        got = team_deck_from_api(eps, r.TeamName)
        if got is None:
            return False
        deck, n_reps = got
        listed = sorted(deck.elements())
        if len(listed) != 60 or not validate(listed).legal:
            return False
        rows.append({"stratum": stratum, "rank": int(r.Rank),
                     "team": r.TeamName, "team_id": int(r.TeamId),
                     "mined_games": int(games.get(r.TeamName, 0)),
                     "replays_used": n_reps, "source": "replay",
                     "archetype": label_archetype(deck, p),
                     "deck": {str(c): n for c, n in sorted(deck.items())}})
        used_teams.add(r.TeamName)
        return True

    for s in range(args.strata):
        lo, hi = s * args.stratum_size + 1, (s + 1) * args.stratum_size
        cand = lb[(lb["Rank"] >= lo) & (lb["Rank"] <= hi)].copy()
        cand["mined"] = cand["TeamName"].map(games).fillna(0)
        cand = cand.sort_values("mined", ascending=False)
        placed = False
        for r in list(cand.itertuples())[:2]:   # bounded: rate-limited API
            if try_team(r, f"{lo}-{hi}"):
                placed = True
                break
        if not placed:
            # priors fallback: the stratum leader's modal mined archetype
            arch = None
            for r in cand.itertuples():
                m = md[md["agent"] == r.TeamName]
                if len(m):
                    arch = m["archetype"].mode().iat[0]
                    break
            deck = priors_fallback(arch, priors) if arch else None
            if deck is None:
                print(f"stratum {lo}-{hi}: UNFILLABLE (no replays, no "
                      f"matching priors list)", flush=True)
                continue
            rows.append({"stratum": f"{lo}-{hi}", "rank": None,
                         "team": None, "team_id": None, "mined_games": 0,
                         "replays_used": 0, "source": "priors-fallback",
                         "archetype": label_archetype(deck, p),
                         "deck": {str(c): n for c, n in sorted(deck.items())}})
        print(f"stratum {lo}-{hi}: "
              f"{rows[-1]['team'] or 'FALLBACK'} "
              f"({rows[-1]['archetype']}, {rows[-1]['source']})", flush=True)

    # ---- proportionality check vs field priors ---------------------------
    tot = sum(e["p"] for e in priors["decks"])
    field = Counter()
    for e in priors["decks"]:
        field[e["a"]] += e["p"] / tot

    def panel_shares() -> Counter:
        c = Counter(r["archetype"] for r in rows)
        return Counter({k: v / len(rows) for k, v in c.items()})

    for arch, fs in sorted(field.items(), key=lambda kv: -kv[1]):
        if fs < 0.05:
            continue
        while (len(rows) < args.max_decks
               and (fs - panel_shares().get(arch, 0)) * 100 > args.gap_pp):
            extra = lb[~lb["TeamName"].isin(used_teams)].copy()
            extra["mined"] = extra["TeamName"].map(games).fillna(0)
            extra = extra.sort_values("mined", ascending=False)
            placed = False
            for r in extra.itertuples():
                m = md[md["agent"] == r.TeamName]
                if not len(m) or m["archetype"].mode().iat[0] != arch:
                    continue
                if try_team(r, f"aug({arch})"):
                    rows[-1]["stratum"] = f"aug {int(r.Rank)}"
                    print(f"augment {arch}: {r.TeamName} rank {r.Rank}",
                          flush=True)
                    placed = True
                    break
            if not placed:
                print(f"augment {arch}: no fillable team left", flush=True)
                break

    # ---- write panel + manifest ------------------------------------------
    panel = {"total": len(rows), "source": "leaderboard-stratified",
             "snapshot": "data/leaderboard_snapshot.csv (2026-08-06)",
             "decks": [{"c": r["deck"], "p": 1, "a": r["archetype"],
                        "team": r["team"], "rank": r["rank"]}
                       for r in rows]}
    Path(args.out).write_text(json.dumps(panel, separators=(",", ":")))

    shares = panel_shares()
    lines = ["# Stratified leaderboard panel — manifest\n",
             f"{len(rows)} decks; uniform cell weights (proportionality "
             "lives in composition). Snapshot: leaderboard 2026-08-06; "
             "episodes via public Kaggle episode API at build time.\n",
             "| stratum | rank | team | mined games | replays | source | "
             "archetype |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['stratum']} | {r['rank']} | {r['team']} | "
                     f"{r['mined_games']} | {r['replays_used']} | "
                     f"{r['source']} | {r['archetype']} |")
    lines += ["", "## Composition vs field priors\n",
              "| archetype | panel share | field share |", "|---|---|---|"]
    for arch, fs in sorted(field.items(), key=lambda kv: -kv[1]):
        if fs >= 0.02 or shares.get(arch):
            lines.append(f"| {arch} | {shares.get(arch, 0):.1%} | {fs:.1%} |")
    Path(args.manifest).write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out} ({len(rows)} decks) and {args.manifest}")


if __name__ == "__main__":
    main()
