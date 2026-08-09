"""Harvest every REAL ladder opponent decklist from our own episode replays.

Why this exists (decisions.md D58): our six `external/*_deck.json` panel decks
are not the field. They forfeit on engine errors 24.9% of the time where real
opponents forfeit 0.0%, none of them carries a damage-wall card though 3 of 80
real opponent decks do, and they steer our search into lethal-decline positions
the ladder never produces. A panel built from decks the ladder actually played
is the only fix that addresses all three.

Every replay's step-0 `visualize` field carries BOTH seats' full 60-card
decklists and `info.TeamNames` identifies the seats, so one GET per episode
yields the opponent's exact list. Episode metadata (`ListEpisodes`) yields the
opponent's rating at the time and the game's outcome, so no replay parsing is
needed for either. Both endpoints are public; no Kaggle token is used.

    python scripts/extract_field_decks.py --out data/analysis/field_decks.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ptcg.creation.pool import POKEMON, pool  # noqa: E402
from ptcg.creation.validator import validate  # noqa: E402

BASE = "https://www.kaggle.com/api/i/competitions.EpisodeService"
HEADERS = {"Content-Type": "application/json",
           "User-Agent": "ptcg-field-panel/1.0"}
PACE_S = 1.5

OUR_TEAM = "Lemmes Yad"
# Damage walls in the engine pool (E11 / D58): Farigiraf ex, Cornerstone Mask
# Ogerpon ex, Drednaw, Milotic ex, Sylveon, Crustle.
WALLS = {83, 117, 158, 207, 330, 345}


def list_episodes(submission_id: int) -> list[dict]:
    for i in range(4):
        try:
            r = requests.post(f"{BASE}/ListEpisodes",
                              json={"submissionId": submission_id},
                              headers=HEADERS, timeout=30)
            if r.status_code == 200:
                time.sleep(PACE_S)
                return r.json().get("episodes") or []
            if r.status_code == 429:
                time.sleep(30 * (i + 1))
                continue
        except requests.RequestException:
            pass
        time.sleep(2 * (i + 1))
    return []


def replay(episode_id: int) -> dict | None:
    for i in range(3):
        try:
            r = requests.get("https://www.kaggleusercontent.com/episodes/"
                             f"{int(episode_id)}.json",
                             headers=HEADERS, timeout=60)
            if r.status_code == 200:
                time.sleep(PACE_S)
                return r.json()
            if r.status_code == 429:
                time.sleep(30 * (i + 1))
                continue
            return None
        except (requests.RequestException, json.JSONDecodeError):
            time.sleep(2 * (i + 1))
    return None


def seat_decks(rep: dict) -> tuple[list[str], list[Counter]] | None:
    steps = rep.get("steps") or []
    if not steps:
        return None
    try:
        vis = steps[0][0]["visualize"][0]["action"]
        decks = [Counter(x) for x in vis
                 if isinstance(x, list) and all(isinstance(i, int) for i in x)]
    except (KeyError, IndexError, TypeError):
        return None
    if len(decks) != 2:
        return None
    return list((rep.get("info") or {}).get("TeamNames") or []), decks


def label_archetype(deck: Counter, p) -> str:
    """Highest-HP Pokemon in the deck — the rule scripts/build_lb_panel.py uses."""
    best, best_hp = None, -1
    for cid in deck:
        c = p.by_id.get(cid)
        if not c or c["cardType"] != POKEMON or not c.get("hp"):
            continue
        if c["hp"] > best_hp:
            best, best_hp = c["name"], c["hp"]
    return best or "(no Pokemon)"


def band(score: float | None) -> str:
    if score is None:
        return "unknown"
    for lo, hi in ((0, 600), (600, 700), (700, 800), (800, 900),
                   (900, 1000)):
        if lo <= score < hi:
            return f"{lo}-{hi}"
    return "1000+"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subs", default="55349268,55362393",
                    help="our submission ids, comma separated")
    ap.add_argument("--out", default=str(ROOT / "data/analysis/field_decks.json"))
    ap.add_argument("--manifest",
                    default=str(ROOT / "data/analysis/FIELD_DECKS.md"))
    ap.add_argument("--cache", default=str(ROOT / "data/analysis/_replay_cache"))
    args = ap.parse_args()

    p = pool()
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)

    episodes: list[dict] = []
    for sid in [int(s) for s in args.subs.split(",")]:
        for e in list_episodes(sid):
            ours = [a for a in e["agents"] if a.get("submissionId") == sid]
            them = [a for a in e["agents"] if a.get("submissionId") != sid]
            if len(ours) != 1 or len(them) != 1:
                continue
            episodes.append({
                "ep": e["id"], "our_sub": sid,
                "our_reward": ours[0].get("reward"),
                "our_score_pre": ours[0].get("initialScore"),
                "opp_sub": them[0].get("submissionId"),
                "opp_team_id": them[0].get("teamId"),
                "opp_score_pre": them[0].get("initialScore"),
            })
    print(f"{len(episodes)} episodes across {args.subs}", flush=True)

    records = []
    for i, ep in enumerate(episodes):
        f = cache / f"{ep['ep']}.json"
        if f.exists():
            rep = json.loads(f.read_text())
        else:
            rep = replay(ep["ep"])
            if rep is None:
                print(f"  {ep['ep']}: replay unavailable", flush=True)
                continue
            f.write_text(json.dumps(rep))
        out = seat_decks(rep)
        if out is None:
            print(f"  {ep['ep']}: no step-0 decklists", flush=True)
            continue
        names, decks = out
        if OUR_TEAM not in names:
            print(f"  {ep['ep']}: our seat not in {names}", flush=True)
            continue
        our_i = names.index(OUR_TEAM)
        opp_i = 1 - our_i
        opp = decks[opp_i]
        listed = sorted(opp.elements())
        rec = dict(ep)
        rec.update({
            "opp_team": names[opp_i],
            "our_seat": our_i,
            "opp_deck": {str(c): n for c, n in sorted(opp.items())},
            "opp_sig": "-".join(f"{c}x{n}" for c, n in sorted(opp.items())),
            "opp_archetype": label_archetype(opp, p),
            "opp_walls": sorted(set(opp) & WALLS),
            "opp_cards": len(listed),
            "legal": bool(len(listed) == 60 and validate(listed).legal),
            "won": ep["our_reward"] == 1,
            "band": band(ep["opp_score_pre"]),
        })
        records.append(rec)
        if (i + 1) % 10 == 0:
            print(f"  ...{i+1}/{len(episodes)}", flush=True)

    # Dedupe into distinct 60-card lists.
    decks: dict[str, dict] = {}
    for r in records:
        d = decks.setdefault(r["opp_sig"], {
            "sig": r["opp_sig"], "deck": r["opp_deck"],
            "archetype": r["opp_archetype"], "walls": r["opp_walls"],
            "legal": r["legal"], "n_episodes": 0, "wins_vs_us": 0,
            "teams": [], "bands": [], "ratings": [], "episodes": [],
        })
        d["n_episodes"] += 1
        d["wins_vs_us"] += 0 if r["won"] else 1
        d["teams"].append(r["opp_team"])
        d["bands"].append(r["band"])
        if r["opp_score_pre"] is not None:
            d["ratings"].append(round(r["opp_score_pre"], 1))
        d["episodes"].append(r["ep"])

    corpus = []
    for i, d in enumerate(sorted(decks.values(),
                                 key=lambda x: -x["n_episodes"])):
        d["deck_id"] = f"field{i:02d}"
        d["teams"] = sorted(set(d["teams"]))
        d["band_counts"] = dict(Counter(d["bands"]))
        d["rating_mean"] = (round(sum(d["ratings"]) / len(d["ratings"]), 1)
                            if d["ratings"] else None)
        d["rating_min"] = min(d["ratings"]) if d["ratings"] else None
        d["rating_max"] = max(d["ratings"]) if d["ratings"] else None
        d["opp_win_rate_vs_us"] = round(d["wins_vs_us"] / d["n_episodes"], 3)
        del d["bands"]
        corpus.append(d)

    payload = {
        "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source_submissions": args.subs,
        "our_team": OUR_TEAM,
        "n_episodes_parsed": len(records),
        "n_distinct_decks": len(corpus),
        "episodes": records,
        "decks": corpus,
    }
    Path(args.out).write_text(json.dumps(payload, indent=1))
    print(f"wrote {args.out}: {len(records)} episodes, {len(corpus)} decks")

    # Human-readable manifest.
    L = ["# Field deck census — real ladder opponents", "",
         f"Built {payload['built']} from submissions {args.subs} "
         f"({len(records)} episodes parsed, {len(corpus)} distinct 60-card "
         "lists).", "",
         "Source: every replay's step-0 `visualize` payload, which carries "
         "both seats' full decklists. Ratings are the opponent's score "
         "immediately before the episode. `opp WR` is the opponent's win rate "
         "against us on that list.", "",
         "## Distinct lists", "",
         "| id | archetype | eps | opp WR | rating mean [min,max] | bands | "
         "walls | legal | teams |",
         "|---|---|---|---|---|---|---|---|---|"]
    for d in corpus:
        bands = ", ".join(f"{k}:{v}" for k, v in sorted(d["band_counts"].items()))
        walls = ", ".join(str(w) for w in d["walls"]) or "-"
        L.append(f"| {d['deck_id']} | {d['archetype']} | {d['n_episodes']} | "
                 f"{d['opp_win_rate_vs_us']:.2f} | {d['rating_mean']} "
                 f"[{d['rating_min']},{d['rating_max']}] | {bands} | {walls} | "
                 f"{'y' if d['legal'] else 'NO'} | {'; '.join(d['teams'])[:60]} |")

    arch = Counter()
    arch_eps = Counter()
    for d in corpus:
        arch[d["archetype"]] += 1
        arch_eps[d["archetype"]] += d["n_episodes"]
    L += ["", "## Archetype census (by episodes played against us)", "",
          "| archetype | distinct lists | episodes | share |",
          "|---|---|---|---|"]
    tot = sum(arch_eps.values())
    for a, n in arch_eps.most_common():
        L.append(f"| {a} | {arch[a]} | {n} | {n/tot:.1%} |")

    b = Counter(r["band"] for r in records)
    L += ["", "## Rating-band census (episodes)", "", "| band | episodes | share |",
          "|---|---|---|"]
    for k in sorted(b):
        L.append(f"| {k} | {b[k]} | {b[k]/len(records):.1%} |")
    wall_eps = sum(1 for r in records if r["opp_walls"])
    L += ["", f"Wall-carrying opponents: {wall_eps}/{len(records)} episodes, "
          f"{sum(1 for d in corpus if d['walls'])}/{len(corpus)} distinct lists.",
          ""]
    Path(args.manifest).write_text("\n".join(L))
    print(f"wrote {args.manifest}")


if __name__ == "__main__":
    main()
