"""One-pass extractor bus over episode replays.

Each daily dump is ~20 GB and is deleted after parsing, so every question we
want answered from a day has to be answered on the single pass that day gets.
This module makes that pass a bus: each episode is parsed once, and every
registered extractor sees it once and emits its own rows.

Workers parse and emit per-episode payloads; the parent routes payloads to the
extractors, which stream to disk. Nothing accumulates in memory except the
2,000-position reservoir.

    python -m ptcg.extract 2026-08-05 --workers 2          # already downloaded
    python scripts/mine_day.py 2026-08-05 --extract        # download, extract, delete

Per mined day, `data/mined/<date>/`:
    decisions.parquet   one row per decision (csv.gz when pyarrow is missing)
    series.parquet      one row per (episode, seat, turn): the board as that
                        seat's turn opened, for turn-to-turn trajectories
    positions.jsonl.gz  2,000 full observations, weighted toward high ratings
    meta.json           episode counts, rating spread, filter survival at cuts

Schema facts this rests on, verified on 2026-08-05 replays:
  * the decision made against `steps[t][seat].observation.select` is recorded at
    `steps[t+1][seat].action`, as a list of indices into `select.option`
    (388/388 decisions in-range and inside min/max; the same-step reading is
    invalid for a third of decisions);
  * the acting seat is the one with `status == "ACTIVE"` — the idle seat carries
    a stale `select`;
  * `observation.logs` and `observation.search_begin_input` are both present on
    the acting seat's observation;
  * there is no rating anywhere in the replay JSON. Seat ratings therefore come
    from a leaderboard snapshot joined on team name, which is a *current*
    rating, not the rating at game time.
"""

from __future__ import annotations

import argparse
import gzip
import heapq
import json
import math
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ptcg.episodes import iter_episode_files  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MINED = DATA / "mined"

CUTS = (800, 900, 1000, 1050)
COMPETITION = "pokemon-tcg-ai-battle"

# Option and select vocabularies, same constants the agent plays against
# (agent/main.py). Kept as a local copy so extraction never imports the engine.
OPTION_TYPE_NAMES = {
    0: "number", 1: "yes", 2: "no", 3: "card", 4: "tool_card", 5: "energy_card",
    6: "energy", 7: "play", 8: "attach", 9: "evolve", 10: "ability",
    11: "discard", 12: "retreat", 13: "attack", 14: "end_turn",
}
SELECT_TYPE_NAMES = {0: "main", 1: "card", 4: "energy", 8: "count", 9: "yes_no"}

DECISION_COLUMNS = [
    ("episode_id", "int64"), ("date", "str"), ("seat", "int8"),
    ("agent_name", "str"), ("opp_name", "str"),
    ("agent_rating", "float64"), ("opp_rating", "float64"),
    ("agent_day_wr", "float64"), ("agent_day_games", "float64"),
    ("step", "int32"), ("turn", "int32"),
    ("context", "int32"), ("select_type", "int32"), ("select_type_name", "str"),
    ("n_options", "int32"), ("min_count", "int32"), ("max_count", "int32"),
    ("chosen", "str"), ("n_chosen", "int32"),
    ("chosen_option_type", "int32"), ("chosen_option_type_name", "str"),
    ("our_archetype", "str"), ("opp_archetype", "str"),
    ("prizes_mine", "int32"), ("prizes_theirs", "int32"),
    ("went_first", "int8"), ("won", "int8"),
]

# One row per (episode, seat, turn): the board as it stood when that seat's own
# turn opened -- `seat_turn` is the seat's turn ordinal, `turn` the game's.
# Everything here is read off the observation's true state
# (`current.players`), never reconstructed from the action stream, so Energy is
# what is sitting on Pokemon rather than a count of attachments made. The two
# damage columns are damage standing on the board at that moment: a knockout
# clears the Pokemon and its damage, so they run low by whatever the knockouts
# were worth, and the prize columns are the complete knockout record.
SERIES_COLUMNS = [
    ("episode_id", "int64"), ("date", "str"), ("seat", "int8"),
    ("agent_name", "str"), ("opp_name", "str"),
    ("agent_rating", "float64"), ("opp_rating", "float64"),
    ("turn", "int32"), ("seat_turn", "int32"), ("step", "int32"),
    ("went_first", "int8"),
    ("our_archetype", "str"), ("opp_archetype", "str"),
    ("energy_in_play", "int32"), ("board_hp", "int32"),
    ("bench_count", "int32"), ("mons_in_play", "int32"),
    ("hand_count", "int32"), ("prizes_remaining", "int32"),
    ("damage_dealt_cumulative", "int32"),
    ("opp_energy_in_play", "int32"), ("opp_board_hp", "int32"),
    ("opp_bench_count", "int32"), ("opp_mons_in_play", "int32"),
    ("opp_hand_count", "int32"), ("opp_prizes_remaining", "int32"),
    ("damage_taken_cumulative", "int32"),
    ("won", "int8"),
]


# --------------------------------------------------------------------------
# per-episode parsing (runs in workers)
# --------------------------------------------------------------------------

_ARCH: dict[int, tuple[bool, float, str]] = {}
_CFG: dict = {}


def _worker_init(cfg: dict) -> None:
    global _ARCH, _CFG
    _CFG = cfg
    import pandas as pd
    from ptcg import load_cards
    cards, _ = load_cards()
    for r in cards.itertuples():
        hp = getattr(r, "hp", None)
        is_pkmn = getattr(r, "is_pokemon", False)
        _ARCH[int(r.card_id)] = (
            False if pd.isna(is_pkmn) else bool(is_pkmn),
            float("nan") if hp is None or pd.isna(hp) else float(hp),
            str(r.name),
        )


def label_archetype_fast(deck: Counter) -> str:
    """Name a deck by its highest-HP Pokémon — `ptcg.meta.label_archetype`."""
    best, best_hp = None, -1.0
    for cid in deck:
        info = _ARCH.get(int(cid))
        if info is None:
            continue
        is_pkmn, hp, name = info
        if not is_pkmn or hp != hp:
            continue
        if hp > best_hp:
            best, best_hp = name, hp
    return best or "(no Pokémon)"


def iter_decisions(steps: list):
    """Yield (step_index, seat, observation, select, chosen_indices).

    The action for the position observed at step t is recorded at step t+1.
    """
    for t in range(len(steps) - 1):
        for seat, ag in enumerate(steps[t]):
            if ag.get("status") != "ACTIVE":
                continue
            obs = ag.get("observation") or {}
            sel = obs.get("select")
            if not sel or not sel.get("option"):
                continue
            nxt = steps[t + 1]
            action = nxt[seat].get("action") if seat < len(nxt) else None
            if not isinstance(action, list):
                action = []
            yield t, seat, obs, sel, [i for i in action if isinstance(i, int)]


def _first_player(steps: list) -> int:
    """Seat that took the first turn, from the earliest board state that has it."""
    for st in steps[:8]:
        for ag in st:
            cur = (ag.get("observation") or {}).get("current")
            if isinstance(cur, dict):
                fp = cur.get("firstPlayer")
                if isinstance(fp, int) and fp >= 0:
                    return fp
    return -1


def _side_state(p: dict) -> dict:
    """Board totals for one player, read off the observation's true state.

    `damage` is the damage standing on that player's Pokemon right now. A
    knocked-out Pokemon leaves the board, so damage summed this way understates
    the damage a side has absorbed over the game by whatever the knockouts were
    worth; the prize counters are the complete record of knockouts.
    """
    energy = hp = dmg = mons = 0
    for zone in ("active", "bench"):
        for mon in (p.get(zone) or []):
            if not isinstance(mon, dict):
                continue
            mons += 1
            energy += len(mon.get("energies") or [])
            h = mon.get("hp")
            mh = mon.get("maxHp")
            if isinstance(h, (int, float)):
                hp += int(h)
                if isinstance(mh, (int, float)):
                    dmg += int(mh) - int(h)
    return {"energy": energy, "hp": hp, "mons": mons,
            "bench": len(p.get("bench") or []), "damage": dmg,
            "hand": int(p.get("handCount") or 0),
            "prizes": len(p.get("prize") or [])}


def _winner(rewards: list) -> int | None:
    if len(rewards) != 2:
        return None
    a, b = rewards
    if a is None and b is None:
        return None
    if a is None:
        return 1 if b is not None else None
    if b is None:
        return 0
    return None if a == b else (0 if a > b else 1)


def episode_payload(path) -> dict | None:
    """Parse one replay and emit every extractor's rows for it."""
    cfg = _CFG
    date = cfg.get("date", "")
    want_dec = cfg.get("decisions", True)
    want_pos = cfg.get("positions", True)
    want_ser = cfg.get("series", True)
    pos_per_ep = cfg.get("pos_per_episode", 2)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError, MemoryError):
        return None

    steps = d.get("steps") or []
    info = d.get("info") or {}
    if not steps:
        return None
    try:
        vis = steps[0][0]["visualize"][0]["action"]
        decks = [Counter(x) for x in vis
                 if isinstance(x, list) and all(isinstance(i, int) for i in x)]
    except (KeyError, IndexError, TypeError):
        decks = []
    agents = list(info.get("TeamNames") or [])
    rewards = list(d.get("rewards") or [])
    winner = _winner(rewards)
    if len(decks) != 2 or len(agents) != 2:
        return None

    arch = [label_archetype_fast(decks[0]), label_archetype_fast(decks[1])]
    fp = _first_player(steps)
    won = [int(winner == 0) if winner is not None else -1,
           int(winner == 1) if winner is not None else -1]

    out = {
        "episode_id": int(info.get("EpisodeId") or d.get("id") or 0),
        "date": date,
        "agents": agents,
        "archetypes": arch,
        "first_player": fp,
        "winner": -1 if winner is None else winner,
        "n_steps": len(steps),
        "decks": [dict(decks[0]), dict(decks[1])],
        "rows": [],
        "positions": [],
        "series": [],
        "n_decisions": 0,
    }

    if not (want_dec or want_pos or want_ser):
        return out

    rng = random.Random(out["episode_id"])
    seen = 0
    rows = out["rows"]
    series = out["series"]
    turn_seen = [set(), set()]   # game-turn numbers already recorded per seat
    for t, seat, obs, sel, chosen in iter_decisions(steps):
        seen += 1
        cur = obs.get("current") or {}
        players = cur.get("players") or []
        me = cur.get("yourIndex", seat)
        if not isinstance(me, int) or me not in (0, 1):
            me = seat
        pm = pt = -1
        if len(players) == 2:
            pm = len(players[me].get("prize") or [])
            pt = len(players[1 - me].get("prize") or [])
        # The state at the top of this seat's own turn: the first decision the
        # seat is asked for inside a turn that belongs to it. Turns alternate
        # from the first player, so turn 1, 3, 5 ... are the first player's;
        # a seat is also asked to decide *during* the opponent's turn (choosing
        # a new Active after a knockout, for one), and those decisions are not
        # a turn of its own. Setup decisions carry turn -1 and are not a turn.
        turn_no = int(cur.get("turn") or -1)
        owns_turn = fp >= 0 and ((turn_no % 2 == 1) == (fp == seat))
        if (want_ser and turn_no > 0 and owns_turn and len(players) == 2
                and turn_no not in turn_seen[seat]):
            turn_seen[seat].add(turn_no)
            # Turn ordinal for this seat, straight off the alternation, so a
            # turn with no decision leaves a hole rather than shifting the index.
            seat_turn = (turn_no + 1) // 2 if fp == seat else turn_no // 2
            mine = _side_state(players[me])
            theirs = _side_state(players[1 - me])
            series.append({
                "episode_id": out["episode_id"], "date": date, "seat": seat,
                "agent_name": agents[seat], "opp_name": agents[1 - seat],
                "agent_rating": None, "opp_rating": None,
                "turn": turn_no, "seat_turn": seat_turn, "step": t,
                "went_first": int(fp == seat) if fp >= 0 else -1,
                "our_archetype": arch[seat], "opp_archetype": arch[1 - seat],
                "energy_in_play": mine["energy"], "board_hp": mine["hp"],
                "bench_count": mine["bench"], "mons_in_play": mine["mons"],
                "hand_count": mine["hand"], "prizes_remaining": mine["prizes"],
                "damage_dealt_cumulative": theirs["damage"],
                "opp_energy_in_play": theirs["energy"],
                "opp_board_hp": theirs["hp"],
                "opp_bench_count": theirs["bench"],
                "opp_mons_in_play": theirs["mons"],
                "opp_hand_count": theirs["hand"],
                "opp_prizes_remaining": theirs["prizes"],
                "damage_taken_cumulative": mine["damage"],
                "won": won[seat],
            })
        opts = sel["option"]
        ctype = -1
        if chosen and 0 <= chosen[0] < len(opts):
            ctype = int(opts[chosen[0]].get("type", -1))
        if want_dec:
            rows.append({
                "episode_id": out["episode_id"], "date": date, "seat": seat,
                "agent_name": agents[seat], "opp_name": agents[1 - seat],
                "agent_rating": None, "opp_rating": None,
                "agent_day_wr": None, "agent_day_games": None,
                "step": t, "turn": int(cur.get("turn") or -1),
                "context": int(sel.get("context", -1)),
                "select_type": int(sel.get("type", -1)),
                "select_type_name": SELECT_TYPE_NAMES.get(
                    sel.get("type"), str(sel.get("type"))),
                "n_options": len(opts),
                "min_count": int(sel.get("minCount", -1)),
                "max_count": int(sel.get("maxCount", -1)),
                "chosen": "|".join(str(i) for i in chosen),
                "n_chosen": len(chosen),
                "chosen_option_type": ctype,
                "chosen_option_type_name": OPTION_TYPE_NAMES.get(ctype, str(ctype)),
                "our_archetype": arch[seat], "opp_archetype": arch[1 - seat],
                "prizes_mine": pm, "prizes_theirs": pt,
                "went_first": int(fp == seat) if fp >= 0 else -1,
                "won": won[seat],
            })
        # Candidate positions: a few per episode, whittled to the day's 2,000
        # by weighted reservoir in the parent (which is where ratings live).
        if want_pos and len(out["positions"]) < pos_per_ep and rng.random() < 0.02:
            out["positions"].append({
                "meta": {
                    "episode_id": out["episode_id"], "date": date, "seat": seat,
                    "agent_name": agents[seat], "opp_name": agents[1 - seat],
                    "our_archetype": arch[seat], "opp_archetype": arch[1 - seat],
                    "step": t, "turn": int(cur.get("turn") or -1),
                    "context": int(sel.get("context", -1)),
                    "chosen": chosen, "won": won[seat],
                    "went_first": int(fp == seat) if fp >= 0 else -1,
                },
                "obs": json.dumps(obs, ensure_ascii=False),
            })
    out["n_decisions"] = seen
    return out


# --------------------------------------------------------------------------
# writers
# --------------------------------------------------------------------------

class RowWriter:
    """Streams rows to parquet, or to csv.gz when pyarrow is absent."""

    def __init__(self, outdir: Path, columns=DECISION_COLUMNS,
                 name: str = "decisions"):
        self.columns = [c for c, _ in columns]
        self.n = 0
        self._pq = None
        self._csv = None
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
            types = {"int64": pa.int64(), "int32": pa.int32(), "int8": pa.int8(),
                     "float64": pa.float64(), "str": pa.string()}
            self._pa = pa
            self.schema = pa.schema([(c, types[t]) for c, t in columns])
            self.path = outdir / f"{name}.parquet"
            self._pq = pq.ParquetWriter(self.path, self.schema,
                                        compression="zstd")
        except ImportError:
            import csv
            self.path = outdir / f"{name}.csv.gz"
            self._fh = gzip.open(self.path, "wt", newline="", encoding="utf-8")
            self._csv = csv.DictWriter(self._fh, fieldnames=self.columns)
            self._csv.writeheader()

    def write(self, rows: list[dict]) -> None:
        if not rows:
            return
        self.n += len(rows)
        if self._pq is not None:
            cols = {c: [r.get(c) for r in rows] for c in self.columns}
            self._pq.write_table(
                self._pa.Table.from_pydict(cols, schema=self.schema))
        else:
            self._csv.writerows(rows)

    def close(self) -> None:
        if self._pq is not None:
            self._pq.close()
        else:
            self._fh.close()


# --------------------------------------------------------------------------
# extractors (parent side)
# --------------------------------------------------------------------------

class Extractor:
    name = "extractor"

    def collect(self, payload: dict) -> None: ...

    def finalize(self, outdir: Path, ctx: dict) -> dict:
        return {}


class Decisions(Extractor):
    name = "decisions"

    def __init__(self, outdir: Path, ratings: "Ratings"):
        self.writer = RowWriter(outdir)
        self.ratings = ratings
        self.buf: list[dict] = []

    def collect(self, payload: dict) -> None:
        rows = payload["rows"]
        if not rows:
            return
        a, b = payload["agents"]
        ra, rb = self.ratings.get(a), self.ratings.get(b)
        wa, wb = self.ratings.day(a), self.ratings.day(b)
        for r in rows:
            first = r["seat"] == 0
            r["agent_rating"], r["opp_rating"] = (ra, rb) if first else (rb, ra)
            r["agent_day_wr"], r["agent_day_games"] = (wa if first else wb)
        self.buf.extend(rows)
        if len(self.buf) >= 20000:
            self.writer.write(self.buf)
            self.buf = []

    def finalize(self, outdir: Path, ctx: dict) -> dict:
        self.writer.write(self.buf)
        self.buf = []
        self.writer.close()
        return {"decisions_rows": self.writer.n,
                "decisions_file": self.writer.path.name}


class Series(Extractor):
    """Per-turn board state, one row per (episode, seat, turn).

    This is the empirical trajectory table: how Energy, board HP, bench width,
    hand and prizes actually move turn to turn, which is what any projection of
    the next few turns has to be fitted and validated against.
    """
    name = "series"

    def __init__(self, outdir: Path, ratings: "Ratings"):
        self.writer = RowWriter(outdir, columns=SERIES_COLUMNS, name="series")
        self.ratings = ratings
        self.buf: list[dict] = []
        self.episodes = 0
        self.turns_per_seat: Counter = Counter()

    def collect(self, payload: dict) -> None:
        rows = payload.get("series") or []
        if not rows:
            return
        self.episodes += 1
        a, b = payload["agents"]
        ra, rb = self.ratings.get(a), self.ratings.get(b)
        for r in rows:
            first = r["seat"] == 0
            r["agent_rating"], r["opp_rating"] = (ra, rb) if first else (rb, ra)
        for s in (0, 1):
            n = sum(1 for r in rows if r["seat"] == s)
            if n:
                self.turns_per_seat[n] += 1
        self.buf.extend(rows)
        if len(self.buf) >= 20000:
            self.writer.write(self.buf)
            self.buf = []

    def finalize(self, outdir: Path, ctx: dict) -> dict:
        self.writer.write(self.buf)
        self.buf = []
        self.writer.close()
        counts = sorted(self.turns_per_seat.elements())
        med = counts[len(counts) // 2] if counts else None
        return {"series_rows": self.writer.n,
                "series_file": self.writer.path.name,
                "series_episodes": self.episodes,
                "series_seat_turns_median": med,
                "series_seat_turns_mean": (
                    round(sum(counts) / len(counts), 2) if counts else None)}


class Positions(Extractor):
    """Weighted reservoir (A-Res) of full observations, 2,000 per day.

    Weight doubles every +100 rating, so a 1200-rated seat is 4x as likely to
    be kept as a 1000-rated one; unrated seats sample as if 1000.
    """
    name = "positions"

    def __init__(self, outdir: Path, ratings: "Ratings", k: int = 2000,
                 seed: int = 0):
        self.outdir = outdir
        self.ratings = ratings
        self.k = k
        self.heap: list[tuple[float, int, dict]] = []
        self.rng = random.Random(seed)
        self.tie = 0
        self.seen = 0

    def collect(self, payload: dict) -> None:
        for cand in payload["positions"]:
            self.seen += 1
            r = self.ratings.get(cand["meta"]["agent_name"])
            w = 2.0 ** (((r if r is not None else 1000.0) - 1000.0) / 100.0)
            w = min(max(w, 1e-6), 1e6)
            key = self.rng.random() ** (1.0 / w)
            cand["meta"]["agent_rating"] = r
            cand["meta"]["sample_weight"] = round(w, 4)
            self.tie += 1
            if len(self.heap) < self.k:
                heapq.heappush(self.heap, (key, self.tie, cand))
            elif key > self.heap[0][0]:
                heapq.heapreplace(self.heap, (key, self.tie, cand))

    def finalize(self, outdir: Path, ctx: dict) -> dict:
        path = outdir / "positions.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            for _, _, cand in sorted(self.heap, key=lambda x: -x[0]):
                head = json.dumps(cand["meta"], ensure_ascii=False)
                fh.write(head[:-1] + ',"observation":' + cand["obs"] + "}\n")
        return {"positions_kept": len(self.heap),
                "positions_candidates": self.seen}


class FirstPlayer(Extractor):
    """Who went first, and did they win — overall and per archetype."""
    name = "first_player"

    def __init__(self):
        self.n = 0
        self.decided = 0
        self.first_wins = 0
        self.unknown_first = 0
        self.by_arch: dict[str, list[int]] = {}
        self.by_pair: dict[str, list[int]] = {}

    def collect(self, payload: dict) -> None:
        self.n += 1
        fp, w = payload["first_player"], payload["winner"]
        if fp < 0:
            self.unknown_first += 1
            return
        if w < 0:
            return
        self.decided += 1
        win = int(w == fp)
        self.first_wins += win
        a = payload["archetypes"]
        e = self.by_arch.setdefault(a[fp], [0, 0])
        e[0] += 1
        e[1] += win
        key = f"{a[fp]} vs {a[1 - fp]}"
        p = self.by_pair.setdefault(key, [0, 0])
        p[0] += 1
        p[1] += win

    def summary(self) -> dict:
        def rate(w, n):
            return round(w / n, 4) if n else None
        arch = {k: {"games_first": v[0], "wins_first": v[1],
                    "wr_first": rate(v[1], v[0])}
                for k, v in sorted(self.by_arch.items(),
                                   key=lambda kv: -kv[1][0])}
        pair = {k: {"games": v[0], "wins_first": v[1], "wr_first": rate(v[1], v[0])}
                for k, v in sorted(self.by_pair.items(),
                                   key=lambda kv: -kv[1][0])[:40]}
        return {"episodes": self.n, "decided": self.decided,
                "first_player_wins": self.first_wins,
                "first_player_wr": rate(self.first_wins, self.decided),
                "unknown_first_player": self.unknown_first,
                "by_archetype_going_first": arch,
                "top_pairings_going_first": pair}

    def finalize(self, outdir: Path, ctx: dict) -> dict:
        s = self.summary()
        (outdir / "first_player.json").write_text(
            json.dumps(s, indent=2, ensure_ascii=False), encoding="utf-8")
        merge_first_player(ctx["date"], s)
        return {"first_player": {k: s[k] for k in
                                 ("episodes", "decided", "first_player_wins",
                                  "first_player_wr", "unknown_first_player")}}


class MetaCounts(Extractor):
    """Episode counts, the rating spread, and survival at the four cuts."""
    name = "meta"

    def __init__(self, ratings: "Ratings"):
        self.ratings = ratings
        self.episodes = 0
        self.decisions = 0
        self.seat_ratings: list[float] = []
        self.rated_seats = 0
        self.unrated_seats = 0
        self.one_ep = {c: 0 for c in CUTS}
        self.both_ep = {c: 0 for c in CUTS}
        self.one_dec = {c: 0 for c in CUTS}
        self.both_dec = {c: 0 for c in CUTS}
        self.agent_games: Counter = Counter()

    def collect(self, payload: dict) -> None:
        self.episodes += 1
        nd = payload["n_decisions"]
        self.decisions += nd
        a, b = payload["agents"]
        self.agent_games[a] += 1
        self.agent_games[b] += 1
        ra, rb = self.ratings.get(a), self.ratings.get(b)
        for r in (ra, rb):
            if r is None:
                self.unrated_seats += 1
            else:
                self.rated_seats += 1
                self.seat_ratings.append(r)
        lo = min(ra if ra is not None else -1, rb if rb is not None else -1)
        hi = max(ra if ra is not None else -1, rb if rb is not None else -1)
        # Decisions split by seat only matters for the one-sided cut; the two
        # seats make within a decision or two of the same number of decisions.
        for c in CUTS:
            if hi >= c:
                self.one_ep[c] += 1
                over = sum(1 for r in (ra, rb) if r is not None and r >= c)
                self.one_dec[c] += int(nd * over / 2)
            if lo >= c:
                self.both_ep[c] += 1
                self.both_dec[c] += nd

    def summary(self) -> dict:
        rs = sorted(self.seat_ratings)
        def q(p):
            if not rs:
                return None
            i = min(len(rs) - 1, max(0, int(round(p * (len(rs) - 1)))))
            return round(rs[i], 1)
        dist = {"n": len(rs), "min": q(0), "p10": q(.10), "p25": q(.25),
                "median": q(.50), "p75": q(.75), "p90": q(.90), "max": q(1.0),
                "mean": round(sum(rs) / len(rs), 1) if rs else None}
        surv = {}
        for c in CUTS:
            surv[str(c)] = {
                "one_sided_episodes": self.one_ep[c],
                "one_sided_episode_pct": round(self.one_ep[c] / self.episodes, 4)
                if self.episodes else None,
                "both_sided_episodes": self.both_ep[c],
                "both_sided_episode_pct": round(self.both_ep[c] / self.episodes, 4)
                if self.episodes else None,
                "one_sided_decisions": self.one_dec[c],
                "both_sided_decisions": self.both_dec[c],
            }
        return {"episodes": self.episodes, "decisions": self.decisions,
                "seats_rated": self.rated_seats,
                "seats_unrated": self.unrated_seats,
                "distinct_agents": len(self.agent_games),
                "rating_distribution": dist, "filter_survival": surv}


@dataclass
class Ratings:
    """Per-agent rating, joined on team name from a leaderboard snapshot.

    The replay JSON carries no rating, so this is a *current* rating standing in
    for the rating at game time. Day-level win rate from `history_agents.csv` is
    carried alongside it as the rating-free fallback.
    """
    lb: dict[str, float] = field(default_factory=dict)
    daywr: dict[str, tuple[float, float]] = field(default_factory=dict)
    source: str = "none"

    def get(self, agent: str) -> float | None:
        return self.lb.get(agent)

    def day(self, agent: str) -> tuple[float | None, float | None]:
        return self.daywr.get(agent, (None, None))


def leaderboard_snapshot(refresh_hours: float = 24.0) -> tuple[dict, str]:
    """Team -> rating from the competition leaderboard, cached on disk."""
    path = DATA / "leaderboard_snapshot.csv"
    fresh = (path.exists()
             and time.time() - path.stat().st_mtime < refresh_hours * 3600)
    if not fresh:
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            import zipfile
            api = KaggleApi()
            api.authenticate()
            tmp = DATA / "_lb"
            tmp.mkdir(exist_ok=True)
            api.competition_leaderboard_download(COMPETITION, path=str(tmp))
            zp = sorted(tmp.glob("*.zip"))[-1]
            with zipfile.ZipFile(zp) as zf:
                name = zf.namelist()[0]
                path.write_bytes(zf.read(name))
            for f in tmp.iterdir():
                f.unlink()
            tmp.rmdir()
        except KeyboardInterrupt:
            raise
        except BaseException as e:
            # BaseException, not Exception: the current kaggle client
            # sys.exit()s when it finds no credentials, and SystemExit would
            # sail past an Exception handler and kill the caller. A missing
            # login must degrade to the cached snapshot like any other failure.
            print(f"  leaderboard refresh failed ({type(e).__name__}: "
                  f"{str(e)[:120]}) — falling back to any cached snapshot")
    if not path.exists():
        return {}, "none"
    import pandas as pd
    df = pd.read_csv(path)
    m = {str(t): float(s) for t, s in zip(df["TeamName"], df["Score"])}
    stamp = datetime.fromtimestamp(path.stat().st_mtime,
                                   timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    return m, f"kaggle leaderboard snapshot {stamp} (current rating, not at game time)"


def load_ratings(date: str) -> Ratings:
    lb, src = leaderboard_snapshot()
    daywr: dict[str, tuple[float, float]] = {}
    agents_csv = DATA / "history_agents.csv"
    if agents_csv.exists():
        import pandas as pd
        df = pd.read_csv(agents_csv)
        df = df[df["date"].astype(str) == date] if date in set(
            df["date"].astype(str)) else df
        g = df.groupby("agent").agg(games=("games", "sum"), wins=("wins", "sum"))
        daywr = {str(a): (float(r.wins / r.games) if r.games else None,
                          float(r.games)) for a, r in g.iterrows()}
    return Ratings(lb=lb, daywr=daywr, source=src)


def merge_first_player(date: str, day: dict) -> Path:
    """Keep one cross-day first-player file, recomputed from the day blocks."""
    path = MINED / "first_player.json"
    doc = {"days": {}}
    if path.exists():
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            doc = {"days": {}}
    doc.setdefault("days", {})[date] = day
    decided = sum(v["decided"] for v in doc["days"].values())
    wins = sum(v["first_player_wins"] for v in doc["days"].values())
    arch: dict[str, list[int]] = {}
    for v in doc["days"].values():
        for a, e in v.get("by_archetype_going_first", {}).items():
            row = arch.setdefault(a, [0, 0])
            row[0] += e["games_first"]
            row[1] += e["wins_first"]
    doc["overall"] = {
        "days": sorted(doc["days"]),
        "episodes": sum(v["episodes"] for v in doc["days"].values()),
        "decided": decided, "first_player_wins": wins,
        "first_player_wr": round(wins / decided, 4) if decided else None,
        "by_archetype_going_first": {
            a: {"games_first": v[0], "wins_first": v[1],
                "wr_first": round(v[1] / v[0], 4) if v[0] else None}
            for a, v in sorted(arch.items(), key=lambda kv: -kv[1][0])},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    return path


def print_first_player(s: dict, top: int = 12) -> None:
    wr = s["first_player_wr"]
    n = s["decided"]
    se = math.sqrt(wr * (1 - wr) / n) if wr and n else 0.0
    print(f"\n  first player wins {s['first_player_wins']}/{n} = "
          f"{wr:.4f} +/- {1.96 * se:.4f} (95%)")
    print(f"  {'archetype going first':<34}{'n':>7}{'wr':>8}")
    for a, e in list(s["by_archetype_going_first"].items())[:top]:
        if e["games_first"] < 30:
            continue
        print(f"  {a[:33]:<34}{e['games_first']:>7}{e['wr_first']:>8.3f}")


# --------------------------------------------------------------------------
# the bus
# --------------------------------------------------------------------------

@dataclass
class DayResult:
    date: str
    outdir: Path
    files: int = 0
    parsed: int = 0
    skipped: int = 0
    meta: dict = field(default_factory=dict)
    deck_rows: list = field(default_factory=list)


def run_day(date: str, outdir: Path | None = None, workers: int = 2,
            limit: int | None = None, positions: int = 2000,
            pos_per_episode: int = 2, want: tuple[str, ...] = (
                "decisions", "series", "positions", "first_player", "meta"),
            deck_rows: bool = False, files: list | None = None,
            chunksize: int = 4) -> DayResult:
    """Run every extractor over one day's replays in a single pass."""
    outdir = Path(outdir or (MINED / date))
    outdir.mkdir(parents=True, exist_ok=True)
    paths = [str(p) for p in (files if files is not None
                              else iter_episode_files(date))]
    if limit:
        paths = paths[:limit]
    res = DayResult(date=date, outdir=outdir, files=len(paths))
    if not paths:
        print(f"  no episode files found for {date}")
        return res

    ratings = load_ratings(date)
    print(f"  ratings: {len(ratings.lb)} teams from {ratings.source}")

    exts: list[Extractor] = []
    dec = ser = pos = fpx = met = None
    if "decisions" in want:
        dec = Decisions(outdir, ratings)
        exts.append(dec)
    if "series" in want:
        ser = Series(outdir, ratings)
        exts.append(ser)
    if "positions" in want:
        pos = Positions(outdir, ratings, k=positions)
        exts.append(pos)
    if "first_player" in want:
        fpx = FirstPlayer()
        exts.append(fpx)
    if "meta" in want:
        met = MetaCounts(ratings)
        exts.append(met)

    cfg = {"date": date, "decisions": "decisions" in want,
           "positions": "positions" in want, "series": "series" in want,
           "pos_per_episode": pos_per_episode}
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init,
                             initargs=(cfg,)) as pool:
        for i, payload in enumerate(pool.map(episode_payload, paths,
                                             chunksize=chunksize), 1):
            if payload is None:
                res.skipped += 1
                continue
            res.parsed += 1
            for e in exts:
                e.collect(payload)
            if deck_rows:
                for s in (0, 1):
                    if payload["winner"] >= 0:
                        res.deck_rows.append({
                            "agent": payload["agents"][s],
                            "won": int(payload["winner"] == s),
                            "n_steps": payload["n_steps"],
                            "deck": payload["decks"][s]})
            payload["rows"] = payload["positions"] = payload["series"] = None
            if i % 250 == 0:
                el = time.perf_counter() - t0
                print(f"    {i}/{len(paths)} episodes  {el:.0f}s  "
                      f"({i / el:.1f}/s, eta {(len(paths) - i) / (i / el) / 60:.0f}m)",
                      flush=True)

    ctx = {"date": date}
    meta = {"date": date,
            "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "files": res.files, "parsed": res.parsed, "skipped": res.skipped,
            "seconds": round(time.perf_counter() - t0, 1),
            "rating_source": ratings.source,
            "workers": workers}
    for e in exts:
        meta.update(e.finalize(outdir, ctx))
    if met is not None:
        meta.update(met.summary())
    (outdir / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    res.meta = meta

    print(f"  parsed {res.parsed}/{res.files} episodes in "
          f"{meta['seconds']:.0f}s -> {outdir}")
    if ser is not None:
        print(f"  series: {meta['series_rows']} rows over "
              f"{meta['series_episodes']} episodes, median "
              f"{meta['series_seat_turns_median']} turns per seat")
    if met is not None:
        print_survival(meta)
    if fpx is not None:
        print_first_player(fpx.summary())
    return res


def print_survival(meta: dict) -> None:
    d = meta.get("rating_distribution") or {}
    print(f"  seat ratings: n={d.get('n')} unrated={meta.get('seats_unrated')} "
          f"p25={d.get('p25')} median={d.get('median')} p75={d.get('p75')} "
          f"max={d.get('max')}")
    print(f"  {'cut':>6}{'1-sided eps':>13}{'%':>8}{'2-sided eps':>13}{'%':>8}"
          f"{'1-sided dec':>13}{'2-sided dec':>13}")
    for c in CUTS:
        s = meta["filter_survival"][str(c)]
        print(f"  {c:>6}{s['one_sided_episodes']:>13}"
              f"{(s['one_sided_episode_pct'] or 0) * 100:>7.1f}%"
              f"{s['both_sided_episodes']:>13}"
              f"{(s['both_sided_episode_pct'] or 0) * 100:>7.1f}%"
              f"{s['one_sided_decisions']:>13}{s['both_sided_decisions']:>13}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dates", nargs="+", help="YYYY-MM-DD, already downloaded")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N episodes (smoke test)")
    ap.add_argument("--positions", type=int, default=2000)
    ap.add_argument("--out", default=None, help="output root (default data/mined)")
    ap.add_argument("--extractors",
                    default="decisions,series,positions,first_player,meta")
    args = ap.parse_args()
    want = tuple(x.strip() for x in args.extractors.split(",") if x.strip())
    for date in args.dates:
        print(f"\n=== extract {date} ===")
        out = Path(args.out) / date if args.out else None
        run_day(date, outdir=out, workers=args.workers, limit=args.limit,
                positions=args.positions, want=want)


if __name__ == "__main__":
    main()
