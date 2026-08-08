"""D47 Phase 1 — sibling-set training table for the neural leaf.

For every qualifying decision point, reconstruct the CANDIDATE set (the
deduped legal options the shipped search would consider), replay each
candidate to its leaf exactly as the runtime search does — determinize with
`_own_hidden` / `_deck_posterior` / `_hidden_from`, `search_begin`,
`search_step(candidate)`, `_greedy_complete` under the rules policy — and
write one row per candidate: the full NAMED feature vector
(`ptcg.leaf.features`), whether the strong player chose it, and the game
outcome. Sibling sets share a `group` id; ranking training pairs within it.

Qualifying seats (decisions-primary, the D47 contract):
  * field seats rated >= --rating-cut (default 1100), our own team excluded;
  * target-archetype seats rated >= --arch-rating-cut (default 1000),
    downsampled per archetype to --arch-target rows/day, for per-archetype
    coverage of the driving cells.

Modes:
    --day 2026-08-05 [--download]     a raw day's replays (the 20 GB dump)
    --positions                       the kept mined positions (prototype)
    --corpus PATH                     an e7 specialist corpus (.jsonl.gz)

    python -m ptcg.leaf.build_table --day 2026-08-05 --workers 8
"""

from __future__ import annotations

import argparse
import glob
import gzip
import importlib.util
import json
import os
import random
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "engine"))

from ptcg.leaf.features import FEATURE_NAMES, leaf_features  # noqa: E402
from ptcg.leaf.lags import LagTracker  # noqa: E402

OUR_TEAM = "Lemmes Yad"
OUT_ROOT = ROOT / "data" / "leaf_train"

META_COLUMNS = [
    ("group", "int64"), ("episode_id", "int64"), ("date", "str"),
    ("seat", "int8"), ("agent_name", "str"), ("agent_rating", "float64"),
    ("our_archetype", "str"), ("opp_archetype", "str"),
    ("pool", "str"), ("step", "int32"), ("turn", "int32"),
    ("context", "int32"), ("n_cand", "int32"), ("cand", "int32"),
    ("is_chosen", "int8"), ("won", "int8"), ("n_det", "int32"),
    ("terminal", "float64"), ("linear_value", "float64"),
]
COLUMNS = META_COLUMNS + [(f"f_{n}", "float64") for n in FEATURE_NAMES]

_W: dict = {}


def load_agent(name: str = "leaf_agent"):
    import ptcg.creation  # noqa: F401  — engine bootstrap
    agent_py = ROOT / "agent" / "main.py"
    cwd = os.getcwd()
    os.chdir(agent_py.parent)
    try:
        spec = importlib.util.spec_from_file_location(name, agent_py)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        os.chdir(cwd)


def _worker_init(cfg: dict) -> None:
    _W["cfg"] = cfg
    M = load_agent()
    if not M.CG_AVAILABLE:
        raise RuntimeError("cg.api did not import; leaves cannot be replayed")
    # E7a discipline: the calibration/pace tables must be loaded, not empty.
    if not M._curves():
        raise RuntimeError("trajectory curves empty — check data/ paths")
    _W["M"] = M
    from ptcg.extract import _worker_init as ext_init
    ext_init({})                      # loads the card table for archetype labels


def sibling_rows(M, obs: dict, chosen: list, lag_ctx: dict, n_det: int,
                 rng: random.Random, stats: Counter) -> list[dict] | None:
    """Rows for one decision, or None with the reason counted."""
    sel = obs.get("select") or {}
    options = sel.get("option") or []
    if len(options) < 2:
        stats["skip_forced"] += 1
        return None
    if not any(o.get("type") in M.MAIN_PRIORITY for o in options):
        stats["skip_not_main"] += 1
        return None
    if not obs.get("search_begin_input"):
        stats["skip_no_sbi"] += 1
        return None
    if not chosen:
        stats["skip_no_choice"] += 1
        return None
    try:
        o = M.to_observation_class(obs)
        me = o.current.yourIndex
    except Exception:
        stats["skip_obs_class"] += 1
        return None

    M._refresh_traj_arch(obs)
    try:
        post = M._deck_posterior(obs, top_k=M.POSTERIOR_TOP_K)
    except Exception:
        post = None
    if not post:
        stats["skip_no_posterior"] += 1
        return None
    M._evo_refresh_pools(post)

    try:
        cand, rep = M._dedup_options(obs, options)
    except Exception:
        stats["skip_dedup"] += 1
        return None
    if len(cand) < 2:
        stats["skip_one_cand"] += 1
        return None
    ch = rep.get(chosen[0], chosen[0])
    if ch not in cand:
        stats["skip_choice_unmapped"] += 1
        return None

    try:
        seen, n_deck, n_hand, n_prize = M._opponent_counts(obs)
    except Exception:
        stats["skip_opp_counts"] += 1
        return None

    from cg.api import search_step, search_end
    counts = post[0][0]
    acc = {i: None for i in cand}
    val = {i: 0.0 for i in cand}
    term = {i: 0.0 for i in cand}
    nev = {i: 0 for i in cand}
    M._SEARCH_ME = me
    try:
        for _ in range(n_det):
            try:
                my_deck, my_prize = M._own_hidden(obs, rng)
                opp_deck, opp_hand, opp_prize = M._hidden_from(
                    counts, seen, n_deck, n_hand, n_prize, rng)
                root = M.search_begin(o, my_deck, my_prize, opp_deck,
                                      opp_prize, opp_hand, [])
            except Exception:
                stats["det_begin_error"] += 1
                continue
            try:
                for i in cand:
                    try:
                        child = search_step(root.searchId, [i])
                        leaf = M._greedy_complete(child, me,
                                                  M._rules_choice_for)
                    except Exception:
                        stats["cand_step_error"] += 1
                        continue
                    lo = leaf.observation
                    cur = lo.current
                    res = getattr(cur, "result", -1)
                    if res is not None and res != -1:
                        term[i] += 1.0 if res == me else -1.0
                        v = 1e6 if res == me else -1e6
                    else:
                        v = M._margin(cur, me)
                    f = leaf_features(M, lo, me, lag_ctx)
                    a = acc[i]
                    if a is None:
                        acc[i] = list(f)
                    else:
                        for k in range(len(f)):
                            a[k] += f[k]
                    val[i] += v
                    nev[i] += 1
            finally:
                try:
                    search_end()
                except Exception:
                    pass
    finally:
        M._SEARCH_ME = None

    live = [i for i in cand if nev[i] > 0]
    if ch not in live or len(live) < 2:
        stats["skip_replay_dead"] += 1
        return None
    rows = []
    for i in live:
        n = nev[i]
        row = {"cand": i, "is_chosen": int(i == ch), "n_det": n,
               "terminal": term[i] / n, "linear_value": val[i] / n}
        f = acc[i]
        for k, name in enumerate(FEATURE_NAMES):
            row[f"f_{name}"] = f[k] / n
        rows.append(row)
    stats["decisions"] += 1
    stats["rows"] += len(rows)
    return rows


def _qualify(cfg: dict, name: str, rating: float | None,
             arch: str) -> tuple[str | None, float]:
    """(pool, keep_probability) for a seat."""
    if name == OUR_TEAM or rating is None:
        return None, 0.0
    if rating >= cfg["rating_cut"]:
        return "field", 1.0
    if rating >= cfg["arch_rating_cut"]:
        p = cfg["arch_keep"].get(arch, 0.0)
        if p > 0:
            return "arch", p
    return None, 0.0


def process_episode(path: str) -> tuple[list[dict], dict] | None:
    """Worker: all training rows for one replay file."""
    M = _W["M"]
    cfg = _W["cfg"]
    stats: Counter = Counter()
    try:
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
    except Exception:
        return None
    steps = d.get("steps") or []
    info = d.get("info") or {}
    agents = list(info.get("TeamNames") or [])
    if not steps or len(agents) != 2:
        return None
    from ptcg.extract import label_archetype_fast, _winner, _first_player
    try:
        vis = steps[0][0]["visualize"][0]["action"]
        decks = [list(x) for x in vis if isinstance(x, list)]
    except (KeyError, IndexError, TypeError):
        return None
    if len(decks) != 2:
        return None
    winner = _winner(d.get("rewards") or [])
    arch = [label_archetype_fast(Counter(decks[0])),
            label_archetype_fast(Counter(decks[1]))]
    ratings = cfg["ratings"]
    pools = {}
    for s in (0, 1):
        pool, p = _qualify(cfg, agents[s], ratings.get(agents[s]), arch[s])
        if pool:
            pools[s] = (pool, p)
    if not pools:
        return [], dict(stats)

    ep_id = int(info.get("EpisodeId") or d.get("id") or 0)
    rng = random.Random(ep_id ^ cfg["seed"])
    won = {0: -1 if winner is None else int(winner == 0),
           1: -1 if winner is None else int(winner == 1)}
    trackers = {s: LagTracker() for s in pools}
    kept: list[dict] = []
    n_seat = Counter()
    for t, seat, obs, sel, chosen in _iter_decisions_cached(steps):
        if seat in pools:
            trackers[seat].observe(obs, (obs.get("current") or {})
                                   .get("yourIndex", seat), seat)
        if seat not in pools:
            continue
        pool, p = pools[seat]
        if n_seat[seat] >= cfg["per_seat_cap"]:
            continue
        if p < 1.0 and rng.random() > p:
            continue
        cur = obs.get("current") or {}
        me = cur.get("yourIndex", seat)
        if me not in (0, 1):
            me = seat
        M._MY_DECK = decks[seat]
        M._EVO_POOLS["us"] = None
        M._EVO_EDGES["us"] = {}
        rows = sibling_rows(M, obs, chosen, trackers[seat].ctx(
            cur.get("turn")), cfg["n_det"], rng, stats)
        if not rows:
            continue
        n_seat[seat] += 1
        group = ep_id * 1000 + t
        for r in rows:
            r.update(group=group, episode_id=ep_id, date=cfg["date"],
                     seat=seat, agent_name=agents[seat],
                     agent_rating=float(ratings.get(agents[seat]) or 0.0),
                     our_archetype=arch[seat], opp_archetype=arch[1 - seat],
                     pool=pool, step=t, turn=int(cur.get("turn") or -1),
                     context=int(sel.get("context", -1)),
                     n_cand=len(rows), won=won[seat])
        kept.extend(rows)
    return kept, dict(stats)


def _iter_decisions_cached(steps):
    from ptcg.extract import iter_decisions
    return iter_decisions(steps)


# ---------------------------------------------------------------- day mode

def arch_keep_probs(date: str, cfg: dict) -> dict:
    """Per-archetype keep probability for the arch-extension pool, sized off
    the day's already-mined decision counts so workers need no coordination."""
    import pandas as pd
    path = ROOT / "data" / "mined" / date / "decisions.parquet"
    if not path.exists():
        return {}
    d = pd.read_parquet(path, columns=[
        "agent_rating", "n_options", "context", "our_archetype", "agent_name"])
    q = d[(d.agent_rating >= cfg["arch_rating_cut"])
          & (d.agent_rating < cfg["rating_cut"])
          & (d.n_options >= 2) & (d.context == 0)
          & (d.agent_name != OUR_TEAM)]
    counts = q.our_archetype.value_counts().to_dict()
    out = {}
    for a in cfg["target_archs"]:
        n = counts.get(a, 0)
        if n:
            out[a] = min(1.0, cfg["arch_target"] / n)
    return out


def specialist_arch_labels() -> list[str]:
    """Archetype labels of the harvested specialist decks + our own."""
    from ptcg.extract import label_archetype_fast, _worker_init as ext_init
    ext_init({})
    labels = set()
    for p in glob.glob(str(ROOT / "external" / "*_deck.json")):
        try:
            deck = json.load(open(p))
            labels.add(label_archetype_fast(Counter(int(c) for c in deck)))
        except Exception:
            continue
    labels.add("Teal Mask Ogerpon ex")
    labels.discard("(no Pokémon)")
    return sorted(labels)


def run_day(date: str, args) -> None:
    from ptcg.extract import load_ratings, iter_episode_files
    from ptcg.leaf.build_table import RowSink   # self, for clarity
    if args.download:
        import kagglehub
        print(f"downloading pokemon-tcg-ai-battle-episodes-{date} (~20 GB)")
        kagglehub.dataset_download(f"kaggle/pokemon-tcg-ai-battle-episodes-{date}")
    files = [str(p) for p in iter_episode_files(date)]
    if args.limit:
        files = files[:args.limit]
    if not files:
        print(f"no episode files for {date} — download first?")
        sys.exit(1)
    ratings = load_ratings(date)
    targets = specialist_arch_labels()
    cfg = {
        "date": date, "seed": args.seed, "n_det": args.n_det,
        "rating_cut": args.rating_cut, "arch_rating_cut": args.arch_rating_cut,
        "arch_target": args.arch_target, "target_archs": targets,
        "per_seat_cap": args.per_seat_cap,
        "ratings": dict(ratings.lb),
    }
    cfg["arch_keep"] = arch_keep_probs(date, cfg)
    print(f"{date}: {len(files)} episodes, {len(cfg['ratings'])} rated teams, "
          f"arch pools: { {k: round(v, 3) for k, v in cfg['arch_keep'].items()} }")

    outdir = OUT_ROOT / date
    outdir.mkdir(parents=True, exist_ok=True)
    sink = RowSink(outdir)
    stats: Counter = Counter()
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers,
                             initializer=_worker_init,
                             initargs=(cfg,)) as pool:
        for k, res in enumerate(pool.map(process_episode, files,
                                         chunksize=2), 1):
            if res is None:
                stats["bad_episode"] += 1
                continue
            rows, st = res
            stats.update(st)
            sink.write(rows)
            if k % 200 == 0:
                el = time.time() - t0
                print(f"  {k}/{len(files)} eps  {stats['decisions']} dec  "
                      f"{stats['rows']} rows  {el:.0f}s  "
                      f"({stats['decisions'] / max(el, 1):.1f} dec/s)",
                      flush=True)
    sink.close()
    meta = {"date": date, "files": len(files), "stats": dict(stats),
            "cfg": {k: v for k, v in cfg.items() if k != "ratings"},
            "n_features": len(FEATURE_NAMES),
            "seconds": round(time.time() - t0, 1)}
    (outdir / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in stats.items()
                      if not k.startswith("f_")}, indent=1))
    print(f"wrote {sink.n} rows -> {sink.path}")
    if args.purge:
        cache = (Path.home() / ".cache" / "kagglehub" / "datasets" / "kaggle"
                 / f"pokemon-tcg-ai-battle-episodes-{date}")
        if cache.exists():
            import shutil
            shutil.rmtree(cache, ignore_errors=True)
            print(f"purged {cache}")


class RowSink:
    def __init__(self, outdir: Path, name: str = "rows"):
        from ptcg.extract import RowWriter
        self._w = RowWriter(outdir, columns=COLUMNS, name=name)
        self.n = 0

    @property
    def path(self):
        return self._w.path

    def write(self, rows):
        self.n += len(rows)
        self._w.write(rows)

    def close(self):
        self._w.close()


# ---------------------------------------------- positions / corpus modes

def run_positions(args) -> None:
    """Prototype pass over the kept mined positions (no download needed)."""
    _worker_init({"date": "positions", "n_det": args.n_det})
    M = _W["M"]
    from ptcg.extract import label_archetype_fast
    priors = json.load(open(ROOT / "agent" / "deck_priors.json"))["decks"]
    best_list: dict = {}
    for d in priors:
        a = d.get("a")
        if a and (a not in best_list or int(d["p"]) > best_list[a][0]):
            cts = {int(k): int(v) for k, v in d["c"].items()}
            best_list[a] = (int(d["p"]),
                            [c for c, n in cts.items() for _ in range(n)])
    outdir = OUT_ROOT / "positions_proto"
    outdir.mkdir(parents=True, exist_ok=True)
    sink = RowSink(outdir)
    stats: Counter = Counter()
    rng = random.Random(args.seed)
    t0 = time.time()
    n_seen = 0
    for p in sorted(glob.glob(str(ROOT / "data/mined/*/positions.jsonl.gz"))):
        for line in gzip.open(p):
            r = json.loads(line)
            if (r.get("agent_rating") or 0) < args.rating_cut:
                continue
            bl = best_list.get(r.get("our_archetype"))
            if bl is None:
                stats["no_selfmodel"] += 1
                continue
            n_seen += 1
            if args.limit and n_seen > args.limit:
                break
            obs = r["observation"]
            M._MY_DECK = bl[1]
            M._EVO_POOLS["us"] = None
            M._EVO_EDGES["us"] = {}
            rows = sibling_rows(M, obs, r.get("chosen") or [],
                                {"depth": 0}, args.n_det, rng, stats)
            if not rows:
                continue
            cur = obs.get("current") or {}
            group = int(r["episode_id"]) * 1000 + int(r["step"])
            for row in rows:
                row.update(group=group, episode_id=int(r["episode_id"]),
                           date=r["date"], seat=int(r.get("seat") or 0),
                           agent_name=r.get("agent_name") or "",
                           agent_rating=float(r.get("agent_rating") or 0),
                           our_archetype=r.get("our_archetype") or "",
                           opp_archetype=r.get("opp_archetype") or "",
                           pool="positions", step=int(r["step"]),
                           turn=int(cur.get("turn") or -1),
                           context=int(r.get("context") or -1),
                           n_cand=len(rows), won=int(r.get("won", -1)))
            sink.write(rows)
        if args.limit and n_seen > args.limit:
            break
    sink.close()
    el = time.time() - t0
    print(json.dumps(dict(stats), indent=1))
    print(f"wrote {sink.n} rows in {el:.0f}s "
          f"({stats['decisions'] / max(el, 1):.2f} dec/s) -> {sink.path}")


def run_corpus(path: str, args) -> None:
    """An e7 specialist corpus: the specialist's own-deck decisions."""
    _worker_init({"date": "corpus", "n_det": args.n_det})
    M = _W["M"]
    fh = gzip.open(path, "rt")
    header = json.loads(fh.readline())
    cell = header.get("cell", Path(path).stem)
    stem = Path(path).name.replace(".jsonl.gz", "")
    name = stem if stem.startswith("selfplay_") else f"corpus_{cell}"
    deck = [int(c) for c in header["deck"]]
    outdir = OUT_ROOT / name
    outdir.mkdir(parents=True, exist_ok=True)
    sink = RowSink(outdir)
    stats: Counter = Counter()
    rng = random.Random(args.seed)
    t0 = time.time()
    tracker = LagTracker()
    last_seed = None
    n_seen = 0
    for line in fh:
        r = json.loads(line)
        obs = r["observation"]
        if r.get("seed") != last_seed:
            tracker.reset()
            last_seed = r.get("seed")
        cur = obs.get("current") or {}
        me = cur.get("yourIndex", 0)
        tracker.observe(obs, me, me)
        n_seen += 1
        if args.limit and n_seen > args.limit:
            break
        M._MY_DECK = deck
        M._EVO_POOLS["us"] = None
        M._EVO_EDGES["us"] = {}
        rows = sibling_rows(M, obs, r.get("chosen") or [],
                            tracker.ctx(cur.get("turn")), args.n_det, rng,
                            stats)
        if not rows:
            continue
        import hashlib
        h = hashlib.md5(f"{name}|{r.get('seed')}|{r.get('i')}".encode())
        group = int.from_bytes(h.digest()[:6], "big")  # deterministic id
        for row in rows:
            row.update(group=group, episode_id=int(r.get("seed") or 0),
                       date=name, seat=0,
                       agent_name=f"specialist_{cell}",
                       agent_rating=0.0, our_archetype=cell,
                       opp_archetype="Teal Mask Ogerpon ex",
                       pool="specialist", step=int(r.get("i") or 0),
                       turn=int(cur.get("turn") or -1),
                       context=int(r.get("ctx") or -1),
                       n_cand=len(rows), won=int(r.get("won", -1)))
        sink.write(rows)
    sink.close()
    el = time.time() - t0
    print(json.dumps(dict(stats), indent=1))
    print(f"wrote {sink.n} rows in {el:.0f}s "
          f"({stats['decisions'] / max(el, 1):.2f} dec/s) -> {sink.path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--day")
    ap.add_argument("--positions", action="store_true")
    ap.add_argument("--corpus")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--purge", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--n-det", type=int, default=2)
    ap.add_argument("--rating-cut", type=float, default=1100.0)
    ap.add_argument("--arch-rating-cut", type=float, default=1000.0)
    ap.add_argument("--arch-target", type=float, default=6000.0)
    ap.add_argument("--per-seat-cap", type=int, default=40)
    ap.add_argument("--seed", type=int, default=47)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if args.positions:
        run_positions(args)
    elif args.corpus:
        run_corpus(args.corpus, args)
    elif args.day:
        run_day(args.day, args)
    else:
        ap.error("one of --day / --positions / --corpus")


if __name__ == "__main__":
    main()
