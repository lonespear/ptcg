"""Download one day of episodes, mine it, append to history, then delete it.

Each daily dump is ~20 GB, so days are processed one at a time and removed once
their aggregates are safely appended.

    python scripts/mine_day.py 2026-08-05
    python scripts/mine_day.py --back 5          # 5 most recent days, newest first
    python scripts/mine_day.py 2026-08-05 --keep # don't delete after mining
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ptcg import load_cards
from ptcg.episodes import parse_episode
from ptcg.meta import aggregate_day, aggregate_decks

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

CARDS_CSV = DATA / "history_cards.csv"
ARCH_CSV = DATA / "history_archetypes.csv"
AGENTS_CSV = DATA / "history_agents.csv"
DECKS_CSV = DATA / "history_decklists.csv"

CACHE = Path.home() / ".cache" / "kagglehub" / "datasets" / "kaggle"
DS_PREFIX = "pokemon-tcg-ai-battle-episodes-"


def _job(path):
    ep = parse_episode(path)
    if ep is None or ep.winner is None or len(ep.agents) != 2:
        return None
    return [
        {"agent": ep.agents[i], "won": int(ep.winner == i),
         "n_steps": ep.n_steps, "deck": dict(ep.decks[i])}
        for i in (0, 1)
    ]


def already_done(date: str) -> bool:
    if not ARCH_CSV.exists():
        return False
    return date in set(pd.read_csv(ARCH_CSV)["date"].astype(str))


def download_day(date: str) -> Path | None:
    import kagglehub
    slug = f"kaggle/{DS_PREFIX}{date}"
    print(f"  downloading {slug} (~20 GB)...")
    try:
        return Path(kagglehub.dataset_download(slug))
    except Exception as e:
        print(f"  download failed: {type(e).__name__}: "
              f"{str(e).replace(chr(10), ' ')[:160]}")
        return None


def purge_day(date: str) -> None:
    """Delete a day's downloaded episodes — and nothing else.

    Guarded so this can only ever remove a directory that lives in the kagglehub
    dataset cache AND is named for an episode dump of this exact date.
    """
    target = CACHE / f"{DS_PREFIX}{date}"
    resolved = target.resolve()
    if not resolved.exists():
        return
    if CACHE.resolve() not in resolved.parents:
        print(f"  refusing to delete outside the kagglehub cache: {resolved}")
        return
    if resolved.name != f"{DS_PREFIX}{date}":
        print(f"  refusing to delete unexpected path: {resolved}")
        return
    size = sum(f.stat().st_size for f in resolved.rglob("*") if f.is_file())
    shutil.rmtree(resolved, ignore_errors=True)
    print(f"  deleted {resolved.name} ({size / 1024**3:.1f} GB freed)")


def append(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        return
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def mine(date: str, by_id: pd.DataFrame, keep: bool,
         decks_only: bool = False, extract: bool = False,
         workers: int | None = None, positions: int = 2000) -> bool:
    print(f"\n=== {date} ===")
    in_history = already_done(date)
    if in_history and not decks_only and not extract:
        print("  already in history — skipping")
        return True
    if decks_only and DECKS_CSV.exists():
        done = set(pd.read_csv(DECKS_CSV)["date"].astype(str))
        if date in done:
            print("  decklists already captured — skipping")
            return True

    path = download_day(date)
    if path is None:
        return False

    files = sorted(path.glob("*.json"))
    print(f"  parsing {len(files)} episodes...")
    t0 = time.perf_counter()
    rows: list[dict] = []
    if extract:
        # One pass, every extractor: the decision rows, the sampled positions
        # and the day's counts come off the same parse as the deck instances.
        from ptcg.extract import run_day
        res = run_day(date, workers=workers or 2, files=files,
                      positions=positions, deck_rows=True)
        rows = res.deck_rows
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for res in pool.map(_job, files, chunksize=16):
                if res:
                    rows.extend(res)
    print(f"  {len(rows)} deck instances in {time.perf_counter() - t0:.0f}s")

    if in_history and extract:
        print("  day already in history — extractor output only, no re-append")
        if not keep:
            purge_day(date)
        return True

    cards, arch, agents = aggregate_day(rows, by_id, date)
    if not decks_only:
        append(cards, CARDS_CSV)
        append(arch, ARCH_CSV)
        append(agents, AGENTS_CSV)
    else:
        print("  decks-only: not re-appending card/archetype/agent rows")

    # Keep only decklists with enough games to mean something — storing all
    # ~10k instances per day would bloat the history for no extra signal.
    decks = aggregate_decks(rows, by_id, date)
    if not decks.empty:
        append(decks[decks["decks"] >= 5], DECKS_CSV)
        print(f"  {len(decks)} distinct decklists "
              f"({int((decks['decks'] >= 5).sum())} with >=5 games kept)")

    if not arch.empty:
        top = arch.assign(wr=arch["wins"] / arch["decks"]).nlargest(5, "decks")
        print("  most-played archetypes:")
        for _, r in top.iterrows():
            print(f"    {r['archetype']:<30} n={int(r['decks']):5d}  "
                  f"wr={r['wr']:.3f}")
    if not agents.empty:
        best = agents[agents["games"] >= 60].assign(
            wr=lambda d: d["wins"] / d["games"]).nlargest(5, "wr")
        print("  top agents (>=60 games):")
        for _, r in best.iterrows():
            print(f"    {r['agent']:<28} {int(r['games']):5d} games  "
                  f"wr={r['wr']:.3f}")

    if not keep:
        purge_day(date)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dates", nargs="*", help="YYYY-MM-DD")
    ap.add_argument("--back", type=int, default=0,
                    help="mine the N most recent days, newest first")
    ap.add_argument("--keep", action="store_true", help="don't delete after mining")
    ap.add_argument("--decks-only", action="store_true",
                    help="capture decklists for a day already aggregated "
                         "(the earliest days were mined before decklists were)")
    ap.add_argument("--extract", action="store_true",
                    help="run the extractor bus over the day as well: "
                         "decisions, sampled positions, meta (data/mined/<date>)")
    ap.add_argument("--workers", type=int, default=None,
                    help="parse workers (default: every core; use 2 when this "
                         "machine is also running a GA)")
    ap.add_argument("--positions", type=int, default=2000,
                    help="full observations sampled per day with --extract")
    args = ap.parse_args()

    os.environ.setdefault("KAGGLE_API_TOKEN", os.environ.get("KAGGLE_API_TOKEN", ""))
    cards, _ = load_cards()
    by_id = cards.set_index("card_id")

    dates = list(args.dates)
    if args.back:
        import kagglehub
        idx = Path(kagglehub.dataset_download(
            "kaggle/pokemon-tcg-ai-battle-episodes-index"))
        man = pd.read_csv(idx / "manifest.csv")
        dates = (man.sort_values("date", ascending=False)["date"]
                 .astype(str).head(args.back).tolist())
        print(f"mining {len(dates)} most recent days: {dates}")

    for d in dates:
        if not mine(d, by_id, args.keep, args.decks_only, args.extract,
                    args.workers, args.positions):
            print(f"  stopping at {d}")
            break

    print("\nhistory files:")
    for p in (CARDS_CSV, ARCH_CSV, AGENTS_CSV, DECKS_CSV):
        if p.exists():
            print(f"  {p.relative_to(ROOT)}  ({len(pd.read_csv(p))} rows)")


if __name__ == "__main__":
    main()
