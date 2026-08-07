"""Falsify the bundle: prove the shipped file's machinery actually runs.

`validate_submission.py` proves an episode completes. That is not the same as
proving the agent did anything — search fired zero times on the ladder for
days behind a passing validation episode, because an `except Exception` turned
a missing import into a silent fallback to the rule policy.

So this runs the bundle the way the grader does — `exec` of the source with no
`__file__`, from the bundle directory, with the repo's own engine off
`sys.path` — and then asserts, on a real battle driven through the exec'd
`agent`:

  * `cg` resolved to the bundle's own copy and `search_begin` fired;
  * the calibration table loaded **from the bundle**, not from the repo;
  * `_pwin` is the fitted step function (monotone, and not the 0.5 that a
    missing table returns);
  * the comeback posture was consulted, and the counters say how often.

Own process on purpose: it loads the engine, and the cabt validation episode
loads its own.

    python scripts/falsify_bundle.py [--games 1]
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "build" / "submission.tar.gz"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=1)
    args = ap.parse_args()

    if not ARCHIVE.exists():
        sys.exit(f"{ARCHIVE} not found — run scripts/build_submission.py first")

    tmp = Path(tempfile.mkdtemp(prefix="ptcg_falsify_"))
    cwd = os.getcwd()
    try:
        with tarfile.open(ARCHIVE) as tar:
            tar.extractall(tmp)
        os.chdir(tmp)
        # The bundle's own cg, never the repo's — the same resolution the
        # grader gets.
        sys.path = [str(tmp)] + [p for p in sys.path
                                 if "ptcg-jonday" not in p or "build" in p]

        import cg.api as api
        from cg.game import battle_finish, battle_select, battle_start

        calls = {"search_begin": 0}
        real_begin = api.search_begin

        def counting_begin(*a, **k):
            calls["search_begin"] += 1
            return real_begin(*a, **k)

        api.search_begin = counting_begin      # bound by the exec below

        src = (tmp / "main.py").read_text(encoding="utf-8")
        g: dict = {"__name__": "kaggle_agent"}
        exec(compile(src, "main.py", "exec"), g)      # noqa: S102 — the point
        if "agent" not in g:
            sys.exit("FAIL: no `agent` callable in the bundle")

        curve = g["_load_calibration"]()
        source = g["_CALIB_SOURCE"]
        print(f"calibration: {len(curve)} breakpoints from {source}")
        if not curve:
            sys.exit("FAIL: no calibration table in the bundle — _pwin would "
                     "return 0.5 and the comeback posture would never fire")
        if not str(Path(source).resolve()).startswith(str(tmp.resolve())):
            sys.exit(f"FAIL: _pwin loaded {source}, outside the bundle")
        lo, hi = g["_pwin"](-1e6), g["_pwin"](1e6)
        print(f"_pwin: {lo:.3f} at -1e6, {hi:.3f} at +1e6, "
              f"{g['_pwin'](0):.3f} at 0")
        if not lo < hi:
            sys.exit("FAIL: _pwin is not monotone over the table")

        deck = g["read_deck_csv"]()
        for game in range(args.games):
            obs, start = battle_start(list(deck), list(deck))
            if obs is None:
                sys.exit(f"FAIL: battle_start (type {start.errorType})")
            steps = 0
            try:
                while steps < 20000:
                    sel = obs.get("select")
                    if sel is None or not sel.get("option"):
                        break
                    obs = battle_select(g["agent"](obs))
                    steps += 1
            finally:
                battle_finish()
            print(f"game {game + 1}: {steps} decisions, "
                  f"turn {obs.get('current', {}).get('turn')}")

        tel = g["TELEMETRY"]
        print(f"search_begin fired {calls['search_begin']} times")
        print(f"telemetry: {tel}")
        if calls["search_begin"] == 0:
            sys.exit("FAIL: search never fired — the agent played rules-only")
        if tel["calibration_missing"]:
            sys.exit("FAIL: _pwin was asked for with no table loaded")
        if tel["main_decisions"] == 0:
            sys.exit("FAIL: the posture was never consulted on a main menu")
        rate = tel["posture_behind"] / tel["main_decisions"]
        print(f"posture fired on {tel['posture_behind']}/"
              f"{tel['main_decisions']} main-menu picks ({rate:.3f})")
        print("\nPASS — search fires, the bundled table loads, the posture runs")
    finally:
        os.chdir(cwd)
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
