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
    ap.add_argument("--opponent", default="external/grimmsnarl/deck.csv",
                    help="field decklist for the other seat; \"\" for a mirror")
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

        # C1/C2: the projection tables have to come out of the bundle too, and
        # the feature has to return a number on a real position rather than
        # fall through its own guard to a silent zero.
        curves, accel = g["_curves"](), g["_accel_rates"]()
        print(f"trajectory curves: {len(curves)} cells from "
              f"{g['_CURVES_SOURCE']}")
        print(f"energy KB: {len(accel)} accelerators from {g['_ACCEL_SOURCE']}")
        if not curves:
            sys.exit("FAIL: no trajectory curves in the bundle — the C2 "
                     "weight would be spent on a flat field rate")
        if not accel:
            sys.exit("FAIL: no energy-mechanics KB in the bundle")
        # Playbook entry 4: the opponent reply table. Asserted only while the
        # agent is configured to read it — SEARCH_OPP_BRANCH is 0 today, the
        # entry having been measured and refused, and the matching assertion is
        # then the other one: that a table nothing reads is not in the archive.
        sources = [("curves", g["_CURVES_SOURCE"]),
                   ("energy KB", g["_ACCEL_SOURCE"])]
        branch = g["SEARCH_OPP_BRANCH"]
        pol = g["_opp_policy"]()
        if branch >= 1:
            print(f"opponent policy: {len(pol.get('cells') or {})} cells, band "
                  f"{g['_opp_band']()}, from {g['_OPP_SOURCE']}")
            if not pol:
                sys.exit("FAIL: SEARCH_OPP_BRANCH is on and there is no "
                         "opponent_policy.json in the bundle — every rollout "
                         "would play their turn under our own priority table, "
                         "which is the configuration three gates rejected")
            sources.append(("opponent policy", g["_OPP_SOURCE"]))
        elif (tmp / "opponent_policy.json").exists():
            sys.exit("FAIL: opponent_policy.json is in the bundle and "
                     "SEARCH_OPP_BRANCH is 0, so nothing will ever open it")
        else:
            print("opponent policy: correctly absent (SEARCH_OPP_BRANCH = 0)")

        for name, src in sources:
            if not str(Path(src).resolve()).startswith(str(tmp.resolve())):
                sys.exit(f"FAIL: {name} loaded {src}, outside the bundle")

        deck = g["read_deck_csv"]()
        # The opponent is a field list, not our own: a same-deck mirror leaves
        # both sides' best affordable attack identical, so the C2 differential
        # is 0 on 98% of positions and asserting it fired would assert nothing.
        # This is a decklist, not code — the bundle's own `cg` and `main.py`
        # are still the only things being exercised.
        opp = deck
        if args.opponent:
            try:
                text = (ROOT / args.opponent).read_text()
                ids = [int(x) for x in text.replace(",", "\n").split()
                       if x.strip()]
                if len(ids) == 60:
                    opp = ids
            except Exception:
                print(f"  (no {args.opponent}; falling back to a mirror)")
        for game in range(args.games):
            obs, start = battle_start(list(deck), list(opp))
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
        traj = g["TELEMETRY_TRAJ"]
        opp = g["TELEMETRY_OPP"]
        print(f"search_begin fired {calls['search_begin']} times")
        print(f"telemetry: {tel}")
        print(f"trajectory: {traj}")
        print(f"opponent: {opp}")
        if traj["feature_errors"]:
            sys.exit(f"FAIL: the C2 feature raised {traj['feature_errors']} "
                     f"times and scored 0.0 behind its guard")
        if traj["curves_missing"]:
            sys.exit("FAIL: the projection fell back to the flat field rate")
        if traj["threat_nonzero"] == 0:
            sys.exit(f"FAIL: the C2 term never moved a margin over "
                     f"{traj['threat_scored']} scored positions")
        if calls["search_begin"] == 0:
            sys.exit("FAIL: search never fired — the agent played rules-only")
        if g["SEARCH_OPP_BRANCH"] >= 1:
            if opp["branch_replies"] == 0:
                sys.exit("FAIL: 2-ply is on and the table was never asked for "
                         "a reply ordering")
            if opp["consulted"] == 0:
                sys.exit("FAIL: the table answered no opponent main menu")
            if opp["policy_missing"]:
                sys.exit(f"FAIL: the table was asked for "
                         f"{opp['policy_missing']} times with nothing loaded")
            miss = opp["miss"] / max(opp["miss"] + sum(
                opp["hit_" + lv] for lv in ("L0", "L1", "L2", "L3")), 1)
            print(f"opponent table answered {opp['consulted']} main menus, "
                  f"{opp['branch_replies']} branch orderings, "
                  f"cell miss rate {miss:.4f}")
            if miss > 0.05:
                sys.exit(f"FAIL: {miss:.3f} of opponent positions found no "
                         f"cell — the backoff is not covering the field")
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
