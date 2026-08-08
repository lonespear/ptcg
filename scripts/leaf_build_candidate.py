"""D47 Phase 3 — assemble the neural-leaf candidate bundle.

Copies the shipped agent (agent/main.py is NEVER edited; the patches land on
the copy), vendors the leaf package flat beside it (leaf_features / leaf_lags
/ leaf_runtime + the trained model), and applies four anchored patches:

  1. neural-leaf setup block before `_evaluate` (env CABT_NEURAL_LEAF,
     default ON in this bundle; linear spine untouched behind the flag);
  2. `_evaluate`'s final `return _margin(...)` gains the net branch with the
     linear margin as in-place fallback on any error;
  3. `_agent` feeds the lag tracker once per decision;
  4. `_refresh_traj_arch` refreshes the evolution pools for the leaf's evo
     features even when CABT_EVO_INTEGRAL is off; and the episode-open call
     resets the tracker.

Every anchor must match exactly once or the build aborts — a silent partial
patch is how E7a happened.

    python scripts/leaf_build_candidate.py --model data/leaf_train/model_h64_multi.npz \
        --out build/agents/neural_leaf_v1
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SETUP = '''

# --- D47 neural leaf (candidate bundle patch; agent/main.py never edited) ---
try:
    NEURAL_LEAF_ENABLED = bool(int(os.environ.get("CABT_NEURAL_LEAF") or 1))
except Exception:
    NEURAL_LEAF_ENABLED = True
_NLEAF = None
_LEAF_TRACKER = None
_LEAF_LAG_CTX = {"depth": 0}
_SELF_MODULE = sys.modules[__name__]
TELEMETRY_NLEAF = {"scored": 0, "errors": 0}
if NEURAL_LEAF_ENABLED:
    try:
        if _HERE and _HERE not in sys.path:
            sys.path.insert(0, _HERE)
        from leaf_runtime import NeuralLeaf as _NeuralLeaf
        from leaf_features import leaf_features as _leaf_features
        from leaf_lags import LagTracker as _LeafLagTracker
        for _p in _bundle_paths(("leaf_model.npz",)):
            try:
                _NLEAF = _NeuralLeaf(_p)
                break
            except Exception:
                continue
        _LEAF_TRACKER = _LeafLagTracker()
    except Exception:
        _NLEAF = None
    if _NLEAF is None:
        NEURAL_LEAF_ENABLED = False


def _evaluate(observation, me: int) -> float:
'''

PATCHES = [
    # 2. the net branch inside _evaluate, linear margin as fallback
    ("    return _margin(cur, me)",
     """    if NEURAL_LEAF_ENABLED and _NLEAF is not None:
        try:
            _f = _leaf_features(_SELF_MODULE, observation, me, _LEAF_LAG_CTX)
            TELEMETRY_NLEAF["scored"] += 1
            return _NLEAF.score_features(_f)
        except Exception:
            TELEMETRY_NLEAF["errors"] += 1
    return _margin(cur, me)"""),
    # 3. lag tracker feed, once per decision
    ("    _refresh_traj_arch(obs)",
     """    _refresh_traj_arch(obs)
    if NEURAL_LEAF_ENABLED and _LEAF_TRACKER is not None:
        try:
            _cur_d = obs.get("current") or {}
            _me_d = _cur_d.get("yourIndex", 0)
            _LEAF_TRACKER.observe(obs, _me_d, _me_d)
            globals()["_LEAF_LAG_CTX"] = _LEAF_TRACKER.ctx(_cur_d.get("turn"))
        except Exception:
            pass"""),
    # 4a. evo pools refresh for the leaf's evolution features
    ("        if EVO_INTEGRAL_ENABLED:",
     "        if EVO_INTEGRAL_ENABLED or NEURAL_LEAF_ENABLED:"),
    # 4b. tracker reset on the episode-open call
    ("                _reset_bank()",
     """                _reset_bank()
                if _LEAF_TRACKER is not None:
                    _LEAF_TRACKER.reset()"""),
]

BUNDLE_FILES = ["deck.csv", "deck_priors.json", "attack_scalers.json"]
DATA_FILES = {
    "calibration_v2.json": ROOT / "data" / "calibration_v2.json",
    "calibration.json": ROOT / "data" / "calibration.json",
    "energy_mechanics.json": ROOT / "data" / "energy_mechanics.json",
    "opponent_policy.json": ROOT / "data" / "opponent_policy.json",
    "trajectory_curves.json": ROOT / "data" / "analysis" / "trajectory_curves.json",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    src = (ROOT / "agent" / "main.py").read_text()

    anchor = "def _evaluate(observation, me: int) -> float:\n"
    if src.count(anchor) != 1:
        sys.exit(f"anchor not unique: {anchor!r}")
    src = src.replace(anchor, SETUP.lstrip("\n"), 1)
    for old, new in PATCHES:
        if src.count(old) != 1:
            sys.exit(f"anchor count {src.count(old)} != 1: {old!r}")
        src = src.replace(old, new, 1)
    (out / "main.py").write_text(src)

    for name in BUNDLE_FILES:
        shutil.copy(ROOT / "agent" / name, out / name)
    for name, p in DATA_FILES.items():
        if p.exists():
            shutil.copy(p, out / name)
    shutil.copy(ROOT / "ptcg" / "leaf" / "features.py", out / "leaf_features.py")
    shutil.copy(ROOT / "ptcg" / "leaf" / "lags.py", out / "leaf_lags.py")
    shutil.copy(ROOT / "ptcg" / "leaf" / "runtime.py", out / "leaf_runtime.py")
    shutil.copy(args.model, out / "leaf_model.npz")
    print(f"built {out}")


if __name__ == "__main__":
    main()
