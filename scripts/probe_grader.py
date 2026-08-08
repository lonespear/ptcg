"""Does the agent's search actually run under the real grader?

The most expensive bug in this project was silent: `cg` is not importable from
inside a function body under the grader, so a lazy import raised
ModuleNotFoundError, an `except Exception` swallowed it, and every submission
for a day played pure rules while every local test passed.

That failure is invisible from inside — the agent still plays, just worse. So it
gets a standing check. This instruments the built bundle *without* depending on
its internals (it appends a wrapper after load rather than patching source), runs
it under kaggle_environments' own `cabt` environment, and reports how often the
search actually reached a decision.

    python scripts/build_submission.py && python scripts/probe_grader.py

Run it after any refactor of agent/main.py.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "build" / "submission.tar.gz"
MARKER = Path(tempfile.gettempdir()) / "ptcg_grader_probe.json"

INSTRUMENT = '''

# ---- appended by scripts/probe_grader.py ----------------------------------
import atexit as _atexit, json as _json, os as _os, tempfile as _tf

_PROBE = {"agent_calls": 0, "search_calls": 0, "search_decided": 0,
          "cg_available": bool(globals().get("CG_AVAILABLE", False)),
          "errors": {}}


def _probe_dump():
    try:
        with open(_os.path.join(_tf.gettempdir(),
                                "ptcg_grader_probe.json"), "w") as fh:
            _json.dump(_PROBE, fh)
    except Exception:
        pass


_atexit.register(_probe_dump)

_orig_agent = agent
_orig_search = globals().get("_search_main")

if _orig_search is not None:
    def _probe_search(obs, options):
        _PROBE["search_calls"] += 1
        try:
            out = _orig_search(obs, options)
        except Exception as e:
            k = type(e).__name__ + ": " + str(e)[:80]
            _PROBE["errors"][k] = _PROBE["errors"].get(k, 0) + 1
            _probe_dump()
            raise
        if out is not None:
            _PROBE["search_decided"] += 1
        return out

    _search_main = _probe_search


# This must be the LAST NEW callable name bound in the module. The grader picks
# `[v for v in env.values() if callable(v)][-1]` — the last callable by dict
# insertion order — and rebinding an existing name (like `agent`) does NOT move
# its position. So the wrapper needs a fresh name defined last, or the grader
# runs some helper as the entry point instead.
def _probe_agent(obs_dict):
    _PROBE["agent_calls"] += 1
    _probe_dump()
    return _orig_agent(obs_dict)


agent = _probe_agent
'''


def main() -> None:
    if not ARCHIVE.exists():
        sys.exit(f"{ARCHIVE} not found — run scripts/build_submission.py first")
    try:
        import kaggle_environments as ke
    except ImportError:
        sys.exit("pip install kaggle-environments")

    tmp = Path(tempfile.mkdtemp(prefix="grader_probe_"))
    cwd = os.getcwd()
    if MARKER.exists():
        MARKER.unlink()
    try:
        with tarfile.open(ARCHIVE) as tar:
            tar.extractall(tmp)
        main_py = tmp / "main.py"
        main_py.write_text(main_py.read_text(encoding="utf-8") + INSTRUMENT,
                           encoding="utf-8")

        os.chdir(tmp)
        env = ke.make("cabt", debug=True)
        env.run([str(main_py), str(main_py)])
        final = env.steps[-1]
        print("statuses:", [s["status"] for s in final],
              "rewards:", [s["reward"] for s in final])
        print("episode steps:", len(env.steps))
    finally:
        os.chdir(cwd)

    print("\n--- probe ---")
    if not MARKER.exists():
        sys.exit("FAIL: no marker written — the agent never ran")
    data = json.loads(MARKER.read_text())
    for k in ("cg_available", "agent_calls", "search_calls", "search_decided"):
        print(f"  {k:<16} {data.get(k)}")
    for k, v in (data.get("errors") or {}).items():
        print(f"  ERROR x{v}: {k}")

    shutil.rmtree(tmp, ignore_errors=True)

    if not data.get("cg_available"):
        sys.exit("\nFAIL: the engine was not importable at module level.")
    if not data.get("search_decided"):
        sys.exit("\nFAIL: search never returned a decision — the agent is "
                 "playing pure rules on the ladder.")
    rate = data["search_decided"] / max(data["search_calls"], 1)
    print(f"\nPASS: search decided {data['search_decided']} of "
          f"{data['search_calls']} calls ({rate:.0%}).")


if __name__ == "__main__":
    main()
