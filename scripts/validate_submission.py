"""Run the real Kaggle validation episode locally, before submitting.

Two submissions were burned on `Validation Episode failed` because a hand-rolled
local test cannot reproduce the grader. The grader **exec()s** main.py rather
than importing it, so `__file__` is undefined — and a normal import always
defines it, which is why the bug was invisible locally.

`kaggle_environments` ships the competition's own `cabt` environment, so the
actual validation (our agent against a copy of itself) runs here:

    python scripts/validate_submission.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "build" / "submission.tar.gz"


def main() -> None:
    if not ARCHIVE.exists():
        sys.exit(f"{ARCHIVE} not found — run scripts/build_submission.py first")
    try:
        import kaggle_environments as ke
    except ImportError:
        sys.exit("kaggle_environments not installed — pip install kaggle-environments")

    tmp = Path(tempfile.mkdtemp(prefix="ptcg_val_"))
    cwd = os.getcwd()
    try:
        with tarfile.open(ARCHIVE) as tar:
            tar.extractall(tmp)
        root_entries = sorted(p.name for p in tmp.iterdir())
        print(f"bundle root: {root_entries}")
        for required in ("main.py", "deck.csv", "cg", "calibration.json"):
            if required not in root_entries:
                sys.exit(f"FAIL: {required} must be at the archive root")

        # Explicit encoding: the file contains "Pokémon", and Windows would
        # otherwise decode it as cp1252 and raise — which silently skipped this
        # whole gate once.
        # The grader resolves the entry point as the LAST callable by dict
        # insertion order — `[v for v in env.values() if callable(v)][-1]` — not
        # by the name `agent`. So a helper defined after agent() silently
        # becomes the submitted agent, and the episode dies as INVALID. This is
        # one edit away at all times, so it is checked rather than remembered.
        import ast
        src_path = tmp / "main.py"
        tree = ast.parse(src_path.read_text(encoding="utf-8"))
        funcs = [n.name for n in tree.body
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        if funcs and funcs[-1] != "agent":
            sys.exit(f"FAIL: agent() must be the last function defined in "
                     f"main.py, but {funcs[-1]}() comes after it. The grader "
                     f"would run {funcs[-1]}() as the agent.")
        print("  agent() is the last callable — grader will pick it")

        src = src_path.read_text(encoding="utf-8")
        if "__file__" in src and "except NameError" not in src:
            sys.exit("FAIL: main.py uses __file__ without a NameError guard — "
                     "the grader exec()s the file and __file__ is undefined")

        os.chdir(tmp)
        env = ke.make("cabt", debug=True)
        main_py = str(tmp / "main.py")
        env.run([main_py, main_py])

        final = env.steps[-1]
        ok = True
        for i, s in enumerate(final):
            status, reward = s["status"], s["reward"]
            print(f"  agent {i}: status={status} reward={reward}")
            if status not in ("DONE", "INACTIVE"):
                ok = False

        # Surface anything the agents printed to stderr — that is where a
        # swallowed exception would show up.
        errs = [e for log in (getattr(env, "logs", None) or []) for e in log
                if isinstance(e, dict) and e.get("stderr")]
        for e in errs[:5]:
            print("  stderr:", e["stderr"][:400])
            ok = False

        print(f"  episode length: {len(env.steps)} steps")
        if not ok:
            sys.exit("\nFAIL — the validation episode would be rejected")
        print("\nPASS — validation episode completed cleanly")
    finally:
        os.chdir(cwd)
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
