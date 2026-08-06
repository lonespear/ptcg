"""Bundle the agent into a submittable archive for the Simulation competition.

    python scripts/build_submission.py                 # build only
    python scripts/build_submission.py --submit -m "heuristic v1"

The archive holds main.py and deck.csv at its root plus the `cg` package copied
from engine/ (the competition runtime imports `cg`, and the sample submission
ships it alongside the agent, so we do the same).

Nothing here is committed — build/ is gitignored, because the bundle contains
the licensed engine library.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGENT = ROOT / "agent"
ENGINE = ROOT / "engine"
BUILD = ROOT / "build"
COMPETITION = "pokemon-tcg-ai-battle"


def write_tables(dest: Path) -> None:
    """Bake the engine's attack/HP tables into the bundle.

    The agent must not import `cg` at run time: the grader already has the
    engine loaded, and a second import re-runs GameInitialize() against the
    live battle. So the numbers get extracted here, at build time, instead.
    """
    import json
    sys.path.insert(0, str(ENGINE))
    from cg.sim import lib

    attacks = json.loads(lib.AllAttack().decode())
    cards = json.loads(lib.AllCard().decode())
    data = {
        "attack_damage": {str(a["attackId"]): (a.get("damage") or 0)
                          for a in attacks},
        "card_hp": {str(c["cardId"]): (c.get("hp") or 0) for c in cards
                    if c.get("hp")},
    }
    dest.write_text(json.dumps(data, separators=(",", ":")))
    print(f"  baked {len(data['attack_damage'])} attacks, "
          f"{len(data['card_hp'])} card HP values "
          f"({dest.stat().st_size / 1024:.0f} KB)")


def build() -> Path:
    staging = BUILD / "submission"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    for name in ("main.py", "deck.csv"):
        src = AGENT / name
        if not src.exists():
            raise FileNotFoundError(f"missing {src}")
        shutil.copy2(src, staging / name)

    write_tables(staging / "agent_data.json")

    archive = BUILD / "submission.tar.gz"
    if archive.exists():
        archive.unlink()
    # Paths must be relative to the archive root, not nested in a folder.
    with tarfile.open(archive, "w:gz") as tar:
        for p in sorted(staging.rglob("*")):
            tar.add(p, arcname=str(p.relative_to(staging)))

    size = archive.stat().st_size / 1024**2
    print(f"built {archive.relative_to(ROOT)} ({size:.1f} MB)")
    with tarfile.open(archive) as tar:
        names = tar.getnames()
    print(f"  {len(names)} entries; root files: "
          f"{[n for n in names if '/' not in n]}")
    return archive


def _basic_energy_ids() -> set[int]:
    """Basic Energy is exempt from the four-copy limit, so it needs flagging."""
    sys.path.insert(0, str(ENGINE))
    try:
        import json

        from cg.sim import lib
        return {c["cardId"] for c in json.loads(lib.AllCard().decode())
                if c.get("cardType") == 5}
    except Exception:
        return set()


def verify(deck_path: Path) -> None:
    """Fail loudly before submitting rather than after."""
    deck = [int(x) for x in deck_path.read_text().replace(",", "\n").split()
            if x.strip()]
    if len(deck) != 60:
        raise ValueError(f"deck has {len(deck)} cards, must be 60")
    counts: dict[int, int] = {}
    for c in deck:
        counts[c] = counts.get(c, 0) + 1
    exempt = _basic_energy_ids()
    over = {c: n for c, n in counts.items() if n > 4 and c not in exempt}
    if over:
        raise ValueError(f"illegal deck — more than 4 copies of {over}")
    print(f"  deck ok: 60 cards, {len(counts)} distinct, "
          f"{sum(n for c, n in counts.items() if c in exempt)} basic energy")


def submit(archive: Path, message: str) -> int:
    cmd = [sys.executable, "-m", "kaggle", "competitions", "submit",
           "-c", COMPETITION, "-f", str(archive), "-m", message]
    print("running:", " ".join(cmd[2:]))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    print(proc.stdout.strip())
    if proc.stderr.strip():
        print("stderr:", proc.stderr.strip())
    return proc.returncode


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("-m", "--message", default="heuristic agent")
    args = ap.parse_args()

    verify(AGENT / "deck.csv")
    archive = build()

    if args.submit:
        # Never submit without reproducing the grader's validation episode.
        rc = subprocess.run(
            [sys.executable, str(Path(__file__).parent / "validate_submission.py")],
            text=True).returncode
        if rc != 0:
            sys.exit("validation failed — not submitting")
    if args.submit:
        rc = submit(archive, args.message)
        sys.exit(rc)
    else:
        print("\nnot submitted (pass --submit to send it)")


if __name__ == "__main__":
    main()
