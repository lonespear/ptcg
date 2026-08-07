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
import os
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


def opp_branch() -> int:
    """The agent's SEARCH_OPP_BRANCH, read off the source it will ship.

    Read rather than imported: importing `agent/main.py` loads `cg`, and this
    script already loads it once for the card tables.
    """
    import re
    src = (AGENT / "main.py").read_text(encoding="utf-8")
    m = re.search(r'CABT_OPP_BRANCH"\) or (\d+)\)', src)
    if m is None:
        raise ValueError("cannot read SEARCH_OPP_BRANCH out of agent/main.py")
    return int(m.group(1))


def agent_flags() -> dict:
    """The posture / protection defaults, read off the source that will ship.

    Same reason as `opp_branch` above: importing agent/main.py loads `cg`, and
    this script already loads it once for the card tables. The environment is
    deliberately not consulted — what ships is the file's own default, not
    whatever the shell that ran the build happened to export.
    """
    import re
    src = (AGENT / "main.py").read_text(encoding="utf-8")
    out = {}
    for key, name in (("postures", "CABT_POSTURES"), ("protect", "CABT_PROTECT"),
                      ("tree_leaf", "CABT_TREE_LEAF"),
                      ("scaled", "CABT_SCALED_DAMAGE")):
        m = re.search(rf'{name}"\) or (\d+)\)', src)
        if m is None:
            raise ValueError(f"cannot read {name} out of agent/main.py")
        out[key] = int(m.group(1))
    return out


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

    # The opponent-deck prior the search needs. Without it the agent still
    # plays, but falls back to rules — so its absence must be loud.
    priors = AGENT / "deck_priors.json"
    if priors.exists():
        shutil.copy2(priors, staging / "deck_priors.json")
        print(f"  priors: {priors.stat().st_size / 1024:.0f} KB")
    else:
        print("  WARNING: no deck_priors.json — search will be disabled. "
              "Run scripts/export_priors.py")

    # The fitted margin -> P(win) table (playbook C3). The agent reads its own
    # verdict off it at run time, and the comeback posture is keyed on that
    # verdict, so a missing table silently flattens a shipped behaviour to
    # nothing — hence the same loud absence the priors get.
    calib = ROOT / "data" / "calibration_v2.json"
    if calib.exists():
        shutil.copy2(calib, staging / "calibration.json")
        print(f"  calibration: {calib.stat().st_size / 1024:.0f} KB")
    else:
        print("  WARNING: no data/calibration_v2.json — _pwin returns 0.5 and "
              "the comeback posture never fires. Run "
              "python -m ptcg.creation.calibration --games 3000 "
              "--out data/calibration_v2.json")

    # The C1/C2 trajectory tables: the fitted turn-to-turn Energy curves the
    # projection reads, and the Energy-mechanics KB it credits a visible
    # accelerator from. Same loud absence as the two above — without them the
    # projection falls back to a flat field rate and the fitted C2 weight is
    # being spent on a number it was not fitted for.
    for src, dest, rebuild in (
            (ROOT / "data" / "analysis" / "trajectory_curves.json",
             "trajectory_curves.json",
             "python -m ptcg.trajectory"),
            (ROOT / "data" / "energy_mechanics.json",
             "energy_mechanics.json",
             "python -c \"from ptcg.energy_mechanics import write_json; "
             "write_json()\""),
            ):
        if src.exists():
            shutil.copy2(src, staging / dest)
            print(f"  {dest}: {src.stat().st_size / 1024:.0f} KB")
        else:
            print(f"  WARNING: no {src.relative_to(ROOT)} — the C1/C2 "
                  f"projection degrades to the field-average rate. "
                  f"Rebuild it with: {rebuild}")

    # The matchup deny-postures and the gust-reach table (postures.json), under
    # the same rule the opponent table gets below: a file in the archive that
    # nothing opens is a bundle audit that lies. Both behaviours that read it
    # were measured and refused, so CABT_POSTURES and CABT_PROTECT default to
    # 0 and this normally copies nothing and says so.
    flags = agent_flags()
    postures = ROOT / "data" / "analysis" / "postures.json"
    # The tree leaf reads this file too, and for the same reason the
    # protection rule does: `attackers_exposed` counts a benched Pokemon only
    # where their archetype can gust it into the Active Spot, which is
    # `gust_reach`. Two of the sixteen features would silently fall back to
    # the play-weighted default without it — a different feature from the one
    # the forest was fitted on.
    if flags["postures"] or flags["protect"] or flags["tree_leaf"]:
        if not postures.exists():
            raise FileNotFoundError(
                f"a posture behaviour is on and {postures} is missing — no "
                f"posture would activate and gust reach would fall back to a "
                f"default. Rebuild it: python -m ptcg.matchup_postures")
        shutil.copy2(postures, staging / "postures.json")
        print(f"  postures: {postures.stat().st_size / 1024:.0f} KB "
              f"(CABT_POSTURES={flags['postures']}, "
              f"CABT_PROTECT={flags['protect']}, "
              f"CABT_TREE_LEAF={flags['tree_leaf']})")
    else:
        print("  postures: not bundled — the deny-postures, the protection "
              "rule and the tree leaf are all off, so nothing in the agent "
              "would read it")

    # The attack-scaler KB (the scaling/flat-damage/resistance bundle). Ships
    # only while CABT_SCALED_DAMAGE defaults on in the source, under the same
    # audit rule as every other switched file: present iff something opens it.
    scalers = AGENT / "attack_scalers.json"
    if flags["scaled"]:
        if not scalers.exists():
            raise FileNotFoundError(
                f"CABT_SCALED_DAMAGE is on and {scalers} is missing — every "
                f"scaling attack would price at its printed number. Rebuild "
                f"it: python -m ptcg.attack_scalers --write")
        shutil.copy2(scalers, staging / "attack_scalers.json")
        print(f"  attack_scalers: {scalers.stat().st_size / 1024:.0f} KB "
              f"(CABT_SCALED_DAMAGE={flags['scaled']})")
    else:
        print("  attack_scalers: not bundled — CABT_SCALED_DAMAGE is 0, so "
              "nothing in the agent would read it")

    # The D34 tree leaf. Under the same rule the two above get: it ships only
    # while the agent is configured to read it, and CABT_TREE_LEAF defaults to
    # 0 because the gate refused it (pooled mirrors 0.2135 over 787). Turning
    # the leaf on without the forest beside it would put `_evaluate` into a
    # guarded except on every scored position and score nothing at all, so a
    # missing file with the flag up is fatal rather than quiet.
    leaf = ROOT / "data" / "analysis" / "tree_leaf.json"
    if flags["tree_leaf"]:
        if not leaf.exists():
            raise FileNotFoundError(
                f"CABT_TREE_LEAF is on and {leaf} is missing — every search "
                f"evaluation would raise into its guard and fall back to the "
                f"linear margin without saying so. Rebuild it: "
                f"/usr/bin/python3 scripts/fit_tree_leaf.py fit")
        shutil.copy2(leaf, staging / "tree_leaf.json")
        print(f"  tree_leaf: {leaf.stat().st_size / 1024:.0f} KB "
              f"(CABT_TREE_LEAF={flags['tree_leaf']})")
    else:
        print("  tree_leaf: not bundled — CABT_TREE_LEAF is 0, so nothing in "
              "the agent would open it")

    # Playbook entry 4: the counted opponent reply table. It ships only while
    # the agent is configured to read it, because a file in the archive that
    # nothing opens is a bundle audit that lies. SEARCH_OPP_BRANCH is 0 at the
    # moment — the entry was measured and refused — so this normally copies
    # nothing and says so.
    branch = opp_branch()
    policy = ROOT / "data" / "opponent_policy.json"
    if branch >= 1:
        if not policy.exists():
            raise FileNotFoundError(
                f"SEARCH_OPP_BRANCH is {branch} and {policy} is missing — the "
                f"rollout would play their turn under our own priority table. "
                f"Rebuild it: /usr/bin/python3 scripts/build_opponent_policy.py")
        shutil.copy2(policy, staging / "opponent_policy.json")
        print(f"  opponent_policy: {policy.stat().st_size / 1024:.0f} KB "
              f"(SEARCH_OPP_BRANCH={branch})")
    else:
        print("  opponent_policy: not bundled — SEARCH_OPP_BRANCH is 0, so "
              "nothing in the agent would read it")

    # The agent reads card/attack metadata through cg.api, exactly as the
    # official sample does, so the package ships with the bundle. (Importing it
    # was never the problem — the first two failures were the __file__ bug.)
    cg_src = ENGINE / "cg"
    if not cg_src.exists():
        raise FileNotFoundError(
            f"missing {cg_src} — copy sample_submission from the competition "
            f"download into engine/")
    shutil.copytree(cg_src, staging / "cg",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    archive = BUILD / "submission.tar.gz"
    if archive.exists():
        archive.unlink()
    # Paths must be relative to the archive root, not nested in a folder.
    with tarfile.open(archive, "w:gz") as tar:
        for p in sorted(staging.rglob("*")):
            # rglob already yields every descendant; letting tar.add recurse
            # into directories on top of that writes each cg/ file twice.
            tar.add(p, arcname=str(p.relative_to(staging)), recursive=False)

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
    if proc.stdout:
        print(proc.stdout.strip())
    if proc.stderr:
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
        # PYTHONIOENCODING keeps the child's own prints from dying on "Pokémon"
        # under the Windows console codepage — a crash there would look like a
        # validation failure, or worse, be mistaken for noise.
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
        for gate in ("falsify_bundle.py", "validate_submission.py"):
            rc = subprocess.run(
                [sys.executable, str(Path(__file__).parent / gate)],
                text=True, env=env).returncode
            if rc != 0:
                sys.exit(f"{gate} failed - not submitting")
            print(f"  {gate} passed")
    if args.submit:
        rc = submit(archive, args.message)
        sys.exit(rc)
    else:
        print("\nnot submitted (pass --submit to send it)")


if __name__ == "__main__":
    main()
