"""Fitness panel where each field deck is piloted by its own specialist.

The default panel plays every opponent list with one generalist pilot, which
flatters decks whose plan the generalist cannot execute. Community agents
harvested from public Kaggle notebooks are deck-specialized: every one of
them beats the rules agent 78-96% piloting its own list, and the Alakazam
agent drops to 27% on a foreign one. So a panel entry whose list matches a
harvested specialist's own deck gets that specialist as its pilot;
everything else keeps the generalist.

Matching is multiset card overlap between the specialist's own 60 and the
panel list (default threshold 40 shared cards) — the same archetype with
different tech, not a name string.

Specialists live in `external/` (gitignored, license unverified); a missing
directory degrades to an all-generalist panel with no error.
"""

import json
from collections import Counter
from pathlib import Path

from .archipelago import load_field_panel

_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL = _ROOT / "external"
MATCH_THRESHOLD = 40


def discover_specialists(external: Path | None = None) -> list[dict]:
    """Harvested agents as {name, agent_path, deck} — pairs of
    external/<name>_agent.py and external/<name>_deck.json."""
    external = Path(external) if external else EXTERNAL
    out = []
    if not external.is_dir():
        return out
    for agent_path in sorted(external.glob("*_agent.py")):
        name = agent_path.name[: -len("_agent.py")]
        deck_path = external / f"{name}_deck.json"
        if not deck_path.exists():
            continue
        try:
            deck = [int(c) for c in json.loads(deck_path.read_text())]
        except (ValueError, json.JSONDecodeError):
            continue
        if len(deck) == 60:
            out.append({"name": name, "agent_path": str(agent_path),
                        "deck": deck})
    return out


def _overlap(a: list[int], b: list[int]) -> int:
    return sum((Counter(a) & Counter(b)).values())


def build_specialist_panel(priors_path: Path, top_n: int = 8,
                           external: Path | None = None,
                           threshold: int = MATCH_THRESHOLD) -> list[dict]:
    """Field panel with a `pilot` spec attached to every entry.

    Each entry gains `pilot`: {"kind": "external", "path": ..., "specialist":
    ..., "overlap": n} when a harvested specialist plays that archetype, else
    {"kind": "generalist"}. An entry takes its highest-overlap specialist; one
    specialist may pilot several entries, since the field carries the same
    archetype at several tech configurations.
    """
    panel = load_field_panel(priors_path, top_n=top_n)
    specialists = discover_specialists(external)

    for entry in panel:
        entry["pilot"] = {"kind": "generalist"}
        if not specialists:
            continue
        n, spec = max(((_overlap(s["deck"], entry["deck"]), s)
                       for s in specialists), key=lambda s: s[0])
        if n >= threshold:
            entry["pilot"] = {"kind": "external", "path": spec["agent_path"],
                              "specialist": spec["name"], "overlap": n}
    return panel


def make_panel_pilots(panel: list[dict], generalist_factory, seed: int = 900):
    """One pilot per panel entry, in panel order.

    `generalist_factory(seed)` builds the fallback pilot. External pilots that
    fail to load fall back to the generalist and the entry is re-marked, so a
    broken harvest degrades the panel instead of killing the run.
    """
    from .pilots import ExternalPilot

    pilots = []
    for i, entry in enumerate(panel):
        spec = entry.get("pilot") or {"kind": "generalist"}
        if spec["kind"] == "external":
            try:
                pilots.append(ExternalPilot(spec["path"]))
                continue
            except Exception as exc:  # noqa: BLE001 — a bad harvest is data
                entry["pilot"] = {"kind": "generalist",
                                  "failed": spec.get("specialist"),
                                  "error": f"{type(exc).__name__}: {exc}"}
        pilots.append(generalist_factory(seed + i))
    return pilots


def panel_report(panel: list[dict]) -> str:
    lines = []
    for i, e in enumerate(panel):
        spec = e["pilot"]
        who = (f"{spec['specialist']} (overlap {spec['overlap']})"
               if spec["kind"] == "external" else "generalist")
        lines.append(f"  [{i}] w={e['weight']:.3f} {e['name'][:30]:32s} {who}")
    covered = sum(e["weight"] for e in panel
                  if e["pilot"]["kind"] == "external")
    lines.append(f"  specialist-piloted weight: {covered:.3f}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(panel_report(build_specialist_panel(_ROOT / "agent" /
                                              "deck_priors.json")))
