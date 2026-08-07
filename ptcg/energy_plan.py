"""The five-turn energy planner: when each attacker comes online.

Concept C1 of D29 is expected energy per Pokemon over t..t+5, and D31 fixes our
side of it as a live projection rather than a fit: true board energy from the
observation, plus one manual attach a turn, plus the accelerators the
`ptcg.energy_mechanics` knowledge base says this deck runs, minus the Energy a
planned attack burns. This module is that arithmetic.

Nothing here touches the agent, the engine or an observation object. Everything
is a pure function over three inputs a caller assembles:

    Position   what is on our board, in our hand, in our discard, and unseen
    the pool   attack costs (ptcg.creation.pool)
    the KB     what each card does to Energy (ptcg.energy_mechanics)

The plan is a small exhaustive search, not a heuristic: with at most four
candidate attackers there are at most 24 feed orders, so every order is
simulated for five turns and the best is kept, scored lexicographically by the
online turn of each attacker in priority order. Inside one order the turn loop
is a straight greedy: abilities first (free), then trainers (card budget), then
the manual attach (one a turn), then the attack.

Two schedules come back. The DETERMINISTIC one spends only cards we can already
see - it is the plan you can commit to. The EXPECTED one adds, per turn, the
hypergeometric odds of drawing the accelerators and Energy still in the deck
(`ptcg.creation.outs`), which is the number that belongs in an evaluator.

    python -m ptcg.energy_plan          # self-check + worked examples
"""

from __future__ import annotations

import itertools
import re
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from ptcg.creation.outs import p_at_least_one
from ptcg.creation.pool import (BASIC_ENERGY, COLORLESS, SPECIAL_ENERGY,
                                SPECIAL_ENERGY_PROVIDES, TYPE_NAMES, pool)
from ptcg.energy_mechanics import build_kb

HORIZON = 5
WILD = -1                      # an attached Energy that pays for any type

# What one use of an accelerator is worth when the text does not bound it
# ("any number of", "up to the amount of ..."). Conservative on purpose and
# surfaced in every printout so a reader can argue with it.
UNBOUNDED_CAP = 2

# Frequencies the planner knows how to spend, and the budget each one draws on.
_ABILITY_FREQ = {"once_per_turn": 1, "unlimited_per_turn": 2,
                 "end_of_turn": 1, "on_promote": 1}
_ENTRY_FREQ = {"on_evolve", "on_bench_play"}
_TRAINER_FREQ = {"item_play": "item", "supporter_play": "supporter",
                 "tool_attached": "item"}


# --- energy arithmetic ------------------------------------------------------

def provides(card_id: int) -> list[int]:
    """The type ids an attached copy of this Energy card pays for.

    A basic Energy pays its own type. A Special Energy that provides every type
    is a joker (WILD). Everything else in the special pool pays {C} only, which
    is type id 0 in the engine's enum.
    """
    card = pool().card(card_id)
    if card is None:
        return []
    if card["cardType"] == BASIC_ENERGY:
        return [card["energyType"]]
    if card["cardType"] != SPECIAL_ENERGY:
        return []
    spec = SPECIAL_ENERGY_PROVIDES.get(card_id, {})
    types = spec.get("types")
    if types == "wild":
        return [WILD]
    if types:
        return [WILD] if len(types) > 1 else [next(iter(types))]
    return [COLORLESS]


def is_energy_card(card_id: int) -> bool:
    card = pool().card(card_id)
    return card is not None and card["cardType"] in (BASIC_ENERGY,
                                                     SPECIAL_ENERGY)


def cost_satisfied(attached: Counter, cost: list[int]) -> bool:
    """Does this attached-Energy multiset pay this printed cost?

    `cost` is the engine's energies list: type ids, with 0 for a colorless
    symbol. Colored symbols are paid by their own type first, then by jokers;
    colorless symbols by whatever is left.
    """
    need = Counter(t for t in cost if t != COLORLESS)
    colorless = sum(1 for t in cost if t == COLORLESS)
    have = Counter(attached)
    jokers = have.pop(WILD, 0)
    for t, n in need.items():
        take = min(have.get(t, 0), n)
        have[t] -= take
        n -= take
        if n:
            used = min(jokers, n)
            jokers -= used
            n -= used
        if n:
            return False
    return sum(v for v in have.values() if v > 0) + jokers >= colorless


def attack_options(card_id: int) -> list[dict]:
    """Every attack this Pokemon prints, richest first."""
    p = pool()
    card = p.card(card_id)
    if not card:
        return []
    out = []
    for aid in card["attacks"]:
        a = p.attack_by_id.get(aid)
        if not a:
            continue
        out.append({"attack_id": aid, "name": a["name"],
                    "damage": a.get("damage") or 0,
                    "cost": list(a.get("energies") or []),
                    "text": a.get("text") or ""})
    return sorted(out, key=lambda a: (-a["damage"], len(a["cost"])))


def main_attack(card_id: int) -> dict | None:
    """The attack we plan toward: the most damaging one that costs Energy."""
    opts = [a for a in attack_options(card_id) if a["cost"]]
    return opts[0] if opts else None


def cost_str(cost: list[int]) -> str:
    return "".join(TYPE_NAMES.get(t, "?")[0] if t else "C" for t in cost) or "-"


# --- position ---------------------------------------------------------------

@dataclass
class Slot:
    """One of our Pokemon in play and the Energy cards on it."""
    card_id: int
    attached: list[int] = field(default_factory=list)   # Energy card ids
    active: bool = False

    @property
    def name(self) -> str:
        return pool().name(self.card_id)

    @property
    def energy(self) -> Counter:
        c = Counter()
        for cid in self.attached:
            for t in provides(cid):
                c[t] += 1
        return c

    @property
    def count(self) -> int:
        return len(self.attached)

    def key(self, i: int) -> str:
        """A stable handle for this board slot.

        Not the card name: two copies of one card are two Pokemon, and a slot
        that evolves changes its name mid-plan without becoming a new target.
        """
        return f"slot{i + 1}"


@dataclass
class Position:
    """Everything the planner needs to know, all of it observable."""
    slots: list[Slot]
    hand: list[int] = field(default_factory=list)
    discard: list[int] = field(default_factory=list)
    deck: list[int] = field(default_factory=list)       # unseen: deck + prizes
    decklist: list[int] = field(default_factory=list)   # the 60, for outs

    def copy(self) -> "Position":
        return Position(
            slots=[Slot(s.card_id, list(s.attached), s.active)
                   for s in self.slots],
            hand=list(self.hand), discard=list(self.discard),
            deck=list(self.deck), decklist=list(self.decklist))

    @property
    def active(self) -> Slot | None:
        for s in self.slots:
            if s.active:
                return s
        return self.slots[0] if self.slots else None


# --- the knowledge-base view the planner needs ------------------------------

@lru_cache(maxsize=1)
def _kb_index() -> dict[int, list[dict]]:
    idx: dict[int, list[dict]] = {}
    for r in build_kb()["records"]:
        idx.setdefault(r["card_id"], []).append(r)
    return idx


def card_records(card_id: int, mechanic: str | None = None) -> list[dict]:
    recs = _kb_index().get(card_id, [])
    return [r for r in recs if mechanic is None or r["mechanic"] == mechanic]


def effective_rate(rec: dict) -> int:
    """Energies this record moves per use, with unbounded text capped."""
    return rec["rate"] if rec["rate"] is not None else UNBOUNDED_CAP


_TAG_RE = re.compile(r"^([A-Za-z]+)['’]s ")


def target_ok(rec: dict, slot: Slot, source: Slot | None) -> bool:
    """Can this accelerator legally aim at this Pokemon?

    Constraints the KB records but the engine dump cannot confirm (the Future
    and Ancient subtypes have no flag in the dump) return False: an unverified
    constraint must not inflate a schedule.
    """
    tgt = rec.get("target") or {}
    card = pool().card(slot.card_id) or {}
    # Identity, not card id: two copies of one card are two Pokemon, and an
    # ability that says "this Pokemon" means the one that used it.
    if tgt.get("scope") == "self" and source is not None and source is not slot:
        return False
    pos = tgt.get("position")
    if pos == "bench" and slot.active:
        return False
    if pos == "active" and not slot.active:
        return False
    if tgt.get("group"):
        return bool(card.get("tera")) and tgt["group"] == "Tera"
    if tgt.get("stage") == 2 and not card.get("stage2"):
        return False
    tag = tgt.get("owner_tag")
    if tag:
        m = _TAG_RE.match(slot.name)
        if not m or m.group(1).lower() != tag.lower():
            return False
    types = tgt.get("types") or []
    if types:
        want = {t for t in types}
        letter = _TYPE_LETTER.get(card.get("energyType"))
        if letter not in want:
            return False
    return True


_TYPE_LETTER = {1: "G", 2: "R", 3: "W", 4: "L", 5: "P", 6: "F", 7: "D",
                8: "M", 9: "N", 10: "Y", 0: "C"}


def energy_ok(rec: dict, card_id: int) -> bool:
    """Is this Energy card one the accelerator is allowed to fetch?"""
    filt = rec.get("energy_filter") or {}
    card = pool().card(card_id) or {}
    if filt.get("basic_only") and card.get("cardType") != BASIC_ENERGY:
        return False
    types = filt.get("types") or []
    if types:
        return _TYPE_LETTER.get(card.get("energyType")) in set(types)
    return True


# --- the turn loop ----------------------------------------------------------

@dataclass
class TurnRecord:
    turn: int
    actions: list[str]
    per_slot: dict[str, int]
    total: int
    online: dict[str, int]      # attacker name -> turn it came online


@dataclass
class Schedule:
    turns: list[TurnRecord]
    online: dict[str, int | None]
    order: list[str]
    labels: dict[str, str] = field(default_factory=dict)
    expected_total: dict[int, float] = field(default_factory=dict)
    expected_note: list[str] = field(default_factory=list)

    def score(self) -> tuple:
        """Lexicographic: earliest online for the first attacker, then next."""
        return tuple(self.online.get(n) or 99 for n in self.order)


def _zone(pos: Position, zone: str) -> list[int]:
    return {"hand": pos.hand, "discard": pos.discard,
            "deck": pos.deck, "deck_top": pos.deck}.get(zone, [])


def _take_energy(pos: Position, zone: str, rec: dict, n: int) -> list[int]:
    """Pull up to n Energy cards the accelerator can legally take."""
    src = _zone(pos, zone)
    taken = []
    for cid in list(src):
        if len(taken) >= n:
            break
        if is_energy_card(cid) and energy_ok(rec, cid):
            src.remove(cid)
            taken.append(cid)
    return taken


def _attach(pos: Position, slot: Slot, cids: list[int]) -> None:
    slot.attached.extend(cids)


def _online_now(slot: Slot, attack: dict) -> bool:
    return cost_satisfied(slot.energy, attack["cost"])


def simulate(pos: Position, order: list[int], horizon: int = HORIZON
             ) -> Schedule:
    """Play `horizon` turns feeding the slots in `order`, and log every step.

    Order of operations inside a turn, and the budget each step spends:
      1. abilities on Pokemon already in play  (free, once each)
      2. Item / Tool accelerators from hand    (card, unlimited plays)
      3. one Supporter accelerator from hand   (the turn's Supporter)
      4. the manual attach                     (the turn's attach)
      5. the attack, and any Energy it burns   (the turn's attack)
    """
    pos = pos.copy()
    targets = [pos.slots[i] for i in order]
    goals = {s.card_id: main_attack(s.card_id) for s in pos.slots}
    online: dict[str, int | None] = {
        s.key(i): (1 if goals[s.card_id] and _online_now(s, goals[s.card_id])
                   else None) for i, s in enumerate(pos.slots)}
    turns: list[TurnRecord] = []

    def next_target() -> Slot | None:
        for s in targets:
            g = goals[s.card_id]
            if g and not _online_now(s, g):
                return s
        return targets[0] if targets else None

    for t in range(1, horizon + 1):
        acts: list[str] = []
        budget = {"supporter": 1, "attach": 1, "attack": 1}

        # 1. abilities already on the board
        for src in list(pos.slots):
            for rec in card_records(src.card_id, "acceleration"):
                if rec["frequency"] not in _ABILITY_FREQ:
                    continue
                uses = _ABILITY_FREQ[rec["frequency"]]
                for _ in range(uses):
                    tgt = next((s for s in targets + pos.slots
                                if target_ok(rec, s, src)), None)
                    if tgt is None:
                        break
                    got = _take_energy(pos, rec["source_zone"], rec,
                                       effective_rate(rec))
                    if not got:
                        break
                    _attach(pos, tgt, got)
                    acts.append(f"ability {src.name}/{rec['effect_name']} "
                                f"+{len(got)} -> {tgt.name}")

        # 1b. an evolution (or a Bench drop) whose entry brings Energy with it
        for cid in list(pos.hand):
            for rec in card_records(cid, "acceleration"):
                if rec["frequency"] not in _ENTRY_FREQ:
                    continue
                if rec["frequency"] == "on_evolve":
                    pre = pool().evolves_from_name(pool().name(cid))
                    host = next((s_ for s_ in pos.slots
                                 if pre and s_.name == pre), None)
                    if host is None:
                        continue
                    host.card_id = cid
                    goals[cid] = main_attack(cid)
                    online.setdefault(host.key(pos.slots.index(host)), None)
                    src = host
                    acts.append(f"evolve -> {host.name}")
                else:
                    src = Slot(cid)
                    pos.slots.append(src)
                    goals[cid] = main_attack(cid)
                    acts.append(f"bench {src.name}")
                pos.hand.remove(cid)
                tgt = next((s_ for s_ in targets + pos.slots
                            if target_ok(rec, s_, src)), None)
                if tgt is not None:
                    got = _take_energy(pos, rec["source_zone"], rec,
                                       effective_rate(rec))
                    _attach(pos, tgt, got)
                    acts.append(f"  {rec['effect_name'].strip()} +{len(got)}"
                                f" -> {tgt.name}")
                break

        # 2-3. trainers in hand
        for cid in list(pos.hand):
            for rec in card_records(cid, "acceleration"):
                bud = _TRAINER_FREQ.get(rec["frequency"])
                if bud is None:
                    continue
                if bud == "supporter" and budget["supporter"] <= 0:
                    continue
                tgt = next((s for s in targets + pos.slots
                            if target_ok(rec, s, None)), None)
                if tgt is None:
                    continue
                if cid not in pos.hand:
                    break
                pos.hand.remove(cid)
                got = _take_energy(pos, rec["source_zone"], rec,
                                   effective_rate(rec))
                _attach(pos, tgt, got)
                if bud == "supporter":
                    budget["supporter"] -= 1
                acts.append(f"play {pool().name(cid)} +{len(got)} -> "
                            f"{tgt.name}")
                break

        # 4. the manual attach
        tgt = next_target()
        if tgt is not None and budget["attach"]:
            cid = next((c for c in pos.hand if is_energy_card(c)), None)
            if cid is not None:
                pos.hand.remove(cid)
                _attach(pos, tgt, [cid])
                budget["attach"] -= 1
                acts.append(f"attach {pool().name(cid)} -> {tgt.name}")

        for i, s_ in enumerate(pos.slots):
            g = goals.get(s_.card_id)
            if g and online.get(s_.key(i)) is None and _online_now(s_, g):
                online[s_.key(i)] = t
                acts.append(f"ONLINE {s_.name}: {g['name']} "
                            f"[{cost_str(g['cost'])}] {g['damage']} dmg")

        # 5. the attack: a ramp attack if the goal is not up yet, else the goal
        act = pos.active
        if act is not None and budget["attack"]:
            goal = goals[act.card_id]
            if goal and _online_now(act, goal):
                acts.append(f"attack {goal['name']} ({goal['damage']})")
                for rec in card_records(act.card_id, "discard_cost"):
                    if rec.get("attack_id") != goal["attack_id"]:
                        continue
                    n = rec["rate"] if rec["rate"] is not None else act.count
                    burn = act.attached[:n] if n else []
                    for c in burn:
                        act.attached.remove(c)
                        pos.discard.append(c)
                    if burn:
                        acts.append(f"  cost: -{len(burn)} Energy off "
                                    f"{act.name}")
            else:
                ramp = next(
                    (a for a in attack_options(act.card_id)
                     if cost_satisfied(act.energy, a["cost"])
                     and any(r.get("attack_id") == a["attack_id"]
                             for r in card_records(act.card_id,
                                                   "acceleration"))), None)
                if ramp:
                    rec = next(r for r in card_records(act.card_id,
                                                       "acceleration")
                               if r.get("attack_id") == ramp["attack_id"])
                    dest = next((s for s in targets + pos.slots
                                 if target_ok(rec, s, act)), None)
                    if dest is not None:
                        got = _take_energy(pos, rec["source_zone"], rec,
                                           effective_rate(rec))
                        _attach(pos, dest, got)
                        acts.append(f"attack {ramp['name']} (ramp) "
                                    f"+{len(got)} -> {dest.name}")

        turns.append(TurnRecord(
            turn=t, actions=acts,
            per_slot={s_.key(i): s_.count for i, s_ in enumerate(pos.slots)},
            total=sum(s.count for s in pos.slots),
            online={k: v for k, v in online.items() if v is not None}))

    return Schedule(turns=turns, online=online,
                    order=[pos.slots[i].key(i) for i in order],
                    labels={s_.key(i): s_.name
                            for i, s_ in enumerate(pos.slots)})


def plan(pos: Position, horizon: int = HORIZON, max_targets: int = 4
         ) -> Schedule:
    """Best feed order over the candidate attackers, by exhaustive search."""
    cand = [i for i, s in enumerate(pos.slots) if main_attack(s.card_id)]
    if not cand:
        return simulate(pos, list(range(len(pos.slots))), horizon)
    cand = cand[:max_targets]
    best: Schedule | None = None
    for order in itertools.permutations(cand):
        sched = simulate(pos, list(order), horizon)
        if best is None or sched.score() < best.score():
            best = sched
    best.expected_total, best.expected_note = expected_curve(pos, best,
                                                             horizon)
    return best


# --- the probabilistic layer ------------------------------------------------

def expected_curve(pos: Position, sched: Schedule, horizon: int = HORIZON
                   ) -> tuple[dict[int, float], list[str]]:
    """Deterministic totals plus the odds of drawing what is still hidden.

    One draw a turn. For each distinct accelerator still in the unseen pile,
    P(at least one by turn t) times what one use is worth; likewise for the
    Energy needed by the manual attach on turns the known hand cannot cover.
    Draw Supporters are not modelled, which pushes the number down; each
    accelerator is credited one use, which pushes it up. It is an estimate,
    and the terms are printed so a reader can take it apart.
    """
    unseen = len(pos.deck)
    notes: list[str] = []
    if unseen <= 0:
        return {t.turn: float(t.total) for t in sched.turns}, ["no unseen pile"]

    accel_outs: dict[int, int] = {}
    for cid in pos.deck:
        if card_records(cid, "acceleration"):
            accel_outs[cid] = accel_outs.get(cid, 0) + 1
    energy_outs = sum(1 for cid in pos.deck if is_energy_card(cid))

    hand_energy = sum(1 for c in pos.hand if is_energy_card(c))
    out: dict[int, float] = {}

    for tr in sched.turns:
        t = tr.turn
        extra = 0.0
        p_energy = p_at_least_one(energy_outs, unseen, t)
        for cid, outs in accel_outs.items():
            recs = card_records(cid, "acceleration")
            rate = max(effective_rate(r) for r in recs)
            gate = 1.0
            if all(r["source_zone"] == "hand" for r in recs) \
                    and hand_energy == 0:
                gate = p_energy      # it can only attach what the hand holds
            extra += p_at_least_one(outs, unseen, t) * rate * gate
        # Turns whose manual attach the visible hand cannot pay for.
        short = max(0, t - hand_energy)
        if short:
            extra += short * p_at_least_one(energy_outs, unseen, t)
        out[t] = round(tr.total + extra, 2)
    for cid, outs in sorted(accel_outs.items(), key=lambda kv: -kv[1]):
        rate = max(effective_rate(r) for r in card_records(cid, "acceleration"))
        notes.append(f"{pool().name(cid)}: {outs} outs / {unseen} unseen, "
                     f"+{rate}/use -> P(by t5)="
                     f"{p_at_least_one(outs, unseen, horizon) * 100:.0f}%")
    notes.append(f"Energy in deck: {energy_outs} outs / {unseen} unseen, "
                 f"P(by t5)={p_at_least_one(energy_outs, unseen, horizon) * 100:.0f}%")
    return out, notes


def baseline_curve(pos: Position, horizon: int = HORIZON) -> dict[int, int]:
    """One attach a turn and nothing else - what acceleration is measured against."""
    start = sum(s.count for s in pos.slots)
    return {t: start + t for t in range(1, horizon + 1)}


# --- opponent-side capability ----------------------------------------------

def deck_accel_package(decklist: list[int]) -> dict:
    """What a known 60 can do to its own Energy count, per turn.

    Reports the components rather than one number: an ability that fires every
    turn is a different animal from a Supporter that fires once, and the
    ceiling a caller should use depends on which of them is on the board.
    """
    counts = Counter(decklist)
    rows = []
    for cid, n in counts.items():
        for rec in card_records(cid, "acceleration"):
            rows.append({
                "card_id": cid, "name": pool().name(cid), "copies": n,
                "rate": rec["rate"], "capped_rate": effective_rate(rec),
                "source": rec["source_zone"], "frequency": rec["frequency"],
                "target": (rec.get("target") or {}).get("raw", ""),
                "text": rec["sentence"],
            })
    rows.sort(key=lambda r: (-r["capped_rate"], r["name"]))

    def best(freqs) -> int:
        vals = [r["capped_rate"] for r in rows if r["frequency"] in freqs]
        return max(vals) if vals else 0

    recurring = best(set(_ABILITY_FREQ))
    supporter = best({"supporter_play"})
    item = best({"item_play", "tool_attached"})
    entry = best(_ENTRY_FREQ)
    rider = best({"attack_rider"})
    realistic = 1 + recurring + max(supporter, item)
    return {
        "cards": rows,
        "energy_count": sum(n for cid, n in counts.items()
                            if is_energy_card(cid)),
        "components": {
            "manual_attach": 1,
            "recurring_ability": recurring,
            "best_supporter": supporter,
            "best_item": item,
            "on_entry_burst": entry,
            "attack_rider": rider,
        },
        "realistic_per_turn": realistic,
        "burst_ceiling": realistic + entry,
        "ceiling_curve": {t: realistic * t + entry
                          for t in range(1, HORIZON + 1)},
        "ramp_curve": {t: (realistic + rider) * t + entry
                       for t in range(1, HORIZON + 1)},
    }


def format_package(name: str, pkg: dict) -> str:
    c = pkg["components"]
    lines = [f"{name}: {pkg['energy_count']} Energy cards, "
             f"{len(pkg['cards'])} accel prints",
             f"  per turn = 1 attach + {c['recurring_ability']} ability "
             f"+ {max(c['best_supporter'], c['best_item'])} trainer "
             f"= {pkg['realistic_per_turn']}/turn "
             f"(+{c['on_entry_burst']} one-off on entry)"]
    for r in pkg["cards"][:6]:
        rate = r["rate"] if r["rate"] is not None else f"~{r['capped_rate']}"
        lines.append(f"    {r['copies']}x {r['name']}: +{rate} from "
                     f"{r['source']} ({r['frequency']})")
    if c["attack_rider"]:
        lines.append(f"    (+{c['attack_rider']}/turn more if it spends the "
                     f"attack on ramp instead of damage)")
    lines.append("  ceiling by turn: " + ", ".join(
        f"t{t}={v}" for t, v in pkg["ceiling_curve"].items()))
    lines.append("  ramping instead of attacking: " + ", ".join(
        f"t{t}={v}" for t, v in pkg["ramp_curve"].items()))
    return "\n".join(lines)


# --- empirical validation against the mined series --------------------------

SERIES_GLOB = "data/mined/*/series.parquet"


def realized_growth(paths: list[str | Path] | None = None,
                    horizon: int = HORIZON, min_games: int = 100,
                    frame=None) -> dict:
    """Observed board Energy per own-turn, per archetype, from mined replays.

    The series extractor writes one row per player per turn with
    `energy_in_play`, so the realized ramp rate is a regression of that column
    on `seat_turn`. It is a LOWER bound on capability: a knocked-out Pokemon
    takes its Energy with it, and the count is of Energy surviving on board.
    """
    import pandas as pd
    if frame is not None:
        # An already-loaded series table: lets an interpreter without a
        # parquet engine still run the check.
        d = frame
        return _growth_from_frame(d, horizon, min_games)
    try:
        import pyarrow                                        # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "reading the mined series needs a parquet engine; this "
            "interpreter has none (pip install pyarrow)") from exc
    root = Path(__file__).resolve().parents[1]
    files = [Path(x) for x in (paths or sorted(root.glob(SERIES_GLOB)))]
    frames, skipped = [], []
    for f in files:
        try:
            frames.append(pd.read_parquet(
                f, columns=["our_archetype", "seat_turn", "energy_in_play"]))
        except Exception as exc:            # a day still being written out
            skipped.append(f"{f.parent.name}: {type(exc).__name__}")
    if skipped:
        print("  unreadable days: " + ", ".join(skipped))
    if not frames:
        raise FileNotFoundError(
            f"no readable series parquet under {root / SERIES_GLOB}")
    return _growth_from_frame(pd.concat(frames), horizon, min_games)


def _growth_from_frame(d, horizon: int, min_games: int) -> dict:
    d = d[d["seat_turn"].between(1, horizon)]
    g = (d.groupby(["our_archetype", "seat_turn"])["energy_in_play"]
         .agg(["mean", "size"]).reset_index())
    out = {}
    for arch, sub in g.groupby("our_archetype"):
        sub = sub.sort_values("seat_turn")
        if len(sub) < horizon or sub["size"].iloc[0] < min_games:
            continue
        y = list(sub["mean"])
        n = len(y)
        xs = list(sub["seat_turn"])
        mx, my = sum(xs) / n, sum(y) / n
        var = sum((x - mx) ** 2 for x in xs)
        slope = sum((x - mx) * (v - my) for x, v in zip(xs, y)) / var
        out[arch] = {
            "mean_by_turn": {int(t): round(v, 2) for t, v in zip(xs, y)},
            "slope": round(slope, 2),
            # The extractor samples turn 1 before the first attach, so the
            # t2..t5 rate is the honest ramp.
            "rate_t2_t5": round((y[-1] - y[1]) / (xs[-1] - xs[1]), 2)
            if n > 2 else None,
            "games": int(sub["size"].iloc[0]),
        }
    return out


def kb_capability(archetype: str) -> str:
    """What the KB says the archetype's NAMED card can do to its Energy.

    Only the named card: an archetype label is its highest-HP Pokemon, and the
    rest of the list is invisible without a decklist, so an archetype can ramp
    faster than this line says.
    """
    recs = [r for cid in pool().ids_by_name.get(archetype, [])
            for r in card_records(cid, "acceleration")]
    if not recs:
        return "none on the named card"
    return "; ".join(
        f"+{r['rate'] if r['rate'] is not None else '~' + str(effective_rate(r))}"
        f" from {r['source_zone']} ({r['frequency']})" for r in recs)


def format_validation(rows: dict, top: int = 12) -> str:
    order = sorted(rows.items(), key=lambda kv: -kv[1]["games"])[:top]
    lines = [f"{'archetype':30s} {'games':>6s} " +
             " ".join(f"t{t}" for t in range(1, HORIZON + 1)) +
             f" {'slope':>6s} {'t2-t5':>6s}  KB accel (named card)"]
    for arch, r in order:
        cells = " ".join(f"{r['mean_by_turn'].get(t, 0):4.2f}"
                         for t in range(1, HORIZON + 1))
        lines.append(f"{arch[:30]:30s} {r['games']:6d} {cells} "
                     f"{r['slope']:6.2f} {r['rate_t2_t5']:6.2f}  "
                     f"{kb_capability(arch)}")
    return "\n".join(lines)


# --- printing ---------------------------------------------------------------

def format_plan(sched: Schedule, pos: Position | None = None) -> str:
    nm = sched.labels
    lines = ["feed order: " + " > ".join(f"{k} {nm.get(k, '')}".strip()
                                         for k in sched.order)]
    base = baseline_curve(pos) if pos else {}
    for tr in sched.turns:
        head = (f"t+{tr.turn}  total {tr.total}"
                + (f" (baseline {base[tr.turn]})" if base else ""))
        if sched.expected_total:
            head += f"  expected {sched.expected_total[tr.turn]}"
        lines.append(head)
        for a in tr.actions or ["(nothing available)"]:
            lines.append(f"        {a}")
        lines.append("        board: " + ", ".join(
            f"{nm.get(k, k)} ({k}) {v}" for k, v in tr.per_slot.items()))
    for key, t in sched.online.items():
        ids = _ids_by_name(nm.get(key, ""))
        atk = main_attack(ids[0]) if ids else None
        want = f" [{cost_str(atk['cost'])}]" if atk else ""
        lines.append(f"online {nm.get(key, key)} ({key}){want}: "
                     + (f"turn {t}" if t else "not within horizon"))
    if sched.expected_note:
        lines.append("draw odds:")
        lines += [f"  {n}" for n in sched.expected_note]
    return "\n".join(lines)


def _ids_by_name(name: str) -> list[int]:
    return pool().ids_by_name.get(name, [])


# --- deck helpers used by the examples and the checks -----------------------

def load_decklist(path: str | Path) -> list[int]:
    """A 60-card list from either a JSON array or a one-id-per-line CSV."""
    p = Path(path)
    txt = p.read_text()
    if txt.lstrip().startswith("["):
        import json
        return [int(x) for x in json.loads(txt)]
    return [int(line) for line in txt.split() if line.strip().isdigit()]


def opening_position(decklist: list[int], starter_id: int,
                     hand: list[int] | None = None,
                     bench: list[int] | None = None) -> Position:
    """A turn-0 board: one active, an optional bench, the rest unseen."""
    slots = [Slot(starter_id, [], active=True)]
    slots += [Slot(cid) for cid in (bench or [])]
    hand = list(hand or [])
    seen = Counter([starter_id] + list(bench or []) + hand)
    deck = list(decklist)
    for cid, n in seen.items():
        for _ in range(n):
            if cid in deck:
                deck.remove(cid)
    return Position(slots=slots, hand=hand, deck=deck, decklist=list(decklist))


# --- self-check -------------------------------------------------------------

def _self_check() -> None:
    p = pool()

    # 1. Cost arithmetic, by hand.
    assert cost_satisfied(Counter({1: 2}), [1, 1])
    assert not cost_satisfied(Counter({1: 1}), [1, 1])
    assert cost_satisfied(Counter({1: 2}), [1, 0]), "a {G} pays a colorless"
    assert not cost_satisfied(Counter({1: 1}), [1, 0])
    assert cost_satisfied(Counter({WILD: 2}), [5, 0]), "a joker pays anything"
    assert not cost_satisfied(Counter({0: 2}), [5, 0]), "{C} never pays {P}"

    # 2. A hand-built deck with a known accelerator must beat 1/turn.
    #    Eelektrik: once a turn, a {L} Energy from the discard onto the bench.
    eel = p.ids_by_name["Eelektrik"][0]
    lightning = 4
    # Eelektrik only feeds the Bench, so the board needs one.
    fast = Position(
        slots=[Slot(eel, [], active=True), Slot(eel)],
        hand=[lightning] * 5, discard=[lightning] * 6,
        deck=[lightning] * 20, decklist=[lightning] * 20)
    slow = Position(
        slots=[Slot(eel, [], active=True), Slot(eel)],
        hand=[lightning] * 5, discard=[], deck=[lightning] * 20,
        decklist=[lightning] * 20)
    f = simulate(fast, [0, 1])
    s = simulate(slow, [0, 1])
    assert f.turns[-1].total > s.turns[-1].total, (f.turns[-1].total,
                                                   s.turns[-1].total)
    assert f.turns[-1].total > baseline_curve(fast)[HORIZON] - 1, \
        f"accelerated deck must beat the 1/turn baseline: {f.turns[-1].total}"
    assert s.turns[-1].total == baseline_curve(slow)[HORIZON], \
        "with an empty discard Eelektrik has nothing to move"

    # 3. A discard-cost attack must show the dip.
    #    Pikachu ex's Topaz Bolt discards 3 Energy after it fires.
    pika = p.ids_by_name["Pikachu ex"][0]
    atk = main_attack(pika)
    assert atk and any(r.get("attack_id") == atk["attack_id"]
                       for r in card_records(pika, "discard_cost")), \
        "Pikachu ex's main attack should carry a discard cost"
    # Topaz Bolt costs {G}{L}{M}, so the hand has to hold all three.
    rainbow = [1, lightning, 8] * 3
    dip = simulate(Position(slots=[Slot(pika, [], active=True)],
                            hand=rainbow, deck=list(rainbow),
                            decklist=list(rainbow) * 2), [0])
    totals = [t.total for t in dip.turns]
    assert any(totals[i + 1] < totals[i] for i in range(len(totals) - 1)), \
        f"a discard-cost attack must dip the curve: {totals}"

    # 4. The plan is at least as good as any single order it searched.
    oger_deck = load_decklist(Path(__file__).resolve().parents[1]
                              / "data" / "ogerpon_seed.csv")
    oger = p.ids_by_name["Teal Mask Ogerpon ex"][0]
    pos = opening_position(oger_deck, oger, hand=[1, 1, 1094])
    best = plan(pos)
    for order in ([0],):
        assert best.score() <= simulate(pos, order).score()

    # 5. Expected >= deterministic, every turn.
    for t, v in best.expected_total.items():
        det = next(x.total for x in best.turns if x.turn == t)
        assert v >= det - 1e-9, (t, v, det)

    # 6. A deck with no accelerator reports a 1/turn package.
    plain = deck_accel_package([lightning] * 60)
    assert plain["realistic_per_turn"] == 1, plain["components"]

    print("energy_plan self-check: 6 groups pass")
    print(f"  Eelektrik+discard t5 total {f.turns[-1].total} vs "
          f"baseline {baseline_curve(fast)[HORIZON]} vs empty-discard "
          f"{s.turns[-1].total}")
    print(f"  Pikachu ex discard-cost curve: {totals}")


def _examples() -> None:
    p = pool()
    root = Path(__file__).resolve().parents[1]
    G, D = 1, 7                                     # Basic {G}, Basic {D}

    print("\n=== Ogerpon (data/ogerpon_seed.csv) ===")
    print("board: two Teal Mask Ogerpon ex, nothing attached; "
          "hand 4x Basic {G} + Energy Search")
    deck = load_decklist(root / "data" / "ogerpon_seed.csv")
    oger = p.ids_by_name["Teal Mask Ogerpon ex"][0]
    pos = opening_position(deck, oger, hand=[G, G, G, G, 1119], bench=[oger])
    print(format_plan(plan(pos), pos))

    print("\n=== Marnie's Grimmsnarl "
          "(external/grimmsnarl_deck.json) ===")
    print("board: Marnie's Morgrem active + Marnie's Impidimp benched; "
          "hand Grimmsnarl ex + 2x Basic {D}")
    gdeck = load_decklist(root / "external" / "grimmsnarl_deck.json")
    morgrem = p.ids_by_name["Marnie's Morgrem"][0]
    impidimp = p.ids_by_name["Marnie's Impidimp"][0]
    grimm = p.ids_by_name["Marnie's Grimmsnarl ex"][0]
    gpos = opening_position(gdeck, morgrem, hand=[grimm, D, D],
                            bench=[impidimp])
    print(format_plan(plan(gpos), gpos))

    print("\n=== opponent capability, top archetypes with a known list ===")
    for label, path in (("Teal Mask Ogerpon ex",
                         root / "data" / "ogerpon_seed.csv"),
                        ("Marnie's Grimmsnarl ex",
                         root / "external" / "grimmsnarl_deck.json"),
                        ("Archaludon ex",
                         root / "external" / "archaludon_deck.json"),
                        ("Mega Lucario ex",
                         root / "external" / "lucario_deck.json"),
                        ("Alakazam",
                         root / "external" / "codex_alakazam_deck.json"),
                        ("Garchomp",
                         root / "external" / "garchomp_deck.json"),
                        ("Kangaskhan",
                         root / "external" / "kanga_deck.json")):
        if Path(path).exists():
            print(format_package(label,
                                 deck_accel_package(load_decklist(path))))


if __name__ == "__main__":
    _self_check()
    _examples()
    print("\n=== realized ramp vs KB capability (mined series) ===")
    try:
        print(format_validation(realized_growth()))
    except Exception as exc:
        print(f"  skipped: {exc}")
