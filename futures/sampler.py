"""Full-turn opponent sampler — Path B Phase 0 (futures/README.md, D45).

Samples an entire opponent turn as a sequence of main-menu action types from
the band/archetype/turn/ordinal reply tables used as a stochastic policy, with
the same L0->L3 backoff and propensity construction agent/main.py uses for its
(dormant) opponent-branch ordering. Stdlib only, so the same file can ship
inside a Phase 1 bundle.

Two entry points:

- ``sample_action(band, arch, turn, ordinal, available, rng)`` — Phase 1 API:
  the engine supplies the truly available types; returns one type name.
- ``sample_turn(band, arch, turn, rng)`` — Phase 0 offline mode: availability
  is itself sampled from the measured availability rates (no engine), with
  hard rules layered on top: end_turn always available; attack and end_turn
  terminal; at most one attach and one retreat per turn.

Where no cell exists at any backoff level (L3 covers every ordinal, so this is
a guard), the fallback is the rule-policy priority the agent ships for its own
forced ordering: attack, evolve, ability, play, attach, retreat, end_turn.

The damage/prize head and disruption head ride along for offline validation:
given the sampled turn attacked (or not), (prizes, chip damage) is drawn from
the train-day empirical joint for (archetype, turn bucket); a sampled play
visit is marked disruption with the archetype's measured rate.
"""

from __future__ import annotations

import json
import os
import random

TYPES = ["ability", "evolve", "play", "attach", "attack", "retreat", "end_turn"]
RULE_PRIORITY = ["attack", "evolve", "ability", "play", "attach", "retreat",
                 "end_turn"]
AVAIL_FLOOR = 0.02          # same guard the agent applies to availability
MAX_VISITS = 25             # holdout p99.9 turn length is ~22 visits

_TURN_BUCKETS = [(2, "1-2"), (5, "3-5"), (9, "6-9")]


def turn_bucket(turn: int) -> str:
    for hi, name in _TURN_BUCKETS:
        if turn <= hi:
            return name
    return "10+"


def ord_bucket(k: int) -> str:
    return str(k) if k < 3 else "3+"


class TurnSampler:
    def __init__(self, policy_path: str | None = None):
        if policy_path is None:
            policy_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "policy_train.json")
        with open(policy_path) as fh:
            self.pol = json.load(fh)
        self.cells = self.pol.get("cells") or {}
        self.avail = self.pol.get("availability") or {}
        self.archetypes = set(self.pol.get("archetypes") or ())
        self.default_band = self.pol.get("default_band", "900-1050")
        self.head = self.pol.get("damage_head") or {}
        self.disrupt = self.pol.get("disruption_head") or {}
        # pre-decode damage-head histograms into (values, cumweights) pairs
        self._hcache: dict = {}

    # -- table lookups ------------------------------------------------------
    def arch_label(self, arch: str) -> str:
        return arch if arch in self.archetypes else "OTHER"

    def _cell(self, band: str, arch: str, tb: str, ob: str) -> dict:
        a = self.arch_label(arch)
        for key in ("L0|%s|%s|%s|%s" % (band or self.default_band, a, tb, ob),
                    "L1|%s|%s|%s" % (a, tb, ob),
                    "L2|%s|%s" % (tb, ob),
                    "L3|%s" % ob):
            cell = self.cells.get(key)
            if cell:
                return cell.get("p") or {}
        return {}

    def _avail_rate(self, tb: str, ob: str) -> dict:
        for key in ("%s|%s" % (tb, ob), ob):
            rec = self.avail.get(key)
            if rec and rec.get("rate"):
                return rec["rate"]
        return {}

    def type_scores(self, band: str, arch: str, turn: int, ordinal: int) -> dict:
        """type -> propensity P(chosen|cell)/P(available|turn,ordinal)."""
        tb, ob = turn_bucket(turn), ord_bucket(ordinal)
        chosen = self._cell(band, arch, tb, ob)
        if not chosen:
            return {}
        avail = self._avail_rate(tb, ob)
        return {t: p / max(avail.get(t, 1.0), AVAIL_FLOOR)
                for t, p in chosen.items()}

    # -- Phase 1 API --------------------------------------------------------
    def sample_action(self, band: str, arch: str, turn: int, ordinal: int,
                      available: list, rng: random.Random) -> str:
        """One action type, sampled over the truly available types."""
        scores = self.type_scores(band, arch, turn, ordinal)
        cand = [(t, scores[t]) for t in available if scores.get(t)]
        total = sum(w for _, w in cand)
        if not cand or total <= 0:
            for t in RULE_PRIORITY:          # fallback: rule-policy priority
                if t in available:
                    return t
            return "end_turn"
        x = rng.random() * total
        for t, w in cand:
            x -= w
            if x <= 0:
                return t
        return cand[-1][0]

    # -- Phase 0 offline mode ----------------------------------------------
    def _offline_available(self, tb: str, ob: str, used: set,
                           rng: random.Random) -> list:
        """v1 (pre-registered): iid per-visit availability draws."""
        rates = self._avail_rate(tb, ob)
        out = []
        for t in TYPES:
            if t == "end_turn":
                out.append(t)
                continue
            if t in ("attach", "retreat") and t in used:
                continue          # hard rule: one per turn (README)
            if rng.random() < rates.get(t, 0.0):
                out.append(t)
        return out

    def _persistent_available(self, tb: str, k: int, state: dict, used: set,
                              rng: random.Random) -> list:
        """v2 (diagnostic): monotone availability within the turn.

        The measured availability of attach *declines* with ordinal only
        because players attach and it leaves the menu, and attack/retreat
        availability *rises* because energy accrues — neither is an iid
        per-visit coin. v2 treats them as unlock processes: attach is drawn
        once at the first visit (its ordinal-0 rate) and persists until used;
        attack and retreat unlock with the hazard implied by the monotone-ized
        cumulative rates and persist (retreat masked after use). play /
        ability / evolve stay iid as in v1. Documented in README; the
        pre-registered verdict is v1's.
        """
        rates_k = self._avail_rate(tb, ord_bucket(k))
        out = ["end_turn"]
        if k == 0:
            state["attach"] = rng.random() < self._avail_rate(tb, "0").get(
                "attach", 0.0)
            state["attack"] = False
            state["retreat"] = False
            state["R"] = {"attack": 0.0, "retreat": 0.0}
        for t in ("attack", "retreat"):
            prev = state["R"][t]
            cur = max(prev, rates_k.get(t, 0.0))
            if not state[t] and cur > prev and prev < 1.0:
                if rng.random() < (cur - prev) / (1.0 - prev):
                    state[t] = True
            state["R"][t] = cur
        if state["attach"] and "attach" not in used:
            out.append("attach")
        if state["attack"]:
            out.append("attack")
        if state["retreat"] and "retreat" not in used:
            out.append("retreat")
        for t in ("ability", "evolve", "play"):
            if rng.random() < rates_k.get(t, 0.0):
                out.append(t)
        return out

    def sample_turn(self, band: str, arch: str, turn: int,
                    rng: random.Random, avail_model: str = "iid") -> dict:
        """A full offline turn: actions plus damage/prize/disruption draws."""
        actions = []
        used: set = set()
        state: dict = {}
        disrupted = False
        for k in range(MAX_VISITS):
            tb, ob = turn_bucket(turn), ord_bucket(k)
            if avail_model == "persistent":
                available = self._persistent_available(tb, k, state, used, rng)
            else:
                available = self._offline_available(tb, ob, used, rng)
            act = self.sample_action(band, arch, turn, k, available, rng)
            actions.append(act)
            used.add(act)
            if act == "play" and rng.random() < self._disr_rate(arch):
                disrupted = True
            if act in ("attack", "end_turn"):
                break
        else:
            actions.append("end_turn")
        attacked = actions[-1] == "attack"
        prizes, chip = self._draw_damage(arch, turn, attacked, rng)
        return {"actions": actions, "attacked": attacked,
                "attached": "attach" in used, "n_attach": int("attach" in used),
                "disrupted": disrupted, "prizes": prizes, "chip": chip}

    # -- heads --------------------------------------------------------------
    def _disr_rate(self, arch: str) -> float:
        rec = (self.disrupt.get(self.arch_label(arch))
               or self.disrupt.get("*") or {})
        return rec.get("rate", 0.0)

    def _head_cell(self, arch: str, tb: str) -> dict:
        for key in ("%s|%s" % (self.arch_label(arch), tb), "*|%s" % tb, "*"):
            cell = self.head.get(key)
            if cell:
                return cell
        return {}

    def _draw_hist(self, key: str, hist: dict, rng: random.Random):
        cached = self._hcache.get(key)
        if cached is None:
            vals, cum, tot = [], [], 0
            for v, c in hist.items():
                tot += c
                vals.append(int(v))
                cum.append(tot)
            cached = (vals, cum, tot)
            self._hcache[key] = cached
        vals, cum, tot = cached
        if not vals:
            return 0
        x = rng.random() * tot
        for v, c in zip(vals, cum):
            if x <= c:
                return v
        return vals[-1]

    def _draw_damage(self, arch: str, turn: int, attacked: bool,
                     rng: random.Random):
        tb = turn_bucket(turn)
        cell = self._head_cell(arch, tb)
        if not cell:
            return 0, 0
        sub = cell["A" if attacked else "N"]
        key = "%s|%s|%s" % (self.arch_label(arch), tb, "A" if attacked else "N")
        prizes = self._draw_hist(key + "|p", sub.get("prizes") or {}, rng)
        chip = self._draw_hist(key + ("|c1" if prizes >= 1 else "|c0"),
                               (sub.get("chip1") if prizes >= 1
                                else sub.get("chip0")) or {}, rng)
        return prizes, chip
