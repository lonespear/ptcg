"""Baseline greedy pilot: the stand-in until chunk 2 (utilization) is wired.

Scores every legal option with named heuristics and picks the best; no
lookahead yet. Every decision keeps a printable trace (feature name +
score per option) — the interpretability seed the Strategy track needs.

Never crashes: any handler error falls back to a minimal legal selection.
"""

import random

from .pool import (
    BASIC_ENERGY, ITEM, POKEMON, SPECIAL_ENERGY, STADIUM, SUPPORTER, TOOL,
    pool,
)

# AreaType
AREA_DECK, AREA_HAND, AREA_DISCARD, AREA_ACTIVE, AREA_BENCH = 1, 2, 3, 4, 5
AREA_LOOKING = 12

# OptionType
OPT_NUMBER, OPT_YES, OPT_NO, OPT_CARD = 0, 1, 2, 3
OPT_PLAY, OPT_ATTACH, OPT_EVOLVE, OPT_ABILITY = 7, 8, 9, 10
OPT_DISCARD, OPT_RETREAT, OPT_ATTACK, OPT_END = 11, 12, 13, 14

# SelectType
SEL_MAIN, SEL_YES_NO, SEL_COUNT = 0, 9, 8

# SelectContext values the pilot special-cases
CTX_SETUP_ACTIVE = 1
CTX_MULLIGAN = 42
CTX_IS_FIRST = 41
CTX_ACTIVATE = 43
CTX_FIRST_EFFECT = 44
CTX_MORE_DEVOLVE = 45
CTX_COIN_HEAD = 46

# contexts where selection is a cost we minimize (pick fewest, cheapest)
COSTLY_CONTEXTS = {8, 9, 10, 11, 20, 26, 27, 29, 30, 31, 32, 36}
# contexts targeting Pokemon for damage (prefer near-KO opponents)
DAMAGE_CONTEXTS = {13, 14, 15}
# contexts where fewer selections can't hurt but more usually help
GREEDY_CONTEXTS_DEFAULT_MAX = True


class GreedyPilot:
    """callable(obs_dict) -> list[int], harness/kaggle compatible."""

    def __init__(self, seed: int | None = None, keep_trace: bool = False):
        self.pool = pool()
        self.rng = random.Random(seed)
        self.keep_trace = keep_trace
        self.trace: list[dict] = []

    # ---- helpers -------------------------------------------------------

    def _me(self, obs):
        return obs["current"]["yourIndex"]

    def _player(self, obs, idx):
        return obs["current"]["players"][idx]

    def _resolve_card_id(self, obs, opt) -> int | None:
        """Card ID an option refers to, or None."""
        area = opt.get("area")
        idx = opt.get("index")
        p = opt.get("playerIndex")
        if area is None or idx is None:
            return None
        sel = obs["select"]
        try:
            if area == AREA_DECK and sel.get("deck"):
                return sel["deck"][idx]["id"]
            if area == AREA_LOOKING:
                c = obs["current"]["looking"][idx]
                return c["id"] if c else None
            player = self._player(obs, p if p is not None else self._me(obs))
            if area == AREA_HAND:
                return player["hand"][idx]["id"]
            if area == AREA_ACTIVE:
                c = player["active"][idx]
                return c["id"] if c else None
            if area == AREA_BENCH:
                return player["bench"][idx]["id"]
            if area == AREA_DISCARD:
                return player["discard"][idx]["id"]
        except (IndexError, KeyError, TypeError):
            return None
        return None

    def _resolve_pokemon(self, obs, player_idx, area, idx) -> dict | None:
        try:
            player = self._player(obs, player_idx)
            if area == AREA_ACTIVE:
                return player["active"][idx]
            if area == AREA_BENCH:
                return player["bench"][idx]
        except (IndexError, KeyError, TypeError):
            pass
        return None

    def _card_value(self, card_id: int | None) -> float:
        """Generic desirability of a card, for pick/discard decisions."""
        if card_id is None:
            return 1.0
        c = self.pool.card(card_id)
        if c is None:
            return 1.0
        if c["cardType"] == POKEMON:
            v = 5.0 + c["hp"] / 100.0
            if c["basic"]:
                v += 1.5
            return v
        return {SUPPORTER: 5.0, ITEM: 4.0, TOOL: 3.0, STADIUM: 2.0,
                BASIC_ENERGY: 3.0, SPECIAL_ENERGY: 3.5}.get(c["cardType"], 2.0)

    def _needs_energy(self, pkm: dict) -> bool:
        c = self.pool.card(pkm["id"])
        if not c or not c["attacks"]:
            return False
        attached = len(pkm.get("energies", []))
        costs = [len(self.pool.attack_by_id[a]["energies"])
                 for a in c["attacks"] if a in self.pool.attack_by_id]
        return bool(costs) and attached < max(costs)

    def _active_can_attack(self, pkm: dict) -> bool:
        c = self.pool.card(pkm["id"])
        if not c or not c["attacks"]:
            return False
        attached = len(pkm.get("energies", []))
        return any(len(self.pool.attack_by_id[a]["energies"]) <= attached
                   for a in c["attacks"] if a in self.pool.attack_by_id)

    # ---- main-phase scoring -------------------------------------------

    def _score_main_option(self, obs, opt) -> tuple[float, str]:
        me = self._me(obs)
        opp = 1 - me
        t = opt["type"]

        if t == OPT_PLAY:
            cid = None
            try:
                cid = self._player(obs, me)["hand"][opt["index"]]["id"]
            except (IndexError, KeyError, TypeError):
                pass
            c = self.pool.card(cid) if cid else None
            if c is None:
                return 4.0, "play:unknown"
            if c["cardType"] == POKEMON:
                return 8.0, f"play:bench-basic({c['name']})"
            if c["cardType"] == SUPPORTER:
                return 7.5, f"play:supporter({c['name']})"
            if c["cardType"] == ITEM:
                return 6.0, f"play:item({c['name']})"
            if c["cardType"] == TOOL:
                return 3.0, f"play:tool({c['name']})"
            if c["cardType"] == STADIUM:
                return 2.5, f"play:stadium({c['name']})"
            return 4.0, f"play:{c['name']}"

        if t == OPT_ATTACH:
            target = self._resolve_pokemon(
                obs, me, opt.get("inPlayArea"), opt.get("inPlayIndex"))
            if target is None:
                return 2.0, "attach:unknown-target"
            needy = self._needs_energy(target)
            active = opt.get("inPlayArea") == AREA_ACTIVE
            if needy and active:
                return 10.0, "attach:active-needs-energy"
            if needy:
                return 6.5, "attach:bench-needs-energy"
            return 1.0, "attach:surplus"

        if t == OPT_EVOLVE:
            return 9.0, "evolve"

        if t == OPT_ABILITY:
            return 3.0, "ability"

        if t == OPT_DISCARD:
            return -2.0, "discard-in-play"

        if t == OPT_RETREAT:
            active = self._player(obs, me)["active"]
            stuck = bool(active and active[0]
                         and not self._active_can_attack(active[0]))
            return (4.0, "retreat:active-stuck") if stuck else (-5.0, "retreat")

        if t == OPT_ATTACK:
            atk = self.pool.attack_by_id.get(opt.get("attackId"))
            dmg = atk["damage"] if atk else 0
            bonus = 0.0
            label = f"attack:{atk['name'] if atk else '?'}"
            try:
                opp_active = self._player(obs, opp)["active"][0]
                my_active = self._player(obs, me)["active"][0]
                opp_card = self.pool.card(opp_active["id"])
                my_card = self.pool.card(my_active["id"])
                eff = dmg
                if opp_card and my_card:
                    if opp_card["weakness"] == my_card["energyType"]:
                        eff *= 2
                    if opp_card["resistance"] == my_card["energyType"]:
                        eff = max(0, eff - 30)
                if opp_active and eff >= opp_active["hp"] > 0:
                    bonus = 20.0
                    label += ":KO"
            except (IndexError, KeyError, TypeError):
                eff = dmg
            return 4.0 + eff / 100.0 + bonus, label

        if t == OPT_END:
            tac = obs["current"].get("turnActionCount", 0)
            return 0.2 + max(0, tac - 25), "end-turn"

        return 1.0, f"opt-type-{t}"

    # ---- entry ---------------------------------------------------------

    def __call__(self, obs: dict) -> list[int]:
        try:
            return self._decide(obs)
        except Exception:
            sel = obs.get("select") or {}
            n = max(sel.get("minCount", 1), 0)
            return list(range(n))

    def _decide(self, obs: dict) -> list[int]:
        sel = obs["select"]
        options = sel["option"]
        ctx = sel.get("context", 0)
        stype = sel.get("type", 0)
        lo, hi = sel["minCount"], sel["maxCount"]

        if stype == SEL_MAIN:
            scored = [(self._score_main_option(obs, o), i)
                      for i, o in enumerate(options)]
            scored.sort(key=lambda s: (-s[0][0], self.rng.random()))
            if self.keep_trace:
                self.trace.append({
                    "context": ctx,
                    "chosen": scored[0][0][1],
                    "options": [(lbl, round(sc, 2))
                                for (sc, lbl), _ in scored[:8]],
                })
            return [scored[0][1]]

        if stype == SEL_YES_NO:
            yes = next((i for i, o in enumerate(options) if o["type"] == OPT_YES), 0)
            no = next((i for i, o in enumerate(options) if o["type"] == OPT_NO), 0)
            if ctx == CTX_MULLIGAN:
                hand = self._player(obs, self._me(obs))["hand"] or []
                has_basic = any(
                    (c := self.pool.card(h["id"])) and c["cardType"] == POKEMON
                    and c["basic"] for h in hand)
                return [no if has_basic else yes]
            if ctx == CTX_MORE_DEVOLVE:
                return [no]
            return [yes]  # IS_FIRST, ACTIVATE, FIRST_EFFECT, COIN_HEAD

        if stype == SEL_COUNT:
            best = max(range(len(options)),
                       key=lambda i: options[i].get("number", 0))
            return [best]

        # card-ish selections: value every option, then take cheap-few for
        # costs, near-KO opponents for damage, best-many otherwise
        if ctx in DAMAGE_CONTEXTS:
            def dmg_key(i):
                o = options[i]
                pkm = self._resolve_pokemon(
                    obs, o.get("playerIndex"), o.get("area"), o.get("index"))
                mine = o.get("playerIndex") == self._me(obs)
                hp = pkm["hp"] if pkm else 999
                return (mine, hp)  # opponents first, lowest HP first
            order = sorted(range(len(options)), key=dmg_key)
            return order[:hi]

        values = [(self._card_value(self._resolve_card_id(obs, o)), i)
                  for i, o in enumerate(options)]
        if ctx in COSTLY_CONTEXTS:
            values.sort(key=lambda v: v[0])
            return [i for _, i in values[:lo]]
        if ctx == CTX_SETUP_ACTIVE:
            values.sort(key=lambda v: -v[0])
            return [values[0][1]]
        values.sort(key=lambda v: -v[0])
        return [i for _, i in values[:hi]]


class JonDayPilot:
    """Adapter over agent/main.py (chunk 2's live agent) for GA evaluation.

    Two things the raw module can't do alone:
    - seat-correct self-model: the module's _MY_DECK global normally comes
      from deck.csv, but GA matches pit two arbitrary decks; the harness
      calls bind_deck() per game and we re-point the global on every call
      (both seats share the module, so per-call re-pointing is required).
    - search throttle: forward search (~4.4 ms/decision) is ladder-grade
      but ~300x too slow for the GA inner loop. search=False gives the
      rules-only policy for fitness; use search=True for gates and
      tournaments.
    """

    def __init__(self, seed: int | None = None, search: bool = False):
        import agent.main as _jon  # real import (not exec): __file__ works,
        self._jon = _jon           # so priors resolve next to the module
        self.search = search
        self.deck: list[int] | None = None

    def bind_deck(self, deck: list[int]) -> None:
        self.deck = list(deck)

    def __call__(self, obs: dict) -> list[int]:
        self._jon._MY_DECK = self.deck
        self._jon.SEARCH_ENABLED = self.search
        return self._jon.agent(obs)


class ExternalPilot:
    """Load a single-file competition agent (main.py style) as a pilot.

    Works for community agents pulled from public Kaggle notebooks: imports
    the module (engine bootstrap must have run), finds its entrypoint, and
    injects the seat's deck into the module's known deck globals per call.
    """

    DECK_GLOBALS = ("my_deck", "_MY_DECK", "MY_DECK")

    def __init__(self, path: str, seed: int | None = None):
        import importlib.util
        import sys
        spec = importlib.util.spec_from_file_location(
            f"ext_agent_{abs(hash(path))}", path)
        mod = importlib.util.module_from_spec(spec)
        # dataclass decorators resolve their defining module through
        # sys.modules, so register before executing
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        self.mod = mod
        self.fn = getattr(mod, "competition_entrypoint", None) or mod.agent
        self.deck: list[int] | None = None

    def bind_deck(self, deck: list[int]) -> None:
        self.deck = list(deck)

    def __call__(self, obs: dict) -> list[int]:
        if self.deck is not None:
            for g in self.DECK_GLOBALS:
                if hasattr(self.mod, g):
                    setattr(self.mod, g, list(self.deck))
        return list(self.fn(obs))
