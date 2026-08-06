"""Decision-stability certificates (D16.4): how often does the agent give the
same answer to the same question?

The agent's forward search samples the hidden information — our own deck and
prizes, and the opponent's deck, hand and prizes — afresh on every call. Two
calls on one identical observation are therefore two different guesses about
the same board, and where they disagree the agent's move is an artifact of the
sample rather than a property of the position. That disagreement rate is the
Strategy rubric's "consistency" criterion expressed as a number.

Method: play G games with the agent under test on seat 0. At every main-phase
decision offering 3 or more options, ask the agent K times about the same
observation before playing the first answer, and record the answers. The agent
is a black box here; nothing reaches into its search.

Output: overall invariance rate (all K answers identical), mean modal
agreement (how large the plurality is when they disagree), the same two
figures per SelectContext, and every unstable state with its option labels and
the frequency of each answer.

Usage: python -m ptcg.creation.stability --games 50
       python -m ptcg.creation.stability --games 20 --k 8 \
              --deck priors:0 --opp priors:3
"""

import argparse
import json
import statistics
import sys
import time
from collections import Counter, defaultdict

from cg.game import battle_finish, battle_select, battle_start

from .goldfish import read_deck
from .pilots import JonDayPilot
from .pool import pool

RESULT = 23

# OptionType — the main menu is any select offering one of these.
OPT_PLAY, OPT_ATTACH, OPT_EVOLVE, OPT_ABILITY = 7, 8, 9, 10
OPT_DISCARD, OPT_RETREAT, OPT_ATTACK, OPT_END = 11, 12, 13, 14
MAIN_TYPES = {OPT_PLAY, OPT_ATTACH, OPT_EVOLVE, OPT_ABILITY,
              OPT_RETREAT, OPT_ATTACK, OPT_END}
OPT_NAMES = {OPT_PLAY: "play", OPT_ATTACH: "attach", OPT_EVOLVE: "evolve",
             OPT_ABILITY: "ability", OPT_DISCARD: "discard",
             OPT_RETREAT: "retreat", OPT_ATTACK: "attack", OPT_END: "end"}

AREA_HAND, AREA_DISCARD, AREA_ACTIVE, AREA_BENCH, AREA_PRIZE = 2, 3, 4, 5, 6
ZONE = {AREA_HAND: "hand", AREA_DISCARD: "discard", AREA_ACTIVE: "active",
        AREA_BENCH: "bench", AREA_PRIZE: "prize"}


def _card_name(cards, obs: dict, area, idx, pidx) -> str | None:
    """Name of the card an (area, index) pair points at, or None."""
    zone = ZONE.get(area)
    if zone is None or idx is None:
        return None
    try:
        card = (obs["current"]["players"][pidx].get(zone) or [])[idx]
    except (IndexError, KeyError, TypeError):
        return None
    if not card:
        return None
    try:
        return cards.name(card["id"])
    except Exception:
        return str(card.get("id"))


def _label(cards, obs: dict, opt: dict) -> str:
    """`attach(Basic {D} Energy -> Impidimp)` — enough to read a disagreement.

    A PLAY option carries a bare hand index with no area, so hand is the
    default zone; ATTACH and EVOLVE carry their board target separately.
    """
    kind = OPT_NAMES.get(opt.get("type"), f"type{opt.get('type')}")
    me = obs["current"]["yourIndex"]
    pidx = opt.get("playerIndex")
    pidx = me if pidx is None else pidx
    area = opt.get("area")
    if area is None and opt.get("index") is not None:
        area = AREA_HAND
    src = _card_name(cards, obs, area, opt.get("index"), pidx)
    tgt = _card_name(cards, obs, opt.get("inPlayArea"),
                     opt.get("inPlayIndex"), me)
    if src and tgt:
        return f"{kind}({src} -> {tgt})"
    if src or tgt:
        return f"{kind}({src or tgt})"
    return kind


def _turn(obs: dict) -> int:
    return int((obs.get("current") or {}).get("turn", 0) or 0)


def probe_game(pilot, opponent, deck: list[int], opp_deck: list[int],
               k: int = 5, seat: int = 0, max_selects: int = 20000) -> dict:
    """One game; returns the decision records taken on `seat`."""
    cards = pool()
    agents = (pilot, opponent) if seat == 0 else (opponent, pilot)
    decks = (deck, opp_deck) if seat == 0 else (opp_deck, deck)
    obs, start = battle_start(*decks)
    if obs is None:
        raise RuntimeError(f"battle_start failed: errorType={start.errorType}")
    for a, d in zip(agents, decks):
        if hasattr(a, "bind_deck"):
            a.bind_deck(d)

    records: list[dict] = []
    selects = 0
    try:
        while selects < max_selects:
            if any(lg["type"] == RESULT for lg in obs["logs"]):
                break
            actor = obs["current"]["yourIndex"]
            sel = obs["select"]
            options = sel["option"]
            is_main = any(o.get("type") in MAIN_TYPES for o in options)
            if actor == seat and len(options) >= 3:
                answers = [tuple(pilot(obs)) for _ in range(k)]
                freq = Counter(answers)
                top, top_n = freq.most_common(1)[0]
                records.append({
                    "turn": _turn(obs),
                    "context": sel.get("context", 0),
                    "select_type": sel.get("type", 0),
                    "main": is_main,
                    "n_options": len(options),
                    "k": k,
                    "invariant": len(freq) == 1,
                    "modal_agreement": top_n / k,
                    "answers": {",".join(map(str, a)): n
                                for a, n in freq.most_common()},
                    "labels": {",".join(map(str, a)):
                               "+".join(_label(cards, obs, options[i])
                                        for i in a if 0 <= i < len(options))
                               for a in freq},
                })
                action = list(top)
            else:
                action = agents[actor](obs)
            obs = battle_select(action)
            selects += 1
        return {"records": records, "selects": selects}
    finally:
        battle_finish()


def stability(deck: list[int], opp_deck: list[int], n_games: int = 50,
              k: int = 5, seed: int = 11, max_unstable: int = 25) -> dict:
    """Invariance over `n_games`, seats alternating so the profile covers
    both the play and the draw."""
    pilot = JonDayPilot(seed=seed, search=True)
    opponent = JonDayPilot(seed=seed + 1, search=False)
    t0 = time.time()
    records: list[dict] = []
    for g in range(n_games):
        out = probe_game(pilot, opponent, deck, opp_deck, k=k, seat=g % 2)
        records.extend(out["records"])

    if not records:
        return {"n_games": n_games, "k": k, "n_decisions": 0,
                "note": "no decision with 3+ options was reached"}

    main = [r for r in records if r["main"]]
    other = [r for r in records if not r["main"]]
    by_ctx: dict[int, list[dict]] = defaultdict(list)
    for r in records:
        by_ctx[r["context"]].append(r)

    def summary(rs: list[dict]) -> dict:
        if not rs:
            return {"n_decisions": 0}
        return {
            "n_decisions": len(rs),
            "invariance_rate": round(sum(r["invariant"] for r in rs)
                                     / len(rs), 4),
            "mean_modal_agreement": round(
                statistics.mean(r["modal_agreement"] for r in rs), 4),
            "mean_options": round(
                statistics.mean(r["n_options"] for r in rs), 2),
        }

    unstable = [r for r in records if not r["invariant"]]
    unstable.sort(key=lambda r: (r["modal_agreement"], -r["n_options"]))

    return {
        "n_games": n_games,
        "k": k,
        "elapsed_s": round(time.time() - t0, 1),
        "overall": summary(main),          # the certificate: main decisions
        "all_selects": summary(records),
        "non_main": summary(other),
        "by_context": {str(c): summary(rs)
                       for c, rs in sorted(by_ctx.items())},
        "by_n_options": {str(n): summary([r for r in main
                                          if r["n_options"] == n])
                         for n in sorted({r["n_options"] for r in main})},
        "n_unstable": len(unstable),
        "unstable_states": unstable[:max_unstable],
    }


def format_stability(rep: dict) -> str:
    if rep.get("n_decisions") == 0:
        return rep.get("note", "no decisions probed")
    o, n = rep["overall"], rep["non_main"]
    lines = [
        f"games {rep['n_games']}   K {rep['k']}   "
        f"probed selects {rep['all_selects']['n_decisions']} "
        f"({o['n_decisions']} main)   ({rep['elapsed_s']}s)",
        f"MAIN decision invariance  {o['invariance_rate']:.1%}   "
        f"mean modal agreement {o['mean_modal_agreement']:.1%}   "
        f"mean options {o['mean_options']}",
    ]
    if n.get("n_decisions"):
        lines.append(f"non-main selects          "
                     f"{n['invariance_rate']:.1%}   over "
                     f"{n['n_decisions']} decisions (no search runs there)")
    lines += ["", "by SelectContext",
              f"{'ctx':>5}{'n':>7}{'invariant':>12}{'modal':>9}{'options':>9}"]
    for c, s in rep["by_context"].items():
        lines.append(f"{c:>5}{s['n_decisions']:>7}"
                     f"{s['invariance_rate']:>11.1%}"
                     f"{s['mean_modal_agreement']:>9.1%}"
                     f"{s['mean_options']:>9.2f}")
    lines += ["", "by option count",
              f"{'opts':>5}{'n':>7}{'invariant':>12}{'modal':>9}"]
    for n, s in rep["by_n_options"].items():
        lines.append(f"{n:>5}{s['n_decisions']:>7}"
                     f"{s['invariance_rate']:>11.1%}"
                     f"{s['mean_modal_agreement']:>9.1%}")
    lines += ["", f"unstable states: {rep['n_unstable']} "
                  f"(worst {len(rep['unstable_states'])} shown)"]
    for r in rep["unstable_states"]:
        picks = "  ".join(f"[{a}] {r['labels'][a]} x{n}"
                          for a, n in r["answers"].items())
        lines.append(f"  turn {r['turn']:>2} ctx {r['context']:>2} "
                     f"opts {r['n_options']:>2}  modal "
                     f"{r['modal_agreement']:.0%}: {picks}")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--games", type=int, default=50)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--deck", default="priors:0",
                    help="deck.json / deck.csv / priors:<rank>")
    ap.add_argument("--opp", default="priors:1")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default=None, help="write the report as JSON")
    a = ap.parse_args()

    rep = stability(read_deck(a.deck), read_deck(a.opp),
                    n_games=a.games, k=a.k, seed=a.seed)
    print(format_stability(rep))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(rep, fh, indent=2)
        print(f"\nwrote {a.out}", file=sys.stderr)
