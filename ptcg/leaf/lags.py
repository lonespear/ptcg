"""Own-turn board snapshots and the 3-lag context the leaf features read.

One implementation for both callers: the offline table builder walks a
replay's decision stream per seat, the runtime agent walks the observation
stream it is handed live. Both push a snapshot at the first decision of each
of the seat's OWN turns and read lags strictly before the current turn, so a
training-time lag and a run-time lag are the same quantity.

Turn ownership follows `ptcg/extract.py`: turns alternate from the first
player, so turn 1, 3, 5... belong to the first player. `firstPlayer` is read
off `current`; when absent the decision's turn parity cannot be attributed
and no snapshot is taken (the lag context simply stays shallower).
"""

from __future__ import annotations

SERIES_KEYS = ("prizes_me", "prizes_them", "energy_me", "energy_them",
               "hand_me", "hand_them", "bench_me", "bench_them")


def _side(player: dict) -> tuple[float, float, float, float]:
    """(prizes, energy, hand, bench) off one side of a dict observation."""
    prizes = float(len(player.get("prize") or []))
    energy = 0.0
    bench_n = 0
    for zone in ("active", "bench"):
        for mon in player.get(zone) or []:
            if mon is None:
                continue
            energy += len(mon.get("energies") or [])
            if zone == "bench":
                bench_n += 1
    hand = float(player.get("handCount") or len(player.get("hand") or []))
    return prizes, energy, hand, float(bench_n)


def snapshot(obs: dict, me: int) -> dict | None:
    """The 8 tracked values from seat `me`'s perspective, or None."""
    cur = obs.get("current") or {}
    players = cur.get("players") or []
    if len(players) != 2:
        return None
    pz_m, en_m, hd_m, bn_m = _side(players[me])
    pz_t, en_t, hd_t, bn_t = _side(players[1 - me])
    return {"prizes_me": pz_m, "prizes_them": pz_t,
            "energy_me": en_m, "energy_them": en_t,
            "hand_me": hd_m, "hand_them": hd_t,
            "bench_me": bn_m, "bench_them": bn_t}


class LagTracker:
    """Per-seat own-turn snapshot buffer.

    observe(obs, me) once per decision the seat is asked for;
    ctx(turn) -> the lag_ctx dict `leaf_features` consumes.
    """

    def __init__(self):
        self.snaps: list[tuple[int, dict]] = []   # (turn, snapshot)
        self._turns_seen: set[int] = set()

    def reset(self) -> None:
        self.snaps.clear()
        self._turns_seen.clear()

    def observe(self, obs: dict, me: int, seat: int | None = None) -> None:
        cur = obs.get("current") or {}
        turn = cur.get("turn")
        if not isinstance(turn, int) or turn <= 0:
            return
        if turn in self._turns_seen:
            return
        fp = cur.get("firstPlayer")
        s = seat if seat is not None else me
        if fp not in (0, 1):
            return
        if (turn % 2 == 1) != (fp == s):
            return                       # not this seat's own turn
        snap = snapshot(obs, me)
        if snap is None:
            return
        self._turns_seen.add(turn)
        self.snaps.append((turn, snap))
        if len(self.snaps) > 8:
            del self.snaps[0]

    def ctx(self, turn: int | None) -> dict:
        """Lags strictly before `turn`, newest first, up to 3 deep."""
        if not isinstance(turn, int):
            turn = 10 ** 9
        prior = [s for t, s in self.snaps if t < turn][-3:]
        prior.reverse()
        out: dict = {"depth": len(prior)}
        for key in SERIES_KEYS:
            out[key] = [s[key] for s in prior]
        return out
