# futures/ — Path B: Full-Turn Futures (Phase 0 / 0.5: sampler realism)

Provenance: D45 (Three Paths ruling, 2026-08-07). Path B is the pursued path;
this folder is the new build. Nothing in `agent/` or the submission bundle is
modified; code here imports repo data read-only.

## The bet

Today's agent does 1-ply search with a linear evaluator and never simulates the
opponent's turn (`SEARCH_OPP_BRANCH = 0`); 16.8% of realized HP loss comes from
mechanics that lookahead never sees. Path B: for each candidate action, play it,
then **sample the opponent's entire turn** from the band/archetype/turn/ordinal
reply tables used as a *stochastic policy* — right on average — where the three
struck 2-ply attempts used them as an adversarial best-reply oracle — right
pointwise — and failed three gates (0.407 / 0.467 / 0.476). Then our reply turn
under a fast policy, 2-3 turns deep, 20-40 rollouts per candidate over
determinized hidden states, futures scored with the existing linear evaluator,
candidates ranked by mean. The grader budget affords it: 600 s bank per episode,
no per-move deadline, the current agent uses ~190 s.

Phase 0 is the pre-registered kill test: **before any rollout engineering, the
turn-sampler must prove it plays like the field.** If it cannot reproduce real
opponent turns offline, rollouts built on it price futures with a fiction and
Path B pauses; the divergences are then the finding.

## Data and holdout discipline

The shipped tables (`data/opponent_policy.json`) were fit on all seven mined
days (2026-07-31 … 2026-08-06), so they cannot be validated against any of
those days. `fit_policy.py` therefore refits the same tables on the six days
**2026-07-31 … 2026-08-05** (train) and every comparison below runs on
**2026-08-06 only** (holdout: 760,265 decision rows, ~65,000 seat-turns, never
seen by the fit). The damage/prize head can only be fit where `series.parquet`
exists (2026-08-04, 2026-08-05 in train); the holdout day has it too.

## What the sampler is

`sampler.py` — dependency-light (stdlib only). Given (rating band, opponent
archetype, game turn), it emits a full opponent turn: a sequence of main-menu
action types (ability / evolve / play / attach / attack / retreat / end_turn),
sampled per visit from the propensity tables
`P(chosen | band, arch, turn-bucket, ordinal) / P(available | turn-bucket,
ordinal)` with the L0→L3 backoff the agent already uses, renormalized over the
types available at that visit. Offline (no engine), availability is itself
sampled from the measured availability rates, with hard rules: `end_turn`
always available, `attack` and `end_turn` terminal (0 of 65,287 holdout turns
continue after an attack), at most one attach and one retreat per turn (the
5.8% of real turns with a second attach are a known simplification, reported in
the attach metric). Where no cell exists at any backoff level, the fallback is
the rule-policy ordering the agent ships (attack-first priority); L3 cells
cover every ordinal, so offline this is a guard, not a path.

Because `series.parquet`'s "cumulative damage" column is actually
damage-counters-on-board (it drops when a damaged mon leaves play), damage per
turn is constructed from paired snapshots: for seat *s*, turn *t*, **chip
damage** = counters on the defender's board at the top of the defender's turn
*t+1* minus counters at the top of *s*'s turn *t* (clamped at 0), and
**prizes taken** = *s*'s prize delta over the same window read from the
defender's snapshot. KO'd mons leave the board, so a KO turn shows small chip
and ≥1 prize; the two observables together are the damage signature. The final
turn of each game has no following snapshot and is excluded — identically in
train and holdout, so the comparison is unbiased, but **terminal-turn damage is
not validated offline** (Phase 1 engine rollouts compute it exactly). The
sampler carries a damage/prize head fit on train days: empirical joint
(prizes, chip | attacked, archetype, turn-bucket), and a disruption head
P(disruption card | play visit, archetype) where the disruption class is
{Judge, Unfair Stamp, Boss's Orders, Prime Catcher, Eri} resolved by card id
from the (uncommitted, licensed) engine dump at fit time — only rates are
stored.

## Validation (`validate_sampler.py`)

For each archetype: every real holdout seat-turn provides conditioning
(band, archetype, turn number); the sampler draws K=3 synthetic turns per real
turn, so the conditioning distribution matches by construction. Compared, real
vs sampled:

1. action-type mix (visit-level distribution over the seven types, multi-option
   visits, pooled and per turn-bucket) — total variation distance;
2. damage per turn — chip mean, chip Wasserstein-1, prizes/turn mean, over all
   non-terminal turns;
3. energy-attach rate — share of turns with ≥1 attach;
4. attack rate — share of turns ending in an attack;
5. disruption usage — P(disruption | play visit) on the holdout positions
   sample, with a Wilson 95% interval (the positions sample is ~2,000
   menus/day, so n is small);
6. attack-choice agreement — on holdout multi-attack menus, does the sampler's
   sub-choice rule (cheapest KO, else biggest hit — the agent's measured rule)
   match the field's pick.

Archetypes validated: the five asked for — Marnie's Grimmsnarl ex, Dudunsparce
(the Dudunsparce/Alakazam engine's corpus label), Archaludon ex, Mega Lopunny
ex, Mega Kangaskhan ex — plus the remaining field-volume leaders Fezandipiti
ex, Dragapult ex, Teal Mask Ogerpon ex. Dudunsparce (8.2k decisions over seven
days) and Archaludon ex (1.6k) are below the tables' TOP_ARCH cut and back off
to OTHER; they are reported with their tiny holdout n and wide intervals, and
their verdicts are labeled OTHER-backoff.

## Pre-registered verdict criteria (fixed before any comparison was run)

Per archetype, the sampler is **GO** iff all of:

- **A. Action mix:** pooled visit-level TV(sampled, real) ≤ **0.10**.
  Justification: distinct archetypes in the shipped tables differ from each
  other by mean TV 0.29-0.36 (the fit's own self-check), so 0.10 keeps the
  sampler ~3x closer to its target deck than decks are to each other — it is
  recognizably playing *that* archetype, not a field average.
- **B. Chip damage:** |mean(sampled) − mean(real)| ≤ **max(10 HP, 15% of real
  mean)** and Wasserstein-1 ≤ **20 HP**. Justification: 10-20 HP is below one
  increment of the smallest common attack step in this pool; a per-turn bias
  smaller than the smallest attack cannot reorder candidate actions whose
  sampled futures differ by a real attack's worth of damage.
- **C. Prizes:** |mean prizes/turn (sampled) − (real)| ≤ **0.05**.
  Justification: over a 3-turn rollout that compounds to <0.15 prizes, small
  against the ≥1-prize gaps the rollout ranking must resolve.
- **D. Attack rate:** |P(turn attacks) sampled − real| ≤ **5 points**.
  Justification: attack frequency drives both damage observables; a 5-point
  error at ~90 HP/attack is ~4.5 HP/turn of systematic bias, inside B's budget.

Attach-rate (±5 pts), disruption (inside the holdout Wilson 95% interval or
±3 pts), and attack-choice agreement (≥0.60, gating only if holdout n ≥ 30 —
multi-attack menus are rare in the positions sample) are **reported flags**,
not kill criteria.

**Phase 0 verdict: GO** iff archetypes passing A-D cover ≥ **70%** of holdout
field volume (share of holdout seat-turns). Otherwise **NO-GO**: Path B pauses
and the divergence table is the deliverable.

## Post-verdict diagnostics (added after the v1 run, labeled as such)

The pre-registered test above ran first and its verdict stands as computed
(see RESULTS.md: NO-GO at 33.1% coverage). Three diagnostics were added
afterwards to attribute the failures; none of them moves the gate:

- `validate_sampler.py --persistent` — a second offline availability
  idealization (monotone unlock instead of iid per-visit draws), to bracket
  the availability surrogate's contribution;
- `menu_replay` (rides the v1 run) — the policy evaluated on the *real*
  holdout menus from the positions sample, no availability model at all: the
  test that isolates the reply tables themselves;
- `diag_damage_head.py` — the damage/prize head drawn against each real
  turn's real attacked flag, separating head error from action-model error.

## Phase 0.5 (engine legality; pre-registered follow-up)

Phase 0's NO-GO decomposed onto the offline availability surrogate, so the
recommended follow-up ran: `sampler.sample_action()` wired into the engine's
real legal-menu loop (`search_begin`/`search_step`) on 1,101 holdout
seat-turns from the mined positions sample, hidden zones determinized from
the replay-derived deck priors through the agent's posterior machinery, the
identical A-D tolerances and 70% coverage rule re-applied. Verdict of
record: **NO-GO at 0% coverage** — real legality fixed A and mostly B, and
the failure relocated to prizes and attack rate (turn-level compounding the
decision-level tests cannot see). Full table, decomposition and disposition:
`RESULTS_phase05.md`.

## Files

- `fit_policy.py` — refit on train days (adapted from
  `scripts/build_opponent_policy.py`, plus damage/prize/disruption heads) →
  `policy_train.json`
- `sampler.py` — the turn sampler (stdlib only; also the Phase 1 API:
  `sample_action(...)` against a live menu)
- `validate_sampler.py` — holdout comparison → `results.json`
  (`--persistent` → `results_v2.json`, diagnostic)
- `diag_damage_head.py` — head-only diagnostic → `results_head_diag.json`
- `RESULTS.md` — Phase 0 table, verdicts, divergences, recommendation
- `validate_sampler_engine.py` — Phase 0.5 engine-playout gate →
  `results_engine.json`
- `diag_engine.py` — Phase 0.5 post-verdict decomposition →
  `results_engine_diag.json`
- `RESULTS_phase05.md` — Phase 0.5 table, decomposition, disposition

Run (system python3 carries pyarrow; the venv does not):

    /usr/bin/python3 futures/fit_policy.py
    /usr/bin/python3 futures/validate_sampler.py

Phase 0.5 needs python ≥ 3.10 for the engine bindings:

    ~/miniforge3/bin/python3 futures/validate_sampler_engine.py
    ~/miniforge3/bin/python3 futures/diag_engine.py
