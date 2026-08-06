# Status — Austin's side (deck creation + shared infra)

*2026-08-07. For Jon: where chunk 1 stands, what we measured, what's in
flight, and the asks. Companion contracts: `ARCHITECTURE.md` (the three-chunk
map) and `UTILIZATION.md` (the pilot contract).*

## Built and verified (all on this branch)

- **`ptcg/creation/`** — validator (hard legality + coherence), fast harness
  (~3 ms/game, seat-swapped, per-seat deck injection), island GA, and the
  **archipelago v2**: one-way DAG migration (mono sources → dual mixers →
  tri terminus), purity floors, explore/refine island temperaments, phased
  compute with plateau exits. `scripts/run_archipelago.py`.
- **Specialist opponent panel** — six community agents harvested from public
  notebooks (gitignored `external/`, license unverified, so never committed):
  Grimmsnarl 96%, Lucario 96%, Archaludon 95%, Alakazam 92%, Garchomp 84%,
  Kangaskhan 78% own-deck win rate vs your rules agent. The fitness panel
  plays each top-8 field list with its own specialist where matched — 89.8%
  of field weight. `ptcg/creation/specialist_panel.py`.
- **Goldfish speed profiler** (`ptcg/creation/goldfish.py`) — median
  first-damage turn + per-turn damage curve per deck. First curves:
  Garchomp opens t2 and fades; Grimmsnarl opens t4 with the biggest t5
  total. These curves (with quantile confidence bands) become the opponent
  threat model later.
- **Matchup matrix** (`scripts/matchup_matrix.py`) — top-8 field decks,
  specialist-piloted, 36 cells × 500 games; running now on a second machine,
  results land as `runs/matrix/matrix.{json,txt}`.

## Measured facts your side should know

- **Deck fitness under a weak or wrong panel is fiction.** Our first 8-hour
  GA run against template decks bred elites that score 0.40–0.54 against the
  real field, while the tutorial starter deck scores 0.784. Your Step-2/Step-6
  README lesson, independently reproduced. Everything now evaluates against
  the mined field, specialist-piloted.
- **Specialists don't generalize.** The strongest public agent wins 96-4 on
  its own deck and loses 27-73 piloting a foreign one — under the harness,
  vs the baseline greedy pilot. There is no public general pilot; the one we
  make is the moat.
- **Two grader-environment bugs found in `agent/main.py`** by running the
  real `kaggle_environments` validation locally: (1) the lazy `cg` imports
  fail under Kaggle's exec model, so **search has likely never fired on the
  ladder** — the deployed agent plays rules-only (fix: one guarded top-level
  import); (2) kaggle_environments calls the *last* callable in the file,
  which is `_agent`, so the never-crash wrapper is dead code (fix: define the
  wrapper last). Fixes are in flight on this branch with before/after
  measurements; worth a resubmission as soon as they land.
- Submission pipeline verified end-to-end locally: validation episode passes,
  bundle audited (nothing licensed/unverified inside), one-command submit
  ready (`scripts/build_submission.py --submit`), ~4 orders of magnitude of
  headroom under the 10-minute clock.

## In flight today

- 8-hour archipelago run vs the specialist panel (Austin's laptop).
- Five deck-agnostic search upgrades being ported into `agent/main.py` from
  a dissection of the strongest public agent: override hysteresis with fair
  sampling, deadline-everywhere budgeting, two evaluator guards, 2-ply
  opponent minimax, derived card-valuation primitives — plus the two grader
  fixes above and an injectable eval weight vector (your UTILIZATION.md
  item 3). Each measured head-to-head before it stays.
- Matchup matrix (~1 h to completion).

## Asks (small)

1. Add an OSI license file (MIT/Apache-2.0) to the public repo — Kaggle's
   public-sharing rule requires it, and the repo currently has none.
2. Read `UTILIZATION.md` — four requests, the weight vector being the one
   that unblocks weight tuning.
3. Confirm the team roster is mirrored on the Strategy-track competition
   (rosters must be identical across divisions).
4. Sanity-check the two grader bugs against your local knowledge — if your
   v9's rating was earned rules-only, the fixed search agent should be worth
   a fresh submission immediately.
