# Status — Austin's side (deck creation + shared infra)

> ## ⚡ ACTION FOR JON: accept PR #1
> **https://github.com/lonespear/ptcg/pull/1** — your `main` (through your
> import-fix `c134f1b` and `FEEDBACK_FOR_AUSTIN.md`) fully merged with our
> stack, gates passed, and v1 already submitted from it (#55306027, the first
> submission where search provably fires on the grader). Point-by-point
> responses to your feedback:
>
> 1. **Bug #2 evidence you asked for:** `kaggle_environments.agent.
>    get_last_callable` returns `[v for v in env.values() if callable(v)][-1]`
>    — the *last* callable in the exec'd file, which was `_agent` (defined
>    after the `agent` wrapper), so the never-crash guard never ran. Observed
>    empirically by asserting the bound callable's identity inside a real cabt
>    episode; the falsification gate in this branch asserts `agent` is bound
>    on every bundle build. Reorder was the fix; it's in the merged file.
> 2. **`collaborate_and_merge.md` exists now** — repo root of this branch.
>    You looked during the ~30-minute window between your fetch and our merge
>    push. It carries the full merge report you wanted, updated to describe
>    what actually happened (including the `obs["step"]` seed defect we found
>    in your position-seeded shuffle — dealt every determinization the same
>    world locally; fixed, A/B'd 44.5%→50.5%).
> 3. **Your import fix `c134f1b` won the reconciliation** — same bug, found
>    independently on both sides; your sys.path loop covers cwd-mismatch
>    cases ours didn't. Your loud-failure suggestion (stderr print when
>    `CG_AVAILABLE` is false) is adopted in spirit: gate-tested every build.
> 4. **Termination-mode fitness: adopted.** The GA's next runs score
>    candidates on win rate AND termination-mode distribution vs real
>    replays, so a deck can't hide a 70% deck-out failure inside a win rate.
>    Your distributional-validation point about self-play blind spots goes
>    into our standing evidence rail.
> 5. **SPRT condition on weight tuning: flagged to Austin as an open call**
>    — our internal gates ratified fixed 500-game samples (games cost ~3 ms
>    on our harness), but anything touching `agent/main.py` is your file and
>    your `field_sprt.py` bar is reasonable for it. Your ~19%-informative-
>    pairs number argues for paired designs either way. Expect Austin's call
>    in this file's next update.
> 6. **Per-deck weight fitting:** agreed and already the architecture — the
>    weight vector is deck-conditioned (plan-module deltas, decisions D10);
>    your energy-term transferability warning is exactly why.

*2026-08-06. For Jon: where chunk 1 stands, what we measured, what's in
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

## Answers to two things your README queues

- **The 2-ply retest is already run, on the shuffled hand model.** Playing the
  opponent's turn out before scoring still costs 9 points of win rate (40.7%
  over 200 games against the same agent at 1 ply), and a single reply with no
  minimum taken over it costs 14 (36.2%). Since one reply is the *worse* of the
  two, the fault is not branching or pessimism, and it is not the determinized
  hand either — it is the opponent **policy** model, which rolls their turn out
  under our own priority table on a guessed decklist. Diagnostics in commit
  `1e1867c`; the mechanism stays live behind `SEARCH_OPP_BRANCH`, defaulted to
  0, ready to re-enable once an opponent policy model exists.
- **Termination-mode distributions are already an output on our side.** Every
  `GameResult` in `ptcg/creation/harness.py` records its end reason (prizes
  taken, deck-out, no active Pokemon, card effect), so the harness and the
  36-cell matchup matrix can report the distribution per cell for any deck we
  generate. Say the word and your 70%-no-Pokemon finding gets checked against
  the whole field, deck by deck.

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
