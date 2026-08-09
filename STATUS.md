**Last updated: 2026-08-09 ~06:00 UTC (Aug 9 morning)**

## Update — Aug 9: **our gate harness was scoring legal moves as crashes**

Jon, read this one first — it invalidates numbers, possibly yours too.

`ptcg/arena.py` treated ANY empty selection as a forfeit. But `[]` is the
engine's own answer whenever a prompt's `minCount` is 0 — the engine's
`SelectData` documents it, and the competition's reference agent returns it.
**All 7,726 "opponent crashes" across 97 of our result files were legal
moves**, typically a specialist saying "done benching" on turn 0 and being
handed a loss. Our agent cannot emit `[]` by construction, so the penalty
fell only on opponents: a 24.9% "opponent forfeit rate" against our 0.0%,
entirely self-inflicted. Fixed.

What we had actually been reading: **archaludon 0.63 where the truth is
0.32**; grimmsnarl 0.85 where it is 0.758. Every "we win 56% of the panel"
claim should read ~44% — against a ladder truth of 0.45. If your gates
score an empty selection as a loss, you have the same inflation.

Two related measurement findings, both costly:
- **300-game cells are noise at our effect sizes**, in both directions. Four
  candidate/cell readings flipped sign at 600 games. We now require 600
  clean games minimum on a named seed block against a baseline pinned to a
  commit, never to the working tree.
- **A specialist panel is not the ladder.** v7 gated +13 pt and delivered
  nothing, because the mechanic it fixed (damage walls) appears in 3 of 80
  real ladder decks and none of our panel decks. We rebuilt the panel from
  80 real ladder decklists recovered from replay frame 0 — worth doing on
  your side too, the lists are public in every replay.

## Update — Aug 9: v8 up — we were throwing away our own free abilities

**v8 submitted** (commit 18e8652): the search may no longer override a
cost-free Ability on one of our own Pokemon. A free Ability expires with the
turn and using one never ends the turn, so ending a turn with one unused is
never right. The rule policy already ranked abilities first; the search
overrode it. On our shipped list that discarded **one in four legal Teal
Dances, 2.1 a game, in every ladder game we have played** — most often for a
manual Energy attach that consumed the very card Teal Dance would have
attached for free, and cost the draw as well.

Gated +5.69 pt pooled clean against a frozen v7 (z=+5.84, 5,252 paired
games, every cell positive, self-mirror 0.5320), and confirmed at +2.70 pt
on the real-ladder-deck panel (z=+2.86). Teal Dance use 75.7% -> 91.9%.
**If your agent lets a search override a free Ability, check this first.**

Counting note that cost us a week: ability options carry no `cardId`, only
`(area, index)`. Keying turn-boards by cardId collapses four Ogerpon into
one row, and Teal Dance is once PER POKEMON per turn — cardId keying hid
three quarters of the problem and read 11% where the truth was 27%.

Three fixes did NOT ship, which may be the more useful half: a Stadium
damage fix (correct, but flat — pricing damage right does not help when no
better action exists), evolution-legality fixes (the measured gain sat in a
term that cannot be a legality correction), and damage-KB corrections whose
value is concentrated in decks we do not play.

## Update — Aug 8 late afternoon: **our 300-game gates were lying to us**

Jon — this is the most useful thing we've learned all week, and it applies
to your gates as much as ours.

We re-ran four candidate/cell combinations at 600 games on a fresh seed
block. **Three of four flipped sign**, always in the same direction, always
understating the candidate:

| cell / arm | 300 games | 600 games |
|---|---|---|
| grimmsnarl / lethal1 | −2.7 pt | **+6.2 pt** (z=+2.79) |
| lucario / lethal2 | −4.7 pt | **+12.5 pt** (z=+5.24) |
| lucario / lethal1 | −1.7 pt | **+12.9 pt** (z=+5.38) |
| self-mirror / lethal2 | 0.4733 | **0.6733** (z=+8.49) |

The consequence: v7 is not a +2.4 pt change as our 300-game gate reported.
Against the frozen v6 arm at 600 games a cell it is **+13 points pooled**
(0.4111 → 0.5411 over 1800 games), and the Dud-Alakazam cell — our worst
matchup, the archetype the 22-day leader plays — goes 0.2817 → 0.4783. The
self-mirror says v7 beats v6 about 2:1.

**We've retired 300-game cells as a decision instrument.** They resolve only
effects above roughly 8-10 points. Ship and kill decisions now need 600
games minimum on a named seed block against a baseline pinned to a commit,
never to the working tree. If you have candidates you killed on ~300-game
evidence, they deserve a rerun — we're re-testing four of ours, including a
neural evaluator that "failed" by 0.7 points against a 3-point bar.

v7 was also re-staged as a result (commit 7656b69): the arm that teaches the
evaluator about damage walls, not just the damage function. The two arms are
indistinguishable on the panel, but only this one stops the search from
*valuing* punches into an immune defender — it cuts wasted attack turns 27%
against the Cornerstone list where the other managed 8%.

## Update — Aug 8 afternoon: v6 crashed back to 782; v7 staged on a real bug

- **The 1000 did not hold.** v6 peaked at 1004.2 on episode 8 and then went
  9W-21L down to 782 over the next 31 episodes. Its twin never launched
  (652). Same shape as v3's 850→730. Treat single-submission peaks as
  variance, not strength, when you pick your final pair.
- **Autopsy of all 39 episodes.** The opponent mix matched our panel almost
  exactly, so this was not an unseen-archetype problem. The real split:
  **8-4 vs Grimmsnarl, 2-17 vs everything else** — our panel-weighted
  fitness was being carried by the single 34.6% Grimmsnarl cell. 14 of 20
  losses came from AHEAD in the prize race. With an all-ex board every
  Pokemon we concede pays 2 prizes while our KOs average 1.37, so seven
  losses had even KO counts and a prize deficit anyway.
- **The bug worth your attention: wall blindness.** The autopsy first flagged
  "payable lethal never taken" in one loss. That was a false positive, and
  chasing it found something better. We *did* attack — into a Cornerstone
  Mask Ogerpon ex, whose ability prevents all damage from attackers that
  have an Ability. Every Pokemon in our deck has one. The engine dealt 0
  four times while both the detector and our own `_damage_against` priced
  the attack at 780-900 into a 240 HP target. Three of 39 episodes met a
  damage wall. The pool has a family of these: Crustle and Sylveon wall ex
  (our whole deck), Milotic ex walls Tera, Farigiraf ex walls Basic ex,
  Drednaw walls hits of 200+. **If your agent prices printed damage without
  a prevention model, it has this bug too.**
- **v7 staged** (commit cf12850): wall conditions parsed from the engine's
  own runtime skill text, plus a hard rule that takes a genuinely
  game-winning lethal ahead of the search. Gated at 300 games a cell,
  paired: pooled 0.5833 vs 0.5592, worst cell -0.98 SE, self-mirror 0.5467,
  and +11.0 pt on the Dud-Alakazam cell (z=+2.76) — our worst matchup and
  the archetype the 22-day leader plays. Registered honestly: no panel cell
  holds a wall card, so those gates price the change's cost, not its
  benefit. A wall-cell test against the real ladder decks that beat us is
  running.
- **A negative result worth your time.** We spent five experiments raising
  our agreement with strong players' decisions. The evolve-timing one moved
  its bucket from 21% to 69% agreement (z=+21) and did not move win rate at
  all (0.576 vs 0.591 over 1200 paired games). Neither did three others. The
  single conversion was a near context-free rule (always secure the kill on
  the lowest-HP target). Copying good players' choices pays only where the
  choice does not depend on their plan — we've stopped using agreement as a
  candidate generator and switched to autopsying our own ladder losses,
  which produced the wall bug directly.

## Update — Aug 8 midday: **v6 crossed 1000 (1004.2, top-100)**

- Eight episodes from submission to 1000: 7W-2L with the last three wins
  against 907/949/924-rated opponents — the band that used to beat us.
  The effect-target fix converts. Judge final convergence at ~30 episodes.

## Update — Aug 8 morning (v6 up: pilot fix + the proven list)

- **v6 SUBMITTED**: the overnight-gated effect-target fix (on ctx-13/14/15
  damage-counter prompts, secure the kill on the lowest-HP opposing target,
  Munkidori prioritized — +4.7pt driving Grimmsnarl over 900 games, 5-cell
  no-regression clean) on the ORIGINAL Ogerpon list. Ladder verdict from
  the overnight pair: both GA deck variants converged below the original
  list (629/692 vs 863), so deck experiments yield to pilot gains for now.
  Scored pair = v5 + v6; 3 team slots remain today.
- D46 GA finished both machines (90 + 30 eras, 37-cell real-ladder panel):
  grimmsnarl-mirror elite 0.64 raw is our best deck but unsubmittable (our
  pilot can't drive the ability engine — measured 0/30, fix queue open);
  the low-energy ENGINE chain hit 0.50 and was still improving at era 65.
  v6-b challenger gate (GA elites under the v6 pilot) is running now.

## Update — Aug 7 late night (v5 up; a pilot gap measured)

- **v5 SUBMITTED** (tonight's last slot): the GA's 0.654 Ogerpon sibling,
  gated 0.407 mean vs v4's 0.367 on the gatekeeper cells. Scored pair is
  now v4 + v5 (champion/challenger; fresh quota in the morning).
- **Measured pilot gap:** our pilot cannot drive ability-engine decks —
  the 0.691 harvest Grimmsnarl AND the field's stock Grimmsnarl netdeck
  both gate at 0.02-0.06 under main.py (0/30 in the mirror) because
  Adrena-Brain/Punk Up never fire. Own-side ability usage is now a
  named pilot workstream; until it lands, Grimmsnarl-class decks are
  panel opponents, not candidates. Large-scale overnight GA (new
  ladder-derived panel, 6 chains incl. a low-energy engine chain) is
  launching tonight on both machines.

## Update — Aug 7 night (GA wound down early; harvest banked)

- The seeded-archipelago v3 runs were **stopped early by decision** (~6 h
  in): the high-value chains (Ogerpon, Darkness) froze at their plateaus
  by hour 6 and the remaining budget was grinding cold chains. Clean
  checkpoint stops on both machines; every chain's best-ever elite is
  banked in `runs/seeded_overnight/HARVEST.md` + `harvest_best.json`
  (decks + full 27-cell matchup profiles). Every chain beat its founder
  baseline; headline: anchored Ogerpon 0.654 uniform panel WR (the v4
  family), Darkness 0.691.
- A redesigned overnight run launches later tonight; the harvest is its
  seed bank. The Archaludon panel ask below stands for that run.

## Update — Aug 7 evening (v4: the GA's first shipped deck)

- **v4 SUBMITTED** (~21:40 UTC; **1 team slot left today**). v3's rating hit
  850 then fell to ~720; replay autopsy of its 18 ladder losses: 5 to one
  Duraludon/Archaludon/Cinderace list — an archetype with zero cells in our
  fitness panel. No episode errors; v3's record (20W-18L) tracks v2's, so
  the crash reads as a panel hole plus Elo convergence, not a pilot bug.
- v4 deck = the seeded GA's Ogerpon-chain elite (raw 0.660 vs founder 0.574
  on the 27-cell panel): 4x Teal Mask / **26 energy** / 30 trainers — the
  scaled-damage pilot prices Myriad Leaf Shower's energy scaling, and the
  GA traded six one-of trainers for six energy. Ship gate (5 loss-archetype
  cells, 30 games each, specialist-piloted): better-or-equal everywhere,
  +10pp on the Archaludon cell (0.30 — improved, still negative),
  bench-out losses 28% vs the old deck's 61%.
- Ask: the Archaludon archetype goes into the fitness panel before the next
  GA selection round; flag if your side has a better list for it.

# Status — Austin's side (deck creation + shared infra)

## Update — Aug 7 PM (v3 shipped: the scaling-attack blindness fix)

- **v3 = v2 + the scaled-damage bundle, gated and SUBMITTED 2026-08-07
  20:50 UTC** (same Ogerpon list, repaired pilot; 2 team submissions left
  today — sequence yours under the latest-2 rule). The threat
  machinery read printed damage only, so Alakazam's Powerful Hand (printed
  0, 20 x own hand size) registered as ZERO threat while Dudunsparce
  engines built 20-card hands — the ladder-autopsy loss mode Austin ruled
  on. `agent/attack_scalers.json` (generator `ptcg/attack_scalers.py`, KB =
  attackId + numbers only) now prices 100 attacks from the observed state:
  hand/energy/bench/damage-counter/prize scalers, flat effect damage
  (Cruel Arrow's printed-0 100), the Gale Thrust from-bench rider, and the
  engine-verified −30 resistance. No new evaluator weights — honest numbers
  feeding the existing screened threat terms. Playbook entry 10 holds the
  full gate: targeted alakazam cell **+3.4 pp** (0.434→0.468, 500/arm,
  paired seeds, replicate +0.6 pp), pooled 8-entry specialist panel
  no-regression (clean −0.81 SE, all-games −0.27 SE, no cell past 2 SE).
  `CABT_SCALED_DAMAGE=0` is the revert.
- Emergent behavior confirmed in trace: Judge into a 20-card Alakazam hand
  now evaluates **+547** on the resulting margin through the ordinary
  1-ply search; the blind build scored it +0.0.
- Limitation on the record: future-turn projections hold scaling
  quantities (hand size, Energy counts) at their observed values — no
  opponent hand-growth model yet.

## Update — Aug 7 PM (seeded archipelago v3 launched, both machines)

- **The GA deck search is LIVE** on both machines: the seeded
  archipelago v3 (`ptcg/creation/seeded.py`, spec + pilot-split table in
  `runs/seeded_overnight/README.md`). Five chains x (explore + refine):
  spec-Ogerpon (anchored refinement of the shipped list), mono-Darkness
  (mirror-weighted Grimmsnarl harvest), mono-Fighting (uniform
  wildcard), rainbow-Kanga (the field's 5-energy toolbox), kanga-counter
  (energy-unconstrained counter chain). Fitness = the 27-cell stratified
  leaderboard panel (`data/panel_lb.json`, manifest
  `data/analysis/PANEL_LB.md`), specialist-piloted; selection decided by
  real/specialist pilots, greedy only screens (D40).
- ~12 h budget each, deep final evals (`final_eval.json`) under real
  pilots at the end. Results + candidate lists tomorrow morning.
- Yesterday's mono-typing sprint findings that shaped this design:
  `data/analysis/MONO_SPRINT.md`.

## Update — Aug 7 (pilot frozen at its ceiling; GA rotation starts tonight)

- **Ladder:** v2 at ~817 (team record, top ~12%), v1 at 686. No new
  submission today — v2 stays the incumbent and the scored-pair member.
  Submission sequencing stays coordinated under the latest-2 rule.
- **The pilot is FROZEN at v2** after a day of systematic refusals, each
  with a diagnosis (all committed here: `b36a6cb`, `570aa37`, `18bd1f7`;
  playbook entries 7–8):
  1. **D34 GBDT tree leaf** — better held-out description (AUC 0.692 vs
     0.661) yet 0.2135 pooled at the play gate. TreeSHAP convicts the
     struck `hand_diff` returning as shape (+0.174 log-odds). Playbook
     entry 7.
  2. **Interaction mining off the dead forest** — 18/18 candidates
     refused. The forest's edge is diffuse (top-5 pairs hold 18% of
     pairwise mass), and expected-incoming-damage anti-prices (wrong
     sign). `data/analysis/INTERACTION_MINING.md`.
  3. **Phase-conditional weights** — the coefficient drift is real
     (bench 202 early vs 12 late) but the gate refused 0.4763/590: a
     state-dependent evaluator hands one search two rulers at the phase
     boundary. Playbook entry 8.

  Net: the 1-ply linear leaf on this feature universe is at its measured
  ceiling — three independent axes now say so with numbers.
- **Tonight: GA rotation begins per D30** — deck STABLE (~100 tuned decks,
  decorrelated matchup weaknesses, per-deck fitted weights). Prep is in
  flight now; overnight runs on both machines.
- **One live screen:** an evolution-aware threat ladder. The shipped
  `_attack_profile` never sees a benched basic's evolved attacks — a
  verified coverage hole. Offline screen only for now.
- **Standing asks (unchanged, still open):**
  1. Post the forum question **in writing** on per-episode deck variation
     — it gates the stable's runtime use.
  2. Mirror the team roster on the Strategy-track competition before
     Sep 6.
  3. Add an OSI license file to the public repo.

---

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
