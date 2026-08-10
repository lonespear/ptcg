# GLIDEPATH — read this first

State of the PTCG agent as of **2026-08-09**, written as a handoff. Pairs with
`CLAUDE.md` (operating rules) and `EDA_FINDINGS.md` (numbered evidence log).

---

## 1. Where things stand

Kaggle **PTCG AI Battle Challenge**, two linked competitions:

- **Simulation** — an agent bundle (`main.py` + `deck.csv` + `cg/`). Has the
  leaderboard. Five submissions/day, resetting 00:00 UTC, **shared across the
  team**.
- **Strategy** — a ≤2000-word writeup, 70% model / 20% deck / 10% report.
  **Not submitted, and not to be submitted without an explicit ask.**

Deadline **13 Sep 2026**. Team `Lemmes Yad` = jonallday + austinsemmel
(+ a contributor committing as Sebastian).

Last confirmed standing, **2026-08-09 ~17:00 UTC**: **rank 778 of 6,679, top
11.6%**, score **822.0** — Austin's **v9**, submitted 16:38 UTC. From 4,567th
four days earlier. Leader ~1,180; two teams above 1,200; median ~631.
**Verify before quoting — see §2d.**

Live submission history is the clearest illustration of §2d. These are all the
same team, days apart, and the *gated* improvements do not order the scores:

| Submission | Local verdict | Live score |
|---|---|---|
| **v9** — engine-verified rules fixes | no gain claimed | **822.0** |
| v8 — free abilities never overridden | **+5.69 pt**, z=+5.84, 5,252 games | 617.7 |
| v7 — wall-aware evaluator | +13 pt pooled at 600 g/cell | 577.3 |
| v6 | — | 771.5 |
| v6-twin — **byte-identical to v6** | — | 655.0 |

v8 carried the strongest local evidence in the project and scored 154 points
*below* v6. v6 and its identical twin differ by 116. **The ladder cannot resolve
a single change**; it can only confirm a direction already established locally.

Deck: **Teal Mask Ogerpon ex**, mono-Grass. Chosen because ~47% of the *global*
field is weak to Grass. See §4 for why that denominator is now suspect.

---

## 2. Read this before trusting any number in this repo

Three measurement faults were found on 2026-08-08/09, two of them in code we
own. **They invalidate a large fraction of this project's recorded results**,
including several conclusions still written confidently elsewhere.

### 2a. The arena scored legal moves as forfeits *(our bug — now fixed)*

`ptcg/arena.py` treated any empty selection as a crash. But `[]` is the engine's
own answer whenever a prompt's `minCount` is 0, and the reference agent returns
it. Our agent never emits `[]` by construction, so the penalty fell **only on
opponents** — Austin measured a **24.9% opponent forfeit rate against our 0.0%**
across 7,726 "crashes" in 97 result files.

Every gauntlet, field-SPRT and matchup number this harness produced is inflated.
Known corrections: **Archaludon 0.63 → 0.32**, Grimmsnarl 0.85 → 0.758, "we win
56% of the panel" → ~44% against a ladder truth of 0.45.

That includes the Archaludon result in this repo's history (**71–9, 0.887**),
which is simply wrong.

### 2b. The engine was never seeded *(engine limitation — Austin fixed)*

`ptcg/arena.py` seeds Python's `random`, but `libcg` draws from
`std::random_device` and exposes no seed. **Every "seeded" run in this
repository's history played different games.** So the paired-seed premise behind
`scripts/field_sprt.py` — both candidates on identical deals — was false, and
its variance reduction never existed.

Austin's fix is `tools/engine_seed` + `ptcg/engine_seed.py` +
`scripts/run_pinned.sh` (a preload replacing the entropy source). Nine runs of
one matchup now return an identical 479–121/600. **Trap:**
`run_pinned.sh env VAR=x python …` runs *unpinned*, because macOS strips
`DYLD_*` for SIP-protected binaries and `/usr/bin/env` is one.

Also: `cg.api` holds one engine-side search handle per process forever, so games
searched through the previous game's leftover state, and which games shared a
process depended on `--workers`.

### 2c. Sample sizes here were too small

Re-running four candidate/cell combinations at 600 games flipped **three of four
sign**, always understating the candidate (e.g. lucario/lethal1 −1.7 pt → +12.9
pt, z=+5.38). Austin's standard is now **600 clean games minimum on a named seed
block, against a baseline pinned to a commit, never to the working tree.**

Most SPRT verdicts recorded in `README.md` were decided on 82–511 games, on an
unseeded engine, with opponent forfeits inflating them. **Treat the whole
decision ledger as provisional.**

### 2d. The ladder noise floor — a number, not a vibe

Every observed pair of identical or behaviourally-identical submissions:

| pair | spread |
|---|---|
| v6 / v6-twin (byte-identical), reading 1 | 93.5 |
| v6 / v6-twin, reading 2 | 116.5 |
| v6 / v6-twin, reading 3 | 142.6 |
| "v3: scaled-damage KB" submitted twice | 150.2 |

**Noise floor: ~125 points, observed range 93–150.** Same agent, same deck,
different launch luck.

**The ship rule, stated numerically:** a single submission cannot resolve a
difference below **~150 rating points**, so nothing ships on ladder evidence
alone. Ship on *local* evidence from the audited harness, then read the ladder
expecting ±150 either way. And judge a submission at roughly 30 settled
episodes, never at its peak — v6 read 1004.2 at 8 episodes and crashed to 782.

Every future settle adds another anchor; keep this table updated, because five
weeks leaves only a handful of informative settles and each should be spent
against a known noise budget.

---

## 3. What is actually solid

Things that survive the above, because they were measured against ground truth
or by direct inspection rather than by playing games:

- **The grader `exec()`s `main.py`.** `__file__` is undefined (cost 2
  submissions). `cg` is not importable from a function body, so engine imports
  must be **module level** — a lazy import silently disabled search on every
  submission for a day.
- **`agent()` must be the LAST callable in the file.** `kaggle_environments`
  resolves the entry point as `[v for v in env.values() if callable(v)][-1]`,
  and rebinding an existing name does not move it. One helper defined after
  `agent()` makes that helper the submission. Guarded in
  `scripts/validate_submission.py`.
- **Opponent-deck inference, graded on 300 replays with both real decklists
  known**: 99.7% coverage, top-1 right 0.63 (turn 1) → 0.84 (turn 3) → 0.94
  (turn 10), well calibrated. When wrong, it shares a median 53/60 cards and is
  the same archetype 80.6% of the time.
- **The metagame is small**: ~120–200 distinct decklists; 40 lists cover 94.4%
  of observed play in 9 KB.
- **Austin's engine-arithmetic fixes (v9)**, each verified against live engine
  output rather than a win rate: Nighttime Mine affordability 150/838 wrong → 0;
  end-of-turn damage counters 2001/15451 → 0; Rare Candy stage skips 87/87 → 0;
  Weakness was being applied to attacks that *place* damage counters, which the
  engine does not treat as dealing damage.
- **Free abilities must never be overridden by search** (Austin's v8). Ending a
  turn with a cost-free Ability unused is never right. Our search was discarding
  one in four legal Teal Dances, 2.1 per game, in every ladder game played.
  Gated +5.69 pt pooled over 5,252 paired games.

---

## 4. The strategic question that is open

Our deck is a best response to the **global** field. It may be the wrong
denominator.

| Source | Archaludon share |
|---|---|
| Our 16-day mining (146,376 replays) | **0.31%** |
| Games *we actually play* (Austin's ladder autopsy) | **14%** |

A 46× gap: mining samples all 6,563 teams, dominated by the copy-paste middle,
while matchmaking pairs us by rating. The same effect shows in the opponent
model, whose top-1 accuracy drops 0.715 → 0.609 against strong agents.

**And Archaludon runs 4× Cinderace — a Fire attacker. Teal Mask Ogerpon ex is
weak to Fire.** They are teching against our plan specifically. The README's
old claim that "nothing in the field plays the Fire we are weak to" was true
globally and false at our rating band.

Austin's related result: **a duel ranks decks correctly and chooses between them
wrongly** — a duel between two Ogerpon decks *is* the mirror (3.5% of the
ladder), while the four gate matchups are 85.6%. Our list loses duels (8th of 9)
but every duel winner fails the field-weighted gate. He also refuted "cheaper
boards help" over 157 candidates: prizes-conceded-per-body vs win rate
correlates **+0.805** — decks conceding fewer prizes per body win *less*.

So: **re-weight the field from the ladder autopsy, not from global mining.** The
gauntlet and the field SPRT harness both need to consume ladder-derived shares.
That is the highest-value unclaimed piece of work.

---

## 5. Repository state

`lonespear/ptcg` is **public** — the engine, `cg/`, card CSVs/PDFs and
`agent/deck.csv` are competition-use-only and gitignored. `data/` and
`agent/deck_priors.json` are aggregate statistics only; keep that distinction.

**Branches have diverged.** Austin's `defense031/ptcg deck-creation` is **64
commits ahead** of our `main`; we hold **12** he lacks.

| Ours only | His only (highlights) |
|---|---|
| `CLAUDE.md` | `PLAYBOOK.md`, `data/analysis/LADDER_AUTOPSY*.md`, `PRIOR_ART.md`, `POSTURES.md`, `INTERACTION_MINING.md` |
| `scripts/probe_grader.py` | fitted models: `tree_leaf.json`, `trajectory_curves.json`, `calibration_v2.json`, `attack_scalers.json`, phase weights |
| `scripts/who_runs_out.py` | ~30 scripts: `fit_*`, `leaf_*`, `run_pinned.sh`, `select_stable.py`, `validate_invariant.py`, `reproduce_check.py` |
| `scripts/rebuild_deck.py` | `tools/engine_seed/` (the seeding fix) |

His `agent/main.py` is **3,371 lines vs our 1,041** and is strictly ahead —
fitted weights, phase-conditional vectors, scaled damage, the v8/v9 fixes.
**Do not submit from our `main`; it would push an older agent over the team's
best.** A worktree at his branch exists at `../ptcg-austin`.

**Merge direction: take his agent, keep our guards.** Our `probe_grader.py`,
the entry-point check in `validate_submission.py`, and the `arena.py` fix in §2a
are things his branch does not have.

---

## 5b. The engine was always seedable — a flag turns it off

Reading `ptcg_engine/ptcgProgram 22/` rather than working around it:

```cpp
// CardMove.h:263
if (state.game->config.deviceRand) {
    std::shuffle(ps.deck.begin(), ps.deck.end(), std::random_device());  // unseedable
} else {
    std::shuffle(ps.deck.begin(), ps.deck.end(), state.game->rng);       // seeded
}
```

`GameConfig` carries a `seed`, the game carries a seeded `mt19937`, and Kaggle's
own episode configs contain a `seed` field. **`ApiBattleStart` (Api.h:33) simply
hardcodes `config.deviceRand = true`**, and then overwrites the seeded generator
with `std::seed_seq{rd(), rd(), rd(), rd()}` at Api.h:77. `ApiAgentStart` — the
*search* path — does neither, so search determinizations were already seeded.

So determinism needs three edits to a local-gating build, not a preload:

1. `config.deviceRand = false`
2. `config.seed = <caller-supplied>` instead of `rd()`
3. `data->game.rng = std::mt19937(config.seed)` instead of the `seed_seq`
4. plus `EffectInstant.h:585`, a second `std::random_device()` target-list shuffle

**Why this matters beyond tidiness:** Austin's `tools/engine_seed` preload is
macOS/Linux only — it needs `DYLD_INSERT_LIBRARIES`/`LD_PRELOAD`, which Windows
does not have. Jon's machine is Windows, so **he currently cannot reproduce any
pinned run**. Rebuilding the engine from the shipped source fixes it on every
platform and needs no interception.

Toolchain is present on Jon's box but off PATH:

```
vcvars64: C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat
cl.exe  : ...\MSVC\14.29.30133\bin\Hostx64\x64\cl.exe
```

Licence note: building locally is squarely "use it to build and test your
competition entries". The build stays gitignored like the rest of the engine,
and **nothing about this ships** — the submission uses Kaggle's own `cg`.

## 5c. The field we actually play — measured, and it is not the mined one

`scripts/build_autopsy_pool.py` rebuilds the opponent pool from
`data/analysis/ladder_autopsy.json`, which records the real opposing decklist
and both ratings for every episode we have played. Filtered to our band
(700–900), 123 episodes, 8 lists covering 63% of them:

| Opponent | Mined share | Actually faced | Error | **Our real win rate** |
|---|---|---|---|---|
| Grimmsnarl/Munkidori | 45.7% | 41.6% | 0.9× | **0.750** |
| Dudunsparce-Alakazam | 0.96% | 15.6% | **16×** | 0.500 |
| Archaludon-Cinderace | 0.29% | 13.0% | **45×** | **0.300** |
| Archaludon-Cinderace (2nd list) | 0.29% | 9.1% | **31×** | **0.286** |
| Mega Lucario | 1.20% | 7.8% | 6.5× | **0.000** (0 of 6) |
| Dudunsparce-Alakazam (2nd) | 0.96% | 6.5% | 6.7× | 0.400 |

**The mined field predicted exactly one opponent** — Grimmsnarl, which is also
the one our deck was built to beat, and it does beat it at 0.750 on 42% of
games. Everything else is wrong by 6× to 45×.

Encounter-weighted, that pool comes to roughly **0.52** — about even, which is
what our rating says.

**Where the rating is actually going:** Archaludon (22% of games across two
lists, ~0.29) and Mega Lucario (7.8%, 0-for-6). Together **~30% of our games at
about 0.2**. That is the entire gap, and neither was in the design.

Mechanically both are the same failure. Mega Lucario ex's Mega Brave does **270
for two Energy**; Teal Mask Ogerpon ex has **210 HP** and concedes **two
prizes**. We are one-shot by a two-Energy attack while giving up double. The
Archaludon lists run Cinderace, a **Fire** attacker, and Ogerpon is weak to
Fire — the same story with the doubling done by Weakness rather than raw damage.

So the thesis is half-right and precisely diagnosable: the Grass plan works
exactly as designed against the deck it targeted, and loses to the two decks
that one-shot a 210 HP two-prize attacker. **Rebuild against this pool, not
against the mined field**, and treat "survives a 270 hit or does not concede
two prizes" as a design constraint rather than a nicety.

## 5d. Search audit — it is not dead weight, but it is small

Measured under `kaggle_environments`' own `cabt` environment
(`scripts/probe_grader.py`), five episodes:

| | run 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| search returned an answer | 24% | 25% | 20% | 39% | 33% |
| …and changed the move | **100%** | 100% | 100% | 100% | 100% |
| net share of decisions changed | 10% | 10% | 6% | 15% | 11% |
| latency median / max (ms) | 13/163 | 17/158 | 9/143 | 17/164 | 17/137 |

Neither of the two failure modes we were worried about. Search is **not**
silently falling back everywhere, and it is **not** running-but-agreeing:

- It declines ~70% of the time **by design** — the confidence gate and the
  `search_margin` hysteresis mean it only overrides when it can show the margin
  against a fairly-sampled alternative.
- When it does return, it disagrees with the rule policy **every single time**.
- Net: it changes roughly **10% of all decisions**, each one deliberately.
- Latency is a non-issue: max 164 ms against a 600 s episode budget.

**So the fork resolves toward keeping search**, but the honest read is that it
is a 10%-of-decisions lever, not a transformation — which is consistent with the
production signal never separating it from noise. Whether those 10% are *good*
decisions remains unresolved, and the ladder cannot answer it at ±150.

The cheap next step is not more search work: it is to log which decision
contexts those 10% fall in, and check a sample against the belief audit Austin
built (`scripts/belief_audit.py`), which compares what the agent believed to
what the engine did. That answers "are the overrides right?" without a single
submission.

## 5e. The rebuild tournament ran, and it is not usable. Here is the check.

Fixing the *field* did not fix the *pilot*. Same eight opponents, same decks,
`current` list, local tournament against ladder truth:

| Opponent | local | ladder | gap | share |
|---|---|---|---|---|
| Grimmsnarl/Munkidori | 1.000 | 0.750 | +0.250 | 41.6% |
| Dudunsparce-Alakazam | 0.840 | 0.500 | +0.340 | 15.6% |
| Archaludon-Cinderace | 0.860 | **0.300** | **+0.560** | 13.0% |
| Archaludon-Cinderace (2) | 0.860 | **0.286** | **+0.574** | 9.1% |
| **Mega Lucario** | 0.780 | **0.000** | **+0.780** | 7.8% |
| **WEIGHTED** | **0.892** | **0.519** | **+0.372** | |

**The harness reports 0.892 where the ladder says 0.519.** And the error is not
uniform — it is +0.25 on the deck our agent knows how to pilot and **+0.78** on
Mega Lucario, which it cannot pilot at all. The bias scales with how much skill
the *opponent's* deck needs, so it is largest exactly on the matchups that are
costing us rating.

This is the symmetric-pilot bias, unchanged: the autopsy pool fixed *which*
decks appear, but both sides are still driven by our own agent. Austin measured
the same thing independently — his real-ladder-deck panel "says we win 74% where
the ladder says 45%".

**Consequence: no deck decision can be made on this harness yet.** The variant
scores it produced (`current` 0.892 against 0.21–0.32 for every single-prize
build) cannot separate "the prize-liability idea is wrong" from "our agent
cannot pilot the variants". Both arms carry the bias, so the comparison is
internally fair, but it is fair about the wrong quantity.

What the run does establish, because it replicates an earlier independent
result: **diluting Ogerpon is catastrophic** — 0.892 → 0.205–0.322 across four
builds, matching the earlier plus4/plus6/plus8 collapse. The Trainers are the
engine. So the prize-liability hypothesis is **untested, not refuted**: the
right test is a list that lowers prize liability *without* cutting the draw
engine, and an opponent driven by something other than us.

**The blocking dependency is an opponent pilot that is not our agent.** Options,
cheapest first: drive pool decks with the competition's reference agent; use
Austin's specialist panel; or replay real ladder episodes and score our agent's
divergence from what the human-tuned opponent actually did. Until one exists,
deck selection has to lean on ladder autopsies, which are slow and cost
submissions.

## 5f. Pilot-agreement does NOT explain the gap. Line closed.

The idea: measure how well our agent pilots each opponent deck by comparing its
choices to what real players did, and use that as a trust score for local
matchups. If we pilot a deck badly, our local number for that matchup is
inflated.

Measured against real replays:

| Deck | agreement | decisions | local-minus-ladder gap |
|---|---|---|---|
| Grimmsnarl/Munkidori | 0.330 | 17,577 | +0.250 |
| Dudunsparce-Alakazam | 0.266 | 12,690 | +0.340 |
| Archaludon-Cinderace | 0.287 | **80** | +0.560 |
| other (Cynthia's Roselia) | 0.318 | 538 | −0.140 |
| **Mega Lucario** | — | **0 found** | +0.780 |

**No usable relationship.** Agreement sits in a narrow 0.27–0.33 band for every
deck, and its ordering does not track the gap: the deck we pilot *best*
(Grimmsnarl, 0.330) has a +0.25 gap, while the one we pilot similarly
(Roselia, 0.318) has a *negative* gap. Archaludon's +0.56 rests on 80 decisions
from 2 player-slots and cannot carry weight either way.

**The solid finding is narrow and worth stating precisely: whatever drives the
gap, it is not divergence _rate_.** That rests on the two well-measured rows
(17,577 and 12,690 decisions).

**The cost story is an inference, not a measurement.** The proposal — that decks
differ in how *punishing* a mistake is, so one misplay into a one-shot deck is
terminal where one into Grimmsnarl is a tempo loss — is mechanistically
plausible and it is the best available explanation. But it is inferred from the
failure of the alternative, and the two rows that motivate it (+0.560 on 80
decisions, +0.780 on zero) are the two worst-measured in the table. Do not build
on it as though it were established.

**Cost is directly measurable with tools already here, and needs no new data.**
For a decision, roll out both our choice and the alternative to terminal and
diff the win probability — that *is* the cost of the divergence. A few hundred
decisions per matchup gives a cost-weighted disagreement number, which is the
quantity the agreement rate was a bad proxy for. If the story holds,
cost-per-divergence against Lucario should dwarf Grimmsnarl at identical
agreement rates. **If it does not, there is a third explanation nobody has
named** — grader-environment differences being the next suspect, given the v8
twin history.

**But do not spend the hours there yet.** The cost story points at the same
place the autopsy already did: if error *tolerance* is the fragility, the fix is
the deck rebuild that is already the top open item. Reserve the rollout
measurement for validating the winning variant — "does this list actually reduce
cost-per-error against one-shot decks?" — rather than running it as a research
direction of its own.

**And the data does not exist to fix it.** Mega Lucario appears **zero** times
in 4,669 episodes, because these decks are rare in the general population
(Archaludon 0.31%) and common only in *our* games — and our own episodes are not
recoverable: **zero of the 187 autopsy episode IDs appear in any public dump.**
The daily datasets are samples, not exhaustive.

**Unblocking deck evaluation therefore needs one of:**

1. **Our own episodes — and these are retrievable without Austin.** The Kaggle
   episode API serves any episode by ID, and the submission page lists our
   games. A small script pulling the last N ladder episodes after each settle
   gives exactly the corpus this instrument needed **and** the standing
   pool-refresh that §5c already wanted as a post-settle step. Build it once;
   it permanently unblocks both this analysis and the autopsy re-weighting.
   **This is the highest-leverage tooling left.**
2. **A non-us opponent driver** — the competition reference agent, or Austin's
   specialist panel, accepting that each is itself biased.
3. **Accept that matchup truth costs submissions**, and spend them deliberately.

Until then, treat every local matchup number as an upper bound, most inflated
exactly where the opponent's deck punishes error hardest.

## 6. If you do one thing next

Austin's branch is **merged** (§5) and the field is **re-weighted** (§5c), so
the original next-steps are done. What remains, in order:

1. **Build the episode puller** (§5f option 1). The Kaggle episode API serves
   any episode by ID and our submission page lists our games. This is the one
   piece of tooling that unblocks everything else — matchup truth, the pilot
   measurement, and the standing pool refresh after each settle.

   **Make it incremental and unconditional, not on-demand.** It should run after
   every settle and *append* to one growing episode store. Episodes age out of
   easy retrieval, the band we play in keeps moving, and every downstream tool
   wants a longitudinal corpus rather than a snapshot. Ten extra lines, and it
   is the difference between an instrument and a chore someone has to remember:
   the pool refresh below stops being a step and becomes a property of the data.
2. **The deck rebuild**, on the prize-liability budget: single-prize bodies so a
   misplay costs one prize instead of two.

   **State the lesson from the failed variants correctly** — they did not fail
   because more Pokémon is wrong, they failed because the slots came out of
   *consistency*. So the search space is **trainer-line-preserving swaps only**.
   That is a much smaller space, small enough that the tournament can be
   **exhaustive over it rather than sampled**.

   Score on the **matchup vector**, not the weighted scalar, and keep the
   flatness criterion live alongside the mean: at rank 778 we are entering the
   band where opponents stop being copy-paste and a spiky vector's blind spots
   get found. Bar: Archaludon and Lucario above ~0.45 with Grimmsnarl still
   above ~0.65 — that takes ~0.52 overall to ~0.61.
3. **Then one consolidated submission** with a written hypothesis, left to
   settle fully. Nothing ships unless its local effect clears the ±150 twin
   spread.

Standing rule after each settle: **re-run the autopsy and re-weight the pool.**
The band we play in moves as the rating moves, and a deck fitted to the old
neighbourhood is fitted to the wrong one.

Do not spend submissions on questions a local test can answer — five a day,
shared with Austin, and one submission carries ±150 points of noise.

**Division of labour**, now that two people share one quota: Austin owns agent
internals and his branch is canonical; this side owns measurement, the opponent
pool, and submission autopsies. One designated submitter per day, and every
submission carries a written hypothesis.
