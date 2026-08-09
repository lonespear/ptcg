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

Last confirmed standing: **rank ~831 of 6,563, top ~12.7%**, score ~818–834,
from 4,567th three days earlier. Leader ~1,182; exactly one team above 1,200;
median 631. **Verify before quoting — see §2.**

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

### 2d. And the leaderboard itself is ±150

Byte-identical submissions scored **833.8 and 691.2**; a repeated submission
scored 863.2 and 713.0. v6 read 1004.2 at 8 episodes and later crashed to 782.
Never conclude from one submission.

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

## 6. If you do one thing next

In order:

1. **Re-run the decision ledger** with the arena fix, the seeded engine, and
   600-game cells. Several verdicts in `README.md` will move; some accepts may
   become rejects.
2. **Re-weight the field** from the ladder autopsy (§4), then re-rank decks.
3. **Merge his branch into ours**, carrying the four guard files forward.

Do not spend submissions to answer questions a local test can answer — the
quota is five a day, shared, and one submission carries ±150 points of noise.
