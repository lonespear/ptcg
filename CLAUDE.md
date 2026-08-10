# Working on this repo

Kaggle **PTCG AI Battle Challenge**. Two linked competitions: *Simulation*
(an agent bundle, has the leaderboard) and *Strategy* (a ≤2000-word writeup,
70% model / 20% deck / 10% report). Deadline **13 Sep 2026**.

**Start with `GLIDEPATH.md`** — current state, what is invalidated, and what to
do next. Then `README.md` for the strategy and `EDA_FINDINGS.md` for the
numbered record. This file is the operating rules.

> Three measurement faults found 2026-08-08/09 invalidate much of the recorded
> evidence: our arena scored legal empty selections as forfeits (opponents only,
> 24.9% vs our 0.0%), the engine was never actually seeded, and 300-game cells
> flip sign at 600. Re-derive before citing a win rate. Details in `GLIDEPATH.md` §2.

## Hard rules

- **Never submit without being asked.** "Keep working" is not authorisation.
  Five per day, resetting **00:00 UTC** (8pm US Eastern), **shared with Austin**,
  and only the latest submission is active — so a worse one replaces a better
  one. Failed validations don't consume quota.
- **Never submit the Strategy writeup.** Not without an explicit, specific ask.
- **Never commit the engine or card data.** `ptcgProgram`, the `cg` package and
  its binaries, and the card CSVs/PDFs are competition-use-only and this repo is
  **public**. `engine/`, `build/`, `agent/deck.csv` and the card files are
  gitignored. `data/` and `agent/deck_priors.json` are aggregate statistics
  only — that distinction is deliberate, keep it.
- **Commit messages carry no Co-Authored-By line.**

## The three traps that have already cost us

1. **The grader `exec()`s `main.py`.** `__file__` is undefined — touching it
   unguarded kills the episode (cost 2 submissions). And `cg` is not importable
   from inside a function body, so **all engine imports go at module level**
   (a lazy import silently disabled search on every submission for a whole day).
2. **`except Exception` around infrastructure hides production failures.** Both
   bugs above degraded quietly instead of failing loudly. Prefer a loud signal.
3. **Leaderboard scores are worth +/-150 points.** An identical agent
   resubmitted scored 833.8 and 691.2; a repeated submission scored 863.2 and
   713.0. Scores also drift live and take hours to settle (724.9 -> 305.8,
   634.9 -> 695.8). **Never treat one submission as evidence** - use it to
   confirm a direction already established locally under SPRT.

## How to decide whether a change is good

- **Use SPRT, not a fixed game count** — `scripts/compare_agents.py --sprt`, or
  `scripts/field_sprt.py` for a play-weighted field on paired seeds. 100-game
  samples carry ±5% and have produced several wrong conclusions here.
- **Compare candidates directly**, never through a stronger third agent — that
  compresses the gap and nearly lost us the search work.
- **Re-test anything surprising on a fresh seed.** 5% error each way means about
  one decision in twenty is wrong.
- **Verify before submitting**: `scripts/build_submission.py --submit` runs the
  real `cabt` validation episode first and refuses on failure. Don't bypass it.

## Benchmarks have misled us four distinct ways

Name the bias before trusting a result:

| Benchmark | Bias |
|---|---|
| Our own weak agent | opponent far too weak — inverted the deck ranking |
| The official reference agent | monoculture, one archetype — cost ~240 rating |
| Field gauntlet | field piloted by us, so absolutes are optimistic |
| **Any self-play measure** | **symmetric failures cancel — caused a retraction** |

Corollary that keeps paying: **move questions out of the game-outcome channel.**
The leaderboard CSV settled a rank-vs-rating confusion, ground-truth replays
graded the opponent model, distributional comparison against real games found a
world-model bug, and reading code found two more. Games are for confirming; the
mined data is for learning.

And **check the instrument too — it is code under test.** Every bug that
invalidated evidence here lived in an instrument, not in the agent: the arena's
forfeit rule, the unseeded engine, a probe copying a formula instead of calling
it, one keying on a deleted counter, a watcher misparsing a timestamp, and an
analysis that measured three decks while silently omitting the two it existed to
measure.

Run `scripts/harness_selftest.py` before trusting a result. For any *new*
measurement tool, four checks catch this whole class:

1. **Symmetry** — swap sides; a rule that can only penalise one seat shows up
   here and nowhere else.
2. **Determinism** — same seed, same game, or say plainly that it is unpaired.
3. **Distribution** — compare against real replays; a private world that ends
   games at half the real length is not measuring the real one.
4. **Coverage — assert what you expected to find, and fail loudly when you do
   not.** A script that reports on 3 of 6 targets must exit non-zero, not print
   a confident table. This is the one that keeps getting missed.

## Conventions

- **Commit rejections with their numbers.** The decision ledger in `README.md`
  is three rejections to two accepts. That record is worth more in a
  methods-scored report than a string of wins, and it stops us re-litigating.
- **Retract in place, prominently.** When a claim is withdrawn, correct it where
  a reader meets it — not only in a later section — and tell Austin, who works
  from `FEEDBACK_FOR_AUSTIN.md`.

## Collaborator

Austin (`defense031/ptcg`) owns `ptcg/creation/` — an island-model GA over
decks. Cross-repo notes go in `FEEDBACK_FOR_AUSTIN.md`. He rebases onto our
`main`.

## Layout

| Path | What |
|---|---|
| `agent/main.py` | the submission: rules + 1-ply search |
| `ptcg/opponent.py` | Bayesian decklist posterior |
| `ptcg/arena.py`, `ptcg/sprt.py` | head-to-head play, sequential testing |
| `scripts/mine_day.py` | download a day of replays → aggregate → delete (20 GB each) |
| `scripts/validate_*.py` | grader episode, posterior vs ground truth, rollout vs replays |
| `scripts/gauntlet.py`, `field_sprt.py` | play-weighted field evaluation |
