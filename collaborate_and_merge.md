# Collaborate & Merge — the plan, for Jon and Jon's agent

*2026-08-06, from Austin's side. This branch (`deck-creation` on
`defense031/ptcg`) now contains the merged system: your updated `main`
(through `ed5453c`) semantically merged with our chunk-1 stack and agent
upgrades. This file explains what was merged, why each resolution went the
way it did, how it was verified, and what to run. Companion docs:
`ARCHITECTURE.md` (three-chunk map), `UTILIZATION.md` (pilot contract),
`STATUS.md` (our side's state and measurements).*

## The one-file collision, and the resolution rule

Everything except `agent/main.py` was disjoint. For `main.py` the rule was
**your hidden-information machinery, our search/eval/grader machinery**:

| Component | Kept | Why |
|---|---|---|
| Guarded top-level `import cg.api` | ours | Under Kaggle's exec model the lazy imports fail: `search_begin` fired **0 times** in the deployed bundle (falsification-tested; 273 calls post-fix). Your ladder agent has been playing rules-only. |
| `agent` wrapper as the **last** callable | ours | `kaggle_environments` binds the last callable in the file; it was binding `_agent`, so the never-crash guard was dead code. |
| `WEIGHTS` + `set_weights()` | ours | Your UTILIZATION item 3: every eval constant is now injectable; the tuner (`ptcg/creation/tuning.py`) optimizes this vector. |
| `_search_main`: margin 0, deadline-everywhere, N_DET=3, `SEARCH_OPP_BRANCH=0` | ours | Measured on 200-game A/Bs: hysteresis margin 500 cost 14 pts (it defends the *weaker* half of your pair); 2-ply cost 9 pts — diagnosis below. Constants carry their measurements in-code. |
| `_evaluate` + no-active guard, derived card primitives | ours | +5 pts rules mode (n=2000). No card IDs anywhere. |
| Bayesian posterior: `_log_choose`, `_deck_posterior`, `_hidden_from`, `CONFIDENCE_GATE` | **yours** | Validated and calibrated (0.63@t1, 0.84@t3, 0.94@t10; claims 0.9+ → right 0.98). Supersedes both older versions. `_predict_opponent` survives only as a thin point-estimate wrapper over your posterior; `_search_main` calls `_deck_posterior`/`_hidden_from` directly so it can average over candidates. |
| Position-seeded shuffling for both sides' hidden cards | **yours, with a fix** | Your seeded deals give paired A/B tests identical worlds — better hygiene than our global rng. But the seed keyed on `obs["step"]`, a kaggle_environments field the raw engine observation does not carry, so locally it collapsed to a constant within a turn and all three determinizations replayed one deal. The key now also carries `turnActionCount` and the option count. Measured: 44.5% before the fix, 50.5% after, same 200 paired games. `_own_hidden` keeps our `rng=` parameter so the call sites stay compatible. |
| 2-ply: your `TWO_PLY` flag vs our `SEARCH_OPP_BRANCH` | ours | Same mechanism, ours branches over their top-K replies and takes the minimum instead of rolling one; your rejection number (0.467) is recorded in the constant's comment beside ours. One flag, not two. |
| `validate_posterior/rollout`, `termination_modes`, `field_sprt`, `misid_cost`, LICENSE, README, `ptcg/opponent.py`, EDA additions | **yours** | Disjoint from our side; took them whole. The distributional-honesty check enters our standing evidence rail. |
| `ptcg/creation/`, `ARCHITECTURE.md`, `UTILIZATION.md`, `STATUS.md`, `.gitignore`, `build_submission.py` | ours | Disjoint from yours; `build_submission.py` carries the `recursive=False` tar fix that stopped every `cg/` file being written into the archive twice. |

## Two results you can use immediately

1. **Your queued 2-ply retest is already run.** On the *shuffled* hand model
   (the configuration your README says was never tested), 2-ply still loses
   ~9 points. Diagnostics (commit `1e1867c`): K=1 without the min is worse
   (36.2%), removing the time budget recovers only 1.5 pts. The residual
   fiction is the opponent **policy** — their turn rolls out under *our*
   priority table. The mechanism sits behind `SEARCH_OPP_BRANCH`, ready to
   re-enable when an opponent policy model exists.
2. **Termination modes are already a gauntlet output on our side.** Every
   `GameResult` in `ptcg/creation/harness.py` carries its end reason; the
   36-cell matchup matrix records them per cell. Your 70%-deck-out finding
   is checkable against any deck we generate, automatically.

## Your #1 queue item is our machine's job

"Rebuild the list with backup attackers while keeping the Grass advantage"
is a constrained deck search: Grass-anchored islands seeded with the Ogerpon
core, purity floors preserving the type edge, your gauntlet + termination
distribution as acceptance. The archipelago runs it as soon as tonight's
weight tuning fixes the pilot (see below). Expect candidate lists within a
day; you get decks + validator reports + matchup rows, you judge.

## How to run what's here (from repo root, venv with pandas/numpy)

```bash
# deck GA vs the specialist-piloted real field (10 workers ~ 7x)
python scripts/run_archipelago.py --hours 8 --pilot jon --workers 10 --git

# weight tuner: evolution strategies vs TD(lambda), then the duel
python -m ptcg.creation.tuning --mode both --es-games 1000 --td-games 3000 \
    --gens 10 --workers 10 --out data/tuned_weights.json --stamp r1
python -m ptcg.creation.tuning --mode duel --games 300 --workers 10 \
    --out data/tuned_weights.json --stamp r1

# 36-cell field matchup matrix, 500 games/cell, resumable
python scripts/matchup_matrix.py --games 500

# D16 instruments: decision stability, calibration, goldfish bands, outs
python -m ptcg.creation.stability --games 50
python -m ptcg.creation.calibration --games 3000
python -m ptcg.creation.goldfish external/<deck>.json 200
python -m ptcg.creation.outs

# submission (validates first, refuses to ship on failure)
python scripts/build_submission.py --submit -m "note"
```

Engine setup is unchanged from your README (copy `sample_submission/` into
`engine/`; nothing licensed is committed). Community specialist agents live
in gitignored `external/` — ask Austin for the harvest script rather than
committing them.

## Verification this merge passed before pushing

1. **Kaggle-exec falsification.** A `cabt` self-play episode run from the built
   bundle, in a clean process with the repo engine off `sys.path`: `cg`
   resolved to the bundle's own copy, `kaggle_environments`' last-callable rule
   bound `agent`, `search_begin` fired **209 times**, both seats DONE over 133
   steps.
2. **200-game A/B, merged vs pre-merge branch agent**, rank-0 Marnie list both
   sides, search on, seats swapped, two 100-game halves on disjoint seeds:
   **101-99, 50.5%**. The merged agent does not regress play. The first run of
   this gate returned 44.5% and is what surfaced the seed defect above.
3. **Smoke.** `ptcg.creation.stability --games 10` clean (74.5% MAIN decision
   invariance over 251 main selects, 100% on non-main); a 6-game `JonDayPilot`
   harness match runs, search vs rules-only, 3-3. `validate_rollout.py` and
   `termination_modes.py` import and show usage.

## Proposed next steps, jointly

1. **PR this branch → `lonespear/ptcg` main.** It returns your posterior
   merged with the grader fixes your live agent needs. Resubmission of the
   fixed agent is worth doing the moment the PR lands — same deck, search
   actually firing.
2. Tonight: our tuner picks the weight vector (ES vs TD(λ), head-to-head).
3. Tomorrow: Ogerpon rebuild via the archipelago (your acceptance test).
4. Chunk 3 wiring: your posterior + our threat curves → the aggression knob
   (decisions D13 in Austin's ledger; ships only if a 500-game A/B wins).
5. Strategy-track entry: your README says "not without being asked" — this
   is the team asking. Entry deadline Sep 6, report Sep 13 UTC, ≤2,000-word
   Writeup, rubric 70/20/10. The evidence tables in this repo are the draft.
