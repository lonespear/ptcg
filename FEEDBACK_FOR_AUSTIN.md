# Feedback + merge notes for Austin

Reply to `STATUS.md` / `ARCHITECTURE.md` / `UTILIZATION.md` on your
`deck-creation` branch. Written 2026-08-06 against `lonespear/ptcg@c134f1b`.

Heads up: there is **no `collaborate_and_merge.md`** on either branch of your
fork — not on `main`, not on `deck-creation`. Jon said to read it, I couldn't
find it, so I read the three docs that are there instead. If it exists locally
you haven't pushed it.

Your fork's `main` is at `1b2a264`; we're 8 commits ahead. Rebase before merging
— several of those commits change `agent/main.py` substantially.

---

## 1. Your lazy-import bug is REAL. It was costing us everything.

This is the single most valuable thing either of us has found. I did not take it
on trust — I instrumented our actual submission bundle and ran it under
`kaggle_environments`' own `cabt` environment (the real validation episode).

**Before the fix:**

```
agent_calls   77
attempts      40          <- _search_main entered
import_ok      0          <- cg.api imported successfully
import_fail   ModuleNotFoundError("No module named 'cg'")
returned       0          <- search reached its option loop
```

The grader `exec()`s `main.py` with the bundle directory as cwd, but `cg` is not
on `sys.path` by the time a *function body* runs. Our
`except Exception: return None` turned that into a silent fallback to the rule
policy on every single decision.

**Consequence: every submission we have ever made scored on rules alone.** v9 sits
at 656.4 on the ladder with its search subsystem completely inert. Everything we
measured about search — 0.72 head-to-head over rules, the 2-ply rejection, the
determinization results — was locally real and competitively nonexistent.

**Fix** (commit `c134f1b`): import at module level like the official sample, after
putting the agent dir on `sys.path`.

```python
for _p in (_HERE, _KAGGLE_DIR, os.getcwd()):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)
try:
    from cg.api import (all_attack, all_card_data, search_begin, search_end,
                        search_step, to_observation_class)
    CG_AVAILABLE = True
except Exception:
    CG_AVAILABLE = False
```

After: `returned 42` in 77 agent calls. Search runs.

**Reproduction harness**, so you can check your own build and re-check after any
refactor: `scripts/validate_submission.py` runs the real `cabt` validation
episode locally and `build_submission.py --submit` refuses to send unless it
passes. If you want the instrumented probe that produced the counts above, say
so and I'll commit it — it lives in scratch right now.

**Generalised lesson worth adopting on both sides:** any `except Exception` around
infrastructure in the agent can silently disable a whole subsystem in
production. We should assert loudly rather than degrade quietly. I'd take a PR
that makes `CG_AVAILABLE == False` print once to stderr.

## 2. Your bug #2 — I have NOT verified it

"Wrapper code was unreachable due to execution model quirks." I could not
reproduce this from the STATUS.md summary alone, and `UTILIZATION.md` doesn't
contain the detail. **I am not treating it as fact until I've reproduced it**,
which is the same standard I applied to bug #1 before acting on it.

Please push the specifics: file, the code shape, and how you observed it. Given
bug #1 was real and expensive, this one gets the same priority.

## 3. Your asks, answered

| Ask | Status |
|---|---|
| Add an OSI license | **Done.** MIT, `LICENSE`, scoped to our code only — engine, card data and anything derived from the engine's card tables are explicitly excluded, since they're competition-use-only and the repo is public. |
| Review `UTILIZATION.md`, esp. weight-vector injection | See §6 — I want it, with one change. |
| Verify team roster alignment | Can't verify from here; Jon's call. |
| Validate the two grader bugs | #1 confirmed and fixed (§1). #2 unverified (§2). |

## 4. The problem your GA should target

This is where `ptcg/creation/` earns its keep, and it's urgent.

Our live deck is the mined Teal Mask Ogerpon ex list. It is a **best response to
the field** — mono-Grass, and ~47% of the metagame (Marnie's Grimmsnarl ex) is
weak to Grass, while nothing in the top six plays the Fire it's weak to. It
beats Marnie's 92–8 in our hands.

**It also runs 4 Pokémon in 60 cards.** Four Teal Mask Ogerpon ex and nothing
else. Four knockouts and the game ends regardless of prizes.

Validating our self-play distributionally against 150 real replays:

| | Real replays | Our self-play |
|---|---|---|
| Game ends: no Pokémon left | 8% | **70%** |
| Median turns | 13 | **8** |
| Attacks per turn | 2.35 | 1.08 |

Real players hold 52.8% with this exact list by protecting those four. Our pilot
does not.

**What I'd want from the GA:** a deck that keeps the Grass matchup edge but
survives our pilot's competence — i.e. optimise against a fitness function that
includes *termination mode*, not just win rate. A deck that wins 60% while
losing 70% of its losses to running out of Pokémon is one bad matchup from
collapsing.

**And the reason no test of mine caught it:** the field-weighted gauntlet pilots
*both sides* with our own agent, so a failure mode that hurts both arms equally
cancels out. Every deck in it was dying the same way; the ranking held while the
absolute behaviour was nowhere near the real field. If your GA fitness is
self-play based, it has the same blind spot — **check candidate decks against
replay distributions, not just against each other.**

## 5. Measurements you can use rather than re-derive

All from `EDA_FINDINGS.md`; scripts are in `scripts/`.

**Deck inference is capped by the metagame's structure, not by our algorithm.**
Graded against ground truth (every replay contains both real decklists —
`scripts/validate_posterior.py`, 300 replays, zero games played):

- Prior covers **99.7%** of opponents
- Top-1 correct: 0.63 turn 1 → 0.84 turn 3 → 0.94 turn 10
- Well calibrated: claims 0.6–0.7 → right 0.70; claims 0.9+ → right 0.98
- **But** when it's wrong it shares a median **53 of 60 cards** with the truth,
  and **80.6%** of misses are the *same archetype*. Genuinely harmful
  misidentification is ~5%, not 37%.

Consequence: multi-determinization over decklists is worth ~nothing (measured:
0.485 over 373 paired games, rejected). Don't spend time there. The hidden
variable that matters is the **shuffle**, not the deck.

**Related bug I found in our own code, same family as yours:** the split of the
opponent's unseen cards into hand/deck/prizes was taken in dict order, and the
priors file stores card IDs ascending — so the simulated opponent's hand was
deterministically their lowest-numbered cards, i.e. **basic Energy, every
rollout, every game**. Every simulated reply came from an opponent holding no
Pokémon and no Trainers. Fixed; now shuffled and seeded from the position.

**Opponent strength matters for inference** (relevant as we climb): coverage
holds at 1.000 against strong agents, but top-1 accuracy drops 0.715 → 0.609.
Their lists are harder to disambiguate early.

**Your 96-4 / 27-73 finding replicates ours.** We measured the best mined
leaderboard decklist losing **32–88** to the trivial starter deck under our
pilot. Deck strength is not separable from the policy that plays it. Two
independent measurements of the same effect is worth stating in the report.

## 6. On `UTILIZATION.md` and weight-vector injection

Yes, and I want it — with one condition.

Our leaf evaluation is currently:

```
score = 1000 × (their prizes − our prizes)      # win condition
      +        (our board HP − their board HP)
      +   30 × (our energy − their energy)      # worth 0.707 and 0.608 on two seeds
```

Those weights are hand-set from unit reasoning, not fitted, so exposing them as
an injectable vector is obviously right and lets your GA co-optimise deck and
pilot weights.

**The condition:** anything tuned that way has to clear the same bar as
everything else — SPRT on the paired field harness
(`scripts/field_sprt.py`), not a fixed game count. We have lost real rating to
noisy 100-game reads more than once. Note also that only ~19% of paired games
are informative (four in five, two agents differing by a whole subsystem reach
identical outcomes on the same deal), so weight fitting will need more games
than it looks like it should.

Also worth knowing before you fit: **the energy term only helped because our
attacker's damage is literally linear in energy** (Myriad Leaf Shower: printed
30, +30 per Energy on both Actives). A weight vector fitted for this deck will
not transfer to a deck without a scaling attacker. Fit per-deck, or include the
deck in the fitting loop.

## 7. Merge plan I'd propose

1. You rebase `deck-creation` onto our `main` (you're 8 behind; `agent/main.py`
   has changed a lot).
2. `ptcg/creation/` merges cleanly as a new package — no conflicts with anything
   we own. I'd take it.
3. Weight-vector injection as a separate PR against `agent/main.py`, since that
   file is the submission and I want its diff readable.
4. Keep `scripts/validate_submission.py` as the gate on both sides. Given bug
   #1, nothing goes to the ladder without a local `cabt` episode passing.

## 8. Two conventions I'd like us to share

- **Commit rejections with their numbers.** Our README carries a table of what
  we tried and what failed with SPRT verdicts — three of five changes rejected.
  "We built it, tested it properly, it didn't help, here's why" is worth more in
  a methods-scored report than a string of wins, and it stops either of us
  re-litigating a settled question.
- **Move questions out of the game-outcome channel wherever possible.** Every
  decisive step this week came from direct measurement instead of playing
  games: the leaderboard CSV settled a rank-vs-rating confusion, ground-truth
  replays settled the posterior's value, distributional comparison found the
  fragile deck, and reading code found both the shuffle bug and yours. Games are
  for confirming; the mined data is for learning.

---

Fastest thing you can do for us: push the bug #2 detail. Fastest thing we can do
for you: point the GA at the 4-Pokémon fragility with a termination-mode-aware
fitness function.
