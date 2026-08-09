# What we learned between v3 and v9

Austin's side, written for Jon, 2026-08-09. Seven submissions, about forty
experiments, and a week of measuring things badly before we noticed. The most
useful parts are the three bugs in our own test harness and the four ideas that
sounded good and did nothing, so those come first.

---

## 1. Three bugs in the harness, not the agent

These cost us more than any agent defect. If your setup shares any of this
plumbing, check for them before trusting a number you have already collected.

### The engine was never seeded

`ptcg/arena.py` took a `seed` argument and passed it to Python's
`random.seed()`. The engine does not use Python's generator. `libcg` draws its
randomness from `std::random_device` and exports no way to seed it, so the seed
never reached the thing that deals the cards.

Every "seeded" run in this repository's history was a fresh deal. No two runs of
the same command ever played the same games, and every gate that reported paired
seeds was paired in seat assignment only.

Fixed by preloading a replacement for that entropy source inside the process
(`tools/engine_seed/`, `ptcg/engine_seed.py`, `scripts/run_pinned.sh`). Nine runs
of one matchup now return the identical 479 wins to 121 out of 600 clean games:
idle, under an eight-process CPU load, and at one, two and four workers.

Two traps we hit:

- `scripts/run_pinned.sh env VAR=x python ...` runs **unpinned**. macOS strips
  `DYLD_*` for SIP-protected binaries and `/usr/bin/env` is one. It fails
  silently, so `ptcg.engine_seed.available()` must be checked inside the actual
  worker process, not in a probe shell beforehand.
- Runs on an idle machine landed within one game of each other and looked
  reproducible. They were not. At the decision level every unpinned game differed.

### One search handle per process, shared across games

`cg.api` allocates a single engine-side `AgentStart()` handle the first time
anything searches and holds it for the life of the process. Every game a worker
played was searching through state the previous game left behind. Which games
shared a process was set by `--workers`, which is why the same seed block scored
differently at two workers and at four. Now reset per game.

### An empty selection was scored as a crash

Our arena treated any empty list returned by an agent as a forfeit. But an empty
list is the engine's own answer whenever a prompt's `minCount` is 0 — "done
benching", declining a search, declining an attachment — and the competition's
reference agent returns it.

All 7,726 "opponent crashes" across 97 of our result files were legal moves. Our
agent cannot emit `[]` by construction, so the penalty fell only on opponents: a
24.9% opponent forfeit rate against our 0.0%, wildly uneven by matchup (58% in
one cell, 0% in another).

What we had been reading versus the truth: Archaludon 0.63 where it is really
0.32. Grimmsnarl 0.85 where it is 0.758. Every "we win 56% of the panel" claim
should have read about 44%, against a ladder truth of 45%.

---

## 2. What our measurements can and cannot see

**A 300-game matchup cell resolves nothing under about 8 points.** We watched
four candidate/cell readings flip sign between 300 and 600 games, in both
directions. One candidate read −2.7 at 300 games and +6.2 at 600; another read
+2.0 and then −5.5.

**Even at 600 games the floor is about 6 points** of win rate at 80% power. With
the engine pinned, pairing buys only a 1.25x variance reduction, because the
engine's random stream is consumed by play: one different decision re-deals the
rest of the game. Rolling the stream back across each agent call would fix that
and we have not built it.

**Our test panels do not reproduce the ladder.** Three separate ways:

- The specialist panel's mechanics are wrong. Our v7 shipped a damage-wall fix
  that gated at +13 points and delivered nothing, because damage walls appear in
  3 of 80 real ladder decks and none of our six panel decks.
- A panel built from real ladder decklists reproduces the decks but not the
  opponents, because we drive them with our own agent — which is measurably bad
  at driving decks that are not ours. It says we win 74% where the ladder says
  45%.
- The ladder itself gives about 30 noisy games per submission.

**Launch luck is larger than any change we made.** Two submissions of *identical
bytes* finished 116 points apart. Another pair, measured behaviorally identical
on the ladder, finished 140 apart. A submission that wins its first several games
climbs high and settles down; one that loses early drops into a range where it
only meets weak opponents, and beating a 600-rated opponent earns almost nothing,
so it climbs back slowly. Judge a submission at roughly 30 games, never at its
peak.

---

## 3. Bugs in the agent, found and fixed

All verified against live engine output rather than against a win rate.

| what was wrong | how wrong | shipped |
|---|---|---|
| Damage-prevention abilities ignored (Cornerstone, Crustle, Sylveon, Milotic, Farigiraf, Drednaw) | priced attacks at 210–300 into defenders taking 0 | v7 |
| Cost-free abilities discarded by the search | 340 of 1,255 legal Teal Dances thrown away, 2.1 a game | v8 |
| Stadium damage modifiers ignored (Full Metal Lab, Neutralization Zone) | 31 of 301 attacks over-predicted, every error high, none low | v9 |
| Attack payability under Nighttime Mine (Tera attacks cost one more Colorless) | 150 of 838 attacks believed affordable that the engine did not offer | v9 |
| End-of-turn damage counters (Froslass family) | 2,001 of 15,451 checkups wrong | v9 |
| Weakness applied to attacks that *place damage counters* | doubled damage the engine pays in full — placing counters is not dealing damage, and the engine says so in its own log field | v9 |
| Resistance hard-coded at 30 | the Duraludon line resists 60 | v9 |
| Attack text waiving resistance ignored | predicted 0, engine dealt 40, 26 of 26 | v9 |
| Our own damage boosters invisible | 108 knockouts we could have taken and did not see | v9 |
| Rare Candy stage skips never modelled | 87 of 87 opponent skips unseen | v9, dormant |

The pattern worth stealing: instrument the agent to record what it *believed* at
each decision, then diff that against what the engine actually did on the next
step. `scripts/belief_audit.py` does this and adding a class is one function.
Every real gain we found came from that, and none came from making the agent
smarter.

Two card-text traps that cost us time:

- `{C}` in ability text means the Colorless **type**, not the generic energy cost
  symbol. Team Rocket's Watchtower switches off `{C}` Pokemon's abilities, and we
  briefly believed it silenced our Grass Ogerpon. It cannot.
- Mega Lucario ex is typed `megaEx`, not `ex`, in the pool's own schema. Any
  condition matching on `ex` misses it.

---

## 4. What did not work, which may save you the trouble

**A learned evaluator.** We trained a small network on 584,827 real decision
points with 84 named features, using pairwise ranking of the choices a strong
player actually made. It beat the linear evaluator on held-out decisions in every
archetype, by 2.45 points of top-1 accuracy. Re-tested at 600 games a cell it was
worth **+0.0003** pooled over 3,599 games an arm. Better decision-fit, zero extra
wins. Notably, the arm trained only on game outcomes ranked *worse* than linear.

**More search.** Raising determinizations 14x moved the mean search from 16ms to
127ms and went 450–450 over 900 games. The override rate — how often search
overruled the rule policy — held flat at 53.9% against 54.2%. At three
determinizations the search has already converged on its answer.

**Copying strong players' decisions.** Five experiments raised our agreement with
1100+ rated agents. One converted. The clearest case raised agreement on a
decision class from 21% to 69% and moved win rate not at all. Matching good
players pays only where the decision does not depend on their plan.

**Changing the deck.** See below; it is its own story.

---

## 5. The deck, settled

Our hand-built Ogerpon list **loses head-to-head duels**. Across 80,000 pinned
games over eleven candidate lists, all driven by the same frozen agent so the only
difference was the 60 cards, it finished 8th of 9. Six challengers beat it past a
Holm correction, the best by 9.5 points.

**And every duel winner then failed the four-matchup gate**, always in the same
place: against the Alakazam deck, where our list plays even at 50.7% and the
challengers collapse to between 22% and 37%.

Both are true because **a duel between two Ogerpon decks is the mirror match,
which is 3.5% of the ladder, while those four matchups are 85.6%.** Switching to
the duel winner buys 9.5 points on 3.5% of games and pays 4.0 points on 85.6% —
about 3 points worse overall. A duel ranks decks correctly against each other and
chooses between them wrongly.

**Cheaper boards do not help.** We built 157 legal mono-Grass lists, all carrying
at least two Teal Mask Ogerpon ex, and varied one-prize Pokemon, energy count,
Stadium count and search cards. One axis explained the space, running the wrong
way: zero one-prize bodies scored 0.470, two scored 0.265, three 0.181, four
0.118. Across all 157 candidates the correlation between conceding fewer prizes
per Pokemon and winning was **+0.805 in the wrong direction**. Cheap bodies do not
attack well enough to be worth their discount.

**A wall deck works mechanically and loses anyway.** Crustle took **zero damage
across 4,497 attacks** while attacking 1,942 times, and doubled game length to
15.6 turns. It still lost badly: Froslass and Munkidori damage through an
*Ability* rather than an attack, so the wall never applies; Mega Lucario ex is
typed `megaEx`; and the Alakazam deck runs no ex Pokemon at all, so the wall does
nothing in the matchup that decides the most games. Its one real gain was +14.3
against Archaludon.

**Why our genetic algorithm's decks looked better than they were.** It screens on
8 games a cell and decides on 6, so a 27-cell fitness is about 160 games with a
standard deviation near 4 win-rate points — and then takes the best of roughly
1,600 evaluations. Re-scoring those winners at 342 games a cell moved them down
3.2 points on average, 19 of 20 down, p = 2e-5. It read one deck as 19.8 points
better than our list; the truth was about a third of that, for a different deck.
The search was never the broken part. The estimator was.

---

## 6. Ladder facts

- **Decks are not secret.** Every replay's step 0 carries both players' full
  60-card lists. Hours after our v2 went up, another team submitted our exact 60.
  We have since recovered 80 opponent decklists this way; it is the best source of
  field data available and it is free.
- **Ratings by submission**: v1 668, v2 785, v3 713 and its duplicate 863, v4 629,
  v5 691, v6 771 (peaked 1004 at eight games, top 100), v6's duplicate 655, v7
  620, v8 629 and converging, v9 just up.
- Only your two most recent submissions score, and the team takes the better of
  the two. Each new submission evicts the older one.

---

## 7. Open questions

1. **No instrument reproduces the real field.** The plausible route is pilots for
   field decks that are not our own agent. This is the binding constraint on
   everything else and it is a multi-day build.
2. **Roll the engine stream back across each agent call** so lookahead does not
   advance the deal. That is what would make paired comparisons actually pay.
3. **One live deck lead**: swapping one N's Plan for one Harlequin read +6.0
   against the Alakazam deck, the only one of 157 candidates to move that matchup
   upward. At 200 games it is well inside noise and needs about 1,200.
4. **Opponent abilities are mostly unmodelled** — by panel weight, 2.12 of 3.15
   sits in cards we ignore. Froslass and Munkidori are the largest.
5. **Final-pair timing.** Given launch luck of 116–140 points on identical agents,
   when the last two submissions go up, and whether we are willing to discard a
   bad pair and re-roll, is worth more than any remaining code change.

---

## 8. The instruments, and how to run them

All on this branch. The experiment logs and the competitor decklists are not —
`data/analysis/*` is gitignored, so you build your own copy of the field data
rather than inheriting ours.

| tool | what it does |
|---|---|
| `scripts/belief_audit.py` | the belief-vs-truth differ. Records what the agent predicted at each decision, diffs it against what the engine did next. Adding a class is one function plus a `PREDICTORS` entry. This is the tool that found every real bug. |
| `scripts/reproduce_check.py` | runs the same command N times and reports the spread. Use it to confirm your own harness reproduces before trusting any A/B. |
| `scripts/run_pinned.sh` | wraps a command with the engine-RNG preload and the deterministic search budget. Builds the preload on first use. |
| `scripts/extract_field_decks.py` | pulls both players' 60-card lists out of replay JSON and builds a field deck corpus. Start here — the lists are public and free. |
| `scripts/build_field_panel.py` | turns that corpus into a weighted panel of matchups. |
| `scripts/field_panel_gate.py` | runs an agent against that panel; `--report tagA tagB` tabulates two arms. |
| `scripts/field_panel_validate.py` | checks the panel against known ladder results. Ours failed this — it says we win 74% where the ladder says 45% — which is how we learned not to trust its absolute level. |
| `scripts/specialist_gate.py`, `scripts/ab_gate.py` | matchup and head-to-head gates. Both now headline the forfeit-excluded win rate and warn below a clean-sample floor. |

Run any gate as `scripts/run_pinned.sh python scripts/specialist_gate.py ...`,
and check `ptcg.engine_seed.available()` inside the worker rather than in a
shell beforehand.

The trained evaluator from the experiment in section 4 is committed at
`data/analysis/leaf/model_h64_multi.npz` with its gate results beside it, in
case the offline result is useful to you even though it won no games.
