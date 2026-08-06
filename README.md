# ptcg — Pokémon TCG AI Battle Challenge

Our entry for the Kaggle **Pokémon TCG AI Battle Challenge**. Final deadline
**13 Sep 2026**.

**Live standing (6 Aug):** `v9` at **656.4**, roughly rank **2905 of 6421** —
**above the median** and still settling. We started the day at 445.8.

| Submission | Score | Rank | |
|---|---|---|---|
| v3 — starter deck, rules only | 544.8 | ~4292 | |
| v4 — Mega Lucario ex | 305.8 | ~6024 | a bad benchmark call |
| v5 — search + Teal Mask Ogerpon ex | 535.8 | ~4391 | |
| **v9 — energy term in the eval** | **656.4** | **~2905** | **top 45%** |

**+182 rating from here is the top decile.**

The score is a rating, not a rank — worth stating because they get confused. The
field:

| | Score | Rank |
|---|---|---|
| Leader | **1182.2** | 1 |
| Top 1% | 1026 | ~64 |
| Top 10% | 838 | ~642 |
| **Median** | **631** | ~3210 |
| **Us** | **520.5** | **4567** |

**Nobody is above 1200.** We are ~110 points below median, and the whole board
spans about 1450 points.

What a rating gain is worth here, because the middle of the board is dense —
1.4% of all teams sit within ±5 points of us:

| Gain | Score | Rank | Percentile |
|---|---|---|---|
| +100 | 620 | ~3352 | top 52% |
| +200 | 720 | ~1998 | top 31% |
| +300 | 820 | ~743 | top 12% |
| +400 | 920 | ~271 | top 4% |
| +517 | 1037 | ~50 | top 0.8% |

So this is not a hopeless gap needing a learned policy — **+300 rating is the
top decile**, and we are still below average. The cheap wins have not been taken
yet.

> Ratings take hours of episodes to settle. An early reading of 724.9 on the
> Mega Lucario submission fell to 305.8 once it had played enough games, and we
> made a bad decision on the strength of it. Numbers here are settled ones.

---

# What we are actually building

There are two linked competitions and it is easy to mix them up.

**Simulation** (`pokemon-tcg-ai-battle`) is the one with a leaderboard. You do
not submit a deck — you submit a **program that plays the game**. Kaggle runs
your program against other people's programs, thousands of matches a day, and
gives you a rating (like an Elo). Your submission is a folder containing:

- `main.py` — a function `agent(observation)` that the game engine calls **every
  single time you have a decision to make**. Play a card? Attach an Energy?
  Which attack? It asks; you answer with the index of the option you want.
- `deck.csv` — your 60-card decklist, as 60 card ID numbers.

**Strategy** (`...-challenge-strategy`) is a written report explaining *why* you
built what you built. It requires a Simulation entry to be eligible. Scored 70%
on the model, 20% on the deck, 10% on the report. **We have not submitted this
and will not without being asked.**

So the deck is only part of it. The program that pilots the deck is the bigger
half, and the two cannot be judged separately — which turned out to be the
single most important thing we learned.

---

# How the agent plays

The whole policy fits in one page of rules, and that is deliberate — the report
has to explain it.

### Rule 1 — do everything free before the thing that ends your turn

In this game, **attacking ends your turn**. Everything else (using an Ability,
evolving, playing a Trainer, attaching an Energy) is free. So the agent ranks
its options and always takes the free ones first:

```
Ability  →  Evolve  →  Play a card  →  Attach Energy  →  ATTACK  →  Retreat  →  End turn
```

A player who attacks the moment they can is throwing away a Rare Candy, a
Supporter draw, and an Energy attachment every turn. Sorting the menu fixes that
without any game knowledge at all.

### Rule 2 — take the knockout, and take the *cheapest* one

When choosing an attack, the agent looks at the defending Pokémon's **remaining**
HP, not its printed HP, and applies Weakness (×2) where it applies. Then:

- If any attack knocks the target out → use the **smallest** one that does.
- Otherwise → use the biggest hit available.

Preferring the cheapest knockout matters because the big attack usually has a
drawback (discard your Energy, can't attack next turn). If 60 damage gets the
prize, there is no reason to spend the 200.

### Rule 3 — count prizes, not damage

This is the rule that moved our score the most.

You win by taking **six Prize cards**, and how many you give up when *your*
Pokémon is knocked out depends on what it is:

| Your Pokémon | Prizes the opponent takes |
|---|---|
| ordinary Pokémon | 1 |
| Pokémon **ex** | 2 |
| **Mega** Pokémon ex | **3** |

A Mega ex with 350 HP feels unkillable, but it is *two knockouts from losing the
game*. So when the agent scores a card — which to promote to the Active spot,
which to fetch when searching the deck — it uses:

```
score = HP + 2 × (best attack damage) − 220 × (extra prizes it concedes)
```

That penalty is what stops it from reflexively leading with the biggest Pokémon
on the board.

### Rule 4 — Energy in play is worth something on its own

This one is the single biggest improvement we made, and it is four lines.

Our attacker, Teal Mask Ogerpon ex, has exactly one attack: **Myriad Leaf
Shower** — 3 Grass, printed at **30 damage**, and *"does 30 more damage for each
Energy attached to both Active Pokémon."* With six Energy on the board that is
**210 damage, not 30**.

So for this deck, **Energy attached *is* the position.** But the agent's
scoring function only counted prizes and HP. That meant two different
attachment choices scored *identically* unless one of them happened to win a
knockout on the spot — which is why attaching Energy was our worst decision even
after we routed it through simulation.

The fix is to score Energy directly:

```
score = 1000 × (their prizes left − our prizes left)   # the win condition
      +        (our board HP − their board HP)
      +   30 × (our Energy in play − their Energy in play)   # <- this
```

30 sits on the HP scale, so it can never outweigh a prize — it just makes
building toward the big attack visible instead of invisible. Measured twice on
independent samples: **0.707** and **0.608** against the previous agent.

The general point: a scoring function has to contain the thing your deck is
actually accumulating, or search cannot see progress toward it.

### Rule 5 — everything else

Attach Energy to the Active Pokémon (it's the one that attacks). Draw the
maximum when asked "how many?". Never mulligan a legal opening hand. When asked
to *discard*, give up the lowest-scoring card; when asked to *gain*, take the
highest.

### One non-obvious implementation note

Options the engine offers you look like `{"type": 3, "area": 2, "index": 5}` —
they carry a **zone and a position, not a card ID**. To know what you are
choosing you have to look it up on the board (`players[i].hand[5]`, or
`select.deck[5]` when searching). We got this wrong at first and the agent was
silently picking option 0 every time.

---

# The deck: Teal Mask Ogerpon ex, and why

The deck is deliberately simple: **4× Teal Mask Ogerpon ex** (210 HP, Grass),
**18 Grass Energy**, and the rest draw and search. One attacker, one Energy
type, no evolution line.

It is chosen entirely to exploit one fact about the field.

**Weakness doubles damage.** And when you mine what people actually play:

- **Marnie's Grimmsnarl ex is 47% of the metagame** — two near-identical lists
  at 41.1% and 5.8% of all decks played. Its Pokémon are **weak to Grass**.
- **Cynthia's Garchomp ex**, another top-six deck, is weak to Grass on all four
  of its Pokémon.
- So roughly **half the field takes double damage** from our only attacker.
- Teal Mask Ogerpon ex is weak only to **Fire** — and no top-six deck plays Fire.
  Our one vulnerability is unexploited.

Measured: our agent piloting this deck beats Marnie's Grimmsnarl ex **92–8**
over 100 games, and scores **0.942** against the play-weighted field.

This is the whole thesis in one card. The field has concentrated a third of its
play into a deck that is *already losing* (46.4% win rate across 57,000 games)
**and** is weak to Grass — and almost nobody is punishing it.

The story of how we got here is the more useful part, because we got it wrong
twice first.

---

# How we got here (and the mistake worth reading)

### Step 1 — we read the metagame off the replays

Kaggle publishes every match played, as public datasets. Each replay contains
**both players' full 60-card decks and who won**. Episode card IDs join straight
to the card list, so replays decode into readable decklists.

We mined six days — **57,108 decks, 5,200 matches a day** — one day at a time
(each dump is 20 GB, so: download, aggregate, delete, next).

What that showed:

- Only **197 of 1,267 cards** are ever played. The real card pool is tiny.
- Only about **120 distinct decklists** exist. Everyone is copying everyone.
- **Marnie's Grimmsnarl ex is 34.5% of the entire metagame and wins 46.4%.**
  One deck in three is a *losing* deck. The field is badly misallocated.
- The best single list was a **Mega Lucario ex build winning 69.5% over 239
  games** — while the *average* Mega Lucario deck won 45.8%. The exact list
  matters far more than the archetype name.

### Step 2 — we copied the winning deck, and it lost badly

We took that 69.5% decklist, gave it to our agent, and played it against the
trivial starter deck that ships with the competition.

**It lost 32–88.**

A top player's decklist encodes *their program's* competence — evolution lines
timed right, Energy placed deliberately, Supporters sequenced. Hand it to a
simple agent and it is worse than a deck with no decisions in it, because every
card it misplays is a card the simple deck never asked it to play.

**Deck strength is not separable from the policy that plays it.**

### Step 3 — we optimised against the wrong opponent for hours

So we built a tournament: every candidate deck against every other, our agent
piloting both sides, 900 games. The starter deck won at 0.883. We tried a
purpose-built Fighting deck (lost 9–51), a version with more Basic Pokémon so we
would never run out (lost 40–160), and swept Energy counts (best result 39–41).

Every experiment agreed: **the starter deck is the best deck.** We submitted it.
It beat a random agent **96.7%** of the time.

It scored **445.8**.

### Step 4 — the benchmark was the problem

The competition ships an *official rule-based sample agent* — a genuinely
competent opponent. The moment we used it as the yardstick instead of our own
weak agent, **the ranking inverted**:

| Deck (our agent piloting) | vs our own weak agent | vs the reference agent |
|---|---|---|
| Starter deck (Mega Abomasnow ex) | **0.883** — tournament winner | **0.194** |
| Mega Lucario ex | 0.337 — last place | **0.276** |

Both right-hand figures are 500 games, so the gap is real. We switched decks,
and an early leaderboard reading of 724.9 seemed to confirm it.

### Step 5 — that was also wrong

The rating settled. Mega Lucario finished at **305.8**; the starter deck it
replaced was at **544.8**. The switch cost us ~240 points.

We had correctly spotted that our own weak agent was a bad yardstick — and then
replaced it with the **official reference agent, which plays a single
archetype**. Beating a Mega Lucario specialist measures one matchup, not a
field. *A single opponent cannot rank decks no matter how strong it is.*

### Step 6 — measure against the field you actually face

The fix was sitting in the mined data the whole time. `scripts/gauntlet.py`
builds the opponent pool from real decklists and weights each by **how often it
is genuinely played**:

| Deck | vs weak agent | vs reference | **field-weighted** | settled rating |
|---|---|---|---|---|
| Starter deck | 0.883 | 0.194 | **0.705** | **544.8** |
| Mega Lucario ex | 0.337 | 0.276 | **0.366** | **305.8** |
| **Teal Mask Ogerpon ex** | — | — | **0.942** | *pending* |

The field-weighted column is the only one that agrees with the leaderboard —
and it is the one that picked the deck described above.

The starter deck's 350 HP wall is unbeatable *by an opponent who cannot knock it
out*. Against one who can, its core problem shows: Mega Abomasnow ex is a Mega
ex, so **every knockout hands over three prizes**. Two knockouts and the game is
gone. Our weak benchmark could never punish that, so it recommended the worst
real deck we tested.

Two warning signs were visible the whole time and we ignored both:

- Our games ended on **turn 5**. Real replays run a median of 146 steps. A
  benchmark whose games finish in a third of the expected time is not measuring
  the same game.
- We treated "beats random 96.7%" as progress. It only measures the floor.

Switching decks on the better benchmark took us from 445.8 → **724.9**.

### Step 7 — the bug that cost two submissions

Our first two submissions failed with `Validation Episode failed` and no detail.
The cause is worth writing down:

> Kaggle **`exec()`s** your `main.py` rather than importing it, so **`__file__`
> is undefined**. Any use of it raises `NameError` and kills the episode.

Every local test passed, because a normal import always defines `__file__`. The
real fix was not the one-line code change — it was discovering that
`kaggle_environments` ships the competition's own `cabt` environment, so the
**actual validation episode can be run locally**. `build_submission.py` now
refuses to submit unless it passes.

---

# What we know is still wrong

Playing the *same* deck, the reference agent beats us roughly 2:1. The gap is
piloting, not cards. Comparing our choices against its choices state-by-state,
we disagree most on:

| Decision | How often we differ |
|---|---|
| Attaching to a target | 90% |
| Which card to play (main menu) | 40% |
| What to fetch when searching the deck | 41% |
| Which Pokémon to promote | 35% |

A first attempt at fixing these made things *worse*. Re-measuring the baseline
at 500 games put it at **0.276**, where 100-game runs had read anywhere from
0.25 to 0.33 — so several conclusions drawn earlier in the day were inside the
noise. Everything here is now measured at 500 games minimum, and that policy
work is parked until it can be judged properly.

**One caveat on that 2:1 figure.** Measured against the reference agent, the
search agent below looked marginal — 0.303 against the rule agent's 0.276,
inside the noise. Head to head on the same deck it wins **0.720 (288–112 over
400 games)**, roughly 10σ.

Both readings are correct, and the reconciliation is the third distinct way a
benchmark misled us in a day: when two candidates are both far weaker than a
common opponent, their gap *measured through that opponent* is compressed.
Compare candidates **directly**. The search work was nearly discarded on the
compressed reading.

---

# Lookahead: guessing the opponent's deck

The rules above are all *reflex* — they never ask "what happens if I do this?"
The engine can answer that: `search_begin` / `search_step` roll the game forward
from the current position.

The catch is that simulating requires the hidden information — the opponent's
deck, hand and prizes. Guessing 60 unknown cards should be hopeless.

**It isn't, because we mined the metagame.** Only ~120 distinct decklists exist,
and the 40 most common cover **94.4% of all observed play** — 9 KB of JSON. So
the agent:

1. Collects every opponent card it can see (Active, Bench, discard, attached
   Energy, evolution chains, Stadium).
2. Keeps only the known decklists **consistent** with that — a list is ruled out
   if we have seen more copies of a card than it runs.
3. Takes the most-played survivor and fills in the remainder as their deck,
   hand and prizes.

Our own hidden cards need no guessing at all: we know our 60-card list, and
everything except deck and prizes is visible to us, so we subtract.

In testing this found a consistent decklist on **30/30 attempts**, at a mean
confidence of 0.71 (the top candidate's share of the surviving plays).

Then, for each option on the main menu, the agent simulates taking it, plays the
rest of its own turn out with the rules above, and scores the position:

```
score = 1000 × (their prizes left − our prizes left)   # prizes are the win condition
      + (our board HP − their board HP)                # tie-break toward the next one
```

It is essentially free: `search_begin` plus branching every option costs **4.4 ms
per decision**, about 0.3 s across a whole episode against a 600 s budget. Any
failure anywhere falls straight back to the rules.

---

# How we decide whether a change is real

Every comparison plays the two agents on the **same deck**, swapping sides each
game so the first-player advantage cancels. What decides it is a **Sequential
Probability Ratio Test** (Wald, 1945 — the same machinery Stockfish's `fishtest`
uses to accept engine patches). It stops the moment the evidence crosses a
threshold instead of at an arbitrary game count.

This was not academic tidiness. Fixed-size samples cost us real rating: a 0.33
that was really 0.276, a "regression" that was inside the noise, and a search
agent nearly discarded because its edge had been measured through the wrong
opponent. 100-game runs carry about ±5%.

What SPRT bought, in one session:

| Change | Verdict | Games to decide |
|---|---|---|
| **Energy term in the evaluation** | **accept, 0.707** | **82** |
| Energy term, replicated on a fresh seed | **accept, 0.608** | 181 |
| Search on every sub-selection too | reject, 0.461 | 230 |
| Static correction for scaling damage | reject, 0.462 | 234 |
| 2-ply rollouts (simulate the reply) | reject, 0.467 | 259 |

Three plausible ideas killed for ~230 games each instead of 500+, and the one
that worked confirmed in 82.

**Why those three failed is the useful part.** All three replace a
state-dependent value with a fixed guess. A flat "assume scaling attacks hit 3×"
is wrong in both directions depending on the board. Searching sub-selections
fails because a card you fetch pays off after the rollout ends. And 2-ply
doubles your exposure to having guessed the opponent's deck wrong — you spend
the extra depth defending against a reply that may be fiction. Simulation beats
static guesses; more simulation on a shaky assumption does not.

Two habits that follow: an accept/reject at 5% error each way means roughly
**one decision in twenty is wrong**, so anything surprising gets re-tested on a
fresh seed before we build on it. And candidates are compared **directly**, never
through a stronger third agent — that compression is what nearly lost us the
search agent.

---

# Repo layout

| Path | What it is |
|---|---|
| `agent/main.py` | The submitted agent |
| `ptcg/data.py` | Card CSV → tidy `cards` + `effects` tables |
| `ptcg/episodes.py` | Replay JSON → decklists + outcomes |
| `ptcg/arena.py` | Plays two agents head-to-head locally |
| `ptcg/meta.py` | Per-day metagame aggregation |
| `scripts/mine_day.py` | Download a day of replays → aggregate → delete |
| `scripts/compare_agents.py` | A vs B, same deck, sides swapped (`--sprt`) |
| `scripts/gauntlet.py` | Decks vs the real field, weighted by play share |
| `ptcg/opponent.py` | Infers the opponent's decklist from what they've shown |
| `ptcg/sprt.py` | Wald's sequential test |
| `scripts/build_submission.py` | Bundle + validate + submit |
| `scripts/validate_submission.py` | Runs the real Kaggle validation episode |
| `EDA_FINDINGS.md` | Full analysis log, numbers and all |

## Setup

```bash
pip install kagglehub pandas numpy matplotlib kaggle-environments
export KAGGLE_API_TOKEN=KGAT_...
python scripts/download_data.py
```

Copy `sample_submission/` from the competition download into `engine/`. It is
**not** in this repo: the battle engine is licensed *competition use only, do
not redistribute*, and this repo is public. `engine/`, `build/`, all binaries,
the card CSVs and `agent/deck.csv` are gitignored for that reason.

**Provenance.** The files under `data/` and `agent/deck_priors.json` are
*aggregate statistics* — decklist frequencies and win counts — computed from the
public competition replay datasets published by Kaggle. No replay content, card
text, or engine code is redistributed here. The MIT `LICENSE` covers this
repository's own code only; Pokémon and all associated names are trademarks of
Nintendo/Creatures/GAME FREAK.

## Submitting

```bash
python scripts/build_submission.py --submit -m "note"
```

This will not send anything unless the local validation episode passes.

## Status

- [x] Metagame mined — 16 days, 146,376 decks, 204 distinct decklists
- [x] Local engine, self-play arena, and a field gauntlet weighted by real play
- [x] Opponent-deck inference from the mined lists (30/30, mean confidence 0.71)
- [x] Forward search via `search_begin` / `search_step`
- [x] SPRT so every change is judged on evidence, not on a game count
- [x] **Agent live at 656.4 — above median, ~rank 2905/6421**
- [ ] **Multi-determinization**: sample from the posterior over consistent
      decklists per rollout instead of taking the mode. The 2-ply rejection is
      evidence that determinization error is now the binding constraint, which
      makes this the right next move rather than more depth.
- [ ] **Confidence-gated depth**: 2-ply failed on average, which is consistent
      with it helping once deck inference is confident and hurting before that.
      Testable by splitting the result early-game vs late-game — no submission
      needed.
- [ ] **Matchup matrix over the top archetypes** (empirical game theory). Our
      deck is a best response to today's field, and the field moves — Marnie's
      Grimmsnarl ex went 64% → 30% in a week. This says whether the counter is
      robust or fragile.
- [ ] Strategy writeup (**draft only — not submitted**)

## Reading list behind the method

The approach has names in the literature, which the strategy report should cite:

- Our search is **Perfect Information Monte Carlo** (Ginsberg's GIB). Its known
  flaw is *strategy fusion* (Frank & Basin 1998); the standard fix is
  **Information Set MCTS** (Cowling, Powley & Whitehouse 2012) — sample a fresh
  determinization per iteration, which is the multi-determinization item above.
- Deck inference is an informal **Bayesian opponent model** (Albrecht & Stone
  2018); making the likelihood explicitly hypergeometric is the principled form.
- The field-weighted gauntlet computes a best response to an empirical strategy
  distribution — **Empirical Game-Theoretic Analysis** (Wellman 2006).
- Ranking agents through a single strong opponent misranks them; see
  **Balduzzi et al., "Re-evaluating Evaluation" (2018)**. We hit this exactly.
- **SPRT**: Wald (1945).

Deliberately *not* used: CFR and its descendants (DeepStack, Libratus, ReBeL)
are the theoretically correct tools for imperfect-information games, but the
state space is enormous, the deadline is real, and deck inference already
collapses the hidden information to near-nothing for 94% of the field.
