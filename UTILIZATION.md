# Chunk 2 — Deck utilization: the pilot contract

What deck creation (chunk 1) needs from the agent, and what it gives back.
The adapter `ptcg/creation/pilots.py::JonDayPilot` already wires the current
`agent/main.py` into GA evaluation; this file is the forward contract.

## Already satisfied (no action)

- `agent(obs_dict) -> list[int]`, never crashes, importable as a module
  (not exec-only), priors resolve next to the module.
- **Deck injection.** GA matches pit two arbitrary decks, so the self-model
  (`_MY_DECK`) cannot come from `deck.csv`. The harness calls
  `bind_deck(deck)` per game and the adapter re-points the module global on
  every call. Nothing to change unless `_MY_DECK` handling moves.
- **Search throttle.** Rules-only mode (~14 ms/game) is the GA inner-loop
  pilot; search mode (~0.25 s/game) is for elite gates, tournaments, and
  the ladder. The adapter toggles `SEARCH_ENABLED`.

## Requested (the chunk-2 work that unblocks the rest)

1. **Deck-general competence.** The GA will hand the pilot thousands of
   decks it has never seen, including evolution-heavy lines. Fitness
   verdicts are only as good as the pilot's ability to execute an alien
   deck's plan. Known soft spots worth attention: Rare Candy / evolution
   timing, bench development for setup decks, retreat judgment. A deck the
   pilot cannot play scores as a bad deck even if it is a good deck.

2. **Interpretability trace (Strategy track is 70% model).** An optional
   trace mode that records, per decision: the context, candidates
   considered, the chosen option, and the evaluation breakdown by named
   term (prizes / board HP / energy-in-play / hand). The report's evidence
   backbone is "here is a decision, here is the printed reasoning" — the
   rule policy already articulates its reasons; this just captures them.

3. **Weight vector as a parameter.** The eval constants (1000 prize,
   1 HP, 30 energy, 5 hand) become an injectable vector with these
   defaults. That single change lets the Trainer tune weights by self-play
   and lets plan modules apply per-archetype deltas (decisions D7/D10)
   with zero further coupling.

4. **Keep the `_predict_opponent` boundary.** Chunk 3's upgrade (posterior
   over consistent lists, hypergeometric likelihood, sampled per
   determinization instead of the mode) replaces exactly that function.
   Signature stays `(obs) -> (deck, hand, prize)`; multi-determinization
   would call it k times per decision.

## What creation returns

- Elite decks as checkpoint JSON (`runs/<id>/latest.json`), validator
  reports attached.
- Matchup grids vs the mined field (the empirical-game-theory matrix on
  the roadmap) — `analysis/tournament.py` machinery in the PTCG_AI repo,
  moving here.
- Pilot-sensitivity flags: any elite whose fitness swings hard between
  rules mode and search mode is pilot-fragile, and the report should know.
