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
- **Search throttle.** Rules-only mode (~16 ms/game) is the GA inner-loop
  pilot; search mode (~3.7 s/game at 3 determinizations and 2-ply) is for
  elite gates, tournaments, and the ladder. The adapter toggles
  `SEARCH_ENABLED`.
- **Weight vector.** `agent.main.WEIGHTS` holds every eval constant
  (`prize`, `hp`, `energy`, `hand`, `no_active`, `search_margin`);
  `set_weights(dict)` re-points it, `set_weights(None)` restores defaults,
  and partial or malformed vectors degrade rather than raise.
  `JonDayPilot(weights=...)` re-points per call, like `_MY_DECK`.
- **Determinization hook.** `_predict_opponent(obs, rng=None)` and
  `_own_hidden(obs, rng=None)` shuffle the hidden pool before splitting it,
  so calling them k times per decision yields k distinct worlds. Chunk 3's
  posterior replaces the body of `_predict_opponent`; the k-sample call site
  in `_search_main` already exists.

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

3. **Tune the weights.** The vector is injectable; nothing has tuned it.
   `search_margin` is the one term with a measured optimum so far (0, not
   the half-prize hysteresis a stronger heuristic would want).

4. **Posterior over opponent lists.** Chunk 3 replaces the body of
   `_predict_opponent` — hypergeometric likelihood over consistent lists
   instead of the single most-played mode, which is what the shuffle
   currently stands in for.

## What creation returns

- Elite decks as checkpoint JSON (`runs/<id>/latest.json`), validator
  reports attached.
- Matchup grids vs the mined field (the empirical-game-theory matrix on
  the roadmap) — `analysis/tournament.py` machinery in the PTCG_AI repo,
  moving here.
- Pilot-sensitivity flags: any elite whose fitness swings hard between
  rules mode and search mode is pilot-fragile, and the report should know.
