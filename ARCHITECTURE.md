# Architecture: three chunks, three seams

The system is three foundational modules. Each is useful alone, each talks to
the others only through small, explicit interfaces, and no chunk imports
another chunk's internals.

```
1. CREATION                 2. UTILIZATION              3. IDENTIFICATION
   ptcg/creation/              agent/main.py               ptcg/opponent.py
   what 60 cards?              how do we play them?        what are they playing?
        |                           |                           |
        |---- pilot interface ------|                           |
        |     callable(obs) -> [i]  |---- posterior interface --|
        |                           |     P(archetype | seen)   |
        |------------- deck interface: list of 60 card IDs -----|
```

## Chunk 1 — Deck creation (`ptcg/creation/`)

An island-model genetic algorithm over 60-card decks (archipelago topology:
mono-type source sets with explore/refine temperaments, dual-type mixer sets
downstream, a gem tri-type terminus; migration is a one-way DAG). Includes:

- `pool.py` — card pool facade over the engine's own card dump
  (self-generates into gitignored `data/engine_dump/` on first run)
- `validator.py` — hard legality (60 cards, 4-per-name, ACE SPEC, Basics)
  plus coherence heuristics (evolution closure, stage-gated energy
  consistency, line ratios, mulligan risk)
- `ga.py` — islands, template seeding, validator-driven repair,
  mutation/crossover, migration, per-era JSON checkpoints
- `harness.py` — fast local matches (seat-swapped; ~3 ms/game)
- `pilots.py` — a baseline greedy pilot so creation runs stand-alone

**Consumes:** a pilot (chunk 2) via `pilot_factory`; a fitness panel — use
`agent/deck_priors.json` (the mined field, play-weighted) via `panel_from`.
**Produces:** elite decks as JSON checkpoints (`runs/<id>/latest.json`).

## Chunk 2 — Deck utilization (`agent/`, `ptcg/arena.py`)

The policy that plays any given deck: rule ordering, prize-aware scoring,
energy-position term, engine-API forward search. Owned by the existing agent
work; creation treats it as a black box behind the pilot interface
`callable(obs_dict) -> list[int]`.

**Consumes:** a deck (chunk 1); an opponent posterior (chunk 3) to
determinize hidden zones in search.
**Produces:** decisions; also the fitness signal when piloting GA evaluations.

## Chunk 3 — Opponent strategy identification (`ptcg/opponent.py`)

From revealed cards to a belief over what the opponent is playing.
Current: consistency filtering over mined decklists, take the mode.
Direction: full Bayesian posterior (hypergeometric likelihood over known
lists), declared by an early turn, feeding (a) plan selection in chunk 2 and
(b) determinization sampling in its search.

**Consumes:** the log/board stream; the mined decklist library.
**Produces:** `P(archetype | observations)` plus a concrete decklist sample.

## Interfaces (the whole contract)

| Seam | Format |
|---|---|
| Deck | `list[int]`, 60 card IDs (validator-legal) |
| Pilot | `callable(obs_dict) -> list[int]`, never crashes, <30 s/move |
| Fitness panel | decklists + play weights (`deck_priors.json` schema) |
| Elite archive | run checkpoint JSON: `{islands: {label: {best, best_deck}}}` |
| Opponent belief | posterior over library entries + sampled hidden zones |

## Licensing rule (unchanged)

The battle engine, card CSVs, and anything derived from the engine's card
tables stay out of this public repo. `ptcg/creation/` regenerates its card
dump locally from the engine at first use; `engine/` is populated by copying
the competition `sample_submission/` and is gitignored.
