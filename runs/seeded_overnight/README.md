# Seeded archipelago v3 — overnight deck search

Run of `scripts/run_seeded.py` (driver `ptcg/creation/seeded.py`),
decisions D38/D38a/D39/D40/D41. Five chains, each an explore + refine
island pair; no Phase B — the whole budget funds the chains plus a deep
final evaluation on reclaimed time. Fitness panel: `data/panel_lb.json`,
the 27-cell stratified leaderboard panel (manifest
`data/analysis/PANEL_LB.md`), uniform cell weights — proportionality
lives in the panel's composition.

## Chains and founders

| chain | energy constraint | objective (cell weights) | founders (exact, head of pop) |
|---|---|---|---|
| spec-Ogerpon | Grass purity; hard anchor >=2x Teal Mask Ogerpon ex in the repair path (D38a) | 55% mass on the 12 Grimmsnarl cells, 45% on the rest, + neutral floor | Jon's shipped list (agent/deck.csv) + 2 D17 sister lists |
| mono-Darkness | Darkness purity | 2/3 mass on the 12 Grimmsnarl cells (the mirror), 1/3 on the rest | 2 field Grimmsnarl lists (p=48306, p=10346), 2 field Fezandipiti lists, constructed 4x Grimmsnarl template |
| mono-Fighting | Fighting purity | uniform (discovery wildcard) | 2 sprint Fighting elites (deep 0.099/0.080), field Garchomp + Mega Lucario lists, constructed Cornerstone template |
| rainbow-Kanga | Grass+Water+Lightning+Psychic+Fighting (union of founder energy bases); energy mutations re-mix the base | uniform | the field's rainbow Mega Kangaskhan list (p=2645, 5 energy types, Crispin/Energy Switch glue) + a uniform energy-rebalance sister |
| kanga-counter | unconstrained — retype mutation (p=0.2) swaps the whole base to a random type and pulls in its attacker lines | 50% mass on the 9 Colorless-threat cells (Mega Kangaskhan, Mega Lopunny, 7x Dudunsparce), 50% on the rest, + neutral floor | the Fighting founder classes (the likely convergence point; the GA may leave) |

After the exact founders, half of each island's remaining slots are
mutated founder variants ("heuristic variants"); the rest are
constraint-satisfying random templates. Below-floor plateau reseeds
from these same classes.

## Neutral-matchup floor (D39, documented choice)

Floored chains (spec-Ogerpon, kanga-counter) pay
`2.0 * max(0, 0.15 - mean_WR_over_non_target_cells)`. A candidate that
zeroes its neutral game pays -0.30 fitness — more than any target-cell
gain can buy back — so a deck cannot win selection on target-cell gains
alone; holding >= 0.15 on neutral cells costs nothing. Target mass
already prices the target cells; the floor only vetoes neutral collapse.

## Pilot split (D40)

| eval | what it decides | candidate pilot | panel side | games/opp |
|---|---|---|---|---|
| screen — every member, every era | which members reach the real tier (with the D11 diversity bonus on explore) | greedy | specialists, except the 10 codex_alakazam cells demoted to the generalist (cost, see below) | 8 |
| real tier — screen-top 6 explore / top 4 refine | elites, refine parent pool, migrants, plateau, below-floor rule | per chain: spec-Ogerpon jon, mono-Darkness grimmsnarl (external), mono-Fighting jon, rainbow-Kanga kanga (external), kanga-counter jon | full specialist panel; codex cells at 3 games | 6 |
| era gate — on elite change | reporting only | jon both sides vs previous elite | — | 500 |
| deep final eval — top 3 per chain | final_eval.json ranking | the chain's real pilot | full specialist panel; codex cells at games/4 | 100 per round, up to 400, wall-checked |

Selection noise beats selection blindness: the real tier plays fewer
games than the screen but every selection-deciding number comes from it.

### Rainbow pilot caveat (flagged per D40)

No competent toolbox pilot exists tonight. Audition on the rainbow
founder vs three panel cells (24 games each, 2026-08-07): jon 0.069,
greedy 0.208, kanga external 0.181. The v2 jon machinery actively
misplays the toolbox; the kanga external at least knows the Mega
Kangaskhan + Crispin core, so it takes the rainbow real tier. The
rainbow chain's real tier is the weakest pilot link in the run.

### Screen-tier panel demotion (operational fallback, flagged)

Panel-side, 10 of 27 cells (8 Dudunsparce, 2 Fezandipiti) matched the
codex_alakazam specialist at ~1-3 s/game (stall games run long); a full
screen eval would have cost ~134 s/deck and collapsed the night to ~25
eras. The screen tier therefore plays those 10 cells with the fast
generalist (~17 s/deck); the real tier and deep eval keep the full
specialist panel (codex cells 3 games real, games/4 deep). Every
selection-deciding eval still sees the specialist panel.

## Hyperparameters (ratified)

Explore pop 24, refine pop 10. Explore selection: rank tournament size
3 over the full population (real-tier members ranked first); refine:
truncation top-4 parent pool + 2-elite preservation. Migration every
10 eras, top 2 by real fitness, explore -> refine within chain, into
the refine island's screen-worst slots. Plateau patience 12 eras at
+0.01 (island fitness, real tier), armed only after the chain's first
migration wave; a plateau whose best raw panel WR (uniform mean, real
tier) is under 0.35 reseeds from founder classes instead of freezing.
D18 termination pricing stays: fitness -= 0.15 * exhaustion share of
losses, per tier. Launch: seed 74, 7 workers, GA 10.0 h, wall 11.5 h.

## Files

- `state.json` / `archive.jsonl` — resumable state; archive records are
  `{"pk": pilot_kind, "k": deck, "pr": profile, "x": frag}` (profile =
  27 per-cell win rates; island fitness is recomputed from it, so one
  eval serves every weighting).
- `era_NNN.json` / `latest.json` — per-era checkpoints. The global
  "elite" is ranked by uniform panel WR (real tier) because island
  fitnesses are weighted differently.
- `final_eval.json` — deep final evaluation, per-opponent W/L +
  termination reasons under the chains' real pilots.
- `founders.json` / `founders_manifest.json` — the founder bank
  (`scripts/build_founders.py`).
