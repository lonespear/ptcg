# Seeded archipelago v4 — D46 large-scale overnight run (2026-08-07/08)

Driver `ptcg/creation/seeded.py` via `scripts/run_seeded.py`; launcher
`scripts/overnight_d46.sh`; keeper + stall watchdogs
`scripts/watchdog_d46.sh` (both inside tmux on each machine). All
D38-D41 rules stand (pilot split, migration every 10 top-2
explore->refine, 0.35 raw-WR floor/reseed, D18 termination pricing,
D11 diversity bonus at the screen tier). D46 sizing: explore pop 36,
refine pop 14, plateau patience 24 eras at +0.005 island fitness.

## Fitness panel (v3, D46)

`data/panel_ladder_v3.json` — 37 cells, all REAL ladder lists from the
autopsy (`scripts/build_panel_v3.py`, manifest
`data/analysis/PANEL_V3.md`). Cell weights are 700+-band encounter
shares, so the weighted panel WR IS the ladder objective; the old
uniform 27-cell `panel_lb.json` is superseded. 86.9% of panel weight is
specialist-piloted (grimmsnarl, codex_alakazam, archaludon, lucario,
garchomp externals). Includes AlphaStarmie's exact stall list (= the
ladder's modal Alakazam variant, cell #1 by weight within the
archetype) and our own shipped v4 Ogerpon list as an opponent cell.

## Chains, constraints, objectives, founders

| chain | constraint | objective | real pilot | founders |
|---|---|---|---|---|
| spec-Ogerpon | Grass purity, >=2x Teal Mask Ogerpon ex (D38a) | panel weights | jon | shipped v4 + v3 harvest elites + D17 sisters (4) |
| grimmsnarl-mirror | Darkness purity | 2/3 mass on Grimmsnarl cells, rest by panel weight | grimmsnarl ext | 0.691 laptop + 0.673 sebastian v3 Darkness elites + 5 autopsy field variants (7) |
| engine | total basic energy <= 10 (repair path), type free | UNIFORM cells (discovery) | jon | AlphaStarmie exact + energyless Lopunny priors + low-energy Fez priors (6) |
| archaludon | energy = union of founder bases (Fire+Metal) | panel weights | archaludon ext | top-5 autopsy Archaludon lists |
| counter-900 | unconstrained; retype mutation p=0.2 | 50% mass on 900+-band archetypes (Grimmsnarl + Alakazam + mono-Kanga), rest by panel weight, + neutral floor | jon | v3 kanga-counter harvest elites + 3 Fighting founder classes (5) |
| rainbow-Kanga | energy = union of founder bases | panel weights | kanga ext | 0.364 laptop v3 elite + Sebastian live-stop pop heads (5) |

Founder bank: `runs/seeded_d46/founders.json`
(`scripts/build_founders_d46.py`; Sebastian pop snapshot in
`seb_v3_rainbow_pops.json`).

Weighting decisions made here (not in the D46 consensus, surfaced for
ratification): with a representative panel, chains without a stated
target objective (spec-Ogerpon, archaludon, rainbow-Kanga) use the
panel's own weights — the old hand-set 55% Grimmsnarl mass on
spec-Ogerpon is superseded by the panel's real 34.6% Grimmsnarl share.
The 0.673 Sebastian elite named as a spec-Ogerpon seed is in fact a
mono-Darkness Grimmsnarl deck (seeded_v3_seb era_000); it seeds
grimmsnarl-mirror instead.

## Machine split

Migration is intra-process, so each machine hosts full explore+refine
pairs; the spec's "d refine on Sebastian / d explore on laptop" becomes
archaludon running FULLY on both machines under different RNG seeds.

| machine | chains | workers | seed | interpreter |
|---|---|---|---|---|
| sebastian (10-core Mini) | spec-Ogerpon, engine, counter-900, archaludon | 10 | 82 | ~/ptcg_fork/.venv/bin/python (3.13) |
| laptop | grimmsnarl-mirror, archaludon, rainbow-Kanga | 7 | 81 | miniforge3 python3.10 |

Per-era eval load: sebastian 4x50=200 screens / laptop 3x50=150 —
about 20 screens per worker on both.

## Runway

Launched ~22:00 ET 2026-08-07. GA budget ~10.5 h (to ~08:30 ET), wall
~11.2 h so the deep final eval (real pilots, 50-game blocks up to 400,
wall-checked) lands `final_eval.json` by ~09:00 ET on each machine.
