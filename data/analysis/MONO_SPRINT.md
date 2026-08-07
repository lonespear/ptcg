# Mono sprint — energy-typing robustness vs the field

Run `runs/mono_sprint/` (seed 73, 7 workers, pop 10, 24 games/opponent GA fitness, plateau patience 12 eras at +0.01). 58 GA eras completed, 1.38h total. Fitness: play-weighted win rate vs the specialist-piloted top-8 field panel minus the D18 termination penalty (0.15 x exhaustion share of losses). Deep eval: elites replayed vs the full panel for tighter CIs; games per opponent as listed (codex-piloted Fezandipiti at a quarter rate, as in GA fitness).

**Read this first.** The sprint answered its question — which energy typings
have measured legs against this field — and its elites are **seed material,
not finished decks**: the best deep win rate on the panel is 0.153 (Grass),
far below shipping strength. The structural findings (1-prize-heavy bodies
everywhere, Fighting owning the field's tail, nobody touching the Grimmsnarl
cells, wins arriving mostly by opponent deck-out) fed the D38 seeded-
archipelago redesign that replaces tonight's planned run.

## Why these four types (selection evidence, measured)

Launched with Grass, Metal, Water, Fighting; Water was swapped for Fire
mid-run by ratified decision (below).

- **Grass** — 41.5% of panel-weighted Pokemon slots are Grass-weak (both
  Grimmsnarl lists including the ex line, plus the whole Garchomp list);
  Grimmsnarl alone is 34-47% of the ladder field. In the 36-cell matchup
  matrix the only field deck with winning records into both Grimmsnarl lists
  is the Grass Ogerpon list (62.2%/63.6%). mono-Grass finished 2nd of 8 in
  archi_r0p Phase A (0.226).
- **Metal** — best measured mono set in archi_r0p at Phase A end (0.227,
  the overall elite from era 16); era-0 elite of the prep smoke test; 15.1%
  weighted weak-slot coverage (Froslass/Snorunt in every Grimmsnarl list).
- **Water** — top refine island through most of archi_r0p Phase A (0.220
  from era 6); Water genes fed archi_r0p's overall Phase-B winner
  (dual-Water+Lightning 0.232).
- **Fighting** — the prep smoke test's Phase-A elite was mono-Fighting
  (0.138-0.159, eras 1-5); Fighting 2x's the field's 1-prize wall that beat
  v2 hardest on the real ladder (Dudunsparce, 1-5 in the autopsy) plus
  Fezandipiti ex, Mega Kangaskhan ex and Mega Lopunny ex.
- Passed over: **Darkness** (2nd-best weakness leverage at 25.5% but worst
  measured island fitness, 0.153, in archi_r0p), **Lightning** (0.199 but
  0.3% weakness coverage), **Psychic** (0.154).

## Mid-run swap: Water out, Fire in (ratified, era 14)

Water's counter target — a Fire meta — is 0% of this field, while Fire is
the counter-counter: it 2x's the Grass decks the field will bring to hunt
Grimmsnarl, and nothing in the current field 2x's Fire back. Water was
retired with its data intact; its rows below are **truncated at era 14**
(the "plateau" tag in the termination table is the retire mechanism, not a
real plateau). Fire started from fresh templates at era 15 and ran ~42 eras
where Grass/Metal/Fighting effectively got their full arcs — its numbers
sit on less wall-clock and must not be compared head-to-head on fitness
alone (it did pass Fighting's frozen best mid-run before its own plateau).

## Termination paths

| island | path | last improvement (era) | set best |
|---|---|---|---|
| mono-Fighting | plateau | 9 | 0.114 |
| mono-Fire | plateau | 45 | 0.108 |
| mono-Grass | plateau | 1 | 0.178 |
| mono-Metal | plateau | 6 | 0.130 |
| mono-Water | plateau | 9 | 0.087 |

## Typing-robustness table (island elites, deep eval)

| island | GA fit | deep WR (95% CI) | decided g | profile min/median/max | exh. loss share | bodies | 1-prize | 2-prize | 3-prize | prizes/body |
|---|---|---|---|---|---|---|---|---|---|---|
| mono-Fighting | 0.110 | 0.099 +/- 0.011 | 2898 | 0.01/0.58/0.89 | 0.13 | 27 | 23 | 4 | 0 | 1.15 |
| mono-Fire | 0.101 | 0.062 +/- 0.011 | 2900 | 0.00/0.06/0.75 | 0.12 | 31 | 24 | 7 | 0 | 1.23 |
| mono-Grass | 0.178 | 0.153 +/- 0.022 | 2900 | 0.06/0.21/0.67 | 0.34 | 19 | 12 | 3 | 4 | 1.58 |
| mono-Metal | 0.130 | 0.080 +/- 0.015 | 2900 | 0.01/0.17/0.58 | 0.24 | 21 | 21 | 0 | 0 | 1.0 |
| mono-Water | 0.092 | 0.064 +/- 0.012 | 2900 | 0.00/0.16/0.74 | 0.30 | 26 | 24 | 2 | 0 | 1.08 |

### Per-opponent deep win rates (island elites)

| island | Marnie's Grimmsnarl ex (0.55) | Fezandipiti ex (0.16) | Marnie's Grimmsnarl ex (0.12) | Cynthia's Garchomp ex (0.06) | Mega Kangaskhan ex (0.03) | Teal Mask Ogerpon ex (0.03) | Fezandipiti ex (0.02) | Mega Lopunny ex (0.02) |
|---|---|---|---|---|---|---|---|---|
| mono-Fighting | 0.02 | 0.06 | 0.01 | 0.01 | 0.89 | 0.72 | 0.81 | 0.58 |
| mono-Fire | 0.02 | 0.06 | 0.02 | 0.01 | 0.75 | 0.00 | 0.32 | 0.47 |
| mono-Grass | 0.12 | 0.21 | 0.07 | 0.07 | 0.67 | 0.06 | 0.62 | 0.31 |
| mono-Metal | 0.02 | 0.17 | 0.02 | 0.01 | 0.58 | 0.01 | 0.39 | 0.46 |
| mono-Water | 0.00 | 0.16 | 0.01 | 0.01 | 0.74 | 0.01 | 0.28 | 0.26 |

### Termination-mode distribution (island elites, deep eval)

| island | wins by | losses by |
|---|---|---|
| mono-Fighting | deck-out 75%, prizes taken 21%, no active Pokemon 4% | prizes taken 87%, no active Pokemon 12%, deck-out 1% |
| mono-Fire | deck-out 83%, prizes taken 10%, no active Pokemon 7% | prizes taken 88%, no active Pokemon 11%, deck-out 0% |
| mono-Grass | deck-out 43%, prizes taken 41%, no active Pokemon 16% | prizes taken 66%, no active Pokemon 34%, deck-out 0% |
| mono-Metal | deck-out 92%, prizes taken 4%, no active Pokemon 4% | prizes taken 76%, no active Pokemon 16%, deck-out 7% |
| mono-Water | deck-out 72%, prizes taken 24%, no active Pokemon 3% | prizes taken 70%, no active Pokemon 29%, deck-out 1% |

## Elite trajectories (set best-so-far by era)

- **mono-Fighting**: e0:-0.003 -> e1:-0.003 -> e2:0.027 -> e3:0.042 -> e4:0.042 -> e5:0.067 -> e6:0.067 -> e7:0.067 -> e8:0.077 -> e9:0.114 -> e10:0.114 -> e11:0.114 -> e12:0.116 -> e13:0.116 -> e14:0.116 -> e15:0.116 -> e16:0.116 -> e17:0.116 -> e18:0.116 -> e19:0.116 -> e20:0.116 -> e21:0.116
- **mono-Fire**: e15:0.020 -> e18:0.039 -> e21:0.039 -> e24:0.040 -> e27:0.040 -> e30:0.070 -> e33:0.070 -> e36:0.077 -> e39:0.096 -> e42:0.096 -> e45:0.108 -> e48:0.108 -> e51:0.108 -> e54:0.108 -> e57:0.108
- **mono-Grass**: e0:0.085 -> e1:0.178 -> e2:0.178 -> e3:0.178 -> e4:0.178 -> e5:0.178 -> e6:0.178 -> e7:0.178 -> e8:0.178 -> e9:0.178 -> e10:0.178 -> e11:0.178 -> e12:0.178 -> e13:0.178
- **mono-Metal**: e0:0.065 -> e1:0.065 -> e2:0.065 -> e3:0.065 -> e4:0.065 -> e5:0.088 -> e6:0.130 -> e7:0.130 -> e8:0.130 -> e9:0.130 -> e10:0.130 -> e11:0.130 -> e12:0.130 -> e13:0.130 -> e14:0.130 -> e15:0.130 -> e16:0.130 -> e17:0.130 -> e18:0.130
- **mono-Water**: e0:-0.001 -> e1:-0.001 -> e2:0.040 -> e3:0.052 -> e4:0.052 -> e5:0.060 -> e6:0.060 -> e7:0.067 -> e8:0.067 -> e9:0.087 -> e10:0.087 -> e11:0.087 -> e12:0.087 -> e13:0.087 -> e14:0.094

## Top decks per island

### mono-Fighting

**#1** (mono-Fighting/explore) GA 0.110, deep 0.099 +/- 0.011 (2898 decided); 27 bodies: 23x1-prize, 4x2-prize, 0x3-prize

- Pokemon: 4x Cornerstone Mask Ogerpon ex, 3x Larry's Starly, 2x Okidogi, 2x Buneary, 2x Nymble, 2x Steven's Baltoy, 2x Munna, 1x Lokix, 1x Landorus, 1x Deino, 1x Lopunny, 1x Scorbunny, 1x Swablu, 1x Magearna, 1x Bouffalant, 1x Yungoos, 1x Sprigatito
- Item: 3x Antique Plume Fossil, 1x Potion, 1x Fighting Gong, 1x Rare Candy, 1x Scramble Switch
- Supporter: 1x Lt. Surge's Bargain, 1x Janine’s Secret Art, 1x Anthea & Concordia, 1x Team Rocket's Ariana
- Stadium: 2x Lumiose City, 1x Dizzying Valley, 1x Risky Ruins, 1x Forest of Vitality, 1x Spikemuth Gym
- Energy: 16x Basic {F} Energy

**#2** (mono-Fighting/refine) GA 0.109, deep 0.080 +/- 0.011 (2898 decided); 19 bodies: 18x1-prize, 1x2-prize, 0x3-prize

- Pokemon: 3x Landorus, 2x Lokix, 2x Larry's Starly, 2x Munna, 1x Okidogi, 1x Regigigas, 1x Nymble, 1x Flutter Mane, 1x Cornerstone Mask Ogerpon ex, 1x Magearna, 1x Tympole, 1x Steven's Baltoy, 1x Wailmer, 1x Skitty
- Item: 2x Antique Plume Fossil, 2x Boxed Order, 1x Fighting Gong, 1x Brilliant Blender, 1x Rare Candy, 1x Hole-Digging Shovel
- Supporter: 1x Janine’s Secret Art
- Stadium: 3x Risky Ruins, 3x Lumiose City, 2x Spikemuth Gym, 2x Forest of Vitality
- Energy: 22x Basic {F} Energy

**#3** (mono-Fighting/refine) GA 0.032, deep 0.069 +/- 0.011 (2900 decided); 18 bodies: 16x1-prize, 2x2-prize, 0x3-prize

- Pokemon: 3x Pansear, 3x Cornerstone Mask Ogerpon, 2x Cynthia's Spiritomb, 2x Chansey, 2x Blissey ex, 1x Mienfoo, 1x Simisear, 1x Onix, 1x Rockruff, 1x Carbink, 1x Hop’s Silicobra
- Item: 2x Antique Plume Fossil, 2x Boxed Order, 1x Fighting Gong, 1x Brilliant Blender, 1x Rare Candy, 1x Hole-Digging Shovel
- Supporter: 1x Janine’s Secret Art
- Stadium: 3x Risky Ruins, 3x Lumiose City, 2x Spikemuth Gym, 2x Forest of Vitality
- Energy: 23x Basic {F} Energy

### mono-Fire

**#1** (mono-Fire/explore) GA 0.101, deep 0.062 +/- 0.011 (2900 decided); 31 bodies: 24x1-prize, 7x2-prize, 0x3-prize

- Pokemon: 4x Wellspring Mask Ogerpon ex, 3x Cynthia's Roselia, 3x Team Rocket's Houndour, 3x Wash Rotom, 2x N’s Darumaka, 2x Goldeen, 2x Regirock ex, 2x Rowlet, 1x Onix, 1x Taillow, 1x Larry's Dunsparce, 1x Vanillite, 1x Magmar, 1x Hearthflame Mask Ogerpon ex, 1x Darumaka, 1x Fidough, 1x Regigigas, 1x Scorbunny
- Item: 1x Antique Jaw Fossil, 1x Glass Trumpet, 1x Reboot Pod
- Supporter: 1x Fennel, 1x Crispin, 1x Brock’s Scouting, 1x Team Rocket's Ariana
- Stadium: 1x Area Zero Underdepths
- Energy: 21x Basic {R} Energy

**#2** (mono-Fire/refine) GA 0.096, deep 0.062 +/- 0.010 (2900 decided); 26 bodies: 18x1-prize, 8x2-prize, 0x3-prize

- Pokemon: 4x Wellspring Mask Ogerpon ex, 3x Cynthia's Roselia, 3x Team Rocket's Houndour, 2x N’s Darumaka, 2x Panpour, 2x Regirock ex, 2x Larry's Dunsparce, 1x Eevee, 1x Goldeen, 1x Tympole, 1x Spritzee, 1x Wash Rotom, 1x Vanillite, 1x Hearthflame Mask Ogerpon ex, 1x Gouging Fire ex
- Item: 1x Antique Sail Fossil
- Tool: 1x Payapa Berry
- Supporter: 1x Carmine, 1x Brock’s Scouting, 1x Cook, 1x Boss’s Orders, 1x Judge
- Stadium: 1x Team Rocket's Watchtower, 1x Area Zero Underdepths
- Energy: 25x Basic {R} Energy

### mono-Grass

**#1** (mono-Grass/explore) GA 0.178, deep 0.153 +/- 0.022 (2900 decided); 19 bodies: 12x1-prize, 3x2-prize, 4x3-prize

- Pokemon: 4x Mega Heracross ex, 4x Golurk, 3x Golett, 3x Zangoose ex, 2x Virizion, 1x Spearow, 1x Hoothoot, 1x Bunnelby
- Item: 3x Antique Cover Fossil, 2x Love Ball, 2x Poké Pad, 2x Deduction Kit, 1x Scoop Up Cyclone
- Supporter: 3x Jacinthe, 1x Amarys
- Stadium: 4x Festival Grounds, 3x Dizzying Valley, 2x Area Zero Underdepths
- Energy: 18x Basic {G} Energy

**#2** (mono-Grass/refine) GA 0.038, deep 0.065 +/- 0.011 (2900 decided); 19 bodies: 18x1-prize, 1x2-prize, 0x3-prize

- Pokemon: 4x Fidough, 3x Bulbasaur, 3x Team Rocket's Porygon, 2x Larry's Rufflet, 2x Farfetch'd, 1x Erika's Oddish, 1x Dachsbun ex, 1x Fan Rotom, 1x Tangela, 1x Erika's Gloom
- Item: 2x Poké Pad, 2x Call Bell, 2x Blowtorch, 2x Rare Candy, 1x Max Rod
- Tool: 3x Hop’s Choice Band
- Supporter: 3x Crispin, 2x Perrin, 1x Carmine
- Energy: 23x Basic {G} Energy

### mono-Metal

**#1** (mono-Metal/explore) GA 0.130, deep 0.080 +/- 0.015 (2900 decided); 21 bodies: 21x1-prize, 0x2-prize, 0x3-prize

- Pokemon: 4x Zamazenta, 4x Tynamo, 4x Lillie’s Comfey, 3x Dialga, 1x Ralts, 1x Nacli, 1x Chi-Yu, 1x Hop’s Rookidee, 1x Helioptile, 1x Fidough
- Item: 3x Energy Swatter, 1x Scoop Up Cyclone, 1x Antique Sail Fossil, 1x Antique Plume Fossil
- Tool: 2x Air Balloon, 2x Counter Gain, 2x Light Ball
- Supporter: 4x Explorer’s Guidance, 1x Amarys
- Stadium: 1x Postwick
- Energy: 21x Basic {M} Energy

**#2** (mono-Metal/refine) GA 0.088, deep 0.055 +/- 0.007 (2900 decided); 17 bodies: 13x1-prize, 4x2-prize, 0x3-prize

- Pokemon: 4x LugiaEX, 3x Zamazenta, 3x Steelix, 3x Mow Rotom, 2x Cornerstone Mask Ogerpon, 2x Onix
- Item: 3x Hand Trimmer, 2x Wondrous Patch, 2x Love Ball
- Tool: 3x Lucky Helmet
- Supporter: 3x Emcee's Hype, 3x Firebreather, 2x Amarys, 2x Brock’s Scouting, 2x Drayton
- Stadium: 1x Festival Grounds
- Energy: 20x Basic {M} Energy

**#3** (mono-Metal/refine) GA 0.076, deep 0.083 +/- 0.013 (2900 decided); 17 bodies: 13x1-prize, 4x2-prize, 0x3-prize

- Pokemon: 4x LugiaEX, 3x Zamazenta, 3x Steelix, 3x Mow Rotom, 2x Cornerstone Mask Ogerpon, 2x Onix
- Item: 3x Hand Trimmer, 2x Wondrous Patch, 2x Love Ball, 1x N’s PP Up
- Tool: 2x Lucky Helmet
- Supporter: 3x Amarys, 3x Emcee's Hype, 3x Firebreather, 3x Drayton, 1x Brock’s Scouting
- Stadium: 1x Festival Grounds
- Energy: 19x Basic {M} Energy

### mono-Water

**#1** (mono-Water/explore) GA 0.092, deep 0.064 +/- 0.012 (2900 decided); 26 bodies: 24x1-prize, 2x2-prize, 0x3-prize

- Pokemon: 4x Spearow, 3x Yungoos, 3x Meditite, 3x Snorunt, 2x Buneary, 2x Misty's Staryu, 1x Mawile, 1x Koraidon, 1x Froslass, 1x Bloodmoon Ursaluna ex, 1x N’s Joltik, 1x Vanillite, 1x Registeel ex, 1x Swinub, 1x Larry's Starly
- Item: 1x Antique Cover Fossil, 1x Call Bell
- Tool: 1x Brave Bangle
- Supporter: 2x Colress’s Tenacity, 2x Morty’s Conviction, 1x Team Rocket's Archer, 1x Lana’s Aid, 1x Team Rocket's Ariana, 1x Cassiopeia
- Stadium: 1x Forest of Vitality
- Energy: 22x Basic {W} Energy

**#2** (mono-Water/refine) GA 0.022, deep 0.072 +/- 0.011 (2900 decided); 23 bodies: 19x1-prize, 2x2-prize, 2x3-prize

- Pokemon: 4x Elgyem, 4x Shelmet, 4x Snorunt, 3x Smoliv, 3x Dolliv, 2x Mega Froslass ex, 2x Arboliva ex, 1x Cubchoo
- Item: 2x Enhanced Hammer, 2x Glass Trumpet, 1x Deduction Kit, 1x Max Rod
- Tool: 3x Core Memory
- Supporter: 3x Lucian, 2x Drayton, 2x Cassiopeia, 1x Lisia’s Appeal, 1x Judge
- Energy: 19x Basic {W} Energy

**#3** (mono-Water/refine) GA 0.022, deep 0.075 +/- 0.011 (2900 decided); 23 bodies: 19x1-prize, 2x2-prize, 2x3-prize

- Pokemon: 4x Elgyem, 4x Shelmet, 4x Snorunt, 3x Smoliv, 3x Dolliv, 2x Mega Froslass ex, 2x Arboliva ex, 1x Cubchoo
- Item: 2x Deduction Kit, 2x Enhanced Hammer, 2x Glass Trumpet, 1x Max Rod
- Tool: 3x Core Memory
- Supporter: 3x Cassiopeia, 3x Lucian, 2x Drayton, 1x Lisia’s Appeal, 1x Judge
- Energy: 17x Basic {W} Energy


## What the sprint measured (findings)

1. **No mono island touches the meta cells.** Best vs the 0.55-weight
   Grimmsnarl list is 0.12 (Grass); vs the specialist-piloted Fezandipiti
   list 0.21 (Grass). The weakness prior does not convert without deck
   quality — the Grass elite holds 2x weakness on Grimmsnarl's whole line
   and still loses 88% of those games.
2. **Fighting owns the field's tail.** Its elite beats the four
   generalist-piloted lists (Kangaskhan 0.89, Ogerpon 0.72, Fez-Grass 0.81,
   Lopunny 0.58) — the widest profile spread of any island (0.01-0.89) —
   while scoring 0.01-0.06 on the specialist cells. Typing robustness and
   meta robustness are different axes.
3. **The GA used its denomination freedom.** Every elite is 1-prize-heavy
   (19-31 bodies, prizes/body 1.0-1.58, vs the shipped deck's 4 bodies at
   2.0) — the exact structural answer the v2 ladder autopsy called for.
4. **Wins arrive by opponent deck-out** (43-92% of elite wins) — the D18
   penalty prices our own exhaustion losses, so the search drifted to
   stall/mill shapes that beat generalist pilots and nothing else. A
   fitness panel this top-heavy plus termination-mode pricing on one side
   only makes mill the local optimum from random starts.
5. **Random-template init + patience 12 froze junk.** Grass froze at era
   13 on an era-1 template draw; migration (every 20) never fired before
   any freeze. The patience clock must not be shorter than the migration
   interval — this is D38's arming rule.

## Recommendation for tonight (superseded into D38, ratified)

The redesigned seeded archipelago replaces the planned lineage-B run.
Sprint output enters it as **founder material**, concretely:

- `scripts/build_founders.py` writes `runs/seeded_overnight/founders.json`:
  per island, 1-2 sprint elites (by deep WR) + the field's best lists of
  that type + a constructed 4x-attacker/19-energy/consensus-engine
  template; `--founders` seeds them at the head of each island population
  (exact where purity floors allow; `founders_manifest.json` records
  classes).
- `--preload-archive runs/mono_sprint/archive.jsonl` warms the fitness
  cache (1,880 evals) — measured effect in the path test: era-0 elite
  0.221 seeded vs 0.006 cold. Note the caches are only valid while the
  fitness panel is unchanged; tonight's stratified-panel run re-scores.
- Island roster per D38: Grass, Fire, Fighting, Darkness, spec-Ogerpon.
  Metal is out (worst weakness leverage of the sprint's four; its elite
  stays available in `final_eval.json` as seed material). Water is out
  (retired mid-sprint; no Fire meta to counter).
