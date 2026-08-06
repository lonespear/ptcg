# EDA pass 1 — what the card pool is made of

Everything below is reproducible with `python scripts/run_eda.py`; figures land in
`figures/`. Numbers are from the EN card list (1,267 unique cards).

## The data, and three things that will bite you

The competition ships 8 files that are really 4 — each exists twice under a
spaced and an underscored name (`EN Card Data.csv` and `EN_Card_Data.csv` are
byte-identical). The two PDFs (305 MB combined) are card-image catalogs; the
CSVs are the machine-readable pool.

The CSV is **one row per effect**, not per card: 2,022 rows collapse to 1,267
cards, with the card's own attributes repeated across its attack/ability rows.
`ptcg/data.py` normalizes this into `cards` (1,267) and `effects` (1,811).

Three encoding traps, all handled in the loader:

1. **`n/a` retreat means free retreat, not missing.** 35 Pokémon carry it, and
   they are exactly the cards you would expect — Shaymin, Emolga, Tynamo,
   Jolteon ex, Tapu Koko ex. Reading it as null silently deletes the entire
   free-retreat tier, which is a real mobility archetype.
2. **Trainer rules text has no Move Name.** Keying effects on `Move Name` drops
   191 of 197 Trainer rows. Their text lives in `Effect Explanation` alone.
3. **`Pokémon Tool` matches the substring "Pokémon"** but is a Trainer subtype.
   A naive `str.contains("Pokémon")` misfiles 27 cards as Pokémon.

The file is UTF-8; the `PokÃ©mon` mojibake you may see is cp1252 decoding, not
corruption. A Japanese mirror (`JP_Card_Data.csv`) exists with identical schema.

## 1. The pool is overwhelmingly Basic Pokémon

| Stage / type | Cards |
|---|---|
| Basic Pokémon | 595 |
| Stage 1 Pokémon | 345 |
| Stage 2 Pokémon | 116 |
| Item | 77 |
| Supporter | 61 |
| Pokémon Tool | 27 |
| Stadium | 26 |
| Special / Basic Energy | 20 |

1,056 Pokémon, 191 Trainers, 20 Energy. **Stage 2 lines are only 116 cards** —
any Stage 2 strategy is drawing from a thin, well-defined slice, and pays two
turns of setup for it. ![](figures/01_pool_composition.png)

## 2. A second prize buys +180 HP

180 cards (14.2%) have a Rule Box (`Pokémon ex`, `Mega Pokémon ex`, `ACE SPEC`).

| | Count | Median HP | Max HP |
|---|---|---|---|
| Rule box (2+ prizes) | 151 | **270** | 380 |
| Single prize | 905 | **90** | 240 |

The exchange rate is the core deckbuilding decision: a rule-box attacker triples
your HP but halves the number of knockouts the opponent needs. A single-prize
deck needs six knockouts against it to lose; an all-ex deck needs three.
![](figures/02_hp_vs_prizes.png)

## 3. Efficiency rises with energy cost — setup time is the real currency

| Energy cost | Attacks | Median damage | Median damage/energy |
|---|---|---|---|
| 1 | 458 | 20 | 20.0 |
| 2 | 421 | 50 | 25.0 |
| 3 | 334 | 100 | 33.3 |
| 4 | 61 | 150 | 37.5 |
| 5 | 6 | 235 | 47.0 |

Damage per energy more than doubles from 1 to 5 energy, so the pool *rewards*
investment — but with one energy attachment per turn, a 3-energy attack is a
turn-3 play at the earliest. The tension the agent has to resolve is not "which
attack hits hardest" but "which attack hits hardest **by the turn it matters**."
![](figures/03_efficiency_frontier.png)

## 4. Damage per energy is a trap metric — 80% of the leaders have a drawback

This is the most actionable finding of the pass. Across all flat-damage attacks,
**15.6%** carry a drawback. Among the **top 40 by damage per energy, 32 (80%)**
do — a 5× enrichment.

The printed energy cost is not the real cost. Palafin ex's Giga Impact is 250
damage for one energy and cannot attack the following turn. Ceruledge's Infernal
Slash is 220 for one, and discards four Fire Energy from hand. Slaking ex swings
for 280 and discards its own energy.

| Hidden cost | Flat attacks |
|---|---|
| Self-damage | 52 |
| Self-discard energy | 49 |
| Cannot attack next turn | 45 |
| Conditional / does nothing | 22 |
| Requires hand resource | 4 |
| Coin flip | 4 |
| Self-status condition | 3 |
| Self-mill | 2 |
| Self-KO / discard | 1 |

Ranking attackers on raw damage/energy puts glass cannons on top. After removing
attacks with drawbacks, the honest leaders are Greninja ex (Shinobi Blade, 170
for one energy, and it *searches your deck*), Flygon ex, and Mega Lucario ex's
Aura Jab. ![](figures/03b_top_efficiency.png)

**Implication for the agent:** the state evaluator needs an *effective* cost that
prices the drawback — a lost turn is worth roughly one full attack of tempo, a
discarded hand card is worth its draw-equivalent. `effects.drawback` and
`effects.is_conditional` in the loader are the hooks for that.

## 5. Weakness is near-deterministic, and Fire is the best-rewarded attacking type

The weakness graph is almost a function of type, not a scatter: Grass→Fire,
Fire→Water, Water→Lightning, Lightning→Fighting, Metal→Fire, Fighting→Grass,
Psychic→Darkness. This makes type exposure *predictable* rather than a gamble.

| Attacking type | Pokémon weak to it | Attackers available | Exposure share |
|---|---|---|---|
| **Fire** | **220** | 103 | 21.5% |
| Fighting | 188 | 121 | 18.4% |
| Lightning | 155 | 76 | 15.2% |
| Grass | 141 | 157 | 13.8% |
| Water | 99 | 137 | 9.7% |
| Darkness | 91 | 116 | 8.9% |
| Metal | 86 | 69 | 8.4% |
| Psychic | 41 | 138 | 4.0% |
| Colorless | 0 | 104 | 0.0% |

Fire is the standout: the fewest-but-not-scarce attackers (103) hitting the most
targets (220). Psychic is the inverse trap — 138 attackers competing for 41
weak targets. **Nothing is weak to Colorless**, so Colorless attackers are pure
flexibility with zero weakness upside.
![](figures/04_weakness_matrix.png) ![](figures/05_weakness_exposure.png)

## 6. Trainers buy consistency, not damage

Keyword classification over 197 Trainer effect rows (approximate — regex, not
parsed rules):

| Effect | Cards |
|---|---|
| Search deck | 39 |
| Draw cards | 24 |
| Heal / remove damage | 13 |
| Switch / gust | 13 |
| Energy acceleration | 8 |
| Recover from discard | 6 |
| Opponent disruption | 4 |

Search and draw are a third of the Trainer pool. Disruption is nearly absent (4
cards), which suggests the metagame is a **race, not a control matchup** — you
win by executing your own plan faster, not by breaking the opponent's.
![](figures/06_trainer_taxonomy.png)

## 7. Mobility is cheap

Retreat cost 0: 35 cards · 1: 531 · 2: 308 · 3: 146 · 4: 36. Over half the pool
retreats for a single energy, so pivoting is usually affordable — which raises
the value of the 13 switch/gust Trainers as *forced* repositioning of the
opponent, not your own.
![](figures/07_retreat_cost.png)

## Where this points next

1. **Build the effective-cost model.** Price each drawback in tempo so attacker
   evaluation stops rewarding glass cannons.
2. **Decide the prize-trade axis first.** Single-prize vs rule-box is the highest
   branch in deck design and everything else follows from it.
3. **Fire or Fighting as the attacking type**, unless the deck concept needs
   Colorless flexibility. Psychic is supply-rich and reward-poor.
4. **Assume a race.** With four disruption cards in the pool, consistency
   (search/draw density) will beat interaction.

## Open blocker

The Simulation-category data is **not accessible** with the current credentials —
both `pokemon-tcg-ai-battle-challenge-simulation` and
`pokemon-tcg-ai-battle-challenge` return 403 ("make sure you are authenticated
and have accepted the competition rules"). Strategy-category entry *requires*
a Simulation-category submission, and the battle engine / API almost certainly
lives there. Accepting the rules on that competition page is the next step;
until then there is no simulator to test any of the above against.
