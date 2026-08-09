# Ladder Autopsy v2 — what actually kills the submitted agent

**Question.** Rank which missing concept costs v2 the most real ladder games,
before building anything new. Candidate families: (1) prize-trade economics
(denomination blindness), (2) lethal exactness (N_DET=3 rollouts missing forced
prize sequences either way).

**Data.** All 53 completed real ladder episodes of submission 55311822 ("v2",
team Lemmes Yad) through 2026-08-07 11:41 UTC, pulled from the public Kaggle
episode API (the one same-submission validation episode excluded). Record
**27W–26L**; rating 600 → 791.4 over the day (the ~817 figure is the team's
earlier peak). Every replay parsed end to end; zero parse errors. Classifier:
`scripts/autopsy_v2.py`; machine-readable companion `ladder_autopsy_v2.json`.

Our shipped deck is the load-bearing context for everything below: it plays
**exactly four Pokémon, all copies of Teal Mask Ogerpon ex** (210 HP, 2 prizes,
one attack: 30 + 30 x every Energy attached to both Actives). Every body we
ever lose concedes 2 prizes; opponents need exactly 3 knockouts to win, always.

## Operational definitions (and what each detector can and cannot see)

- **(a) Lost the race structurally** — the opponent scored first and we never
  reached prize parity at any point of the game, and no other detector fired.
- **(b) Traded badly** — we held the prize race at some point (differential of
  prizes-taken >= 0 after some knockout) and finished behind it, with at least
  one 2-prize concession; or we promoted a multi-prize body over an available
  1-prize one (that promotion detector cannot fire for this deck — every body
  is the same 2-prize card — and indeed never did).
- **(c) Missed lethal** — at a main-phase decision we had a *listed* attack
  option whose computable damage (printed base + computable riders, weakness
  x2, resistance -30; Ogerpon's energy-scaling attack is computed exactly)
  KOs their Active, whose prize value covers all our remaining prizes, and the
  game did not end that turn. Coin-flip and text-conditional attacks are
  excluded from lethal claims. Single-attack, active-target only: multi-prize
  sequences via gust or bench damage are out of scope (our deck plays none).
- **(d) Allowed visible lethal** — we lost on prizes, and at our last
  main-phase decision the opponent's eventual game-winning attack was already
  payable from their attached energy, its computable damage KO'd our Active for
  their remaining prizes, and we had a real dodge (retreat legal + a bench body
  that survives that attack or doesn't yield the needed prizes) and didn't
  take it.
- **(e) Resource death** — the game ended by deck-out or no-Active against us
  (with only four bodies in the deck, this is the bench-out loss mode), or we
  had no legal attack on 2+ of our last 4 turns.
- **(f) Other** — nothing fired and the race wasn't one-sided.

A loss carries every tag it earns. Primary = proximate cause: a game that
ended no-Active or deck-out died of (e) whatever else was true; prize-decided
games rank c > d > b > a.

**Detector validation (wins as control).** The lethal detector saw a computable
lethal in 23 of 27 wins and the agent converted **all 23 on the turn they
appeared** — the detector is live and the agent's lethal conversion is clean.
Zero wins show a race inversion. False-positive risk of the lethal detector is
therefore low for *our* attack (exact formula, verified against HpChange logs);
for *opponent* attacks (class d) unmatched damage riders make the estimate a
lower bound, so d is if anything undercounted — but see the d evidence below.

## The distribution

53 episodes, 27W–26L.

**Primary cause (26 losses):**

| primary | count |
|---|---|
| b — traded badly | **14** |
| e — resource death | **11** |
| a — structural race loss | 1 |
| c — missed lethal | **0** |
| d — allowed visible lethal | **0** |
| f — other | 0 |

**All tags:** b: 20, e: 13 (8 losses carry both), a: 1. Termination modes in
losses: prizes 15, no-Active 10, deck-out 1. (Wins: prizes 25, no-Active 2.)

**Per-archetype W/L (all 53):** Fezandipiti ex 8–7, Marnie's Grimmsnarl ex
10–3, Dudunsparce **1–5**, Archaludon ex 3–4, Mega Lucario ex 1–3, Mega
Starmie ex 2–0, Teal Mask Ogerpon ex (mirror) **0–2**, Team Rocket's Mewtwo ex
0–1, Crustle 0–1, Cynthia's Garchomp ex 1–0, N's Zoroark ex 1–0.

**The trade ledger, aggregated.** In losses we scored 49 KOs for 61 prizes
(**1.24 prizes/KO**) while conceding 63 KOs for 126 prizes (**2.00 —
definitionally, every body we own is a 2-prize ex**). Nine losses show the
purest denomination signature: KO count even or in our favor, prize race lost
anyway. The 2.00 concession rate is identical in wins — it is a property of
the deck, not of any in-game choice.

## Worked examples — class b (traded badly / denomination blindness)

**90611299 vs Marnie's Grimmsnarl ex (L, 9 turns, 1–0 final).** We out-KO'd
nothing — the KO count finished 3–3 and we still lost. T4 we KO Munkidori
(+1); T5 they KO Ogerpon (-2); T6 and T8 we KO two Grimmsnarl ex (+2, +2 —
race +3, prizes-taken 5 v 2); T9 they KO **two** Ogerpon in one turn (2+2) and
take exactly their last 4 prizes. Three of their KOs = 6 prizes; our three =
5. One prize short, and the shortfall is precisely the Munkidori/Grimmsnarl
denomination mix we were fed.

**90698334 vs Fezandipiti ex/Alakazam (L, 10 turns, 1–0 final).** KO count
**4–3 in our favor**, race led at +1 or +2 after turns 3, 5, 7, 9. Our victims:
Dunsparce (1), Kadabra (1), Alakazam (1), Alakazam (1, **+1 via Briar** — the
agent correctly played Briar at their exactly-2-prizes window and cashed the
extra prize). Total: 5. Their victims: Ogerpon, Ogerpon, Ogerpon = 6. They win
on their third KO, turn 10. Even a correctly played denomination tech card
couldn't close a 1.25-vs-2.00 exchange rate.

**90596164 vs Dudunsparce (L, 12 turns, 2–0 final).** The arithmetic is
unwinnable at even tempo: their board offers almost only 1-prize bodies, so we
need 5–6 KOs; they need 3. We took 4 KOs for 4 prizes (one Briar-boosted),
they took 3 KOs for 6, winning T12. Dudunsparce is our worst matchup (1–5) and
every loss there looks like this. No in-game trade decision changes 3-KOs-to-win
vs 6.

## Class c/d — the lethal-exactness evidence

**c = 0 of 26.** No loss contains a turn where a computable lethal was on the
board and the agent played something else. The same detector fired 23 times in
wins and the agent converted every one the same turn. Within the detector's
scope (single attack into the Active — which is our deck's entire move space),
lethal exactness is clean.

**d = 0 of 26 actionable.** Three losses ended with a next-turn lethal that was
fully computable at our last decision — and in all three the dodge condition
fails, so there was nothing the agent could have done at that point:

- **90592260** (Archaludon ex): their Archaludon, 220 computable damage into
  our 210 HP Ogerpon, 2 prizes needed. No surviving bench body of lower value —
  every alternative is another 210 HP, 2-prize Ogerpon in the same range.
- **90593844** (Archaludon ex): 230 into our 100-HP damaged Ogerpon; same
  no-dodge structure (we scored 0 prizes all game; the one a-primary loss).
- **90640463** (Crustle): 120 into our exactly-120-HP Ogerpon at turn 19;
  bench bodies identical, no safe swap.

The mono-body deck makes "dodge" nearly a null concept: everything we could
promote is the same HP, the same 2 prizes.

## WHICH CONCEPT FAMILY THE LOSSES INDICT

**Prize-trade economics — by incidence, and it isn't close. Lethal exactness
is exonerated on this sample.**

- Missed our lethal: **0/26**. Walked into a dodgeable visible lethal:
  **0/26** (3 visible, 0 dodgeable). The N_DET=3 concern about forced prize
  sequences produces no measurable body count on the real ladder. Do not build
  lethal exactness first.
- Denomination economics: **14/26 primary, 20/26 tagged** — we led or held the
  prize race mid-game and lost it to the 2-per-KO concession rate; 9 losses
  with KO parity or better. The 11 resource deaths are the same economics in
  another costume: 2–3 knockouts against a 4-body deck ends the game
  regardless of prizes (10 no-Active losses), and the audit found **zero**
  cases where the agent declined to bench a body it held — the bodies were
  simply gone.
- **But the indictment splits across lanes.** The 2.00 prizes-per-KO we
  concede is fixed at deck construction, not at decision time: with four
  identical 2-prize bodies there is no cheaper body to give, no promotion
  choice, no denomination-aware dodge. The losses to Dudunsparce (1–5) and
  Crustle (0–1) are arithmetic: 3 KOs for them vs 5–6 for us. That is the
  **GA/deck lane**: bodies with mixed prize denominations (or simply more
  bodies) change the ledger more than any agent feature can. The
  agent-side prize-trade feature (prizes-at-risk term in evaluation: when to
  commit the last body, when to hold an attack, Briar timing) is the right
  *second* build — the agent already plays Briar's extra-prize window
  correctly, so the hooks exist.

**Build order the numbers support:** (1) deck-lane fix to the concession rate
(GA objective should price own-board prizes-at-risk and body count), (2)
agent-side prize-trade/prizes-at-risk awareness, (3) lethal exactness — not
until some future autopsy shows a nonzero incidence.

## Classifier limitations

- Lethal detection covers single attacks into the Active with computable
  damage (exact for our Ogerpon attack, validated against replay HpChange
  logs; printed-flat + 7 common rider patterns for opponents, a lower bound).
  Multi-attack sequences, gust lines, bench damage and ability damage are out
  of scope — our deck can't execute them, so class c is exact for us, but
  class d could in principle miss rider- or ability-based opponent lethals.
- "Dodge available" ignores opponent gust responses (would only make dodges
  rarer — strengthening the d=0 finding) and energy the opponent attaches on
  their own turn (payability judged on attached energy only, conservative).
- Class b's race-inversion test reads outcomes, not counterfactuals: it shows
  the race flipped, not that a specific alternative decision existed. For this
  deck the promotion/denomination counterfactuals are provably empty (all
  bodies identical), which is why the verdict routes most of b to the deck
  lane rather than claiming agent blunders.
- Opponent archetypes labeled from the deck action at `steps[1]` of the
  opponent seat (the v1 labeling bug is fixed; v1's "unknown" opponents are
  now fully attributed).
- 53 games is one day of ladder play; per-archetype splits carry small-n
  noise (Mewtwo, Crustle, Garchomp, Zoroark are single games).
