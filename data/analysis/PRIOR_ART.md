# PRIOR_ART — external survey for the 1000+ question

*Compiled 2026-08-07. Question: our agent is 1-ply search over a fitted linear evaluator
(~850-950 ladder); is the ceiling of that model below the top, and what do the strongest
agents in this format-class actually run?*

## Bottom line

The public evidence from this competition says the 1000+ ceiling is NOT primarily an
architecture ceiling — it is a deck/matchup ceiling first and an evaluator-quality ceiling
second. Multiple teams with heavier machinery (ISMCTS, RL, deep search) sit at or below our
band; the teams above 1000 got there through deck selection against the mined live meta and
pilot tuning, and the documented top of the ladder (~1300+) is held by agents whose search
is shallow-to-moderate but whose opponent/meta model is good. Naive deep search with a bad
opponent model has been independently measured to be *worse than no search* by at least
three teams. A 1-ply + good evaluator architecture placed 2nd of 33 in the closest academic
analogue (Hearthstone 2018). Path B (flat rollout ranking) is the right next step *only if*
the determinization it rolls out against is defensible.

---

## (a) This competition's public footprint

Public code is plentiful — 60+ GitHub repos reference the competition. No Kaggle forum
writeup from a confirmed top-10 team was found (the Strategy track deadline is 13 Sep, so
strong teams are withholding). The most instructive repos, with real ladder numbers:

| Repo | Architecture | Ladder result / key claim |
|---|---|---|
| [henriquetakahiroito/pokemon-tcg-ai-battle](https://github.com/henriquetakahiroito/pokemon-tcg-ai-battle) | Determinized flat-UCB MCTS on the engine `search_begin/search_step` API + numpy MLP (32→64→64→1) value net, episode mining of 6,533 replays | **948 live** after 4 deck pivots; observed ladder top: Tea Party's Walrein at **1326**. Writeup (WRITEUP.md) findings: Elo and win-rate diverge 500+ pts; a deck's Elo ceiling ≈ number of top-tier archetypes it auto-loses (1 hole → 1326, 2 holes → caps ~850); offline gauntlets are non-predictive of ladder |
| [cha7ura/kaggle-pokemon-tcg](https://github.com/cha7ura/kaggle-pokemon-tcg) | XGBoost card-policy (131-dim features) + lethal gate; explicit negative results ledger | ~**900–978** live via meta-counter deck (Dragapult counter to #1 Trevenant). "The ONE proven lever: meta-counter (rock-paper-scissors)." Policy learning plateaued at top-1 0.488→0.529 (**RF≈GBM≫linear**; bottleneck = features, not data). **Forward search validated WORSE** (bad opponent determinization); self-play gate failed; offline sim does not predict ladder |
| [willgitdata/kaggle-pokemon-tcg-ai](https://github.com/willgitdata/kaggle-pokemon-tcg-ai) | Determinized turn-search on the engine search API + self-play-evolved evaluation weights + Boss-threat opponent model; later **imitation of the #1 pilot** (decision-agreement diffing) | Local gauntlet MLE Elo 1304; **live plateau ~810–877**. Broke its plateau not with deeper search but by measuring decision-agreement vs the #1 agent (40% overall, energy-attach 23.7%) and patching the biggest gap → 58% mirror A/B win |
| [brunoramosmartins/ptcgabc-ismcts](https://github.com/brunoramosmartins/ptcgabc-ismcts) | SO-ISMCTS, pre-registered hypotheses, 500-seed paired tests | With opponent list KNOWN (mirror): ISMCTS 0.780 vs heuristic. With hidden info filled by dummy cards: **0.506 — the entire search advantage erased** (paired −27.4 pp, p=1.2e-17). "Deep search against a wrong opponent model is not neutral, it is confidently wrong." Heuristic rollouts did NOT beat random rollouts at fixed budget (p=0.279) |
| [yowayani517/pokemon-tcg-ai-agent](https://github.com/yowayani517/pokemon-tcg-ai-agent) | Compared 4 approaches head-to-head locally | Turn-goal planning **85%** > rule-based wall 82% > self-play RL (LightGBM→numpy) 54% > raw forward search **53%** ("naive search degrades in a stochastic game") |
| [souyuukou/pokemon-tcg-ai-battle3/4/5](https://github.com/souyuukou/pokemon-tcg-ai-battle5) | Heaviest public engineering: exact information-safe turn search, C++20 core, hypergeometric prize belief, opponent-deck priors from mined replays, semantic-action transpositions, value-only UCT past thresholds | No ladder number published; the belief-tracking design (condition replay-derived deck priors on revealed cards) is the state of the art in this pool |
| [wmh/ptcg-abc](https://github.com/wmh/ptcg-abc) | Pure rule-based per-context scoring, 3 decks + meta analysis | Best **836** (Bellibolt). Simple+consistent deck beat their fancier combo deck (532) |
| [sutesute0000/ptcg-ai-battle-lopunny](https://github.com/sutesute0000/ptcg-ai-battle-lopunny) | Imitation via divergence analysis: replay top pilots' games through own agent, decode disagreements to card names, derive piloting rules | No opponent modeling at all, by design — "pick the deck with the highest top-tier win rate and impose its win pattern" |
| [knightynite/ptcg-ai-battle-agent](https://github.com/knightynite/ptcg-ai-battle-agent) | Rule-based + measurement discipline (paired-world CRN harness, decision diffs) | Strategy-track writeup is about measurement, not search |
| Also seen | PPO notebook ([kaggle](https://www.kaggle.com/code/hmnshudhmn24/pok-mon-tcg-ai-battle-challenge-ppo-agent)), DRL repos, one LLM-workspace repo (sota1111) | No RL/LLM agent reports a ladder number above the rule-based pack; one repo reports RL+MCTS at **580 μ** vs their own rule-based 660 |

Excluded per task: defense031/ptcg (us), lonespear/ptcg (upstream). Kaggle forum itself:
nothing substantive beyond beginner guides; the leaderboard page is JS-walled to fetch.

**Convergent findings across independent teams (treat these as ground truth):**
1. Deck/meta choice moves the ladder more than pilot architecture (cha7ura, takahiroito, wmh).
2. Determinized search helps ONLY with a credible opponent model; with a naive fill it is
   measurably harmful (brunoramosmartins, cha7ura, yowayani — three independent replications).
3. Offline gauntlets against fixed baselines do not predict live Elo (takahiroito, cha7ura).
4. Ladder Elo ≠ win rate; ratings drift and are only comparable within a snapshot
   (takahiroito, brunoramosmartins both measured it).
5. The matchup-hole framework: Elo ceiling ≈ count of top-tier auto-losses (takahiroito).

## (b) The architecture leaderboard of the genre

- **Hearthstone AI Competition (2018–2020, IEEE CoG):** winning strategies were MCTS, RHEA,
  and greedy lookahead; the 2018 runner-up (2nd of 33) was a **greedy one-step lookahead
  with a coevolution-tuned evaluation function** — literally our architecture with a better
  evaluator ([arXiv:2410.19681](https://arxiv.org/html/2410.19681v1),
  [results](https://hearthstoneai.github.io/results2018.html)). The 2019 best agent was
  **ISMCTS + sparse sampling**. Best published academic result: MCTS + supervised value
  network ([Świechowski et al. 2018](https://arxiv.org/pdf/1808.04794)).
- **Strategy Card Game AI Competition / LOCM (2019–2023):** early editions won by search
  (flat MC, shallow minimax/MCTS) + handmade or statistically-learned heuristics; the final
  edition was **dominated by neural-network RL**, with end-to-end-trained ByteRL winning
  "by a large margin" ([Kowalski & Miernik, arXiv:2305.11814](https://arxiv.org/pdf/2305.11814)).
  Note the arc: heuristics+shallow search reign until someone pays the RL training bill.
  ByteRL itself was later shown highly exploitable ([arXiv:2404.16689](https://arxiv.org/pdf/2404.16689)).
- **MTG:** Forge's shipped AI is per-card heuristic (`canPlayAI()`) and famously cannot
  combine two effects; no search-based MTG bot has topped anything public. Draft AI is where
  NNs won ([RyanSaxe/mtg](https://github.com/RyanSaxe/mtg)).
- **Yu-Gi-Oh:** production bots (WindBot) are rule-based; the research frontier is
  [ygo-agent](https://github.com/sbl1996/ygo-agent) (deep RL + MCTS + league training), not
  deployed on any ladder comparable to ours.
- **Kaggle simulation analogues:** rules-based agents won Halite, Kore, and Lux AI S2's
  main prize class per the microRTS retrospective ([arXiv:2402.08112](https://arxiv.org/html/2402.08112v1));
  Lux S1 was won by deep RL (IMPALA+UPGO, [IsaiahPressman/Kaggle_Lux_AI_2021](https://github.com/IsaiahPressman/Kaggle_Lux_AI_2021)).
  Pattern: hand-tuned policies with good domain features stay competitive until a team
  invests serious RL infrastructure; mid-ladder plateaus break on *evaluation quality and
  opponent modeling*, not on adding generic search.

## (c) Where 1-ply linear + Path B sits, and what 1000+ requires

**The map (imperfect-information search ladder, cheap → expensive):**
1-ply static eval (us) → flat rollout ranking over determinizations = **PIMC at the root**
(Path B) → determinized UCT per world → single-observer ISMCTS (one tree over information
sets; fixes strategy fusion) → MO-ISMCTS / belief-weighted determinization → RL-trained
value/policy inside search. Canon: [Cowling, Powley & Whitehouse 2012](https://eprints.whiterose.ac.uk/id/eprint/75048/1/CowlingPowleyWhitehouse2012.pdf);
Long et al. 2010 ("Understanding the Success of PIMC") shows PIMC-style determinization is
near-optimal precisely in games with high leaf correlation and fast disambiguation — a
prize-race TCG mostly qualifies, which is why full ISMCTS bought brunoramosmartins only
+3.8 pp (n.s.) over plain PIMC *when the opponent model was right*, and nothing when wrong.

**Practical read at 2 vCPU, ~600 s/episode, ~200 decisions (~2-3 s/decision):**
Path B (rank top-k root actions by rollouts across a few determinized worlds) is the correct
price point — full ISMCTS at 1000 iterations/decision does not fit the budget and its
measured marginal value over root-level determinized rollouts in this exact game is ~4 pp
in the best case. But every competitor measurement says the binding constraint is the
determinizer, not the search: rollouts against a dummy-filled opponent deck will *lower*
our Elo. Our fitted evaluator is not the bottleneck evidence either — cha7ura's careful
policy-learning plateau (linear ≪ tree models, features the bottleneck) says a linear form
loses real accuracy, but their tree-model agent still needed a meta-counter deck to break 900.

**What the evidence says 1000+ actually requires, in order:**
1. A deck with ≤1 auto-loss among current top-tier archetypes (matchup-hole framework;
   the mined meta shifts mid-competition, so this needs a fresh replay read near lock).
2. Pilot quality on that deck's specific win pattern (imitation/divergence vs top pilots
   beat generic search improvements for two teams).
3. Only then, search — rollout ranking with a replay-derived opponent prior, not a uniform fill.
The 1326 top observed was a rule-based/shallow agent on the right deck. Nothing published
suggests 1000+ *requires* MCTS or RL; several things published show naive versions hurt.

## (d) Three steal-able ideas

1. **Replay-derived opponent determinization** (souyuukou battle5; the fix
   brunoramosmartins priced): build archetype priors from mined Kaggle episode JSONs,
   condition on cards the opponent has revealed this game, and sample determinizations from
   that posterior. This is the single gate that decides whether Path B adds or subtracts Elo.
   Cheap fallback measured as "cheapest deployable fix": assume the opponent plays a known
   top-meta list (or our own), never a dummy fill.
2. **Decision-agreement diffing against top pilots** (willgitdata ORACLE.md; sutesute0000
   divergence.py): replay top-rated agents' episodes through our agent, bucket disagreements
   by move type (attach/evolve/retreat/support), and patch the largest gap. willgitdata
   measured 23.7% agreement on energy-attach vs the #1, fixed that one dimension, and won
   the A/B 58% — a targeted evaluator upgrade, exactly compatible with our linear model
   (add/refit terms where the diff concentrates).
3. **Matchup-hole deck audit before any more pilot work** (takahiroito WRITEUP.md): mine
   current episodes, reconstruct the live archetype distribution, count our auto-losses
   among top-tier decks. Two holes caps ~850 — our exact band. If we have two, no evaluator
   or search work clears 1000; a deck/tech change might. His full pipeline (episode miner,
   deck reconstruction from logs) is public in `tools/`.

## Sources not linked above

- Competition: [pokemon-tcg-ai-battle](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle),
  [strategy track](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle-challenge-strategy/overview),
  [cabt engine docs](https://matsuoinstitute.github.io/cabt/)
- ISMCTS vs determinization in Dou Di Zhu: [Whitehouse et al.](https://www.semanticscholar.org/paper/67e1f4795c461a5467d6009b1efdaa36aad03a40)
- PIMC critique/practice: [AI Factory newsletter](https://www.aifactory.co.uk/newsletter/2013_01_reduce_burden.htm)
- AlphaZero-style baselines surprisingly strong in imperfect info: [AlphaZe**](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2023.1014561/full)
