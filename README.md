# ptcg — Pokémon TCG AI Battle Challenge

Entry for the Kaggle **Pokémon TCG AI Battle Challenge — Strategy Category**.
Final submission deadline **13 Sep 2026**.

The Strategy track is judged on Model Score (70%), Deck Score (20%), and Report
Score (10%) — the deliverable is a ≤2,000-word Kaggle Writeup explaining the
strategic reasoning behind the agent, not just a leaderboard placement.

## Setup

```bash
pip install kagglehub pandas numpy matplotlib
export KAGGLE_API_TOKEN=KGAT_...        # or write it to ~/.kaggle/access_token
python scripts/download_data.py         # ~598 MB
```

`ptcg/data.py` finds the extracted data in the kagglehub cache automatically; set
`PTCG_DATA_DIR` to override.

## Layout

| Path | What it is |
|---|---|
| `ptcg/data.py` | Loads the CSV into tidy `cards` + `effects` tables |
| `ptcg/episodes.py` | Parses `cabt` replay JSON into decklists + outcomes |
| `ptcg/viz.py` | Chart styling — one validated palette for every figure |
| `scripts/download_data.py` | kagglehub download |
| `scripts/run_eda.py` | Card-pool EDA; writes `figures/` |
| `scripts/mine_meta.py` | Mines replays into the live metagame; writes `data/` |
| `EDA_FINDINGS.md` | **Findings so far — read this first** |

## Using the data

```python
from ptcg import load_cards

cards, effects = load_cards("EN")   # 1,267 cards, 1,811 attacks/abilities
```

`cards` carries HP, type, weakness, retreat, and `has_rule_box` (the prize-risk
flag). `effects` carries parsed energy costs, damage, and a `drawback`
classification — see `EDA_FINDINGS.md` §4 for why that last one matters.

## Replay mining

The Simulation episodes are published as **public datasets** (no competition
access needed) and each replay carries both 60-card decks plus the winner:

```bash
python -c "import kagglehub; print(kagglehub.dataset_download('kaggle/pokemon-tcg-ai-battle-episodes-index'))"
python scripts/mine_meta.py
```

A single day is ~20 GB on disk and ~5,200 matches. Set `PTCG_EPISODE_DIR` if the
kagglehub cache lives elsewhere.

## Submitting

```bash
python scripts/build_submission.py                      # build + inspect
python scripts/build_submission.py --submit -m "note"   # validates, then submits
```

`build_submission.py --submit` will not send anything unless
`scripts/validate_submission.py` passes. That script runs the competition's own
`cabt` environment from `kaggle_environments` — the real validation episode,
agent against a copy of itself. Two submissions were lost to a bug a hand-rolled
test could not see (see `EDA_FINDINGS.md` §16), so this gate is not optional.

## Status

- [x] Data downloaded and normalized
- [x] EDA pass 1 — pool composition, efficiency, weakness graph, Trainer taxonomy
- [x] EDA pass 2 — live metagame from 5,197 replays
- [x] EDA pass 3 — six days, 57,108 deck instances, decklist-level win rates
- [x] Local battle engine + self-play arena
- [x] **Agent live on the Simulation leaderboard** (Heuristic v3, rating ~446
      vs a top of ~1206 — a working baseline, not a competitive one)
- [ ] Improve the policy: card-selection contexts (`TO_HAND` is 11.7% of
      decisions and currently answered with index 0), deliberate retreat,
      prize/board awareness
- [ ] Search-based policy (the engine exposes `SearchBegin`/`SearchStep`)
- [ ] Writeup for the Strategy track (**not submitted — draft only**)
