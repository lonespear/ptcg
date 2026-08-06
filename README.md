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
| `ptcg/viz.py` | Chart styling — one validated palette for every figure |
| `scripts/download_data.py` | kagglehub download |
| `scripts/run_eda.py` | First EDA pass; writes `figures/` |
| `EDA_FINDINGS.md` | **Findings so far — read this first** |

## Using the data

```python
from ptcg import load_cards

cards, effects = load_cards("EN")   # 1,267 cards, 1,811 attacks/abilities
```

`cards` carries HP, type, weakness, retreat, and `has_rule_box` (the prize-risk
flag). `effects` carries parsed energy costs, damage, and a `drawback`
classification — see `EDA_FINDINGS.md` §4 for why that last one matters.

## Status

- [x] Data downloaded and normalized
- [x] EDA pass 1 — pool composition, efficiency, weakness graph, Trainer taxonomy
- [ ] **Blocked:** Simulation-category access (403 — competition rules not yet
      accepted). Strategy entry requires a Simulation submission.
- [ ] Effective-cost model for attack evaluation
- [ ] Deck concept + agent policy
- [ ] Writeup
