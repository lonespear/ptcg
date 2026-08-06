# ptcg — Pokémon TCG AI Battle Challenge

Kaggle hackathon entry for the **Pokemon TCG AI Battle Challenge (Strategy)** competition.

## Setup

```bash
pip install kagglehub
```

Auth (one of):

```bash
mkdir -p ~/.kaggle && echo $KAGGLE_API_TOKEN > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token
# or
export KAGGLE_API_TOKEN=KGAT_...
```

## Download the data

```bash
python download_data.py
```

Files land in the kagglehub cache (`~/.cache/kagglehub/competitions/...`); the script prints the path.
