"""Download the Pokemon TCG AI Battle Challenge competition data via kagglehub."""

import kagglehub

# Download latest version
path = kagglehub.competition_download('pokemon-tcg-ai-battle-challenge-strategy')

print("Path to competition files:", path)
