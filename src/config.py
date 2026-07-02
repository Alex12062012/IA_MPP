"""
Config centrale du projet MPP-IA.
Si les noms de colonnes des CSV Kaggle diffèrent de ce qui est prévu ici,
c'est LE SEUL fichier à modifier (data_pipeline.py lit tout via ce mapping).
"""

# --- Chemins ---
DATA_DIR = "data"
MODELS_DIR = "models"
LOGS_DIR = "logs"

RAW_CLUB_CSV = f"{DATA_DIR}/club_matches_raw.csv"        # adamgbor/club-football-match-data-2000-2025
RAW_INTL_CSV = f"{DATA_DIR}/intl_matches_raw.csv"        # martj42/international-football-results
PROCESSED_CSV = f"{DATA_DIR}/processed.csv"

# --- football-data.org (résultats récents, boucle de réentraînement incrémental) ---
FOOTBALL_DATA_API_KEY = "f7693eb3b75a49419a09ee7095716e4b"
FOOTBALL_DATA_BASE_URL = "https://api.football-data.org/v4"

# --- Kaggle datasets (slugs pour l'API kaggle) ---
KAGGLE_CLUB_DATASET = "adamgbor/club-football-match-data-2000-2025"
KAGGLE_INTL_DATASET = "martj42/international-football-results-from-1872-to-2017"

# --- Mapping de colonnes attendu pour le dataset "club" ---
# NOTE: à vérifier/corriger après le premier chargement réel (voir data_pipeline.inspect_columns)
CLUB_COLUMN_MAP = {
    "date": "MatchDate",
    "home_team": "HomeTeam",
    "away_team": "AwayTeam",
    "home_goals": "FTHome",
    "away_goals": "FTAway",
    "home_odds": "OddHome",
    "draw_odds": "OddDraw",
    "away_odds": "OddAway",
    "home_elo": "HomeElo",
    "away_elo": "AwayElo",
    "league": "Division",
}

INTL_COLUMN_MAP = {
    "date": "date",
    "home_team": "home_team",
    "away_team": "away_team",
    "home_goals": "home_score",
    "away_goals": "away_score",
    "tournament": "tournament",
    "neutral": "neutral",
}

# --- Features finales utilisées par le modèle ---
FEATURE_COLUMNS = [
    "home_form_5", "away_form_5",           # points/match sur les 5 derniers
    "home_form_10", "away_form_10",
    "home_goals_avg_5", "away_goals_avg_5",  # buts marqués moyenne 5 derniers
    "home_conceded_avg_5", "away_conceded_avg_5",
    "home_elo", "away_elo",
    "h2h_home_winrate",                      # % victoires domicile sur confrontations passées
    "days_since_last_home", "days_since_last_away",
    "is_neutral",
    "home_odds_implied", "draw_odds_implied", "away_odds_implied",  # probas implicites cotes (normalisées)
]

TARGET_HOME_GOALS = "home_goals"
TARGET_AWAY_GOALS = "away_goals"

# --- Split temporel (jamais de fuite de données futures) ---
TEST_SET_START_DATE = "2024-07-01"   # dernière saison = test pur, jamais vu en train/val
VAL_FRACTION = 0.15                  # pris sur la portion train (temporellement, juste avant test)

# --- Poisson / plafond de buts pour la matrice de score exact ---
MAX_GOALS = 8   # matrice 0..8 buts pour chaque équipe (largement suffisant)

# --- NAS (recherche d'architecture génétique) ---
NAS_SEARCH_SPACE = {
    "n_layers": [1, 2, 3],
    "hidden_size": [8, 16, 32, 64],
    "dropout": [0.0, 0.1, 0.2, 0.3],
    "activation": ["relu", "tanh"],
    "lr": [1e-2, 5e-3, 1e-3],
}

# Profil "recherche qualitative multi-nuits" — pensé pour tourner sur plusieurs sessions
# (reprise automatique via checkpoint à chaque relance, resume=True). Pas de raccourci sur la
# qualité : population large, patience généreuse, on relance chaque nuit jusqu'à ce que la
# fitness plafonne dans les logs (voir README).
NAS_POPULATION_SIZE = 150
NAS_ELITE_SIZE = 15
NAS_N_GENERATIONS = 300              # volontairement haut, le budget temps arrête avant si besoin
NAS_EPOCHS_PER_CANDIDATE = 60
NAS_EARLY_STOPPING_PATIENCE = 12
NAS_TIME_BUDGET_SECONDS = 11 * 3600  # marge de sécu avant que Colab tue la session (~12h max)
NAS_CHECKPOINT_EVERY_N_GEN = 1

# Entraîne plusieurs candidats EN PARALLÈLE sur le même GPU (threads + CUDA streams).
# Le GPU était sous-utilisé (~14%) en séquentiel car le réseau est minuscule — le paralléliser
# comble le temps mort entre kernels. Ajuste selon la VRAM dispo (8 est safe sur un T4 15GB
# pour ce genre de petit MLP).
NAS_PARALLEL_WORKERS = 8

# Batch size volontairement gros : le modèle est minuscule, le vrai coût n'est pas le calcul
# GPU mais l'overhead Python/dispatch répété à chaque mini-lot. Avec 228k lignes de train,
# batch=256 -> ~894 itérations/epoch (lent). batch=8192 -> ~28 itérations/epoch (~30x moins).
NAS_BATCH_SIZE = 8192

# --- Réentraînement incrémental (après chaque journée) ---
INCREMENTAL_EPOCHS = 15
INCREMENTAL_LR = 5e-4
