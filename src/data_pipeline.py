"""
data_pipeline.py
Charge les CSV Kaggle (club + international), construit les features,
et produit un split train/val/test temporel propre (pas de fuite de données futures).

Usage (dans Colab, après avoir téléchargé les CSV bruts dans data/):
    from src import data_pipeline as dp
    df = dp.build_dataset()
    train_df, val_df, test_df = dp.temporal_split(df)
"""

import pandas as pd
import numpy as np
from datetime import datetime

from . import config as cfg


def inspect_columns(csv_path: str) -> list:
    """Aide au debug : affiche les vraies colonnes d'un CSV avant de toucher au mapping.
    A LANCER EN PREMIER après le téléchargement, pour vérifier config.CLUB_COLUMN_MAP."""
    df = pd.read_csv(csv_path, nrows=5)
    print(f"Colonnes de {csv_path} :")
    for c in df.columns:
        print(f"  - {c}")
    return list(df.columns)


def _safe_rename(df: pd.DataFrame, colmap: dict) -> pd.DataFrame:
    """Renomme selon colmap ({nom_standard: nom_reel_dans_le_csv}),
    ignore les colonnes absentes (avec warning) plutôt que de planter."""
    rename_map = {src: dst for dst, src in colmap.items() if src in df.columns}
    missing = [src for dst, src in colmap.items() if src not in df.columns]
    if missing:
        print(f"[WARN] Colonnes manquantes (ignorées, vérifie config.py) : {missing}")
    df = df.rename(columns=rename_map)
    return df


def load_club_matches() -> pd.DataFrame:
    df = pd.read_csv(cfg.RAW_CLUB_CSV)
    df = _safe_rename(df, cfg.CLUB_COLUMN_MAP)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_team", "away_team", "home_goals", "away_goals"])
    df["is_neutral"] = 0
    df["source"] = "club"
    return df


def load_intl_matches() -> pd.DataFrame:
    df = pd.read_csv(cfg.RAW_INTL_CSV)
    df = _safe_rename(df, cfg.INTL_COLUMN_MAP)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_team", "away_team", "home_goals", "away_goals"])
    if "neutral" in df.columns:
        df["is_neutral"] = df["neutral"].astype(str).str.lower().isin(["true", "1", "yes"]).astype(int)
    else:
        df["is_neutral"] = 0
    # pas de cotes dispo sur ce dataset -> colonnes vides, gérées en NaN puis imputées
    for c in ["home_odds", "draw_odds", "away_odds", "home_elo", "away_elo"]:
        if c not in df.columns:
            df[c] = np.nan
    df["source"] = "intl"
    return df


def _implied_probs(row) -> tuple:
    """Convertit cotes décimales en probabilités implicites normalisées (overround retiré)."""
    o_h, o_d, o_a = row.get("home_odds"), row.get("draw_odds"), row.get("away_odds")
    if pd.isna(o_h) or pd.isna(o_d) or pd.isna(o_a) or o_h <= 1 or o_d <= 1 or o_a <= 1:
        return (np.nan, np.nan, np.nan)
    inv = np.array([1 / o_h, 1 / o_d, 1 / o_a])
    norm = inv / inv.sum()  # retire la marge du bookmaker
    return tuple(norm)


def _rolling_team_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construit forme, moyenne buts marqués/encaissés, jours de repos, h2h — sans fuite temporelle
    (chaque ligne n'utilise que des matchs STRICTEMENT antérieurs)."""
    df = df.sort_values("date").reset_index(drop=True)

    # Historique long format : une ligne par (équipe, match) pour calculer les rolling stats facilement
    home_rows = df[["date", "home_team", "home_goals", "away_goals"]].copy()
    home_rows.columns = ["date", "team", "goals_for", "goals_against"]
    home_rows["points"] = np.where(home_rows.goals_for > home_rows.goals_against, 3,
                            np.where(home_rows.goals_for == home_rows.goals_against, 1, 0))
    home_rows["is_home"] = 1

    away_rows = df[["date", "away_team", "away_goals", "home_goals"]].copy()
    away_rows.columns = ["date", "team", "goals_for", "goals_against"]
    away_rows["points"] = np.where(away_rows.goals_for > away_rows.goals_against, 3,
                            np.where(away_rows.goals_for == away_rows.goals_against, 1, 0))
    away_rows["is_home"] = 0

    long_df = pd.concat([home_rows, away_rows]).sort_values("date").reset_index(drop=True)

    def team_rolling(team, upto_date, n):
        hist = long_df[(long_df.team == team) & (long_df.date < upto_date)].tail(n)
        if len(hist) == 0:
            return np.nan, np.nan, np.nan, np.nan  # form, goals_avg, conceded_avg, days_since_last
        form = hist.points.mean()
        goals_avg = hist.goals_for.mean()
        conceded_avg = hist.goals_against.mean()
        days_since = (upto_date - hist.date.max()).days
        return form, goals_avg, conceded_avg, days_since

    # NOTE: boucle explicite pour rester lisible et débuggable ; pour un dataset >100k lignes,
    # envisager un groupby+shift vectorisé si trop lent sur Colab.
    records = []
    for _, row in df.iterrows():
        hf5, hg5, hc5, h_days = team_rolling(row.home_team, row.date, 5)
        af5, ag5, ac5, a_days = team_rolling(row.away_team, row.date, 5)
        hf10, _, _, _ = team_rolling(row.home_team, row.date, 10)
        af10, _, _, _ = team_rolling(row.away_team, row.date, 10)

        h2h = df[((df.home_team == row.home_team) & (df.away_team == row.away_team) & (df.date < row.date)) |
                 ((df.home_team == row.away_team) & (df.away_team == row.home_team) & (df.date < row.date))].tail(5)
        if len(h2h) > 0:
            home_wins = ((h2h.home_team == row.home_team) & (h2h.home_goals > h2h.away_goals)).sum() + \
                        ((h2h.away_team == row.home_team) & (h2h.away_goals > h2h.home_goals)).sum()
            h2h_rate = home_wins / len(h2h)
        else:
            h2h_rate = np.nan

        records.append({
            "home_form_5": hf5, "away_form_5": af5,
            "home_form_10": hf10, "away_form_10": af10,
            "home_goals_avg_5": hg5, "away_goals_avg_5": ag5,
            "home_conceded_avg_5": hc5, "away_conceded_avg_5": ac5,
            "days_since_last_home": h_days, "days_since_last_away": a_days,
            "h2h_home_winrate": h2h_rate,
        })

    feat_df = pd.DataFrame(records)
    return pd.concat([df.reset_index(drop=True), feat_df], axis=1)


def build_dataset(include_intl: bool = True) -> pd.DataFrame:
    """Point d'entrée principal. Charge, fusionne, construit toutes les features."""
    club = load_club_matches()
    frames = [club]
    if include_intl:
        intl = load_intl_matches()
        frames.append(intl)
    df = pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)

    df["home_elo"] = df.get("home_elo", pd.Series(np.nan, index=df.index))
    df["away_elo"] = df.get("away_elo", pd.Series(np.nan, index=df.index))

    implied = df.apply(_implied_probs, axis=1, result_type="expand")
    implied.columns = ["home_odds_implied", "draw_odds_implied", "away_odds_implied"]
    df = pd.concat([df, implied], axis=1)

    print(f"[INFO] {len(df)} matchs bruts avant feature engineering (attention : peut être lent, "
          f"c'est normal pour la boucle rolling sur un gros dataset).")
    df = _rolling_team_features(df)

    # Impute les NaN (premiers matchs d'une équipe sans historique) avec la médiane globale
    for col in cfg.FEATURE_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = 0.0

    df.to_csv(cfg.PROCESSED_CSV, index=False)
    print(f"[INFO] Dataset processed sauvegardé : {cfg.PROCESSED_CSV} ({len(df)} lignes)")
    return df


def build_feature_vector_for_match(df: pd.DataFrame, home_team: str, away_team: str,
                                    match_date: str, is_neutral: int = 0) -> np.ndarray:
    """Construit le vecteur de features pour un match FUTUR (pas encore joué), à partir de
    l'historique connu dans df. C'est ce qu'on appelle avant chaque pronostic MPP.
    match_date : string 'YYYY-MM-DD'."""
    match_date = pd.to_datetime(match_date)

    # Important : ne repartir QUE des colonnes brutes (pas des features déjà calculées dans df,
    # sinon _rolling_team_features les recalcule et les concatène en double).
    raw_cols = ["date", "home_team", "away_team", "home_goals", "away_goals",
                "home_odds", "draw_odds", "away_odds"]
    base = df[[c for c in raw_cols if c in df.columns]].copy()

    fake_row = pd.DataFrame([{
        "date": match_date, "home_team": home_team, "away_team": away_team,
        "home_goals": np.nan, "away_goals": np.nan,
        "home_odds": np.nan, "draw_odds": np.nan, "away_odds": np.nan,
    }])
    combined = pd.concat([base, fake_row], ignore_index=True).sort_values("date").reset_index(drop=True)
    combined_feat = _rolling_team_features(combined)
    combined_feat["is_neutral"] = is_neutral

    implied = combined_feat.apply(_implied_probs, axis=1, result_type="expand")
    implied.columns = ["home_odds_implied", "draw_odds_implied", "away_odds_implied"]
    combined_feat = pd.concat([combined_feat, implied], axis=1)

    target_row = combined_feat[(combined_feat.date == match_date) &
                                (combined_feat.home_team == home_team) &
                                (combined_feat.away_team == away_team)].iloc[-1]

    for col in cfg.FEATURE_COLUMNS:
        if col not in target_row.index or pd.isna(target_row[col]):
            target_row[col] = df[col].median() if col in df.columns else 0.0

    return target_row[cfg.FEATURE_COLUMNS].values.astype(np.float32)


def fetch_recent_results(days_back: int = 3) -> pd.DataFrame:
    """Récupère les résultats de matchs récents via football-data.org, au format brut
    (mêmes colonnes que load_club_matches) — à appeler après chaque journée pour la boucle
    de réentraînement incrémental. Nécessite requests (déjà présent sur Colab)."""
    import requests
    from datetime import timedelta

    date_to = datetime.now().date()
    date_from = date_to - timedelta(days=days_back)

    headers = {"X-Auth-Token": cfg.FOOTBALL_DATA_API_KEY}
    params = {"dateFrom": str(date_from), "dateTo": str(date_to), "status": "FINISHED"}
    resp = requests.get(f"{cfg.FOOTBALL_DATA_BASE_URL}/matches", headers=headers, params=params)

    if resp.status_code != 200:
        print(f"[WARN] football-data.org a répondu {resp.status_code} : {resp.text[:200]}")
        return pd.DataFrame(columns=["date", "home_team", "away_team", "home_goals", "away_goals"])

    matches = resp.json().get("matches", [])
    rows = []
    for m in matches:
        score = m.get("score", {}).get("fullTime", {})
        if score.get("home") is None or score.get("away") is None:
            continue
        rows.append({
            "date": m["utcDate"][:10],
            "home_team": m["homeTeam"]["name"],
            "away_team": m["awayTeam"]["name"],
            "home_goals": score["home"],
            "away_goals": score["away"],
            "home_odds": np.nan, "draw_odds": np.nan, "away_odds": np.nan,
            "home_elo": np.nan, "away_elo": np.nan,
            "is_neutral": 0, "source": "football-data-live",
        })
    df = pd.DataFrame(rows)
    print(f"[INFO] {len(df)} matchs terminés récupérés (derniers {days_back} jours)")
    return df


def temporal_split(df: pd.DataFrame):
    """Split temporel strict : test = dernière période jamais vue, val = juste avant test."""
    df = df.sort_values("date").reset_index(drop=True)
    test_start = pd.to_datetime(cfg.TEST_SET_START_DATE)

    test_df = df[df.date >= test_start].copy()
    trainval_df = df[df.date < test_start].copy()

    n_val = int(len(trainval_df) * cfg.VAL_FRACTION)
    val_df = trainval_df.iloc[-n_val:].copy()
    train_df = trainval_df.iloc[:-n_val].copy()

    print(f"[INFO] Split -> train={len(train_df)} | val={len(val_df)} | test={len(test_df)}")
    return train_df, val_df, test_df
