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
    (chaque ligne n'utilise que des matchs STRICTEMENT antérieurs).
    Version vectorisée (groupby+shift+rolling) — nécessaire dès qu'on dépasse quelques milliers
    de matchs, la version boucle-par-ligne+filtre-dataframe est O(n²) et devient inutilisable
    au-delà de ~10-20k lignes (des heures sur un dataset de plusieurs centaines de milliers)."""
    df = df.sort_values("date").reset_index(drop=True)
    df["match_idx"] = df.index

    # --- Forme / buts marqués-encaissés / jours de repos (perspective équipe) ---
    home_rows = df[["match_idx", "date", "home_team", "home_goals", "away_goals"]].copy()
    home_rows.columns = ["match_idx", "date", "team", "goals_for", "goals_against"]
    home_rows["points"] = np.where(home_rows.goals_for > home_rows.goals_against, 3,
                            np.where(home_rows.goals_for == home_rows.goals_against, 1, 0))
    home_rows["is_home"] = 1

    away_rows = df[["match_idx", "date", "away_team", "away_goals", "home_goals"]].copy()
    away_rows.columns = ["match_idx", "date", "team", "goals_for", "goals_against"]
    away_rows["points"] = np.where(away_rows.goals_for > away_rows.goals_against, 3,
                            np.where(away_rows.goals_for == away_rows.goals_against, 1, 0))
    away_rows["is_home"] = 0

    long_df = pd.concat([home_rows, away_rows], ignore_index=True)
    long_df = long_df.sort_values(["team", "date"]).reset_index(drop=True)

    g = long_df.groupby("team", sort=False)
    long_df["form_5"] = g["points"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    long_df["form_10"] = g["points"].transform(lambda s: s.shift(1).rolling(10, min_periods=1).mean())
    long_df["goals_avg_5"] = g["goals_for"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    long_df["conceded_avg_5"] = g["goals_against"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    long_df["days_since_last"] = g["date"].transform(lambda s: (s - s.shift(1)).dt.days)

    keep_cols = ["match_idx", "form_5", "form_10", "goals_avg_5", "conceded_avg_5", "days_since_last"]
    home_feat = long_df[long_df.is_home == 1][keep_cols].rename(columns={
        "form_5": "home_form_5", "form_10": "home_form_10",
        "goals_avg_5": "home_goals_avg_5", "conceded_avg_5": "home_conceded_avg_5",
        "days_since_last": "days_since_last_home",
    })
    away_feat = long_df[long_df.is_home == 0][keep_cols].rename(columns={
        "form_5": "away_form_5", "form_10": "away_form_10",
        "goals_avg_5": "away_goals_avg_5", "conceded_avg_5": "away_conceded_avg_5",
        "days_since_last": "days_since_last_away",
    })

    df = df.merge(home_feat, on="match_idx", how="left").merge(away_feat, on="match_idx", how="left")

    # --- Face-à-face (h2h) : pour chaque match, taux de victoire de l'équipe à domicile actuelle
    # contre CET adversaire précis, sur les 5 confrontations précédentes (peu importe qui recevait
    # à l'époque). Vectorisé via une table "team vs opponent" en double perspective. ---
    base = df[["match_idx", "date", "home_team", "away_team", "home_goals", "away_goals"]]
    home_persp = pd.DataFrame({
        "match_idx": base.match_idx, "date": base.date,
        "team": base.home_team, "opponent": base.away_team,
        "won": (base.home_goals > base.away_goals).astype(int),
        "is_home_perspective": 1,
    })
    away_persp = pd.DataFrame({
        "match_idx": base.match_idx, "date": base.date,
        "team": base.away_team, "opponent": base.home_team,
        "won": (base.away_goals > base.home_goals).astype(int),
        "is_home_perspective": 0,
    })
    h2h_long = pd.concat([home_persp, away_persp], ignore_index=True)
    h2h_long = h2h_long.sort_values(["team", "opponent", "date"]).reset_index(drop=True)

    gh = h2h_long.groupby(["team", "opponent"], sort=False)
    h2h_long["h2h_winrate"] = gh["won"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())

    h2h_home = h2h_long[h2h_long.is_home_perspective == 1][["match_idx", "h2h_winrate"]] \
        .rename(columns={"h2h_winrate": "h2h_home_winrate"})

    df = df.merge(h2h_home, on="match_idx", how="left")
    df = df.drop(columns=["match_idx"])
    return df




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
