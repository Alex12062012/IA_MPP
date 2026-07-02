"""
predict.py
Pipeline d'inférence : donne deux équipes -> sort vainqueur + % confiance,
score exact + % confiance. Même format que le skill pronostic-sportif existant.

Usage :
    from src import predict
    result = predict.predict_match(features_row, model_path="models/final_model.pt")
    # ou predict.predict_match_ensemble(features_row, "models/ensemble.pt") pour l'ensemble
"""

import torch
import numpy as np

from . import config as cfg
from .model import ScorePredictorNet, score_matrix, outcome_probs_from_matrix, best_score_from_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_single_model(model_path):
    ckpt = torch.load(model_path, map_location=device)
    model = ScorePredictorNet(ckpt["input_dim"], ckpt["config"]["n_layers"],
                               ckpt["config"]["hidden_size"], ckpt["config"]["dropout"],
                               ckpt["config"]["activation"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def predict_match(feature_vector: np.ndarray, model_path=f"{cfg.MODELS_DIR}/final_model.pt") -> dict:
    """feature_vector : array 1D dans l'ordre exact de config.FEATURE_COLUMNS."""
    model = _load_single_model(model_path)
    x = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        lambdas = model(x).cpu().numpy()[0]
    return _format_prediction(lambdas[0], lambdas[1])


def predict_match_ensemble(feature_vector: np.ndarray,
                            ensemble_path=f"{cfg.MODELS_DIR}/ensemble.pt") -> dict:
    """Moyenne pondérée des lambdas prédits par chaque modèle de l'ensemble
    (poids = fitness relative de chaque candidat lors du NAS)."""
    ckpt = torch.load(ensemble_path, map_location=device)
    x = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0).to(device)

    all_lambdas = []
    for config, state_dict in zip(ckpt["configs"], ckpt["state_dicts"]):
        model = ScorePredictorNet(ckpt["input_dim"], config["n_layers"], config["hidden_size"],
                                   config["dropout"], config["activation"]).to(device)
        model.load_state_dict(state_dict)
        model.eval()
        with torch.no_grad():
            all_lambdas.append(model(x).cpu().numpy()[0])

    weights = np.array(ckpt["weights"])
    lambdas = np.average(np.array(all_lambdas), axis=0, weights=weights)
    return _format_prediction(lambdas[0], lambdas[1])


def _format_prediction(lambda_home: float, lambda_away: float) -> dict:
    matrix = score_matrix(lambda_home, lambda_away)
    home_win, draw, away_win = outcome_probs_from_matrix(matrix)
    (best_h, best_a), best_score_proba = best_score_from_matrix(matrix)

    outcomes = {"domicile": float(home_win), "nul": float(draw), "exterieur": float(away_win)}
    winner = max(outcomes, key=outcomes.get)

    return {
        "lambda_home": round(float(lambda_home), 2),
        "lambda_away": round(float(lambda_away), 2),
        "vainqueur": winner,
        "vainqueur_confiance_pct": round(outcomes[winner] * 100, 1),
        "score_exact": f"{int(best_h)}-{int(best_a)}",
        "score_exact_confiance_pct": round(float(best_score_proba) * 100, 1),
        "detail_probas": {
            "domicile_pct": round(outcomes["domicile"] * 100, 1),
            "nul_pct": round(outcomes["nul"] * 100, 1),
            "exterieur_pct": round(outcomes["exterieur"] * 100, 1),
        },
    }
