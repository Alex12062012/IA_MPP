"""
model.py
Le réseau prédit deux valeurs : lambda_home, lambda_away (buts attendus par équipe).
On en déduit ensuite toute la distribution des scores possibles via une matrice de
Poisson (approche Dixon-Coles simplifiée) -> probas vainqueur/nul + score exact le plus probable.

C'est ce qui permet une sortie directement exploitable : "vainqueur X%", "score exact Y%".
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.stats import poisson

from . import config as cfg

ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh}


class ScorePredictorNet(nn.Module):
    """MLP simple. L'architecture (n_layers, hidden_size, dropout, activation) est
    ce que le NAS génétique fait varier — voir nas.py."""

    def __init__(self, input_dim: int, n_layers: int, hidden_size: int,
                 dropout: float, activation: str):
        super().__init__()
        act_cls = ACTIVATIONS[activation]
        layers = []
        in_dim = input_dim
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, hidden_size), act_cls(), nn.Dropout(dropout)]
            in_dim = hidden_size
        layers.append(nn.Linear(in_dim, 2))  # sortie : raw_lambda_home, raw_lambda_away
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        raw = self.net(x)
        # softplus pour garantir lambda > 0 (contrainte physique : buts attendus positifs)
        lambdas = torch.nn.functional.softplus(raw) + 1e-4
        return lambdas  # shape (batch, 2) -> [lambda_home, lambda_away]


def poisson_nll_loss(pred_lambdas, true_goals):
    """Negative log-likelihood Poisson — la vraie loss adaptée au problème
    (mieux que MSE brut sur des buts, qui sont des comptages, pas des continus)."""
    lam_h, lam_a = pred_lambdas[:, 0], pred_lambdas[:, 1]
    gh, ga = true_goals[:, 0], true_goals[:, 1]
    dist_h = torch.distributions.Poisson(lam_h)
    dist_a = torch.distributions.Poisson(lam_a)
    nll = -(dist_h.log_prob(gh) + dist_a.log_prob(ga))
    return nll.mean()


def score_matrix(lambda_home: float, lambda_away: float, max_goals: int = cfg.MAX_GOALS):
    """Matrice (max_goals+1) x (max_goals+1) des probas de chaque score exact,
    en supposant indépendance des deux Poisson (simplification standard Dixon-Coles de base)."""
    goals_range = np.arange(0, max_goals + 1)
    p_home = poisson.pmf(goals_range, lambda_home)
    p_away = poisson.pmf(goals_range, lambda_away)
    matrix = np.outer(p_home, p_away)
    matrix = matrix / matrix.sum()  # renormalise (le tronquage à max_goals perd un peu de masse)
    return matrix


def outcome_probs_from_matrix(matrix: np.ndarray):
    """Retourne (proba_victoire_domicile, proba_nul, proba_victoire_exterieur)."""
    home_win = np.tril(matrix, -1).sum()
    draw = np.trace(matrix)
    away_win = np.triu(matrix, 1).sum()
    return home_win, draw, away_win


def best_score_from_matrix(matrix: np.ndarray):
    """Retourne ((buts_home, buts_away), proba) du score le plus probable."""
    idx = np.unravel_index(np.argmax(matrix), matrix.shape)
    return idx, matrix[idx]
