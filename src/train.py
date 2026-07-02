"""
train.py
Deux usages :
1. train_final_model() : à partir de la meilleure archi trouvée par le NAS, entraîne un modèle
   final propre (ou un ensemble des top-N candidats), sauvegarde les poids.
2. incremental_retrain() : réentraînement rapide (quelques secondes) sur les nouveaux résultats
   de matchs, à lancer après chaque journée. Ne repart PAS de zéro, fine-tune le modèle existant.
"""

import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from . import config as cfg
from .model import ScorePredictorNet, poisson_nll_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model_from_config(input_dim: int, individual_config: dict, state_dict=None):
    model = ScorePredictorNet(input_dim, individual_config["n_layers"],
                               individual_config["hidden_size"], individual_config["dropout"],
                               individual_config["activation"]).to(device)
    if state_dict is not None:
        model.load_state_dict(state_dict)
    return model


def train_final_model(X_train, y_train, X_val, y_val, input_dim, individual_config,
                       epochs=100, patience=15, save_path=f"{cfg.MODELS_DIR}/final_model.pt"):
    """Réentraîne à fond (plus d'epochs, patience plus large) la meilleure archi trouvée par
    le NAS, sur train complet, avec le val set pour l'early stopping."""
    model = build_model_from_config(input_dim, individual_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=individual_config["lr"])

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                              torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=cfg.NAS_BATCH_SIZE, shuffle=True)
    val_X = torch.tensor(X_val, dtype=torch.float32).to(device)
    val_y = torch.tensor(y_val, dtype=torch.float32).to(device)

    best_val_nll = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = poisson_nll_loss(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_nll = poisson_nll_loss(model(val_X), val_y).item()

        if val_nll < best_val_nll:
            best_val_nll = val_nll
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[INFO] Early stopping à l'epoch {epoch}")
                break

    torch.save({
        "state_dict": best_state,
        "config": individual_config,
        "input_dim": input_dim,
        "val_nll": best_val_nll,
    }, save_path)
    print(f"[INFO] Modèle final sauvegardé : {save_path} (val_nll={best_val_nll:.4f})")
    return best_state, best_val_nll


def build_ensemble(top_candidates: list, input_dim: int, save_path=f"{cfg.MODELS_DIR}/ensemble.pt"):
    """top_candidates: liste de (fitness, config, state_dict) — typiquement les 5-10 meilleurs
    du NAS. Sauvegarde l'ensemble pour moyenne pondérée des prédictions en inférence."""
    fitnesses = np.array([c[0] for c in top_candidates])
    weights = np.exp(fitnesses - fitnesses.max())  # softmax-like, stable numériquement
    weights = weights / weights.sum()

    torch.save({
        "configs": [c[1] for c in top_candidates],
        "state_dicts": [c[2] for c in top_candidates],
        "weights": weights.tolist(),
        "input_dim": input_dim,
    }, save_path)
    print(f"[INFO] Ensemble de {len(top_candidates)} modèles sauvegardé : {save_path}")
    print(f"[INFO] Poids relatifs : {[round(w, 3) for w in weights]}")


def incremental_retrain(model_path, new_X, new_y, epochs=cfg.INCREMENTAL_EPOCHS,
                         lr=cfg.INCREMENTAL_LR):
    """A lancer après chaque journée de matchs (résultats réels ajoutés au dataset).
    Fine-tune le modèle existant sur les nouvelles données uniquement — quelques secondes.
    NE remplace PAS le NAS complet, qui reste à relancer périodiquement (toutes les 4-6 semaines,
    voir README) pour ne pas sur-optimiser sur du bruit récent."""
    ckpt = torch.load(model_path, map_location=device)
    model = build_model_from_config(ckpt["input_dim"], ckpt["config"], ckpt["state_dict"])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X = torch.tensor(new_X, dtype=torch.float32).to(device)
    y = torch.tensor(new_y, dtype=torch.float32).to(device)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = poisson_nll_loss(model(X), y)
        loss.backward()
        optimizer.step()

    ckpt["state_dict"] = model.state_dict()
    torch.save(ckpt, model_path)
    print(f"[INFO] Réentraînement incrémental terminé ({epochs} epochs, {len(new_X)} nouveaux matchs). "
          f"Loss finale : {loss.item():.4f}")
