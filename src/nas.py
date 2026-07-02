"""
nas.py
Neural Architecture Search par algorithme génétique (population d'archis, pas de poids figés
à la main). Chaque "individu" = un dict d'hyperparamètres d'archi. Fitness = NLL Poisson sur
le set de validation après un entraînement gradient rapide (early stopping).

Conçu pour tourner plusieurs heures sans surveillance sur Colab :
- checkpoint de la population + logs après chaque génération (Google Drive)
- budget de temps dur (NAS_TIME_BUDGET_SECONDS) : s'arrête proprement avant la fin de la nuit
- reprise possible depuis le dernier checkpoint si la session a coupé
"""

import json
import time
import random
import os
import contextlib
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from . import config as cfg
from .model import ScorePredictorNet, poisson_nll_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def random_individual():
    sp = cfg.NAS_SEARCH_SPACE
    return {
        "n_layers": random.choice(sp["n_layers"]),
        "hidden_size": random.choice(sp["hidden_size"]),
        "dropout": random.choice(sp["dropout"]),
        "activation": random.choice(sp["activation"]),
        "lr": random.choice(sp["lr"]),
    }


def mutate(individual: dict, rate: float = 0.3) -> dict:
    sp = cfg.NAS_SEARCH_SPACE
    child = dict(individual)
    for key in child:
        if random.random() < rate:
            child[key] = random.choice(sp[key])
    return child


def crossover(parent_a: dict, parent_b: dict) -> dict:
    return {k: random.choice([parent_a[k], parent_b[k]]) for k in parent_a}


def train_candidate(individual: dict, input_dim: int, train_dataset: TensorDataset, val_X, val_y,
                     epochs=cfg.NAS_EPOCHS_PER_CANDIDATE, patience=cfg.NAS_EARLY_STOPPING_PATIENCE):
    """Entraîne un candidat avec early stopping. Retourne (fitness = -val_nll, state_dict, individual).
    Prend un TensorDataset (pas un DataLoader partagé) : chaque appel (potentiellement dans un
    thread séparé, voir train_population_parallel) crée son propre DataLoader pour éviter les
    conflits d'itérateur entre threads."""
    # CUDA stream dédié : permet à plusieurs candidats de tourner en parallèle sur le même GPU
    # au lieu de se mettre en file (le réseau est petit, le GPU était sous-utilisé en séquentiel).
    stream = torch.cuda.Stream() if torch.cuda.is_available() else None
    stream_ctx = torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()

    with stream_ctx:
        loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        model = ScorePredictorNet(input_dim, individual["n_layers"], individual["hidden_size"],
                                   individual["dropout"], individual["activation"]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=individual["lr"])

        best_val_nll = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(epochs):
            model.train()
            for xb, yb in loader:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                optimizer.zero_grad()
                pred = model(xb)
                loss = poisson_nll_loss(pred, yb)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_pred = model(val_X.to(device, non_blocking=True))
                val_nll = poisson_nll_loss(val_pred, val_y.to(device, non_blocking=True)).item()

            if val_nll < best_val_nll:
                best_val_nll = val_nll
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break  # early stopping

        if stream is not None:
            stream.synchronize()

    fitness = -best_val_nll  # fitness plus haute = mieux (on minimise la NLL)
    return fitness, best_state, individual


def train_population_parallel(population, input_dim, train_dataset, val_X, val_y,
                               max_workers=cfg.NAS_PARALLEL_WORKERS):
    """Entraîne toute une génération de candidats en parallèle (threads + CUDA streams).
    Remplace la boucle séquentielle — gros gain de vitesse quand le GPU est sous-utilisé
    par un modèle aussi petit (observé ~14% d'utilisation en séquentiel)."""
    results = [None] * len(population)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(train_candidate, ind, input_dim, train_dataset, val_X, val_y): i
            for i, ind in enumerate(population)
        }
        for future in futures:
            i = futures[future]
            fitness, state, ind = future.result()
            results[i] = (fitness, ind, state)
    return results


def save_checkpoint(path, generation, population, fitnesses, best_ever):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "generation": generation,
        "population": population,
        "fitnesses": fitnesses,
        "best_ever": best_ever,  # (fitness, individual, state_dict)
    }, path)


def load_checkpoint(path):
    if os.path.exists(path):
        return torch.load(path, map_location=device)
    return None


def log_generation(log_path, generation, fitnesses, best_individual, elapsed):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    line = {
        "generation": generation,
        "best_fitness": max(fitnesses),
        "mean_fitness": float(np.mean(fitnesses)),
        "worst_fitness": min(fitnesses),
        "best_individual": best_individual,
        "elapsed_seconds": elapsed,
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(line) + "\n")
    print(f"[GEN {generation}] best={line['best_fitness']:.4f} "
          f"mean={line['mean_fitness']:.4f} elapsed={elapsed:.0f}s")


def run_nas(X_train, y_train, X_val, y_val, input_dim,
            checkpoint_path=f"{cfg.MODELS_DIR}/nas_checkpoint.pt",
            log_path=f"{cfg.LOGS_DIR}/nas_log.jsonl",
            resume=True):
    """Boucle principale du NAS génétique. A appeler depuis le notebook Colab.
    Reprend automatiquement depuis checkpoint_path si resume=True et qu'un checkpoint existe."""

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                              torch.tensor(y_train, dtype=torch.float32))
    val_X = torch.tensor(X_val, dtype=torch.float32)
    val_y = torch.tensor(y_val, dtype=torch.float32)

    ckpt = load_checkpoint(checkpoint_path) if resume else None
    if ckpt:
        print(f"[INFO] Reprise depuis checkpoint, génération {ckpt['generation']}")
        population = ckpt["population"]
        start_gen = ckpt["generation"] + 1
        best_ever = ckpt["best_ever"]
    else:
        population = [random_individual() for _ in range(cfg.NAS_POPULATION_SIZE)]
        start_gen = 0
        best_ever = (-float("inf"), None, None)

    start_time = time.time()

    for gen in range(start_gen, cfg.NAS_N_GENERATIONS):
        elapsed_total = time.time() - start_time
        if elapsed_total > cfg.NAS_TIME_BUDGET_SECONDS:
            print(f"[INFO] Budget temps atteint ({elapsed_total:.0f}s), arrêt propre.")
            break

        gen_start = time.time()
        results = train_population_parallel(population, input_dim, train_ds, val_X, val_y)

        results.sort(key=lambda r: r[0], reverse=True)
        fitnesses = [r[0] for r in results]

        if results[0][0] > best_ever[0]:
            best_ever = results[0]

        elite = results[:cfg.NAS_ELITE_SIZE]
        elite_individuals = [r[1] for r in elite]

        # nouvelle génération : élite conservée telle quelle + reste = crossover/mutation des élites
        new_population = list(elite_individuals)
        while len(new_population) < cfg.NAS_POPULATION_SIZE:
            pa, pb = random.sample(elite_individuals, 2)
            child = mutate(crossover(pa, pb))
            new_population.append(child)
        population = new_population

        gen_elapsed = time.time() - gen_start
        log_generation(log_path, gen, fitnesses, elite[0][1], gen_elapsed)

        if gen % cfg.NAS_CHECKPOINT_EVERY_N_GEN == 0:
            save_checkpoint(checkpoint_path, gen, population, fitnesses, best_ever)

    print(f"[INFO] NAS terminé. Meilleure fitness jamais atteinte : {best_ever[0]:.4f}")
    print(f"[INFO] Meilleure archi : {best_ever[1]}")
    return best_ever  # (fitness, individual_config, state_dict)
