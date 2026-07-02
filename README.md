# MPP-IA

IA maison pour pronostiquer les scores foot (façon Mon Petit Prono), entraînée par
recherche d'architecture génétique (NAS), 100% utilisable depuis un navigateur (Colab).

## Principe

Un petit réseau de neurones prédit `lambda_home` et `lambda_away` (buts attendus par
équipe), sur la base de features tabulaires (forme, Elo, cotes implicites, historique
face-à-face...). On en déduit ensuite, via une matrice de Poisson, toutes les probas :
vainqueur (domicile/nul/extérieur) + % de confiance, score exact + % de confiance.

L'architecture du réseau (nb de couches, taille, dropout, activation, learning rate)
n'est pas fixée à la main : elle est trouvée par un **algorithme génétique** qui fait
évoluer une population de configurations sur plusieurs générations (voir `src/nas.py`).

## Setup rapide (tout depuis le navigateur, via Colab)

1. Crée un repo GitHub `mpp-ia`, uploade tout ce dossier dedans (`src/`, `requirements.txt`,
   `notebook_colab.ipynb`, `.gitignore`) via l'interface web GitHub (drag & drop).
2. Ouvre [Google Colab](https://colab.research.google.com), `Fichier > Ouvrir un notebook >
   GitHub`, colle l'URL de ton repo, ouvre `notebook_colab.ipynb`.
3. `Exécution > Modifier le type d'exécution > GPU (T4)`.
4. Exécute les cellules dans l'ordre (voir la cellule d'intro du notebook).

## Vérification importante avant le premier run complet

Les noms de colonnes dans `src/config.py` (`CLUB_COLUMN_MAP`, `INTL_COLUMN_MAP`) sont
des **suppositions** basées sur la description publique des datasets Kaggle. La cellule 3
du notebook (`dp.inspect_columns`) affiche les vrais noms de colonnes après téléchargement —
si ça ne matche pas, corrige `config.py` (c'est le seul fichier à toucher).

## Lancer le run de la nuit (NAS)

Profil par défaut dans `config.py` (`NAS_POPULATION_SIZE=120`, `NAS_N_GENERATIONS=60`,
`NAS_TIME_BUDGET_SECONDS=11*3600`) : pensé pour ~10-12h, avec arrêt propre avant la fin.

- Checkpoint après chaque génération sur Google Drive (`models/nas_checkpoint.pt`) —
  si Colab coupe la session, relance juste la cellule 5, ça reprend automatiquement.
- Logs génération par génération dans `logs/nas_log.jsonl` (fitness best/mean/worst).
- Au réveil : regarde si la fitness (`best_fitness`) plafonne depuis plusieurs générations
  → si oui, pas besoin de relancer aussi long la prochaine fois.

## Après le NAS : routine du matin

1. **Lire les logs** (`logs/nas_log.jsonl`) — évolution de la fitness par génération.
2. **Entraînement final** (cellule 6) — reprend la meilleure archi, entraîne à fond.
3. **Backtest** (cellule 7) — comparaison à une baseline naïve sur le test set jamais vu.
   Si ton modèle ne bat pas la baseline, revoir les features avant de re-NAS.
4. **Prédiction** (cellule 8) — format `{vainqueur, vainqueur_confiance_pct, score_exact,
   score_exact_confiance_pct, detail_probas}`.
5. **Incrémental** (cellule 9) — après chaque journée, fine-tune sur les nouveaux résultats
   (quelques secondes, pas besoin de re-NAS).

## Cadence recommandée

- Réentraînement incrémental : après chaque journée de matchs.
- NAS complet (nouvelle nuit) : toutes les 4-6 semaines, pas plus souvent — sinon risque
  de sur-optimiser sur du bruit récent plutôt que d'apprendre un vrai signal.

## Limites connues

- Colab free : pas de garantie 100% de survie sur 10-12h (timeout d'inactivité possible).
  Le checkpointing Drive rattrape le coup, mais accepte le risque qu'il faille relancer
  une deuxième nuit si ça coupe tôt.
- Le mapping de colonnes (`config.py`) est une supposition à vérifier au premier run.
- Le modèle suppose indépendance entre buts domicile/extérieur (simplification Poisson
  standard) — améliorable plus tard avec un vrai Dixon-Coles (terme de corrélation) si tu
  veux pousser la précision.

## Structure

```
mpp-ia/
  src/
    config.py       # TOUT ce qui est ajustable (colonnes, features, hyperparams NAS)
    data_pipeline.py # chargement + feature engineering + split temporel
    model.py         # réseau + matrice Poisson (score exact / vainqueur)
    nas.py            # algo génétique de recherche d'archi, avec checkpointing
    train.py          # entraînement final + réentraînement incrémental
    predict.py        # inférence -> format de sortie MPP
  notebook_colab.ipynb # tout le pipeline, orchestré cellule par cellule
  requirements.txt
```
