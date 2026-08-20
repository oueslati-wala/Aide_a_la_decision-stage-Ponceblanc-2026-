# Estimateur d'offres — Ponceblanc & LBFI

Application d'aide à la décision pour le pricing des devis.

Le commercial saisit les informations de l'offre (client, produit, quantité, coût total).
Le modèle propose un **prix de vente total recommandé (€)**, une **probabilité d'acceptation**, des scénarios (prudent / recommandé / ambitieux) et une courbe de sensibilité.

Les deux sources (Ponceblanc, LBFI) sont **entièrement séparées** : données, features et modèles ne sont jamais mélangés.

Ouvrir l'URL affichée (généralement http://localhost:8501).

### Variables d'environnement optionnelles

| Variable | Rôle |
|----------|------|
| `DEVIS_XLSX` | Chemin vers `Query_tableau_devis_with_costs.xlsx` |
| `PONCEBLANC_CSV` / `LBFI_CSV` | CSV de secours par source |
| `DEVIS_VERBOSE=1` | Logs de résolution des feuilles Excel |

Par défaut, les données sont lues depuis `data/Query_tableau_devis_with_costs.xlsx`.

---

## Structure

```
3/
├── vis.py              # UI Streamlit (point d'entrée)
├── predict.py          # API de prédiction (QuoteEstimator)
├── features.py         # Chargement & features par source
├── train_models.py     # Entraînement (classifieur + régresseurs)
├── eval_models.py      # Évaluation hors UI
├── requirements.txt
├── data/
│   └── Query_tableau_devis_with_costs.xlsx
└── models/
    ├── ponceblanc/           # modèle Ponceblanc (coeff × coût → prix)
    └── lbfi/                 # modèle LBFI (même logique euro)
```

Chaque dossier `models/<source>/` contient :

- `classifier_best.joblib`
- `regressor_marge_{best,lower,upper}.joblib`
- `product_clusters.json`, `client_encoding.json`
- `margin_feature_columns.json`, `regressor_target.json`, `metrics.json`

> **Note** : `data/` et `models/` sont exclus du dépôt (`.gitignore`) — ils contiennent des données clients réelles et des artefacts régénérables. Après avoir cloné le dépôt, placez `Query_tableau_devis_with_costs.xlsx` dans `data/`, puis lancez `python train_models.py` pour générer les modèles.

---

## Logique métier

| Composant | Rôle |
|-----------|------|
| **Régresseur** | Prédit un coefficient `prix/coût` sur les devis **acceptés**, puis `prix = coeff × coût` |
| **Classifieur** | Estime P(acceptation) pour un prix candidat donné |
| **Scénarios** | Prix prudents / ambitieux / maximisant la proba, + ligne éditable « Votre choix » |
| **Historique** | Filtres client / produit / bande de quantité pour justification |

Les taux de marge affichés dans l'UI sont **dérivés** (affichage seulement) et ne sont jamais des entrées du modèle.

Le modèle de production est **prix / coefficient** pour les deux sources (Ponceblanc et LBFI).

---

## Réentraîner

```bash
python train_models.py          # les deux sources
# ou logique interne : une source à la fois selon le script
```

Les artefacts sont écrits sous `models/<source>/`.  
Relancer Streamlit après un retrain (ou vider le cache Streamlit).

---

## Limites connues

- Clients / produits rares → proba et reco moins fiables.
- Coefficients extrêmes (typiquement > 3–5× le coût) hors zone d'entraînement.
- L'outil quantifie l'habitude historique ; il ne remplace pas le jugement commercial.

---

## Métriques (ordre de grandeur actuel)

| Source | Devis | Taux accept. | ROC-AUC classif. | MAE coeff (acceptés) |
|--------|------:|-------------:|-----------------:|---------------------:|
| Ponceblanc | ~1 950 | ~35 % | ~0.72 | ~0.42 |
| LBFI | ~6 100 | ~31 % | ~0.77 | ~0.08 |

Valeurs exactes dans `models/*/metrics.json` et page **Performance des modèles**.
