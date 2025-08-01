# 🏠 Projet MLOps : Prédiction des prix des maisons à Ames, Iowa

## Auteurs

- Mouhamed DIENG  
- Ndaraw FALL  
- Isabelle Olive Kantoussan
 
## Présentation du projet

Ce projet vise à construire un pipeline MLOps pour prédire le prix de vente de maisons à Ames (Iowa, USA), à partir d'un jeu de données de 79 variables. L'objectif est de fournir un outil fiable pour l'estimation immobilière, utilisable par des professionnels, data scientists ou agences.

Le pipeline couvre les étapes suivantes : analyse exploratoire, feature engineering, expérimentation et sélection de modèles, suivi des expériences, déploiement d'une API, interface utilisateur web, tests et automatisation.

---

## Étapes du projet

**1. Analyse exploratoire des données**  
- Nettoyage, visualisation, enrichissement du dataset Ames Housing.
- Détection des valeurs manquantes, outliers, création de nouvelles variables.

**2. Expérimentation et choix du modèle**  
- Construction de pipelines de prétraitement.
- Test et comparaison de modèles de régression (ElasticNet, XGBoost, LightGBM…).
- Suivi des expériences avec MLflow.
- Sélection du meilleur modèle et sauvegarde pour le déploiement.

**3. Déploiement et automatisation**  
- API FastAPI pour servir le modèle (prédiction unitaire ou batch).
- Interface Streamlit pour la saisie manuelle ou par fichier CSV.
- Dockerisation de l'API pour un déploiement facile.
- Script d'exécution automatique des notebooks.
- Pipeline CI/CD avec GitHub Actions.

**4. Tests et validation**  
- Tests unitaires sur la qualité des données et la robustesse de l'API.

---

## Structure du projet

```
.
├── .github/
│   └── workflows/
│       └── main.yml
├── front/
│   └── house_price_front.py
├── house_price_dataset/
│   └── ames_housing.csv
├── notebooks/
│   ├── house_price_01_analyse.ipynb
│   ├── house_price_02_essais.ipynb
│   ├── mlruns/
│   ├── models/
│   ├── output_files/
│   └── reports/
├── serving/
│   ├── app.py
│   ├── Dockerfile
│   ├── requirements.txt
│   └── utils/
│       ├── inputs.py
│       ├── loaders.py
│       └── __init__.py
├── settings/
│   ├── params.py
│   └── __init__.py
├── src/
│   └── __init__.py
├── tests/
│   ├── test_api.py
│   ├── test_datasets.py
│   └── __init__.py
├── pyproject.toml
├── pytest.ini
├── README.md
├── requirements-nr.txt
├── requirements-tests.txt
└── run_house_price.sh
```
---

## Données

- **Source :** `house_price_dataset/ames_housing.csv`
- **Variables :** 79 features (surface, quartier, qualité, etc.), cible = `SalePrice`
  
---

## Fichiers clés

- `notebooks/house_price_01_analyse.ipynb` : Analyse exploratoire et préparation des données.
- `notebooks/house_price_02_essais.ipynb` : Expérimentation et sélection des modèles.
- `serving/app.py` : API FastAPI pour la prédiction.
- `front/house_price_front.py` : Interface utilisateur Streamlit.
- `run_house_price.sh` : Script d'exécution automatique des notebooks.
- `settings/params.py` : Paramètres globaux du projet.
- `tests/` : Tests unitaires (API et datasets).

---

## Lancer le projet

### 1. Installer les dépendances

- **Analyse/modélisation :**  
  ```bash
  pip install -r requirements-nr.txt
  ```

- **API :**  
  ```bash
  pip install -r serving/requirements.txt
  ```

### 2. Exécuter les notebooks

- **Script automatique (Linux/Mac) :**  
  ```bash
  bash run_house_price.sh
  ```

- **Ou manuellement dans `notebooks/`**

### 3. Démarrer l'API

```bash
cd serving
uvicorn app:app --reload
```

### 4. Lancer l'interface Streamlit

```bash
streamlit run front/house_price_front.py
```

### 5. Lancer les tests

```bash
pytest
```

---

## Déploiement Docker

Un Dockerfile est fourni pour déployer l'API :

```bash
cd serving
docker build -t house-price-api .
docker run -p 8000:8000 house-price-api
```