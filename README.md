# Projet MLOps : Prédiction des prix des maisons à Ames, Iowa

## Description du projet

Ce projet a pour objectif de construire un modèle de prédiction du prix final de chaque maison à Ames, Iowa, en utilisant un ensemble de 79 variables explicatives. Il met en œuvre un pipeline MLOps complet pour assurer la robustesse, la reproductibilité et la facilité de déploiement du modèle.

### Contexte métier

Dans le secteur de l'immobilier, la détermination précise du prix d'une propriété est cruciale. Ce projet vise à fournir un algorithme de prédiction des prix des maisons aux équipes immobilières, leur permettant d'estimer la valeur des biens en se basant sur un large éventail de caractéristiques. Cela inclut des aspects architecturaux, des conditions de vente, et des informations sur le quartier, entre autres.

## Architecture du projet

Le projet est structuré autour des principes clés du MLOps, intégrant les composants suivants :

- **Analyse exploratoire des données (EDA)** : Compréhension approfondie du jeu de données et identification des facteurs clés influençant les prix.
- **Expérimentation et suivi des modèles** : Utilisation de MLFlow pour gérer les différentes expérimentations de modèles, suivre les métriques de performance et versionner les modèles.
- **API REST pour les prédictions** : Une interface de programmation applicative (API) basée sur FastAPI pour servir le modèle de prédiction en temps réel ou en lot.
- **Tests unitaires automatisés** : Des tests rigoureux pour valider l'intégrité des données et le bon fonctionnement de l'API.
- **Pipeline CI/CD avec GitHub Actions** : Automatisation des processus de vérification du code, d'exécution des notebooks, de construction d'images Docker et de déploiement continu.
- **Containerisation avec Docker** : Empaquetage de l'application API dans un conteneur pour garantir un environnement d'exécution cohérent et portable.

## Structure du projet

```
mlops_house_price_prediction-main/
├── .github/
│   └── workflows/
│       └── main.yml                 # Pipeline CI/CD GitHub Actions
├── API/
│   ├── utils/
│   │   ├── inputs.py               # Modèles Pydantic pour la validation des entrées de l'API
│   │   └── loaders.py              # Fonctions pour charger le modèle de prédiction
│   ├── app.py                      # Application FastAPI principale de l'API
│   ├── Dockerfile                  # Instructions pour construire l'image Docker de l'API
│   └── requirements.txt            # Dépendances Python spécifiques à l'API
├── house_price_dataset/
│   └── ames_housing.csv            # Le jeu de données des prix des maisons à Ames, Iowa
├── notebooks/
│   ├── house_price_01_analyse.ipynb    # Notebook d'analyse exploratoire des données
│   ├── house_price_02_essais.ipynb     # Notebook d'expérimentation et de sélection des modèles
│   └── house_price_03_stabilite.ipynb  # Notebook pour l'analyse de la stabilité du modèle (optionnel)
├── tests/
│   ├── test_datasets.py            # Tests unitaires pour la validation du jeu de données
│   ├── test_api.py                 # Tests unitaires pour la validation des endpoints de l'API
│   └── __init__.py                 # Fichier d'initialisation du module de tests
├── output_files/                   # Répertoire pour les fichiers de sortie générés par les notebooks (ex: modèles sérialisés, rapports intermédiaires)
├── models/                         # Répertoire pour les modèles de machine learning sauvegardés localement
├── logs/                           # Répertoire pour les logs d'exécution des notebooks et de l'API
├── requirements-nr.txt            # Dépendances Python pour l'exécution des notebooks (non-root)
├── requirements-tests.txt         # Dépendances Python pour l'exécution des tests unitaires
├── run_house_price.sh            # Script shell pour exécuter les notebooks de manière séquentielle
└── README.md                     # Ce fichier de documentation du projet
```

## Installation et utilisation

### Prérequis

Assurez-vous d'avoir les éléments suivants installés sur votre système :

- **Python 3.11+** : Langage de programmation principal.
- **pip** : Gestionnaire de paquets Python.
- **Docker** (optionnel) : Pour la construction et l'exécution des conteneurs.

### Installation des dépendances

Pour installer toutes les dépendances nécessaires au projet, exécutez les commandes suivantes depuis la racine du projet :

```bash
# Installation des dépendances pour les notebooks et l'analyse
pip install -r requirements-nr.txt

# Installation des dépendances pour les tests unitaires
pip install -r requirements-tests.txt

# Installation des dépendances spécifiques à l'API
cd API
pip install -r requirements.txt
cd ..
```

### Exécution des notebooks

Les notebooks d'analyse et d'expérimentation peuvent être exécutés séquentiellement à l'aide du script shell fourni :

```bash
# Rendre le script exécutable
chmod +x run_house_price.sh

# Exécuter tous les notebooks
./run_house_price.sh
```

Ce script exécutera `house_price_01_analyse.ipynb` et `house_price_02_essais.ipynb`. Les logs d'exécution seront sauvegardés dans le répertoire `logs/`.

### Démarrage de l'API

Pour démarrer l'API de prédiction, naviguez dans le répertoire `API` et exécutez le fichier `app.py` :

```bash
cd API
python app.py
```

L'API sera accessible à l'adresse par défaut : `http://localhost:8000`.
La documentation interactive de l'API (Swagger UI) est disponible à : `http://localhost:8000/docs`.

### Endpoints de l'API

L'API expose les endpoints suivants :

- `GET /api/health` : Vérifie l'état de santé de l'API. Retourne un statut `healthy` si l'API est opérationnelle.
- `GET /api/model/info` : Fournit des informations détaillées sur le modèle de machine learning actuellement utilisé par l'API.
- `POST /api/predict` : Permet d'obtenir une prédiction de prix pour une seule maison en fournissant ses caractéristiques au format JSON.
- `POST /api/predict/batch` : Permet d'obtenir des prédictions de prix pour plusieurs maisons en téléchargeant un fichier CSV ou XLSX contenant leurs caractéristiques.

#### Exemple d'utilisation de l'endpoint `/api/predict`

```bash
curl -X POST "http://localhost:8000/api/predict" \
     -H "Content-Type: application/json" \
     -d 
```

### Exécution des tests

Les tests unitaires peuvent être exécutés en utilisant `pytest` depuis la racine du projet :

```bash
# Exécuter les tests spécifiques au jeu de données
python -m pytest tests/test_datasets.py -v

# Exécuter les tests spécifiques à l'API
python -m pytest tests/test_api.py -v

# Exécuter tous les tests du projet
python -m pytest tests/ -v
```

## Pipeline CI/CD

Le projet intègre un pipeline d'intégration et de déploiement continus (CI/CD) via GitHub Actions. Ce pipeline est déclenché à chaque push ou pull request sur la branche `main` et exécute les étapes suivantes :

1. **Vérification du code** : Effectue des contrôles de linting (avec `ruff`) et de formatage (avec `black`), puis exécute les tests unitaires.
2. **Exécution des notebooks** : Lance les notebooks d'analyse et d'expérimentation pour s'assurer de leur bon fonctionnement et de la reproductibilité des résultats.
3. **Build et push Docker** : Construit l'image Docker de l'API et la pousse vers un registre Docker configuré.
4. **Déploiement** : Déploie automatiquement la nouvelle version de l'API sur la plateforme Render (ou une autre plateforme configurée).

### Variables d'environnement requises pour le CI/CD

Pour que le pipeline de déploiement fonctionne correctement, les secrets suivants doivent être configurés dans votre dépôt GitHub :

- `DOCKER_USERNAME` : Votre nom d'utilisateur pour Docker Hub ou un autre registre Docker.
- `DOCKER_PASSWORD` : Votre mot de passe pour le registre Docker.
- `DOCKER_REGISTRY` : L'URL complète de votre image Docker (ex: `votre_utilisateur/house-price-api`).
- `RENDER_API_KEY` : Votre clé API pour la plateforme de déploiement Render.
- `RENDER_SERVICE_ID` : L'identifiant de votre service sur Render.

## Suivi des expériences avec MLFlow

MLFlow est utilisé pour le suivi des expérimentations de machine learning. Il permet de :

- Enregistrer les paramètres des modèles entraînés.
- Suivre les métriques de performance (RMSE, R², MAE, MAPE) pour chaque exécution.
- Gérer les versions des modèles et les archiver.
- Faciliter la comparaison visuelle des différentes expérimentations via son interface utilisateur.

### Démarrage de l'interface utilisateur MLFlow

Pour visualiser les expérimentations, naviguez dans le répertoire `notebooks` et lancez l'interface utilisateur MLFlow :

```bash
cd notebooks
mlflow ui
```

L'interface sera accessible dans votre navigateur à l'adresse par défaut : `http://localhost:5000`.

## Modèles testés

Le projet évalue la performance de plusieurs algorithmes de régression pour la prédiction des prix des maisons :

- **Régression Linéaire** : Un modèle de base pour établir une référence.
- **Random Forest Regressor** : Un modèle d'ensemble robuste, efficace pour capturer les relations non linéaires.
- **Gradient Boosting Regressor** : Un autre modèle d'ensemble puissant, connu pour sa précision.

Les performances de ces modèles sont évaluées à l'aide des métriques suivantes :

- **RMSE** (Root Mean Squared Error) : Mesure la taille moyenne des erreurs de prédiction.
- **R²** (Coefficient de détermination) : Indique la proportion de la variance de la variable dépendante qui est prévisible à partir des variables indépendantes.
- **MAE** (Mean Absolute Error) : Mesure la moyenne des différences absolues entre les prédictions et les observations réelles.
- **MAPE** (Mean Absolute Percentage Error) : Exprime l'erreur en pourcentage de la valeur réelle, utile pour l'interprétation métier.

## Variables importantes

L'analyse exploratoire et l'expérimentation des modèles ont permis d'identifier les variables les plus influentes sur le prix des maisons. Parmi celles-ci, les plus significatives incluent :

- `GrLivArea` : Surface habitable au-dessus du sol (en pieds carrés).
- `OverallQual` : Qualité générale du matériau et de la finition (sur une échelle de 1 à 10).
- `YearBuilt` : Année de construction originale.
- `TotalBsmtSF` : Surface totale du sous-sol (en pieds carrés).
- `GarageCars` : Taille du garage en capacité de voitures.
- `FullBath` : Nombre de salles de bain complètes au-dessus du niveau du sol.
- `Neighborhood` : Quartier dans Ames, Iowa.

## Containerisation

L'API de prédiction est conteneurisée à l'aide de Docker, ce qui garantit un environnement d'exécution isolé et reproductible.

### Construction de l'image Docker

Depuis le répertoire `API`, vous pouvez construire l'image Docker :

```bash
cd API
docker build -t house-price-api .
```

### Exécution du conteneur Docker

Une fois l'image construite, vous pouvez lancer le conteneur :

```bash
docker run -p 8000:8000 house-price-api
```

L'API sera alors accessible via `http://localhost:8000` sur votre machine hôte.

## Déploiement

Le projet est configuré pour un déploiement continu. Chaque modification poussée sur la branche `main` de votre dépôt GitHub déclenchera automatiquement le pipeline CI/CD, qui inclura la construction d'une nouvelle image Docker et son déploiement sur la plateforme de votre choix (par exemple, Render, comme configuré dans le workflow GitHub Actions).

## Monitoring et maintenance

### Logs

Les logs détaillés des exécutions des notebooks sont automatiquement enregistrés dans le répertoire `logs/`, organisés par année et par mois. Ces logs sont essentiels pour le débogage et le suivi des performances du pipeline.

### Analyse de stabilité

Le notebook `house_price_03_stabilite.ipynb` est prévu pour l'analyse de la stabilité du modèle. Il permet d'évaluer comment les performances du modèle évoluent au fil du temps et de déterminer la fréquence optimale de réentraînement pour maintenir la précision des prédictions.

## Contribution

Les contributions sont les bienvenues ! Pour contribuer à ce projet, veuillez suivre les étapes suivantes :

1. Forker le dépôt du projet.
2. Cloner votre fork localement : `git clone https://github.com/votre_utilisateur/mlops_house_price_prediction.git`.
3. Créer une nouvelle branche pour vos modifications : `git checkout -b feature/votre-fonctionnalite`.
4. Effectuer vos modifications et les commiter : `git commit -am 'Ajout de ma nouvelle fonctionnalité'`.
5. Pousser votre branche vers votre fork : `git push origin feature/votre-fonctionnalite`.
6. Ouvrir une Pull Request depuis votre fork vers le dépôt principal.

## Licence

Ce projet est distribué sous la licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## Support

Pour toute question, suggestion ou problème, n'hésitez pas à ouvrir une issue sur le dépôt GitHub du projet.


