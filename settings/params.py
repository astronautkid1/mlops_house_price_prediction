"""
Configuration et paramètres pour le projet de prédiction des prix des maisons.

Ce module centralise tous les paramètres de configuration utilisés dans le projet,
incluant les chemins de fichiers, les paramètres de modélisation, et les constantes
utilisées à travers les différents composants du pipeline MLOps.
"""

from pathlib import Path
import os
from typing import Dict, List, Any, Optional
import pendulum

# =============================================================================
# CONFIGURATION GÉNÉRALE DU PROJET
# =============================================================================

PROJECT_NAME = "House Price Prediction"
PROJECT_VERSION = "1.0.0"
PROJECT_DESCRIPTION = "Prédiction des prix des maisons à Ames, Iowa"
PROJECT_AUTHOR = "Data Science Team"

# Date et timezone
TIMEZONE = "UTC"
CURRENT_DATE = pendulum.now(tz=TIMEZONE)

# =============================================================================
# CHEMINS ET RÉPERTOIRES
# =============================================================================

# Répertoire racine du projet
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "house_price_dataset"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
OUTPUT_DIR = PROJECT_ROOT / "output_files"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
LOGS_DIR = PROJECT_ROOT / "logs"
API_DIR = PROJECT_ROOT / "API"
TESTS_DIR = PROJECT_ROOT / "tests"

# Fichiers de données
DATASET_FILENAME = "ames_housing.csv"
DATASET_PATH = DATA_DIR / DATASET_FILENAME

# Fichiers de sortie
ENRICHED_DATASET_FILENAME = "house_price_01_analyse_dataset_enrichi.csv"
ANALYSIS_METADATA_FILENAME = "house_price_01_analyse_metadata.json"
MODEL_COMPARISON_FILENAME = "house_price_02_essais_comparaison.csv"
PREDICTIONS_EXAMPLE_FILENAME = "house_price_02_essais_predictions_exemple.csv"

# =============================================================================
# PARAMÈTRES DE DONNÉES
# =============================================================================

# Variable cible
TARGET_VARIABLE = "SalePrice"

# Variables numériques principales (basées sur l'analyse exploratoire)
MAIN_NUMERIC_FEATURES = [
    "GrLivArea",
    "OverallQual",
    "YearBuilt",
    "TotalBsmtSF",
    "GarageArea",
    "OverallCond",
    "YearRemodAdd",
    "1stFlrSF",  # Nom correct du dataset
    "2ndFlrSF",  # Nom correct du dataset
    "BsmtFinSF1",
    "LotFrontage",
    "LotArea",
    "MasVnrArea",
    "BsmtUnfSF",
    "GarageCars",
    "BedroomAbvGr",
    "KitchenAbvGr",
    "TotRmsAbvGrd",
    "Fireplaces",
    "FullBath",
    "HalfBath",
    "BsmtFullBath",
    "BsmtHalfBath",
    "YrSold",
    "MoSold",
    "PoolArea",
    "ScreenPorch",
    "3SsnPorch",  # Nom correct du dataset
    "EnclosedPorch",
    "OpenPorchSF",
    "WoodDeckSF",
    "LowQualFinSF",
    "BsmtFinSF2",
    "GarageYrBlt",
    "MiscVal",  # Colonne présente dans le dataset
]

# Variables catégorielles principales
MAIN_CATEGORICAL_FEATURES = [
    "MSZoning",
    "Street",
    "LotShape",
    "LandContour",
    "Utilities",
    "LotConfig",
    "LandSlope",
    "Neighborhood",
    "Condition1",
    "BldgType",
    "HouseStyle",
    "RoofStyle",
    "Exterior1st",
    "Exterior2nd",
    "ExterQual",
    "ExterCond",
    "Foundation",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "Heating",
    "HeatingQC",
    "CentralAir",
    "Electrical",
    "KitchenQual",
    "Functional",
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "SaleType",
    "SaleCondition",
    "MSSubClass",
    "Alley",
    "Condition2",
    "RoofMatl",
    "MasVnrType",
    "BsmtFinType2",
    "FireplaceQu",
    "PoolQC",
    "Fence",
    "MiscFeature",
    "PavedDrive",  # Colonne présente dans le dataset
]

# Variables dérivées (feature engineering)
DERIVED_FEATURES = [
    "HouseAge",
    "YearsSinceRemod",
    "TotalSF",
    "QualityScore",
    "TotalBathrooms",
    "HasGarage",
    "HasBasement",
    "HasFireplace",
    "LivingAreaRatio",
]

# Seuils pour le filtrage des variables catégorielles
MAX_CATEGORICAL_CARDINALITY = 25
MAX_MISSING_PERCENTAGE = 0.5

# =============================================================================
# PARAMÈTRES DE MODÉLISATION
# =============================================================================

# Division des données
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2
RANDOM_STATE = 42

# Validation croisée
CV_FOLDS = 5
CV_SCORING = "neg_mean_squared_error"

# Transformation de la variable cible
USE_LOG_TRANSFORM = True  # Basé sur l'analyse de la distribution

# Prétraitement
NUMERIC_IMPUTATION_STRATEGY = "median"
CATEGORICAL_IMPUTATION_STRATEGY = "most_frequent"
NUMERIC_SCALER = "RobustScaler"  # Robuste aux outliers
CATEGORICAL_ENCODER = "OneHotEncoder"

# Détection d'outliers
OUTLIER_METHOD = "IQR"  # ou "zscore"
OUTLIER_THRESHOLD = 1.5  # pour IQR
ZSCORE_THRESHOLD = 3.0  # pour Z-score

# =============================================================================
# CONFIGURATION DES MODÈLES
# =============================================================================

# Modèles à tester avec leurs hyperparamètres
MODELS_CONFIG = {
    "LinearRegression": {
        "class": "sklearn.linear_model.LinearRegression",
        "params": {},
        "description": "Régression linéaire simple - Modèle de référence",
    },
    "Ridge": {
        "class": "sklearn.linear_model.Ridge",
        "params": {"regressor__alpha": [0.1, 1.0, 10.0, 100.0]},
        "description": "Régression Ridge avec régularisation L2",
    },
    "Lasso": {
        "class": "sklearn.linear_model.Lasso",
        "params": {
            "regressor__alpha": [0.01, 0.1, 1.0, 10.0],
            "regressor__max_iter": [2000],
        },
        "description": "Régression Lasso avec régularisation L1",
    },
    "ElasticNet": {
        "class": "sklearn.linear_model.ElasticNet",
        "params": {
            "regressor__alpha": [0.01, 0.1, 1.0],
            "regressor__l1_ratio": [0.1, 0.5, 0.9],
            "regressor__max_iter": [2000],
        },
        "description": "Régression ElasticNet combinant L1 et L2",
    },
    "RandomForest": {
        "class": "sklearn.ensemble.RandomForestRegressor",
        "params": {
            "regressor__n_estimators": [100, 200],
            "regressor__max_depth": [10, 20, None],
            "regressor__min_samples_split": [2, 5],
            "regressor__min_samples_leaf": [1, 2],
        },
        "description": "Random Forest - Ensemble d'arbres de décision",
    },
    "GradientBoosting": {
        "class": "sklearn.ensemble.GradientBoostingRegressor",
        "params": {
            "regressor__n_estimators": [100, 200],
            "regressor__learning_rate": [0.05, 0.1, 0.2],
            "regressor__max_depth": [3, 5, 7],
        },
        "description": "Gradient Boosting - Ensemble séquentiel d'arbres",
    },
    "ExtraTrees": {
        "class": "sklearn.ensemble.ExtraTreesRegressor",
        "params": {
            "regressor__n_estimators": [100, 200],
            "regressor__max_depth": [10, 20, None],
            "regressor__min_samples_split": [2, 5],
        },
        "description": "Extra Trees - Arbres extrêmement randomisés",
    },
    "SVR": {
        "class": "sklearn.svm.SVR",
        "params": {
            "regressor__C": [0.1, 1.0, 10.0],
            "regressor__epsilon": [0.01, 0.1, 0.2],
            "regressor__kernel": ["rbf", "linear"],
        },
        "description": "Support Vector Regression",
    },
}

# Modèle par défaut (fallback)
DEFAULT_MODEL = "GradientBoosting"

# =============================================================================
# MÉTRIQUES D'ÉVALUATION
# =============================================================================

# Métriques principales
PRIMARY_METRICS = ["RMSE", "R2", "MAE", "MAPE"]

# Seuils de performance acceptables
PERFORMANCE_THRESHOLDS = {
    "RMSE": 30000,  # Erreur quadratique moyenne acceptable
    "R2": 0.80,  # R² minimum acceptable
    "MAE": 20000,  # Erreur absolue moyenne acceptable
    "MAPE": 15.0,  # Erreur absolue moyenne en pourcentage
}

# =============================================================================
# CONFIGURATION MLFLOW
# =============================================================================

# Expérimentation
MLFLOW_EXPERIMENT_NAME = "House Price Prediction"
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"  # ou une URI distante
MLFLOW_ARTIFACT_ROOT = "./mlruns"

# Tags par défaut pour les runs
DEFAULT_MLFLOW_TAGS = {
    "project": PROJECT_NAME,
    "version": PROJECT_VERSION,
    "target": TARGET_VARIABLE,
    "dataset": "Ames Housing",
}

# =============================================================================
# CONFIGURATION API
# =============================================================================

# Serveur
API_HOST = "0.0.0.0"
API_PORT = 8000
API_WORKERS = 1
API_RELOAD = False

# Modèle pour l'API
API_MODEL_PATH = MODELS_DIR / "house_price_best_model_gradientboosting.pkl"
API_METADATA_PATH = MODELS_DIR / "house_price_best_model_gradientboosting_metadata.json"

# Limites de l'API
MAX_BATCH_SIZE = 1000
MAX_REQUEST_SIZE = 10 * 1024 * 1024  # 10 MB

# CORS
CORS_ORIGINS = ["*"]  # À restreindre en production
CORS_METHODS = ["GET", "POST"]
CORS_HEADERS = ["*"]

# =============================================================================
# CONFIGURATION DES TESTS
# =============================================================================

# Données de test
TEST_DATA_SIZE = 100
TEST_RANDOM_STATE = 123

# Seuils pour les tests de qualité des données
DATA_QUALITY_THRESHOLDS = {
    "max_missing_percentage": 0.8,
    "min_unique_values": 2,
    "max_unique_values": 1000,
    "min_numeric_range": 0.01,
    "max_outlier_percentage": 0.1,
}

# =============================================================================
# CONFIGURATION DU LOGGING
# =============================================================================

# Format des logs
LOG_FORMAT = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS!UTC}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - {message}"

# Niveaux de log
LOG_LEVEL = "INFO"
LOG_ROTATION = "1 day"
LOG_RETENTION = "30 days"

# Fichiers de log
LOG_FILES = {
    "general": LOGS_DIR / "house_price_general.log",
    "api": LOGS_DIR / "house_price_api.log",
    "training": LOGS_DIR / "house_price_training.log",
    "errors": LOGS_DIR / "house_price_errors.log",
}

# =============================================================================
# CONFIGURATION CI/CD
# =============================================================================

# GitHub Actions
GITHUB_WORKFLOW_TRIGGERS = ["push", "pull_request"]
GITHUB_PYTHON_VERSION = "3.11"
GITHUB_OS = "ubuntu-latest"

# Docker
DOCKER_IMAGE_NAME = "house-price-api"
DOCKER_TAG = "latest"
DOCKER_PORT = 8000

# Déploiement
DEPLOYMENT_PLATFORM = "render"  # ou "heroku", "aws", etc.
DEPLOYMENT_REGION = "us-east-1"

# =============================================================================
# VARIABLES D'ENVIRONNEMENT
# =============================================================================

# Variables d'environnement avec valeurs par défaut
ENV_VARS = {
    "MODEL_PATH": str(API_MODEL_PATH),
    "MODEL_METADATA_PATH": str(API_METADATA_PATH),
    "MLFLOW_TRACKING_URI": MLFLOW_TRACKING_URI,
    "LOG_LEVEL": LOG_LEVEL,
    "API_HOST": API_HOST,
    "API_PORT": str(API_PORT),
    "CORS_ORIGINS": ",".join(CORS_ORIGINS),
    "MAX_BATCH_SIZE": str(MAX_BATCH_SIZE),
}

# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================


def get_env_var(key: str, default: Optional[str] = None) -> str:
    """Récupère une variable d'environnement avec une valeur par défaut."""
    return os.getenv(key, default or ENV_VARS.get(key, ""))


def create_directories() -> None:
    """Crée tous les répertoires nécessaires au projet."""
    directories = [
        DATA_DIR,
        OUTPUT_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        LOGS_DIR,
        NOTEBOOKS_DIR,
        API_DIR,
        TESTS_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def get_model_config(model_name: str) -> Dict[str, Any]:
    """Récupère la configuration d'un modèle spécifique."""
    return MODELS_CONFIG.get(model_name, MODELS_CONFIG[DEFAULT_MODEL])


def get_all_features() -> List[str]:
    """Retourne la liste complète de toutes les features."""
    return MAIN_NUMERIC_FEATURES + MAIN_CATEGORICAL_FEATURES + DERIVED_FEATURES


def get_feature_types() -> Dict[str, List[str]]:
    """Retourne un dictionnaire des features par type."""
    return {
        "numeric": MAIN_NUMERIC_FEATURES
        + [
            f
            for f in DERIVED_FEATURES
            if f not in ["HasGarage", "HasBasement", "HasFireplace"]
        ],
        "categorical": MAIN_CATEGORICAL_FEATURES,
        "binary": ["HasGarage", "HasBasement", "HasFireplace"],
    }


def validate_config() -> bool:
    """Valide la configuration du projet."""
    try:
        # Vérifier que les répertoires essentiels existent ou peuvent être créés
        create_directories()

        # Vérifier que les paramètres critiques sont définis
        assert TARGET_VARIABLE is not None, "TARGET_VARIABLE doit être défini"
        assert (
            len(MAIN_NUMERIC_FEATURES) > 0
        ), "Au moins une feature numérique doit être définie"
        assert 0 < TEST_SIZE < 1, "TEST_SIZE doit être entre 0 et 1"
        assert CV_FOLDS > 1, "CV_FOLDS doit être supérieur à 1"

        return True

    except Exception as e:
        print(f"Erreur de validation de la configuration: {e}")
        return False


# =============================================================================
# INITIALISATION
# =============================================================================

# Créer les répertoires au moment de l'import
create_directories()

# Validation de la configuration
if not validate_config():
    raise RuntimeError(
        "Configuration invalide. Vérifiez les paramètres dans settings/params.py"
    )

# Export des constantes principales pour faciliter l'import
__all__ = [
    "PROJECT_NAME",
    "PROJECT_VERSION",
    "TARGET_VARIABLE",
    "MAIN_NUMERIC_FEATURES",
    "MAIN_CATEGORICAL_FEATURES",
    "DERIVED_FEATURES",
    "MODELS_CONFIG",
    "PRIMARY_METRICS",
    "PERFORMANCE_THRESHOLDS",
    "API_HOST",
    "API_PORT",
    "MLFLOW_EXPERIMENT_NAME",
    "get_env_var",
    "get_model_config",
    "get_all_features",
    "get_feature_types",
]
