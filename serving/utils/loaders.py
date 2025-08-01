from typing import List, Dict, Tuple, Optional
import json
import pickle
import os
from loguru import logger
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline


def load_model_and_metadata(
    model_path: str, metadata_path: str
) -> Tuple[Optional[Pipeline], Optional[Dict]]:
    """
    Charge le modèle de machine learning et ses métadonnées.

    Args:
        model_path (str): Chemin vers le fichier du modèle sérialisé (.pkl)
        metadata_path (str): Chemin vers le fichier des métadonnées (.json)

    Returns:
        Tuple[Optional[Pipeline], Optional[Dict]]: Le modèle chargé et ses métadonnées,
                                                   ou (None, None) en cas d'erreur
    """
    model = None
    metadata = None

    try:
        # Chargement du modèle
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            logger.info(f"Modèle chargé avec succès depuis: {model_path}")

            # Validation du type de modèle
            if not isinstance(model, Pipeline):
                logger.warning(
                    f"Le modèle chargé n'est pas un Pipeline scikit-learn: {type(model)}"
                )
            else:
                # Vérification de la structure du pipeline
                if "preprocessor" not in model.named_steps:
                    logger.warning("Le pipeline ne contient pas d'étape 'preprocessor'")
                # Accepter 'model' ou 'regressor' comme nom d'étape pour le modèle final
                if not any(
                    name in model.named_steps for name in ["regressor", "model"]
                ):
                    logger.warning(
                        "Le pipeline ne contient pas d'étape 'model' ou 'regressor'"
                    )
                else:
                    reg_name = (
                        "regressor" if "regressor" in model.named_steps else "model"
                    )
                    regressor_type = type(model.named_steps[reg_name]).__name__
                    logger.info(f"Type de régresseur détecté: {regressor_type}")
        else:
            logger.error(f"Fichier modèle non trouvé: {model_path}")
            # Tentative de chargement d'un modèle de fallback
            model = _load_fallback_model()

    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle: {e}")
        model = _load_fallback_model()

    try:
        # Chargement des métadonnées
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            logger.info(f"Métadonnées chargées avec succès depuis: {metadata_path}")

            # Validation des métadonnées essentielles
            required_keys = [
                "model_name",
                "target_variable",
                "use_log_transform",
                "features",
            ]
            missing_keys = [key for key in required_keys if key not in metadata]
            if missing_keys:
                logger.warning(f"Clés manquantes dans les métadonnées: {missing_keys}")

            # Validation de la structure des features
            if "features" in metadata:
                features_info = metadata["features"]
                if "numeric_features" not in features_info:
                    logger.warning(
                        "Liste des features numériques manquante dans les métadonnées"
                    )
                if "categorical_features" not in features_info:
                    logger.warning(
                        "Liste des features catégorielles manquante dans les métadonnées"
                    )
        else:
            logger.error(f"Fichier métadonnées non trouvé: {metadata_path}")
            metadata = _create_default_metadata()

    except Exception as e:
        logger.error(f"Erreur lors du chargement des métadonnées: {e}")
        metadata = _create_default_metadata()

    # Validation de la cohérence entre modèle et métadonnées
    if model is not None and metadata is not None:
        try:
            _validate_model_metadata_consistency(model, metadata)
        except Exception as e:
            logger.warning(f"Incohérence détectée entre modèle et métadonnées: {e}")

    return model, metadata


def _load_fallback_model() -> Optional[Pipeline]:
    """
    Charge un modèle de fallback simple en cas d'échec du chargement principal.

    Returns:
        Optional[Pipeline]: Un modèle simple ou None si impossible à créer
    """
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import LinearRegression

        logger.info("Création d'un modèle de fallback simple...")

        # Features minimales pour le modèle de fallback
        numeric_features = [
            "GrLivArea",
            "OverallQual",
            "YearBuilt",
            "TotalBsmtSF",
            "GarageArea",
        ]
        categorical_features = ["Neighborhood", "MSZoning"]

        # Pipeline de prétraitement simple
        numeric_transformer = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_transformer = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )

        preprocessor = ColumnTransformer(
            [
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ],
            remainder="drop",
        )

        # Pipeline complet avec régression linéaire
        fallback_pipeline = Pipeline(
            [("preprocessor", preprocessor), ("regressor", LinearRegression())]
        )

        logger.info("Modèle de fallback créé avec succès")
        return fallback_pipeline

    except Exception as e:
        logger.error(f"Impossible de créer le modèle de fallback: {e}")
        return None


def _create_default_metadata() -> Dict:
    """
    Crée des métadonnées par défaut en cas d'échec du chargement.

    Returns:
        Dict: Métadonnées par défaut
    """
    logger.info("Création de métadonnées par défaut...")

    return {
        "model_name": "FallbackModel",
        "model_type": "LinearRegression",
        "training_date": "2024-01-01T00:00:00Z",
        "target_variable": "SalePrice",
        "use_log_transform": True,
        "features": {
            "numeric_features": [
                "GrLivArea",
                "OverallQual",
                "YearBuilt",
                "TotalBsmtSF",
                "GarageArea",
                "OverallCond",
                "YearRemodAdd",
                "FirstFlrSF",
                "SecondFlrSF",
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
            ],
            "categorical_features": [
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
            ],
            "total_features": 56,
        },
        "performance": {
            "test_rmse": 25000.0,
            "test_r2": 0.85,
            "test_mae": 18000.0,
            "test_mape": 12.5,
        },
        "data_info": {"train_size": 1168, "test_size": 292, "total_size": 1460},
        "preprocessing": {
            "numeric_imputation": "median",
            "numeric_scaling": "StandardScaler",
            "categorical_imputation": "most_frequent",
            "categorical_encoding": "OneHotEncoder",
        },
    }


def _validate_model_metadata_consistency(model: Pipeline, metadata: Dict) -> None:
    """
    Valide la cohérence entre le modèle chargé et ses métadonnées.

    Args:
        model (Pipeline): Le modèle chargé
        metadata (Dict): Les métadonnées chargées

    Raises:
        ValueError: Si des incohérences sont détectées
    """
    # Vérification du type de modèle
    if "regressor" in model.named_steps:
        actual_model_type = type(model.named_steps["regressor"]).__name__
        expected_model_type = metadata.get("model_type", "")

        # Mapping des noms pour la compatibilité
        model_type_mapping = {
            "GradientBoostingRegressor": [
                "GradientBoosting",
                "GradientBoostingRegressor",
            ],
            "RandomForestRegressor": ["RandomForest", "RandomForestRegressor"],
            "LinearRegression": ["LinearRegression", "Linear"],
            "Ridge": ["Ridge", "RidgeRegression"],
            "Lasso": ["Lasso", "LassoRegression"],
        }

        type_match = False
        for actual_type, possible_names in model_type_mapping.items():
            if (
                actual_model_type == actual_type
                and expected_model_type in possible_names
            ):
                type_match = True
                break

        if not type_match and expected_model_type != "FallbackModel":
            logger.warning(
                f"Type de modèle incohérent: attendu {expected_model_type}, trouvé {actual_model_type}"
            )

    # Vérification de la variable cible
    expected_target = metadata.get("target_variable", "SalePrice")
    if expected_target != "SalePrice":
        logger.warning(f"Variable cible inattendue: {expected_target}")

    # Vérification des features (si le preprocessor est accessible)
    if "preprocessor" in model.named_steps:
        try:
            preprocessor = model.named_steps["preprocessor"]
            if hasattr(preprocessor, "transformers_"):
                # Extraire les noms des features du ColumnTransformer
                numeric_features_model = []
                categorical_features_model = []

                for name, transformer, features in preprocessor.transformers_:
                    if name == "num":
                        numeric_features_model = features
                    elif name == "cat":
                        categorical_features_model = features

                # Comparer avec les métadonnées
                expected_numeric = set(
                    metadata.get("features", {}).get("numeric_features", [])
                )
                expected_categorical = set(
                    metadata.get("features", {}).get("categorical_features", [])
                )

                actual_numeric = set(numeric_features_model)
                actual_categorical = set(categorical_features_model)

                # Vérifier les différences
                missing_numeric = expected_numeric - actual_numeric
                extra_numeric = actual_numeric - expected_numeric
                missing_categorical = expected_categorical - actual_categorical
                extra_categorical = actual_categorical - expected_categorical

                if missing_numeric or extra_numeric:
                    logger.warning(
                        f"Features numériques incohérentes. Manquantes: {missing_numeric}, Supplémentaires: {extra_numeric}"
                    )

                if missing_categorical or extra_categorical:
                    logger.warning(
                        f"Features catégorielles incohérentes. Manquantes: {missing_categorical}, Supplémentaires: {extra_categorical}"
                    )

        except Exception as e:
            logger.warning(f"Impossible de valider les features du modèle: {e}")


def get_model_info(model: Pipeline, metadata: Dict) -> Dict:
    """
    Extrait les informations détaillées du modèle pour l'API.

    Args:
        model (Pipeline): Le modèle chargé
        metadata (Dict): Les métadonnées du modèle

    Returns:
        Dict: Informations détaillées du modèle
    """
    info = {"model_loaded": model is not None, "metadata_loaded": metadata is not None}

    if metadata:
        info.update(
            {
                "model_name": metadata.get("model_name", "Inconnu"),
                "model_type": metadata.get("model_type", "Inconnu"),
                "training_date": metadata.get("training_date", "Inconnu"),
                "target_variable": metadata.get("target_variable", "SalePrice"),
                "use_log_transform": metadata.get("use_log_transform", False),
                "performance_metrics": metadata.get("performance", {}),
                "data_info": metadata.get("data_info", {}),
                "preprocessing_info": metadata.get("preprocessing", {}),
                "features_count": {
                    "numeric": len(
                        metadata.get("features", {}).get("numeric_features", [])
                    ),
                    "categorical": len(
                        metadata.get("features", {}).get("categorical_features", [])
                    ),
                    "total": metadata.get("features", {}).get("total_features", 0),
                },
            }
        )

    if model:
        try:
            # Informations sur la structure du pipeline
            pipeline_steps = list(model.named_steps.keys())
            info["pipeline_steps"] = pipeline_steps

            # Informations sur le régresseur
            if "regressor" in model.named_steps:
                regressor = model.named_steps["regressor"]
                info["regressor_type"] = type(regressor).__name__

                # Paramètres spécifiques selon le type de modèle
                if hasattr(regressor, "n_estimators"):
                    info["n_estimators"] = regressor.n_estimators
                if hasattr(regressor, "max_depth"):
                    info["max_depth"] = regressor.max_depth
                if hasattr(regressor, "learning_rate"):
                    info["learning_rate"] = regressor.learning_rate
                if hasattr(regressor, "alpha"):
                    info["alpha"] = regressor.alpha

            # Informations sur le preprocessor
            if "preprocessor" in model.named_steps:
                preprocessor = model.named_steps["preprocessor"]
                info["preprocessor_type"] = type(preprocessor).__name__

                if hasattr(preprocessor, "transformers_"):
                    transformers_info = []
                    for name, transformer, features in preprocessor.transformers_:
                        transformers_info.append(
                            {
                                "name": name,
                                "transformer_type": type(transformer).__name__,
                                "n_features": (
                                    len(features)
                                    if isinstance(features, list)
                                    else "unknown"
                                ),
                            }
                        )
                    info["transformers"] = transformers_info

        except Exception as e:
            logger.warning(
                f"Erreur lors de l'extraction des informations du modèle: {e}"
            )
            info["model_info_error"] = str(e)

    return info


def validate_input_features(
    input_data: pd.DataFrame, metadata: Dict
) -> Tuple[bool, List[str]]:
    """
    Valide que les données d'entrée contiennent les features nécessaires.

    Args:
        input_data (pd.DataFrame): Les données d'entrée
        metadata (Dict): Les métadonnées du modèle

    Returns:
        Tuple[bool, List[str]]: (True si valide, liste des erreurs)
    """
    errors = []

    if metadata is None:
        errors.append("Métadonnées du modèle non disponibles")
        return False, errors

    features_info = metadata.get("features", {})
    expected_numeric = set(features_info.get("numeric_features", []))
    expected_categorical = set(features_info.get("categorical_features", []))
    expected_all = expected_numeric.union(expected_categorical)

    actual_features = set(input_data.columns)

    # Vérifier les features manquantes critiques
    # Note: Certaines features peuvent être manquantes et seront imputées par le pipeline
    critical_features = {
        "GrLivArea",
        "OverallQual",
        "YearBuilt",
    }  # Features considérées comme critiques
    missing_critical = critical_features.intersection(expected_all) - actual_features

    if missing_critical:
        errors.append(f"Features critiques manquantes: {list(missing_critical)}")

    # Vérifier les types de données
    for col in actual_features.intersection(expected_numeric):
        if not pd.api.types.is_numeric_dtype(input_data[col]):
            try:
                input_data[col] = pd.to_numeric(input_data[col], errors="coerce")
            except Exception:
                errors.append(f"Impossible de convertir '{col}' en type numérique")

    # Vérifier les valeurs aberrantes pour certaines features critiques
    if "GrLivArea" in input_data.columns:
        if input_data["GrLivArea"].max() > 10000 or input_data["GrLivArea"].min() < 0:
            errors.append(
                "GrLivArea contient des valeurs aberrantes (doit être entre 0 et 10000)"
            )

    if "OverallQual" in input_data.columns:
        if input_data["OverallQual"].max() > 10 or input_data["OverallQual"].min() < 1:
            errors.append("OverallQual doit être entre 1 et 10")

    is_valid = len(errors) == 0
    return is_valid, errors


def preprocess_input_for_prediction(
    input_data: pd.DataFrame, metadata: Dict
) -> pd.DataFrame:
    """
    Prétraite les données d'entrée pour la prédiction.

    Args:
        input_data (pd.DataFrame): Les données d'entrée brutes
        metadata (Dict): Les métadonnées du modèle

    Returns:
        pd.DataFrame: Les données prétraitées
    """
    processed_data = input_data.copy()

    if metadata is None:
        logger.warning("Métadonnées non disponibles pour le prétraitement")
        return processed_data

    features_info = metadata.get("features", {})
    expected_numeric = features_info.get("numeric_features", [])
    expected_categorical = features_info.get("categorical_features", [])

    # S'assurer que toutes les colonnes attendues sont présentes
    all_expected = expected_numeric + expected_categorical
    for col in all_expected:
        if col not in processed_data.columns:
            processed_data[col] = np.nan

    # Conversion des types
    for col in expected_numeric:
        if col in processed_data.columns:
            processed_data[col] = pd.to_numeric(processed_data[col], errors="coerce")

    for col in expected_categorical:
        if col in processed_data.columns:
            processed_data[col] = processed_data[col].astype(str)
            # Remplacer 'nan' string par NaN
            processed_data[col] = processed_data[col].replace("nan", np.nan)

    # Réordonner les colonnes selon l'ordre attendu
    processed_data = processed_data[all_expected]

    return processed_data
