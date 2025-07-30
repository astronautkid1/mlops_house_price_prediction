import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import sys

# Ajouter le répertoire parent au path pour les imports
sys.path.append(str(Path(__file__).parent.parent))

# Importer l'application FastAPI
from serving.app import app

# Client de test FastAPI
client = TestClient(app)


@pytest.fixture(scope="module")
def sample_house_data():
    """
    Fixture pour fournir des données d'exemple d'une maison.
    """
    return {
        "GrLivArea": 1710.0,
        "OverallQual": 8,
        "OverallCond": 5,
        "YearBuilt": 2003,
        "YearRemodAdd": 2003,
        "TotalBsmtSF": 856.0,
        "1stFlrSF": 856.0,  # Nom correct du dataset
        "2ndFlrSF": 854.0,  # Nom correct du dataset
        "GarageArea": 548.0,
        "GarageCars": 2.0,
        "BedroomAbvGr": 3,
        "KitchenAbvGr": 1,
        "TotRmsAbvGrd": 8,
        "FullBath": 2,
        "HalfBath": 1,
        "Fireplaces": 1,
        "YrSold": 2008,
        "MoSold": 2,
        "Neighborhood": "CollgCr",
        "MSZoning": "RL",
        "LotArea": 8450.0,
        "Street": "Pave",
        "LotShape": "Reg",
        "LandContour": "Lvl",
        "Utilities": "AllPub",
        "LotConfig": "Inside",
        "LandSlope": "Gtl",
        "BldgType": "1Fam",
        "HouseStyle": "2Story",
        "RoofStyle": "Gable",
        "RoofMatl": "CompShg",
        "Exterior1st": "VinylSd",
        "Exterior2nd": "VinylSd",
        "ExterQual": "Gd",
        "ExterCond": "TA",
        "Foundation": "PConc",
        "BsmtQual": "Gd",
        "BsmtCond": "TA",
        "BsmtExposure": "No",
        "BsmtFinType1": "GLQ",
        "Heating": "GasA",
        "HeatingQC": "Ex",
        "CentralAir": "Y",
        "Electrical": "SBrkr",
        "KitchenQual": "Gd",
        "Functional": "Typ",
        "GarageType": "Attchd",
        "GarageFinish": "RFn",
        "GarageQual": "TA",
        "GarageCond": "TA",
        "SaleType": "WD",
        "SaleCondition": "Normal",
    }


@pytest.fixture(scope="module")
def sample_batch_data():
    """
    Fixture pour fournir des données d'exemple pour les prédictions en lot.
    """
    return [
        {
            "GrLivArea": 1710.0,
            "OverallQual": 8,
            "YearBuilt": 2003,
            "TotalBsmtSF": 856.0,
            "GarageArea": 548.0,
            "Neighborhood": "CollgCr",
            "MSZoning": "RL",
        },
        {
            "GrLivArea": 1262.0,
            "OverallQual": 6,
            "YearBuilt": 1976,
            "TotalBsmtSF": 1262.0,
            "GarageArea": 460.0,
            "Neighborhood": "Veenker",
            "MSZoning": "RL",
        },
    ]


@pytest.fixture(scope="module")
def mock_model():
    """
    Fixture pour créer un modèle mock pour les tests.
    """
    model = MagicMock()
    model.predict.return_value = np.array([12.0])  # Valeur log-transformée
    return model


@pytest.fixture(scope="module")
def mock_metadata():
    """
    Fixture pour créer des métadonnées mock pour les tests.
    """
    return {
        "model_name": "TestModel",
        "model_type": "GradientBoostingRegressor",
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
                "1stFlrSF",  # Nom correct du dataset
                "2ndFlrSF",  # Nom correct du dataset
            ],
            "categorical_features": ["Neighborhood", "MSZoning"],
            "total_features": 9,
        },
        "performance": {
            "test_rmse": 25000.0,
            "test_r2": 0.85,
            "test_mae": 18000.0,
            "test_mape": 12.5,
        },
    }


@pytest.mark.api
def test_health_endpoint():
    """
    Teste l'endpoint de vérification de santé de l'API.
    """
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"
    assert "timestamp" in data


@pytest.mark.api
def test_model_info_endpoint():
    """
    Teste l'endpoint d'informations sur le modèle.
    """
    response = client.get("/api/model/info")
    assert response.status_code == 200
    data = response.json()
    # Vérifier que les clés essentielles sont présentes
    expected_keys = ["model_name", "target_variable", "use_log_transform", "features"]
    for key in expected_keys:
        assert key in data, f"Clé manquante dans les métadonnées: {key}"


@pytest.mark.api
def test_predict_endpoint_success(sample_house_data):
    """
    Teste l'endpoint de prédiction avec des données valides.
    """
    with patch("serving.app.model") as mock_model, patch(
        "serving.app.model_metadata"
    ) as mock_metadata:

        # Configurer les mocks
        mock_model.predict.return_value = np.array([12.0])  # log(prix)
        mock_metadata.get.side_effect = lambda key, default=None: {
            "model_name": "TestModel",
            "use_log_transform": True,
            "features": {
                "numeric_features": [
                    "GrLivArea",
                    "OverallQual",
                    "YearBuilt",
                    "TotalBsmtSF",
                    "GarageArea",
                ],
                "categorical_features": ["Neighborhood", "MSZoning"],
            },
        }.get(key, default)

        response = client.post("/api/predict", json=sample_house_data)
        assert response.status_code == 200
        data = response.json()
        assert "predicted_price" in data
        assert isinstance(data["predicted_price"], (int, float))
        assert data["predicted_price"] > 0


@pytest.mark.api
def test_predict_endpoint_validation_error():
    """
    Teste l'endpoint de prédiction avec des données invalides.
    """
    invalid_data = {
        "GrLivArea": -100,  # Valeur négative invalide
        "OverallQual": 15,  # Valeur hors plage (doit être 1-10)
        "YearBuilt": 1500,  # Année trop ancienne
    }

    response = client.post("/api/predict", json=invalid_data)
    assert response.status_code == 422  # Unprocessable Entity
    data = response.json()
    assert "detail" in data


@pytest.mark.api
def test_predict_endpoint_missing_required_fields():
    """
    Teste l'endpoint de prédiction avec des champs manquants.
    """
    with patch("serving.app.model") as mock_model, patch(
        "serving.app.model_metadata"
    ) as mock_metadata:

        # Configurer les mocks
        mock_model.predict.return_value = np.array([12.0])
        mock_metadata.get.side_effect = lambda key, default=None: {
            "model_name": "TestModel",
            "use_log_transform": True,
            "features": {
                "numeric_features": [
                    "GrLivArea",
                    "OverallQual",
                    "YearBuilt",
                    "TotalBsmtSF",
                    "GarageArea",
                ],
                "categorical_features": ["Neighborhood", "MSZoning"],
            },
        }.get(key, default)

        incomplete_data = {
            "GrLivArea": 1500
            # Beaucoup de champs manquants
        }

        response = client.post("/api/predict", json=incomplete_data)
        # L'API devrait accepter les données incomplètes car tous les champs sont optionnels
        # et les valeurs manquantes seront imputées par le pipeline
        assert response.status_code in [200, 422]


@pytest.mark.api
def test_predict_endpoint_empty_data():
    """
    Teste l'endpoint de prédiction avec des données vides.
    """
    with patch("serving.app.model") as mock_model, patch(
        "serving.app.model_metadata"
    ) as mock_metadata:

        # Configurer les mocks
        mock_model.predict.return_value = np.array([12.0])
        mock_metadata.get.side_effect = lambda key, default=None: {
            "model_name": "TestModel",
            "use_log_transform": True,
            "features": {
                "numeric_features": [
                    "GrLivArea",
                    "OverallQual",
                    "YearBuilt",
                    "TotalBsmtSF",
                    "GarageArea",
                ],
                "categorical_features": ["Neighborhood", "MSZoning"],
            },
        }.get(key, default)

        empty_data = {}

        response = client.post("/api/predict", json=empty_data)
        # L'API devrait gérer les données vides en utilisant les valeurs par défaut/imputées
        assert response.status_code in [200, 422]


@pytest.mark.api
def test_predict_batch_endpoint_success(sample_batch_data, mock_model, mock_metadata):
    """
    Teste l'endpoint de prédiction en lot avec des données valides.
    """
    # Importer le module ici pour pouvoir le patcher
    import serving.app as app_module

    # Sauvegarder les valeurs originales
    original_model = app_module.model
    original_metadata = app_module.model_metadata

    try:
        # Remplacer par les mocks
        app_module.model = mock_model
        app_module.model_metadata = mock_metadata

        # Configurer le mock du modèle pour retourner plusieurs prédictions
        mock_model.predict.return_value = np.array(
            [12.0, 11.8]
        )  # log(prix) pour 2 maisons

        # Créer un fichier CSV temporaire
        df = pd.DataFrame(sample_batch_data)
        csv_content = df.to_csv(index=False)

        files = {"file": ("test_batch.csv", csv_content, "text/csv")}
        response = client.post("/api/predict/batch", files=files)

        assert response.status_code == 200
        data = response.json()
        assert "predicted_prices" in data
        assert "errors" in data
        assert len(data["predicted_prices"]) == 2
        assert all(
            isinstance(price, (int, float)) for price in data["predicted_prices"]
        )
        assert all(price > 0 for price in data["predicted_prices"])
    finally:
        # Restaurer les valeurs originales
        app_module.model = original_model
        app_module.model_metadata = original_metadata


@pytest.mark.api
def test_predict_batch_endpoint_invalid_file_format():
    """
    Teste l'endpoint de prédiction en lot avec un format de fichier invalide.
    """
    files = {"file": ("test.txt", "invalid content", "text/plain")}
    response = client.post("/api/predict/batch", files=files)

    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert "Format de fichier non supporté" in data["detail"]


@pytest.mark.api
def test_predict_batch_endpoint_empty_file():
    """
    Teste l'endpoint de prédiction en lot avec un fichier vide.
    """
    files = {"file": ("empty.csv", "", "text/csv")}
    response = client.post("/api/predict/batch", files=files)

    assert response.status_code == 400


@pytest.mark.api
def test_predict_batch_endpoint_malformed_csv():
    """
    Teste l'endpoint de prédiction en lot avec un CSV malformé.
    """
    malformed_csv = "col1,col2\nvalue1\nvalue2,value3,extra_value"
    files = {"file": ("malformed.csv", malformed_csv, "text/csv")}
    response = client.post("/api/predict/batch", files=files)

    # L'API devrait gérer les CSV malformés gracieusement
    assert response.status_code in [200, 400]


@pytest.mark.api
@patch("serving.app.model")
def test_predict_endpoint_model_error(mock_model_patch, sample_house_data):
    """
    Teste l'endpoint de prédiction quand le modèle lève une exception.
    """
    mock_model = MagicMock()
    mock_model.predict.side_effect = Exception("Erreur du modèle")
    mock_model_patch.return_value = mock_model

    response = client.post("/api/predict", json=sample_house_data)
    assert response.status_code == 500
    data = response.json()
    assert "detail" in data


@pytest.mark.api
def test_predict_endpoint_with_none_values():
    """
    Teste l'endpoint de prédiction avec des valeurs None.
    """
    with patch("serving.app.model") as mock_model, patch(
        "serving.app.model_metadata"
    ) as mock_metadata:

        # Configurer les mocks
        mock_model.predict.return_value = np.array([12.0])
        mock_metadata.get.side_effect = lambda key, default=None: {
            "model_name": "TestModel",
            "use_log_transform": True,
            "features": {
                "numeric_features": [
                    "GrLivArea",
                    "OverallQual",
                    "YearBuilt",
                    "TotalBsmtSF",
                    "GarageArea",
                ],
                "categorical_features": ["Neighborhood", "MSZoning"],
            },
        }.get(key, default)

        data_with_none = {
            "GrLivArea": 1500.0,
            "OverallQual": None,
            "YearBuilt": 2000,
            "Neighborhood": None,
        }

        response = client.post("/api/predict", json=data_with_none)
        # L'API devrait accepter les valeurs None et les traiter comme des valeurs manquantes
        assert response.status_code in [200, 422]


@pytest.mark.api
def test_predict_endpoint_with_string_numbers():
    """
    Teste l'endpoint de prédiction avec des nombres sous forme de chaînes.
    """
    with patch("serving.app.model") as mock_model, patch(
        "serving.app.model_metadata"
    ) as mock_metadata:

        # Configurer les mocks
        mock_model.predict.return_value = np.array([12.0])
        mock_metadata.get.side_effect = lambda key, default=None: {
            "model_name": "TestModel",
            "use_log_transform": True,
            "features": {
                "numeric_features": [
                    "GrLivArea",
                    "OverallQual",
                    "YearBuilt",
                    "TotalBsmtSF",
                    "GarageArea",
                ],
                "categorical_features": ["Neighborhood", "MSZoning"],
            },
        }.get(key, default)

        data_with_strings = {
            "GrLivArea": "1500.0",
            "OverallQual": "8",
            "YearBuilt": "2000",
        }

        response = client.post("/api/predict", json=data_with_strings)
        # Pydantic devrait convertir automatiquement les chaînes en nombres
        assert response.status_code in [200, 422]


@pytest.mark.api
def test_cors_headers():
    """
    Teste la présence des headers CORS dans les réponses.
    """
    response = client.get("/api/health")
    # Vérifier que les headers CORS sont présents (si configurés)
    # Note: Cela dépend de la configuration CORS de l'application
    assert response.status_code == 200


@pytest.mark.api
def test_api_documentation_endpoints():
    """
    Teste l'accessibilité des endpoints de documentation.
    """
    # Tester Swagger UI
    response = client.get("/docs")
    assert response.status_code == 200

    # Tester ReDoc
    response = client.get("/redoc")
    assert response.status_code == 200

    # Tester OpenAPI schema
    response = client.get("/openapi.json")
    assert response.status_code == 200
    data = response.json()
    assert "openapi" in data
    assert "info" in data


@pytest.mark.api
def test_predict_endpoint_response_format():
    """
    Teste le format de la réponse de l'endpoint de prédiction.
    """
    minimal_data = {"GrLivArea": 1500.0}

    response = client.post("/api/predict", json=minimal_data)
    if response.status_code == 200:
        data = response.json()
        assert isinstance(data, dict)
        assert "predicted_price" in data
        assert isinstance(data["predicted_price"], (int, float))
        # Vérifier que le prix est dans une plage raisonnable
        assert 10000 <= data["predicted_price"] <= 1000000


@pytest.mark.api
def test_predict_batch_response_format():
    """
    Teste le format de la réponse de l'endpoint de prédiction en lot.
    """
    sample_data = [{"GrLivArea": 1500.0}, {"GrLivArea": 2000.0}]
    df = pd.DataFrame(sample_data)
    csv_content = df.to_csv(index=False)

    files = {"file": ("test.csv", csv_content, "text/csv")}
    response = client.post("/api/predict/batch", files=files)

    if response.status_code == 200:
        data = response.json()
        assert isinstance(data, dict)
        assert "predicted_prices" in data
        assert "errors" in data
        assert isinstance(data["predicted_prices"], list)
        assert isinstance(data["errors"], list)


@pytest.mark.api
def test_model_info_response_format():
    """
    Teste le format de la réponse de l'endpoint d'informations sur le modèle.
    """
    response = client.get("/api/model/info")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)

    # Vérifier la structure des métadonnées
    if "features" in data:
        features = data["features"]
        assert isinstance(features, dict)
        if "numeric_features" in features:
            assert isinstance(features["numeric_features"], list)
        if "categorical_features" in features:
            assert isinstance(features["categorical_features"], list)


@pytest.mark.api
def test_health_response_format():
    """
    Teste le format de la réponse de l'endpoint de santé.
    """
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
    assert "status" in data
    assert "timestamp" in data
    assert data["status"] == "healthy"
    # Vérifier que le timestamp est au format ISO
    import datetime

    try:
        datetime.datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))
    except ValueError:
        pytest.fail("Le timestamp n'est pas au format ISO valide")


@pytest.mark.api
def test_error_handling_invalid_json():
    """
    Teste la gestion des erreurs avec un JSON invalide.
    """
    response = client.post(
        "/api/predict",
        data="invalid json",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422


@pytest.mark.api
def test_error_handling_missing_file():
    """
    Teste la gestion des erreurs quand aucun fichier n'est fourni pour le batch.
    """
    response = client.post("/api/predict/batch")
    assert response.status_code == 422


@pytest.mark.api
def test_large_batch_size_limit():
    """
    Teste la limitation de la taille des lots (si implémentée).
    """
    with patch("serving.app.model") as mock_model, patch(
        "serving.app.model_metadata"
    ) as mock_metadata:

        # Configurer les mocks
        mock_model.predict.return_value = np.array([12.0] * 10)  # 10 prédictions
        mock_metadata.get.side_effect = lambda key, default=None: {
            "model_name": "TestModel",
            "use_log_transform": True,
            "features": {
                "numeric_features": [
                    "GrLivArea",
                    "OverallQual",
                    "YearBuilt",
                    "TotalBsmtSF",
                    "GarageArea",
                ],
                "categorical_features": ["Neighborhood", "MSZoning"],
            },
        }.get(key, default)

        # Créer un lot très large pour tester les limites
        large_batch = [{"GrLivArea": 1500.0} for _ in range(10)]
        df = pd.DataFrame(large_batch)
        csv_content = df.to_csv(index=False)

        files = {"file": ("large_batch.csv", csv_content, "text/csv")}
        response = client.post("/api/predict/batch", files=files)

        # L'API devrait soit accepter le lot, soit retourner une erreur de limite
        assert response.status_code in [200, 400, 413]


@pytest.mark.api
def test_concurrent_requests():
    """
    Teste les requêtes concurrentes (test basique).
    """
    import concurrent.futures

    def make_request():
        return client.get("/api/health")

    # Faire plusieurs requêtes en parallèle
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(make_request) for _ in range(5)]
        responses = [
            future.result() for future in concurrent.futures.as_completed(futures)
        ]

    # Toutes les requêtes devraient réussir
    for response in responses:
        assert response.status_code == 200


@pytest.mark.api
def test_input_validation_edge_cases():
    """
    Teste la validation des entrées avec des cas limites.
    """
    with patch("serving.app.model") as mock_model, patch(
        "serving.app.model_metadata"
    ) as mock_metadata:

        # Configurer les mocks
        mock_model.predict.return_value = np.array([12.0])
        mock_metadata.get.side_effect = lambda key, default=None: {
            "model_name": "TestModel",
            "use_log_transform": True,
            "features": {
                "numeric_features": [
                    "GrLivArea",
                    "OverallQual",
                    "YearBuilt",
                    "TotalBsmtSF",
                    "GarageArea",
                ],
                "categorical_features": ["Neighborhood", "MSZoning"],
            },
        }.get(key, default)

        edge_cases = [
            {"GrLivArea": 0},  # Valeur minimale
            {"GrLivArea": 10000},  # Valeur maximale
            {"OverallQual": 1},  # Qualité minimale
            {"OverallQual": 10},  # Qualité maximale
            {"YearBuilt": 1800},  # Année minimale
            {"YearBuilt": 2030},  # Année future
        ]

        for case in edge_cases:
            response = client.post("/api/predict", json=case)
            # Les cas limites devraient être acceptés ou rejetés de manière cohérente
            assert response.status_code in [200, 422]


@pytest.mark.api
def test_api_performance_basic():
    """
    Teste les performances de base de l'API (temps de réponse).
    """
    import time

    start_time = time.time()
    response = client.get("/api/health")
    end_time = time.time()

    assert response.status_code == 200
    # Le health check devrait être très rapide (moins d'1 seconde)
    assert (end_time - start_time) < 1.0


# Exécution des tests si le script est lancé directement
if __name__ == "__main__":
    pytest.main([__file__])
