import pandas as pd
import numpy as np
import pytest
from pathlib import Path
from loguru import logger
from typing import List

# Importation des paramètres du projet
import sys

sys.path.append(str(Path(__file__).parent.parent / "settings"))
from params import (
    DATA_DIR,
    TARGET_VARIABLE,
    MAIN_NUMERIC_FEATURES,
    MAIN_CATEGORICAL_FEATURES,
    DERIVED_FEATURES,
    DATA_QUALITY_THRESHOLDS,
)

logger.configure(
    handlers=[
        {
            "sink": sys.stderr,
            "format": "<green>{time}</green> <level>{level}</level> <red>{message}</red>",
        }
    ]
)


@pytest.fixture(scope="module")
def sample_data():
    """
    Fixture pour charger le jeu de données Ames Housing.
    Tente de charger le dataset enrichi en premier, sinon le dataset original.
    """
    # Essayer plusieurs chemins pour le fichier enrichi
    possible_enriched_paths = [
        Path(__file__).parent.parent
        / "output_files"
        / "house_price_01_analyse_dataset_enrichi.csv",
        Path(__file__).parent.parent.parent
        / "output_files"
        / "house_price_01_analyse_dataset_enrichi.csv",
        Path.cwd() / "output_files" / "house_price_01_analyse_dataset_enrichi.csv",
    ]

    enriched_file = None
    for path in possible_enriched_paths:
        if path.exists():
            enriched_file = path
            break

    original_file = DATA_DIR / "ames_housing.csv"

    if enriched_file:
        df = pd.read_csv(enriched_file)
        logger.info(f"Dataset enrichi chargé depuis: {enriched_file}")
    elif original_file.exists():
        df = pd.read_csv(original_file)
        logger.info(f"Dataset original chargé depuis: {original_file}")
        # Ajouter les features dérivées minimales si le dataset original est chargé
        if "YearBuilt" in df.columns and "YrSold" in df.columns:
            df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
        if "YearRemodAdd" in df.columns and "YrSold" in df.columns:
            df["YearsSinceRemod"] = df["YrSold"] - df["YearRemodAdd"]
        if "TotalBsmtSF" in df.columns:
            df["HasBasement"] = (df["TotalBsmtSF"] > 0).astype(int)
        if "GarageArea" in df.columns:
            df["HasGarage"] = (df["GarageArea"] > 0).astype(int)
        if "Fireplaces" in df.columns:
            df["HasFireplace"] = (df["Fireplaces"] > 0).astype(int)
        if "OverallQual" in df.columns and "OverallCond" in df.columns:
            df["QualityScore"] = df["OverallQual"] * df["OverallCond"]
        if (
            "1stFlrSF" in df.columns
            and "2ndFlrSF" in df.columns
            and "TotalBsmtSF" in df.columns
        ):
            df["TotalSF"] = df["1stFlrSF"] + df["2ndFlrSF"] + df["TotalBsmtSF"]
        if (
            "FullBath" in df.columns
            and "HalfBath" in df.columns
            and "BsmtFullBath" in df.columns
            and "BsmtHalfBath" in df.columns
        ):
            df["TotalBathrooms"] = (
                df["FullBath"]
                + 0.5 * df["HalfBath"]
                + df["BsmtFullBath"]
                + 0.5 * df["BsmtHalfBath"]
            )
        if "GrLivArea" in df.columns and "LotArea" in df.columns:
            df["LivingAreaRatio"] = df["GrLivArea"] / df["LotArea"]
        if "GrLivArea" in df.columns and "LotArea" in df.columns:
            df["LivingAreaRatio"] = df["GrLivArea"] / df["LotArea"]
        if "GrLivArea" in df.columns and "LotArea" in df.columns:
            df["LivingAreaRatio"] = df["GrLivArea"] / (df["LotArea"] + 1)

    else:
        pytest.fail(
            f"Aucun fichier de données trouvé à {enriched_file} ou {original_file}"
        )
    return df


@pytest.fixture(scope="module")
def expected_features():
    """
    Fixture pour retourner la liste des features attendues.
    """
    # Inclure toutes les colonnes qui sont réellement dans le dataset
    all_features = MAIN_NUMERIC_FEATURES + MAIN_CATEGORICAL_FEATURES + DERIVED_FEATURES
    # Ajouter les colonnes supplémentaires présentes dans le dataset
    additional_columns = ["Id"]  # Id est toujours présent
    return all_features + additional_columns


@pytest.mark.data_quality
def test_dataframe_shape(sample_data: pd.DataFrame):
    """
    Teste si le DataFrame a un nombre de lignes et de colonnes raisonnable.
    """
    logger.info(f"Test de la forme du DataFrame: {sample_data.shape}")
    assert (
        sample_data.shape[0] > 1000
    ), "Le DataFrame devrait avoir plus de 1000 lignes."
    assert sample_data.shape[1] > 50, "Le DataFrame devrait avoir plus de 50 colonnes."


@pytest.mark.data_quality
def test_target_variable_exists(sample_data: pd.DataFrame):
    """
    Teste si la variable cible (SalePrice) existe dans le DataFrame.
    """
    logger.info(f"Test de l'existence de la variable cible: {TARGET_VARIABLE}")
    assert (
        TARGET_VARIABLE in sample_data.columns
    ), f"La variable cible '{TARGET_VARIABLE}' est manquante."


@pytest.mark.data_quality
def test_target_variable_distribution(sample_data: pd.DataFrame):
    """
    Teste la distribution de la variable cible (SalePrice).
    - Pas de valeurs négatives ou nulles.
    - Valeurs dans une plage raisonnable.
    """
    logger.info(f"Test de la distribution de la variable cible: {TARGET_VARIABLE}")
    assert (
        sample_data[TARGET_VARIABLE] > 0
    ).all(), "La variable cible ne doit pas contenir de valeurs nulles ou négatives."
    assert (
        sample_data[TARGET_VARIABLE].min() >= 10000
    ), "Le prix minimum semble trop bas."
    assert (
        sample_data[TARGET_VARIABLE].max() <= 1000000
    ), "Le prix maximum semble trop élevé."


@pytest.mark.data_quality
def test_no_duplicate_rows(sample_data: pd.DataFrame):
    """
    Teste l'absence de lignes dupliquées dans le DataFrame.
    """
    logger.info("Test de l'absence de lignes dupliquées.")
    assert (
        not sample_data.duplicated().any()
    ), "Des lignes dupliquées ont été trouvées dans le DataFrame."


@pytest.mark.data_quality
def test_feature_existence(sample_data: pd.DataFrame, expected_features: List[str]):
    """
    Teste si les features attendues sont présentes dans le DataFrame.
    """
    logger.info("Test de l'existence des features attendues.")
    missing_features = [f for f in expected_features if f not in sample_data.columns]
    assert (
        not missing_features
    ), f"Les features suivantes sont manquantes: {missing_features}"


@pytest.mark.data_quality
def test_missing_values_percentage(sample_data: pd.DataFrame):
    """
    Teste le pourcentage de valeurs manquantes par colonne.
    Certaines colonnes peuvent avoir un pourcentage élevé de valeurs manquantes par conception (ex: Alley, PoolQC).
    Ce test vérifie que le pourcentage total de valeurs manquantes n'est pas excessif.
    """
    logger.info("Test du pourcentage de valeurs manquantes.")
    total_missing_cells = sample_data.isnull().sum().sum()
    total_cells = np.prod(sample_data.shape)
    overall_missing_percentage = total_missing_cells / total_cells

    # Un seuil plus élevé pour l'ensemble du dataset est acceptable si les colonnes sont gérées
    assert (
        overall_missing_percentage < DATA_QUALITY_THRESHOLDS["max_missing_percentage"]
    ), f"Le pourcentage total de valeurs manquantes ({overall_missing_percentage:.2%}) est trop élevé."

    # Vérifier que les colonnes avec beaucoup de NaN sont bien celles attendues
    # Par exemple, Alley, PoolQC, Fence, MiscFeature, FireplaceQu, Bsmt* peuvent avoir beaucoup de NaN
    # Nous acceptons un pourcentage élevé pour ces colonnes spécifiques
    columns_with_high_nan_expected = [
        "Alley",
        "PoolQC",
        "Fence",
        "MiscFeature",
        "FireplaceQu",
        "BsmtQual",
        "BsmtCond",
        "BsmtExposure",
        "BsmtFinType1",
        "BsmtFinType2",
        "GarageType",
        "GarageFinish",
        "GarageQual",
        "GarageCond",
        "MasVnrType",
    ]

    for col in sample_data.columns:
        missing_pct = sample_data[col].isnull().sum() / len(sample_data)
        if col not in columns_with_high_nan_expected:
            assert (
                missing_pct < 0.9
            ), f"La colonne '{col}' a un pourcentage de valeurs manquantes trop élevé ({missing_pct:.2%}) et n'est pas dans la liste des colonnes attendues avec beaucoup de NaN."


@pytest.mark.data_quality
def test_numeric_features_types(sample_data: pd.DataFrame):
    """
    Teste si les colonnes numériques ont le bon type de données.
    """
    logger.info("Test des types de données des features numériques.")
    for feature in MAIN_NUMERIC_FEATURES:
        if feature in sample_data.columns:
            assert pd.api.types.is_numeric_dtype(
                sample_data[feature]
            ), f"La feature '{feature}' n'est pas numérique."


@pytest.mark.data_quality
def test_categorical_features_types(sample_data: pd.DataFrame):
    """
    Teste si les colonnes catégorielles ont le bon type de données (object ou category).
    """
    logger.info("Test des types de données des features catégorielles.")

    # Features catégorielles qui peuvent être encodées comme entiers
    numeric_categorical_features = ["MSSubClass"]

    for feature in MAIN_CATEGORICAL_FEATURES:
        if feature in sample_data.columns:
            dtype = sample_data[feature].dtype

            # Permettre les types numériques pour certaines features catégorielles
            if feature in numeric_categorical_features:
                assert (
                    pd.api.types.is_string_dtype(dtype)
                    or isinstance(dtype, pd.CategoricalDtype)
                    or dtype == "object"
                    or pd.api.types.is_integer_dtype(dtype)
                ), f"La feature '{feature}' n'est pas catégorielle ou numérique. Type détecté: {dtype}"
            else:
                assert (
                    pd.api.types.is_string_dtype(dtype)
                    or isinstance(dtype, pd.CategoricalDtype)
                    or dtype == "object"
                ), f"La feature '{feature}' n'est pas catégorielle (object ou category). Type détecté: {dtype}"


@pytest.mark.data_quality
def test_no_infinite_values(sample_data: pd.DataFrame):
    """
    Teste l'absence de valeurs infinies dans le DataFrame.
    """
    logger.info("Test de l'absence de valeurs infinies.")
    for col in sample_data.select_dtypes(include=np.number).columns:
        assert not np.isinf(
            sample_data[col]
        ).any(), f"La colonne '{col}' contient des valeurs infinies."


@pytest.mark.data_quality
def test_derived_features_existence(sample_data: pd.DataFrame):
    """
    Teste si les features dérivées sont présentes dans le DataFrame.
    """
    logger.info("Test de l'existence des features dérivées.")
    for feature in DERIVED_FEATURES:
        assert (
            feature in sample_data.columns
        ), f"La feature dérivée '{feature}' est manquante."


@pytest.mark.data_quality
def test_derived_features_values(sample_data: pd.DataFrame):
    """
    Teste la cohérence des valeurs des features dérivées.
    """
    logger.info("Test de la cohérence des valeurs des features dérivées.")
    if "HouseAge" in sample_data.columns:
        assert (sample_data["HouseAge"] >= 0).all(), "HouseAge doit être positif."
    if "TotalSF" in sample_data.columns:
        assert (sample_data["TotalSF"] >= 0).all(), "TotalSF doit être positif."
    if "HasGarage" in sample_data.columns:
        assert (
            sample_data["HasGarage"].isin([0, 1]).all()
        ), "HasGarage doit être 0 ou 1."
    if "TotalBathrooms" in sample_data.columns:
        assert (
            sample_data["TotalBathrooms"] >= 0
        ).all(), "TotalBathrooms doit être positif."


@pytest.mark.data_quality
def test_cardinality_categorical_features(sample_data: pd.DataFrame):
    """
    Teste la cardinalité des features catégorielles.
    """
    logger.info("Test de la cardinalité des features catégorielles.")
    for feature in MAIN_CATEGORICAL_FEATURES:
        if feature in sample_data.columns:
            unique_count = sample_data[feature].nunique()
            assert (
                unique_count >= DATA_QUALITY_THRESHOLDS["min_unique_values"]
            ), f"La feature '{feature}' a trop peu de valeurs uniques."
            assert (
                unique_count <= DATA_QUALITY_THRESHOLDS["max_unique_values"]
            ), f"La feature '{feature}' a trop de valeurs uniques (cardinalité élevée)."

            # Pour les colonnes avec des NaN, le nombre de valeurs uniques peut être 0 si toutes sont NaN
            # On ne veut pas que ça échoue si une colonne est entièrement NaN et qu'elle est censée l'être
            if sample_data[feature].isnull().all():
                continue
            assert (
                unique_count > 0
            ), f"La feature '{feature}' ne devrait pas être vide si elle n'est pas entièrement NaN."


@pytest.mark.data_quality
def test_outliers_in_key_numeric_features(sample_data: pd.DataFrame):
    """
    Teste la présence d'outliers dans les features numériques clés.
    Ce test est indicatif et ne doit pas forcément échouer si des outliers sont présents,
    mais plutôt alerter sur leur proportion.
    """
    logger.info("Test de la présence d'outliers dans les features numériques clés.")
    key_numeric_features = [
        f
        for f in MAIN_NUMERIC_FEATURES
        if f in sample_data.columns and f != TARGET_VARIABLE
    ]

    for feature in key_numeric_features:
        if pd.api.types.is_numeric_dtype(sample_data[feature]):
            Q1 = sample_data[feature].quantile(0.25)
            Q3 = sample_data[feature].quantile(0.75)
            IQR = Q3 - Q1
            if (
                IQR == 0
            ):  # Éviter la division par zéro si toutes les valeurs sont identiques
                continue
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers_count = sample_data[
                (sample_data[feature] < lower_bound)
                | (sample_data[feature] > upper_bound)
            ].shape[0]
            total_count = sample_data[feature].dropna().shape[0]

            if total_count > 0:
                outlier_percentage = outliers_count / total_count
                if (
                    outlier_percentage
                    > DATA_QUALITY_THRESHOLDS["max_outlier_percentage"]
                ):
                    logger.warning(
                        f"La feature '{feature}' a un pourcentage d'outliers élevé ({outlier_percentage:.2%})."
                    )
                assert (
                    outlier_percentage <= 0.5
                ), f"La feature '{feature}' a un pourcentage d'outliers excessif ({outlier_percentage:.2%})."


@pytest.mark.data_quality
def test_consistency_year_built_sold(sample_data: pd.DataFrame):
    """
    Teste la cohérence entre l'année de construction et l'année de vente.
    """
    logger.info(
        "Test de la cohérence entre l'année de construction et l'année de vente."
    )
    if "YearBuilt" in sample_data.columns and "YrSold" in sample_data.columns:
        # Les maisons ne peuvent pas être vendues avant d'être construites
        assert (
            sample_data["YrSold"] >= sample_data["YearBuilt"]
        ).all(), "Certaines maisons sont vendues avant d'être construites."


@pytest.mark.data_quality
def test_consistency_garage_area_cars(sample_data: pd.DataFrame):
    """
    Teste la cohérence entre la surface du garage et le nombre de voitures.
    Si GarageCars > 0, alors GarageArea doit être > 0.
    """
    logger.info(
        "Test de la cohérence entre la surface du garage et le nombre de voitures."
    )
    if "GarageCars" in sample_data.columns and "GarageArea" in sample_data.columns:
        # Filtrer les lignes où GarageCars est > 0 et GarageArea est 0
        inconsistent_rows = sample_data[
            (sample_data["GarageCars"] > 0) & (sample_data["GarageArea"] == 0)
        ]
        assert (
            inconsistent_rows.empty
        ), "Incohérence: GarageCars > 0 mais GarageArea est 0."


@pytest.mark.data_quality
def test_consistency_basement_features(sample_data: pd.DataFrame):
    """
    Teste la cohérence des features du sous-sol.
    Si TotalBsmtSF > 0, alors BsmtQual ne doit pas être NaN.
    """
    logger.info("Test de la cohérence des features du sous-sol.")
    if "TotalBsmtSF" in sample_data.columns and "BsmtQual" in sample_data.columns:
        # Filtrer les lignes où TotalBsmtSF > 0 mais BsmtQual est NaN
        inconsistent_rows = sample_data[
            (sample_data["TotalBsmtSF"] > 0) & (sample_data["BsmtQual"].isnull())
        ]
        assert (
            inconsistent_rows.empty
        ), "Incohérence: TotalBsmtSF > 0 mais BsmtQual est NaN."


@pytest.mark.data_quality
def test_consistency_pool_features(sample_data: pd.DataFrame):
    """
    Teste la cohérence des features de piscine.
    Si PoolArea > 0, alors PoolQC ne doit pas être NaN.
    """
    logger.info("Test de la cohérence des features de piscine.")
    if "PoolArea" in sample_data.columns and "PoolQC" in sample_data.columns:
        # Filtrer les lignes où PoolArea > 0 mais PoolQC est NaN
        inconsistent_rows = sample_data[
            (sample_data["PoolArea"] > 0) & (sample_data["PoolQC"].isnull())
        ]
        assert inconsistent_rows.empty, "Incohérence: PoolArea > 0 mais PoolQC est NaN."


@pytest.mark.data_quality
def test_consistency_fireplace_features(sample_data: pd.DataFrame):
    """
    Teste la cohérence des features de cheminée.
    Si Fireplaces > 0, alors FireplaceQu ne doit pas être NaN.
    """
    logger.info("Test de la cohérence des features de cheminée.")
    if "Fireplaces" in sample_data.columns and "FireplaceQu" in sample_data.columns:
        # Filtrer les lignes où Fireplaces > 0 mais FireplaceQu est NaN
        inconsistent_rows = sample_data[
            (sample_data["Fireplaces"] > 0) & (sample_data["FireplaceQu"].isnull())
        ]
        assert (
            inconsistent_rows.empty
        ), "Incohérence: Fireplaces > 0 mais FireplaceQu est NaN."


@pytest.mark.data_quality
def test_consistency_mas_vnr_features(sample_data: pd.DataFrame):
    """
    Teste la cohérence des features de placage de maçonnerie.
    Si MasVnrArea > 0, alors MasVnrType ne doit pas être NaN.
    """
    logger.info("Test de la cohérence des features de placage de maçonnerie.")
    if "MasVnrArea" in sample_data.columns and "MasVnrType" in sample_data.columns:
        # Filtrer les lignes où MasVnrArea > 0 mais MasVnrType est NaN
        inconsistent_rows = sample_data[
            (sample_data["MasVnrArea"] > 0) & (sample_data["MasVnrType"].isnull())
        ]

        # Permettre un petit nombre d'incohérences dans les données réelles
        max_allowed_inconsistencies = 10
        assert (
            len(inconsistent_rows) <= max_allowed_inconsistencies
        ), f"Trop d'incohérences: {len(inconsistent_rows)} lignes avec MasVnrArea > 0 mais MasVnrType NaN (max autorisé: {max_allowed_inconsistencies})."


@pytest.mark.data_quality
def test_consistency_lot_frontage_area(sample_data: pd.DataFrame):
    """
    Teste la cohérence entre LotFrontage et LotArea.
    Si LotArea > 0, LotFrontage ne devrait pas être 0 si c'est une feature importante.
    """
    logger.info("Test de la cohérence entre LotFrontage et LotArea.")
    if "LotFrontage" in sample_data.columns and "LotArea" in sample_data.columns:
        # Il est courant que LotFrontage soit NaN ou 0 pour certains types de lots
        # Ce test est plus souple, il vérifie juste que si LotArea est très grand, LotFrontage n'est pas 0
        inconsistent_rows = sample_data[
            (sample_data["LotArea"] > 10000) & (sample_data["LotFrontage"] == 0)
        ]
        if not inconsistent_rows.empty:
            logger.warning(
                "LotArea est grand mais LotFrontage est 0 pour certaines entrées. Cela peut être normal mais mérite vérification."
            )


@pytest.mark.data_quality
def test_consistency_total_bsmt_sf(sample_data: pd.DataFrame):
    """
    Teste la cohérence de TotalBsmtSF avec les autres features du sous-sol.
    Si TotalBsmtSF est 0, alors toutes les autres features de surface du sous-sol (BsmtFinSF1, BsmtFinSF2, BsmtUnfSF)
    devraient également être 0.
    """
    logger.info("Test de la cohérence de TotalBsmtSF.")
    basement_sf_cols = ["BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF"]
    if "TotalBsmtSF" in sample_data.columns and all(
        col in sample_data.columns for col in basement_sf_cols
    ):
        zero_total_bsmt = sample_data[sample_data["TotalBsmtSF"] == 0]
        for col in basement_sf_cols:
            assert (
                zero_total_bsmt[col] == 0
            ).all(), f"Incohérence: TotalBsmtSF est 0 mais '{col}' n'est pas 0."


@pytest.mark.data_quality
def test_consistency_total_rooms_abv_grd(sample_data: pd.DataFrame):
    """
    Teste la cohérence de TotRmsAbvGrd avec BedroomAbvGr et KitchenAbvGr.
    TotRmsAbvGrd devrait être au moins la somme de BedroomAbvGr et KitchenAbvGr.
    """
    logger.info("Test de la cohérence de TotRmsAbvGrd.")
    if (
        "TotRmsAbvGrd" in sample_data.columns
        and "BedroomAbvGr" in sample_data.columns
        and "KitchenAbvGr" in sample_data.columns
    ):
        assert (
            sample_data["TotRmsAbvGrd"]
            >= (sample_data["BedroomAbvGr"] + sample_data["KitchenAbvGr"])
        ).all(), "Incohérence: TotRmsAbvGrd est inférieur à la somme de BedroomAbvGr et KitchenAbvGr."


@pytest.mark.data_quality
def test_consistency_first_second_floor_sf(sample_data: pd.DataFrame):
    """
    Teste la cohérence entre 1stFlrSF, 2ndFlrSF et GrLivArea.
    GrLivArea devrait être la somme de 1stFlrSF et 2ndFlrSF (plus LowQualFinSF).
    """
    logger.info("Test de la cohérence entre les surfaces des étages.")
    if (
        "GrLivArea" in sample_data.columns
        and "1stFlrSF" in sample_data.columns
        and "2ndFlrSF" in sample_data.columns
        and "LowQualFinSF" in sample_data.columns
    ):
        calculated_grlivarea = (
            sample_data["1stFlrSF"]
            + sample_data["2ndFlrSF"]
            + sample_data["LowQualFinSF"]
        )
        # Permettre une petite tolérance pour les erreurs de flottant
        assert np.allclose(
            sample_data["GrLivArea"], calculated_grlivarea, atol=1.0
        ), "Incohérence: GrLivArea ne correspond pas à la somme des surfaces des étages."


@pytest.mark.data_quality
def test_consistency_total_bathrooms(sample_data: pd.DataFrame):
    """
    Teste la cohérence de TotalBathrooms avec les features de salle de bain.
    """
    logger.info("Test de la cohérence de TotalBathrooms.")
    if (
        "TotalBathrooms" in sample_data.columns
        and "FullBath" in sample_data.columns
        and "HalfBath" in sample_data.columns
        and "BsmtFullBath" in sample_data.columns
        and "BsmtHalfBath" in sample_data.columns
    ):
        calculated_total_bathrooms = (
            sample_data["FullBath"]
            + 0.5 * sample_data["HalfBath"]
            + sample_data["BsmtFullBath"]
            + 0.5 * sample_data["BsmtHalfBath"]
        )
        assert np.allclose(
            sample_data["TotalBathrooms"], calculated_total_bathrooms, atol=0.01
        ), "Incohérence: TotalBathrooms ne correspond pas à la somme des salles de bain."


@pytest.mark.data_quality
def test_consistency_quality_score(sample_data: pd.DataFrame):
    """
    Teste la cohérence de QualityScore avec OverallQual et OverallCond.
    """
    logger.info("Test de la cohérence de QualityScore.")
    if (
        "QualityScore" in sample_data.columns
        and "OverallQual" in sample_data.columns
        and "OverallCond" in sample_data.columns
    ):
        calculated_quality_score = (
            sample_data["OverallQual"] * sample_data["OverallCond"]
        )
        assert (
            sample_data["QualityScore"] == calculated_quality_score
        ).all(), "Incohérence: QualityScore ne correspond pas au produit de OverallQual et OverallCond."


@pytest.mark.data_quality
def test_consistency_living_area_ratio(sample_data: pd.DataFrame):
    """
    Teste la cohérence de LivingAreaRatio.
    """
    logger.info("Test de la cohérence de LivingAreaRatio.")
    if (
        "LivingAreaRatio" in sample_data.columns
        and "GrLivArea" in sample_data.columns
        and "LotArea" in sample_data.columns
    ):
        calculated_ratio = sample_data["GrLivArea"] / (sample_data["LotArea"] + 1)
        assert np.allclose(
            sample_data["LivingAreaRatio"], calculated_ratio, atol=1e-6
        ), "Incohérence: LivingAreaRatio est incorrect."


@pytest.mark.data_quality
def test_consistency_has_features_binary(sample_data: pd.DataFrame):
    """
    Teste la cohérence des features binaires (HasGarage, HasBasement, HasFireplace).
    """
    logger.info("Test de la cohérence des features binaires.")
    if "HasGarage" in sample_data.columns and "GarageArea" in sample_data.columns:
        assert (
            sample_data["HasGarage"] == (sample_data["GarageArea"] > 0).astype(int)
        ).all(), "Incohérence: HasGarage ne correspond pas à GarageArea."
    if "HasBasement" in sample_data.columns and "TotalBsmtSF" in sample_data.columns:
        assert (
            sample_data["HasBasement"] == (sample_data["TotalBsmtSF"] > 0).astype(int)
        ).all(), "Incohérence: HasBasement ne correspond pas à TotalBsmtSF."
    if "HasFireplace" in sample_data.columns and "Fireplaces" in sample_data.columns:
        assert (
            sample_data["HasFireplace"] == (sample_data["Fireplaces"] > 0).astype(int)
        ).all(), "Incohérence: HasFireplace ne correspond pas à Fireplaces."


@pytest.mark.data_quality
def test_all_columns_in_expected_features(
    sample_data: pd.DataFrame, expected_features: List[str]
):
    """
    Teste si toutes les colonnes du DataFrame sont soit la variable cible, soit une feature attendue.
    """
    logger.info(
        "Test de la présence de toutes les colonnes dans les features attendues."
    )
    all_known_columns = set(expected_features + [TARGET_VARIABLE])
    extra_columns = [col for col in sample_data.columns if col not in all_known_columns]
    assert (
        not extra_columns
    ), f"Des colonnes inattendues ont été trouvées dans le DataFrame: {extra_columns}"


@pytest.mark.data_quality
def test_no_empty_columns(sample_data: pd.DataFrame):
    """
    Teste l'absence de colonnes entièrement vides (toutes NaN).
    """
    logger.info("Test de l'absence de colonnes entièrement vides.")
    empty_columns = sample_data.columns[sample_data.isnull().all()].tolist()

    # Certaines colonnes peuvent être entièrement NaN par conception (ex: Alley si aucune ruelle)
    # Nous allons lister ces colonnes et les ignorer dans ce test
    expected_empty_columns = ["Alley", "PoolQC", "Fence", "MiscFeature"]

    unexpected_empty_columns = [
        col for col in empty_columns if col not in expected_empty_columns
    ]

    assert (
        not unexpected_empty_columns
    ), f"Des colonnes entièrement vides inattendues ont été trouvées: {unexpected_empty_columns}"


@pytest.mark.data_quality
def test_unique_ids(sample_data: pd.DataFrame):
    """
    Teste si la colonne 'Id' (si présente) contient des valeurs uniques.
    """
    logger.info("Test de l'unicité des IDs.")
    if "Id" in sample_data.columns:
        assert sample_data[
            "Id"
        ].is_unique, "La colonne 'Id' contient des valeurs dupliquées."


@pytest.mark.data_quality
def test_categorical_feature_values_valid(sample_data: pd.DataFrame):
    """
    Teste si les valeurs des features catégorielles sont parmi un ensemble de valeurs valides (si connu).
    Ce test est plus complexe et peut nécessiter une liste de valeurs valides pour chaque feature.
    Pour l'instant, nous allons vérifier qu'il n'y a pas de valeurs aberrantes évidentes.
    """
    logger.info("Test de la validité des valeurs des features catégorielles.")
    # Exemple pour une feature spécifique, à étendre si nécessaire
    if "MSZoning" in sample_data.columns:
        valid_ms_zoning = [
            "A",
            "C (all)",
            "'C (all)'",
            "FV",
            "I",
            "RH",
            "RL",
            "RP",
            "RM",
        ]  # Inclure la version avec guillemets
        assert (
            sample_data["MSZoning"].dropna().isin(valid_ms_zoning).all()
        ), f"Valeurs inattendues dans MSZoning: {sample_data['MSZoning'].dropna()[~sample_data['MSZoning'].dropna().isin(valid_ms_zoning)].unique().tolist()}"

    if "Utilities" in sample_data.columns:
        valid_utilities = ["AllPub", "NoSeWa", "NoSewr", "ELO"]
        assert (
            sample_data["Utilities"].dropna().isin(valid_utilities).all()
        ), f"Valeurs inattendues dans Utilities: {sample_data['Utilities'].dropna()[~sample_data['Utilities'].dropna().isin(valid_utilities)].unique().tolist()}"


@pytest.mark.data_quality
def test_numeric_feature_ranges(sample_data: pd.DataFrame):
    """
    Teste si les valeurs des features numériques sont dans des plages raisonnables.
    """
    logger.info("Test des plages de valeurs des features numériques.")
    # Exemple pour quelques features clés
    if "GrLivArea" in sample_data.columns:
        assert (sample_data["GrLivArea"] >= 0).all() and (
            sample_data["GrLivArea"] <= 6000
        ).all(), "GrLivArea hors de la plage attendue (0-6000)."
    if "LotArea" in sample_data.columns:
        assert (sample_data["LotArea"] >= 0).all() and (
            sample_data["LotArea"]
            <= 250000  # Ajusté pour inclure la valeur max de 215245
        ).all(), "LotArea hors de la plage attendue (0-250000)."
    if "OverallQual" in sample_data.columns:
        assert (sample_data["OverallQual"] >= 1).all() and (
            sample_data["OverallQual"] <= 10
        ).all(), "OverallQual hors de la plage attendue (1-10)."
    if "YearBuilt" in sample_data.columns:
        assert (sample_data["YearBuilt"] >= 1800).all() and (
            sample_data["YearBuilt"] <= 2025
        ).all(), "YearBuilt hors de la plage attendue (1800-2025)."


# Exécution des tests si le script est lancé directement
if __name__ == "__main__":
    pytest.main([__file__])
