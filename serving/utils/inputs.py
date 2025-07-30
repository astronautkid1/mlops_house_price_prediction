from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Optional, List


class HouseFeatures(BaseModel):
    """
    Modèle Pydantic pour valider les caractéristiques d'une maison.

    Ce modèle définit les champs requis et optionnels pour effectuer une prédiction
    du prix d'une maison. Il inclut des validations pour s'assurer que les données
    sont dans les plages attendues.
    """

    # Variables numériques principales
    GrLivArea: Optional[float] = Field(
        None,
        description="Surface habitable au-dessus du sol (pieds carrés)",
        ge=0,
        le=10000,
    )
    OverallQual: Optional[int] = Field(
        None,
        description="Qualité générale du matériau et de la finition (1-10)",
        ge=1,
        le=10,
    )
    OverallCond: Optional[int] = Field(
        None, description="Condition générale de la maison (1-10)", ge=1, le=10
    )
    YearBuilt: Optional[int] = Field(
        None, description="Année de construction originale", ge=1800, le=2030
    )
    YearRemodAdd: Optional[int] = Field(
        None, description="Année de rénovation/ajout", ge=1800, le=2030
    )
    TotalBsmtSF: Optional[float] = Field(
        None, description="Surface totale du sous-sol (pieds carrés)", ge=0, le=10000
    )
    FirstFlrSF: Optional[float] = Field(
        None,
        description="Surface du premier étage (pieds carrés)",
        ge=0,
        le=10000,
        alias="1stFlrSF",  # Mappage vers le nom réel de la colonne
    )
    SecondFlrSF: Optional[float] = Field(
        None,
        description="Surface du deuxième étage (pieds carrés)",
        ge=0,
        le=10000,
        alias="2ndFlrSF",  # Mappage vers le nom réel de la colonne
    )
    LowQualFinSF: Optional[float] = Field(
        None,
        description="Surface finie de faible qualité (pieds carrés)",
        ge=0,
        le=5000,
    )
    BsmtFinSF1: Optional[float] = Field(
        None,
        description="Surface finie du sous-sol type 1 (pieds carrés)",
        ge=0,
        le=5000,
    )
    BsmtFinSF2: Optional[float] = Field(
        None,
        description="Surface finie du sous-sol type 2 (pieds carrés)",
        ge=0,
        le=5000,
    )
    BsmtUnfSF: Optional[float] = Field(
        None, description="Surface non finie du sous-sol (pieds carrés)", ge=0, le=5000
    )

    # Variables de surface et dimensions
    LotFrontage: Optional[float] = Field(
        None,
        description="Pieds linéaires de rue connectés à la propriété",
        ge=0,
        le=500,
    )
    LotArea: Optional[float] = Field(
        None, description="Taille du lot (pieds carrés)", ge=0, le=200000
    )
    MasVnrArea: Optional[float] = Field(
        None,
        description="Surface de placage de maçonnerie (pieds carrés)",
        ge=0,
        le=2000,
    )

    # Variables de garage
    GarageArea: Optional[float] = Field(
        None, description="Taille du garage (pieds carrés)", ge=0, le=2000
    )
    GarageCars: Optional[float] = Field(
        None, description="Taille du garage en capacité de voitures", ge=0, le=5
    )
    GarageYrBlt: Optional[float] = Field(
        None, description="Année de construction du garage", ge=1800, le=2030
    )

    # Variables de salles et espaces
    BedroomAbvGr: Optional[int] = Field(
        None, description="Nombre de chambres au-dessus du niveau du sol", ge=0, le=10
    )
    KitchenAbvGr: Optional[int] = Field(
        None, description="Nombre de cuisines au-dessus du niveau du sol", ge=0, le=5
    )
    TotRmsAbvGrd: Optional[int] = Field(
        None,
        description="Nombre total de pièces au-dessus du niveau du sol",
        ge=0,
        le=20,
    )
    Fireplaces: Optional[int] = Field(
        None, description="Nombre de cheminées", ge=0, le=5
    )

    # Variables de salles de bain
    FullBath: Optional[int] = Field(
        None,
        description="Salles de bain complètes au-dessus du niveau du sol",
        ge=0,
        le=5,
    )
    HalfBath: Optional[int] = Field(
        None, description="Demi-salles de bain au-dessus du niveau du sol", ge=0, le=5
    )
    BsmtFullBath: Optional[int] = Field(
        None, description="Salles de bain complètes au sous-sol", ge=0, le=5
    )
    BsmtHalfBath: Optional[int] = Field(
        None, description="Demi-salles de bain au sous-sol", ge=0, le=5
    )

    # Variables de vente
    YrSold: Optional[int] = Field(None, description="Année de vente", ge=2000, le=2030)
    MoSold: Optional[int] = Field(None, description="Mois de vente", ge=1, le=12)

    # Variables diverses
    PoolArea: Optional[float] = Field(
        None, description="Surface de la piscine (pieds carrés)", ge=0, le=2000
    )
    ScreenPorch: Optional[float] = Field(
        None, description="Surface du porche grillagé (pieds carrés)", ge=0, le=1000
    )
    ThreeSsnPorch: Optional[float] = Field(
        None,
        description="Surface du porche trois saisons (pieds carrés)",
        ge=0,
        le=1000,
        alias="3SsnPorch",  # Mappage vers le nom réel de la colonne
    )
    EnclosedPorch: Optional[float] = Field(
        None, description="Surface du porche fermé (pieds carrés)", ge=0, le=1000
    )
    OpenPorchSF: Optional[float] = Field(
        None, description="Surface du porche ouvert (pieds carrés)", ge=0, le=1000
    )
    WoodDeckSF: Optional[float] = Field(
        None, description="Surface de la terrasse en bois (pieds carrés)", ge=0, le=2000
    )

    # Variables catégorielles principales
    MSSubClass: Optional[str] = Field(
        None, description="Type de logement impliqué dans la vente"
    )
    MSZoning: Optional[str] = Field(
        None, description="Classification générale du zonage"
    )
    Street: Optional[str] = Field(
        None, description="Type d'accès routier à la propriété"
    )
    Alley: Optional[str] = Field(
        None, description="Type d'accès par ruelle à la propriété"
    )
    LotShape: Optional[str] = Field(None, description="Forme générale de la propriété")
    LandContour: Optional[str] = Field(None, description="Planéité de la propriété")
    Utilities: Optional[str] = Field(None, description="Type d'utilitaires disponibles")
    LotConfig: Optional[str] = Field(None, description="Configuration du lot")
    LandSlope: Optional[str] = Field(None, description="Pente de la propriété")
    Neighborhood: Optional[str] = Field(
        None, description="Emplacements physiques dans les limites de la ville d'Ames"
    )
    Condition1: Optional[str] = Field(
        None, description="Proximité de diverses conditions"
    )
    Condition2: Optional[str] = Field(
        None,
        description="Proximité de diverses conditions (si plus d'une est présente)",
    )
    BldgType: Optional[str] = Field(None, description="Type de logement")
    HouseStyle: Optional[str] = Field(None, description="Style de logement")
    RoofStyle: Optional[str] = Field(None, description="Type de toit")
    RoofMatl: Optional[str] = Field(None, description="Matériau du toit")
    Exterior1st: Optional[str] = Field(
        None, description="Revêtement extérieur de la maison"
    )
    Exterior2nd: Optional[str] = Field(
        None, description="Revêtement extérieur de la maison (si plus d'un matériau)"
    )
    MasVnrType: Optional[str] = Field(None, description="Type de placage de maçonnerie")
    ExterQual: Optional[str] = Field(None, description="Qualité du matériau extérieur")
    ExterCond: Optional[str] = Field(
        None, description="Condition actuelle du matériau extérieur"
    )
    Foundation: Optional[str] = Field(None, description="Type de fondation")

    # Variables de sous-sol
    BsmtQual: Optional[str] = Field(None, description="Hauteur du sous-sol")
    BsmtCond: Optional[str] = Field(None, description="Condition générale du sous-sol")
    BsmtExposure: Optional[str] = Field(
        None, description="Murs du sous-sol de niveau jardin ou sortie"
    )
    BsmtFinType1: Optional[str] = Field(
        None, description="Qualité de la zone finie du sous-sol"
    )
    BsmtFinType2: Optional[str] = Field(
        None, description="Qualité de la deuxième zone finie (si présente)"
    )

    # Variables de chauffage et climatisation
    Heating: Optional[str] = Field(None, description="Type de chauffage")
    HeatingQC: Optional[str] = Field(
        None, description="Qualité et condition du chauffage"
    )
    CentralAir: Optional[str] = Field(None, description="Climatisation centrale")
    Electrical: Optional[str] = Field(None, description="Système électrique")

    # Variables de cuisine et intérieur
    KitchenQual: Optional[str] = Field(None, description="Qualité de la cuisine")
    Functional: Optional[str] = Field(
        None, description="Évaluation de la fonctionnalité de la maison"
    )
    FireplaceQu: Optional[str] = Field(None, description="Qualité de la cheminée")

    # Variables de garage
    GarageType: Optional[str] = Field(None, description="Emplacement du garage")
    GarageFinish: Optional[str] = Field(
        None, description="Finition intérieure du garage"
    )
    GarageQual: Optional[str] = Field(None, description="Qualité du garage")
    GarageCond: Optional[str] = Field(None, description="Condition du garage")

    # Variables de piscine et clôture
    PoolQC: Optional[str] = Field(None, description="Qualité de la piscine")
    Fence: Optional[str] = Field(None, description="Qualité de la clôture")
    MiscFeature: Optional[str] = Field(
        None,
        description="Caractéristique diverse non couverte dans d'autres catégories",
    )

    # Variables de vente
    SaleType: Optional[str] = Field(None, description="Type de vente")
    SaleCondition: Optional[str] = Field(None, description="Condition de vente")

    # Variables dérivées (features engineering)
    HouseAge: Optional[float] = Field(
        None, description="Âge de la maison (calculé)", ge=0, le=200
    )
    YearsSinceRemod: Optional[float] = Field(
        None, description="Années depuis la rénovation (calculé)", ge=0, le=200
    )
    TotalSF: Optional[float] = Field(
        None, description="Surface totale (calculé)", ge=0, le=20000
    )
    QualityScore: Optional[float] = Field(
        None, description="Score de qualité combiné (calculé)", ge=0, le=100
    )
    TotalBathrooms: Optional[float] = Field(
        None, description="Nombre total de salles de bain (calculé)", ge=0, le=10
    )
    HasGarage: Optional[int] = Field(
        None, description="Présence d'un garage (0/1)", ge=0, le=1
    )
    HasBasement: Optional[int] = Field(
        None, description="Présence d'un sous-sol (0/1)", ge=0, le=1
    )
    HasFireplace: Optional[int] = Field(
        None, description="Présence d'une cheminée (0/1)", ge=0, le=1
    )
    LivingAreaRatio: Optional[float] = Field(
        None, description="Ratio surface habitable/terrain", ge=0, le=1
    )

    @field_validator("YearRemodAdd")
    @classmethod
    def validate_remod_year(cls, v, info):
        """Valide que l'année de rénovation n'est pas antérieure à l'année de construction."""
        if (
            v is not None
            and "YearBuilt" in info.data
            and info.data["YearBuilt"] is not None
        ):
            if v < info.data["YearBuilt"]:
                raise ValueError(
                    "L'année de rénovation ne peut pas être antérieure à l'année de construction"
                )
        return v

    @field_validator("GarageYrBlt")
    @classmethod
    def validate_garage_year(cls, v, info):
        """Valide que l'année de construction du garage est cohérente."""
        if (
            v is not None
            and "YearBuilt" in info.data
            and info.data["YearBuilt"] is not None
        ):
            if v < info.data["YearBuilt"] - 5:  # Tolérance de 5 ans
                raise ValueError("L'année de construction du garage semble incohérente")
        return v

    @field_validator("TotRmsAbvGrd")
    @classmethod
    def validate_total_rooms(cls, v, info):
        """Valide que le nombre total de pièces est cohérent avec les autres compteurs."""
        if v is not None:
            bedrooms = info.data.get("BedroomAbvGr", 0) or 0
            kitchens = info.data.get("KitchenAbvGr", 0) or 0
            bathrooms = info.data.get("FullBath", 0) or 0

            min_rooms = bedrooms + kitchens + bathrooms
            if v < min_rooms:
                raise ValueError(
                    f"Le nombre total de pièces ({v}) semble trop faible par rapport aux autres compteurs"
                )
        return v

    @field_validator("GarageCars")
    @classmethod
    def validate_garage_cars_area(cls, v, info):
        """Valide la cohérence entre la capacité du garage et sa surface."""
        if v is not None and v > 0:
            garage_area = info.data.get("GarageArea", 0) or 0
            if garage_area == 0:
                raise ValueError(
                    "Un garage avec une capacité de voitures doit avoir une surface > 0"
                )
        return v

    model_config = ConfigDict(
        # Permettre l'utilisation de champs non définis (pour la flexibilité)
        extra="ignore",
        # Valider les assignations
        validate_assignment=True,
        # Utiliser les types enum pour une meilleure validation
        use_enum_values=True,
        # Schema extra pour la documentation
        json_schema_extra={
            "example": {
                "GrLivArea": 1710.0,
                "OverallQual": 8,
                "OverallCond": 5,
                "YearBuilt": 2003,
                "YearRemodAdd": 2003,
                "TotalBsmtSF": 856.0,
                "FirstFlrSF": 856.0,
                "SecondFlrSF": 854.0,
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
        },
    )


class HouseFeaturesBatch(BaseModel):
    """
    Modèle Pydantic pour valider un lot de caractéristiques de maisons.

    Utilisé pour les prédictions en lot via l'endpoint /api/predict/batch.
    """

    houses: List[HouseFeatures] = Field(
        ..., description="Liste des caractéristiques des maisons"
    )

    @field_validator("houses")
    @classmethod
    def validate_houses_list(cls, v):
        """Valide que la liste n'est pas vide et ne dépasse pas une certaine limite."""
        if not v:
            raise ValueError("La liste des maisons ne peut pas être vide")
        if len(v) > 1000:  # Limite pour éviter les surcharges
            raise ValueError(
                "Le nombre de maisons ne peut pas dépasser 1000 par requête"
            )
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "houses": [
                    {
                        "GrLivArea": 1710.0,
                        "OverallQual": 8,
                        "YearBuilt": 2003,
                        "TotalBsmtSF": 856.0,
                        "GarageArea": 548.0,
                        "Neighborhood": "CollgCr",
                    },
                    {
                        "GrLivArea": 1262.0,
                        "OverallQual": 6,
                        "YearBuilt": 1976,
                        "TotalBsmtSF": 1262.0,
                        "GarageArea": 460.0,
                        "Neighborhood": "Veenker",
                    },
                ]
            }
        }
    )


# Modèles de réponse pour la documentation API
class PredictionResponse(BaseModel):
    """Modèle de réponse pour une prédiction simple."""

    predicted_price: float = Field(..., description="Prix prédit en dollars")


class BatchPredictionResponse(BaseModel):
    """Modèle de réponse pour les prédictions en lot."""

    predicted_prices: List[float] = Field(
        ..., description="Liste des prix prédits en dollars"
    )
    errors: List[dict] = Field(
        ..., description="Liste des erreurs de validation par ligne"
    )


class HealthResponse(BaseModel):
    """Modèle de réponse pour le check de santé."""

    status: str = Field(..., description="Statut de l'API")
    timestamp: str = Field(..., description="Timestamp de la vérification")


class ModelInfoResponse(BaseModel):
    """Modèle de réponse pour les informations du modèle."""

    model_name: str = Field(..., description="Nom du modèle")
    model_type: str = Field(..., description="Type de modèle")
    training_date: str = Field(..., description="Date d'entraînement")
    performance: dict = Field(..., description="Métriques de performance")
