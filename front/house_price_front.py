import streamlit as st
import pandas as pd
import requests
from io import StringIO

# ------------------------- Configuration -------------------------
st.set_page_config(
    page_title="Prédicteur de Prix Immobilier",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS personnalisé pour une interface plus belle (correction du problème d'affichage blanc)
st.markdown(
    """
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    
    .prediction-result {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 2rem 0;
    }
    
    .error-box {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    
    .success-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    
    /* Correction pour les sélecteurs - supprimer background blanc */
    .stSelectbox > div > div > div {
        background-color: transparent !important;
    }
    
    .stNumberInput > div > div > input {
        background-color: transparent !important;
    }
    
    /* Améliorer la visibilité des options */
    .stSelectbox label,
    .stNumberInput label {
        color: inherit !important;
        font-weight: 600 !important;
    }
    
</style>
""",
    unsafe_allow_html=True,
)

# En-tête principal
st.markdown(
    """
<div class="main-header">
    <h1>🏠 Prédicteur de Prix Immobilier</h1>
    <p>Intelligence artificielle pour l'estimation de biens immobiliers</p>
</div>
""",
    unsafe_allow_html=True,
)

# Sidebar pour la sélection du mode
with st.sidebar:
    st.image(
        "https://images.unsplash.com/photo-1560518883-ce09059eeffa?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Mnx8aG91c2V8ZW58MHx8MHx8fDA%3D&w=1000&q=80",
        width=300,
    )
    st.markdown("### 📌 Mode de prédiction")
    mode = st.radio(
        "Choisissez votre méthode :",
        ["🏡 Prédiction individuelle", "📊 Prédiction en lot (CSV)"],
        help="Sélectionnez le mode de prédiction souhaité",
    )

# ------------------------- Mode Manuel -------------------------
if mode == "🏡 Prédiction individuelle":
    st.markdown("## 📝 Caractéristiques de la propriété")

    # Organisation en colonnes pour une meilleure présentation
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🏠 Informations générales")
        with st.container():
            gr_liv_area = st.number_input(
                "Surface habitable (sq ft)",
                min_value=0,
                value=1710,
                help="Surface habitable au-dessus du sol (GrLivArea)",
            )

            overall_qual = st.select_slider(
                "Qualité globale (OverallQual)",
                options=list(range(1, 11)),
                value=7,
                help="1=Très pauvre → 10=Excellent",
            )

            year_built = st.number_input(
                "Année de construction (YearBuilt)",
                min_value=1800,
                max_value=2025,
                value=2003,
                help="Année de construction originale",
            )

            year_remod_add = st.number_input(
                "Année de rénovation (YearRemodAdd)",
                min_value=1800,
                max_value=2025,
                value=2003,
                help="Année de rénovation (même année si pas de rénovation)",
            )

    with col2:
        st.markdown("### 🏗️ Surfaces et structures")
        with st.container():
            total_bsmt_sf = st.number_input(
                "Surface sous-sol (TotalBsmtSF)",
                min_value=0,
                value=856,
                help="Surface totale du sous-sol en sq ft",
            )

            first_flr_sf = st.number_input(
                "Surface RDC (1stFlrSF)",
                min_value=0,
                value=856,
                help="Surface du rez-de-chaussée en sq ft",
            )

            second_flr_sf = st.number_input(
                "Surface étage (2ndFlrSF)",
                min_value=0,
                value=854,
                help="Surface du deuxième étage en sq ft",
            )

            garage_area = st.number_input(
                "Surface garage (GarageArea)",
                min_value=0,
                value=548,
                help="Surface du garage en sq ft",
            )

            garage_cars = st.select_slider(
                "Places de garage (GarageCars)",
                options=list(range(0, 6)),
                value=2,
                help="Capacité du garage en nombre de voitures",
            )

    with col3:
        st.markdown("### 🏘️ Localisation et qualité")
        with st.container():
            # Neighborhoods corrects selon la documentation Ames Housing
            neighborhood = st.selectbox(
                "Quartier (Neighborhood)",
                options=[
                    "Blmngtn",
                    "Blueste",
                    "BrDale",
                    "BrkSide",
                    "ClearCr",
                    "CollgCr",
                    "Crawfor",
                    "Edwards",
                    "Gilbert",
                    "Greens",
                    "GrnHill",
                    "IDOTRR",
                    "Landmrk",
                    "MeadowV",
                    "Mitchel",
                    "Names",
                    "NoRidge",
                    "NPkVill",
                    "NridgHt",
                    "NWAmes",
                    "OldTown",
                    "SWISU",
                    "Sawyer",
                    "SawyerW",
                    "Somerst",
                    "StoneBr",
                    "Timber",
                    "Veenker",
                ],
                index=5,  # CollgCr par défaut
                help="Quartier physique dans la ville d'Ames",
            )

            # MSZoning corrects selon la documentation
            ms_zoning = st.selectbox(
                "Zone résidentielle (MSZoning)",
                options=["A", "C", "FV", "I", "RH", "RL", "RP", "RM"],
                index=5,  # RL (Residential Low Density) par défaut
                help="A=Agriculture, C=Commercial, FV=Floating Village, RH=Residential High Density, RL=Residential Low Density",
            )

            # HeatingQC corrects
            heating_qc = st.selectbox(
                "Qualité du chauffage (HeatingQC)",
                options=["Ex", "Gd", "TA", "Fa", "Po"],
                index=1,  # Gd par défaut
                help="Ex=Excellent, Gd=Bon, TA=Typique/Moyen, Fa=Correct, Po=Pauvre",
            )

            central_air = st.selectbox(
                "Climatisation centrale (CentralAir)",
                options=["Y", "N"],
                index=0,  # Y par défaut
                help="Y=Oui, N=Non",
            )

            # KitchenQual corrects
            kitchen_qual = st.selectbox(
                "Qualité de la cuisine (KitchenQual)",
                options=["Ex", "Gd", "TA", "Fa", "Po"],
                index=1,  # Gd par défaut
                help="Ex=Excellent, Gd=Bon, TA=Typique/Moyen, Fa=Correct, Po=Pauvre",
            )

    # Section des caractéristiques avancées (repliable)
    with st.expander("🔧 Caractéristiques avancées", expanded=False):
        col_adv1, col_adv2 = st.columns(2)

        with col_adv1:
            # Functional corrects
            functional = st.selectbox(
                "Fonctionnalité (Functional)",
                options=["Typ", "Min1", "Min2", "Mod", "Maj1", "Maj2", "Sev", "Sal"],
                index=0,  # Typ par défaut
                help="Typ=Typique, Min1/2=Déductions mineures, Mod=Modérée, Maj1/2=Majeures, Sev=Sévère, Sal=Récupération",
            )

            # FireplaceQu corrects
            fireplace_qu = st.selectbox(
                "Qualité de la cheminée (FireplaceQu)",
                options=["Ex", "Gd", "TA", "Fa", "Po", "NA"],
                index=5,  # NA par défaut (pas de cheminée)
                help="Ex=Excellent, Gd=Bon, TA=Moyen, Fa=Correct, Po=Pauvre, NA=Pas de cheminée",
            )

            # GarageType corrects
            garage_type = st.selectbox(
                "Type de garage (GarageType)",
                options=[
                    "2Types",
                    "Attchd",
                    "Basment",
                    "BuiltIn",
                    "CarPort",
                    "Detchd",
                    "NA",
                ],
                index=1,  # Attchd par défaut
                help="Attchd=Attaché, Detchd=Détaché, BuiltIn=Intégré, Basment=Sous-sol, CarPort=Carport, NA=Pas de garage",
            )

        with col_adv2:
            # GarageFinish corrects
            garage_finish = st.selectbox(
                "Finition du garage (GarageFinish)",
                options=["Fin", "RFn", "Unf", "NA"],
                index=1,  # RFn par défaut
                help="Fin=Fini, RFn=Finition rugueuse, Unf=Non fini, NA=Pas de garage",
            )

            # GarageQual corrects
            garage_qual = st.selectbox(
                "Qualité du garage (GarageQual)",
                options=["Ex", "Gd", "TA", "Fa", "Po", "NA"],
                index=2,  # TA par défaut
                help="Ex=Excellent, Gd=Bon, TA=Typique/Moyen, Fa=Correct, Po=Pauvre, NA=Pas de garage",
            )

            # PavedDrive corrects
            paved_drive = st.selectbox(
                "Allée pavée (PavedDrive)",
                options=["Y", "P", "N"],
                index=0,  # Y par défaut
                help="Y=Pavée, P=Partiellement pavée, N=Terre/Gravier",
            )

            # SaleCondition corrects
            sale_condition = st.selectbox(
                "Condition de vente (SaleCondition)",
                options=["Normal", "Abnorml", "AdjLand", "Alloca", "Family", "Partial"],
                index=0,  # Normal par défaut
                help="Normal=Vente normale, Abnorml=Anormale, AdjLand=Terrain adjacent, Family=Famille, Partial=Partielle",
            )

    # Bouton de prédiction stylisé
    st.markdown("---")
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])

    with col_btn2:
        predict_button = st.button(
            "🔮 Prédire le prix de la propriété",
            type="primary",
            use_container_width=True,
        )

    if predict_button:
        # Préparation des données - conversion des valeurs NA en None
        fireplace_qu_val = None if fireplace_qu == "NA" else fireplace_qu
        garage_type_val = None if garage_type == "NA" else garage_type
        garage_finish_val = None if garage_finish == "NA" else garage_finish
        garage_qual_val = None if garage_qual == "NA" else garage_qual

        input_data = {
            "GrLivArea": gr_liv_area,
            "OverallQual": overall_qual,
            "OverallCond": 5,  # valeur par défaut
            "YearBuilt": year_built,
            "YearRemodAdd": year_remod_add,
            "TotalBsmtSF": total_bsmt_sf,
            "FirstFlrSF": first_flr_sf,
            "SecondFlrSF": second_flr_sf,
            "GarageArea": garage_area,
            "GarageCars": garage_cars,
            "HeatingQC": heating_qc,
            "CentralAir": central_air,
            "KitchenQual": kitchen_qual,
            "Functional": functional,
            "FireplaceQu": fireplace_qu_val,
            "GarageType": garage_type_val,
            "GarageFinish": garage_finish_val,
            "GarageQual": garage_qual_val,
            "PavedDrive": paved_drive,
            "SaleCondition": sale_condition,
            "Neighborhood": neighborhood,
            "MSZoning": ms_zoning,
        }

        # Validation des champs obligatoires
        required_values = [
            gr_liv_area,
            overall_qual,
            year_built,
            total_bsmt_sf,
            garage_area,
        ]

        if any(v in [None, "", 0] for v in required_values):
            st.markdown(
                """
            <div class="error-box">
                ❗ Veuillez remplir tous les champs obligatoires pour obtenir une prédiction précise.
            </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            with st.spinner("🔄 Calcul de la prédiction en cours..."):
                try:
                    response = requests.post(
                        # "http://localhost:8000/api/predict",
                        "https://house-price-prediction-f3xa.onrender.com/api/predict",
                        json=input_data,
                        timeout=30,
                    )

                    if response.status_code == 200:
                        prediction = response.json().get(
                            "predicted_price", "Non défini"
                        )

                        # Affichage stylisé du résultat
                        st.markdown(
                            f"""
                        <div class="prediction-result">
                            💰 Prix estimé de la propriété<br>
                            <span style="font-size: 2rem;">${prediction:,.2f}</span>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )

                        # Métriques supplémentaires
                        col_m1, col_m2, col_m3 = st.columns(3)
                        with col_m1:
                            st.metric(
                                "Prix par sq ft",
                                f"${prediction/gr_liv_area:.2f}",
                                help="Prix par pied carré habitable",
                            )
                        with col_m2:
                            st.metric(
                                "Surface totale",
                                f"{gr_liv_area:,} sq ft",
                                help="Surface habitable",
                            )
                        with col_m3:
                            st.metric(
                                "Qualité",
                                f"{overall_qual}/10",
                                help="Note de qualité globale",
                            )

                    else:
                        st.markdown(
                            f"""
                        <div class="error-box">
                            ❌ Erreur de prédiction ({response.status_code}): {response.text}
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )

                except requests.exceptions.ConnectionError:
                    st.markdown(
                        """
                    <div class="error-box">
                        ❌ Impossible de se connecter au serveur de prédiction (localhost:8000)<br>
                        Vérifiez que l'API est en fonctionnement.
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )
                except requests.exceptions.Timeout:
                    st.markdown(
                        """
                    <div class="error-box">
                        ⏱️ Délai de connexion dépassé. Veuillez réessayer.
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

# ------------------------- Mode CSV -------------------------
else:
    st.markdown("## 📊 Prédiction en lot par fichier CSV")

    col_upload1, col_upload2 = st.columns([2, 1])

    with col_upload1:
        st.markdown(
            """
        ### 📤 Upload de votre fichier
        Uploadez un fichier CSV contenant les caractéristiques des propriétés à évaluer.
        """
        )

        uploaded_file = st.file_uploader(
            "Choisir un fichier CSV",
            type=["csv"],
            help="Le fichier doit contenir les colonnes requises pour la prédiction",
        )

    with col_upload2:
        st.markdown(
            """
        ### 📋 Format requis
        **Colonnes obligatoires:**
        - **GrLivArea**: Surface habitable
        - **OverallQual**: Qualité (1-10)
        - **YearBuilt**: Année construction
        - **TotalBsmtSF**: Surface sous-sol
        - **GarageArea**: Surface garage
        - **Neighborhood**: Quartier (voir liste)
        - **MSZoning**: Zone résidentielle
        
        **Quartiers valides:**
        Blmngtn, Blueste, BrDale, BrkSide, ClearCr, CollgCr, Crawfor, Edwards, Gilbert, etc.
        
        **Zones valides:**
        A, C, FV, I, RH, RL, RP, RM
        """
        )

    if uploaded_file is not None:
        try:
            # Lecture et affichage du fichier
            df = pd.read_csv(uploaded_file)

            st.markdown(
                """
            <div class="success-box">
                ✅ Fichier chargé avec succès !
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Métriques du fichier
            col_info1, col_info2, col_info3 = st.columns(3)
            with col_info1:
                st.metric("Nombre de lignes", len(df))
            with col_info2:
                st.metric("Nombre de colonnes", len(df.columns))
            with col_info3:
                st.metric("Taille du fichier", f"{uploaded_file.size / 1024:.1f} KB")

            # Aperçu des données
            st.markdown("### 👀 Aperçu des données")
            st.dataframe(df.head(10), use_container_width=True)

            # Bouton de prédiction
            col_batch1, col_batch2, col_batch3 = st.columns([1, 2, 1])
            with col_batch2:
                batch_predict_button = st.button(
                    "🚀 Lancer les prédictions en lot",
                    type="primary",
                    use_container_width=True,
                )

            if batch_predict_button:
                with st.spinner("🔄 Traitement des prédictions..."):
                    try:
                        response = requests.post(
                            # "http://localhost:8000/api/predict/batch",
                            "https://house-price-prediction-f3xa.onrender.com/api/predict/batch",
                            files={
                                "file": (
                                    uploaded_file.name,
                                    uploaded_file.getvalue(),
                                    "text/csv",
                                )
                            },
                            timeout=60,
                        )

                        if response.status_code == 200:
                            result_data = response.json()

                            st.markdown(
                                """
                            <div class="success-box">
                                ✅ Prédictions terminées avec succès !
                            </div>
                            """,
                                unsafe_allow_html=True,
                            )

                            # Traitement et affichage des résultats
                            if "predicted_prices" in result_data:
                                predictions = result_data["predicted_prices"]
                                errors = result_data.get("errors", [])

                                # Gérer le cas où predictions est une liste ou un dictionnaire
                                if isinstance(predictions, list):
                                    # Si c'est une liste, créer un dictionnaire avec les indices
                                    predictions_dict = {
                                        str(i): price
                                        for i, price in enumerate(predictions)
                                    }
                                    prices_values = predictions
                                elif isinstance(predictions, dict):
                                    # Si c'est déjà un dictionnaire
                                    predictions_dict = predictions
                                    prices_values = list(predictions.values())
                                else:
                                    st.error(
                                        "Format de données de prédiction non reconnu"
                                    )
                                    predictions_dict = {}
                                    prices_values = []

                                # Métriques des résultats
                                col_res1, col_res2, col_res3, col_res4 = st.columns(4)
                                with col_res1:
                                    st.metric(
                                        "Prédictions réussies", len(predictions_dict)
                                    )
                                with col_res2:
                                    st.metric("Erreurs", len(errors))
                                with col_res3:
                                    if prices_values:
                                        avg_price = sum(prices_values) / len(
                                            prices_values
                                        )
                                        st.metric("Prix moyen", f"${avg_price:,.2f}")
                                with col_res4:
                                    if prices_values:
                                        max_price = max(prices_values)
                                        st.metric("Prix maximum", f"${max_price:,.2f}")

                                # Création du DataFrame des résultats
                                results_list = []
                                for idx, price in predictions_dict.items():
                                    try:
                                        row_data = df.iloc[int(idx)].to_dict()
                                        row_data["Predicted_Price"] = f"${price:,.2f}"
                                        row_data["Index"] = idx
                                        results_list.append(row_data)
                                    except (IndexError, ValueError) as e:
                                        st.warning(
                                            f"Erreur lors du traitement de la ligne {idx}: {e}"
                                        )

                                if results_list:
                                    results_df = pd.DataFrame(results_list)

                                    # Réorganiser les colonnes pour mettre le prix en premier
                                    cols = ["Index", "Predicted_Price"] + [
                                        col
                                        for col in results_df.columns
                                        if col not in ["Index", "Predicted_Price"]
                                    ]
                                    results_df = results_df[cols]

                                    st.markdown("### 💰 Résultats des prédictions")
                                    st.dataframe(results_df, use_container_width=True)

                                    # Préparation du CSV pour téléchargement
                                    csv_buffer = StringIO()
                                    results_df.to_csv(csv_buffer, index=False)
                                    csv_data = csv_buffer.getvalue()

                                    # Bouton de téléchargement
                                    st.download_button(
                                        label="💾 Télécharger les résultats (CSV)",
                                        data=csv_data,
                                        file_name=f"predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                        mime="text/csv",
                                        use_container_width=True,
                                    )

                                # Affichage des erreurs s'il y en a
                                if errors:
                                    st.markdown("### ⚠️ Erreurs rencontrées")
                                    for error in errors:
                                        st.error(
                                            f"Ligne {error.get('index', 'N/A')}: {error.get('message', 'Erreur inconnue')}"
                                        )

                        else:
                            st.markdown(
                                f"""
                            <div class="error-box">
                                ❌ Erreur lors du traitement ({response.status_code}): {response.text}
                            </div>
                            """,
                                unsafe_allow_html=True,
                            )

                    except requests.exceptions.ConnectionError:
                        st.markdown(
                            """
                        <div class="error-box">
                            ❌ Impossible de se connecter au serveur de prédiction (localhost:8000)
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )
                    except requests.exceptions.Timeout:
                        st.markdown(
                            """
                        <div class="error-box">
                            ⏱️ Délai de traitement dépassé. Le fichier est peut-être trop volumineux.
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )

        except Exception as e:
            st.markdown(
                f"""
            <div class="error-box">
                ❌ Erreur lors de la lecture du fichier: {str(e)}
            </div>
            """,
                unsafe_allow_html=True,
            )

# Footer
st.markdown("---")
st.markdown(
    """
<div style="text-align: center; color: #666; padding: 2rem;">
    🏠 Prédicteur de Prix Immobilier - Développé avec ❤️ et Streamlit<br>
    <small>Utilise l'intelligence artificielle pour des estimations précises basées sur le dataset Ames Housing</small>
</div>
""",
    unsafe_allow_html=True,
)
