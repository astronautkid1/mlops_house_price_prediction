# Dockerfile pour l'API de prédiction des prix des maisons
# Utilise une image Python officielle comme base
FROM python:3.11-slim-buster

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier les fichiers de dépendances et installer les packages
# Utiliser un fichier requirements.txt dédié pour l'API
COPY serving/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code de l'API (depuis le répertoire serving)
COPY serving/ ./serving/
COPY models/ ./models/
COPY settings/ ./settings/

# Exposer le port sur lequel l'API va écouter
EXPOSE 8000

# Définir les variables d'environnement pour le modèle et les métadonnées
# Ces chemins sont relatifs au WORKDIR /app
ENV MODEL_PATH="/app/models/house_price_best_model_elasticnet.pkl"
ENV MODEL_METADATA_PATH="/app/models/house_price_best_model_elasticnet_metadata.json"

# Commande pour lancer l'API avec Uvicorn
# Utilise gunicorn pour la production pour une meilleure gestion des processus
# et des workers. Le nombre de workers peut être ajusté en fonction des ressources.
CMD ["gunicorn", "serving.app:app", "--workers", "1", "--bind", "0.0.0.0:8000", "--worker-class", "uvicorn.workers.UvicornWorker"]

