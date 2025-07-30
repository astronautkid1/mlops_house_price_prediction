#!/bin/bash

# Script pour exécuter les notebooks Jupyter de manière séquentielle
# Utilise papermill pour l'exécution et nbconvert pour la conversion en HTML

# Détecter le répertoire racine du script pour rendre les chemins relatifs
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

# Configuration des chemins relatifs
NOTEBOOKS_DIR="${SCRIPT_DIR}/notebooks"
OUTPUT_DIR="${SCRIPT_DIR}/output_files"
LOGS_DIR="${SCRIPT_DIR}/logs"
REPORTS_DIR="${SCRIPT_DIR}/reports"

# Créer les répertoires de logs, rapports et output s'ils n'existent pas
mkdir -p "$LOGS_DIR"
mkdir -p "$REPORTS_DIR"
mkdir -p "$OUTPUT_DIR"

# Nom du fichier de log pour cette exécution
LOG_FILE="${LOGS_DIR}/notebook_execution_$(date +%Y%m%d_%H%M%S).log"

# Liste des notebooks à exécuter dans l'ordre
NOTEBOOKS=(
    "house_price_01_analyse.ipynb"
    "house_price_02_essais.ipynb"
)

echo "$(date +%Y-%m-%d_%H-%M-%S) - Début de l'exécution des notebooks" | tee -a "$LOG_FILE"

# Installer papermill et nbconvert si non installés
pip install papermill nbconvert jupyter ipykernel --quiet

# Boucle sur chaque notebook
for notebook in "${NOTEBOOKS[@]}"; do
    INPUT_NOTEBOOK="${NOTEBOOKS_DIR}/${notebook}"
    OUTPUT_NOTEBOOK="${OUTPUT_DIR}/executed_${notebook}"
    REPORT_HTML="${REPORTS_DIR}/${notebook%.ipynb}.html"

    echo "$(date +%Y-%m-%d_%H-%M-%S) - Exécution de ${notebook}..." | tee -a "$LOG_FILE"
    
    # Exécuter le notebook avec papermill
    papermill "$INPUT_NOTEBOOK" "$OUTPUT_NOTEBOOK" \
        --log-output \
        --stdout-file "$LOG_FILE" \
        --stderr-file "$LOG_FILE" \
        --log-level INFO
    
    if [ $? -eq 0 ]; then
        echo "$(date +%Y-%m-%d_%H-%M-%S) - ${notebook} exécuté avec succès." | tee -a "$LOG_FILE"
        
        # Convertir le notebook exécuté en HTML pour un rapport facile à lire
        echo "$(date +%Y-%m-%d_%H-%M-%S) - Conversion de ${notebook} en HTML..." | tee -a "$LOG_FILE"
        jupyter nbconvert --to html "$OUTPUT_NOTEBOOK" --output "$REPORT_HTML" --template full --TagRemovePreprocessor.remove_cell_tags="['skip-html']"
        
        if [ $? -eq 0 ]; then
            echo "$(date +%Y-%m-%d_%H-%M-%S) - Rapport HTML généré: ${REPORT_HTML}" | tee -a "$LOG_FILE"
        else
            echo "$(date +%Y-%m-%d_%H-%M-%S) - ERREUR: Échec de la conversion HTML pour ${notebook}." | tee -a "$LOG_FILE"
        fi
    else
        echo "$(date +%Y-%m-%d_%H-%M-%S) - ERREUR: Échec de l'exécution de ${notebook}. Voir le log pour plus de détails." | tee -a "$LOG_FILE"
        exit 1 # Arrêter le script en cas d'erreur
    fi
done

echo "$(date +%Y-%m-%d_%H-%M-%S) - Fin de l'exécution des notebooks." | tee -a "$LOG_FILE"


