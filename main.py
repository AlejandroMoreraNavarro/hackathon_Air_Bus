"""
Script principal (Point d'entrée) pour la détection LiDAR.
Orchestre la détection des câbles, des éoliennes, la fusion et la visualisation.
"""

import os
import sys

# Importation de tes modules depuis le dossier src/
from src.detect_cables import run_cable_detection
from src.detect_turbines import run_turbine_detection
from src.fusion_arbitre import run_fusion
from src.visualization import visualize_frame

# ==========================================
# CONFIGURATION DES CHEMINS
# ==========================================
# On suppose que ton fichier .h5 est dans un dossier "data/raw/"
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
OUTPUT_DIR = os.path.join(DATA_DIR, 'output')

# Fichier d'entrée
H5_FILENAME = 'scene_1.h5'
H5_PATH = os.path.join(RAW_DIR, H5_FILENAME)

# Fichiers de sortie (Générés par le script)
CABLES_CSV = os.path.join(OUTPUT_DIR, 'predictions_cables.csv')
TURBINES_CSV = os.path.join(OUTPUT_DIR, 'predictions_turbines.csv')
FINAL_CSV = os.path.join(OUTPUT_DIR, 'predictions_FINAL.csv')

def main():
    print("==================================================")
    print("   🚀 PIPELINE DE DÉTECTION LIDAR - DÉMARRAGE   ")
    print("==================================================\n")

    # 1. Vérification de l'environnement
    if not os.path.exists(RAW_DIR):
        os.makedirs(RAW_DIR)
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    if not os.path.exists(H5_PATH):
        print(f"❌ ERREUR : Le fichier d'entrée est introuvable !")
        print(f"Veuillez placer '{H5_FILENAME}' dans le dossier '{RAW_DIR}'.")
        sys.exit(1)

    # 2. Détection des Câbles
    print("\n--- ÉTAPE 1 : DÉTECTION DES CÂBLES ---")
    run_cable_detection(H5_PATH, CABLES_CSV)

    # 3. Détection des Éoliennes
    print("\n--- ÉTAPE 2 : DÉTECTION DES ÉOLIENNES ---")
    run_turbine_detection(H5_PATH, TURBINES_CSV)

    # 4. Fusion et Arbitrage (Antennes, Pylônes)
    print("\n--- ÉTAPE 3 : FUSION ET ARBITRAGE SPATIAL ---")
    run_fusion(H5_PATH, CABLES_CSV, TURBINES_CSV, FINAL_CSV)

    # 5. Visualisation (Optionnelle)
    print("\n==================================================")
    print("   ✅ PIPELINE TERMINÉ AVEC SUCCÈS !   ")
    print("==================================================\n")
    
    choix = input("Voulez-vous visualiser la première frame (Frame 0) dans votre navigateur ? (o/n) : ")
    if choix.lower() == 'o':
        visualize_frame(H5_PATH, FINAL_CSV, frame_index=0)
    else:
        print("Fermeture du programme. À bientôt !")

if __name__ == "__main__":
    main()