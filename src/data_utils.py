"""
Utilitaires pour la manipulation des données LiDAR et du Ground Truth.
"""

import h5py
import numpy as np
import pandas as pd

def get_frame_id_from_prediction(h5_path, pred_ego_yaw, pred_ego_x=None, tolerance=1e-4):
    """
    Retrouve l'ID (index) de la frame correspondant à une ligne de prédiction.
    
    Arguments:
    - h5_path : Chemin vers le fichier de la scène (ex: 'scene_10.h5')
    - pred_ego_yaw : La valeur 'ego_yaw' de ta ligne de prédiction
    - pred_ego_x : (Optionnel) Ajoute une vérification sur X pour plus de sécurité
    """
    with h5py.File(h5_path, 'r') as f:
        dataset = f['lidar_points']
        
        # On extrait les limites des frames
        ego_yaw_all = dataset['ego_yaw'][:]
        changes = np.where(np.diff(ego_yaw_all) != 0)[0] + 1
        boundaries = [0] + changes.tolist()
        
        # On cherche la frame correspondante
        for frame_idx, start_idx in enumerate(boundaries):
            frame_yaw = ego_yaw_all[start_idx]
            
            # On utilise une tolérance car les flottants CSV vs H5 peuvent varier infimement
            if abs(frame_yaw - pred_ego_yaw) < tolerance:
                if pred_ego_x is not None:
                    frame_x = dataset['ego_x'][start_idx]
                    if abs(frame_x - pred_ego_x) > tolerance:
                        continue # Faux positif sur le yaw, on continue
                return frame_idx
                
    return -1 # Frame non trouvée

def count_objects_in_scene_frame(GT_CSV, h5_path, frame_index, tolerance=1e-2):
    """
    Compte et renvoie les objets du ground truth pour une scène et une frame données.
    
    Arguments:
    - GT_CSV : Chemin vers 'ground_truth_train.csv'
    - h5_path : Chemin vers le fichier de la scène (ex: 'scene_10.h5')
    - frame_index : Le numéro de la frame (0, 1, 2...)
    """
    # 1. Récupérer l'ego_pose de la frame demandée dans le H5
    with h5py.File(h5_path, 'r') as f:
        dataset = f['lidar_points']
        ego_yaw_all = dataset['ego_yaw'][:]
        changes = np.where(np.diff(ego_yaw_all) != 0)[0] + 1
        boundaries = [0] + changes.tolist()
        
        if frame_index >= len(boundaries):
            raise ValueError(f"La frame {frame_index} n'existe pas dans ce fichier.")
            
        start_idx = boundaries[frame_index]
        target_yaw = ego_yaw_all[start_idx]
        target_x = dataset['ego_x'][start_idx]
        target_y = dataset['ego_y'][start_idx]

    # 2. Chercher cette pose dans le Ground Truth CSV
    df_gt = pd.read_csv(GT_CSV)
    
    # On filtre avec une tolérance pour les imprécisions des flottants
    mask = (
        (abs(df_gt['ego_yaw'] - target_yaw) < tolerance) &
        (abs(df_gt['ego_x'] - target_x) < tolerance) &
        (abs(df_gt['ego_y'] - target_y) < tolerance)
    )
    
    objects_in_frame = df_gt[mask]
    
    return len(objects_in_frame), objects_in_frame