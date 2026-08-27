"""
Module de visualisation 3D interactive des nuages de points et des prédictions (Plotly).
"""

import h5py
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os

# --- PARAMÈTRES ---
BG_VOXEL_SIZE = 5.0 

# Couleurs par classe
COLORS_PTS = {0: 'mediumorchid', 1: 'cyan', 2: 'darkorange', 3: 'blue'}
COLORS_PRED_BOX = {0: 'magenta', 1: 'cyan', 2: 'orange', 3: 'red'}
CLASS_NAMES = {0: 'Antenna', 1: 'Cable', 2: 'Pole', 3: 'Wind turbine'}


def voxel_grid_downsample(points, voxel_size):
    """Sous-échantillonnage uniforme du nuage de points pour alléger l'affichage."""
    if len(points) == 0: return np.array([])
    vox_indices = np.floor(points / voxel_size).astype(int)
    voxel_map = {}
    for i, idx_tuple in enumerate(zip(vox_indices[:,0], vox_indices[:,1], vox_indices[:,2])):
        if idx_tuple not in voxel_map:
            voxel_map[idx_tuple] = points[i]
    return np.array(list(voxel_map.values()))


def draw_bboxes(df_boxes, color_dict, prefix):
    """Fonction utilitaire pour dessiner les arêtes des Bounding Boxes 3D."""
    lines_trace = []
    added_legends = set() 
    
    for i, row in df_boxes.iterrows():
        cx, cy, cz = row['bbox_center_x'], row['bbox_center_y'], row['bbox_center_z']
        bw, bl, bh = row['bbox_width'], row['bbox_length'], row['bbox_height']
        byaw = row.get('bbox_yaw', 0.0)
        cid = int(row['class_ID'])
        
        box_color = color_dict.get(cid, 'black')
        class_name = CLASS_NAMES.get(cid, f'Cls {cid}')
        legend_name = f"{prefix} {class_name}"
        
        show_leg = legend_name not in added_legends
        added_legends.add(legend_name)
        
        dx, dy, dz = bl/2.0, bw/2.0, bh/2.0 
        c, s = np.cos(byaw), np.sin(byaw)
        rot_mat = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        
        corners_local = np.array([
            [dx, dy, dz], [dx, -dy, dz], [-dx, -dy, dz], [-dx, dy, dz],
            [dx, dy, -dz], [dx, -dy, -dz], [-dx, -dy, -dz], [-dx, dy, -dz]
        ])
        corners = np.dot(corners_local, rot_mat.T) + np.array([cx, cy, cz])
        
        lines = [(0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4), (0,4), (1,5), (2,6), (3,7)]
        for j, (p1, p2) in enumerate(lines):
            lines_trace.append(go.Scatter3d(
                x=[corners[p1, 0], corners[p2, 0]], 
                y=[corners[p1, 1], corners[p2, 1]], 
                z=[corners[p1, 2], corners[p2, 2]],
                mode='lines', line=dict(color=box_color, width=4), 
                showlegend=(show_leg and j==0), 
                name=legend_name
            ))
    return lines_trace


def visualize_frame(h5_path, pred_csv_path, frame_index=0):
    """
    Charge une frame spécifique depuis le fichier H5 et affiche les prédictions
    correspondantes dans une fenêtre de navigateur interactive via Plotly.
    """
    print(f"📊 Préparation de la visualisation pour la frame {frame_index}...")
    
    with h5py.File(h5_path, 'r') as f:
        ego_yaw_all = f['lidar_points']['ego_yaw'][:]
        changes = np.where(np.diff(ego_yaw_all) != 0)[0] + 1
        boundaries = [0] + changes.tolist() + [len(ego_yaw_all)]
        
        if frame_index >= len(boundaries) - 1:
            print(f"❌ La frame {frame_index} n'existe pas. Max: {len(boundaries)-2}")
            return
            
        start_idx = boundaries[frame_index]
        end_idx = boundaries[frame_index+1]
        df_frame = pd.DataFrame(f['lidar_points'][start_idx:end_idx])
        df_frame = df_frame[df_frame['distance_cm'] > 0]
        
        if df_frame.empty:
            print(f"⚠️ Frame {frame_index} vide.")
            return
            
        target_yaw = df_frame['ego_yaw'].iloc[0]
        target_x = df_frame['ego_x'].iloc[0]
        target_y = df_frame['ego_y'].iloc[0]
        
        az = np.deg2rad(df_frame['azimuth_raw'] / 100.0)
        el = np.deg2rad(df_frame['elevation_raw'] / 100.0)
        dist = df_frame['distance_cm'] / 100.0
        points = np.column_stack((
            dist * np.cos(el) * np.cos(az),
            -dist * np.cos(el) * np.sin(az),
            dist * np.sin(el)
        ))

    # --- Filtrage Strict Prédictions ---
    try:
        df_preds = pd.read_csv(pred_csv_path)
        mask_pred = (abs(df_preds['ego_x'] - target_x) < 1e-2) & \
                    (abs(df_preds['ego_y'] - target_y) < 1e-2) & \
                    (abs(df_preds['ego_yaw'] - target_yaw) < 1e-3)
        frame_preds = df_preds[mask_pred]
    except FileNotFoundError:
        print(f"⚠️ Fichier de prédictions introuvable : {pred_csv_path}")
        frame_preds = pd.DataFrame()

    # Identification des points dans les boîtes prédites
    point_classes = np.full(len(points), -1)
    
    for _, row in frame_preds.iterrows():
        cx, cy, cz = row['bbox_center_x'], row['bbox_center_y'], row['bbox_center_z']
        bw, bl, bh = row['bbox_width'], row['bbox_length'], row['bbox_height']
        byaw = row.get('bbox_yaw', 0.0)
        cid = int(row['class_ID'])
        
        pts_centered = points - np.array([cx, cy, cz])
        c, s = np.cos(-byaw), np.sin(-byaw)
        rot_inv = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        pts_local = np.dot(pts_centered, rot_inv.T)
        
        in_box = (
            (pts_local[:, 0] >= -bl/2.0) & (pts_local[:, 0] <= bl/2.0) &
            (pts_local[:, 1] >= -bw/2.0) & (pts_local[:, 1] <= bw/2.0) &
            (pts_local[:, 2] >= -bh/2.0) & (pts_local[:, 2] <= bh/2.0)
        )
        point_classes[in_box] = cid

    # --- Construction de la Scène 3D ---
    fig = go.Figure()
    
    # Nuage de points (Fond)
    bg_mask = (point_classes == -1)
    bg_plot = voxel_grid_downsample(points[bg_mask], BG_VOXEL_SIZE) 
    if len(bg_plot) > 0:
        fig.add_trace(go.Scatter3d(x=bg_plot[:, 0], y=bg_plot[:, 1], z=bg_plot[:, 2],
            mode='markers', marker=dict(size=1.5, color='darkgrey', opacity=1.0), name='Fond'
        ))
        
    # Points colorés par classe prédite
    for cid in np.unique(point_classes):
        if cid == -1: continue
        cls_pts = points[point_classes == cid]
        color = COLORS_PTS.get(cid, 'blue')
        name = CLASS_NAMES.get(cid, f'Classe {cid}')
        fig.add_trace(go.Scatter3d(x=cls_pts[:, 0], y=cls_pts[:, 1], z=cls_pts[:, 2],
            mode='markers', marker=dict(size=3, color=color, opacity=1.0), name=f'Pts: {name}'
        ))

    # Boîtes
    if not frame_preds.empty:
        pred_traces = draw_bboxes(frame_preds, COLORS_PRED_BOX, 'Pred:')
        for trace in pred_traces: fig.add_trace(trace)

    fig.update_layout(
        title=f"Frame {frame_index} | Objets Prédits: {len(frame_preds)}", 
        scene=dict(aspectmode='data', bgcolor='white'),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    print("🌐 Ouverture de la visualisation dans le navigateur...")
    fig.show()