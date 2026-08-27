"""
Module de fusion et d'arbitrage spatial des prédictions (Câbles, Éoliennes, Antennes, Pylônes).
"""

import numpy as np
import pandas as pd
from scipy.ndimage import minimum_filter, maximum_filter
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components
import h5py
import os

# ==========================================
# 1. FONCTIONS DE BASE
# ==========================================

def get_robust_ground_mask(points, cell_size=0.5, height_threshold=0.25):
    x_min, y_min = np.min(points[:, :2], axis=0)
    x_indices = ((points[:, 0] - x_min) / cell_size).astype(np.int32)
    y_indices = ((points[:, 1] - y_min) / cell_size).astype(np.int32)
    
    df = pd.DataFrame({'x': x_indices, 'y': y_indices, 'z': points[:, 2]})
    min_z_map = df.groupby(['x', 'y'])['z'].min().reset_index()
    
    grid_w, grid_h = x_indices.max() + 1, y_indices.max() + 1
    z_grid = np.full((grid_w, grid_h), np.nan)
    z_grid[min_z_map['x'], min_z_map['y']] = min_z_map['z']
    
    z_grid_filled = pd.DataFrame(z_grid).interpolate(axis=0).ffill().bfill().values
    
    ground_surface = minimum_filter(z_grid_filled, size=5)
    ground_surface = maximum_filter(ground_surface, size=5)
    
    ground_z_at_points = ground_surface[x_indices, y_indices]
    return (points[:, 2] - ground_z_at_points) > height_threshold


def get_3d_clusters_from_grid(points, mask_no_ground, grid_resolution=1.0, density_threshold=1):
    pts_no_ground = points[mask_no_ground]
    x, y = pts_no_ground[:, 0], pts_no_ground[:, 1]
    x_min, y_min = x.min(), y.min()
    
    x_idx = ((x - x_min) / grid_resolution).astype(np.int32)
    y_idx = ((y - y_min) / grid_resolution).astype(np.int32)
    
    df_pts = pd.DataFrame({'x_idx': x_idx, 'y_idx': y_idx, 'pt_idx': np.arange(len(x))})
    cell_counts = df_pts.groupby(['x_idx', 'y_idx']).size().reset_index(name='count')
    dense_cells = cell_counts[cell_counts['count'] > density_threshold].copy()
    
    if len(dense_cells) == 0:
        return np.full(len(x), -1)
    
    dense_cells['x_coord'] = dense_cells['x_idx'] * grid_resolution
    dense_cells['y_coord'] = dense_cells['y_idx'] * grid_resolution
    
    eps_distance = grid_resolution * 1.5 
    clustering = DBSCAN(eps=eps_distance, min_samples=1).fit(dense_cells[['x_coord', 'y_coord']])
    dense_cells['cluster_label'] = clustering.labels_
    
    df_pts = df_pts.merge(dense_cells[['x_idx', 'y_idx', 'cluster_label']], on=['x_idx', 'y_idx'], how='left')
    return df_pts['cluster_label'].fillna(-1).values.astype(np.int32)

# ==========================================
# 2. LOGIQUE MÉTIER
# ==========================================

def process_lidar_frame(points, grid_resolution=1.0, density_threshold=1, min_height_meters=30, max_z_gap=3.0, merge_distance=20):
    mask_no_ground = get_robust_ground_mask(points)
    cluster_labels = get_3d_clusters_from_grid(points, mask_no_ground, grid_resolution, density_threshold)
    
    global_labels = np.full(len(points), -1)
    global_labels[mask_no_ground] = cluster_labels
    
    unique_clusters = set(global_labels)
    unique_clusters.discard(-1)
    
    next_new_label = max(unique_clusters) + 1 if unique_clusters else 0
    split_clusters = set()

    for cluster_id in list(unique_clusters):
        mask_cluster = (global_labels == cluster_id)
        cluster_points = points[mask_cluster]
        
        z_values = np.sort(cluster_points[:, 2])
        z_diffs = np.diff(z_values)
        
        if np.any(z_diffs > max_z_gap):
            gap_index = np.argmax(z_diffs)
            split_z_value = z_values[gap_index] + (z_diffs[gap_index] / 2.0)
            
            mask_top_part = mask_cluster & (points[:, 2] > split_z_value)
            
            global_labels[mask_top_part] = next_new_label
            split_clusters.add(cluster_id)      
            split_clusters.add(next_new_label)  
            next_new_label += 1
        else:
            split_clusters.add(cluster_id)

    filtered_clusters = set()
    for cluster_id in list(split_clusters):
        mask_cluster = (global_labels == cluster_id)
        cluster_points = points[mask_cluster]
        
        if len(cluster_points) == 0: continue
            
        cluster_height = cluster_points[:, 2].max() - cluster_points[:, 2].min()
        
        if cluster_height < min_height_meters:
            global_labels[mask_cluster] = -1 
        else:
            filtered_clusters.add(cluster_id)

    filtered_list = list(filtered_clusters)
    n_clusters = len(filtered_list)
    
    if n_clusters > 1:
        adj_matrix = np.zeros((n_clusters, n_clusters), dtype=bool)
        trees = {}
        for i, c_id in enumerate(filtered_list):
            trees[i] = cKDTree(points[global_labels == c_id][:, :3])
            
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                matches = trees[i].query_ball_tree(trees[j], r=merge_distance)
                if any(len(m) > 0 for m in matches):
                    adj_matrix[i, j] = True
                    adj_matrix[j, i] = True
                    
        n_components, component_labels = connected_components(adj_matrix, directed=False)
        
        new_global_labels = np.full_like(global_labels, -1)
        final_clusters = set()
        
        for i, old_c_id in enumerate(filtered_list):
            new_comp_id = component_labels[i] 
            new_global_labels[global_labels == old_c_id] = new_comp_id
            final_clusters.add(new_comp_id)
            
        global_labels = new_global_labels
    else:
        final_clusters = filtered_clusters

    return points, mask_no_ground, global_labels

# ==========================================
# 3. L'ARBITRE DE CLASSIFICATION SPATIALE
# ==========================================

def point_in_oriented_bbox(px, py, cx, cy, w, l, yaw, margin):
    dx, dy = px - cx, py - cy
    c, s = np.cos(-yaw), np.sin(-yaw)
    lx = dx * c - dy * s
    ly = dx * s + dy * c
    return (abs(lx) <= (l / 2.0 + margin)) and (abs(ly) <= (w / 2.0 + margin))

def run_fusion(h5_path, cables_csv, turbines_csv, output_final_csv):
    """
    Pipeline principal pour fusionner les CSV, arbitrer les nouvelles classes
    et appliquer le filtre spatial KDTree.
    """
    MAX_LIDAR_DISTANCE = 250.0  
    CORRIDOR_MARGIN = 15.0      
    TURBINE_MATCH_DIST = 50.0   
    ANTENNA_MAX_RATIO = 2.5     

    print("🚀 Démarrage de la Fusion de Modèles...")

    try:
        df_cables = pd.read_csv(cables_csv)
        df_turbines = pd.read_csv(turbines_csv)
        print(f"✅ Contextes chargés : {len(df_cables)} câbles, {len(df_turbines)} éoliennes.")
    except FileNotFoundError:
        print("⚠️ Fichiers Câbles ou Éoliennes introuvables. Mode aveugle.")
        df_cables, df_turbines = pd.DataFrame(), pd.DataFrame()

    # 🚨 Stockage exclusif des nouvelles prédictions
    new_predictions = []

    with h5py.File(h5_path, 'r') as f:
        dataset = f['lidar_points']
        ego_yaw_all = dataset['ego_yaw'][:]
        changes = np.where(np.diff(ego_yaw_all) != 0)[0] + 1
        boundaries = [0] + changes.tolist() + [len(ego_yaw_all)]
        del ego_yaw_all
        
        total_frames = len(boundaries) - 1
        
        for n_index in range(total_frames):
            if n_index % 5 == 0:
                print(f"⏳ Traitement de la frame {n_index}/{total_frames}...")
                
            df_frame = pd.DataFrame(dataset[boundaries[n_index]:boundaries[n_index+1]])
            df_frame = df_frame[df_frame['distance_cm'] > 0]
            df_frame = df_frame[(df_frame['distance_cm'] / 100.0) < MAX_LIDAR_DISTANCE] 
            
            if df_frame.empty: continue
                
            ego_pose = (df_frame['ego_x'].iloc[0], df_frame['ego_y'].iloc[0], 
                        df_frame['ego_z'].iloc[0], df_frame['ego_yaw'].iloc[0])
            
            az = np.deg2rad(df_frame['azimuth_raw'] / 100.0)
            el = np.deg2rad(df_frame['elevation_raw'] / 100.0)
            dist = df_frame['distance_cm'] / 100.0
            points = np.column_stack((
                dist * np.cos(el) * np.cos(az), -dist * np.cos(el) * np.sin(az), dist * np.sin(el)
            ))
            
            # 🚨 FIX : Masque strict au centimètre près sur X, Y et Yaw pour isoler LA bonne frame
            frame_cables = pd.DataFrame()
            frame_turbines = pd.DataFrame()
            
            if not df_cables.empty:
                mask_c = (abs(df_cables['ego_x'] - ego_pose[0]) < 1e-2) & \
                         (abs(df_cables['ego_y'] - ego_pose[1]) < 1e-2) & \
                         (abs(df_cables['ego_yaw'] - ego_pose[3]) < 1e-3)
                frame_cables = df_cables[mask_c]
                
            if not df_turbines.empty:
                mask_t = (abs(df_turbines['ego_x'] - ego_pose[0]) < 1e-2) & \
                         (abs(df_turbines['ego_y'] - ego_pose[1]) < 1e-2) & \
                         (abs(df_turbines['ego_yaw'] - ego_pose[3]) < 1e-3)
                frame_turbines = df_turbines[mask_t]

            # Appel du clustering du collègue (avec ses vrais paramètres pour tuer les faux arbres)
            _, _, labels = process_lidar_frame(
                points,
                grid_resolution=1.0, 
                density_threshold=1, 
                min_height_meters=30.0, 
                max_z_gap=3.0,
                merge_distance=20.0
            )
            
            unique_ids = set(labels) - {-1}
            
            for cid in unique_ids:
                cluster_pts = points[labels == cid]
                
                min_x, max_x = np.min(cluster_pts[:, 0]), np.max(cluster_pts[:, 0])
                min_y, max_y = np.min(cluster_pts[:, 1]), np.max(cluster_pts[:, 1])
                min_z, max_z = np.min(cluster_pts[:, 2]), np.max(cluster_pts[:, 2])
                
                cx, cy, cz = (max_x + min_x)/2.0, (max_y + min_y)/2.0, (max_z + min_z)/2.0
                bl, bw, bh = max(max_x - min_x, 1.0), max(max_y - min_y, 1.0), max_z - min_z
                
                # --- RÈGLE 1 : ÉOLIENNE DÉJÀ CONNUE ? ---
                is_known_turbine = False
                for _, t_row in frame_turbines.iterrows():
                    if np.sqrt((cx - t_row['bbox_center_x'])**2 + (cy - t_row['bbox_center_y'])**2) < TURBINE_MATCH_DIST:
                        is_known_turbine = True
                        break
                if is_known_turbine: continue

                # --- RÈGLE 2 : PYLÔNE DANS CÂBLES ? ---
                is_pylon = False
                for _, c_row in frame_cables.iterrows():
                    if point_in_oriented_bbox(cx, cy, c_row['bbox_width'], c_row['bbox_length'], 
                                              c_row['bbox_center_x'], c_row['bbox_center_y'], 
                                              c_row['bbox_yaw'], margin=CORRIDOR_MARGIN):
                        is_pylon = True
                        break
                
                if is_pylon:
                    assigned_class, assigned_label = 2, 'Pole'
                else:
                    # --- RÈGLE 3 : ANTENNE OU PALES ? ---
                    ratio = max(bl / bw, bw / bl)
                    if ratio <= ANTENNA_MAX_RATIO and bl < 20.0 and bw < 20.0:
                        assigned_class, assigned_label = 0, 'Antenna'
                    else:
                        continue
                        
                # Ajout du nouvel objet uniquement
                new_predictions.append({
                    'ego_x': ego_pose[0], 'ego_y': ego_pose[1], 'ego_z': ego_pose[2], 'ego_yaw': ego_pose[3],
                    'bbox_center_x': cx, 'bbox_center_y': cy, 'bbox_center_z': cz,
                    'bbox_width': bw, 'bbox_length': bl, 'bbox_height': bh,
                    'bbox_yaw': 0.0,
                    'class_ID': assigned_class, 'class_label': assigned_label
                })


    # ==========================================
    # 4. CONCATÉNATION FINALE & EXPORTATION
    # ==========================================
    df_new = pd.DataFrame(new_predictions)

    # Concaténation pure de tes CSV originaux + Nouveaux objets
    frames_to_concat = []
    if not df_cables.empty: frames_to_concat.append(df_cables)
    if not df_turbines.empty: frames_to_concat.append(df_turbines)
    if not df_new.empty: frames_to_concat.append(df_new)

    if frames_to_concat:
        df_final = pd.concat(frames_to_concat, ignore_index=True)
        
        # =========================================================
        # 🚨 LE FILTRE POST-TRAITEMENT : DISTANCE POINT-PAR-POINT
        # =========================================================
        cables = df_final[df_final['class_ID'] == 1]
        targets_idx = df_final[df_final['class_ID'].isin([0, 3])].index # Antennes et Éoliennes
        
        changed_count = 0
        for idx in targets_idx:
            row = df_final.loc[idx]
            px, py = row['bbox_center_x'], row['bbox_center_y']
            
            # On isole les câbles de LA bonne frame
            frame_cables = cables[abs(cables['ego_yaw'] - row['ego_yaw']) < 1e-3]
            if frame_cables.empty: continue
                
            # 1. On "voxelise" les boîtes de câbles en générant 1 point tous les 2 mètres
            cable_pts = []
            for _, c_row in frame_cables.iterrows():
                cx, cy = c_row['bbox_center_x'], c_row['bbox_center_y']
                bl, byaw = c_row['bbox_length'], c_row['bbox_yaw']
                c, s = np.cos(byaw), np.sin(byaw)
                
                # On marche le long de la ligne centrale du câble
                for step in np.arange(-bl/2.0, bl/2.0, 2.0):
                    cable_pts.append([cx + step * c, cy + step * s])
                    
            if not cable_pts: continue
            
            # 2. Recherche du point du câble le plus proche via KDTree (Zéro erreur trigonométrique !)
            tree = cKDTree(cable_pts)
            dist, _ = tree.query([px, py])
            
            # 3. Validation de la distance (Bord à Câble < 50m)
            target_radius = max(row['bbox_width'], row['bbox_length']) / 2.0
            if (dist - target_radius) < 50.0:
                df_final.at[idx, 'class_ID'] = 2
                df_final.at[idx, 'class_label'] = 'Pole'
                changed_count += 1
                
        if changed_count > 0:
            print(f"🔄 Règle Spatiale (KDTree Point-par-Point) : {changed_count} objets convertis en Pylônes !")
        # =========================================================
        
        cols = ['ego_x', 'ego_y', 'ego_z', 'ego_yaw', 'bbox_center_x', 'bbox_center_y', 
                'bbox_center_z', 'bbox_width', 'bbox_length', 'bbox_height', 'bbox_yaw', 
                'class_ID', 'class_label']
        df_final = df_final[cols]
        df_final.to_csv(output_final_csv, index=False)
        
        print(f"\n🎉 FUSION TERMINÉE ! Fichier généré : {output_final_csv}")
        print("\n📊 Bilan des détections (Toutes Frames) :")
        print(df_final['class_label'].value_counts())
    else:
        print("\n⚠️ Aucune prédiction générée.")