"""
Module de détection des éoliennes à partir de données LiDAR.
"""

import h5py
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import cv2
import os
from collections import defaultdict

# --- PARAMÈTRES ---
# Critères Pilier
GRID_RES = 0.5
MIN_PTS_IN_COLUMN = 50         
MIN_RADIUS = 1.0               
MAX_RADIUS = 4.5               
MIN_GAP_ANGLE = 160.0          
MAX_RADIAL_THICKNESS = 0.4     # 🚨 Réduit pour éliminer les faux positifs (arbres pleins)
MAX_Z_GAP = 2.5                
MIN_TOTAL_HEIGHT = 15.0        
NMS_DISTANCE = 10.0             

# Paramètres Voxel Growing (Moitié Haute)
BLADE_SEARCH_RADIUS = 60.0     
VOXEL_SIZE = 3.0               


def run_turbine_detection(h5_path, output_csv):
    """
    Pipeline principal pour l'extraction des éoliennes (piliers ancrés + nacelle/pales).
    """
    print(f"🚀 Génération CSV : Piliers ancrés + PCA Orientée sur {os.path.basename(h5_path)}")
    predictions = []

    with h5py.File(h5_path, 'r') as f:
        dataset = f['lidar_points']
        ego_yaw_all = dataset['ego_yaw'][:]
        changes = np.where(np.diff(ego_yaw_all) != 0)[0] + 1
        boundaries = [0] + changes.tolist() + [len(ego_yaw_all)]
        del ego_yaw_all
        
        total_frames = len(boundaries) - 1
        
        for n_index in range(total_frames):
            df_frame = pd.DataFrame(dataset[boundaries[n_index]:boundaries[n_index+1]])
            df_frame = df_frame[df_frame['distance_cm'] > 0]
            if df_frame.empty: continue
                
            ego_pose = (df_frame['ego_x'].iloc[0], df_frame['ego_y'].iloc[0], 
                        df_frame['ego_z'].iloc[0], df_frame['ego_yaw'].iloc[0])
            
            az = np.deg2rad(df_frame['azimuth_raw'] / 100.0)
            el = np.deg2rad(df_frame['elevation_raw'] / 100.0)
            dist = df_frame['distance_cm'] / 100.0
            points = np.column_stack((
                dist * np.cos(el) * np.cos(az),
                -dist * np.cos(el) * np.sin(az),
                dist * np.sin(el)
            ))
            
            gx = np.floor((points[:, 0] - np.min(points[:, 0])) / 10.0)
            gy = np.floor((points[:, 1] - np.min(points[:, 1])) / 10.0)
            df_g = pd.DataFrame({'gx': gx, 'gy': gy, 'z': points[:, 2]})
            ground_map = df_g.groupby(['gx', 'gy'])['z'].min().to_dict()
            
            z_ground_local = np.array([ground_map.get((xi, yi), 0) for xi, yi in zip(gx, gy)])
            high_pts = points[(points[:, 2] - z_ground_local) > 10.0]
            
            if len(high_pts) < 100: continue

            x_min, y_min = np.min(high_pts[:, 0]), np.min(high_pts[:, 1])
            px = ((high_pts[:, 0] - x_min) / GRID_RES).astype(int)
            py = ((high_pts[:, 1] - y_min) / GRID_RES).astype(int)
            
            heatmap, _, _ = np.histogram2d(px, py, bins=[np.max(px)+2, np.max(py)+2])
            img_dense = np.zeros_like(heatmap.T, dtype=np.uint8)
            img_dense[heatmap.T >= MIN_PTS_IN_COLUMN] = 255
            img_closed = cv2.morphologyEx(img_dense, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            contours, _ = cv2.findContours(img_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            frame_predictions = []

            for cnt in contours:
                (cx_px, cy_px), _ = cv2.minEnclosingCircle(cnt)
                cx = cx_px * GRID_RES + x_min
                cy = cy_px * GRID_RES + y_min
                
                dists_2d = np.linalg.norm(points[:, :2] - [cx, cy], axis=1)
                cyl_pts = points[dists_2d <= MAX_RADIUS + 1.0] 
                
                if len(cyl_pts) < 100: continue
                    
                z_sorted = np.sort(cyl_pts[:, 2])
                z_gaps = np.diff(z_sorted)
                break_indices = np.where(z_gaps > MAX_Z_GAP)[0]
                
                seg_starts = np.insert(break_indices + 1, 0, 0)
                seg_ends = np.append(break_indices, len(z_sorted) - 1)
                
                max_h, best_z_min, best_z_max = 0, 0, 0
                for s_idx, e_idx in zip(seg_starts, seg_ends):
                    h = z_sorted[e_idx] - z_sorted[s_idx]
                    if h > max_h:
                        max_h, best_z_min, best_z_max = h, z_sorted[s_idx], z_sorted[e_idx]
                        
                if max_h < MIN_TOTAL_HEIGHT: continue
                    
                z_mid = (best_z_max + best_z_min) / 2.0
                slice_mask = (cyl_pts[:, 2] >= z_mid - 2.5) & (cyl_pts[:, 2] <= z_mid + 2.5)
                slice_pts = cyl_pts[slice_mask]
                
                if len(slice_pts) < 30: continue
                    
                pts_2d = np.array(slice_pts[:, :2], dtype=np.float32)
                (scx, scy), s_radius = cv2.minEnclosingCircle(pts_2d)
                
                # --- TESTS PILIER ---
                if MIN_RADIUS <= s_radius <= MAX_RADIUS:
                    d_center = np.linalg.norm(pts_2d - [scx, scy], axis=1)
                    if np.std(d_center) > MAX_RADIAL_THICKNESS or np.mean(d_center) < s_radius * 0.7:
                        continue 
                        
                    angles = np.degrees(np.arctan2(pts_2d[:, 1] - scy, pts_2d[:, 0] - scx))
                    angles = np.sort(angles % 360)
                    max_gap = max(np.max(np.diff(angles)), 360.0 - (angles[-1] - angles[0]))
                    
                    if max_gap >= MIN_GAP_ANGLE:
                        # =========================================================
                        # ✅ PILIER VALIDÉ ! 
                        # =========================================================
                        
                        # 1. 🚨 FIX : Pousser le pilier vers le bas (Le point le plus bas de la colonne)
                        dist_to_pillar = np.linalg.norm(points[:, :2] - [scx, scy], axis=1)
                        pillar_col_pts = points[dist_to_pillar <= s_radius * 1.2]
                        if len(pillar_col_pts) > 0:
                            best_z_min = np.min(pillar_col_pts[:, 2]) # On ancre la boîte à la racine exacte !
                            best_z_max = np.max(pillar_col_pts[:, 2])

                        # 2. 🚨 FIX : Voxel Growing UNIQUEMENT sur la moitié haute du pilier
                        mid_z = (best_z_max + best_z_min) / 2.0
                        local_mask = (
                            (points[:, 0] >= scx - BLADE_SEARCH_RADIUS) & (points[:, 0] <= scx + BLADE_SEARCH_RADIUS) &
                            (points[:, 1] >= scy - BLADE_SEARCH_RADIUS) & (points[:, 1] <= scy + BLADE_SEARCH_RADIUS) &
                            (points[:, 2] >= mid_z) # <-- Bloque totalement l'aspiration du sol/buissons !
                        )
                        local_pts = points[local_mask]
                        
                        if len(local_pts) > 0:
                            voxels = np.floor(local_pts / VOXEL_SIZE).astype(int)
                            voxel_map = defaultdict(list)
                            for idx, v in enumerate(voxels):
                                voxel_map[tuple(v)].append(idx)
                                
                            hub_mask = (
                                (local_pts[:, 0] >= scx - 5.0) & (local_pts[:, 0] <= scx + 5.0) &
                                (local_pts[:, 1] >= scy - 5.0) & (local_pts[:, 1] <= scy + 5.0) &
                                (local_pts[:, 2] >= best_z_max - 5.0) & (local_pts[:, 2] <= best_z_max + 5.0)
                            )
                            seed_indices = np.where(hub_mask)[0]
                            visited_voxels = set(tuple(v) for v in voxels[seed_indices])
                            active_queue = list(visited_voxels)
                            
                            offsets = [(dx, dy, dz) for dx in [-1,0,1] for dy in [-1,0,1] for dz in [-1,0,1] if not (dx==0 and dy==0 and dz==0)]
                            occupied_voxels = set(voxel_map.keys())
                            
                            while active_queue:
                                cx_v, cy_v, cz_v = active_queue.pop(0)
                                for dx, dy, dz in offsets:
                                    neighbor = (cx_v + dx, cy_v + dy, cz_v + dz)
                                    if neighbor in occupied_voxels and neighbor not in visited_voxels:
                                        visited_voxels.add(neighbor)
                                        active_queue.append(neighbor)
                                        
                            turbine_indices = []
                            for v in visited_voxels:
                                turbine_indices.extend(voxel_map[v])
                                
                            # 3. 🚨 FIX : PCA pour l'Orientation et les dimensions
                            if len(turbine_indices) > 5:
                                turbine_pts = local_pts[turbine_indices]
                                xy_pts = turbine_pts[:, :2]
                                
                                pca = PCA(n_components=2).fit(xy_pts)
                                main_dir = pca.components_[0]  # Direction du rotor (Length)
                                ortho_dir = pca.components_[1] # Épaisseur de nacelle (Width)
                                byaw = np.arctan2(main_dir[1], main_dir[0])
                                
                                proj_main = np.dot(xy_pts, main_dir)
                                proj_ortho = np.dot(xy_pts, ortho_dir)
                                
                                # On s'assure que la boîte ne soit jamais plus fine que le pilier lui-même
                                bl = max(np.max(proj_main) - np.min(proj_main), s_radius * 2.0)
                                bw = max(np.max(proj_ortho) - np.min(proj_ortho), s_radius * 2.0)
                                
                                center_main = (np.max(proj_main) + np.min(proj_main)) / 2.0
                                center_ortho = (np.max(proj_ortho) + np.min(proj_ortho)) / 2.0
                                center_xy = center_main * main_dir + center_ortho * ortho_dir
                                
                                bx, by = center_xy[0], center_xy[1]
                                final_max_z = max(best_z_max, np.max(turbine_pts[:, 2]))
                                
                            else:
                                # Sécurité : Si aucune pale n'est trouvée (Pilier nu)
                                bx, by = scx, scy
                                bl, bw = s_radius * 2.0, s_radius * 2.0
                                byaw = 0.0
                                final_max_z = best_z_max

                        # 4. Construction de la Boîte
                        bh = final_max_z - best_z_min # Le plancher absolu !
                        bz = best_z_min + (bh / 2.0)
                        
                        frame_predictions.append({
                            'ego_x': ego_pose[0], 'ego_y': ego_pose[1], 'ego_z': ego_pose[2], 'ego_yaw': ego_pose[3],
                            'bbox_center_x': bx, 'bbox_center_y': by, 'bbox_center_z': bz,
                            'bbox_width': bw, 'bbox_length': bl, 'bbox_height': bh,
                            'bbox_yaw': byaw,
                            'pillar_cx': scx, 'pillar_cy': scy,
                            'class_ID': 3, 'class_label': 'Wind turbine'
                        })

            # 🚨 FILTRE NMS
            if frame_predictions:
                frame_predictions.sort(key=lambda x: x['bbox_width'])
                kept_predictions = []
                
                for current_box in frame_predictions:
                    overlap = False
                    for kept_box in kept_predictions:
                        dist = np.sqrt((current_box['pillar_cx'] - kept_box['pillar_cx'])**2 + 
                                       (current_box['pillar_cy'] - kept_box['pillar_cy'])**2)
                        if dist < NMS_DISTANCE:
                            overlap = True
                            break
                            
                    if not overlap:
                        kept_predictions.append(current_box)
                
                for p in kept_predictions:
                    p.pop('pillar_cx')
                    p.pop('pillar_cy')
                    predictions.append(p)

    # --- 3. EXPORT CSV ---
    df_preds = pd.DataFrame(predictions)

    if not df_preds.empty:
        cols = ['ego_x', 'ego_y', 'ego_z', 'ego_yaw', 'bbox_center_x', 'bbox_center_y', 
                'bbox_center_z', 'bbox_width', 'bbox_length', 'bbox_height', 'bbox_yaw', 
                'class_ID', 'class_label']
        df_preds = df_preds[cols]
        df_preds.to_csv(output_csv, index=False)
        print(f"\n🎉 PERFECTION ATTEINTE ! {len(df_preds)} éoliennes orientées et ancrées sauvegardées.")
    else:
        print("\n⚠️ Aucune éolienne validée.")