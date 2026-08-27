"""
Module de détection des câbles à partir de données LiDAR.
"""

import h5py
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
import os

# --- PARAMÈTRES GLOBAUX ---
MIN_HEIGHT = 10.0              
MIN_SEGMENT_LENGTH = 1.5       
ANGLE_TOLERANCE = 10.0         
CORRIDOR_WIDTH = 25.0          
MIN_SPANNED_LENGTH = 50.0      
MIN_LINEAR_DENSITY = 0.5       

def angle_diff(a, b):
    diff = abs(a - b)
    return min(diff, 180 - diff)

def compute_gap_segmented_bboxes(points, gap_threshold=10.0):
    """
    Crée des Bounding Boxes le long du câble, coupées s'il y a un trou > 2m.
    Garde l'orientation globale, mais ajuste la Longueur ET la Largeur 
    pour coller exactement aux points de chaque segment !
    """
    if len(points) < 5:
        return []

    # 1. PCA GLOBALE : On calcule l'orientation sur TOUT le câble pour la stabilité
    xy_pts = points[:, :2]
    pca = PCA(n_components=2).fit(xy_pts)
    main_dir = pca.components_[0]  # Axe de la longueur (Orientation du câble)
    ortho_dir = pca.components_[1] # Axe de la largeur (Perpendiculaire)
    yaw = np.arctan2(main_dir[1], main_dir[0])

    # Projections de tous les points sur l'axe principal (pour trouver les trous)
    proj_main = np.dot(xy_pts, main_dir)

    # 2. TRI DES POINTS : On trie le long de l'axe principal
    sort_idx = np.argsort(proj_main)
    proj_main_sorted = proj_main[sort_idx]
    points_sorted = points[sort_idx]

    # 3. IDENTIFICATION DES COUPURES (Les "Trous" > 10m)
    segments = []
    current_segment = [0]

    for i in range(1, len(proj_main_sorted)):
        if proj_main_sorted[i] - proj_main_sorted[i-1] > gap_threshold:
            segments.append(current_segment)
            current_segment = [i]
        else:
            current_segment.append(i)
    segments.append(current_segment)

    # 4. CRÉATION DES SOUS-BOÎTES SUR-MESURE
    bboxes = []
    for seg_indices in segments:
        if len(seg_indices) < 3: 
            continue # On ignore les miettes isolées

        seg_pts = points_sorted[seg_indices]
        seg_xy = seg_pts[:, :2]
        
        # Projections LOCALES du segment sur les axes GLOBAUX
        seg_proj_main = np.dot(seg_xy, main_dir)
        seg_proj_ortho = np.dot(seg_xy, ortho_dir)

        # -- A. Longueur --
        seg_min_main = np.min(seg_proj_main)
        seg_max_main = np.max(seg_proj_main)
        length = max(seg_max_main - seg_min_main, 1.0) # Au moins 1m de long
        local_main_center = (seg_max_main + seg_min_main) / 2.0

        # -- B. Largeur (NOUVEAU: Ajustement Local Exact) --
        seg_min_ortho = np.min(seg_proj_ortho)
        seg_max_ortho = np.max(seg_proj_ortho)
        width = max(seg_max_ortho - seg_min_ortho, 0.3) # Au moins 30cm de large (très serré)
        local_ortho_center = (seg_max_ortho + seg_min_ortho) / 2.0

        # Centre X, Y reconstruit à partir des centres locaux
        center_xy = local_main_center * main_dir + local_ortho_center * ortho_dir
        center_x, center_y = center_xy[0], center_xy[1]

        # -- C. Hauteur robuste (Percentiles 5% - 95%) --
        z_vals = seg_pts[:, 2]
        z_min = np.percentile(z_vals, 5)
        z_max = np.percentile(z_vals, 100)
        height = max(z_max - z_min, 0.3) # Au moins 30cm de haut
        center_z = (z_max + z_min) / 2.0

        # Ajout de la boîte (Le yaw reste identique pour toutes les boîtes du câble !)
        bboxes.append((center_x, center_y, center_z, width, length, height, yaw))

    return bboxes


def run_cable_detection(h5_path, output_csv):
    """
    Pipeline principal pour l'extraction des câbles.
    """
    print(f"🚀 Lancement de l'extraction sur : {os.path.basename(h5_path)}")
    predictions = []

    with h5py.File(h5_path, 'r') as f:
        dataset = f['lidar_points']
        ego_yaw_all = dataset['ego_yaw'][:]
        
        # Identification des frames
        changes = np.where(np.diff(ego_yaw_all) != 0)[0] + 1
        boundaries = [0] + changes.tolist() + [len(ego_yaw_all)]
        del ego_yaw_all
        
        total_frames = len(boundaries) - 1
        print(f"📊 {total_frames} frames détectées. Traitement en cours...")
        
        for n_index in range(total_frames):
            if n_index % 10 == 0:
                print(f"   ⏳ Traitement Frame {n_index}/{total_frames}...")
                
            # --- A. Extraction et Conversion ---
            frame_data = dataset[boundaries[n_index]:boundaries[n_index+1]]
            df_frame = pd.DataFrame(frame_data)
            df_frame = df_frame[df_frame['distance_cm'] > 0]
            
            if df_frame.empty: continue
                
            ego_pose = (df_frame['ego_x'].iloc[0], df_frame['ego_y'].iloc[0], 
                        df_frame['ego_z'].iloc[0], df_frame['ego_yaw'].iloc[0])
            
            azimuth = np.deg2rad(df_frame['azimuth_raw'] / 100.0)
            elevation = np.deg2rad(df_frame['elevation_raw'] / 100.0)
            dist = df_frame['distance_cm'] / 100.0

            x = dist * np.cos(elevation) * np.cos(azimuth)
            y = -dist * np.cos(elevation) * np.sin(azimuth) # Left-Handed
            z = dist * np.sin(elevation)
            points = np.column_stack((x, y, z))
            
            # --- B. Filtre du Sol ---
            df = pd.DataFrame(points, columns=['x', 'y', 'z'])
            df['gx'] = np.floor((df['x'] - df['x'].min()) / 5.0)
            df['gy'] = np.floor((df['y'] - df['y'].min()) / 5.0)
            valid_ground = df.groupby(['gx', 'gy'])['z'].agg(['min', 'count']).reset_index()
            valid_ground = valid_ground[valid_ground['count'] >= 5]
            
            if valid_ground.empty: continue
                
            tree_g = cKDTree(valid_ground[['gx', 'gy']].values)
            _, idx = tree_g.query(df[['gx', 'gy']].values)
            mask_high = (df['z'] - valid_ground['min'].values[idx]) > MIN_HEIGHT
            high_points = points[mask_high]
            
            if len(high_points) < 10: continue
                
            # --- C. Graphe KNN et Longueur ---
            tree = cKDTree(high_points[:, :2])
            distances, indices = tree.query(high_points[:, :2], k=3)
            
            segments_info = []
            angles = []
            for i in range(len(high_points)):
                p0 = high_points[i][:2]
                for j in [1, 2]:
                    if distances[i, j] > MIN_SEGMENT_LENGTH:
                        idx_n = indices[i, j]
                        p_n = high_points[idx_n][:2]
                        angle = np.degrees(np.arctan2(p_n[1] - p0[1], p_n[0] - p0[0])) % 180
                        segments_info.append((p0, p_n, angle, i, idx_n, distances[i, j]))
                        angles.append(angle)
                        
            # --- D. Directions Majoritaires ---
            if not angles: continue
            counts, bins = np.histogram(angles, bins=36, range=(0, 180))
            sorted_bins = np.argsort(counts)[::-1]
            
            top_angles = []
            for b in sorted_bins:
                if counts[b] < 5: break
                ang = (bins[b] + bins[b+1]) / 2.0
                if all(angle_diff(ang, a) > 15 for a in top_angles):
                    top_angles.append(ang)
                    if len(top_angles) == 2: break
                        
            # --- E. Validation des Couloirs et Création des BBox ---
            for majority_angle in top_angles:
                aligned = [info for info in segments_info if angle_diff(info[2], majority_angle) <= ANGLE_TOLERANCE]
                if not aligned: continue
                    
                rad_angle = np.radians(majority_angle)
                normal_vec = np.array([-np.sin(rad_angle), np.cos(rad_angle)])
                dir_vec = np.array([np.cos(rad_angle), np.sin(rad_angle)])
                
                projs_norm = np.array([np.dot((info[0]+info[1])/2.0, normal_vec) for info in aligned])
                
                # Meilleur couloir
                best_count, best_center = 0, 0
                for p in projs_norm:
                    count = np.sum((projs_norm >= p - CORRIDOR_WIDTH/2) & (projs_norm <= p + CORRIDOR_WIDTH/2))
                    if count > best_count:
                        best_count, best_center = count, p
                        
                corridor_segs = [aligned[i] for i, proj in enumerate(projs_norm) if abs(proj - best_center) <= CORRIDOR_WIDTH/2]
                if not corridor_segs: continue
                    
                # Densité Linéaire
                proj_along = []
                total_len = 0.0
                cable_indices_3d = set()
                
                for info in corridor_segs:
                    proj_along.extend([np.dot(info[0], dir_vec), np.dot(info[1], dir_vec)])
                    total_len += info[5]
                    cable_indices_3d.update([info[3], info[4]])
                
                spanned_length = max(proj_along) - min(proj_along)
                
                # Validation Finale
                if spanned_length > MIN_SPANNED_LENGTH and (total_len / spanned_length) > MIN_LINEAR_DENSITY:
                    cable_points = high_points[list(cable_indices_3d)]
                    
                    # Appel de la fonction de segmentation (Découpe tous les 15 mètres)
                    bboxes = compute_gap_segmented_bboxes(cable_points, gap_threshold=10.0) 
                    
                    total_length = sum([bbox[4] for bbox in bboxes]) # index 4 = length
                    if len(bboxes) <= 2 and total_length < 30.0:
                        continue
                        
                    for bbox in bboxes:
                        cx, cy, cz, width, length, height, yaw = bbox
                        predictions.append({
                            'ego_x': ego_pose[0], 'ego_y': ego_pose[1], 'ego_z': ego_pose[2], 'ego_yaw': ego_pose[3],
                            'bbox_center_x': cx, 'bbox_center_y': cy, 'bbox_center_z': cz,
                            'bbox_width': width, 'bbox_length': length, 'bbox_height': height,
                            'bbox_yaw': yaw,
                            'class_ID': 1, 'class_label': 'Cable'
                        })
                        
    # --- 4. SAUVEGARDE FORMAT AIRBUS ---
    df_preds = pd.DataFrame(predictions)
    cols = ['ego_x', 'ego_y', 'ego_z', 'ego_yaw', 'bbox_center_x', 'bbox_center_y', 
            'bbox_center_z', 'bbox_width', 'bbox_length', 'bbox_height', 'bbox_yaw', 
            'class_ID', 'class_label']

    if not df_preds.empty:
        df_preds = df_preds[cols]
        df_preds.to_csv(output_csv, index=False)
        print(f"\n🎉 SUCCÈS ! {len(df_preds)} câbles détectés dans toute la scène.")
        print(f"📁 Fichier sauvegardé : {output_csv}")
    else:
        print("\n⚠️ Aucun câble n'a validé les critères stricts dans cette scène.")