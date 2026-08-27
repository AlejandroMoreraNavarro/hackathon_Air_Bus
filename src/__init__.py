"""
Package principal pour la détection LiDAR.
"""

# On importe les fonctions utiles pour qu'elles soient accessibles directement via "src"
from .data_utils import get_frame_id_from_prediction, count_objects_in_scene_frame
from .detect_cables import compute_gap_segmented_bboxes
from .fusion_arbitre import process_lidar_frame, point_in_oriented_bbox
from .visualization import voxel_grid_downsample