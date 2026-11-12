import math
import numpy as np
from typing import Tuple, List, Optional


def get_foot_point(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """
    Trích xuất tọa độ điểm tiếp đất (Foot Point - trung điểm cạnh đáy Bbox).
    bbox: (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, float(y2))


def calculate_torso_angle(shoulder_pt: Tuple[float, float], hip_pt: Tuple[float, float]) -> float:
    """
    Tính góc nghiêng của trục thân người theta so với phương ngang (trục hoành Ox).
    - Đứng thẳng: theta ≈ 70° - 90°
    - Ngã nằm ngang: theta < 30° - 40°
    """
    dx = abs(shoulder_pt[0] - hip_pt[0])
    dy = abs(shoulder_pt[1] - hip_pt[1])
    
    if dx == 0 and dy == 0:
        return 90.0
    
    angle_rad = math.atan2(dy, dx)
    return math.degrees(angle_rad)


def calculate_aspect_ratio(bbox: Tuple[float, float, float, float]) -> float:
    """
    Tính tỷ lệ co giãn khung bao Aspect Ratio: AR = Height / Width.
    - Đứng / đi: AR >= 1.5
    - Ngã nằm dài trên sàn: AR <= 0.8
    """
    x1, y1, x2, y2 = bbox
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    return h / w


def normalize_skeleton_root_relative(
    keypoints_17: np.ndarray, 
    bbox: Tuple[float, float, float, float]
) -> np.ndarray:
    """
    Chuẩn hóa 17 điểm khớp xương theo Root-Relative (trung điểm 2 khớp hông)
    và co giãn theo chiều cao H của Bounding Box.
    
    Đầu vào: keypoints_17 shape (17, 2) hoặc (17, 3)
    Đầu ra: vector 34 chiều phẳng (Flattened 34-dim array)
    """
    # Keypoint 11: Hông trái, Keypoint 12: Hông phải (theo chuẩn COCO 0-indexed)
    hip_left = keypoints_17[11, :2]
    hip_right = keypoints_17[12, :2]
    
    root_x = (hip_left[0] + hip_right[0]) / 2.0
    root_y = (hip_left[1] + hip_right[1]) / 2.0
    
    _, y1, _, y2 = bbox
    height = max(1.0, y2 - y1)
    
    norm_kpts = np.zeros((17, 2), dtype=np.float32)
    for k in range(17):
        norm_kpts[k, 0] = (keypoints_17[k, 0] - root_x) / height
        norm_kpts[k, 1] = (keypoints_17[k, 1] - root_y) / height
        
    return norm_kpts.flatten()


def is_same_person(kpts_a: Optional[np.ndarray], box_a, kpts_b: Optional[np.ndarray], box_b,
                   conf_thresh: float = 0.3, max_mean_dist: float = 0.15, min_common: int = 5) -> bool:
    """
    Hai detection có phải cùng một người không (YOLO đôi khi vẽ 2 khung lồng nhau cho một người,
    nhất là khi người nằm ngang hoặc ảnh bị xoay).
    So bộ khớp xương: nếu >= min_common khớp cùng nhìn thấy và khoảng cách trung bình giữa các khớp
    tương ứng < max_mean_dist x đường chéo khung nhỏ hơn thì coi là trùng.
    Hai người thật đứng sát / ôm vật nhau vẫn có khớp xương lệch nhau rõ nên không bị gộp.
    """
    if kpts_a is None or kpts_b is None:
        return False
    a, b = np.asarray(kpts_a), np.asarray(kpts_b)
    if a.shape != b.shape or a.ndim != 2 or a.shape[0] != 17:
        return False
    visible = np.ones(17, dtype=bool)
    if a.shape[1] > 2:
        visible = (a[:, 2] >= conf_thresh) & (b[:, 2] >= conf_thresh)
    visible &= np.isfinite(a[:, :2]).all(axis=1) & np.isfinite(b[:, :2]).all(axis=1)
    if int(visible.sum()) < min_common:
        return False
    diag = min(math.hypot(box_a[2] - box_a[0], box_a[3] - box_a[1]),
               math.hypot(box_b[2] - box_b[0], box_b[3] - box_b[1]))
    if diag <= 0:
        return False
    dist = np.linalg.norm(a[visible, :2] - b[visible, :2], axis=1).mean()
    return float(dist) / diag < max_mean_dist
