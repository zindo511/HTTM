import math
import numpy as np
import cv2
from typing import Tuple, List, Optional


def get_foot_point(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """
    Trích xuất tọa độ điểm tiếp đất (Foot Point - trung điểm cạnh đáy Bbox).
    bbox: (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, float(y2))


def is_point_in_polygon(point: Tuple[float, float], polygon: np.ndarray) -> bool:
    """
    Kiểm tra điểm tiếp đất có nằm trong đa giác vùng cấm ROI hay không (Point-in-Polygon).
    Sử dụng cv2.pointPolygonTest: >= 0 là nằm trong hoặc trên biên đa giác.
    """
    if polygon is None or len(polygon) < 3:
        return False
    # Đảm bảo định dạng int32 cho cv2.pointPolygonTest
    pts = np.array(polygon, dtype=np.int32)
    dist = cv2.pointPolygonTest(pts, (float(point[0]), float(point[1])), False)
    return dist >= 0


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


def calculate_trajectory_displacement(points: List[Tuple[float, float]]) -> float:
    """
    Tính bán kính dịch chuyển tối đa so với tâm trung bình của quỹ đạo (Spatial Confinement).
    Phục vụ nhận diện Lảng vảng (Loitering).
    """
    if len(points) < 2:
        return 0.0
    
    arr = np.array(points)
    center = np.mean(arr, axis=0)
    distances = np.linalg.norm(arr - center, axis=1)
    return float(np.max(distances))


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
