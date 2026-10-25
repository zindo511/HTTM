import cv2
import numpy as np
from typing import List, Dict, Any, Tuple


# Màu sắc chuẩn (BGR)
COLOR_GREEN = (0, 255, 0)       # Normal
COLOR_RED = (0, 0, 255)         # Intrusion / Fall Alert
COLOR_ORANGE = (0, 140, 255)     # Running Alert
COLOR_YELLOW = (0, 255, 255)     # Loitering Alert
COLOR_BLUE = (255, 0, 0)
COLOR_WHITE = (255, 255, 255)

# 19 đường xương kết nối 17 khớp (COCO Skeleton)
SKELETON_PAIRS = [
    (0, 1), (0, 2), (1, 3), (2, 4),               # Vùng đầu
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),      # Vùng tay
    (5, 11), (6, 12), (11, 12),                   # Thân trên
    (11, 13), (13, 15), (12, 14), (14, 16)        # Chân
]


def draw_bounding_box(frame: np.ndarray, bbox: Tuple[float, float, float, float], label: str, state: str = "NORMAL"):
    x1, y1, x2, y2 = map(int, bbox)
    
    # Chọn màu khung theo trạng thái hành vi
    if "INTRUSION" in state or "FALL" in state:
        color = COLOR_RED
    elif "LOITER" in state:
        color = COLOR_YELLOW
    elif "RUN" in state:
        color = COLOR_ORANGE
    else:
        color = COLOR_GREEN

    # Vẽ hộp chữ nhật bao quanh người
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # Vẽ nhãn phía trên hộp
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - 20), (x1 + w + 10, y1), color, -1)
    cv2.putText(frame, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)


def draw_foot_point(frame: np.ndarray, foot_point: Tuple[float, float], is_inside: bool = False):
    fx, fy = int(foot_point[0]), int(foot_point[1])
    color = COLOR_RED if is_inside else COLOR_BLUE
    cv2.circle(frame, (fx, fy), 5, color, -1)
    cv2.circle(frame, (fx, fy), 8, color, 1)


def draw_skeleton(frame: np.ndarray, keypoints: np.ndarray, conf_thresh: float = 0.4):
    """
    Vẽ 17 điểm khớp xương và các đường nối cơ thể người (COCO Format).
    """
    if keypoints is None or len(keypoints) < 17:
        return
        
    pts = {}
    for i in range(17):
        x, y = int(keypoints[i, 0]), int(keypoints[i, 1])
        # Nếu có cột confidence ở vị trí 2
        conf = keypoints[i, 2] if keypoints.shape[1] > 2 else 1.0
        if conf >= conf_thresh and x > 0 and y > 0:
            pts[i] = (x, y)
            cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)

    # Vẽ đường nối giữa các khớp
    for p1, p2 in SKELETON_PAIRS:
        if p1 in pts and p2 in pts:
            cv2.line(frame, pts[p1], pts[p2], (255, 100, 0), 2)


def draw_roi_zones(frame: np.ndarray, zones: List[Dict[str, Any]]):
    """
    Vẽ các vùng cấm hoặc vùng giám sát đa giác lên khung hình.
    """
    overlay = frame.copy()
    
    for zone in zones:
        polygon = np.array(zone.get("polygon", []), dtype=np.int32)
        if len(polygon) < 3:
            continue
            
        zone_type = zone.get("type", "RESTRICTED_INTRUSION")
        color = COLOR_RED if zone_type == "RESTRICTED_INTRUSION" else COLOR_YELLOW
        name = zone.get("name", zone.get("id", "Zone"))
        
        # Tô màu mờ trong đa giác
        cv2.fillPoly(overlay, [polygon], color)
        # Vẽ đường viền đa giác
        cv2.polylines(frame, [polygon], isClosed=True, color=color, thickness=2)
        
        # Đặt tên vùng cấm
        top_pt = polygon[0]
        cv2.putText(frame, f"[ROI] {name}", (top_pt[0], top_pt[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Pha trộn hiệu ứng trong suốt alpha = 0.2
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)


def draw_hud(frame: np.ndarray, fps: float, active_tracks: int, alerts_count: int):
    """
    Vẽ bảng thông số hệ thống (Heads-Up Display) ở góc trên màn hình.
    """
    h, w, _ = frame.shape
    cv2.rectangle(frame, (10, 10), (320, 80), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (320, 80), (100, 100, 100), 1)
    
    cv2.putText(frame, f"FPS: {fps:.1f} (Drop-Oldest Zero Lag)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(frame, f"Nguoi trong tam nhin: {active_tracks:02d}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Tong so canh bao: {alerts_count:02d}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255) if alerts_count > 0 else (200, 200, 200), 1)
