import cv2
import numpy as np
from typing import List, Dict, Any, Tuple


# Màu sắc chuẩn (BGR)
COLOR_GREEN = (0, 255, 0)       # Normal
COLOR_RED = (0, 0, 255)         # Fall Alert
COLOR_ORANGE = (0, 140, 255)     # Running Alert
COLOR_YELLOW = (0, 255, 255)     # Fighting Suspect
COLOR_MAGENTA = (180, 0, 255)    # Fighting Alert
COLOR_BLUE = (255, 0, 0)
COLOR_WHITE = (255, 255, 255)

# 19 đường xương kết nối 17 khớp (COCO Skeleton)
SKELETON_PAIRS = [
    (0, 1), (0, 2), (1, 3), (2, 4),               # Vùng đầu
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),      # Vùng tay
    (5, 11), (6, 12), (11, 12),                   # Thân trên
    (11, 13), (13, 15), (12, 14), (14, 16)        # Chân
]

STATE_VN_LABELS = {
    "NORMAL": "BINH THUONG",
    "RUNNING_ALERT": "CHAY BUT TOC",
    "FALLING_ALERT": "TE NGA!",
    "FIGHTING_SUSPECT": "NGHI NGO AU DA",
    "FIGHTING_ALERT": "AU DA / DANH NHAU!"
}


def draw_bounding_box(frame: np.ndarray, bbox: Tuple[float, float, float, float], label: str, state: str = "NORMAL"):
    x1, y1, x2, y2 = map(int, bbox)
    
    # Chọn màu khung và độ dày theo trạng thái hành vi
    thickness = 2
    if "FALL" in state:
        color = COLOR_RED
        thickness = 3
    elif state == "FIGHTING_SUSPECT":
        color = COLOR_YELLOW
    elif "FIGHT" in state:
        color = COLOR_MAGENTA
        thickness = 3
    elif "RUN" in state:
        color = COLOR_ORANGE
        thickness = 2
    else:
        color = COLOR_GREEN

    # Vẽ hộp chữ nhật bao quanh người
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    
    # Vẽ nhãn phía trên hộp
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top_y = max(20, y1)
    label_x = max(0, min(x1, frame.shape[1] - w - 10))
    cv2.rectangle(frame, (label_x, top_y - 20), (label_x + w + 10, top_y), color, -1)
    cv2.putText(frame, label, (label_x + 5, top_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1, cv2.LINE_AA)


def draw_foot_point(frame: np.ndarray, foot_point: Tuple[float, float]):
    fx, fy = int(foot_point[0]), int(foot_point[1])
    color = COLOR_BLUE
    cv2.circle(frame, (fx, fy), 5, color, -1)
    cv2.circle(frame, (fx, fy), 8, color, 1)


def draw_debug_metrics(frame: np.ndarray, bbox: Tuple[float, float, float, float], metrics: Dict[str, Any]):
    """
    Hiển thị các số đo chẩn đoán bên cạnh khung người.
    Tự động thu gọn trên video khổ dọc/nhỏ để không che khuất đối tượng.
    """
    if not metrics:
        return
    h_frame, w_frame = frame.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)

    # Trên video hẹp (như 360x640), chỉ hiện tối đa 3 chỉ số quan trọng nhất
    if w_frame < 500:
        priority_keys = [k for k in ("Fight", "AR", "V") if k in metrics]
        lines = [metrics[k] for k in priority_keys]
    else:
        lines = list(metrics.values())

    if not lines:
        return

    line_h = 15
    box_w = 155 if w_frame < 500 else 185
    tx = x2 + 6 if x2 + 6 + box_w <= w_frame else max(4, x1 - box_w - 6)
    ty = max(line_h + 4, min(y1 + line_h, h_frame - line_h * len(lines) - 6))

    overlay = frame.copy()
    cv2.rectangle(overlay, (tx - 3, ty - 12), (tx + box_w, ty + line_h * (len(lines) - 1) + 5), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    for i, (text, ok) in enumerate(lines):
        color = COLOR_GREEN if ok else COLOR_WHITE
        cv2.putText(frame, text, (tx, ty + i * line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)


def draw_skeleton(frame: np.ndarray, keypoints: np.ndarray, conf_thresh: float = 0.4):
    """
    Vẽ 17 điểm khớp xương và các đường nối cơ thể người (COCO Format).
    """
    if keypoints is None or len(keypoints) < 17:
        return
        
    pts = {}
    for i in range(17):
        x, y = int(keypoints[i, 0]), int(keypoints[i, 1])
        conf = keypoints[i, 2] if keypoints.shape[1] > 2 else 1.0
        if conf >= conf_thresh and x > 0 and y > 0:
            pts[i] = (x, y)
            cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)

    for p1, p2 in SKELETON_PAIRS:
        if p1 in pts and p2 in pts:
            cv2.line(frame, pts[p1], pts[p2], (255, 100, 0), 2)


def draw_fighting_overlay(frame: np.ndarray, fight_info: Dict[str, Any]):
    """
    Vẽ cảnh báo trực quan nổi bật khi phát hiện Ẩu đả / Đánh nhau:
    1. Khung viền đỏ/tím bao quanh cụm đối tượng đang xô xát kèm nhãn rõ ràng.
    2. Đường nối tâm giữa 2 đối tượng.
    3. Banner cảnh báo khẩn cấp màu đỏ nổi bật phía trên màn hình.
    """
    if not fight_info or not fight_info.get("is_fighting", False):
        return

    h, w = frame.shape[:2]

    # 1. Vẽ đường nối giữa 2 tâm người đang xô xát
    for pt_a, pt_b in fight_info.get("links", []):
        ax, ay = int(pt_a[0]), int(pt_a[1])
        bx, by = int(pt_b[0]), int(pt_b[1])
        mx, my = (ax + bx) // 2, (ay + by) // 2
        cv2.line(frame, (ax, ay), (bx, by), COLOR_RED, 3, cv2.LINE_AA)
        cv2.circle(frame, (mx, my), 7, COLOR_YELLOW, -1)
        cv2.circle(frame, (mx, my), 9, COLOR_RED, 2)

    # 2. Vẽ vùng bao cụm xô xát (Conflict Box) với nền đỏ bán trong suốt
    conflict_boxes = fight_info.get("conflict_boxes", [])
    if conflict_boxes:
        overlay = frame.copy()
        for (bx1, by1, bx2, by2, lbl) in conflict_boxes:
            x1, y1 = max(2, int(bx1) - 8), max(25, int(by1) - 8)
            x2, y2 = min(w - 2, int(bx2) + 8), min(h - 2, int(by2) + 8)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), COLOR_RED, -1)
        cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)

        for (bx1, by1, bx2, by2, lbl) in conflict_boxes:
            x1, y1 = max(2, int(bx1) - 8), max(25, int(by1) - 8)
            x2, y2 = min(w - 2, int(bx2) + 8), min(h - 2, int(by2) + 8)
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_RED, 3)

            tag = f"[ ! {lbl} ! ]"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            ty = max(44, y1 - 22)
            cv2.rectangle(frame, (x1, ty - th - 8), (min(w - 2, x1 + tw + 12), ty + 2), COLOR_RED, -1)
            cv2.putText(frame, tag, (x1 + 6, ty - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_YELLOW, 2, cv2.LINE_AA)

    # 3. Vẽ viền khung hình nháy đỏ & Banner cảnh báo lớn ở phía trên/dưới HUD
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), COLOR_RED, 4)

    banner_text = "CANH BAO: PHAT HIEN AU DA / DANH NHAU!"
    font_scale = 0.52 if w < 500 else 0.7
    thickness = 2
    (tw, th), _ = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    if tw > w - 44:
        font_scale *= (w - 44) / tw
        (tw, th), _ = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    bx1 = max(6, (w - tw) // 2 - 14)
    bx2 = min(w - 6, (w + tw) // 2 + 14)
    by1 = 88
    by2 = by1 + th + 16

    cv2.rectangle(frame, (bx1, by1), (bx2, by2), COLOR_RED, -1)
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), COLOR_YELLOW, 2)
    cv2.putText(
        frame,
        banner_text,
        (bx1 + 14, by2 - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        COLOR_WHITE,
        thickness,
        cv2.LINE_AA
    )


def draw_hud(frame: np.ndarray, fps: float, active_tracks: int, alerts_count: int):
    """
    Vẽ bảng thông số hệ thống (Heads-Up Display) ở góc trên màn hình.
    """
    h, w, _ = frame.shape
    box_w = min(w - 20, 310)
    cv2.rectangle(frame, (10, 10), (10 + box_w, 78), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (10 + box_w, 78), (100, 100, 100), 1)
    
    cv2.putText(frame, f"FPS mau: {fps:.1f}", (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Nguoi trong tam nhin: {active_tracks:02d}", (18, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Tong so canh bao: {alerts_count:02d}", (18, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 255) if alerts_count > 0 else (200, 200, 200), 1, cv2.LINE_AA)
