import time
import os
from typing import Dict, Any, List, Optional
import cv2
import numpy as np
from ultralytics import YOLO

from src.tracking.track_manager import TrackManager
from src.behaviors.intrusion import IntrusionDetector
from src.behaviors.loitering import LoiteringDetector
from src.behaviors.fall_run import MotionAndFallDetector
from src.models.bilstm_verifier import BiLSTMInferenceWrapper
from src.core.alert_manager import AlertManager
from src.database.db_service import DatabaseService
from src.utils.geometry import get_foot_point, is_point_in_polygon
from src.utils.visualizer import draw_bounding_box, draw_foot_point, draw_skeleton, draw_roi_zones, draw_hud


class SurveillancePipeline:
    """
    Nhạc trưởng điều phối toàn bộ hệ thống trinh sát (Core Pipeline):
    - Tầng 1: Đọc frame từ CaptureThread
    - Tầng 2: YOLOv8-Pose + ByteTrack -> Lấy Track ID & 17 khớp xương
    - Tầng 3: Chạy 3 module hành vi (Xâm nhập, Lảng vảng, Chạy & Ngã)
    - Tầng 4: Gọi AlertManager ghi nhận sự kiện & render khung hình
    """
    def __init__(self, config: Dict[str, Any], ws_broadcast_callback=None):
        self.config = config
        self.sys_cfg = config.get("system", {})
        self.track_cfg = config.get("tracking", {})
        self.behav_cfg = config.get("behaviors", {})
        
        # 1. Cơ sở dữ liệu và Alert Manager
        self.db_service = DatabaseService()
        self.alert_manager = AlertManager(
            snapshots_dir=config.get("web", {}).get("snapshots_dir", "data/snapshots"),
            db_service=self.db_service,
            ws_broadcast_callback=ws_broadcast_callback
        )
        
        # 2. Khởi tạo mô hình AI (YOLO-Pose & Bi-LSTM)
        model_pose_path = self.sys_cfg.get("model_pose_path", "yolov8n-pose.pt")
        print(f"[INFO] Đang nạp mô hình YOLO-Pose: {model_pose_path}...")
        self.yolo_model = YOLO(model_pose_path)
        
        self.bilstm_model = None
        if self.sys_cfg.get("enable_bilstm_tier2", False):
            bilstm_path = self.sys_cfg.get("model_bilstm_path", "weights/fall_bilstm.pt")
            if os.path.exists(bilstm_path):
                self.bilstm_model = BiLSTMInferenceWrapper(model_path=bilstm_path)

        # 3. Quản lý bám vết đối tượng (TrackManager)
        self.track_manager = TrackManager(
            track_ttl=self.track_cfg.get("track_ttl", 5.0),
            exit_grace_period=self.track_cfg.get("grace_period_exit", 2.0)
        )

        # 4. Các module hành vi nghiệp vụ
        self.intrusion_detector = IntrusionDetector(self.behav_cfg.get("intrusion", {}), self.alert_manager)
        self.loitering_detector = LoiteringDetector(self.behav_cfg.get("loitering", {}), self.alert_manager)
        self.motion_fall_detector = MotionAndFallDetector(
            config_run=self.behav_cfg.get("running", {}),
            config_fall=self.behav_cfg.get("falling", {}),
            bilstm_model=self.bilstm_model,
            alert_manager=self.alert_manager
        )

        # 5. Danh sách vùng cấm đa giác ROI
        self.zones = config.get("zones", [])
        
        # Thống kê hiệu năng
        self.total_alerts = 0
        self.last_time = time.time()
        self.fps = 0.0

    def update_zones(self, new_zones: List[Dict[str, Any]]):
        """Cho phép người dùng cập nhật lại vùng ROI ngay trên giao diện web"""
        self.zones = new_zones

    def process_frame(self, frame: np.ndarray, timestamp: float) -> np.ndarray:
        # Tính FPS tức thời
        dt = timestamp - self.last_time
        if dt > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt) if self.fps > 0 else (1.0 / dt)
        self.last_time = timestamp

        annotated_frame = frame.copy()
        
        # Vẽ các vùng cấm ROI lên khung hình trước
        draw_roi_zones(annotated_frame, self.zones)

        # 1. Lan truyền tiến YOLO-Pose + ByteTrack
        # Chỉ nhận diện lớp 0 (person), sử dụng tracker bytetrack
        results = self.yolo_model.track(
            source=frame,
            persist=True,
            classes=[0],
            tracker="bytetrack.yaml",
            conf=self.sys_cfg.get("confidence_threshold", 0.4),
            verbose=False
        )[0]

        active_count = 0
        current_time = timestamp

        # 2. Xử lý các đối tượng phát hiện và bám vết
        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            track_ids = results.boxes.id.int().cpu().numpy()
            
            # Keypoints: shape (N, 17, 3) hoặc (N, 17, 2)
            has_kpts = results.keypoints is not None
            kpts_data = results.keypoints.data.cpu().numpy() if has_kpts else None

            for i, (bbox, tid) in enumerate(zip(boxes, track_ids)):
                active_count += 1
                foot_point = get_foot_point(bbox)
                kpts = kpts_data[i] if has_kpts else None

                # Cập nhật thông tin vào TrackManager
                track = self.track_manager.update_track(
                    track_id=int(tid),
                    bbox=tuple(bbox),
                    foot_point=foot_point,
                    keypoints=kpts,
                    current_time=current_time
                )

                # 3. Phân tích 3 hành vi nghiệp vụ
                self.intrusion_detector.process(track, current_time, frame, self.zones)
                self.loitering_detector.process(track, current_time, frame, self.zones)
                self.motion_fall_detector.process(track, current_time, frame, self.zones)

                # 4. Render đồ họa trực quan
                label = f"ID: {tid:02d} [{track.current_state}]"
                draw_bounding_box(annotated_frame, bbox, label, state=track.current_state)
                
                # Kiểm tra xem chân có nằm trong bất kỳ vùng cấm nào không để đổi màu điểm tiếp đất
                is_foot_in_any = False
                for z in self.zones:
                    poly = np.array(z.get("polygon", []), dtype=np.int32)
                    if is_point_in_polygon(foot_point, poly):
                        is_foot_in_any = True
                        break
                draw_foot_point(annotated_frame, foot_point, is_inside=is_foot_in_any)
                
                if kpts is not None:
                    draw_skeleton(annotated_frame, kpts)

        # 5. Dọn dẹp bộ nhớ RAM cho các track đã rời đi quá hạn TTL
        self.track_manager.purge_stale_tracks(current_time)

        # 6. Vẽ thông số HUD lên màn hình
        draw_hud(annotated_frame, self.fps, active_count, self.db_service.count_total_events())

        return annotated_frame
