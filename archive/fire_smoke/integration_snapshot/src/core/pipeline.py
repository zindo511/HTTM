import time
import threading
import os
from typing import Dict, Any, List, Optional
import cv2
import numpy as np
import torch
from ultralytics import YOLO

from src.tracking.track_manager import TrackManager
from src.behaviors.fall_run import MotionAndFallDetector
from src.behaviors.fighting import FightingDetector
from src.behaviors.fire_smoke import FireSmokeDetector
from src.models.bilstm_verifier import BiLSTMInferenceWrapper
from src.core.alert_manager import AlertManager
from src.database.db_service import DatabaseService
from src.utils.geometry import get_foot_point
from src.utils.visualizer import (
    draw_bounding_box,
    draw_foot_point,
    draw_skeleton,
    draw_hud,
    draw_debug_metrics,
    draw_fighting_overlay,
    draw_fire_smoke_overlay,
    STATE_VN_LABELS,
)


class SurveillancePipeline:
    """
    Nhạc trưởng điều phối toàn bộ hệ thống trinh sát (Core Pipeline):
    - Tầng 1: Đọc frame từ CaptureThread
    - Tầng 2: YOLOv8-Pose + ByteTrack -> Lấy Track ID & 17 khớp xương
    - Tầng 3: Chạy các module hành vi (Chạy nhanh, Té ngã, Đánh nhau)
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

        # Thiết bị suy luận: tự lùi về CPU nếu cấu hình yêu cầu CUDA mà máy không có
        self.device = str(self.sys_cfg.get("device", "cpu"))
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            print(f"[WARNING] Cấu hình device={self.device} nhưng không có CUDA, chuyển sang CPU.")
            self.device = "cpu"
        print(f"[INFO] Thiết bị suy luận: {self.device}")

        self.bilstm_model = None
        if self.sys_cfg.get("enable_bilstm_tier2", False):
            bilstm_path = self.sys_cfg.get("model_bilstm_path", "weights/fall_bilstm.pt")
            if os.path.exists(bilstm_path):
                self.bilstm_model = BiLSTMInferenceWrapper(model_path=bilstm_path, device=self.device)
            else:
                print(f"[WARNING] enable_bilstm_tier2=true nhưng không thấy {bilstm_path}; "
                      "té ngã chỉ được xác nhận bằng FSM hình học.")

        # 3. Quản lý bám vết đối tượng (TrackManager)
        self.track_manager = TrackManager(
            track_ttl=self.track_cfg.get("track_ttl", 5.0),
            max_missing_kpts=self.track_cfg.get("max_missing_kpts", 5),
            max_fill_seconds=self.track_cfg.get("max_fill_seconds", 0.25),
            motion_history_seconds=max(2.0, self.behav_cfg.get("running", {}).get("window_seconds", 0.5) + 0.5)
        )

        # 4. Các module hành vi nghiệp vụ
        self.motion_fall_detector = MotionAndFallDetector(
            config_run=self.behav_cfg.get("running", {}),
            config_fall=self.behav_cfg.get("falling", {}),
            bilstm_model=self.bilstm_model,
            alert_manager=self.alert_manager
        )
        self.fighting_detector = FightingDetector(self.behav_cfg.get("fighting", {}), self.alert_manager)

        # Mô hình phân loại clip cho đánh nhau (tuỳ chọn, train bằng train_fight_colab.ipynb)
        self.fight_model = None
        self.last_fight_probability = None
        clip_cfg = (self.behav_cfg.get("fighting", {}) or {}).get("clip_model") or {}
        if clip_cfg.get("enabled", False):
            clip_path = clip_cfg.get("path", "weights/fight_clip.pt")
            if os.path.exists(clip_path):
                from src.models.fight_classifier import FightClipInference
                self.fight_model = FightClipInference(clip_path, device=self.device)
                print(f"[INFO] Đã nạp mô hình đánh nhau: {clip_path} "
                      f"(ngưỡng đề xuất {self.fight_model.recommended_threshold:.2f})")
        else:
                print(f"[WARNING] clip_model.enabled=true nhưng không thấy {clip_path}; "
                      "đánh nhau chỉ dùng luật theo cặp người.")
        self._configure_fight_threshold()

        # Mô hình phát hiện sớm Khói & Lửa (train bằng train_fire_smoke_colab.ipynb)
        fire_cfg = self.behav_cfg.get("fire_smoke", {}) or {}
        self.fire_yolo_model = None
        self._fire_lock = threading.Lock()
        self.fire_smoke_detector = FireSmokeDetector(
            config=fire_cfg,
            fire_model=None,
            alert_manager=self.alert_manager
        )
        # Tắt trong config thì không nạp mô hình (tiết kiệm CPU); bật lại được từ dashboard.
        self.set_fire_smoke_enabled(bool(fire_cfg.get("enabled", True)))

        # Thống kê hiệu năng
        self.total_alerts = 0
        self.last_time = time.time()
        self.fps = 0.0

    @property
    def fire_model_path(self) -> str:
        return (self.behav_cfg.get("fire_smoke", {}) or {}).get("model_path", "weights/fire_smoke_yolov8n.pt")

    def set_fire_smoke_enabled(self, enabled: bool) -> bool:
        """Bật/tắt phát hiện Khói & Lửa lúc đang chạy. Trả về trạng thái thực tế sau khi đổi."""
        detector = self.fire_smoke_detector
        with self._fire_lock:
            if enabled and self.fire_yolo_model is None:
                if not os.path.exists(self.fire_model_path):
                    print(f"[INFO] Chưa có {self.fire_model_path} (hãy train bằng "
                          "train_fire_smoke_colab.ipynb và chép vào weights/).")
                    enabled = False
                else:
                    print(f"[INFO] Đang nạp mô hình Khói & Lửa: {self.fire_model_path}...")
                    self.fire_yolo_model = YOLO(self.fire_model_path)
            detector.fire_model = self.fire_yolo_model
            if detector.enabled != enabled:
                detector.reset()
            detector.enabled = enabled
        if not enabled:
            # Bỏ nhãn cảnh báo lửa còn sót trên các track đang hiển thị
            for track in list(self.track_manager.tracks.values()):
                if track.current_state == "FIRE_ALERT":
                    track.current_state = "NORMAL"
        return enabled

    @property
    def fire_smoke_enabled(self) -> bool:
        return bool(self.fire_smoke_detector.enabled and self.fire_yolo_model is not None)

    def request_reset(self):
        """Đánh dấu cần xoá trạng thái bám vết (gọi khi đổi nguồn video, an toàn giữa các luồng)."""
        self._reset_requested = True

    def _configure_fight_threshold(self):
        """Nếu config không đặt ngưỡng thì dùng ngưỡng được chọn lúc train (lưu trong checkpoint)."""
        if self.fight_model is not None and self.fighting_detector.model_threshold is None:
            self.fighting_detector.model_threshold = self.fight_model.recommended_threshold

    def _reset_tracking_state(self):
        self.track_manager.tracks.clear()
        self.fighting_detector = FightingDetector(self.behav_cfg.get("fighting", {}), self.alert_manager)
        self._configure_fight_threshold()
        if self.fight_model is not None:
            self.fight_model.reset()
        if hasattr(self, "fire_smoke_detector") and self.fire_smoke_detector is not None:
            self.fire_smoke_detector.reset()
        # Buộc ByteTrack khởi tạo lại để ID không nối từ video cũ sang video mới
        if getattr(self.yolo_model, "predictor", None) is not None:
            self.yolo_model.predictor = None
        self.fps = 0.0

    def process_frame(self, frame: np.ndarray, timestamp: float) -> np.ndarray:
        if getattr(self, "_reset_requested", False):
            self._reset_requested = False
            self._reset_tracking_state()
            self.last_time = timestamp

        # Tính FPS tức thời
        dt = timestamp - self.last_time
        if dt > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt) if self.fps > 0 else (1.0 / dt)
        self.last_time = timestamp

        annotated_frame = frame.copy()

        # 1. Lan truyền tiến YOLO-Pose + ByteTrack
        results = self.yolo_model.track(
            source=frame,
            persist=True,
            classes=[0],
            tracker=self.track_cfg.get("tracker_path", "bytetrack.yaml"),
            imgsz=self.sys_cfg.get("inference_imgsz", 640),
            conf=self.sys_cfg.get("confidence_threshold", 0.4),
            device=self.device,
            verbose=False
        )[0]

        active_count = 0
        current_time = timestamp

        active_tracks = []
        track_render_items = []

        # 2. Xử lý các đối tượng phát hiện và bám vết
        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            track_ids = results.boxes.id.int().cpu().numpy()

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

                # 3a. Phân tích Chạy & Té ngã
                # ID mới sinh ra giữa cú ngã: nhận lại trạng thái ngã của ID vừa mất
                if (len(track.bbox_history) == 1
                        and self.behav_cfg.get("falling", {}).get("enabled", True)):
                    self.motion_fall_detector.adopt_fall_candidate(
                        track, self.track_manager.tracks.values(), current_time)
                self.motion_fall_detector.process(track, current_time, annotated_frame)

                active_tracks.append(track)
                track_render_items.append((track, bbox, int(tid), foot_point, kpts))

        # 3b. Phân tích hành vi Ẩu đả / Đánh nhau
        fight_prob = self.fight_model.update(frame, current_time) if self.fight_model is not None else None
        self.last_fight_probability = fight_prob
        fight_info = self.fighting_detector.process_all(
            active_tracks, current_time, annotated_frame, model_prob=fight_prob)

        # 3c. Phân tích Khói & Lửa sớm + Hành vi Châm lửa
        fire_info = self.fire_smoke_detector.process_all(
            active_tracks, current_time, frame, device=self.device
        )

        # Preserve a latched fall even when other detectors change display state.
        for track in active_tracks:
            if track.fall_phase == "FALL_LATCHED":
                track.current_state = "FALLING_ALERT"

        # 4. Render đồ họa trực quan lên annotated_frame
        for track, bbox, tid, foot_point, kpts in track_render_items:
            vn_state = STATE_VN_LABELS.get(track.current_state, track.current_state)
            label = f"ID: {tid:02d} [{vn_state}]"
            draw_bounding_box(annotated_frame, bbox, label, state=track.current_state)

            if bbox[3] >= frame.shape[0] - 5:
                foot_point = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
            draw_foot_point(annotated_frame, foot_point)

            if kpts is not None:
                draw_skeleton(annotated_frame, kpts)

            if self.sys_cfg.get("debug_overlay", True):
                draw_debug_metrics(annotated_frame, bbox, getattr(track, "debug_metrics", {}))

        # Vẽ khung bao vùng ẩu đả và vùng Khói & Lửa lên annotated_frame
        draw_fighting_overlay(annotated_frame, fight_info)
        draw_fire_smoke_overlay(annotated_frame, fire_info)

        # 4b. Phát cảnh báo Ẩu đả kèm ảnh snapshot ĐÃ ĐƯỢC VẼ KHUNG & BANNER RÕ RÀNG
        for alert_item in fight_info.get("new_alerts", []):
            if self.alert_manager:
                self.alert_manager.trigger_alert(
                    event_type="FIGHTING",
                    track_id=alert_item["track_id"],
                    frame=annotated_frame,
                    confidence=alert_item["confidence"],
                    extra_info=alert_item["extra_info"]
                )

        # 4c. Phát cảnh báo Khói & Lửa kèm ảnh snapshot đã vẽ khung
        for alert_item in fire_info.get("new_alerts", []):
            if self.alert_manager:
                self.alert_manager.trigger_alert(
                    event_type="FIRE_SMOKE",
                    track_id=alert_item["track_id"],
                    frame=annotated_frame,
                    confidence=alert_item["confidence"],
                    extra_info=alert_item["extra_info"]
                )

        # 5. Dọn dẹp bộ nhớ RAM cho các track đã rời đi quá hạn TTL
        self.track_manager.purge_stale_tracks(current_time)

        # 6. Vẽ thông số HUD lên màn hình
        draw_hud(annotated_frame, self.fps, active_count, self.db_service.count_total_events())

        return annotated_frame
