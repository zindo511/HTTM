import math
from typing import Dict, Any, List, Optional
import numpy as np
from src.behaviors.base import BaseBehaviorDetector
from src.behaviors.fall_fsm import FallFSMMixin
from src.behaviors.running import RunningDetector
from src.tracking.track_manager import TrackHistory
from src.utils.geometry import calculate_torso_angle


class MotionAndFallDetector(FallFSMMixin, BaseBehaviorDetector):
    """Temporal fall FSM (with optional Bi-LSTM) plus abnormal-running detection."""
    def __init__(self, config_run: Dict[str, Any], config_fall: Dict[str, Any], bilstm_model=None, alert_manager=None):
        super().__init__(config_fall, alert_manager)
        self.config_run = config_run
        self.config_fall = config_fall
        self.bilstm_model = bilstm_model
        self.running = RunningDetector(config_run, alert_manager)
        # Gọi qua _compute_run_speed để test/script có thể thay thế cách đo tốc độ
        self.running.compute_speed = lambda track, frame_shape=None: self._compute_run_speed(track)

        # Ngưỡng ngã (3-Phase FSM)
        self.ar_max = self.config_fall.get("aspect_ratio_max", 0.8)
        self.angle_max = self.config_fall.get("torso_angle_max", 40.0)
        self.vy_drop_min = self.config_fall.get("v_y_drop_min", 0.8)
        self.bilstm_fall_thresh = self.config_fall.get("bilstm_fall_threshold", 0.85)
        self.cooldown_fall = self.config_fall.get("cooldown_seconds", 15.0)

    def process(
        self,
        track: TrackHistory,
        current_time: float,
        frame: np.ndarray,
        zones: Optional[List[Dict[str, Any]]] = None  # không dùng, giữ để tương thích
    ):
        track.debug_metrics = {}
        if track.current_state == "RUNNING_ALERT":
            track.current_state = "NORMAL"
        if not math.isfinite(current_time):
            self.running.reset(track)
            return
        if not track.bbox_history or not track.timestamp_history:
            return

        kpts = (track.keypoints_history[-1] if track.keypoints_history
                and track.missing_kpts_count == 0 else None)
        shoulder = self._midpoint(kpts, 5, 6)
        hip = self._midpoint(kpts, 11, 12)
        angle = (calculate_torso_angle(shoulder, hip)
                 if shoulder is not None and hip is not None else None)
        body_lean = 90.0 - angle if angle is not None else None

        # 1. Té ngã có ưu tiên cao hơn chạy
        if self.config_fall.get("enabled", True):
            self._process_fall(track, current_time, frame, angle, hip)
            if track.fall_phase != "NORMAL":
                self.running.reset(track)
                return
        else:
            self._reset_fall(track)

        # 2. Chạy bất thường: tốc độ + pose + ngữ cảnh, FSM NORMAL/SUSPECTED/CONFIRMED/COOLDOWN
        self.running.enabled = self.config_run.get("enabled", True)
        if self.running.enabled:
            self._frame_shape = frame.shape[:2] if frame is not None else None
            self.running.alert_manager = self.alert_manager  # đồng bộ nếu alert_manager bị thay sau khi khởi tạo
            self.running.process(track, current_time, frame, body_lean)
        else:
            self.running.reset(track)

    def _compute_run_speed(self, track: TrackHistory) -> Optional[float]:
        """Vận tốc chạy (BH/s, hoặc m/s khi có homography); dùng chung cho các script đánh giá."""
        return RunningDetector.compute_speed(self.running, track, getattr(self, "_frame_shape", None))

    def _trigger_fall_alert(self, track: TrackHistory, current_time: float, frame: np.ndarray, conf: float, detail: str):
        track.current_state = "FALLING_ALERT"
        if track.can_trigger_alert("FALLING", current_time, self.cooldown_fall):
            if self.alert_manager:
                self.alert_manager.trigger_alert(
                    event_type="FALLING",
                    track_id=track.track_id,
                    frame=frame,
                    confidence=conf,
                    extra_info=detail
                )
