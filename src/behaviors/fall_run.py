import math
from typing import Dict, Any, List, Optional
import numpy as np
from src.behaviors.base import BaseBehaviorDetector
from src.behaviors.fall_fsm import FallFSMMixin
from src.tracking.track_manager import TrackHistory
from src.utils.geometry import calculate_torso_angle


class MotionAndFallDetector(FallFSMMixin, BaseBehaviorDetector):
    """Running kinematics and temporal fall FSM with optional Bi-LSTM verification."""
    def __init__(self, config_run: Dict[str, Any], config_fall: Dict[str, Any], bilstm_model=None, alert_manager=None):
        super().__init__(config_fall, alert_manager)
        self.config_run = config_run
        self.config_fall = config_fall
        self.bilstm_model = bilstm_model

        # Ngưỡng chạy
        self.v_norm_thresh = self.config_run.get("v_norm_threshold", 1.5)
        self.body_lean_min = self.config_run.get("body_lean_min", 15.0)
        self.cooldown_run = self.config_run.get("cooldown_seconds", 4.0)
        self.run_window = self.config_run.get("window_seconds", 0.5)
        self.run_confirm_frames = self.config_run.get("confirm_frames", 3)

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
        zones: List[Dict[str, Any]]
    ):
        track.debug_metrics = {}

        # Chỉ reset trạng thái RUNNING_ALERT; còn FALLING_ALERT do FSM Pha 3 (FALL_LATCHED) quản lý
        if track.current_state == "RUNNING_ALERT":
            track.current_state = "NORMAL"

        if not track.bbox_history or not track.timestamp_history:
            return
        kpts = (track.keypoints_history[-1] if track.keypoints_history
                and track.missing_kpts_count == 0 else None)
        shoulder = self._midpoint(kpts, 5, 6)
        hip = self._midpoint(kpts, 11, 12)
        angle = (calculate_torso_angle(shoulder, hip)
                 if shoulder is not None and hip is not None else None)
        body_lean = 90.0 - angle if angle is not None else 0.0
        if self.config_fall.get("enabled", True):
            self._process_fall(track, current_time, frame, angle, hip)
            if track.fall_phase != "NORMAL":
                track.run_streak = 0
                return
        else:
            self._reset_fall(track)

        # ---------------------------------------------------------
        # 1. KIỂM TRA HÀNH VI CHẠY (RUNNING KINEMATICS)
        # ---------------------------------------------------------
        # Không xét chạy nếu đối tượng đang bị khóa cảnh báo ngã (FALL_LATCHED)
        if self.config_run.get("enabled", True) and track.fall_phase != "FALL_LATCHED":
            v_norm = self._compute_run_speed(track)
            streak = getattr(track, "run_streak", 0)

            if v_norm is None:
                track.run_streak = 0
            else:
                track.debug_metrics["V"] = (f"V={v_norm:.2f} (>={self.v_norm_thresh})", v_norm >= self.v_norm_thresh)
                track.debug_metrics["Lean"] = (f"Lean={body_lean:.0f} (>={self.body_lean_min:.0f})", body_lean >= self.body_lean_min)

                is_fast = v_norm >= self.v_norm_thresh and (
                    body_lean >= self.body_lean_min or v_norm >= self.v_norm_thresh * 1.5
                )
                track.run_streak = streak + 1 if is_fast else 0
                track.debug_metrics["Run"] = (f"RunFrames={track.run_streak}/{self.run_confirm_frames}", is_fast)

                if track.run_streak >= self.run_confirm_frames:
                    track.current_state = "RUNNING_ALERT"
                    if track.can_trigger_alert("RUNNING", current_time, self.cooldown_run):
                        if self.alert_manager:
                            self.alert_manager.trigger_alert(
                                event_type="RUNNING",
                                track_id=track.track_id,
                                frame=frame,
                                confidence=min(1.0, v_norm / 2.5),
                                extra_info=f"Chạy bứt tốc: {v_norm:.1f} BH/s"
                            )
                    return

    def _compute_run_speed(self, track: TrackHistory) -> Optional[float]:
        """
        Vận tốc chuẩn hóa V_norm (Body Height / giây) trên cửa sổ thời gian `run_window` giây.
        """
        n = min(len(track.timestamp_history), len(track.bbox_history))
        if n < 4:
            return None
        ts = list(track.timestamp_history)[-n:]
        bb = list(track.bbox_history)[-n:]
        t_now = ts[-1]

        j = None
        for i in range(n - 1, -1, -1):
            if t_now - ts[i] >= self.run_window:
                j = i
                break
        if j is None:
            return None

        def mean_center(lo, hi):
            lo, hi = max(0, lo), min(n, hi)
            cx = sum((b[0] + b[2]) / 2.0 for b in bb[lo:hi]) / (hi - lo)
            cy = sum((b[1] + b[3]) / 2.0 for b in bb[lo:hi]) / (hi - lo)
            t = sum(ts[lo:hi]) / (hi - lo)
            return cx, cy, t

        old_x, old_y, old_t = mean_center(j - 1, j + 2)
        new_x, new_y, new_t = mean_center(n - 3, n)
        dt = new_t - old_t
        if not math.isfinite(dt) or dt <= 0:
            return None

        heights = sorted(max(1.0, b[3] - b[1]) for b in bb[j:])
        ref_height = heights[len(heights) // 2]
        return math.hypot(new_x - old_x, new_y - old_y) / (dt * ref_height)

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
