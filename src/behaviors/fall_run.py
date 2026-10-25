import math
from typing import Dict, Any, List, Optional
import numpy as np
from src.behaviors.base import BaseBehaviorDetector
from src.tracking.track_manager import TrackHistory
from src.utils.geometry import calculate_aspect_ratio, calculate_torso_angle, normalize_skeleton_root_relative


class MotionAndFallDetector(BaseBehaviorDetector):
    """
    Module 3: Phát hiện Chạy bứt tốc (Running) và Té ngã (Falling).
    - Chạy: Đo vận tốc chuẩn hóa V_norm (BH/s) và độ đổ người về phía trước.
    - Ngã: Kiến trúc 2 tầng:
      + Tier 1: Bộ lọc hình thái (AR <= 0.8, góc trục thân theta < 40°, tốc độ rơi Vy).
      + Tier 2: Gọi mạng Bi-LSTM thẩm định chuỗi thời gian 30 frames (chống nhầm cúi/ngồi).
    """
    def __init__(self, config_run: Dict[str, Any], config_fall: Dict[str, Any], bilstm_model=None, alert_manager=None):
        super().__init__(config_fall, alert_manager)
        self.config_run = config_run
        self.config_fall = config_fall
        self.bilstm_model = bilstm_model
        
        # Ngưỡng chạy
        self.v_norm_thresh = self.config_run.get("v_norm_threshold", 1.5)
        self.body_lean_min = self.config_run.get("body_lean_min", 15.0)
        self.cooldown_run = self.config_run.get("cooldown_seconds", 4.0)
        
        # Ngưỡng ngã
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
        if len(track.bbox_history) < 5 or len(track.keypoints_history) == 0:
            return

        current_bbox = track.bbox_history[-1]
        x1, y1, x2, y2 = current_bbox
        current_height = max(1.0, y2 - y1)
        kpts = track.keypoints_history[-1]

        # ---------------------------------------------------------
        # 1. KIỂM TRA HÀNH VI CHẠY (RUNNING KINEMATICS)
        # ---------------------------------------------------------
        if self.config_run.get("enabled", True):
            # Tính vận tốc chuẩn hóa trong cửa sổ trượt 5 frames (~0.16s)
            prev_bbox = track.bbox_history[-5]
            prev_foot = ((prev_bbox[0] + prev_bbox[2]) / 2.0, prev_bbox[3])
            curr_foot = ((x1 + x2) / 2.0, y2)
            
            # Giả định tốc độ 30 FPS -> delta_t xấp xỉ 5 / 30 = 0.167s
            dt = 5.0 / 30.0
            dist_px = math.hypot(curr_foot[0] - prev_foot[0], curr_foot[1] - prev_foot[1])
            v_norm = dist_px / (dt * current_height)  # Đơn vị: Body Heights / second

            if v_norm >= self.v_norm_thresh:
                track.current_state = "RUNNING_ALERT"
                if track.can_trigger_alert("RUNNING", current_time, self.cooldown_run):
                    if self.alert_manager:
                        self.alert_manager.trigger_alert(
                            event_type="RUNNING",
                            track_id=track.track_id,
                            frame=frame,
                            confidence=min(1.0, v_norm / 2.5),
                            extra_info=f"Vận tốc: {v_norm:.1f} BH/s"
                        )
                return

        # ---------------------------------------------------------
        # 2. KIỂM TRA HÀNH VI TÉ NGÃ (FALLING DETECTION - 2 TIERS)
        # ---------------------------------------------------------
        if self.config_fall.get("enabled", True):
            # TIER 1: FAST RULE-BASED GATEKEEPER
            ar = calculate_aspect_ratio(current_bbox)
            
            # Tính góc trục thân theta (giữa vai và hông)
            # COCO Keypoints: 5=Left Shoulder, 6=Right Shoulder, 11=Left Hip, 12=Right Hip
            shoulder_pt = ((kpts[5, 0] + kpts[6, 0]) / 2.0, (kpts[5, 1] + kpts[6, 1]) / 2.0)
            hip_pt = ((kpts[11, 0] + kpts[12, 0]) / 2.0, (kpts[11, 1] + kpts[12, 1]) / 2.0)
            torso_angle = calculate_torso_angle(shoulder_pt, hip_pt)
            
            # Kiểm tra xem có dấu hiệu sụp đổ tư thế hay không
            is_tier1_triggered = (ar <= self.ar_max) or (torso_angle <= self.angle_max)
            
            if is_tier1_triggered:
                # TIER 2: DEEP BI-LSTM VERIFIER (Nếu có nạp model)
                if self.bilstm_model is not None and len(track.keypoints_history) >= 30:
                    # Lấy cửa sổ 30 frames gần nhất và chuẩn hóa Root-Relative
                    seq_features = []
                    hist_slice = list(track.keypoints_history)[-30:]
                    bbox_slice = list(track.bbox_history)[-30:]
                    
                    for k_frame, b_frame in zip(hist_slice, bbox_slice):
                        norm_vec = normalize_skeleton_root_relative(k_frame, b_frame)
                        seq_features.append(norm_vec)
                    
                    # Nạp vào mạng Bi-LSTM (Input shape: 1, 30, 34)
                    features_arr = np.expand_dims(np.array(seq_features, dtype=np.float32), axis=0)
                    prob_normal, prob_running, prob_fall = self.bilstm_model.predict(features_arr)
                    
                    if prob_fall >= self.bilstm_fall_thresh:
                        self._trigger_fall_alert(track, current_time, frame, prob_fall, "Tier 2 Verified")
                    else:
                        # Bác bỏ báo động giả (người cúi nhặt đồ hoặc ngồi bệt)
                        track.current_state = "NORMAL"
                else:
                    # Nếu chưa gắn Bi-LSTM model: Dùng Heuristic của Tier 1 để cảnh báo trực tiếp
                    self._trigger_fall_alert(track, current_time, frame, 0.9, f"Tier 1 (AR={ar:.2f}, θ={torso_angle:.1f}°)")

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
