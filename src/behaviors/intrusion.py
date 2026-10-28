from typing import Dict, Any, List
import numpy as np
from src.behaviors.base import BaseBehaviorDetector
from src.tracking.track_manager import TrackHistory
from src.utils.geometry import is_point_in_polygon


class IntrusionDetector(BaseBehaviorDetector):
    """
    Module 1: Phát hiện hành vi Xâm nhập vùng cấm đa giác ảo (Restricted Polygon ROI).
    Áp dụng bộ lọc chống rung lắc k = 5 khung hình liên tiếp trước khi phát chuông.
    """
    def __init__(self, config: Dict[str, Any], alert_manager=None):
        super().__init__(config, alert_manager)
        self.min_confirm_frames = self.config.get("min_confirm_frames", 5)
        self.cooldown = self.config.get("cooldown_seconds", 5.0)

    def process(
        self, 
        track: TrackHistory, 
        current_time: float, 
        frame: np.ndarray, 
        zones: List[Dict[str, Any]]
    ):
        if not self.enabled or len(track.foot_positions) == 0:
            if track.current_state == "INTRUSION_ALERT":
                track.current_state = "NORMAL"
            track.consecutive_inside_frames = 0
            return

        current_foot = track.foot_positions[-1]

        for zone in zones:
            if zone.get("type") != "RESTRICTED_INTRUSION":
                continue

            polygon = np.array(zone.get("polygon", []), dtype=np.int32)
            if is_point_in_polygon(current_foot, polygon):
                # Tích lũy số khung hình liên tiếp nằm trong vùng cấm
                track.consecutive_inside_frames += 1
                
                # Chỉ kích hoạt cảnh báo khi đã vượt qua bộ lọc chống nhiễu k khung hình
                if track.consecutive_inside_frames >= self.min_confirm_frames:
                    track.current_state = "INTRUSION_ALERT"
                    
                    if track.can_trigger_alert("INTRUSION", current_time, self.cooldown):
                        if self.alert_manager:
                            self.alert_manager.trigger_alert(
                                event_type="INTRUSION",
                                track_id=track.track_id,
                                frame=frame,
                                zone_id=zone.get("id"),
                                confidence=1.0,
                                extra_info=f"Xâm nhập vào vùng: {zone.get('name')}"
                            )
                return

        # Nếu không nằm trong bất kỳ vùng cấm nào
        track.consecutive_inside_frames = 0

        if track.current_state == "INTRUSION_ALERT":
            track.current_state = "NORMAL"
