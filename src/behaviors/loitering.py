from typing import Dict, Any, List
import numpy as np
from src.behaviors.base import BaseBehaviorDetector
from src.tracking.track_manager import TrackHistory
from src.utils.geometry import is_point_in_polygon, calculate_trajectory_displacement


class LoiteringDetector(BaseBehaviorDetector):
    """
    Module 2: Phát hiện hành vi Lảng vảng (Loitering).
    Thỏa mãn 2 điều kiện:
    1. Thời gian lưu trú (Dwell time) >= threshold_seconds (mặc định 10s).
    2. Bán kính dịch chuyển quỹ đạo (Displacement) <= max_radius_pixels.
    Bảo lưu bộ đếm thời gian thông qua vùng đệm gia hạn Grace Period 2.0s chống rung biên và che khuất.
    """
    def __init__(self, config: Dict[str, Any], alert_manager=None):
        super().__init__(config, alert_manager)
        self.threshold_seconds = self.config.get("threshold_seconds", 10.0)
        self.max_radius = self.config.get("max_radius_pixels", 60.0)
        self.cooldown = self.config.get("cooldown_seconds", 10.0)

    def process(
        self, 
        track: TrackHistory, 
        current_time: float, 
        frame: np.ndarray, 
        zones: List[Dict[str, Any]]
    ):
        if not self.enabled or len(track.foot_positions) == 0:
            return

        current_foot = track.foot_positions[-1]
        is_inside_loiter_zone = False

        for zone in zones:
            if zone.get("type") != "LOITERING_ZONE":
                continue

            polygon = np.array(zone.get("polygon", []), dtype=np.int32)
            if is_point_in_polygon(current_foot, polygon):
                is_inside_loiter_zone = True
                track.last_seen_inside_zone = current_time
                track.current_zone_id = zone.get("id")

                # Bắt đầu bấm giờ lưu trú nếu mới bước vào
                if track.zone_entry_time is None:
                    track.zone_entry_time = current_time

                # Tính tổng thời gian lưu trú trong vùng
                dwell_time = current_time - track.zone_entry_time

                if dwell_time >= self.threshold_seconds:
                    # Kiểm tra bán kính dịch chuyển trong cửa sổ gần nhất
                    displacement = calculate_trajectory_displacement(list(track.foot_positions))
                    if displacement <= self.max_radius:
                        track.current_state = "LOITERING_ALERT"
                        
                        if track.can_trigger_alert("LOITERING", current_time, self.cooldown):
                            if self.alert_manager:
                                self.alert_manager.trigger_alert(
                                    event_type="LOITERING",
                                    track_id=track.track_id,
                                    frame=frame,
                                    zone_id=zone.get("id"),
                                    confidence=1.0,
                                    extra_info=f"Lảng vảng {dwell_time:.1f}s tại: {zone.get('name')}"
                                )
                break

        # Nếu chân ra ngoài vùng giám sát: áp dụng cơ chế Grace Period
        if not is_inside_loiter_zone and track.zone_entry_time is not None:
            time_since_last_seen = current_time - track.last_seen_inside_zone
            # Chỉ reset bộ đếm khi đối tượng thực sự rời đi vượt quá Grace Period (2.0s)
            if time_since_last_seen > track.exit_grace_period:
                track.zone_entry_time = None
                track.current_zone_id = None
                if track.current_state == "LOITERING_ALERT":
                    track.current_state = "NORMAL"
