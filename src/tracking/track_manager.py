import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class TrackHistory:
    """
    Quản lý vòng đời và lưu trữ lịch sử chuyển động của một đối tượng track_id.
    """
    track_id: int
    first_seen: float
    last_seen: float
    foot_positions: deque = field(default_factory=lambda: deque(maxlen=30))
    # 90 mẫu (~1.5s ở 60 FPS) để cửa sổ vận tốc chạy 0.5s luôn đủ dữ liệu khi FPS cao.
    bbox_history: deque = field(default_factory=lambda: deque(maxlen=90))
    keypoints_history: deque = field(default_factory=lambda: deque(maxlen=30))

    timestamp_history: deque = field(default_factory=lambda: deque(maxlen=90))
    keypoint_bbox_history: deque = field(default_factory=lambda: deque(maxlen=30))
    keypoint_observed_history: deque = field(default_factory=lambda: deque(maxlen=30))
    missing_kpts_count: int = 0
    last_valid_kpts_time: Optional[float] = None
    fall_phase: str = "NORMAL"
    fall_impact_time: Optional[float] = None
    fall_lying_since: Optional[float] = None
    fall_recovery_since: Optional[float] = None
    fall_last_time: Optional[float] = None
    fall_reference_height: Optional[float] = None
    fall_height_samples: deque = field(default_factory=lambda: deque(maxlen=180))
    fall_motion_samples: deque = field(default_factory=lambda: deque(maxlen=120))

    current_state: str = "NORMAL"
    current_zone_id: Optional[str] = None
    zone_entry_time: Optional[float] = None
    last_seen_inside_zone: float = 0.0
    exit_grace_period: float = 2.0  # Vùng đệm 2 giây chống rung biên và che khuất
    
    # Bộ đếm khung hình liên tục bên trong vùng cấm (chống nhiễu rung lắc)
    consecutive_inside_frames: int = 0
    
    # Ma trận lưu mốc thời gian cảnh báo gần nhất để chống spam: {"LOITERING": t1, "RUNNING": t2}
    last_alert_time: Dict[str, float] = field(default_factory=dict)

    def can_trigger_alert(self, event_type: str, current_time: float, cooldown: float) -> bool:
        """
        Kiểm tra cơ chế Cooldown chống Spam chuông báo động theo từng sự kiện.
        """
        last_time = self.last_alert_time.get(event_type)
        if last_time is None or (current_time - last_time) >= cooldown:
            self.last_alert_time[event_type] = current_time
            return True
        return False


class TrackManager:
    """
    Quản lý tập trung toàn bộ các đối tượng đang theo dõi trong RAM.
    Có cơ chế dọn rác TTL (Time-to-Live) để tránh rò rỉ bộ nhớ khi người rời khỏi khung nhìn.
    """
    def __init__(self, track_ttl: float = 5.0, exit_grace_period: float = 2.0,
                 max_missing_kpts: int = 5, max_fill_seconds: float = 0.25):
        self.track_ttl = track_ttl
        self.max_missing_kpts = max_missing_kpts
        self.max_fill_seconds = max_fill_seconds
        self.exit_grace_period = exit_grace_period
        self.tracks: Dict[int, TrackHistory] = {}

    def get_or_create(self, track_id: int, current_time: float) -> TrackHistory:
        if track_id not in self.tracks:
            self.tracks[track_id] = TrackHistory(
                track_id=track_id,
                first_seen=current_time,
                last_seen=current_time,
                last_seen_inside_zone=current_time,
                exit_grace_period=self.exit_grace_period
            )
        return self.tracks[track_id]

    def update_track(
        self, 
        track_id: int, 
        bbox: Tuple[float, float, float, float], 
        foot_point: Tuple[float, float],
        keypoints: Optional[np.ndarray], 
        current_time: float
    ) -> TrackHistory:
        track = self.get_or_create(track_id, current_time)
        track.last_seen = current_time
        track.bbox_history.append(bbox)
        track.timestamp_history.append(current_time)
        track.foot_positions.append(foot_point)
        valid = (keypoints is not None and keypoints.ndim == 2
                 and keypoints.shape[0] == 17 and keypoints.shape[1] in (2, 3)
                 and np.isfinite(keypoints).all()
                 and np.any(np.any(keypoints[:, :2] != 0, axis=1)
                            & (keypoints[:, 2] >= 0.3 if keypoints.shape[1] == 3 else True)))
        if (track.last_valid_kpts_time is not None
                and current_time - track.last_valid_kpts_time > self.max_fill_seconds):
            track.keypoints_history.clear()
            track.keypoint_bbox_history.clear()
            track.keypoint_observed_history.clear()
        if valid:
            track.missing_kpts_count = 0
            track.last_valid_kpts_time = current_time
            track.keypoints_history.append(keypoints.copy())
            track.keypoint_bbox_history.append(bbox)
            track.keypoint_observed_history.append(True)
        else:
            track.missing_kpts_count += 1
            if track.missing_kpts_count <= self.max_missing_kpts and track.keypoints_history:
                track.keypoints_history.append(track.keypoints_history[-1].copy())
                # Keep the original coordinate scale paired with the filled pose.
                track.keypoint_bbox_history.append(track.keypoint_bbox_history[-1])
                track.keypoint_observed_history.append(False)
            else:
                track.keypoints_history.clear()
                track.keypoint_bbox_history.clear()
                track.keypoint_observed_history.clear()
        return track

    def purge_stale_tracks(self, current_time: float) -> List[int]:
        """
        Xóa bỏ các track đã mất dấu quá thời gian TTL (ví dụ 5 giây không thấy lại).
        """
        stale_ids = []
        for tid, track in list(self.tracks.items()):
            if (current_time - track.last_seen) > self.track_ttl:
                stale_ids.append(tid)
                del self.tracks[tid]
        return stale_ids
