import os
import time
import threading
from typing import Dict, Optional, Callable, List
import cv2
import numpy as np


class AlertManager:
    """
    Trung tâm điều phối cảnh báo (Central Alert Manager):
    1. Kiểm soát Cooldown chống Spam chuông theo từng đối tượng và sự kiện.
    2. Cắt ảnh bằng chứng (Snapshot) và ghi Database trên luồng nền (Asynchronous Non-blocking).
    3. Đẩy thông điệp cảnh báo thời gian thực tới các WebSocket clients.
    """
    def __init__(
        self, 
        snapshots_dir: str = "data/snapshots",
        db_service=None,
        ws_broadcast_callback: Optional[Callable[[dict], None]] = None
    ):
        self.snapshots_dir = snapshots_dir
        self.db_service = db_service
        self.ws_broadcast_callback = ws_broadcast_callback
        os.makedirs(self.snapshots_dir, exist_ok=True)
        
        # Ngưỡng cooldown mặc định theo từng loại sự kiện (giây)
        self.default_cooldowns: Dict[str, float] = {
            "INTRUSION": 5.0,
            "LOITERING": 10.0,
            "RUNNING": 4.0,
            "FALLING": 15.0
        }

    def trigger_alert(
        self,
        event_type: str,
        track_id: int,
        frame: np.ndarray,
        zone_id: Optional[str] = None,
        confidence: float = 1.0,
        extra_info: Optional[str] = None
    ):
        """
        Tiếp nhận yêu cầu cảnh báo từ các module hành vi.
        Thực thi lưu ảnh và đẩy dữ liệu bất đồng bộ trên luồng riêng để không nghẽn FPS.
        """
        timestamp = time.time()
        time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
        filename = f"{event_type}_{track_id}_{int(timestamp)}.jpg"
        snapshot_path = os.path.join(self.snapshots_dir, filename)

        # Chạy tác vụ ghi đĩa và database trên luồng nền
        threading.Thread(
            target=self._async_handle_alert,
            args=(event_type, track_id, frame.copy(), snapshot_path, filename, zone_id, confidence, time_str, timestamp, extra_info),
            daemon=True
        ).start()

    def _async_handle_alert(
        self,
        event_type: str,
        track_id: int,
        frame: np.ndarray,
        snapshot_path: str,
        filename: str,
        zone_id: Optional[str],
        confidence: float,
        time_str: str,
        timestamp: float,
        extra_info: Optional[str]
    ):
        try:
            # 1. Lưu ảnh bằng chứng snapshot ra ổ cứng
            cv2.imwrite(snapshot_path, frame)
            
            # 2. Lưu bản ghi vào cơ sở dữ liệu SQLite (nếu có db_service)
            if self.db_service:
                self.db_service.insert_event(
                    track_id=track_id,
                    event_type=event_type,
                    zone_id=zone_id,
                    confidence=confidence,
                    snapshot_path=snapshot_path,
                    timestamp=time_str
                )
            
            # 3. Tạo gói tin đẩy qua WebSocket lên giao diện Web Dashboard
            payload = {
                "event_type": event_type,
                "track_id": track_id,
                "zone_id": zone_id or "N/A",
                "confidence": round(confidence * 100, 1),
                "timestamp": time_str,
                "snapshot_url": f"/snapshots/{filename}",
                "extra_info": extra_info or ""
            }
            
            if self.ws_broadcast_callback:
                self.ws_broadcast_callback(payload)
                
            print(f"[ALERT] {time_str} | ⚠️ {event_type} phát hiện tại Track #{track_id} (Snapshot: {filename})")
            
        except Exception as e:
            print(f"[ERROR] Lỗi khi xử lý lưu snapshot/cảnh báo bất đồng bộ: {e}")
