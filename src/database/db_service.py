import sqlite3
import os
import struct
import threading
from typing import List, Dict, Optional, Any


class DatabaseService:
    """
    Quản lý lưu trữ cơ sở dữ liệu SQLite cho hệ thống trinh sát.
    Bao gồm 2 bảng: TRACKS, EVENTS.
    """
    def __init__(self, db_path: str = "data/surveillance.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

        # Bộ đếm sự kiện giữ trên RAM để HUD đọc mỗi frame mà không truy vấn SQLite
        self._count_lock = threading.Lock()
        self._event_counts: Dict[str, int] = self._load_event_counts()

    def _get_connection(self):
        return sqlite3.connect(self.db_path, timeout=10.0)

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # 1. Bảng TRACKS (Đối tượng theo dõi)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS tracks (
                track_id INTEGER PRIMARY KEY,
                first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                final_status TEXT DEFAULT 'NORMAL'
            )
            """)

            # 2. Bảng EVENTS (Nhật ký cảnh báo sự kiện vi phạm)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                track_id INTEGER,
                zone_id TEXT,
                event_type TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                snapshot_path TEXT,
                timestamp TEXT NOT NULL,
                resolution_status TEXT DEFAULT 'PENDING'
            )
            """)
            # CSDL cũ chưa có cột mô tả sự kiện -> bổ sung tại chỗ
            columns = {row[1] for row in cursor.execute("PRAGMA table_info(events)")}
            if "extra_info" not in columns:
                cursor.execute("ALTER TABLE events ADD COLUMN extra_info TEXT")
            conn.commit()

    def insert_event(
        self,
        track_id: int,
        event_type: str,
        zone_id: Optional[str],
        confidence: float,
        snapshot_path: str,
        timestamp: str,
        extra_info: Optional[str] = None
    ) -> int:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO events (track_id, zone_id, event_type, confidence, snapshot_path, timestamp, extra_info)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (track_id, zone_id, event_type, float(confidence), snapshot_path, timestamp, extra_info))
            conn.commit()
            event_id = cursor.lastrowid

        with self._count_lock:
            self._event_counts[event_type] = self._event_counts.get(event_type, 0) + 1
        return event_id

    def _load_event_counts(self) -> Dict[str, int]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT event_type, COUNT(*) FROM events GROUP BY event_type")
            return {event_type: count for event_type, count in cursor.fetchall()}

    def count_total_events(self) -> int:
        """Tổng số cảnh báo đã ghi nhận (dùng cho HUD trên video)"""
        with self._count_lock:
            return sum(self._event_counts.values())

    def count_events_by_type(self) -> Dict[str, int]:
        """Số cảnh báo theo từng loại hành vi (dùng cho thanh footer trên web)"""
        with self._count_lock:
            counts = dict(self._event_counts)
        for event_type in ("RUNNING", "FALLING", "FIGHTING"):
            counts.setdefault(event_type, 0)
        return counts

    def get_recent_events(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM events ORDER BY event_id DESC LIMIT ?", (limit,))
            rows = cursor.fetchall()
            events = []
            for row in rows:
                d = dict(row)
                # Bản ghi cũ lưu numpy.float32 thành BLOB 4 byte -> giải mã lại về float
                if isinstance(d["confidence"], bytes):
                    d["confidence"] = struct.unpack("<f", d["confidence"])[0] if len(d["confidence"]) == 4 else 0.0
                events.append(d)
            return events

    def delete_event(self, event_id: int) -> bool:
        """Xoá 1 cảnh báo theo ID và đồng bộ lại bộ đếm."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM events WHERE event_id = ?", (int(event_id),))
            deleted = cursor.rowcount > 0
            conn.commit()
        if deleted:
            with self._count_lock:
                self._event_counts = self._load_event_counts()
        return deleted

    def clear_events(self, event_type: Optional[str] = None) -> int:
        """Xoá toàn bộ cảnh báo hoặc xoá theo loại hành vi và đồng bộ lại bộ đếm."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if event_type and event_type != "ALL":
                cursor.execute("DELETE FROM events WHERE event_type = ?", (event_type,))
            else:
                cursor.execute("DELETE FROM events")
            deleted = cursor.rowcount
            conn.commit()
        with self._count_lock:
            self._event_counts = self._load_event_counts()
        return deleted

