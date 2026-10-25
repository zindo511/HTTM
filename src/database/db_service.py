import sqlite3
import os
import json
from typing import List, Dict, Optional, Any


class DatabaseService:
    """
    Quản lý lưu trữ cơ sở dữ liệu SQLite cho hệ thống trinh sát.
    Bao gồm 3 bảng chính: ZONES, TRACKS, EVENTS.
    """
    def __init__(self, db_path: str = "data/surveillance.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _get_connection(self):
        return sqlite3.connect(self.db_path, timeout=10.0)

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # 1. Bảng ZONES (Vùng cấm & vùng giám sát)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS zones (
                zone_id TEXT PRIMARY KEY,
                zone_name TEXT NOT NULL,
                zone_type TEXT NOT NULL,
                polygon_points TEXT NOT NULL,
                loiter_threshold REAL DEFAULT 10.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)

            # 2. Bảng TRACKS (Đối tượng theo dõi)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS tracks (
                track_id INTEGER PRIMARY KEY,
                first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                final_status TEXT DEFAULT 'NORMAL'
            )
            """)

            # 3. Bảng EVENTS (Nhật ký cảnh báo sự kiện vi phạm)
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
            conn.commit()

    def insert_event(
        self, 
        track_id: int, 
        event_type: str, 
        zone_id: Optional[str], 
        confidence: float, 
        snapshot_path: str, 
        timestamp: str
    ) -> int:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO events (track_id, zone_id, event_type, confidence, snapshot_path, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (track_id, zone_id, event_type, confidence, snapshot_path, timestamp))
            conn.commit()
            return cursor.lastrowid

    def get_recent_events(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM events ORDER BY event_id DESC LIMIT ?", (limit,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def save_zone(self, zone_id: str, zone_name: str, zone_type: str, polygon_points: list, loiter_threshold: float = 10.0):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT OR REPLACE INTO zones (zone_id, zone_name, zone_type, polygon_points, loiter_threshold)
            VALUES (?, ?, ?, ?, ?)
            """, (zone_id, zone_name, zone_type, json.dumps(polygon_points), loiter_threshold))
            conn.commit()

    def load_zones(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM zones")
            rows = cursor.fetchall()
            result = []
            for row in rows:
                d = dict(row)
                d["polygon_points"] = json.loads(d["polygon_points"])
                result.append(d)
            return result
