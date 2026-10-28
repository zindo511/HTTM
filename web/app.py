import os
import json
import asyncio
import shutil
from typing import List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import cv2

app = FastAPI(title="AI Surveillance System")

# Mount các thư mục tài nguyên tĩnh và ảnh snapshot
os.makedirs("data/snapshots", exist_ok=True)
os.makedirs("web/static", exist_ok=True)

app.mount("/static", StaticFiles(directory="web/static"), name="static")
app.mount("/snapshots", StaticFiles(directory="data/snapshots"), name="snapshots")

templates = Jinja2Templates(directory="web/templates")

# Quản lý các kết nối WebSocket đang hoạt động
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"[WS] Dashboard kết nối. Tổng: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        # Chuyển dữ liệu thành JSON trước khi gửi.
        # Nếu lỗi, callback bên dưới sẽ ghi rõ nguyên nhân.
        text = json.dumps(message, ensure_ascii=False)

        for connection in list(self.active_connections):
            try:
                await connection.send_text(text)
            except Exception as exc:
                print(f"[WS ERROR] Gửi tới dashboard thất bại: {exc}")
                self.disconnect(connection)

manager = ConnectionManager()

# Tham chiếu toàn cục tới pipeline và capture thread (sẽ được main.py gán vào)
global_pipeline = None
global_capture = None
global_loop = None


def broadcast_alert_sync(payload: dict):
    """Nhận cảnh báo từ luồng nền và chuyển sang event loop của server."""
    loop = global_loop

    if loop is None or loop.is_closed() or not loop.is_running():
        print("[WS WARNING] Event loop chưa hoạt động, chưa gửi được cảnh báo.")
        return

    future = asyncio.run_coroutine_threadsafe(
        manager.broadcast(payload),
        loop
    )

    def report_result(done):
        try:
            done.result()
        except Exception as exc:
            print(f"[WS ERROR] Không gửi được cảnh báo: {exc}")

    future.add_done_callback(report_result)


def _db_zone_to_pipeline_zone(row: dict) -> dict:
    """Chuyển bản ghi bảng zones (SQLite) về định dạng zone dùng trong pipeline"""
    return {
        "id": row["zone_id"],
        "name": row["zone_name"],
        "type": row["zone_type"],
        "polygon": row["polygon_points"],
    }


@app.on_event("startup")
async def load_zones_from_db():
    global global_loop
    global_loop = asyncio.get_running_loop()
    print("[WS] Đã gắn event loop của server.")

    if not global_pipeline:
        return

    existing_ids = {z.get("id") for z in global_pipeline.zones}

    for row in global_pipeline.db_service.load_zones():
        if row["zone_id"] not in existing_ids:
            global_pipeline.zones.append(_db_zone_to_pipeline_zone(row))
            existing_ids.add(row["zone_id"])


@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.websocket("/ws/alerts")
async def websocket_alerts_endpoint(websocket: WebSocket):
    await manager.connect(websocket)

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        print("[WS] Dashboard ngắt kết nối.")
    except Exception as exc:
        print(f"[WS ERROR] Kết nối dashboard: {exc}")
    finally:
        manager.disconnect(websocket)


def generate_mjpeg_stream():
    """Hàm sinh luồng MJPEG thời gian thực từ camera pipeline"""
    global global_capture, global_pipeline
    while True:
        if not global_capture or not global_pipeline:
            continue

        item = global_capture.read(timeout=1.0)
        if item is None:
            continue

        frame, timestamp = item
        # Đưa frame qua toàn bộ pipeline phân tích AI
        annotated_frame = global_pipeline.process_frame(frame, timestamp)

        # Nén JPEG để truyền qua HTTP
        ret, jpeg = cv2.imencode(".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")


@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(
        generate_mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.post("/api/zones")
async def add_zone(request: Request):
    global global_pipeline
    data = await request.json()
    if global_pipeline:
        global_pipeline.zones.append(data)
        # Lưu vào SQLite
        global_pipeline.db_service.save_zone(
            zone_id=data.get("id"),
            zone_name=data.get("name"),
            zone_type=data.get("type", "RESTRICTED_INTRUSION"),
            polygon_points=data.get("polygon", [])
        )
    return {"status": "success", "zones": global_pipeline.zones if global_pipeline else []}


@app.post("/api/zones/clear")
async def clear_zones():
    global global_pipeline
    if global_pipeline:
        global_pipeline.zones = []
        global_pipeline.db_service.clear_all_zones()
    return {"status": "cleared"}


@app.get("/api/events")
async def get_events():
    global global_pipeline
    if not global_pipeline:
        return []
    # Trả về cùng định dạng với gói tin WebSocket /ws/alerts để frontend dùng chung
    events = []
    for row in global_pipeline.db_service.get_recent_events(limit=30):
        snapshot_name = os.path.basename((row.get("snapshot_path") or "").replace("\\", "/"))
        events.append({
            "event_type": row["event_type"],
            "track_id": row["track_id"],
            "zone_id": row["zone_id"] or "N/A",
            "confidence": round((row["confidence"] or 0) * 100, 1),
            "timestamp": row["timestamp"],
            "snapshot_url": f"/snapshots/{snapshot_name}",
            "extra_info": "",
        })
    return events


@app.get("/api/events/stats")
async def get_event_stats():
    global global_pipeline
    if not global_pipeline:
        return {"total": 0, "by_type": {}}
    by_type = global_pipeline.db_service.count_events_by_type()
    return {"total": sum(by_type.values()), "by_type": by_type}


@app.post("/api/change_source")
async def change_source(request: Request):
    """Đổi nguồn video ngay tức thì mà không cần khởi động lại server"""
    global global_capture
    data = await request.json()
    new_source = data.get("source")
    if new_source is not None:
        if str(new_source).isdigit():
            new_source = int(new_source)
        if global_capture:
            global_capture.stop()
            from src.core.capture import CaptureThread
            global_capture = CaptureThread(source=new_source, buffer_size=2)
            global_capture.start_capture()
        return {"status": "success", "new_source": str(new_source)}
    return {"status": "error", "message": "Source missing"}


@app.post("/api/upload_video")
async def upload_video(file: UploadFile = File(...)):
    """Tải file video MP4 từ trình duyệt lên và phát tự động"""
    global global_capture
    dest_path = os.path.join("data/test_videos", file.filename)
    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    if global_capture:
        global_capture.stop()
        from src.core.capture import CaptureThread
        global_capture = CaptureThread(source=dest_path, buffer_size=2)
        global_capture.start_capture()
    return {"status": "success", "video_path": dest_path}

