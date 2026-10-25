import os
import json
import asyncio
from typing import List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
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

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in list(self.active_connections):
            try:
                await connection.send_text(json.dumps(message))
            except Exception:
                self.disconnect(connection)

manager = ConnectionManager()

# Tham chiếu toàn cục tới pipeline và capture thread (sẽ được main.py gán vào)
global_pipeline = None
global_capture = None
global_loop = None


def broadcast_alert_sync(payload: dict):
    """Callback đồng bộ được gọi từ AlertManager chạy trên luồng nền"""
    global global_loop
    if global_loop and manager.active_connections:
        asyncio.run_coroutine_threadsafe(manager.broadcast(payload), global_loop)


@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.websocket("/ws/alerts")
async def websocket_alerts_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Giữ kết nối lắng nghe
            await websocket.receive_text()
    except WebSocketDisconnect:
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
    return {"status": "cleared"}


@app.get("/api/events")
async def get_events():
    global global_pipeline
    if global_pipeline:
        return global_pipeline.db_service.get_recent_events(limit=30)
    return []
