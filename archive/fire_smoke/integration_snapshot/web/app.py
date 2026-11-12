import os
import json
import asyncio
import shutil
from typing import List
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from src.core.shared_stream import SharedVideoStream

app = FastAPI(title="AI Surveillance System")

# Mount các thư mục tài nguyên tĩnh và ảnh snapshot
os.makedirs("data/snapshots", exist_ok=True)
os.makedirs("web/static", exist_ok=True)

app.mount("/static", StaticFiles(directory="web/static"), name="static")
app.mount("/snapshots", StaticFiles(directory="data/snapshots"), name="snapshots")

templates = Jinja2Templates(directory="web/templates")
DEMO_CLIPS = (
    ("Đi bộ", "data/test_videos/people_walking.mp4"),
    ("Chạy nhanh", "data/test_videos/public/sphar/running__casia_angleview_p01_run_a1.mp4"),
    ("Té ngã", "data/test_videos/fall_sample.mp4"),
    ("Đánh nhau", "data/test_videos/fight1.mp4"),
    ("Khói & Lửa", "data/test_videos/khoi.mp4"),
)

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
shared_stream = SharedVideoStream(lambda: global_capture, lambda: global_pipeline)


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


@app.on_event("startup")
async def start_shared_stream():
    global global_loop
    global_loop = asyncio.get_running_loop()
    print("[WS] Đã gắn event loop của server.")

    if not global_pipeline:
        return

    shared_stream.start()


@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={
        "demo_clips": [(name, path) for name, path in DEMO_CLIPS if Path(path).is_file()],
        "current_source": str(global_capture.source) if global_capture is not None else None,
        "fire_smoke": _fire_smoke_status(),
    })


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


@app.on_event("shutdown")
async def stop_shared_stream():
    shared_stream.stop()
    if global_capture:
        global_capture.stop()


def generate_mjpeg_stream():
    return shared_stream.subscribe()


@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(
        generate_mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


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
            "extra_info": row.get("extra_info") or "",
        })
    return events


@app.get("/api/events/stats")
async def get_event_stats():
    global global_pipeline
    if not global_pipeline:
        return {"total": 0, "by_type": {}}
    by_type = global_pipeline.db_service.count_events_by_type()
    return {"total": sum(by_type.values()), "by_type": by_type}


def _fire_smoke_status() -> dict:
    available = global_pipeline is not None and Path(global_pipeline.fire_model_path).is_file()
    return {
        "enabled": bool(global_pipeline is not None and global_pipeline.fire_smoke_enabled),
        "available": available,
    }


@app.get("/api/detectors/fire_smoke")
async def get_fire_smoke():
    return _fire_smoke_status()


@app.post("/api/detectors/fire_smoke")
def set_fire_smoke(payload: dict):
    """Bật/tắt mô hình Khói & Lửa (def thường: nạp mô hình chạy ở threadpool, không chặn server)."""
    if global_pipeline is None:
        return _error("Pipeline chưa chạy", 503)
    enabled = payload.get("enabled")
    if not isinstance(enabled, bool):
        return _error("Thiếu trường enabled (true/false)")
    if enabled and not Path(global_pipeline.fire_model_path).is_file():
        return _error(f"Chưa có mô hình {global_pipeline.fire_model_path}", 404)
    global_pipeline.set_fire_smoke_enabled(enabled)
    return {"status": "success", **_fire_smoke_status()}


UPLOAD_DIR = "data/test_videos"
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def _error(message: str, status_code: int = 400) -> JSONResponse:
    return JSONResponse({"status": "error", "message": message}, status_code=status_code)


def _switch_source(new_source) -> bool:
    """Mở nguồn mới trước; chỉ thay nguồn cũ khi mở thành công để stream không bị chết."""
    global global_capture
    from src.core.capture import CaptureThread

    old_capture = global_capture
    same_device = (old_capture is not None and isinstance(new_source, int)
                   and old_capture.source == new_source)
    if same_device:
        # Webcam không mở được 2 lần cùng lúc: buộc phải đóng trước.
        old_capture.stop()
    camera = global_pipeline.config.get("camera", {}) if global_pipeline else {}
    new_capture = CaptureThread(source=new_source, buffer_size=camera.get("buffer_size", 2),
                                file_sample_fps=camera.get("file_sample_fps"),
                                max_frame_side=camera.get("max_frame_side"))
    if not new_capture.start_capture():
        if same_device:
            global_capture = None
        return False
    global_capture = new_capture
    if old_capture is not None and not same_device:
        old_capture.stop()
    return True


@app.post("/api/change_source")
async def change_source(request: Request):
    """Đổi nguồn video ngay tức thì mà không cần khởi động lại server"""
    data = await request.json()
    new_source = data.get("source")
    if new_source is None or str(new_source).strip() == "":
        return _error("Thiếu nguồn video")
    if str(new_source).isdigit():
        new_source = int(new_source)
    if not _switch_source(new_source):
        return _error(f"Không mở được nguồn video: {new_source}")
    return {"status": "success", "new_source": str(new_source)}


@app.post("/api/upload_video")
async def upload_video(file: UploadFile = File(...)):
    """Tải file video từ trình duyệt lên và phát tự động"""
    # Chỉ giữ tên file, bỏ mọi thành phần đường dẫn do client gửi lên.
    filename = os.path.basename((file.filename or "").replace("\\", "/"))
    if not filename or os.path.splitext(filename)[1].lower() not in ALLOWED_VIDEO_EXTENSIONS:
        return _error("Chỉ nhận file video: " + ", ".join(sorted(ALLOWED_VIDEO_EXTENSIONS)))

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    dest_path = os.path.join(UPLOAD_DIR, filename).replace("\\", "/")
    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    if not _switch_source(dest_path):
        return _error("Đã lưu file nhưng không mở được video này")
    return {"status": "success", "video_path": dest_path}

