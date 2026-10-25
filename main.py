import os
import sys
import yaml
import asyncio
import uvicorn

from src.core.capture import CaptureThread
from src.core.pipeline import SurveillancePipeline
import web.app as web_module


def load_config(config_path: str = "configs/config.yaml") -> dict:
    if not os.path.exists(config_path):
        print(f"[ERROR] Không tìm thấy file cấu hình: {config_path}")
        sys.exit(1)
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    print("=" * 70)
    print("   HỆ THỐNG TRINH SÁT AN NINH THỜI GIAN THỰC (AI SURVEILLANCE)")
    print("   Phát hiện 3 hành vi bất thường: Xâm nhập | Lảng vảng | Chạy / Té ngã")
    print("=" * 70)

    # 1. Tải cấu hình
    config = load_config()
    cam_cfg = config.get("camera", {})
    web_cfg = config.get("web", {})

    # 2. Khởi tạo luồng thu hình Tầng 1 (CaptureThread với Drop-Oldest Queue)
    source = cam_cfg.get("source", 0)
    buffer_size = cam_cfg.get("buffer_size", 2)
    capture_thread = CaptureThread(source=source, buffer_size=buffer_size)
    if not capture_thread.start_capture():
        print("[ERROR] Không thể khởi chạy luồng camera. Vui lòng kiểm tra lại thiết bị hoặc file video!")
        return

    # 3. Khởi tạo Nhạc trưởng điều phối Tầng 2 & 3 (SurveillancePipeline)
    pipeline = SurveillancePipeline(
        config=config,
        ws_broadcast_callback=web_module.broadcast_alert_sync
    )

    # 4. Gắn kết nối vào module Web (Tầng 4)
    web_module.global_capture = capture_thread
    web_module.global_pipeline = pipeline

    host = web_cfg.get("host", "0.0.0.0")
    port = web_cfg.get("port", 8000)

    print(f"\n[INFO] Hệ thống đã sẵn sàng! Mời bạn mở trình duyệt truy cập:")
    print(f"       👉 http://localhost:{port} (hoặc http://127.0.0.1:{port})")
    print(f"[INFO] Nhấn Ctrl+C để dừng hệ thống.\n")

    # 5. Khởi chạy Web Server FastAPI & WebSocket
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        web_module.global_loop = loop
        
        server_config = uvicorn.Config(web_module.app, host=host, port=port, loop="asyncio", log_level="warning")
        server = uvicorn.Server(server_config)
        loop.run_until_complete(server.serve())
    except KeyboardInterrupt:
        print("\n[INFO] Đang đóng hệ thống...")
    finally:
        capture_thread.stop()
        print("[INFO] Hệ thống đã tắt an toàn.")


if __name__ == "__main__":
    main()
