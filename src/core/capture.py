import time
import sys
import threading
from queue import Queue, Full
import cv2
from typing import Optional, Union, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


class DropOldestQueue:
    """
    Hàng đợi Circular Buffer (Latest-Frame Buffer):
    Chỉ giữ tối đa N khung hình mới nhất. Khi hàng đợi đầy,
    nó chủ động loại bỏ (drop) khung hình cũ nhất để nạp khung hình mới vào,
    đảm bảo độ trễ của luồng video luôn tiệm cận 0ms.
    """
    def __init__(self, maxsize: int = 2):
        self.maxsize = maxsize
        self.queue = Queue(maxsize=maxsize)
        self.lock = threading.Lock()

    def put(self, item):
        with self.lock:
            if self.queue.full():
                try:
                    # Bỏ khung hình cũ nhất
                    self.queue.get_nowait()
                except Exception:
                    pass
            self.queue.put(item)

    def get(self, timeout: Optional[float] = 1.0):
        return self.queue.get(timeout=timeout)

    def empty(self) -> bool:
        return self.queue.empty()


class CaptureThread(threading.Thread):
    """
    Luồng thu nhận video độc lập (Producer Thread):
    Đọc luồng RTSP / Webcam / Video file ở tốc độ cao nhất của camera
    và đẩy vào DropOldestQueue để luồng AI xử lý mà không gây nghẽn buffer.
    """
    def __init__(self, source: Union[int, str], buffer_size: int = 2):
        super().__init__(daemon=True)
        self.source = source
        self.buffer = DropOldestQueue(maxsize=buffer_size)
        self.running = False
        self.cap: Optional[cv2.VideoCapture] = None
        self.fps: float = 30.0
        self.frame_width: int = 1280
        self.frame_height: int = 720

    def start_capture(self) -> bool:
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            print(f"[ERROR] Không thể kết nối tới nguồn video: {self.source}")
            return False
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
        
        self.running = True
        self.is_video_file = isinstance(self.source, str) and not str(self.source).startswith(("rtsp://", "http://", "https://"))
        self.start()
        print(f"[INFO] Khởi chạy luồng Capture thành công. Nguồn: {self.source} ({self.frame_width}x{self.frame_height} @ {self.fps:.1f} FPS)")
        return True

    def run(self):
        # Tính khoảng cách thời gian giữa 2 khung hình để video phát đúng tốc độ thực tế (1x)
        frame_interval = 1.0 / max(1.0, self.fps)
        last_frame_time = time.time()

        while self.running:
            if not self.cap or not self.cap.isOpened():
                break
                
            # Nếu là file video trên đĩa: Điều tiết nhịp đọc khớp với FPS gốc của video (chống chạy tua nhanh)
            if self.is_video_file:
                elapsed = time.time() - last_frame_time
                wait_time = frame_interval - elapsed
                if wait_time > 0:
                    time.sleep(wait_time)
                last_frame_time = time.time()

            ret, frame = self.cap.read()
            if not ret:
                # Nếu là file video hết thì tua lại từ đầu để loop test
                if self.is_video_file:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    last_frame_time = time.time()
                    continue
                else:
                    print("[WARNING] Mất tín hiệu luồng camera...")
                    time.sleep(1.0)
                    continue

            timestamp = time.time()
            self.buffer.put((frame, timestamp))

    def read(self, timeout: float = 1.0) -> Optional[Tuple[any, float]]:
        """Lấy khung hình mới nhất từ buffer cho luồng AI"""
        try:
            return self.buffer.get(timeout=timeout)
        except Exception:
            return None

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        print("[INFO] Đã dừng luồng Capture.")
