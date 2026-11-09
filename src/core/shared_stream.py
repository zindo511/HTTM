"""One inference producer; subscribers only consume the latest encoded frame."""
import logging
import threading

import cv2


class SharedVideoStream:
    def __init__(self, capture_provider, pipeline_provider):
        self.capture_provider = capture_provider
        self.pipeline_provider = pipeline_provider
        self.condition = threading.Condition()
        self.stopped = threading.Event()
        self.sequence = 0
        self.jpeg = None
        self.thread = None

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.stopped.clear()
        self.thread = threading.Thread(target=self._run, daemon=True, name="shared-ai")
        self.thread.start()

    def stop(self):
        self.stopped.set()
        with self.condition:
            self.condition.notify_all()
        if self.thread:
            self.thread.join(timeout=5)

    def _run(self):
        while not self.stopped.is_set():
            capture = self.capture_provider()
            pipeline = self.pipeline_provider()
            if capture is None or pipeline is None:
                self.stopped.wait(0.1)
                continue
            item = capture.read(timeout=0.5)
            if item is None or capture is not self.capture_provider():
                continue
            try:
                frame = pipeline.process_frame(*item)
                ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    with self.condition:
                        self.jpeg = jpeg.tobytes()
                        self.sequence += 1
                        self.condition.notify_all()
            except Exception:
                logging.exception("Shared AI frame processing failed")
                self.stopped.wait(0.05)

    def subscribe(self):
        sequence = -1
        while not self.stopped.is_set():
            with self.condition:
                self.condition.wait_for(
                    lambda: self.stopped.is_set() or (
                        self.jpeg is not None and self.sequence != sequence), timeout=1
                )
                if self.stopped.is_set():
                    return
                if self.jpeg is None or self.sequence == sequence:
                    continue
                sequence, jpeg = self.sequence, self.jpeg
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
