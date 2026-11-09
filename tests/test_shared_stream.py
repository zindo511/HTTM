import sys
import threading
import unittest
from unittest.mock import Mock, patch
from queue import Queue, Empty

import cv2
import numpy as np

from src.core.shared_stream import SharedVideoStream


class SharedStreamTests(unittest.TestCase):
    def test_real_jpeg_and_recovery_after_bad_frame(self):
        frames = Queue()
        frames.put((np.zeros((64, 64, 3), dtype=np.uint8), 0))
        frames.put((np.zeros((64, 64, 3), dtype=np.uint8), .1))
        def read(timeout):
            try:
                return frames.get(timeout=timeout)
            except Empty:
                return None
        capture = Mock(read=read)
        pipeline = Mock()
        pipeline.process_frame.side_effect = [ValueError('bad frame'), np.zeros((64, 64, 3), dtype=np.uint8)]
        stream = SharedVideoStream(lambda: capture, lambda: pipeline)
        with self.assertLogs(level='ERROR'):
            stream.start()
            try:
                with stream.condition:
                    self.assertTrue(stream.condition.wait_for(lambda: stream.sequence == 1, timeout=5))
                payload = next(stream.subscribe())
                jpeg = payload.split(b'\r\n\r\n', 1)[1][:-2]
                decoded = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
                self.assertEqual(decoded.shape, (64, 64, 3))
                self.assertEqual(pipeline.process_frame.call_count, 2)
            finally:
                stream.stop()

    def test_two_subscribers_share_one_inference(self):
        capture = Mock()
        delivered = threading.Event()
        def read(timeout):
            if not delivered.is_set():
                delivered.set()
                return object(), 1.0
            stream.stopped.wait(timeout)
            return None
        capture.read.side_effect = read
        pipeline = Mock()
        stream = SharedVideoStream(lambda: capture, lambda: pipeline)
        encoded = Mock()
        encoded.tobytes.return_value = b'jpeg'
        with patch('src.core.shared_stream.cv2.imencode', return_value=(True, encoded)):
            stream.start()
            try:
                first = next(stream.subscribe())
                second = next(stream.subscribe())
                self.assertEqual(first, second)
                pipeline.process_frame.assert_called_once()
            finally:
                stream.stop()


if __name__ == '__main__':
    unittest.main()
