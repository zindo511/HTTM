import time
import unittest
from unittest.mock import Mock, patch

import cv2

from src.core.capture import CaptureThread


class SampledReplayTests(unittest.TestCase):
    def test_slow_reader_gets_all_selected_frames_once_with_media_timestamps(self):
        video = Mock()
        video.isOpened.return_value = True
        video.get.side_effect = lambda prop: 120 if prop == cv2.CAP_PROP_FPS else 64
        # grab() and read() advance the same stream; skipped frames are only grabbed.
        stream = iter([(True, frame) for frame in range(8)] + [(False, None)] * 2)
        video.read.side_effect = lambda: next(stream)
        video.grab.side_effect = lambda: next(stream)[0]

        with patch("src.core.capture.cv2.VideoCapture", return_value=video):
            capture = CaptureThread("test.mp4", buffer_size=1, file_sample_fps=60)
            self.assertTrue(capture.start_capture())
            try:
                # Let the second sample reach the full queue before consuming it.
                time.sleep(0.05)
                samples = [capture.read(timeout=1) for _ in range(4)]
                self.assertEqual(samples, [(index, index / 120) for index in (0, 2, 4, 6)])
                capture.join(timeout=1)
                self.assertFalse(capture.is_alive())
                self.assertIsNone(capture.read(timeout=0.01))
                video.set.assert_not_called()
            finally:
                capture.stop()
