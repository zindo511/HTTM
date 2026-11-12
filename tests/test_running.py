import unittest
from unittest.mock import Mock

from src.behaviors.fall_run import MotionAndFallDetector
from src.tracking.track_manager import TrackManager


class RunningTests(unittest.TestCase):
    def setUp(self):
        self.manager = TrackManager()
        self.alerts = Mock()
        self.detector = MotionAndFallDetector({}, {"enabled": False}, alert_manager=self.alerts)

    def step(self, t, speed=3.0, x=None):
        x = speed * 200 * t if x is None else x
        track = self.manager.update_track(1, (x, 100, x + 60, 300), (x, 300), None, t)
        self.detector.process(track, t, None)
        return track

    def test_speed_and_confirmation_at_multiple_fps(self):
        for fps in (10, 15, 30, 60, 120):
            with self.subTest(fps=fps):
                self.setUp()
                first = None
                for i in range(fps * 2):
                    t = i / fps
                    track = self.step(t)
                    if track.current_state == "RUNNING_ALERT" and first is None:
                        first = t
                self.assertAlmostEqual(self.detector._compute_run_speed(track), 3.0)
                self.assertIsNotNone(first)
                self.assertGreaterEqual(first + 1e-9, .65)
                self.assertLessEqual(first, .65 + 2 / fps)
                self.assertEqual(self.alerts.trigger_alert.call_count, 1)

    def test_walking_does_not_alert(self):
        for i in range(120):
            self.step(i / 30, speed=.7)
        self.alerts.trigger_alert.assert_not_called()

    def test_gap_requires_new_speed_window_and_confirmation(self):
        for i in range(31):
            self.step(i / 30)
        self.alerts.reset_mock()
        for i in range(20):
            t = 2 + i / 30
            track = self.step(t, x=3000 + i * 20)
            self.assertNotEqual(track.current_state, "RUNNING_ALERT")
        track = self.step(2 + 20 / 30, x=3400)
        self.assertEqual(track.current_state, "RUNNING_ALERT")

    def test_short_fast_burst_does_not_confirm(self):
        self.detector._compute_run_speed = Mock(return_value=3.0)
        for t in (0, .05, .1):
            self.step(t)
        self.detector._compute_run_speed.return_value = .5
        self.step(.15)
        self.alerts.trigger_alert.assert_not_called()

    def test_reversed_time_starts_new_window(self):
        for i in range(31):
            self.step(i / 30)
        track = self.step(0)
        self.assertIsNone(self.detector._compute_run_speed(track))
        self.assertIsNone(track.run_fast_since)

    def test_history_is_time_bounded_without_changing_pose_buffer(self):
        for i in range(1201):
            track = self.step(i / 120)
        self.assertLessEqual(len(track.run_motion_history), 242)
        self.assertEqual(len(track.bbox_history), 30)
        self.assertGreaterEqual(track.run_motion_history[-1][0] - track.run_motion_history[0][0], 2)


if __name__ == "__main__":
    unittest.main()
