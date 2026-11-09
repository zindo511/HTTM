import unittest
from unittest.mock import Mock
import numpy as np

from src.behaviors.fall_run import MotionAndFallDetector
from src.tracking.track_manager import TrackManager


class FallTests(unittest.TestCase):
    def setUp(self):
        self.manager = TrackManager()
        self.alerts = Mock()
        self.detector = MotionAndFallDetector({"enabled": False}, {}, alert_manager=self.alerts)

    def step(self, t, box=(100, 100, 160, 300), pose=None, scale=1):
        box = tuple(v * scale for v in box)
        track = self.manager.update_track(1, box, (box[0], box[3]), pose, t)
        self.detector.process(track, t, None, [])
        return track

    def warm(self, fps=30, scale=1):
        for i in range(fps + 1):
            track = self.step(i / fps, scale=scale)
        self.assertAlmostEqual(track.fall_reference_height, 200 * scale)
        return track

    def fall(self, fps=30, scale=1, axial=False):
        self.warm(fps, scale)
        # Fast downward motion while still upright, then horizontal posture later.
        for i in range(1, int(0.2 * fps) + 1):
            t = i / fps
            self.step(1 + t, (100, 100 + 400*t, 160, 300 + 400*t), scale=scale)
        box = (100, 290, 170, 380) if axial else (100, 290, 300, 380)
        for i in range(1, fps + 1):
            track = self.step(1.2 + i / fps, box, scale=scale)
        return track, box

    def test_sequential_fall_and_latch_at_multiple_fps_and_scales(self):
        for fps in (10, 15, 30, 60):
            for scale in (0.5, 1, 2):
                with self.subTest(fps=fps, scale=scale):
                    self.setUp()
                    track, box = self.fall(fps, scale)
                    self.assertEqual(track.fall_phase, "FALL_LATCHED")
                    self.assertEqual(track.current_state, "FALLING_ALERT")
                    for i in range(1, fps + 1):
                        self.step(2.2 + i/fps, box, scale=scale)
                    self.assertEqual(track.current_state, "FALLING_ALERT")
                    self.assertEqual(self.alerts.trigger_alert.call_count, 1)

    def test_axial_height_collapse(self):
        track, _ = self.fall(axial=True)
        self.assertEqual(track.fall_phase, "FALL_LATCHED")

    def test_recovery_needs_sustained_upright_posture(self):
        track, box = self.fall()
        for i in range(1, 10):
            self.step(2.2+i/30)
        self.assertEqual(track.fall_phase, "FALL_LATCHED")
        self.step(2.54, box)
        for i in range(1, 20):
            self.step(2.54+i/30)
        self.assertEqual(track.fall_phase, "NORMAL")
        self.assertEqual(track.current_state, "NORMAL")

    def test_slow_lowering_does_not_trigger(self):
        self.warm()
        for i in range(1, 121):
            f = i/120
            track = self.step(1+i/30, (100, 100+190*f, 160+140*f, 300+80*f))
        for i in range(1, 31):
            self.step(5+i/30, (100, 290, 300, 380))
        self.assertEqual(track.fall_phase, "NORMAL")
        self.alerts.trigger_alert.assert_not_called()

    def test_stumble_expires_without_lying(self):
        self.warm()
        for i in range(1, 7):
            self.step(1+i/30, (100, 100+400*i/30, 160, 300+400*i/30))
        for i in range(1, 61):
            track = self.step(1.2+i/30)
        self.assertEqual(track.fall_phase, "NORMAL")
        self.alerts.trigger_alert.assert_not_called()

    def test_gap_does_not_count_as_lying_time(self):
        self.warm()
        self.step(1.1, (100, 160, 160, 360))
        self.step(1.2, (100, 290, 300, 380))
        track = self.step(2, (100, 290, 300, 380))
        self.assertEqual(track.fall_phase, "NORMAL")
        self.alerts.trigger_alert.assert_not_called()

    def test_running_cannot_short_circuit_fall(self):
        self.detector.config_run = {"enabled": True}
        self.detector._compute_run_speed = lambda track: 10.0
        track, _ = self.fall()
        self.assertEqual(track.current_state, "FALLING_ALERT")

    def test_confidence_fallback(self):
        pose = np.zeros((17, 3))
        pose[5] = (100, 100, 0.9)
        pose[6] = (0, 0, 0.1)
        np.testing.assert_array_equal(self.detector._midpoint(pose, 5, 6), [100, 100])
        pose[5, 2] = 0.1
        self.assertIsNone(self.detector._midpoint(pose, 5, 6))
        pose[5] = (np.nan, 100, 1)
        self.assertIsNone(self.detector._midpoint(pose, 5, 6))

    def test_forward_fill_exactly_five_frames_and_reacquisition(self):
        pose = np.ones((17, 3))
        for i in range(30):
            track = self.step(i/30, pose=pose)
        for i in range(1, 6):
            self.step((29+i)/30)
            self.assertEqual(len(track.keypoints_history), 30)
            self.assertFalse(track.keypoint_observed_history[-1])
        self.step(35/30)
        self.assertEqual(len(track.keypoints_history), 0)
        self.step(36/30, pose=pose)
        self.assertEqual(len(track.keypoints_history), 1)
        self.assertEqual(track.missing_kpts_count, 0)

    def test_long_gap_does_not_fill_and_bad_pose_is_missing(self):
        pose = np.ones((17, 3))
        self.step(0, pose=pose)
        track = self.step(1, pose=np.zeros((17, 3)))
        self.assertEqual(len(track.keypoints_history), 0)

    def test_bilstm_accepts_and_rejects(self):
        self.detector.bilstm_model = Mock()
        track = self.manager.get_or_create(1, 0)
        self.assertIsNone(self.detector._verify_fall(track))
        for _ in range(30):
            track.keypoints_history.append(np.ones((17, 3)))
            track.keypoint_bbox_history.append((0, 0, 100, 200))
        self.detector.bilstm_model.predict.return_value = (0.02, 0.03, 0.95)
        self.assertEqual(self.detector._verify_fall(track), 0.95)
        self.assertEqual(self.detector.bilstm_model.predict.call_args.args[0].shape, (1, 30, 34))
        self.detector.bilstm_model.predict.return_value = (0.9, 0.05, 0.05)
        self.assertIsNone(self.detector._verify_fall(track))


if __name__ == "__main__":
    unittest.main()
