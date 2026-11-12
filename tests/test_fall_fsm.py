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

    def test_short_gap_pauses_lying_confirmation(self):
        self.warm()
        self.step(1.1, (100, 160, 160, 360))
        box = (100, 290, 300, 380)
        for t in (1.2, 1.28, 1.36, 1.44, 1.52, 1.60, 1.68):
            self.step(t, box)
        track = self.step(2.0, box)
        self.assertEqual(track.fall_phase, "LYING_VERIFY")
        self.alerts.trigger_alert.assert_not_called()
        track = self.step(2.08, box)
        self.assertEqual(track.fall_phase, "FALL_LATCHED")
        self.alerts.trigger_alert.assert_called_once()

    def test_standing_on_return_discards_prior_low_pose_time(self):
        self.warm()
        self.step(1.1, (100, 160, 160, 360))
        box = (100, 290, 300, 380)
        for t in (1.2, 1.28, 1.36, 1.44, 1.52, 1.60, 1.68):
            self.step(t, box)
        track = self.step(2.0)
        self.assertEqual(track.fall_phase, "IMPACT_PENDING")
        self.assertIsNone(track.fall_lying_since)
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

    def step_id(self, tid, t, box):
        track = self.manager.update_track(tid, box, (box[0], box[3]), None, t)
        if len(track.bbox_history) == 1:
            self.detector.adopt_fall_candidate(track, self.manager.tracks.values(), t)
        self.detector.process(track, t, None, [])
        return track

    def start_fall_then_lose_id(self):
        self.warm()
        for i in range(1, 7):
            t = i / 30
            self.step(1 + t, (100, 100 + 400*t, 160, 300 + 400*t))
        lying = (100, 290, 300, 380)
        for i in range(1, 4):
            old = self.step(1.2 + i / 30, lying)
        self.assertEqual(old.fall_phase, "LYING_VERIFY")
        return old, lying

    def test_fall_survives_tracker_id_switch(self):
        old, lying = self.start_fall_then_lose_id()
        # ID 1 vanishes for 0.5 s, then the lying person comes back as ID 2.
        for i in range(0, 30):
            new = self.step_id(2, 1.8 + i / 30, lying)
        self.assertEqual(new.fall_phase, "FALL_LATCHED")
        self.assertEqual(old.fall_phase, "NORMAL")
        self.assertEqual(self.alerts.trigger_alert.call_count, 1)
        self.assertEqual(self.alerts.trigger_alert.call_args.kwargs["track_id"], 2)

    def test_no_handover_to_distant_or_late_track(self):
        _, lying = self.start_fall_then_lose_id()
        far = (900, 290, 1100, 380)
        for i in range(0, 30):
            other = self.step_id(2, 1.4 + i / 30, far)
        self.assertEqual(other.fall_phase, "NORMAL")
        for i in range(0, 30):
            late = self.step_id(3, 3.0 + i / 30, lying)
        self.assertEqual(late.fall_phase, "NORMAL")
        self.alerts.trigger_alert.assert_not_called()

    def slow_collapse(self, end_box, seconds=0.9, fps=15):
        self.warm(fps)
        n = int(seconds * fps)
        for i in range(1, n + 1):
            k = i / n
            box = tuple(a + (b - a) * k for a, b in zip((100, 100, 160, 300), end_box))
            track = self.step(1 + i / fps, box)
        for i in range(1, fps + 1):
            track = self.step(1 + seconds + i / fps, end_box)
        return track

    def test_slow_collapse_into_lying_posture_latches(self):
        # Height 200 -> 90 over 0.9 s: hip speed stays below v_y_drop_min.
        track = self.slow_collapse((60, 210, 300, 300))
        self.assertEqual(track.fall_phase, "FALL_LATCHED")
        self.assertEqual(self.alerts.trigger_alert.call_count, 1)

    def test_occluded_legs_do_not_count_as_slow_fall(self):
        # Box shrinks from the bottom (walking behind furniture) but stays upright.
        track = self.slow_collapse((100, 100, 160, 200))
        for i in range(1, 8):
            track = self.step(2.9 + i / 15, (100, 100, 160, 200))
        self.assertEqual(track.fall_phase, "NORMAL")
        self.alerts.trigger_alert.assert_not_called()

    @staticmethod
    def side_pose(box, knee_drop):
        """Shoulders at the box top, hips mid-box, knees `knee_drop` x torso below the hips."""
        x1, y1, x2, y2 = box
        pose = np.zeros((17, 3))
        pose[:, 2] = 1.0
        cx, torso = (x1 + x2) / 2, (y2 - y1) * 0.4
        pose[5:7, :2] = (cx, y1 + 10)
        pose[11:13, :2] = (cx, y1 + 10 + torso)
        pose[13:15, :2] = (cx + torso, y1 + 10 + torso + knee_drop * torso)
        return pose

    def collapse_with_pose(self, knee_drop):
        start, end = (100, 100, 160, 300), (60, 210, 300, 300)
        for i in range(16):
            self.step(i / 15, start, pose=self.side_pose(start, knee_drop))
        for i in range(1, 14):
            k = i / 13
            box = tuple(a + (b - a) * k for a, b in zip(start, end))
            # Thighs keep their pre-fall orientation until the body is down.
            track = self.step(1 + i / 15, box, pose=self.side_pose(box, knee_drop if k < 0.5 else 0.0))
        for i in range(1, 16):
            track = self.step(1.9 + i / 15, end)
        return track

    def test_sitting_then_lying_down_is_not_a_slow_fall(self):
        track = self.collapse_with_pose(knee_drop=0.1)  # thighs horizontal: sitting
        self.assertNotEqual(track.fall_phase, "FALL_LATCHED")
        self.alerts.trigger_alert.assert_not_called()

    def test_standing_then_slow_collapse_still_latches(self):
        track = self.collapse_with_pose(knee_drop=0.9)  # thighs vertical: standing
        self.assertEqual(track.fall_phase, "FALL_LATCHED")

    def test_bending_over_after_sudden_dip_is_not_lying(self):
        self.warm()
        for i in range(1, 7):
            t = i / 30
            self.step(1 + t, (100, 100 + 400*t, 160, 300 + 400*t))
        # Torso horizontal (shoulders level with hips) but the box keeps 90% height.
        bent = (100, 120, 320, 300)
        pose = np.zeros((17, 3))
        pose[:, 2] = 1.0
        pose[5:7, :2] = (300, 180)
        pose[11:13, :2] = (200, 180)
        for i in range(1, 40):
            track = self.step(1.2 + i / 30, bent, pose=pose)
        self.assertNotEqual(track.fall_phase, "FALL_LATCHED")
        self.alerts.trigger_alert.assert_not_called()

    def test_brief_pose_jitter_does_not_restart_lying_timer(self):
        self.warm()
        for i in range(1, 7):
            t = i / 30
            self.step(1 + t, (100, 100 + 400*t, 160, 300 + 400*t))
        lying, upright = (100, 290, 300, 380), (100, 230, 160, 380)
        for i in range(1, 5):
            track = self.step(1.2 + i / 30, lying)
        since = track.fall_lying_since
        self.step(1.2 + 5 / 30, upright)  # one bad frame (~33 ms)
        self.assertEqual(track.fall_lying_since, since)
        for i in range(6, 30):
            track = self.step(1.2 + i / 30, lying)
        self.assertEqual(track.fall_phase, "FALL_LATCHED")

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
