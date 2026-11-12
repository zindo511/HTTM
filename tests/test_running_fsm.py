import math
import unittest
from datetime import datetime
from unittest.mock import Mock

import numpy as np

from src.behaviors.running import RunningDetector, point_in_polygon
from src.tracking.track_manager import TrackManager


def running_pose(x, phase):
    """Pose 17 khớp có vung chân/tay ngược pha và thân nghiêng về trước."""
    k = np.zeros((17, 3))
    k[:, 2] = 0.9
    swing = 40 * math.sin(phase)
    k[5] = (x + 30, 140, .9)
    k[6] = (x + 30, 140, .9)
    k[11] = (x + 20, 200, .9)
    k[12] = (x + 20, 200, .9)
    k[9] = (x + 30 + swing, 190, .9)
    k[10] = (x + 30 - swing, 190, .9)
    k[15] = (x + 20 + swing, 295, .9)
    k[16] = (x + 20 - swing, 295, .9)
    k[7], k[8] = (k[5] + k[9]) / 2, (k[6] + k[10]) / 2
    return k


class RunningFSMTests(unittest.TestCase):
    def make(self, **cfg):
        self.manager = TrackManager()
        self.alerts = Mock()
        self.det = RunningDetector(cfg, self.alerts, clock=lambda: datetime(2026, 1, 1, 23, 0))

    def step(self, t, speed, pose=False, lean=None, x=None, frame=None):
        x = speed * 200 * t if x is None else x
        kpts = running_pose(x, t * 12) if pose else None
        track = self.manager.update_track(1, (x, 100, x + 60, 300), (x + 30, 300), kpts, t)
        track.debug_metrics = {}
        self.det.process(track, t, frame, lean)
        return track

    def test_fsm_full_cycle_single_alert(self):
        self.make(cooldown_seconds=3.0)
        states = []
        for i in range(60):  # chạy 2 giây @30fps
            states.append(self.step(i / 30, 3.0).run_state)
        self.assertIn("SUSPECTED", states)
        self.assertEqual(states[-1], "CONFIRMED")
        self.assertEqual(self.alerts.trigger_alert.call_count, 1)
        # Dừng lại -> COOLDOWN, sau đó về NORMAL
        x = 3.0 * 200 * 2
        for i in range(30):
            track = self.step(2 + i / 30, 0, x=x)
        self.assertEqual(track.run_state, "COOLDOWN")
        for i in range(60):
            track = self.step(3 + i / 30, 0, x=x)
        self.assertEqual(track.run_state, "NORMAL")
        self.assertEqual(self.alerts.trigger_alert.call_count, 1)

    def test_continuous_long_run_alerts_once(self):
        self.make(cooldown_seconds=1.0)
        for i in range(300):
            self.step(i / 30, 3.0)
        self.assertEqual(self.alerts.trigger_alert.call_count, 1)

    def test_pose_lowers_required_speed(self):
        # 1.6 BH/s: trên ngưỡng 1.3 nhưng dưới ngưỡng không-pose 1.95
        self.make()
        for i in range(60):
            self.step(i / 30, 1.6)
        self.alerts.trigger_alert.assert_not_called()
        self.make()
        for i in range(60):
            track = self.step(i / 30, 1.6, pose=True, lean=20)
        self.assertEqual(self.alerts.trigger_alert.call_count, 1)
        self.assertTrue(self.det.pose_cues(track, 20)["legs"])

    def test_fast_walk_does_not_alert(self):
        self.make()
        for i in range(120):
            self.step(i / 30, 1.0, pose=True, lean=20)
        self.alerts.trigger_alert.assert_not_called()

    def test_outside_zone_is_ignored(self):
        zone = {"name": "hanh lang", "points": [[0, 0], [100, 0], [100, 1000], [0, 1000]]}
        self.make(zones=[zone])
        for i in range(60):
            track = self.step(i / 30, 3.0, x=500 + i * 20)
        self.alerts.trigger_alert.assert_not_called()
        self.assertEqual(track.run_state, "NORMAL")

    def test_inside_normalized_zone_alerts(self):
        zone = {"name": "sanh", "normalized": True, "points": [[0, 0], [1, 0], [1, 1], [0, 1]]}
        self.make(zones=[zone])
        frame = np.zeros((1000, 4000, 3), dtype=np.uint8)
        for i in range(60):
            self.step(i / 30, 3.0, frame=frame)
        self.assertEqual(self.alerts.trigger_alert.call_count, 1)
        self.assertIn("sanh", self.alerts.trigger_alert.call_args.kwargs["extra_info"])

    def test_active_hours_overnight(self):
        self.make(active_hours=["22:00-06:00"])
        self.assertTrue(self.det.check_context(self.manager.update_track(1, (0, 0, 1, 1), (0, 1), None, 0), None)[0])
        self.make(active_hours=["08:00-17:00"])
        for i in range(60):
            self.step(i / 30, 3.0)
        self.alerts.trigger_alert.assert_not_called()

    def test_camera_shake_is_rejected(self):
        self.make()
        for i in range(90):
            x = 100 + (60 if i % 2 else -60)  # rung qua lại, không dịch chuyển thật
            self.step(i / 30, 0, x=x)
        self.alerts.trigger_alert.assert_not_called()

    def test_short_occlusion_keeps_suspicion(self):
        self.make(confirm_seconds=0.3)
        for i in range(20):
            self.step(i / 30, 3.0)
        # bỏ 2 frame (che khuất ngắn) rồi tiếp tục
        for i in range(23, 40):
            self.step(i / 30, 3.0)
        self.assertEqual(self.alerts.trigger_alert.call_count, 1)

    def test_homography_metric_speed(self):
        # 1 pixel = 0.01 m: chạy 600 px/s = 6 m/s
        self.make(homography=[[0.01, 0, 0], [0, 0.01, 0], [0, 0, 1]])
        for i in range(60):
            track = self.step(i / 30, 3.0)
        self.assertAlmostEqual(self.det.compute_speed(track), 6.0, places=5)
        self.assertEqual(self.alerts.trigger_alert.call_count, 1)

    def test_point_in_polygon(self):
        square = [(0, 0), (10, 0), (10, 10), (0, 10)]
        self.assertTrue(point_in_polygon(5, 5, square))
        self.assertFalse(point_in_polygon(15, 5, square))


def frontal_pose(box, phase, swing=True):
    """Pose nhìn chính diện: gối/cổ chân và cổ tay nâng hạ luân phiên theo phương y."""
    x1, y1, x2, y2 = box
    h, cx = y2 - y1, (x1 + x2) / 2
    lift = 0.12 * h * math.sin(phase) if swing else 0.0
    k = np.zeros((17, 3))
    k[:, 2] = 0.9
    k[5], k[6] = (cx - .12 * h, y1 + .2 * h, .9), (cx + .12 * h, y1 + .2 * h, .9)
    k[11], k[12] = (cx - .08 * h, y1 + .5 * h, .9), (cx + .08 * h, y1 + .5 * h, .9)
    k[13], k[14] = (cx - .08 * h, y1 + .72 * h - lift, .9), (cx + .08 * h, y1 + .72 * h + lift, .9)
    k[15], k[16] = (cx - .08 * h, y1 + .95 * h - lift, .9), (cx + .08 * h, y1 + .95 * h + lift, .9)
    k[9], k[10] = (cx - .15 * h, y1 + .45 * h + lift, .9), (cx + .15 * h, y1 + .45 * h - lift, .9)
    k[7], k[8] = (k[5] + k[9]) / 2, (k[6] + k[10]) / 2  # tay duỗi thẳng
    return k


def bent_arm_pose(box):
    """Đứng yên, khuỷu tay gập 90° (tư thế chạy bộ nhìn chính diện)."""
    k = frontal_pose(box, 0.0, swing=False)
    x1, y1, x2, y2 = box
    h = y2 - y1
    k[7], k[8] = k[5] + (0, .15 * h, 0), k[6] + (0, .15 * h, 0)
    k[9], k[10] = k[7] + (.12 * h, 0, 0), k[8] + (-.12 * h, 0, 0)
    k[:, 2] = .9
    return k


class FrontalRunningTests(unittest.TestCase):
    """Người chạy thẳng về phía camera: tâm khung gần như đứng yên, khung to dần."""
    FRAME = np.zeros((720, 1280, 3), dtype=np.uint8)  # f ≈ 1280 px

    def run_toward(self, speed_bh, pose, fps=10, seconds=2.0, swing=True, gait=0.0, bent=False):
        manager, alerts = TrackManager(), Mock()
        det = RunningDetector({"focal_length_px": 1280}, alerts)
        track = None
        for i in range(int(seconds * fps)):
            t = i / fps
            dist = 10.0 - speed_bh * t          # khoảng cách theo chiều cao người
            # Khung co lại khi nhấc chân (chu kỳ bước ~0.8 s)
            h = 1280 / dist * (1 - gait * (0.5 + 0.5 * math.sin(t * 2 * math.pi / 0.8)))
            y2 = 300 + 0.9 * h
            box = (640 - .2 * h, y2 - h, 640 + .2 * h, y2)
            if bent:
                kpts = bent_arm_pose(box)
            else:
                kpts = frontal_pose(box, t * 2 * math.pi * 1.5, swing) if pose else None
            track = manager.update_track(1, box, (640, y2), kpts, t)
            track.debug_metrics = {}
            det.process(track, t, self.FRAME, None)
        return det, track, alerts

    def test_running_toward_camera_alerts(self):
        det, track, alerts = self.run_toward(3.0, pose=True)
        self.assertEqual(alerts.trigger_alert.call_count, 1)
        self.assertTrue(track.run_depth_dominant)

    def test_depth_speed_is_measured(self):
        det, track, _ = self.run_toward(3.0, pose=False)
        lateral, depth = det._depth_speed(track, self.FRAME.shape[:2])
        self.assertAlmostEqual(depth, 3.0, places=3)
        self.assertLess(lateral, 0.05)

    def test_bbox_growth_alone_is_not_enough(self):
        _, _, alerts = self.run_toward(3.0, pose=False)
        alerts.trigger_alert.assert_not_called()
        _, _, alerts = self.run_toward(3.0, pose=True, swing=False)
        alerts.trigger_alert.assert_not_called()

    def test_gait_modulated_bbox_still_measures_depth(self):
        # Khung co giãn 8% theo bước chân (như video thật) vẫn đo được tốc độ tiến lại gần
        det, track, alerts = self.run_toward(2.0, pose=True, seconds=3.0, gait=0.08)
        self.assertGreater(det._depth_speed(track, self.FRAME.shape[:2])[1], 1.5)
        self.assertEqual(alerts.trigger_alert.call_count, 1)

    def test_bent_arms_count_as_running_pose(self):
        _, _, alerts = self.run_toward(2.0, pose=True, seconds=3.0, bent=True)
        self.assertEqual(alerts.trigger_alert.call_count, 1)

    def test_walking_toward_camera_does_not_alert(self):
        _, _, alerts = self.run_toward(0.8, pose=True, seconds=4.0)
        alerts.trigger_alert.assert_not_called()


if __name__ == "__main__":
    unittest.main()
