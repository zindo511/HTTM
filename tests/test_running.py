import math
import unittest
from unittest.mock import Mock

from src.behaviors.fall_run import MotionAndFallDetector
from src.tracking.track_manager import TrackManager
from tests.poses import bbox, skeleton

RUN_CFG = {'v_norm_threshold': 1.5, 'body_lean_min': 15.0, 'window_seconds': 0.5,
           'confirm_frames': 3, 'cooldown_seconds': 4.0}


class RunningTests(unittest.TestCase):
    def setUp(self, config=None):
        self.alerts = Mock()
        self.manager = TrackManager()
        # Fall FSM stays on, as in production, so running and falling must coexist.
        self.detector = MotionAndFallDetector(config or dict(RUN_CFG), {}, alert_manager=self.alerts)

    def move(self, speed_bh, seconds, fps=15, h=200, lean=0.0, tid=1, t0=0.0, x0=100.0,
             swing=True, jitter=0.0):
        """Person translating horizontally at `speed_bh` body-heights per second."""
        states = []
        track = None
        for i in range(int(seconds * fps) + 1):
            t = t0 + i / fps
            cx = x0 + speed_bh * h * (t - t0) + (jitter * (-1) ** i)
            pose = skeleton(cx, 100, h, swing=math.sin(t * 2 * math.pi * 2) if swing else 0,
                            lean_deg=lean)
            track = self.manager.update_track(tid, bbox(cx, 100, h), (cx, 100 + h), pose, t)
            self.detector.process(track, t, None, [])
            states.append(track.current_state)
        return track, states

    def run_alerts(self):
        return [c.kwargs for c in self.alerts.trigger_alert.call_args_list
                if c.kwargs['event_type'] == 'RUNNING']

    def test_standing_with_bbox_jitter_never_alerts(self):
        _, states = self.move(0.0, 5, jitter=4, swing=False)
        self.assertNotIn('RUNNING_ALERT', states)
        self.assertEqual(self.run_alerts(), [])

    def test_walking_does_not_alert(self):
        _, states = self.move(0.8, 5, lean=5)
        self.assertNotIn('RUNNING_ALERT', states)

    def test_running_with_forward_lean_alerts_once_per_cooldown(self):
        _, states = self.move(2.2, 3, lean=25)
        self.assertIn('RUNNING_ALERT', states)
        alerts = self.run_alerts()
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]['track_id'], 1)
        self.assertGreater(alerts[0]['confidence'], 0.5)

    def test_upright_needs_much_higher_speed(self):
        _, states = self.move(1.8, 3, lean=0)
        self.assertNotIn('RUNNING_ALERT', states)
        self.setUp()
        _, states = self.move(2.6, 3, lean=0)
        self.assertIn('RUNNING_ALERT', states)

    def test_alert_clears_after_stopping(self):
        track, _ = self.move(2.5, 2, lean=25)
        self.assertEqual(track.current_state, 'RUNNING_ALERT')
        x_end = 100 + 2.5 * 200 * 2
        track, states = self.move(0.0, 2, lean=0, t0=2 + 1 / 15, x0=x_end, swing=False)
        self.assertEqual(track.current_state, 'NORMAL')

    def test_first_alert_time_is_fps_independent(self):
        for fps in (10, 15, 30, 60):
            with self.subTest(fps=fps):
                self.setUp()
                _, states = self.move(2.5, 2, fps=fps, lean=25)
                first = states.index('RUNNING_ALERT') / fps
                # 0.5s velocity window + confirmation; must not depend on FPS much.
                self.assertGreaterEqual(first, 0.5)
                self.assertLessEqual(first, 0.85)

    def test_speed_is_normalised_by_body_height(self):
        for h in (60, 200, 500):
            with self.subTest(height=h):
                self.setUp()
                _, states = self.move(2.2, 3, h=h, lean=25)
                self.assertIn('RUNNING_ALERT', states)
                self.setUp()
                _, states = self.move(0.8, 3, h=h, lean=25)
                self.assertNotIn('RUNNING_ALERT', states)

    def test_tracker_id_switch_teleport_does_not_alert(self):
        # ByteTrack can hand an ID to someone on the other side of the frame.
        self.move(0.0, 2, x0=100, swing=False)
        _, states = self.move(0.0, 2, x0=900, t0=2 + 1 / 15, swing=False, lean=20)
        self.assertNotIn('RUNNING_ALERT', states)
        self.assertEqual(self.run_alerts(), [])

    def test_reappearing_after_occlusion_does_not_alert(self):
        # Walks behind a pillar for 1s and reappears 1 BH further on.
        self.move(0.5, 2, x0=100)
        _, states = self.move(0.5, 2, x0=100 + 0.5 * 200 * 3, t0=3.0, lean=20)
        self.assertNotIn('RUNNING_ALERT', states)

    def test_disabled(self):
        self.setUp({**RUN_CFG, 'enabled': False})
        _, states = self.move(3.0, 3, lean=30)
        self.assertNotIn('RUNNING_ALERT', states)


if __name__ == '__main__':
    unittest.main()
