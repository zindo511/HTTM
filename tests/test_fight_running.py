"""Fighting vs. running interactions, with real (synthetic) skeletons, no mocks."""
import math
import random
import unittest

from src.behaviors.fall_run import MotionAndFallDetector
from src.behaviors.fighting import FightingDetector
from src.tracking.track_manager import TrackManager
from tests.poses import bbox, skeleton

FIGHT_CFG = {'max_distance_bh': 1.15, 'max_height_ratio': 2.0, 'limb_speed_threshold': 2.1,
             'hold_seconds': 1.5, 'confirm_frames': 4, 'confirm_seconds': 0.15,
             'cooldown_seconds': 6.0}
RUN_CFG = {'v_norm_threshold': 1.5, 'body_lean_min': 15.0, 'window_seconds': 0.5,
           'confirm_frames': 3, 'cooldown_seconds': 4.0}
H = 200
_rng = random.Random(1)
ARMS = [_rng.uniform(-2.4, 2.4) for _ in range(200)]


class FightRunningTests(unittest.TestCase):
    def simulate(self, people, seconds, fps=15):
        """people: {tid: fn(t) -> (cx, swing, lean)}. Returns (fight_alerts, states per tid)."""
        manager = TrackManager()
        fight = FightingDetector(dict(FIGHT_CFG))
        motion = MotionAndFallDetector(dict(RUN_CFG), {})
        alerts, states = [], {tid: [] for tid in people}
        for i in range(int(seconds * fps) + 1):
            t = i / fps
            active = []
            for tid, fn in people.items():
                cx, swing, lean = fn(t)
                track = manager.update_track(tid, bbox(cx, 100, H), (cx, 100 + H),
                                             skeleton(cx, 100, H, swing=swing, lean_deg=lean), t)
                motion.process(track, t, None, [])
                active.append(track)
            alerts += fight.process_all(active, t, None)['new_alerts']
            for track in active:
                states[track.track_id].append(track.current_state)
        return alerts, states

    @staticmethod
    def jog(x0, speed_bh, phase=0.0, lean=20):
        # Running arm swing at 2.5 Hz: wrists move ~4 BH/s relative to the torso.
        return lambda t: (x0 + speed_bh * H * t, math.sin(2 * math.pi * 2.5 * t + phase), lean)

    @staticmethod
    def flurry(x0, seed=0):
        # Flailing fists: a new arm position every 0.1s (FPS independent), up to 0.6 BH reach.
        rng = random.Random(seed)
        arms = [rng.uniform(-2.4, 2.4) for _ in range(200)]
        return lambda t: (x0, arms[int(t / 0.1)], 0)

    @staticmethod
    def still(x0):
        return lambda t: (x0, 0.0, 0)

    def test_joggers_side_by_side_are_not_fighting(self):
        for fps in (12, 15, 30):
            with self.subTest(fps=fps):
                alerts, states = self.simulate({1: self.jog(100, 2.0), 2: self.jog(100 + 0.6 * H, 2.0, 1.3)}, 4, fps)
                self.assertEqual(alerts, [])
                self.assertNotIn('FIGHTING_ALERT', states[1])
                # They are still detected as running.
                self.assertIn('RUNNING_ALERT', states[1])

    def test_runners_crossing_in_opposite_directions_are_not_fighting(self):
        for fps in (12, 15, 30):
            with self.subTest(fps=fps):
                alerts, _ = self.simulate({1: self.jog(100, 2.0), 2: self.jog(100 + 4 * H, -2.0, 0.7)}, 3, fps)
                self.assertEqual(alerts, [])

    def test_punching_while_standing_is_fighting(self):
        for fps in (12, 15, 30):
            with self.subTest(fps=fps):
                alerts, states = self.simulate({1: self.flurry(300), 2: self.flurry(300 + 0.45 * H, 7)}, 2, fps)
                self.assertEqual(len(alerts), 1)
                self.assertIn('FIGHTING_ALERT', states[2])

    def test_charge_then_strike_is_fighting(self):
        # #1 sprints at a standing #2 and starts hitting once in reach.
        stop_x = 300 + 4.5 * H - 0.45 * H
        def attacker(t):
            if t < 1.0:
                return 300 + 3.6 * H * t + 0 * t, math.sin(2 * math.pi * 2.5 * t), 25
            return stop_x, ARMS[int(t / 0.1)], 0
        alerts, states = self.simulate({1: attacker, 2: self.still(300 + 4.5 * H)}, 3)
        self.assertGreaterEqual(len(alerts), 1)
        self.assertIn('FIGHTING_ALERT', states[1])

    def test_fight_state_wins_over_running_state(self):
        # A sprinting attacker that ends in a brawl is shown as fighting, not running.
        stop_x = 300 + 4.5 * H - 0.45 * H
        def attacker(t):
            if t < 1.0:
                return 300 + 3.6 * H * t, math.sin(2 * math.pi * 2.5 * t), 25
            return stop_x, ARMS[int(t / 0.1)], 0
        _, states = self.simulate({1: attacker, 2: self.still(300 + 4.5 * H)}, 3)
        self.assertEqual(states[1][-1], 'FIGHTING_ALERT')


if __name__ == '__main__':
    unittest.main()
