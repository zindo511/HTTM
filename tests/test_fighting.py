import unittest
from unittest.mock import Mock
import numpy as np
from src.behaviors.fighting import FightingDetector
from src.tracking.track_manager import TrackManager

class FightingTests(unittest.TestCase):
    def test_contract_and_hold(self):
        detector = FightingDetector({'confirm_frames': 2, 'hold_seconds': 1.5})
        manager = TrackManager()
        tracks = [manager.update_track(i, (100+i*10,100,180+i*10,300), (140,300), None, 0) for i in (1,2)]
        detector._compute_limb_speed = Mock(return_value=3)
        detector._compute_torso_agitation = Mock(return_value=0)
        detector._check_strike_zone = Mock(return_value='BODY')
        self.assertEqual(detector.process_all([], 0, None)['new_alerts'], [])
        self.assertFalse(detector.process_all(tracks, 0, None)['is_fighting'])
        self.assertEqual(tracks[0].current_state, 'FIGHTING_SUSPECT')
        result = detector.process_all(tracks, .1, None)
        self.assertTrue(result['is_fighting'])
        self.assertEqual(len(result['new_alerts']), 1)
        detector._compute_limb_speed.return_value = 0
        self.assertTrue(detector.process_all(tracks, .5, None)['is_fighting'])
        self.assertFalse(detector.process_all(tracks, 2, None)['is_fighting'])

    def make_pair(self, config=None):
        detector = FightingDetector(config or {'confirm_frames': 2})
        manager = TrackManager()
        tracks = [manager.update_track(i, (100+i*10,100,180+i*10,300),
                                       (140,300), None, 0) for i in (1,2)]
        detector._compute_limb_speed = Mock(return_value=3)
        detector._compute_torso_agitation = Mock(return_value=0)
        detector._check_strike_zone = Mock(return_value='BODY')
        return detector, tracks

    def test_occlusion_keeps_region_and_fall_priority(self):
        detector, tracks = self.make_pair()
        detector.process_all(tracks, 0, None)
        detector.process_all(tracks, .1, None)
        tracks[0].current_state = 'FALLING_ALERT'
        result = detector.process_all(tracks[:1], .2, None)
        self.assertTrue(result['is_fighting'])
        self.assertEqual(len(result['conflict_boxes']), 1)
        self.assertEqual(tracks[0].current_state, 'FALLING_ALERT')
        self.assertTrue(detector.process_all([], .3, None)['is_fighting'])
        self.assertFalse(detector.process_all([], 2, None)['is_fighting'])
        detector.process_all([], 5, None)
        self.assertFalse(detector.pair_streaks)
        self.assertFalse(detector.pair_visuals)

    def test_stationary_close_people_do_not_alert(self):
        detector, tracks = self.make_pair()
        detector._compute_limb_speed.return_value = 0
        for i in range(30):
            self.assertFalse(detector.process_all(tracks, i/30, None)['is_fighting'])
        self.assertTrue(all(t.current_state == 'NORMAL' for t in tracks))

    def test_far_people_do_not_alert_even_with_fast_limbs(self):
        detector, tracks = self.make_pair()
        tracks[1].bbox_history[-1] = (1000, 100, 1080, 300)
        for i in range(30):
            self.assertFalse(detector.process_all(tracks, i/30, None)['is_fighting'])

    def test_gap_does_not_combine_old_suspicion(self):
        detector, tracks = self.make_pair()
        detector.process_all(tracks, 0, None)
        self.assertFalse(detector.process_all(tracks, 1, None)['is_fighting'])

    def test_foreground_background_overlap_does_not_alert(self):
        detector, tracks = self.make_pair()
        tracks[1].bbox_history[-1] = (100, 170, 120, 220)
        for i in range(30):
            self.assertFalse(detector.process_all(tracks, i/30, None)['is_fighting'])

    def test_cooldown_suppresses_repeated_alerts(self):
        detector, tracks = self.make_pair()
        alerts = []
        for i in range(30):
            alerts.extend(detector.process_all(tracks, i/30, None)['new_alerts'])
        self.assertEqual(len(alerts), 1)

    def test_disabled_contract(self):
        detector, tracks = self.make_pair({'enabled': False})
        self.assertEqual(detector.process_all(tracks, 0, None), {
            'is_fighting': False, 'links': [], 'conflict_boxes': [], 'new_alerts': []})

    def test_actual_pose_strikes_and_uniform_translation(self):
        for strike in (False, True):
            with self.subTest(strike=strike):
                detector = FightingDetector({'confirm_frames': 4})
                manager = TrackManager()
                alerts = []
                for i in range(15):
                    tracks = []
                    for tid, x in ((1, 100), (2, 180)):
                        x += i * 3 if not strike else 0
                        pose = np.ones((17, 3), dtype=np.float32)
                        pose[:, 0], pose[:, 1], pose[:, 2] = x + 40, 180, .9
                        if strike and tid == 1:
                            pose[9, 0] = x + 60 + (i % 8) * 45
                        tracks.append(manager.update_track(tid, (x, 100, x+80, 300),
                                                           (x+40,300), pose, i/30))
                    alerts.extend(detector.process_all(tracks, i/30, None)['new_alerts'])
                self.assertEqual(bool(alerts), strike)

    def test_time_confirmation_at_different_fps(self):
        for fps in (10, 15, 30, 60):
            with self.subTest(fps=fps):
                detector, tracks = self.make_pair({'confirm_seconds': .15})
                first_alert = None
                for i in range(fps):
                    result = detector.process_all(tracks, i/fps, None)
                    if result['new_alerts'] and first_alert is None:
                        first_alert = i/fps
                self.assertIsNotNone(first_alert)
                self.assertGreaterEqual(first_alert, .15)
                self.assertLessEqual(first_alert, .15 + 1/fps)

if __name__ == '__main__':
    unittest.main()

