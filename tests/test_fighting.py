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


class ScuffleTests(unittest.TestCase):
    """Đánh nhẹ / giằng co kéo dài: đòn không đủ nhanh nhưng liên tục vươn vào người đối phương."""

    def simulate(self, seconds, limb=1.0, zone='BODY', walk=0.0, prob=0.03, fps=10):
        det = FightingDetector({'confirm_seconds': 0.15, 'clip_model': {'threshold': .45}})
        det._compute_limb_speed = Mock(return_value=limb)
        det._compute_torso_agitation = Mock(return_value=0.2)
        det._check_strike_zone = Mock(return_value=zone)
        manager, alerts = TrackManager(), []
        for i in range(int(seconds * fps)):
            t = i / fps
            dx = walk * 200 * t                     # cả cặp cùng di chuyển (BH/s, người cao 200 px)
            tracks = [manager.update_track(1, (100 + dx, 100, 180 + dx, 300), (0, 0), None, t),
                      manager.update_track(2, (170 + dx, 100, 250 + dx, 300), (0, 0), None, t)]
            for a in det.process_all(tracks, t, None, model_prob=prob)['new_alerts']:
                alerts.append((t, a))
        return alerts

    def test_sustained_light_scuffle_alerts_despite_model_veto(self):
        alerts = self.simulate(4.0)
        self.assertEqual(len(alerts), 1)
        self.assertGreaterEqual(alerts[0][0], 1.8)   # cần đủ cửa sổ 2 giây

    def test_short_scuffle_does_not_alert(self):
        self.assertEqual(self.simulate(1.5), [])

    def test_standing_close_and_talking_does_not_alert(self):
        self.assertEqual(self.simulate(6.0, limb=0.4), [])

    def test_no_reach_into_opponent_does_not_alert(self):
        self.assertEqual(self.simulate(6.0, zone=None), [])

    def test_walking_together_does_not_alert(self):
        self.assertEqual(self.simulate(6.0, walk=1.0), [])


class MergedPairModelTests(unittest.TestCase):
    """Hai người ôm vật nhau bị YOLO gộp thành 1 khung: mô hình clip vẫn phải xác nhận được."""

    def setUp(self):
        self.det = FightingDetector({'confirm_frames': 99, 'clip_model': {'threshold': .45}})
        self.det._compute_limb_speed = Mock(return_value=0)   # tắt luật theo cặp, chỉ xét mô hình
        self.det._compute_torso_agitation = Mock(return_value=0)
        self.manager = TrackManager()

    def frame(self, t, boxes, prob):
        tracks = [self.manager.update_track(i, b, (0, 0), None, t) for i, b in boxes]
        return self.det.process_all(tracks, t, None, model_prob=prob)['new_alerts']

    def test_pair_merges_into_one_box_still_alerts(self):
        a, b = (100, 100, 180, 300), (170, 100, 250, 300)
        alerts = []
        for i in range(5):                       # 0.5 s: 2 người sát nhau, mô hình chưa chắc
            alerts += self.frame(i / 10, [(1, a), (2, b)], .2)
        for i in range(5, 20):                   # gộp thành 1 khung lớn, mô hình chắc chắn
            alerts += self.frame(i / 10, [(3, (105, 95, 245, 305))], .8)
        self.assertEqual(len(alerts), 1)
        self.assertIn('#3', alerts[0]['extra_info'])

    def test_single_person_without_prior_pair_does_not_alert(self):
        alerts = []
        for i in range(30):
            alerts += self.frame(i / 10, [(1, (100, 100, 180, 300))], .9)
        self.assertEqual(alerts, [])

    def test_merge_memory_expires(self):
        a, b = (100, 100, 180, 300), (170, 100, 250, 300)
        self.frame(0, [(1, a), (2, b)], .2)
        alerts = []
        for i in range(40, 60):                  # 4 s sau mới có 1 khung với xác suất cao
            alerts += self.frame(i / 10, [(3, (105, 95, 245, 305))], .9)
        self.assertEqual(alerts, [])


if __name__ == '__main__':
    unittest.main()

