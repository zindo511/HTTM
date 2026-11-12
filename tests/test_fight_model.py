import os
import tempfile
import unittest

import numpy as np
import torch

from src.behaviors.fighting import FightingDetector
from src.models.fight_classifier import DEFAULT_CONFIG, FightClipClassifier, FightClipInference
from src.tracking.track_manager import TrackManager
import tests.test_fighting as rule_tests


class ClipModelDecisionTests(unittest.TestCase):
    CONFIG = {'clip_model': {'threshold': 0.7, 'veto_threshold': 0.2, 'confirm_seconds': 0.5},
              'hold_seconds': 1.5, 'cooldown_seconds': 6.0}

    def make_tracks(self, count=2):
        manager = TrackManager()
        return [manager.update_track(7 + i, (100 + i * 80, 100, 180 + i * 80, 300),
                                     (140 + i * 80, 300), None, 0) for i in range(count)]

    def test_model_alerts_with_two_people_and_holds_after_occlusion(self):
        detector, tracks = FightingDetector(self.CONFIG), self.make_tracks()
        self.assertFalse(detector.process_all(tracks, 0.0, None, model_prob=0.9)['is_fighting'])
        self.assertFalse(detector.process_all(tracks, 0.3, None, model_prob=0.9)['is_fighting'])
        result = detector.process_all(tracks, 0.6, None, model_prob=0.9)
        self.assertTrue(result['is_fighting'])
        self.assertEqual([a['track_id'] for a in result['new_alerts']], [7])
        self.assertEqual(tracks[0].current_state, 'FIGHTING_ALERT')
        # Trong thời gian cooldown không báo lặp, nhưng vẫn giữ vùng cảnh báo.
        again = detector.process_all(tracks[:1], 1.0, None, model_prob=0.9)
        self.assertTrue(again['is_fighting'])
        self.assertEqual(again['new_alerts'], [])

    def test_high_model_score_with_one_person_does_not_alert(self):
        detector, tracks = FightingDetector(self.CONFIG), self.make_tracks(count=1)
        for t in (0.0, 0.3, 0.6, 1.2):
            result = detector.process_all(tracks, t, None, model_prob=0.99)
            self.assertFalse(result['is_fighting'])
            self.assertEqual(result['new_alerts'], [])

    def test_losing_second_person_restarts_confirmation(self):
        detector, tracks = FightingDetector(self.CONFIG), self.make_tracks()
        detector.process_all(tracks, 0.0, None, model_prob=0.9)
        detector.process_all(tracks[:1], 0.3, None, model_prob=0.9)
        self.assertFalse(detector.process_all(tracks, 0.6, None, model_prob=0.9)['is_fighting'])
        self.assertTrue(detector.process_all(tracks, 1.2, None, model_prob=0.9)['is_fighting'])

    def test_probability_dip_restarts_confirmation(self):
        detector, tracks = FightingDetector(self.CONFIG), self.make_tracks()
        detector.process_all(tracks, 0.0, None, model_prob=0.9)
        detector.process_all(tracks, 0.3, None, model_prob=0.4)
        self.assertFalse(detector.process_all(tracks, 0.6, None, model_prob=0.9)['is_fighting'])

    def test_model_does_not_assign_fight_to_a_confirmed_fall(self):
        detector, tracks = FightingDetector(self.CONFIG), self.make_tracks()
        tracks[0].current_state = 'FALLING_ALERT'
        for t in (0.0, 0.3, 0.6):
            result = detector.process_all(tracks, t, None, model_prob=0.95)
            self.assertFalse(result['is_fighting'])
            self.assertEqual(result['new_alerts'], [])
        self.assertEqual(tracks[0].current_state, 'FALLING_ALERT')

    def test_no_alert_without_people_or_without_threshold(self):
        detector = FightingDetector(self.CONFIG)
        for t in (0.0, 0.6, 1.2):
            self.assertFalse(detector.process_all([], t, None, model_prob=0.99)['is_fighting'])
        no_threshold = FightingDetector({'clip_model': {}})
        tracks = self.make_tracks()
        for t in (0.0, 0.6, 1.2):
            self.assertFalse(no_threshold.process_all(tracks, t, None, model_prob=0.99)['is_fighting'])

    def test_low_probability_vetoes_pair_rules(self):
        detector, tracks = rule_tests.FightingTests.make_pair(None, {**self.CONFIG, 'confirm_frames': 2})
        detector.process_all(tracks, 0.0, None, model_prob=0.05)
        result = detector.process_all(tracks, 0.1, None, model_prob=0.05)
        self.assertFalse(result['is_fighting'])
        self.assertEqual(result['new_alerts'], [])

    def test_rules_still_work_when_model_is_unsure(self):
        detector, tracks = rule_tests.FightingTests.make_pair(None, {**self.CONFIG, 'confirm_frames': 2})
        detector.process_all(tracks, 0.0, None, model_prob=0.4)
        result = detector.process_all(tracks, 0.1, None, model_prob=0.4)
        self.assertTrue(result['is_fighting'])
        self.assertEqual(len(result['new_alerts']), 1)


class ClipModelRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = os.path.join(cls.tmp.name, 'fight_clip.pt')
        model = FightClipClassifier(DEFAULT_CONFIG['hidden'], DEFAULT_CONFIG['dropout'])
        torch.save({'state_dict': model.state_dict(), 'config': dict(DEFAULT_CONFIG), 'threshold': 0.65},
                   cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_forward_shape(self):
        model = FightClipClassifier().eval()
        with torch.no_grad():
            self.assertEqual(tuple(model(torch.zeros(2, 4, 3, 64, 64)).shape), (2,))

    def test_runtime_needs_full_window_and_resets_on_gap(self):
        runner = FightClipInference(self.path)
        self.assertAlmostEqual(runner.recommended_threshold, 0.65)
        frame = np.random.default_rng(0).integers(0, 255, (120, 160, 3), dtype=np.uint8)
        n = runner.num_frames
        for i in range(n - 1):
            self.assertIsNone(runner.update(frame, i * runner.interval))
        p = runner.update(frame, (n - 1) * runner.interval)
        self.assertTrue(0.0 <= p <= 1.0)
        # Khung hình đến quá sớm không được lấy mẫu: giữ nguyên kết quả cũ.
        self.assertEqual(runner.update(frame, (n - 1) * runner.interval + 0.01), p)
        # Đứt quãng dài -> bỏ cửa sổ cũ, phải gom lại từ đầu.
        self.assertIsNone(runner.update(frame, 100.0))
        self.assertEqual(len(runner.features), 1)

    def test_sampling_rate_matches_training_for_any_input_fps(self):
        frame = np.zeros((60, 80, 3), dtype=np.uint8)
        for input_fps in (8, 10, 12.5, 15, 25, 30):
            with self.subTest(input_fps=input_fps):
                runner = FightClipInference(self.path)
                runner.features = __import__('collections').deque(maxlen=1000)
                for i in range(int(input_fps * 10)):
                    runner.update(frame, i / input_fps)
                times = [t for t, _ in runner.features]
                rate = (len(times) - 1) / (times[-1] - times[0])
                # ~8 khung/giây như lúc train, không bị giảm còn 5 ở nguồn 10 fps
                self.assertAlmostEqual(rate, 1 / runner.interval, delta=1.0)


if __name__ == '__main__':
    unittest.main()
