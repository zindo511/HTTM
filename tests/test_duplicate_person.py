import unittest
from unittest.mock import Mock

import numpy as np

from src.behaviors.fighting import FightingDetector
from src.tracking.track_manager import TrackManager
from src.utils.geometry import is_same_person


def pose_at(x, y, h, jitter=0.0, seed=0):
    """Bộ khớp xương đứng thẳng tại (x, y) cao h pixel."""
    rng = np.random.default_rng(seed)
    rel = np.array([
        (0, 0), (-.03, -.02), (.03, -.02), (-.06, 0), (.06, 0),      # mặt
        (-.12, .18), (.12, .18), (-.16, .33), (.16, .33),            # vai, khuỷu
        (-.18, .47), (.18, .47), (-.08, .52), (.08, .52),            # cổ tay, hông
        (-.08, .75), (.08, .75), (-.08, .97), (.08, .97),            # gối, cổ chân
    ])
    k = np.zeros((17, 3))
    k[:, 0] = x + rel[:, 0] * h + rng.normal(0, jitter * h, 17)
    k[:, 1] = y + rel[:, 1] * h + rng.normal(0, jitter * h, 17)
    k[:, 2] = .9
    return k


class SamePersonTests(unittest.TestCase):
    def test_nested_boxes_with_same_skeleton_are_one_person(self):
        k = pose_at(200, 100, 200)
        outer, inner = (150, 90, 250, 300), (165, 100, 240, 290)
        self.assertTrue(is_same_person(k, outer, pose_at(200, 100, 200, jitter=.01, seed=1), inner))

    def test_two_people_side_by_side_are_different(self):
        a, b = pose_at(200, 100, 200), pose_at(290, 100, 200)
        self.assertFalse(is_same_person(a, (150, 90, 250, 300), b, (240, 90, 340, 300)))

    def test_grappling_overlap_is_still_two_people(self):
        # Ôm vật nhau: khung chồng lấn mạnh nhưng khớp xương lệch nửa thân người
        a, b = pose_at(200, 100, 200), pose_at(245, 110, 200)
        self.assertFalse(is_same_person(a, (150, 90, 250, 300), b, (195, 100, 295, 310)))

    def test_missing_pose_never_merges(self):
        self.assertFalse(is_same_person(None, (0, 0, 1, 1), pose_at(0, 0, 10), (0, 0, 1, 1)))


class FightDuplicateTests(unittest.TestCase):
    def run_pair(self, kpts_a, kpts_b, box_a, box_b):
        det = FightingDetector({"confirm_frames": 2})
        det._compute_limb_speed = Mock(return_value=3)
        det._compute_torso_agitation = Mock(return_value=0)
        det._check_strike_zone = Mock(return_value="HEAD")
        manager, alerts = TrackManager(), []
        for i in range(10):
            t = i / 10
            tracks = [manager.update_track(1, box_a, (0, 0), kpts_a, t),
                      manager.update_track(2, box_b, (0, 0), kpts_b, t)]
            alerts += det.process_all(tracks, t, None)["new_alerts"]
        return alerts

    def test_one_person_with_two_boxes_does_not_fight(self):
        k = pose_at(200, 100, 200)
        alerts = self.run_pair(k, pose_at(200, 100, 200, jitter=.01, seed=2),
                               (150, 90, 250, 300), (165, 100, 240, 290))
        self.assertEqual(alerts, [])

    def test_two_real_people_still_fight(self):
        alerts = self.run_pair(pose_at(200, 100, 200), pose_at(290, 100, 200),
                               (150, 90, 250, 300), (240, 90, 340, 300))
        self.assertEqual(len(alerts), 1)


class PipelineDedupeTests(unittest.TestCase):
    def test_keeps_higher_confidence_duplicate(self):
        from src.core.pipeline import SurveillancePipeline
        boxes = np.array([(165, 100, 240, 290), (150, 90, 250, 300), (400, 90, 500, 300)], dtype=float)
        kpts = np.stack([pose_at(200, 100, 200, jitter=.01, seed=3), pose_at(200, 100, 200),
                         pose_at(450, 100, 200)])
        keep = SurveillancePipeline._drop_duplicate_people(boxes, kpts, np.array([.4, .9, .8]))
        self.assertEqual(keep, {1, 2})


class CaptureRotateTests(unittest.TestCase):
    def test_rotates_live_stream_but_not_files(self):
        from src.core.capture import CaptureThread
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        live = CaptureThread(source="http://192.168.1.2:4747/video", rotate=90)
        self.assertEqual(live._downscale(frame).shape[:2], (1280, 720))
        f = CaptureThread(source=__file__, rotate=90)
        self.assertEqual(f._downscale(frame).shape[:2], (720, 1280))


if __name__ == "__main__":
    unittest.main()
