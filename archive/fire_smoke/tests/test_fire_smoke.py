import unittest
import numpy as np
from src.behaviors.fire_smoke import FireSmokeDetector
from src.tracking.track_manager import TrackManager


class FireSmokeTests(unittest.TestCase):
    def setUp(self):
        self.manager = TrackManager()
        self.detector = FireSmokeDetector({
            "enabled": True,
            "conf_threshold": 0.35,
            "smoke_conf_threshold": 0.40,
            "hand_radius_bh": 0.40,
            "confirm_seconds": 0.35,
            "max_observation_gap": 0.45,
            "cooldown_seconds": 6.0,
        })
        self.dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def _make_person_with_wrist(self, t: float, wrist_xy=(220.0, 180.0)):
        kpts = np.zeros((17, 3), dtype=np.float32)
        kpts[0] = [200.0, 120.0, 0.9]   # Mũi
        kpts[9] = [wrist_xy[0], wrist_xy[1], 0.9]  # Cổ tay trái
        track = self.manager.update_track(
            track_id=1,
            bbox=(160.0, 100.0, 240.0, 300.0),  # BH = 200
            foot_point=(200.0, 300.0),
            keypoints=kpts,
            current_time=t,
        )
        return track

    def test_person_hand_fire_ignition_triggers_alert_after_confirm(self):
        fire_det = [{
            "bbox": (215.0, 165.0, 240.0, 195.0),
            "center": (227.5, 180.0),
            "conf": 0.82,
            "cls_id": 0,
            "label": "FIRE",
            "vn_label": "LỬA",
        }]
        alerts = []
        for i in range(10):
            t = i * 0.1
            track = self._make_person_with_wrist(t)
            info = self.detector.process_all([track], t, self.dummy_frame, raw_detections=fire_det)
            if info["new_alerts"]:
                alerts.extend(info["new_alerts"])

        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["track_id"], 1)
        self.assertIn("châm lửa", alerts[0]["extra_info"])
        self.assertEqual(track.current_state, "FIRE_ALERT")

    def test_short_light_flash_does_not_trigger_false_alarm(self):
        fire_det = [{
            "bbox": (215.0, 165.0, 240.0, 195.0),
            "center": (227.5, 180.0),
            "conf": 0.75,
            "cls_id": 0,
            "label": "FIRE",
            "vn_label": "LỬA",
        }]
        for t in (0.0, 0.1, 0.2):
            track = self._make_person_with_wrist(t)
            info = self.detector.process_all([track], t, self.dummy_frame, raw_detections=fire_det)
            self.assertEqual(len(info["new_alerts"]), 0)

        # Sang 0.8s đốm sáng biến mất -> reset và không cảnh báo
        track = self._make_person_with_wrist(0.8)
        info = self.detector.process_all([track], 0.8, self.dummy_frame, raw_detections=[])
        self.assertEqual(len(info["new_alerts"]), 0)
        self.assertEqual(track.current_state, "NORMAL")

    def test_scene_level_smoke_and_fire_triggers_early_warning(self):
        smoke_det = [{
            "bbox": (450.0, 40.0, 580.0, 160.0),
            "center": (515.0, 100.0),
            "conf": 0.78,
            "cls_id": 1,
            "label": "SMOKE",
            "vn_label": "KHÓI",
        }]
        alerts = []
        for i in range(8):
            t = i * 0.1
            info = self.detector.process_all([], t, self.dummy_frame, raw_detections=smoke_det)
            alerts.extend(info["new_alerts"])

        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["track_id"], 0)
        self.assertIn("Đám khói", alerts[0]["extra_info"])

    def test_white_halogen_lamp_glare_is_rejected_while_warm_fire_is_accepted(self):
        frame = np.full((360, 640, 3), 80, dtype=np.uint8)
        # 1. Bóng đèn pha trắng lạnh ở góc trái (tâm lóa trắng 255, quầng sáng hơi xanh B > R)
        frame[60:200, 70:180] = (195, 185, 175)  # BGR: B=195 > R=175 (warmth = -20)
        frame[80:160, 90:160] = (255, 255, 255)  # Tâm lóa trắng tinh
        gray = np.mean(frame, axis=2).astype(np.uint8)

        lamp_res = self.detector._verify_and_classify_region(
            frame, gray, gray, (70, 60, 180, 200), conf=0.45, cls_id=0
        )
        self.assertIsNone(lamp_res)

        # 2. Ngọn lửa thật ở thùng rác giữa phòng (quầng cam/đỏ R > B rõ rệt)
        frame[200:270, 555:615] = (70, 140, 210)  # BGR: R=210 > B=70 (warmth = +140)
        gray2 = np.mean(frame, axis=2).astype(np.uint8)
        fire_res = self.detector._verify_and_classify_region(
            frame, gray2, gray, (555, 200, 615, 270), conf=0.75, cls_id=0
        )
        self.assertEqual(fire_res, (0, 0.75))


class FireSmokeToggleTests(unittest.TestCase):
    def make_pipeline(self, model_exists):
        import threading
        from unittest.mock import patch
        from src.core.pipeline import SurveillancePipeline
        from src.tracking.track_manager import TrackManager
        pipe = SurveillancePipeline.__new__(SurveillancePipeline)
        pipe.behav_cfg = {"fire_smoke": {"model_path": "weights/x.pt"}}
        pipe.fire_yolo_model = None
        pipe._fire_lock = threading.Lock()
        pipe.track_manager = TrackManager()
        pipe.fire_smoke_detector = FireSmokeDetector({"enabled": False})
        self.exists = patch("src.core.pipeline.os.path.exists", return_value=model_exists)
        self.yolo = patch("src.core.pipeline.YOLO", return_value="model")
        self.exists.start(); self.loader = self.yolo.start()
        self.addCleanup(self.exists.stop); self.addCleanup(self.yolo.stop)
        return pipe

    def test_enable_loads_model_once_and_disable_clears_fire_state(self):
        pipe = self.make_pipeline(model_exists=True)
        self.assertTrue(pipe.set_fire_smoke_enabled(True))
        self.assertTrue(pipe.fire_smoke_enabled)
        self.assertEqual(pipe.fire_smoke_detector.fire_model, "model")
        track = pipe.track_manager.get_or_create(1, 0.0)
        track.current_state = "FIRE_ALERT"
        self.assertFalse(pipe.set_fire_smoke_enabled(False))
        self.assertFalse(pipe.fire_smoke_enabled)
        self.assertEqual(track.current_state, "NORMAL")
        pipe.set_fire_smoke_enabled(True)
        self.loader.assert_called_once()

    def test_enable_without_weights_stays_off(self):
        pipe = self.make_pipeline(model_exists=False)
        self.assertFalse(pipe.set_fire_smoke_enabled(True))
        self.assertFalse(pipe.fire_smoke_detector.enabled)
        self.loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
