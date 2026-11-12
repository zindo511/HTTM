from collections import deque
import math
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np
from src.tracking.track_manager import TrackHistory


class FireSmokeDetector:
    """
    Phát hiện sớm Khói & Lửa (Smoke & Fire Detection) kết hợp phân tích hành vi Châm lửa:
    1. Chạy mô hình YOLOv8n (`weights/fire_smoke_yolov8n.pt`) nhận diện 2 lớp: `0: fire` (Lửa), `1: smoke` (Khói).
    2. Tầng kiểm định vật lý (Chromaticity Halo + Temporal Flicker/Plume):
       - Loại bỏ hoàn toàn bóng đèn pha trắng / đèn huỳnh quang cố định (tâm lóa trắng `RGB > 235` nhưng quầng sáng lạnh `R - B < 4.0`).
       - Phát hiện sớm cột khói xám bốc lên (`motion_diff >= 4.5`, độ bão hòa thấp `S < 80`, không phải bóng đèn lóa).
    3. Liên kết với 17 khớp xương người (`YOLOv8n-Pose`):
       - Nếu lửa/khói nằm gần khớp Cổ tay (9, 10) hoặc Mặt (0) -> Cảnh báo hành vi Châm lửa / Tạo khói gắn với Track ID.
       - Nếu lửa/khói xuất hiện độc lập ở không gian rộng / ngoài trời -> Cảnh báo sớm đám cháy / khói môi trường.
    """
    CLASS_NAMES = {0: "FIRE", 1: "SMOKE"}
    CLASS_VN = {0: "LỬA", 1: "KHÓI"}

    def __init__(self, config: Dict[str, Any], fire_model=None, alert_manager=None):
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)
        self.fire_model = fire_model
        self.alert_manager = alert_manager

        self.conf_thresh = float(self.config.get("conf_threshold", 0.35))
        self.smoke_conf_thresh = float(self.config.get("smoke_conf_threshold", 0.38))
        self.plume_conf_thresh = float(self.config.get("plume_conf_threshold", 0.16))
        self.hand_radius_bh = float(self.config.get("hand_radius_bh", 0.40))
        self.confirm_seconds = float(self.config.get("confirm_seconds", 0.35))
        self.max_gap_seconds = float(self.config.get("max_observation_gap", 0.45))
        self.cooldown_seconds = float(self.config.get("cooldown_seconds", 6.0))
        self.infer_every_n_frames = max(1, int(self.config.get("infer_every_n_frames", 2)))
        self.imgsz = int(self.config.get("imgsz", 640))

        self.reset()

    def reset(self):
        self._frame_counter = 0
        self._cached_detections: List[Dict[str, Any]] = []
        self._gray_history: deque = deque(maxlen=8)
        self._last_time: Optional[float] = None
        self._scene_active_since: Optional[float] = None
        self._scene_last_seen: Optional[float] = None
        self._scene_last_alert: float = -1e9
        self._scene_last_kind: Optional[str] = None

    def _verify_and_classify_region(
        self,
        frame: np.ndarray,
        gray: np.ndarray,
        ref_gray: np.ndarray,
        box: Tuple[float, float, float, float],
        conf: float,
        cls_id: int,
    ) -> Optional[Tuple[int, float]]:
        """
        Kiểm định vật lý vùng ứng viên Lửa / Khói:
        - Loại bỏ bóng đèn pha trắng / đèn LED cố định (quầng sáng lạnh R - B < 4.0 hoặc lóa trắng tĩnh).
        - Chuyển vùng khói bốc lên sớm (cột khói xám chuyển động, không lóa đèn) sang lớp SMOKE (1).
        """
        h, w = frame.shape[:2]
        x1 = max(0, min(w - 1, int(box[0])))
        y1 = max(0, min(h - 1, int(box[1])))
        x2 = max(x1 + 1, min(w, int(box[2])))
        y2 = max(y1 + 1, min(h, int(box[3])))

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        s_mean = float(hsv[:, :, 1].mean())
        v_max = crop.max(axis=2)

        # Tỷ lệ điểm ảnh lóa trắng tinh (bóng đèn pha / đèn huỳnh quang)
        white_glare_ratio = float(
            ((crop[:, :, 0] > 235) & (crop[:, :, 1] > 235) & (crop[:, :, 2] > 235)).mean()
        )

        # Độ ấm màu (Red - Blue) ở vùng quầng sáng không bị bão hòa trắng (70 < V < 245)
        halo_pixels = crop[(v_max > 70) & (v_max < 245)]
        if len(halo_pixels) > 0:
            halo_warmth = float(
                (halo_pixels[:, 2].astype(np.float32) - halo_pixels[:, 0].astype(np.float32)).mean()
            )
        else:
            halo_warmth = float(crop[:, :, 2].mean()) - float(crop[:, :, 0].mean())

        # Độ biến thiên mức xám theo thời gian (~0.4-0.5s trước)
        motion_diff = float(cv2.absdiff(gray[y1:y2, x1:x2], ref_gray[y1:y2, x1:x2]).mean())

        if cls_id == 0:  # Ứng viên FIRE
            if halo_warmth < 4.0:
                # Quầng sáng lạnh (trắng/xanh): nếu có tâm lóa trắng -> chắc chắn là bóng đèn chiếu sáng!
                if white_glare_ratio >= 0.06:
                    return None
                # Nếu không lóa trắng mà có chuyển động bốc lên của đám xám mờ -> Khói sớm (SMOKE)
                if motion_diff >= 4.8 and s_mean < 80.0 and conf >= self.plume_conf_thresh:
                    return 1, max(conf, 0.45)
                return None

            # Quầng sáng ấm (R - B >= 4.0): kiểm tra ngưỡng tin cậy của Lửa
            if conf < self.conf_thresh:
                return None
            return 0, conf

        # Ứng viên SMOKE (cls_id == 1)
        if white_glare_ratio >= 0.12 and halo_warmth < 2.0:
            # Loại bỏ trường hợp quầng lóa bóng đèn bị nhầm thành khói
            return None
        if conf < self.smoke_conf_thresh:
            return None
        return 1, conf

    def _detect(self, frame: np.ndarray, device: str = "cpu") -> List[Dict[str, Any]]:
        if self.fire_model is None or frame is None:
            return []

        self._frame_counter += 1
        if self._frame_counter % self.infer_every_n_frames != 1 and self._cached_detections:
            return self._cached_detections

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ref_gray = self._gray_history[0] if self._gray_history else gray
        self._gray_history.append(gray)

        results = self.fire_model.predict(
            source=frame,
            conf=min(self.plume_conf_thresh, self.conf_thresh, self.smoke_conf_thresh),
            imgsz=self.imgsz,
            device=device,
            verbose=False
        )[0]

        detections: List[Dict[str, Any]] = []
        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            clss = results.boxes.cls.int().cpu().numpy()

            for box, conf, cls_id in zip(boxes, confs, clss):
                x1, y1, x2, y2 = map(float, box)
                if x2 <= x1 or y2 <= y1:
                    continue
                verified = self._verify_and_classify_region(
                    frame, gray, ref_gray, (x1, y1, x2, y2), float(conf), int(cls_id)
                )
                if verified is None:
                    continue
                final_cls, final_conf = verified
                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "center": ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                    "conf": final_conf,
                    "cls_id": final_cls,
                    "label": self.CLASS_NAMES.get(final_cls, "FIRE"),
                    "vn_label": self.CLASS_VN.get(final_cls, "LỬA"),
                })

        self._cached_detections = detections
        return detections

    @staticmethod
    def _extract_person_anchors(track: TrackHistory) -> List[Tuple[float, float, str]]:
        """Lấy các điểm neo Cổ tay trái (9), Cổ tay phải (10) và Mũi/Mặt (0) nếu đủ độ tự tin."""
        anchors: List[Tuple[float, float, str]] = []
        if not track.keypoints_history:
            return anchors
        kpts = track.keypoints_history[-1]
        if kpts is None or len(kpts) < 11:
            return anchors
        for idx, name in ((9, "tay trái"), (10, "tay phải"), (0, "vùng mặt")):
            x, y = float(kpts[idx, 0]), float(kpts[idx, 1])
            conf = float(kpts[idx, 2]) if kpts.shape[1] > 2 else 1.0
            if conf >= 0.25 and x > 0 and y > 0:
                anchors.append((x, y, name))
        return anchors

    def process_all(
        self,
        tracks: List[TrackHistory],
        current_time: float,
        frame: np.ndarray,
        device: str = "cpu",
        raw_detections: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "is_active": False,
            "detections": [],
            "links": [],
            "new_alerts": [],
        }
        if not self.enabled or not math.isfinite(current_time):
            return info

        if self._last_time is not None and current_time < self._last_time:
            self.reset()
        self._last_time = current_time

        detections = (
            raw_detections
            if raw_detections is not None
            else self._detect(frame, device=device)
        )
        info["detections"] = detections

        matched_det_indices = set()

        # 1. Kiểm tra tương quan giữa từng người (TrackHistory) và vùng Lửa / Khói
        for track in tracks:
            if getattr(track, "current_state", "NORMAL") == "FIRE_ALERT":
                track.current_state = "NORMAL"

            last_seen = getattr(track, "fire_last_seen", None)
            if last_seen is not None and current_time - last_seen > self.max_gap_seconds:
                track.fire_active_since = None

            if not track.bbox_history:
                continue

            px1, py1, px2, py2 = track.bbox_history[-1]
            bh = max(1.0, float(py2 - py1))
            anchors = self._extract_person_anchors(track)

            best_match = None
            best_dist_bh = float("inf")

            for d_idx, det in enumerate(detections):
                fx, fy = det["center"]
                # Kiểm tra khoảng cách tới bàn tay / mặt
                for ax, ay, anchor_name in anchors:
                    dist_bh = math.hypot(fx - ax, fy - ay) / bh
                    if dist_bh <= self.hand_radius_bh and dist_bh < best_dist_bh:
                        best_dist_bh = dist_bh
                        best_match = (d_idx, det, (ax, ay), anchor_name)

                # Nếu khuất khớp cổ tay, kiểm tra đốm lửa nằm ngay trong vùng thân người
                if best_match is None:
                    margin = 0.12 * bh
                    if (px1 - margin <= fx <= px2 + margin) and (py1 - margin <= fy <= py2 + margin):
                        person_center = ((px1 + px2) / 2.0, (py1 + py2) / 2.0)
                        best_match = (d_idx, det, person_center, "cơ thể")

            if best_match is not None:
                d_idx, det, anchor_pt, anchor_name = best_match
                matched_det_indices.add(d_idx)
                track.fire_last_seen = current_time
                if getattr(track, "fire_active_since", None) is None:
                    track.fire_active_since = current_time

                duration = current_time - track.fire_active_since
                if not hasattr(track, "debug_metrics") or track.debug_metrics is None:
                    track.debug_metrics = {}
                track.debug_metrics["Fire"] = (
                    f"{det['label']}={duration:.2f}/{self.confirm_seconds:.2f}s",
                    duration + 1e-9 >= self.confirm_seconds,
                )
                info["links"].append((anchor_pt, det["center"], det["vn_label"]))

                if duration + 1e-9 >= self.confirm_seconds:
                    info["is_active"] = True
                    if getattr(track, "fall_phase", "NORMAL") != "FALL_LATCHED":
                        track.current_state = "FIRE_ALERT"
                    if track.can_trigger_alert("FIRE_SMOKE", current_time, self.cooldown_seconds):
                        action_desc = (
                            f"Phát hiện hành vi châm lửa ({anchor_name})"
                            if det["cls_id"] == 0
                            else f"Phát hiện khói bốc lên gần người ({anchor_name})"
                        )
                        info["new_alerts"].append({
                            "track_id": track.track_id,
                            "confidence": det["conf"],
                            "extra_info": action_desc,
                        })
            else:
                if last_seen is None or current_time - last_seen > self.max_gap_seconds:
                    track.fire_active_since = None

        # 2. Kiểm tra Lửa / Khói độc lập trong không gian rộng / ngoài trời (không gắn với tay người)
        unmatched = [det for idx, det in enumerate(detections) if idx not in matched_det_indices]
        if unmatched:
            if self._scene_last_seen is not None and current_time - self._scene_last_seen > self.max_gap_seconds:
                self._scene_active_since = None
                self._scene_last_kind = None
            self._scene_last_seen = current_time
            if self._scene_active_since is None:
                self._scene_active_since = current_time

            scene_duration = current_time - self._scene_active_since
            if scene_duration + 1e-9 >= self.confirm_seconds:
                info["is_active"] = True
                has_fire = any(d["cls_id"] == 0 for d in unmatched)
                has_smoke = any(d["cls_id"] == 1 for d in unmatched)
                current_kind = "BOTH" if (has_fire and has_smoke) else ("FIRE" if has_fire else "SMOKE")
                # Cho phép báo động ngay khi chuyển cấp độ từ Khói (SMOKE) sang Bùng lửa (FIRE)
                escalated = (self._scene_last_kind == "SMOKE" and has_fire)
                if escalated or (current_time - self._scene_last_alert >= self.cooldown_seconds):
                    self._scene_last_alert = current_time
                    self._scene_last_kind = current_kind
                    top_det = max(unmatched, key=lambda d: d["conf"])
                    kind_str = "Lửa và Khói" if (has_fire and has_smoke) else ("Ngọn lửa" if has_fire else "Đám khói")
                    info["new_alerts"].append({
                        "track_id": 0,
                        "confidence": top_det["conf"],
                        "extra_info": f"Cảnh báo sớm: Phát hiện {kind_str} trong khu vực giám sát!",
                    })
        else:
            if self._scene_last_seen is not None and current_time - self._scene_last_seen > self.max_gap_seconds:
                self._scene_active_since = None
                self._scene_last_kind = None

        return info
