"""
Phát hiện hành vi CHẠY BẤT THƯỜNG cho từng người (track_id).

Kết hợp 3 tiêu chí:
  1. Tốc độ di chuyển (bắt buộc): V = quãng đường điểm chân / (chiều cao người * Δt), đơn vị BH/s.
     Chạy thẳng về phía / ra xa camera: tâm khung gần như đứng yên nhưng khung to/nhỏ dần.
     Theo mô hình pinhole, khoảng cách tính theo chiều cao người là D = f / h, nên tốc độ theo
     chiều sâu = |dD/dt| (hồi quy tuyến tính trong cửa sổ để khử nhiễu bbox).
     Nếu tốc độ chủ yếu đến từ chiều sâu thì BẮT BUỘC có pose giống dáng chạy (đi bộ về phía
     camera cũng làm khung to dần).
     Nếu cấu hình `homography` (ảnh -> mặt sàn, mét) thì đo bằng m/s trên mặt sàn,
     loại bỏ méo phối cảnh khi người ở xa hoặc chạy về phía camera.
  2. Chuyển động cơ thể (nên có): độ nghiêng thân, biên độ vung chân, biên độ vung tay.
     Thiếu pose hoặc pose không giống dáng chạy -> chỉ xác nhận khi tốc độ rất cao.
  3. Ngữ cảnh (bắt buộc): điểm chân nằm trong vùng giám sát và thời điểm nằm trong khung giờ
     cấu hình. Không cấu hình vùng/khung giờ = toàn bộ khung hình, mọi lúc.

Máy trạng thái theo từng track:
  NORMAL     -> SUSPECTED : tốc độ vượt ngưỡng trong ngữ cảnh cấm chạy
  SUSPECTED  -> CONFIRMED : duy trì dấu hiệu chạy đủ `confirm_seconds` (cho phép rớt tín hiệu ngắn)
  SUSPECTED  -> NORMAL    : hết dấu hiệu lâu hơn `drop_grace_seconds`
  CONFIRMED  -> COOLDOWN  : người dừng chạy (phát đúng 1 cảnh báo cho mỗi sự kiện)
  COOLDOWN   -> NORMAL    : hết `cooldown_seconds` kể từ lần cảnh báo
"""
import math
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.tracking.track_manager import TrackHistory

NORMAL, SUSPECTED, CONFIRMED, COOLDOWN = "NORMAL", "SUSPECTED", "CONFIRMED", "COOLDOWN"

# Chỉ số khớp theo chuẩn COCO
L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = 11, 12, 13, 14, 15, 16


def point_in_polygon(x: float, y: float, polygon: Sequence[Sequence[float]]) -> bool:
    """Ray casting: điểm (x, y) có nằm trong đa giác hay không."""
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i][0], polygon[i][1]
        x2, y2 = polygon[(i + 1) % n][0], polygon[(i + 1) % n][1]
        if (y1 > y) != (y2 > y):
            if x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
                inside = not inside
    return inside


def _parse_hhmm(text: str) -> int:
    hh, mm = str(text).strip().split(":")
    return int(hh) * 60 + int(mm)


class RunningDetector:
    def __init__(self, config: Dict[str, Any], alert_manager=None,
                 clock: Optional[Callable[[], datetime]] = None):
        self.config = config or {}
        self.alert_manager = alert_manager
        self.clock = clock or datetime.now
        c = self.config
        self.enabled = c.get("enabled", True)

        # 1. Tốc độ
        self.v_thresh = float(c.get("v_norm_threshold", 1.3))
        self.strong_factor = float(c.get("strong_speed_factor", 1.5))
        self.window = float(c.get("window_seconds", 0.5))
        self.max_gap = float(c.get("max_observation_gap", 0.3))
        self.min_path_efficiency = float(c.get("min_path_efficiency", 0.5))
        h = c.get("homography")
        self.homography = np.asarray(h, dtype=np.float64).reshape(3, 3) if h else None
        self.mps_thresh = float(c.get("speed_mps_threshold", 2.5))
        self.depth_enabled = c.get("depth_speed", True)
        self.focal_px = c.get("focal_length_px")  # None -> focal_length_ratio x chiều rộng khung
        self.focal_ratio = float(c.get("focal_length_ratio", 0.6))  # camera giám sát góc rộng ~80°
        self.edge_margin = float(c.get("edge_margin_px", 3))
        self.depth_window = float(c.get("depth_window_seconds", 1.0))
        self.depth_min_r2 = float(c.get("depth_min_r2", 0.6))

        # 2. Pose
        self.body_lean_min = float(c.get("body_lean_min", 15.0))
        self.leg_swing_min = float(c.get("leg_swing_min", 0.25))
        self.arm_swing_min = float(c.get("arm_swing_min", 0.15))
        self.knee_lift_min = float(c.get("knee_lift_min", 0.08))
        self.arm_bent_max_deg = float(c.get("arm_bent_max_deg", 120.0))
        self.arm_bent_ratio = float(c.get("arm_bent_ratio", 0.6))
        self.pose_min_cues = int(c.get("pose_min_cues", 1))
        self.kpt_conf = float(c.get("keypoint_confidence", 0.3))

        # 3. Ngữ cảnh
        self.zones = [z for z in (c.get("zones") or []) if len(z.get("points") or []) >= 3]
        self.active_hours = [
            (_parse_hhmm(a), _parse_hhmm(b))
            for a, b in (str(s).split("-") for s in (c.get("active_hours") or []))
        ]

        # Máy trạng thái
        self.confirm_seconds = float(c.get("confirm_seconds", 0.15))
        self.drop_grace = float(c.get("drop_grace_seconds", 0.2))
        self.cooldown = float(c.get("cooldown_seconds", 4.0))

    # ------------------------------------------------------------------
    # Vòng xử lý chính
    # ------------------------------------------------------------------
    def process(self, track: TrackHistory, current_time: float, frame: Optional[np.ndarray],
                body_lean: Optional[float] = None) -> None:
        metrics = track.debug_metrics if isinstance(getattr(track, "debug_metrics", None), dict) else {}
        track.debug_metrics = metrics
        if track.current_state == "RUNNING_ALERT":
            track.current_state = "NORMAL"

        if not self.enabled or not math.isfinite(current_time):
            self.reset(track)
            return
        if track.run_last_time is not None:
            if current_time <= track.run_last_time:
                return
            if current_time - track.run_last_time > self.max_gap:
                track.run_fast_since = None  # mất dấu lâu: đo lại từ đầu
                if track.run_state == SUSPECTED:
                    track.run_state = NORMAL
        track.run_last_time = current_time

        frame_shape = frame.shape[:2] if frame is not None else None
        speed = self.compute_speed(track, frame_shape)
        in_context, ctx_label = self.check_context(track, frame_shape)
        pose_cues = self.pose_cues(track, body_lean)
        depth_dominant = speed is not None and track.run_depth_dominant
        indicator = self._is_running(speed, pose_cues, depth_dominant)

        if speed is not None:
            unit = "m/s" if self.homography is not None else "BH/s"
            tag = " sau" if depth_dominant else ""
            metrics["V"] = (f"V={speed:.2f}{unit}{tag} (>={self._threshold():.2f})", speed >= self._threshold())
        if pose_cues is not None:
            metrics["Pose"] = (f"Pose={sum(pose_cues.values())}/3 cues", sum(pose_cues.values()) >= self.pose_min_cues)
        metrics["Ctx"] = (f"Ctx={ctx_label}", in_context)

        running_now = indicator and in_context
        if running_now:
            track.run_last_fast_time = current_time
        recently_fast = (track.run_last_fast_time is not None
                         and 0 <= current_time - track.run_last_fast_time <= self.drop_grace)

        # Thời gian duy trì dấu hiệu chạy (chịu được rớt tín hiệu ngắn drop_grace)
        if not recently_fast:
            track.run_fast_since = None
        elif running_now and track.run_fast_since is None:
            track.run_fast_since = current_time
        sustained = (running_now and track.run_fast_since is not None
                     and current_time - track.run_fast_since + 1e-9 >= self.confirm_seconds)

        state = track.run_state
        if state == COOLDOWN:
            if track.run_alert_time is None or current_time - track.run_alert_time >= self.cooldown:
                state = NORMAL
            elif sustained:
                track.current_state = "RUNNING_ALERT"  # cùng sự kiện: hiển thị nhưng không báo lại

        if state == NORMAL and running_now:
            state = SUSPECTED
        elif state == SUSPECTED and not recently_fast:
            state = NORMAL

        if state == SUSPECTED and running_now:
            duration = current_time - track.run_fast_since
            metrics["Run"] = (f"Run={duration:.2f}/{self.confirm_seconds:.2f}s", True)
            if sustained:
                state = CONFIRMED
                track.run_alert_time = current_time
                self._alert(track, current_time, frame, speed, pose_cues, ctx_label)

        if state == CONFIRMED:
            if recently_fast:
                track.current_state = "RUNNING_ALERT"
            else:
                state = COOLDOWN

        track.run_state = state
        metrics["RunFSM"] = (f"Run:{state}", state in (SUSPECTED, CONFIRMED))

    def reset(self, track: TrackHistory) -> None:
        track.run_fast_since = None
        if track.run_state in (SUSPECTED, CONFIRMED):
            track.run_state = COOLDOWN if track.run_alert_time is not None else NORMAL

    # ------------------------------------------------------------------
    # Tiêu chí 1: tốc độ
    # ------------------------------------------------------------------
    def _threshold(self) -> float:
        return self.mps_thresh if self.homography is not None else self.v_thresh

    def _is_running(self, speed: Optional[float], cues: Optional[Dict[str, bool]],
                    depth_dominant: bool = False) -> bool:
        if speed is None:
            return False
        thresh = self._threshold()
        if speed < thresh:
            return False
        pose_ok = cues is not None and sum(cues.values()) >= self.pose_min_cues
        if depth_dominant:
            # Chỉ dựa vào khung to dần thì không đủ: phải thấy dáng chạy (tay/chân)
            return pose_ok
        # Pose là tiêu chí phụ: không có/không khớp thì cần tốc độ rõ ràng hơn
        return pose_ok or speed >= thresh * self.strong_factor

    def _segment(self, track: TrackHistory):
        """Đoạn quan sát liên tục cuối cùng (bỏ qua bbox lỗi, mất dấu, thời gian lùi)."""
        history = list(track.run_motion_history)
        start = 0
        for i, (t, box) in enumerate(history):
            if (not math.isfinite(t) or not np.isfinite(box).all()
                    or box[2] <= box[0] or box[3] <= box[1]):
                start = i + 1
            elif i and (t <= history[i - 1][0] or t - history[i - 1][0] > self.max_gap):
                start = i
        return history[start:]

    def compute_speed(self, track: TrackHistory, frame_shape: Optional[Tuple[int, int]] = None) -> Optional[float]:
        """
        Tốc độ trên cửa sổ `window_seconds`: BH/s (mặc định) hoặc m/s (khi có homography).
        Lấy max giữa tốc độ trên mặt phẳng ảnh và tốc độ 3D (ngang + chiều sâu từ cỡ khung).
        Trả về None khi chưa đủ dữ liệu; 0 khi chuyển động không nhất quán (rung camera, nhiễu bbox).
        """
        track.run_depth_dominant = False
        planar = self._planar_speed(track, frame_shape)
        if planar is None or self.homography is not None or not self.depth_enabled:
            return planar
        parts = self._depth_speed(track, frame_shape)
        if parts is None:
            return planar
        lateral, depth = parts
        # Di chuyển chủ yếu tiến/lùi so với camera (kể cả điểm chân trượt dọc ảnh do phối cảnh)
        track.run_depth_dominant = depth > lateral
        return max(planar, math.hypot(lateral, depth))

    def _depth_speed(self, track: TrackHistory, frame_shape) -> Optional[Tuple[float, float]]:
        """(tốc độ ngang, tốc độ theo chiều sâu) theo BH/s, hồi quy trên cửa sổ đo."""
        focal = self.focal_px or (frame_shape[1] * self.focal_ratio if frame_shape is not None else None)
        history = self._segment(track)
        if not focal or len(history) < 4:
            return None
        t_now = history[-1][0]
        # Cửa sổ dài hơn một chu kỳ bước chân: khung co lại khi nhấc chân rồi giãn ra khi tiếp đất
        j = next((i for i in range(len(history) - 1, -1, -1)
                  if t_now - history[i][0] + 1e-9 >= self.depth_window), None)
        if j is None:
            return None
        samples = history[j:]
        if len(samples) < 4:
            return None
        if frame_shape is not None:
            fh, fw = frame_shape
            m = self.edge_margin
            # Khung chạm mép ảnh -> người bị cắt, chiều cao không còn phản ánh khoảng cách
            if any(b[1] <= m or b[3] >= fh - m or b[0] <= m or b[2] >= fw - m for _, b in samples):
                return None
        ts = np.array([t for t, _ in samples], dtype=np.float64)
        hs = np.array([max(1.0, b[3] - b[1]) for _, b in samples], dtype=np.float64)
        xs = np.array([(b[0] + b[2]) / 2.0 for _, b in samples], dtype=np.float64)
        ts = ts - ts.mean()
        denom = float((ts ** 2).sum())
        if denom <= 0:
            return None
        depth = focal / hs                       # khoảng cách tới camera (đơn vị chiều cao người)
        lateral = xs / float(np.median(hs))      # vị trí ngang (đơn vị chiều cao người)
        slope_d = float((ts * (depth - depth.mean())).sum() / denom)
        slope_x = float((ts * (lateral - lateral.mean())).sum() / denom)
        # Chiều sâu chỉ dao động (rung camera, che khuất, khung giật) -> đường thẳng khớp kém
        ss_tot = float(((depth - depth.mean()) ** 2).sum())
        ss_res = float(((depth - depth.mean() - slope_d * ts) ** 2).sum())
        if ss_tot <= 0 or 1.0 - ss_res / ss_tot < self.depth_min_r2:
            slope_d = 0.0
        return abs(slope_x), abs(slope_d)

    def _planar_speed(self, track: TrackHistory, frame_shape: Optional[Tuple[int, int]] = None) -> Optional[float]:
        history = self._segment(track)
        n = len(history)
        if n < 4:
            return None
        ts = [t for t, _ in history]
        bb = [b for _, b in history]
        t_now = ts[-1]
        j = next((i for i in range(n - 1, -1, -1) if t_now - ts[i] + 1e-9 >= self.window), None)
        if j is None:
            return None

        # Ưu tiên điểm chân (tiếp đất). Bbox chạm mép dưới khung hình -> chân bị cắt, dùng tâm.
        use_center = frame_shape is not None and any(b[3] >= frame_shape[0] - 2 for b in bb[j:])

        def ref_point(b):
            x = (b[0] + b[2]) / 2.0
            y = (b[1] + b[3]) / 2.0 if use_center else b[3]
            return self._to_ground(x, y)

        pts = [ref_point(b) for b in bb]

        def mean_point(lo, hi):
            lo, hi = max(0, lo), min(n, hi)
            xs, ys = zip(*pts[lo:hi])
            return sum(xs) / len(xs), sum(ys) / len(ys), sum(ts[lo:hi]) / (hi - lo)

        ox, oy, ot = mean_point(j - 1, j + 2)
        nx, ny, nt = mean_point(n - 3, n)
        dt = nt - ot
        if not math.isfinite(dt) or dt <= 0:
            return None
        net = math.hypot(nx - ox, ny - oy)

        # Rung camera / bbox giật: quãng đường tích lũy lớn nhưng dịch chuyển thực nhỏ
        if self.min_path_efficiency > 0 and net > 0:
            step = max(self.window / 5.0, 1e-6)
            path, last_t, last_p = 0.0, ts[j], pts[j]
            for t, p in zip(ts[j + 1:], pts[j + 1:]):
                if t - last_t + 1e-9 >= step:
                    path += math.hypot(p[0] - last_p[0], p[1] - last_p[1])
                    last_t, last_p = t, p
            path += math.hypot(pts[-1][0] - last_p[0], pts[-1][1] - last_p[1])
            if path > 0 and net / path < self.min_path_efficiency:
                return 0.0

        if self.homography is not None:
            return net / dt
        heights = sorted(max(1.0, b[3] - b[1]) for b in bb[j:])
        return net / (dt * heights[len(heights) // 2])

    def _to_ground(self, x: float, y: float) -> Tuple[float, float]:
        if self.homography is None:
            return x, y
        gx, gy, w = self.homography @ np.array([x, y, 1.0])
        if abs(w) < 1e-12:
            return x, y
        return float(gx / w), float(gy / w)

    # ------------------------------------------------------------------
    # Tiêu chí 2: chuyển động cơ thể
    # ------------------------------------------------------------------
    def pose_cues(self, track: TrackHistory, body_lean: Optional[float]) -> Optional[Dict[str, bool]]:
        """Các dấu hiệu dáng chạy trên các pose quan sát thật trong cửa sổ gần nhất."""
        pairs = [
            (k, b) for k, b, seen in zip(track.keypoints_history, track.keypoint_bbox_history,
                                          track.keypoint_observed_history) if seen
        ][-15:]
        if len(pairs) < 3:
            return None

        def series(i, j, axis=0):
            out = []
            for k, b in pairs:
                if k.shape[1] > 2 and (k[i, 2] < self.kpt_conf or k[j, 2] < self.kpt_conf):
                    continue
                h = max(1.0, b[3] - b[1])
                out.append((k[i, axis] - k[j, axis]) / h)
            return out

        def amplitude(values):
            return max(values) - min(values) if len(values) >= 3 else 0.0

        # Nhìn ngang: sải chân/tay theo phương x. Nhìn chính diện: nâng gối, vung tay theo phương y.
        stride = amplitude(series(L_ANKLE, R_ANKLE))
        knee_lift = max(amplitude(series(L_KNEE, R_KNEE, axis=1)),
                        amplitude(series(L_ANKLE, R_ANKLE, axis=1)))
        arms = max(amplitude(series(w, sh, axis=axis))
                   for w, sh in ((L_WRIST, L_SHOULDER), (R_WRIST, R_SHOULDER)) for axis in (0, 1))
        return {
            "lean": body_lean is not None and body_lean >= self.body_lean_min,
            "legs": stride >= self.leg_swing_min or knee_lift >= self.knee_lift_min,
            # Vung tay rộng, hoặc khuỷu tay gập ~90° (tư thế chạy; đi bộ tay thường duỗi)
            "arms": arms >= self.arm_swing_min or self._arms_bent_ratio(pairs) >= self.arm_bent_ratio,
        }

    def _arms_bent_ratio(self, pairs) -> float:
        """Tỷ lệ frame có ít nhất một khuỷu tay gập (góc vai-khuỷu-cổ tay trên ảnh nhỏ)."""
        seen = bent = 0
        for k, _ in pairs:
            frame_bent = frame_seen = False
            for s, e, w in ((L_SHOULDER, L_ELBOW, L_WRIST), (R_SHOULDER, R_ELBOW, R_WRIST)):
                if k.shape[1] > 2 and min(k[s, 2], k[e, 2], k[w, 2]) < self.kpt_conf:
                    continue
                v1, v2 = k[s, :2] - k[e, :2], k[w, :2] - k[e, :2]
                n1, n2 = float(np.hypot(*v1)), float(np.hypot(*v2))
                if n1 < 1e-6 or n2 < 1e-6:
                    continue
                frame_seen = True
                cos = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
                frame_bent |= math.degrees(math.acos(cos)) <= self.arm_bent_max_deg
            seen += frame_seen
            bent += frame_bent
        return bent / seen if seen >= 3 else 0.0

    # ------------------------------------------------------------------
    # Tiêu chí 3: ngữ cảnh
    # ------------------------------------------------------------------
    def check_context(self, track: TrackHistory, frame_shape: Optional[Tuple[int, int]]) -> Tuple[bool, str]:
        if self.active_hours:
            now = self.clock()
            minute = now.hour * 60 + now.minute
            if not any((a <= minute < b) if a <= b else (minute >= a or minute < b)
                       for a, b in self.active_hours):
                return False, "ngoai gio"
        if not self.zones:
            return True, "toan khung"
        if not track.bbox_history:
            return False, "?"
        x1, _, x2, y2 = track.bbox_history[-1]
        fx, fy = (x1 + x2) / 2.0, y2
        for zone in self.zones:
            pts = zone["points"]
            if zone.get("normalized", False):
                if frame_shape is None:
                    continue
                h, w = frame_shape
                pts = [(p[0] * w, p[1] * h) for p in pts]
            if point_in_polygon(fx, fy, pts):
                return True, str(zone.get("name", "vung"))
        return False, "ngoai vung"

    # ------------------------------------------------------------------
    def _alert(self, track, current_time, frame, speed, cues, ctx_label):
        if not track.can_trigger_alert("RUNNING", current_time, self.cooldown):
            return
        if not self.alert_manager:
            return
        unit = "m/s" if self.homography is not None else "BH/s"
        cue_text = ",".join(k for k, v in (cues or {}).items() if v) or "khong co pose"
        self.alert_manager.trigger_alert(
            event_type="RUNNING",
            track_id=track.track_id,
            frame=frame,
            confidence=float(min(1.0, speed / (self._threshold() * 1.7))),
            extra_info=f"Chạy nhanh bất thường: {speed:.1f} {unit} | pose: {cue_text} | vùng: {ctx_label}",
        )
