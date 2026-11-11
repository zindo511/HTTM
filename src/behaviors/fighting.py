import math
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from src.tracking.track_manager import TrackHistory


# Các khớp tứ chi dùng để đo vận tốc ra đòn (COCO 17 Keypoints):
# 7: Khuỷu trái, 8: Khuỷu phải, 9: Cổ tay trái, 10: Cổ tay phải, 15: Cổ chân trái, 16: Cổ chân phải
LIMB_KEYPOINTS = (7, 8, 9, 10, 15, 16)
WRIST_ELBOW_KEYPOINTS = (7, 8, 9, 10)
ANKLE_KEYPOINTS = (15, 16)
HEAD_KEYPOINTS = (0, 1, 2, 3, 4)


class FightingDetector:
    """
    Module 4: Phát hiện hành vi Ẩu đả / Đánh nhau (Multi-Cue Human-like Fight Perception).
    Kết hợp 3 nhóm dấu hiệu như cảm nhận của mắt người thật:
    1. Đòn đánh tứ chi (Limb Strike): Vung tay/chân nhanh (V_limb >= ngưỡng) xâm phạm vùng Đầu/Thân đối phương.
    2. Vật lộn áp sát (Grappling / Clinch): Hai người chồng lấn sâu (IoU cao hoặc D_norm <= 0.68) kèm giật lắc thân mạnh
       (bắt được cả khi cổ tay bị che khuất lúc ôm vật nhau).
    3. Lao vào va chạm (Collision / Rush): Vận tốc áp sát vào nhau V_closing lớn kèm khoảng cách sát.
    """
    def __init__(self, config: Dict[str, Any], alert_manager=None):
        self.config = config
        self.alert_manager = alert_manager
        self.enabled = config.get("enabled", True)

        self.max_distance_bh = config.get("max_distance_bh", 1.15)
        self.limb_speed_thresh = config.get("limb_speed_threshold", 2.2)
        self.confirm_frames = config.get("confirm_frames", 4)
        self.confirm_seconds = config.get("confirm_seconds")
        self.pair_candidate_since = {}
        self.cooldown = config.get("cooldown_seconds", 6.0)
        self.kpt_conf_thresh = config.get("kpt_conf_threshold", 0.3)
        self.max_height_ratio = config.get("max_height_ratio", 2.0)

        self.hold_seconds = config.get("hold_seconds", 1.5)
        self.pair_alert_until = {}
        self.pair_visuals = {}

        self.pair_streaks: Dict[Tuple[int, int], int] = {}
        self.pair_last_seen: Dict[Tuple[int, int], float] = {}
        self.pair_prev_dist: Dict[Tuple[int, int], Tuple[float, float]] = {}

        # Hai người cùng di chuyển nhanh (chạy bộ cạnh nhau / chạy lướt qua nhau) không phải ẩu đả.
        self.pass_by_speed_bh = config.get("pass_by_speed_bh", 1.0)
        self.body_window = config.get("body_window_seconds", 0.5)

    def process_all(
        self,
        active_tracks: List[TrackHistory],
        current_time: float,
        frame: np.ndarray
    ) -> Dict[str, Any]:
        result = {"is_fighting": False, "links": [], "conflict_boxes": [], "new_alerts": []}
        if not self.enabled:
            return result

        for t in active_tracks:
            if t.current_state in ("FIGHTING_ALERT", "FIGHTING_SUSPECT"):
                t.current_state = "NORMAL"

        n = len(active_tracks)
        limb_speeds: Dict[int, float] = {}
        torso_agitation: Dict[int, float] = {}
        body_speeds: Dict[int, float] = {}
        for t in active_tracks:
            limb_speeds[t.track_id] = self._compute_limb_speed(t)
            torso_agitation[t.track_id] = self._compute_torso_agitation(t)
            body_speeds[t.track_id] = self._compute_body_speed(t)

        for i in range(n):
            for j in range(i + 1, n):
                ta = active_tracks[i]
                tb = active_tracks[j]
                if len(ta.bbox_history) == 0 or len(tb.bbox_history) == 0:
                    continue

                pair_key = (min(ta.track_id, tb.track_id), max(ta.track_id, tb.track_id))
                last_seen = self.pair_last_seen.get(pair_key)
                if last_seen is not None and current_time - last_seen > 0.3:
                    self.pair_streaks[pair_key] = 0
                    self.pair_prev_dist.pop(pair_key, None)
                    self.pair_candidate_since.pop(pair_key, None)
                self.pair_last_seen[pair_key] = current_time

                box_a = ta.bbox_history[-1]
                box_b = tb.bbox_history[-1]

                ca = ((box_a[0] + box_a[2]) / 2.0, (box_a[1] + box_a[3]) / 2.0)
                cb = ((box_b[0] + box_b[2]) / 2.0, (box_b[1] + box_b[3]) / 2.0)
                ha = max(1.0, box_a[3] - box_a[1])
                hb = max(1.0, box_b[3] - box_b[1])
                # Foreground and distant background people can overlap in 2D
                # without interacting. Reject that perspective mismatch.
                both_upright = (ha / max(1.0, box_a[2] - box_a[0]) > 1.2 and
                                hb / max(1.0, box_b[2] - box_b[0]) > 1.2)
                if both_upright and max(ha, hb) / min(ha, hb) > self.max_height_ratio:
                    self.pair_streaks[pair_key] = 0
                    self.pair_candidate_since.pop(pair_key, None)
                    self.pair_prev_dist.pop(pair_key, None)
                    continue
                avg_h = (ha + hb) / 2.0

                center_dist = math.hypot(ca[0] - cb[0], ca[1] - cb[1])
                d_norm = center_dist / avg_h
                iou = self._compute_iou(box_a, box_b)
                is_close = (d_norm <= self.max_distance_bh) or (iou >= 0.05)

                # Tính vận tốc lao vào nhau (Closing velocity V_close)
                v_closing = 0.0
                if pair_key in self.pair_prev_dist:
                    prev_d, prev_t = self.pair_prev_dist[pair_key]
                    dt_pair = current_time - prev_t
                    if 0.03 <= dt_pair <= 0.6:
                        v_closing = max(0.0, (prev_d - d_norm) / dt_pair)
                self.pair_prev_dist[pair_key] = (d_norm, current_time)

                va = limb_speeds.get(ta.track_id, 0.0)
                vb = limb_speeds.get(tb.track_id, 0.0)
                max_v_limb = max(va, vb)

                agit_a = torso_agitation.get(ta.track_id, 0.0)
                agit_b = torso_agitation.get(tb.track_id, 0.0)
                max_agit = max(agit_a, agit_b)

                # Kiểm tra vùng chạm đòn (Đầu / Thân / Đá chân)
                strike_zone_a = self._check_strike_zone(ta, tb, box_b, avg_h)
                strike_zone_b = self._check_strike_zone(tb, ta, box_a, avg_h)
                strike_zone = strike_zone_a or strike_zone_b
                is_reaching = (strike_zone is not None) or (iou >= 0.05) or (d_norm <= 0.80)

                # --- 3 NHÓM DẤU HIỆU NHẬN BIẾT ẨU ĐẢ ---
                # 1. Ra đòn tay/chân tốc độ cao (Limb Strike)
                cue_strike = is_close and is_reaching and (max_v_limb >= self.limb_speed_thresh)
                # 2. Ôm vật / giằng co sát người (Grappling / Clinch - bắt cả khi khuất cổ tay)
                cue_grapple = ((iou >= 0.14 or d_norm <= 0.62) and max_agit >= 1.15 and max_v_limb >= self.limb_speed_thresh * 0.6)
                # 3. Lao thẳng vào va chạm mạnh (Aggressive Charge)
                cue_charge = (d_norm <= 0.85 and v_closing >= 1.35 and max_v_limb >= self.limb_speed_thresh * 0.75)

                # Cả hai đều đang di chuyển nhanh => chạy cùng nhau / lướt qua nhau, không phải đánh nhau.
                # (Lao vào đánh: nạn nhân đứng yên nên không bị loại.)
                passing_by = min(body_speeds.get(ta.track_id, 0.0),
                                 body_speeds.get(tb.track_id, 0.0)) >= self.pass_by_speed_bh

                is_fighting_now = (cue_strike or cue_grapple or cue_charge) and not passing_by

                prev_streak = self.pair_streaks.get(pair_key, 0)
                if is_fighting_now:
                    self.pair_candidate_since.setdefault(pair_key, current_time)
                    # Nếu đòn đánh cực mạnh vào vùng đầu hoặc lao vào cực nhanh thì tăng nhanh bộ đếm
                    step = 2 if (strike_zone == "HEAD" and max_v_limb >= self.limb_speed_thresh * 1.35) else 1
                    streak = min(self.confirm_frames, prev_streak + step)
                else:
                    self.pair_candidate_since.pop(pair_key, None)
                    streak = max(0, prev_streak - 1)
                self.pair_streaks[pair_key] = streak
                candidate_seconds = current_time - self.pair_candidate_since.get(pair_key, current_time)
                confirmed = (streak >= self.confirm_frames if self.confirm_seconds is None else
                             is_fighting_now and candidate_seconds >= self.confirm_seconds)

                if d_norm <= self.max_distance_bh * 1.4:
                    for t, v_self in ((ta, va), (tb, vb)):
                        if not hasattr(t, "debug_metrics") or t.debug_metrics is None:
                            t.debug_metrics = {}
                        t.debug_metrics["Dist"] = (f"Dist={d_norm:.2f} (<={self.max_distance_bh})", is_close)
                        t.debug_metrics["V_limb"] = (f"V_arm={v_self:.2f} (>={self.limb_speed_thresh})", v_self >= self.limb_speed_thresh)
                        progress = (f"Fight={streak}/{self.confirm_frames}" if self.confirm_seconds is None else
                                    f"Fight={candidate_seconds:.2f}/{self.confirm_seconds:.2f}s")
                        t.debug_metrics["Fight"] = (progress, confirmed)

                if is_fighting_now and confirmed:
                    self.pair_alert_until[pair_key] = current_time + self.hold_seconds
                held = current_time < self.pair_alert_until.get(pair_key, float("-inf"))
                if streak > 0 and not held:
                    for track in (ta, tb):
                        if track.current_state not in ("FIGHTING_ALERT", "FALLING_ALERT"):
                            track.current_state = "FIGHTING_SUSPECT"
                if held:
                    self.pair_visuals[pair_key] = ((ca, cb), (
                        min(box_a[0], box_b[0]), min(box_a[1], box_b[1]),
                        max(box_a[2], box_b[2]), max(box_a[3], box_b[3]), "AU DA"
                    ))

                    primary_track = ta if ta.track_id < tb.track_id else tb
                    if is_fighting_now and primary_track.can_trigger_alert("FIGHTING", current_time, self.cooldown):
                        conf = min(0.99, max(0.80, max_v_limb / (self.limb_speed_thresh * 1.4)))
                        desc = self._build_human_description(
                            ta.track_id, tb.track_id, strike_zone, cue_grapple, cue_charge, max_v_limb
                        )
                        result["new_alerts"].append({
                            "track_id": primary_track.track_id,
                            "confidence": conf, "extra_info": desc,
                        })

        # Keep the last conflict region through a brief tracking dropout too.
        tracks_by_id = {t.track_id: t for t in active_tracks}
        for pair, until in self.pair_alert_until.items():
            if current_time >= until:
                continue
            result["is_fighting"] = True
            link, box = self.pair_visuals[pair]
            result["links"].append(link)
            result["conflict_boxes"].append(box)
            for tid in pair:
                track = tracks_by_id.get(tid)
                if track is not None and track.current_state != "FALLING_ALERT":
                    track.current_state = "FIGHTING_ALERT"
        self._cleanup_stale_pairs(current_time)
        return result

    @staticmethod
    def _build_human_description(
        id_a: int,
        id_b: int,
        strike_zone: Optional[str],
        cue_grapple: bool,
        cue_charge: bool,
        max_v_limb: float
    ) -> str:
        """Sinh câu mô tả tự nhiên như người quan sát trực tiếp."""
        if strike_zone == "HEAD":
            action = "vung đòn tấn công vùng đầu"
        elif strike_zone == "KICK":
            action = "tung cú đá tấn công"
        elif cue_charge:
            action = "lao vào xô xát mạnh"
        elif cue_grapple:
            action = "áp sát giằng co, vật lộn"
        else:
            action = "đánh nhau, vung tay quyết liệt"
        return f"Ẩu đả khẩn cấp! ID #{id_a} & #{id_b} đang {action} (V={max_v_limb:.1f} BH/s)"

    def _compute_limb_speed(self, track: TrackHistory) -> float:
        """Tính vận tốc cực đại của các khớp tứ chi tương đối so với tâm thân người (BH/s)."""
        n = min(
            len(track.keypoints_history),
            len(track.keypoint_bbox_history),
            len(track.timestamp_history)
        )
        if n < 3:
            return 0.0

        kpts_curr = track.keypoints_history[-1]
        bbox_curr = track.keypoint_bbox_history[-1]
        t_curr = track.timestamp_history[-1]

        step_back = min(3, n - 1)
        kpts_prev = track.keypoints_history[-1 - step_back]
        bbox_prev = track.keypoint_bbox_history[-1 - step_back]
        t_prev = track.timestamp_history[-1 - step_back]

        dt = t_curr - t_prev
        if not math.isfinite(dt) or dt <= 0.01 or dt > 1.0:
            return 0.0

        torso_curr = self._get_torso_center(kpts_curr, bbox_curr)
        torso_prev = self._get_torso_center(kpts_prev, bbox_prev)
        h_curr = max(1.0, bbox_curr[3] - bbox_curr[1])
        h_prev = max(1.0, bbox_prev[3] - bbox_prev[1])
        avg_h = (h_curr + h_prev) / 2.0

        max_speed = 0.0
        for idx in LIMB_KEYPOINTS:
            if not self._is_valid_kpt(kpts_curr, idx) or not self._is_valid_kpt(kpts_prev, idx):
                continue

            rel_curr_x = kpts_curr[idx, 0] - torso_curr[0]
            rel_curr_y = kpts_curr[idx, 1] - torso_curr[1]
            rel_prev_x = kpts_prev[idx, 0] - torso_prev[0]
            rel_prev_y = kpts_prev[idx, 1] - torso_prev[1]

            disp_rel = math.hypot(rel_curr_x - rel_prev_x, rel_curr_y - rel_prev_y)
            speed_rel = disp_rel / (dt * avg_h)

            if speed_rel > max_speed:
                max_speed = speed_rel

        return max_speed

    def _compute_body_speed(self, track: TrackHistory) -> float:
        """Vận tốc tịnh tiến của cả người (BH/s) trên cửa sổ `body_window` giây."""
        n = min(len(track.bbox_history), len(track.timestamp_history))
        if n < 3:
            return 0.0
        ts = track.timestamp_history
        bb = track.bbox_history
        t_now = ts[-1]
        j = None
        for i in range(n - 1, -1, -1):
            if t_now - ts[-n + i] >= self.body_window * 0.6:
                j = -n + i
                break
        if j is None:
            # Track mới xuất hiện: dùng mẫu cũ nhất nếu đã đủ ~0.1s để không bỏ lọt cặp chạy cùng nhau.
            j = -n
            if t_now - ts[j] < 0.1:
                return 0.0
        dt = t_now - ts[j]
        if dt <= 0 or dt > self.body_window * 2:
            return 0.0
        b0, b1 = bb[j], bb[-1]
        h = max(1.0, (b0[3] - b0[1] + b1[3] - b1[1]) / 2.0)
        dx = (b1[0] + b1[2] - b0[0] - b0[2]) / 2.0
        dy = (b1[1] + b1[3] - b0[1] - b0[3]) / 2.0
        return math.hypot(dx, dy) / (dt * h)

    def _compute_torso_agitation(self, track: TrackHistory) -> float:
        """Đo mức độ giật lắc của thân người (BH/s) để phát hiện vật lộn khi tay bị che khuất."""
        n = min(len(track.bbox_history), len(track.timestamp_history))
        if n < 4:
            return 0.0
        b_curr = track.bbox_history[-1]
        b_prev = track.bbox_history[-3]
        dt = track.timestamp_history[-1] - track.timestamp_history[-3]
        if not math.isfinite(dt) or dt <= 0.02 or dt > 0.8:
            return 0.0
        h = max(1.0, b_curr[3] - b_curr[1])
        cx_curr, cy_curr = (b_curr[0] + b_curr[2]) / 2.0, (b_curr[1] + b_curr[3]) / 2.0
        cx_prev, cy_prev = (b_prev[0] + b_prev[2]) / 2.0, (b_prev[1] + b_prev[3]) / 2.0
        return math.hypot(cx_curr - cx_prev, cy_curr - cy_prev) / (dt * h)

    def _check_strike_zone(
        self,
        attacker: TrackHistory,
        defender: TrackHistory,
        opp_bbox: Tuple[float, float, float, float],
        avg_h: float
    ) -> Optional[str]:
        """
        Kiểm tra khớp tay/chân của attacker có vươn tới vùng Đầu ("HEAD"), Thân ("BODY")
        hoặc tung cú đá ("KICK") về phía defender hay không.
        """
        if len(attacker.keypoints_history) == 0:
            return None
        kpts_att = attacker.keypoints_history[-1]
        kpts_def = defender.keypoints_history[-1] if len(defender.keypoints_history) > 0 else None

        # Tọa độ vùng đầu của đối phương
        head_pts = []
        if kpts_def is not None:
            for idx in HEAD_KEYPOINTS:
                if self._is_valid_kpt(kpts_def, idx):
                    head_pts.append((float(kpts_def[idx, 0]), float(kpts_def[idx, 1])))

        margin = 0.22 * avg_h
        ox1, oy1, ox2, oy2 = opp_bbox[0] - margin, opp_bbox[1] - margin, opp_bbox[2] + margin, opp_bbox[3] + margin
        head_y_limit = opp_bbox[1] + 0.35 * (opp_bbox[3] - opp_bbox[1])

        # Kiểm tra đòn tay vào Đầu hoặc Thân
        for idx in WRIST_ELBOW_KEYPOINTS:
            if not self._is_valid_kpt(kpts_att, idx):
                continue
            kx, ky = float(kpts_att[idx, 0]), float(kpts_att[idx, 1])
            if head_pts:
                for hx, hy in head_pts:
                    if math.hypot(kx - hx, ky - hy) <= 0.28 * avg_h:
                        return "HEAD"
            if ox1 <= kx <= ox2 and oy1 <= ky <= oy2:
                if ky <= head_y_limit:
                    return "HEAD"
                return "BODY"

        # Kiểm tra đòn đá chân (Ankle nhấc cao hoặc đá sang khung đối phương)
        for idx in ANKLE_KEYPOINTS:
            if not self._is_valid_kpt(kpts_att, idx):
                continue
            kx, ky = float(kpts_att[idx, 0]), float(kpts_att[idx, 1])
            if ox1 <= kx <= ox2 and oy1 <= ky <= (opp_bbox[3] - 0.15 * avg_h):
                return "KICK"

        return None

    def _get_torso_center(self, kpts: np.ndarray, bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
        valid_pts = []
        for idx in (5, 6, 11, 12):
            if self._is_valid_kpt(kpts, idx):
                valid_pts.append((float(kpts[idx, 0]), float(kpts[idx, 1])))
        if valid_pts:
            cx = sum(p[0] for p in valid_pts) / len(valid_pts)
            cy = sum(p[1] for p in valid_pts) / len(valid_pts)
            return cx, cy
        return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)

    def _is_valid_kpt(self, kpts: np.ndarray, idx: int) -> bool:
        if kpts is None or kpts.ndim != 2 or idx >= kpts.shape[0]:
            return False
        x, y = float(kpts[idx, 0]), float(kpts[idx, 1])
        if not math.isfinite(x) or not math.isfinite(y) or x <= 1.0 or y <= 1.0:
            return False
        if kpts.shape[1] >= 3 and float(kpts[idx, 2]) < self.kpt_conf_thresh:
            return False
        return True

    @staticmethod
    def _compute_iou(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if inter <= 0.0:
            return 0.0
        area_a = max(1.0, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
        area_b = max(1.0, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
        return inter / (area_a + area_b - inter)

    def _cleanup_stale_pairs(self, current_time: float):
        for pair_key, last_t in list(self.pair_last_seen.items()):
            if current_time - last_t > max(3.0, self.hold_seconds):
                self.pair_last_seen.pop(pair_key, None)
                self.pair_streaks.pop(pair_key, None)
                self.pair_prev_dist.pop(pair_key, None)
                self.pair_alert_until.pop(pair_key, None)
                self.pair_visuals.pop(pair_key, None)
                self.pair_candidate_since.pop(pair_key, None)
