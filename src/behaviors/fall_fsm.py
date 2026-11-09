"""Temporal fall candidate, posture verification and persistent alert latch."""
import math
import numpy as np

from src.utils.geometry import calculate_aspect_ratio, normalize_skeleton_root_relative


class FallFSMMixin:
    def _midpoint(self, kpts, left, right):
        if kpts is None or kpts.ndim != 2 or kpts.shape[0] <= right or kpts.shape[1] < 2:
            return None
        points = []
        for i in (left, right):
            row = kpts[i]
            if (np.isfinite(row).all() and np.any(row[:2] != 0)
                    and (len(row) < 3 or row[2] >= self.config_fall.get("keypoint_confidence", 0.3))):
                points.append(row[:2])
        return np.mean(points, axis=0) if points else None

    @staticmethod
    def _reset_fall(track):
        track.fall_phase = "NORMAL"
        track.fall_impact_time = None
        track.fall_lying_since = None
        track.fall_recovery_since = None
        if track.current_state == "FALLING_ALERT":
            track.current_state = "NORMAL"

    def _process_fall(self, track, now, frame, angle, hip):
        cfg = self.config_fall
        if not math.isfinite(now):
            return
        previous_time = track.fall_last_time
        if previous_time is not None and now <= previous_time:
            if track.fall_phase == "FALL_LATCHED":
                track.current_state = "FALLING_ALERT"
            return
        gap = previous_time is not None and now - previous_time > cfg.get("max_observation_gap", 0.3)
        track.fall_last_time = now
        if gap:
            track.fall_motion_samples.clear()
            track.fall_height_samples.clear()
            track.fall_lying_since = track.fall_recovery_since = None
            if track.fall_phase != "FALL_LATCHED":
                self._reset_fall(track)
                track.fall_reference_height = None
        bbox = track.bbox_history[-1]
        if not np.isfinite(bbox).all() or bbox[3] <= bbox[1] or bbox[2] <= bbox[0]:
            track.fall_lying_since = track.fall_recovery_since = None
            if track.fall_phase == "FALL_LATCHED":
                track.current_state = "FALLING_ALERT"
            return
        height = bbox[3] - bbox[1]
        ar = calculate_aspect_ratio(bbox)
        center = (bbox[1] + bbox[3]) / 2.0
        samples = track.fall_motion_samples
        samples.append((now, center, bbox[1], float(hip[1]) if hip is not None else None))
        window = cfg.get("velocity_window_seconds", 0.2)
        while len(samples) > 2 and samples[1][0] <= now - window:
            samples.popleft()
        ref = track.fall_reference_height
        vy = top_speed = 0.0
        if ref is not None and len(samples) > 1:
            old = samples[0]
            dt = now - old[0]
            if dt >= window * 0.5:
                # Never mix hip and bounding-box center between endpoints.
                displacement = hip[1] - old[3] if hip is not None and old[3] is not None else center - old[1]
                vy = float(displacement / (ref * dt))
                top_speed = float((bbox[1] - old[2]) / (ref * dt))
        ratio = height / ref if ref is not None else 1.0
        low = (ar <= self.ar_max or (angle is not None and angle <= self.angle_max)
               or ratio <= cfg.get("lying_height_ratio", 0.55))
        # Height loss alone also describes slow squats: require rapid motion.
        impact = ref is not None and (vy >= self.vy_drop_min or (
            ratio <= cfg.get("impact_height_ratio", 0.65)
            and top_speed >= cfg.get("top_drop_min", 1.0)))
        track.debug_metrics.update({
            "AR": (f"AR={ar:.2f}", ar <= self.ar_max),
            "Angle": (f"Angle={angle:.0f}" if angle is not None else "Angle=N/A", angle is not None and angle <= self.angle_max),
            "Vy": (f"Vy={vy:.2f} BH/s", vy >= self.vy_drop_min),
            "Height": (f"H/Href={ratio:.2f}" if ref is not None else "Href=warming", ratio <= 0.65),
        })
        if track.fall_phase == "FALL_LATCHED":
            track.current_state = "FALLING_ALERT"
            recovered = (ar > cfg.get("recovery_aspect_ratio", 1.2)
                         and ratio > cfg.get("recovery_height_ratio", 0.8)
                         and (angle is None or angle > self.angle_max))
            if recovered:
                if track.fall_recovery_since is None:
                    track.fall_recovery_since = now
                if now - track.fall_recovery_since >= cfg.get("recovery_seconds", 0.5):
                    self._reset_fall(track)
                    track.fall_motion_samples.clear()
                    track.fall_height_samples.clear()
            else:
                track.fall_recovery_since = None
        else:
            if track.fall_phase == "NORMAL" and impact:
                track.fall_phase = "IMPACT_PENDING"
                track.fall_impact_time = now
            if track.fall_phase in ("IMPACT_PENDING", "LYING_VERIFY"):
                if now - track.fall_impact_time > cfg.get("impact_window_seconds", 1.2):
                    self._reset_fall(track)
                elif low:
                    track.fall_phase = "LYING_VERIFY"
                    if track.fall_lying_since is None:
                        track.fall_lying_since = now
                    if now - track.fall_lying_since >= cfg.get("lying_seconds", 0.5):
                        confidence = self._verify_fall(track)
                        if confidence is not None:
                            track.fall_phase = "FALL_LATCHED"
                            self._trigger_fall_alert(track, now, frame, confidence, "Temporal FSM verified")
                        else:
                            self._reset_fall(track)
                else:
                    track.fall_phase = "IMPACT_PENDING"
                    track.fall_lying_since = None
            # Freeze standing height for the entire candidate and latched event.
            if track.fall_phase == "NORMAL" and ar > cfg.get("standing_aspect_ratio", 1.3) and not low and not impact:
                heights = track.fall_height_samples
                heights.append((now, height))
                while heights and now - heights[0][0] > cfg.get("reference_window_seconds", 1.5):
                    heights.popleft()
                if len(heights) >= 3 and now - heights[0][0] >= cfg.get("reference_warmup_seconds", 0.3):
                    track.fall_reference_height = float(np.median([h for _, h in heights]))
        track.debug_metrics["Fall"] = (f"Fall={track.fall_phase}", track.fall_phase == "FALL_LATCHED")

    def _verify_fall(self, track):
        if self.bilstm_model is None:
            return 0.9  # Heuristic score, not a calibrated probability.
        if len(track.keypoints_history) < 30 or len(track.keypoint_bbox_history) < 30:
            return None
        features = [normalize_skeleton_root_relative(k, b) for k, b in zip(
            list(track.keypoints_history)[-30:], list(track.keypoint_bbox_history)[-30:])]
        _, _, probability = self.bilstm_model.predict(np.asarray([features], dtype=np.float32))
        return float(probability) if probability >= self.bilstm_fall_thresh else None
