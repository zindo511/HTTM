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

    def _legs_extended(self, kpts):
        """True when standing (a thigh points down), False when sitting, None if unknown."""
        conf = self.config_fall.get("keypoint_confidence", 0.3)
        if kpts is None or kpts.ndim != 2 or kpts.shape[0] < 17:
            return None

        def point(i):
            row = kpts[i]
            ok = (np.isfinite(row).all() and np.any(row[:2] != 0)
                  and (len(row) < 3 or row[2] >= conf))
            return row[:2] if ok else None

        best = None
        for shoulder, hip, knee in ((5, 11, 13), (6, 12, 14)):
            s, h, k = point(shoulder), point(hip), point(knee)
            if s is None or h is None or k is None:
                continue
            torso = float(np.hypot(*(h - s)))
            if torso < 1e-6:
                continue
            drop = float(k[1] - h[1]) / torso
            best = drop if best is None else max(best, drop)
        if best is None:
            return None
        return best >= self.config_fall.get("standing_thigh_ratio", 0.5)

    @staticmethod
    def _reset_fall(track):
        track.fall_phase = "NORMAL"
        track.fall_impact_time = None
        track.fall_lying_since = None
        track.fall_recovery_since = None
        track.fall_impact_kind = None
        track.fall_not_low_since = None
        if track.current_state == "FALLING_ALERT":
            track.current_state = "NORMAL"

    def adopt_fall_candidate(self, track, tracks, now):
        """Hand an in-progress fall over to a track ID created during the fall.

        The tracker often loses the person when the box flips from standing to
        lying and starts a new ID. That ID never saw the standing height or the
        impact, so without this the fall can never be confirmed.
        """
        cfg = self.config_fall
        max_gap = cfg.get("handover_gap_seconds", 1.0)
        if (max_gap <= 0 or not math.isfinite(now) or len(track.bbox_history) != 1
                or track.fall_phase != "NORMAL"):
            return None
        bbox = np.asarray(track.bbox_history[-1], dtype=float)
        if not np.isfinite(bbox).all() or bbox[3] <= bbox[1] or bbox[2] <= bbox[0]:
            return None
        center = np.array([(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0])
        best, best_dist = None, None
        for donor in tracks:
            ref = donor.fall_reference_height
            if (donor is track or donor.fall_phase == "NORMAL" or not ref
                    or donor.fall_last_time is None or donor.last_seen >= now
                    or not 0 < now - donor.fall_last_time <= max_gap
                    or not donor.bbox_history):
                continue
            old = np.asarray(donor.bbox_history[-1], dtype=float)
            if not np.isfinite(old).all():
                continue
            # A lying body is not taller than the person standing.
            if bbox[3] - bbox[1] > ref * 1.1:
                continue
            old_center = np.array([(old[0] + old[2]) / 2.0, (old[1] + old[3]) / 2.0])
            dist = float(np.linalg.norm(center - old_center)) / ref
            if dist <= cfg.get("handover_max_distance_bh", 1.0) and (best is None or dist < best_dist):
                best, best_dist = donor, dist
        if best is None:
            return None
        gap = now - best.fall_last_time
        track.fall_phase = best.fall_phase
        track.fall_impact_time = best.fall_impact_time
        track.fall_impact_kind = best.fall_impact_kind
        track.fall_reference_height = best.fall_reference_height
        # Same rule as a short reacquire: unobserved time adds nothing to lying.
        track.fall_lying_since = best.fall_lying_since + gap if best.fall_lying_since is not None else None
        track.fall_last_time = None
        for event, last in best.last_alert_time.items():
            track.last_alert_time[event] = max(last, track.last_alert_time.get(event, last))
        if best.fall_phase == "FALL_LATCHED":
            track.current_state = "FALLING_ALERT"
        self._reset_fall(best)
        best.fall_reference_height = None
        return best.track_id

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
            track.fall_collapse_samples.clear()
            track.fall_recovery_since = None
            if (track.fall_phase == "LYING_VERIFY" and track.fall_lying_since is not None
                    and now - previous_time <= cfg.get("lying_reacquire_gap_seconds", 0.4)):
                # Pause confirmation across a short gap. A low pose on return
                # can continue the timer, but the unobserved time adds nothing.
                track.fall_lying_since += now - previous_time
            elif track.fall_phase != "FALL_LATCHED":
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
        # A horizontal torso alone also describes bending over (picking something
        # up, brawling): only trust it once the box is low and not tall.
        torso_flat = (angle is not None and angle <= self.angle_max
                      and ratio <= cfg.get("flat_torso_max_height_ratio", 0.8)
                      and ar <= cfg.get("flat_torso_max_aspect_ratio", 1.5))
        posture_low = ar <= self.ar_max or torso_flat
        low = posture_low or ratio <= cfg.get("lying_height_ratio", 0.55)
        # Height loss alone also describes slow squats: require rapid motion.
        impact = ref is not None and (vy >= self.vy_drop_min or (
            ratio <= cfg.get("impact_height_ratio", 0.65)
            and top_speed >= cfg.get("top_drop_min", 1.0)))
        # Slow collapse: a large height loss within ~1 s (e.g. sinking backwards
        # or falling towards the camera), too slow for the velocity test above.
        collapse_samples = track.fall_collapse_samples
        kpts = (track.keypoints_history[-1] if track.keypoints_history
                and track.missing_kpts_count == 0 else None)
        collapse_samples.append((now, height, self._legs_extended(kpts)))
        while collapse_samples and now - collapse_samples[0][0] > cfg.get("collapse_window_seconds", 1.0):
            collapse_samples.popleft()
        collapse = False
        if ref is not None and not impact and cfg.get("collapse_window_seconds", 1.0) > 0:
            # Only a drop that starts from a standing pose: sitting then lying
            # down (bed, sofa) also loses height quickly.
            standing = [h for _, h, legs in collapse_samples if legs is not False]
            if standing:
                peak = max(standing) / ref
                collapse = (ratio <= cfg.get("collapse_height_ratio", 0.55)
                            and peak - ratio >= cfg.get("collapse_drop_min", 0.4))
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
            if track.fall_phase == "NORMAL" and (impact or collapse):
                track.fall_phase = "IMPACT_PENDING"
                track.fall_impact_time = now
                track.fall_impact_kind = "fast" if impact else "collapse"
            if track.fall_phase in ("IMPACT_PENDING", "LYING_VERIFY"):
                # A slow collapse alone could be an occluded lower body: it must be
                # confirmed by a lying posture, not by box height.
                lying = posture_low if track.fall_impact_kind == "collapse" else low
                if not lying and track.fall_phase == "LYING_VERIFY" and not gap:
                    if track.fall_not_low_since is None:
                        track.fall_not_low_since = now
                    # Tolerate brief pose jitter instead of restarting the timer.
                    if now - track.fall_not_low_since <= cfg.get("lying_grace_seconds", 0.2):
                        lying = True
                else:
                    track.fall_not_low_since = None
                if now - track.fall_impact_time > cfg.get("impact_window_seconds", 1.2):
                    self._reset_fall(track)
                elif lying:
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
                    track.fall_not_low_since = None
            # Freeze standing height for the entire candidate and latched event.
            if (track.fall_phase == "NORMAL" and ar > cfg.get("standing_aspect_ratio", 1.3)
                    and not low and not impact and not collapse):
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
