"""Compare YOLO/ByteTrack settings on local clips, without operational alerts.

Cache raw predictions so every tracker variant receives identical detections.
Coverage measures frames with an active ID, not ground-truth detection accuracy.
"""
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker

from src.behaviors.fall_run import MotionAndFallDetector
from src.tracking.track_manager import TrackManager


CLIPS = [
    "sphar/running__casia_horizontalview_p01_run_a2.mp4",
    "sphar/running__casia_angleview_p01_run_a1.mp4",
    "sphar/falling__casia_horizontalview_p08_faint_a1.mp4",
    "ur_fall/fall-01.mp4",
    "sphar/walking__casia_horizontalview_p02_walk_a1.mp4",
    "ur_fall/adl-01.mp4",
]
VARIANTS = [
    ("baseline_640", 640, .4, .25),
    ("low_conf_640", 640, .1, .25),
    ("sensitive_640", 640, .1, .15),
    ("low_conf_960", 960, .1, .25),
    ("sensitive_960", 960, .1, .15),
    ("selected_640", 640, .1, .4, True, .4),
]


def predict_clip(path, size, out, model_path, fps_target):
    cache = out / f"{path.stem}-{size}-detections.json"
    key = [path.stat().st_size, path.stat().st_mtime_ns, size, fps_target,
           str(model_path), Path(model_path).stat().st_mtime_ns]
    if cache.exists():
        saved = json.loads(cache.read_text())
        if saved.get("key") == key:
            return saved
    model = YOLO(model_path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    stride = max(1, round(fps / fps_target))
    frames, index = [], 0
    elapsed = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index % stride == 0:
            start = time.perf_counter()
            result = model.predict(frame, imgsz=size, conf=.1, classes=[0], device="cpu", verbose=False)[0]
            elapsed += time.perf_counter() - start
            frames.append({"time": index / fps, "shape": list(frame.shape[:2]),
                           "boxes": result.boxes.data.cpu().tolist(),
                           "kpts": result.keypoints.data.cpu().tolist()})
        index += 1
    cap.release()
    saved = {"key": key, "inference_seconds": elapsed, "frames": frames}
    cache.write_text(json.dumps(saved), encoding="utf-8")
    return saved


def replay(saved, cfg, conf, high, fuse_score=True, new_track_thresh=None):
    args = SimpleNamespace(track_high_thresh=high, track_low_thresh=.1,
                           new_track_thresh=high if new_track_thresh is None else new_track_thresh, track_buffer=30,
                           match_thresh=.8, fuse_score=fuse_score)
    tracker = BYTETracker(args)
    manager = TrackManager()
    events = []
    now = 0.0

    class Sink:
        def trigger_alert(self, **kwargs):
            events.append({"time": round(now, 3), "type": kwargs["event_type"]})

    detector = MotionAndFallDetector(cfg["behaviors"]["running"], cfg["behaviors"]["falling"], alert_manager=Sink())
    detected_frames = tracked_frames = 0
    ids = set()
    max_speed = 0.0
    for row in saved["frames"]:
        now = row["time"]
        raw = np.asarray(row["boxes"], dtype=np.float32).reshape(-1, 6)
        mask = raw[:, 4] >= conf
        boxes = Boxes(raw[mask], tuple(row["shape"]))
        poses = np.asarray(row["kpts"], dtype=np.float32).reshape(-1, 17, 3)[mask]
        detected_frames += bool(len(boxes))
        tracks = tracker.update(boxes)
        tracked_frames += bool(len(tracks))
        for item in tracks:
            box, tid, source = tuple(item[:4]), int(item[4]), int(item[-1])
            ids.add(tid)
            track = manager.update_track(tid, box, (box[0], box[3]), poses[source], now)
            detector.process(track, now, None)
            speed = detector._compute_run_speed(track)
            if speed is not None:
                max_speed = max(max_speed, speed)
        manager.purge_stale_tracks(now)
    total = len(saved["frames"])
    return {"frames": total, "detected_frames": detected_frames, "tracked_frames": tracked_frames,
            "coverage": round(tracked_frames / max(total, 1), 3), "track_ids": len(ids),
            "max_speed": round(max_speed, 3), "events": events,
            "inference_ms_per_frame": round(saved["inference_seconds"] / max(total, 1) * 1000, 2)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/motion_tracking_review"))
    parser.add_argument("--fps", type=float, default=15)
    parser.add_argument("--clips", nargs="*", default=CLIPS)
    options = parser.parse_args()
    options.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    cfg = yaml.safe_load(Path("configs/config.yaml").read_text(encoding="utf-8"))
    rows = []
    for clip in options.clips:
        saved_by_size = {}
        for name, size, conf, high, *tracker_options in VARIANTS:
            if size not in saved_by_size:
                saved_by_size[size] = predict_clip(Path("data/test_videos/public") / clip, size,
                                                  options.output, cfg["system"]["model_pose_path"], options.fps)
            row = {"clip": clip, "variant": name, **replay(saved_by_size[size], cfg, conf, high, *tracker_options)}
            rows.append(row)
            print(json.dumps(row), flush=True)
            (options.output / "comparison.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
