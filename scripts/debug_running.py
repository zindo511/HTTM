"""
In số liệu chẩn đoán hành vi chạy theo từng frame cho một video:
ID, độ tin cậy, khung, tốc độ (mặt phẳng ảnh / chiều sâu), dấu hiệu pose, ngữ cảnh, trạng thái FSM.

    python scripts/debug_running.py data/test_videos/runn.mp4 --fps 10
"""
import argparse
import json
import sys
from pathlib import Path

if sys.platform == "win32":  # tránh treo khi import (xem main.py)
    import platform

    def _skip_wmi(*args, **kwargs):
        raise OSError("WMI query skipped")

    platform._wmi_query = _skip_wmi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from unittest.mock import Mock  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from src.behaviors.fall_run import MotionAndFallDetector  # noqa: E402
from src.tracking.track_manager import TrackManager  # noqa: E402
from src.utils.geometry import get_foot_point  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--fps", type=float, default=None, help="Nhịp lấy mẫu (mặc định theo config)")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--focal", type=float, default=None, help="Ghi đè focal_length_px để thử")
    parser.add_argument("--cache", type=Path, default=None,
                        help="File .json lưu kết quả YOLO; chạy lại sẽ không cần suy luận")
    args = parser.parse_args()
    cache = json.loads(args.cache.read_text()) if args.cache and args.cache.exists() else {}

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    sys_cfg, beh = cfg["system"], cfg["behaviors"]
    target_fps = args.fps or cfg["camera"].get("file_sample_fps", 10)

    cap = cv2.VideoCapture(args.video)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    stride = max(1, round(src_fps / target_fps))
    if args.focal:
        beh["running"]["focal_length_px"] = args.focal
    model = None
    manager = TrackManager(motion_history_seconds=max(2.0, beh["running"].get("window_seconds", 0.5) + 0.5))
    alerts = Mock()
    det = MotionAndFallDetector(beh["running"], beh["falling"], alert_manager=alerts)
    run = det.running
    max_side = cfg["camera"].get("max_frame_side")

    print(f"{'t':>5} {'id':>3} {'conf':>4} {'h_px':>5} {'edge':>4} {'V_img':>5} {'V_lat':>5} {'V_sau':>5} "
          f"{'V':>5} {'pose(lean,legs,arms)':>20} {'ctx':>10} state")
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index % stride:
            index += 1
            continue
        if max_side and max(frame.shape[:2]) > max_side:
            s = max_side / max(frame.shape[:2])
            frame = cv2.resize(frame, (int(frame.shape[1] * s), int(frame.shape[0] * s)))
        t = index / src_fps
        index += 1
        key = str(index - 1)
        if key not in cache:
            if model is None:
                model = YOLO(sys_cfg["model_pose_path"])
            r = model.track(frame, persist=True, classes=[0],
                            tracker=cfg["tracking"].get("tracker_path", "bytetrack.yaml"),
                            imgsz=sys_cfg.get("inference_imgsz", 640),
                            conf=sys_cfg.get("confidence_threshold", 0.1), device="cpu", verbose=False)[0]
            found = r.boxes is not None and r.boxes.id is not None
            cache[key] = {
                "boxes": r.boxes.xyxy.tolist() if found else [],
                "conf": r.boxes.conf.tolist() if found else [],
                "ids": r.boxes.id.int().tolist() if found else [],
                "kpts": r.keypoints.data.tolist() if found and r.keypoints is not None else [],
            }
        row = cache[key]
        if not row["ids"]:
            print(f"{t:5.1f}   -  không có người được bám vết")
            continue
        boxes = np.array(row["boxes"], dtype=np.float32)
        confs = np.array(row["conf"], dtype=np.float32)
        ids = np.array(row["ids"], dtype=int)
        kpts = np.array(row["kpts"], dtype=np.float32) if row["kpts"] else None
        fh, fw = frame.shape[:2]
        for i, (box, tid) in enumerate(zip(boxes, ids)):
            track = manager.update_track(int(tid), tuple(box), get_foot_point(box),
                                         kpts[i] if kpts is not None else None, t)
            n_alerts = alerts.trigger_alert.call_count
            det.process(track, t, frame)
            shape = frame.shape[:2]
            planar = run._planar_speed(track, shape)
            parts = run._depth_speed(track, shape)
            body = track.debug_metrics.get("Pose", ("-",))[0]
            cues = run.pose_cues(track, None)
            cue_txt = ",".join("1" if v else "0" for v in cues.values()) if cues else "khong pose"
            edge = "".join(c for c, hit in (("T", box[1] <= 3), ("B", box[3] >= fh - 3),
                                            ("L", box[0] <= 3), ("R", box[2] >= fw - 3)) if hit) or "-"
            v = det._compute_run_speed(track)
            fmt = lambda x: f"{x:5.2f}" if x is not None else "    -"
            ctx = track.debug_metrics.get("Ctx", ("-",))[0].replace("Ctx=", "")
            alert = "  <== CANH BAO" if alerts.trigger_alert.call_count > n_alerts else ""
            print(f"{t:5.1f} {tid:3d} {confs[i]:4.2f} {box[3]-box[1]:5.0f} {edge:>4} {fmt(planar)} "
                  f"{fmt(parts[0] if parts else None)} {fmt(parts[1] if parts else None)} {fmt(v)} "
                  f"{cue_txt:>20} {ctx:>10} {track.run_state}{'/sau' if track.run_depth_dominant else ''}{alert}")
        manager.purge_stale_tracks(t)
    if args.cache:
        args.cache.write_text(json.dumps(cache))
    print(f"\nTổng số cảnh báo chạy: {alerts.trigger_alert.call_count}")


if __name__ == "__main__":
    main()
