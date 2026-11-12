r"""Chạy toàn bộ pipeline (cả 3 hành vi) trên các clip công khai và đếm đúng/sai theo clip.

Chạy:  python -m scripts.evaluate_public_clips
Nhãn mong đợi suy ra từ tên file (xem EXPECTED_BY_PREFIX). Mỗi clip là một mẫu:
- clip dương tính của hành vi X: đúng nếu có ít nhất một cảnh báo X;
- mọi cảnh báo loại khác với nhãn của clip được tính là báo giả.
Kết quả ghi vào data/public_evaluation/results.json. Không ghi vào SQLite/snapshot vận hành.
"""
import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from unittest.mock import patch

import cv2
import torch
import yaml

from src.core.pipeline import SurveillancePipeline

EVENTS = ("RUNNING", "FALLING", "FIGHTING")

# (thư mục con, tiền tố tên file) -> hành vi mong đợi (None = clip bình thường)
EXPECTED_BY_PREFIX = [
    ("ur_fall", "fall-", "FALLING"),
    ("ur_fall", "adl-", None),
    ("fight_surv", "fi", "FIGHTING"),
    ("fight_surv", "nofi", None),
    ("sphar", "running__", "RUNNING"),
    ("sphar", "walking__", None),
    ("sphar", "fighting__", "FIGHTING"),
    ("sphar", "falling__", "FALLING"),
]


def expected_label(path: Path):
    for folder, prefix, label in sorted(EXPECTED_BY_PREFIX, key=lambda r: -len(r[1])):
        if path.parent.name == folder and path.name.startswith(prefix):
            return label
    raise ValueError(f"Chưa khai báo nhãn cho {path}")


def run_clip(pipeline, path: Path, target_fps: float, events: list, clock: dict, video_out=None):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    stride = max(1, round(fps / target_fps))
    pipeline.request_reset()
    events.clear()
    writer = None
    index = 0
    max_people = 0
    sampled_frames = tracked_frames = 0
    fall_phases = set()
    standing_reference = False
    run_speed_available = run_speed_reached = run_condition_met = False
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index % stride == 0:
            now = index / fps
            clock["t"] = now
            rendered = pipeline.process_frame(frame, now)
            tracks = pipeline.track_manager.tracks.values()
            max_people = max(max_people, len(tracks))
            active = [track for track in tracks if track.last_seen == now]
            sampled_frames += 1
            tracked_frames += bool(active)
            for track in active:
                fall_phases.add(track.fall_phase)
                standing_reference |= track.fall_reference_height is not None
                metrics = track.debug_metrics
                run_speed_available |= "V" in metrics
                run_speed_reached |= bool(metrics.get("V", (None, False))[1])
                run_condition_met |= bool(metrics.get("Run", (None, False))[1])
            if video_out is not None:
                if writer is None:
                    h, w = rendered.shape[:2]
                    writer = cv2.VideoWriter(str(video_out), cv2.VideoWriter_fourcc(*"mp4v"),
                                             fps / stride, (w, h))
                writer.write(rendered)
        index += 1
    cap.release()
    if writer:
        writer.release()
    if not sampled_frames:
        raise RuntimeError(f"Video không có frame: {path}")
    return {"seconds": round(index / fps, 2), "max_tracks": max_people, "events": list(events),
            "diagnostics": {"sampled_frames": sampled_frames, "tracked_frames": tracked_frames,
                            "fall_phases": sorted(fall_phases), "standing_reference": standing_reference,
                            "run_speed_available": run_speed_available, "run_speed_reached": run_speed_reached,
                            "run_condition_met": run_condition_met}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/test_videos/public"))
    parser.add_argument("--output", type=Path, default=Path("data/public_evaluation"))
    parser.add_argument("--fps", type=float, default=15, help="Tốc độ lấy mẫu mô phỏng khâu AI")
    parser.add_argument("--save-video", action="store_true", help="Lưu video đã vẽ khung của từng clip")
    parser.add_argument("--only", default="", help="Chỉ chạy clip có tên chứa chuỗi này")
    args = parser.parse_args()
    torch.set_num_threads(2)

    cfg = yaml.safe_load(Path("configs/config.yaml").read_text(encoding="utf-8"))
    cfg["web"]["snapshots_dir"] = str(args.output / "snapshots")
    args.output.mkdir(parents=True, exist_ok=True)

    events, clock = [], {"t": 0.0}
    with patch("src.core.pipeline.DatabaseService") as db:
        db.return_value.count_total_events.side_effect = lambda: len(events)
        pipeline = SurveillancePipeline(cfg)

    class Sink:
        def trigger_alert(self, event_type, track_id, frame, confidence=1.0, extra_info=None, **_):
            events.append({"event_type": event_type, "track_id": int(track_id),
                           "seconds": round(clock["t"], 2), "extra_info": extra_info})

    sink = Sink()
    pipeline.alert_manager = sink
    pipeline.motion_fall_detector.alert_manager = sink
    # request_reset tạo lại FightingDetector từ pipeline.alert_manager nên không cần gán riêng.

    clips = sorted(p for p in args.root.rglob("*") if p.suffix.lower() in (".mp4", ".avi")
                   and args.only in p.name)
    rows = []
    started = time.monotonic()
    for i, path in enumerate(clips, 1):
        label = expected_label(path)
        out_video = (args.output / f"{path.parent.name}__{path.stem}.mp4") if args.save_video else None
        row = run_clip(pipeline, path, args.fps, events, clock, out_video)
        fired = sorted({e["event_type"] for e in row["events"]})
        row.update(video=f"{path.parent.name}/{path.name}", expected=label, fired=fired)
        rows.append(row)
        print(f"[{i}/{len(clips)}] {row['video']}: mong đợi={label} | phát ra={fired}", flush=True)

    summary = {}
    for ev in EVENTS:
        pos = [r for r in rows if r["expected"] == ev]
        neg = [r for r in rows if r["expected"] != ev]
        tp = sum(ev in r["fired"] for r in pos)
        fp = sum(ev in r["fired"] for r in neg)
        summary[ev] = {
            "positive_clips": len(pos), "detected": tp, "missed": len(pos) - tp,
            "other_clips": len(neg), "false_alarm_clips": fp,
            "recall": round(tp / len(pos), 3) if pos else None,
            "precision": round(tp / (tp + fp), 3) if tp + fp else None,
        }
    by_source = defaultdict(Counter)
    for r in rows:
        key = f"{r['video'].split('/')[0]}:{r['expected']}"
        by_source[key]["clips"] += 1
        by_source[key]["correct"] += (r["fired"] == ([r["expected"]] if r["expected"] else []))

    report = {"target_fps": args.fps, "device": pipeline.device,
              "fight_clip_model_enabled": pipeline.fight_model is not None,
              "elapsed_seconds": round(time.monotonic() - started, 1),
              "summary": summary, "by_source": {k: dict(v) for k, v in by_source.items()},
              "clips": rows}
    (args.output / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                              encoding="utf-8")
    print("\n=== TÓM TẮT (theo clip) ===")
    for ev, s in summary.items():
        print(f"{ev:9s} bắt được {s['detected']}/{s['positive_clips']} | "
              f"báo giả trên {s['false_alarm_clips']}/{s['other_clips']} clip khác | "
              f"recall={s['recall']} precision={s['precision']}")
    for k, v in sorted(by_source.items()):
        print(f"  {k:28s} đúng hoàn toàn {v['correct']}/{v['clips']}")


if __name__ == "__main__":
    main()
