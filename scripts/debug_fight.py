"""
In số liệu chẩn đoán đánh nhau theo từng khung cho một video: số người, xác suất mô hình clip,
và với mỗi cặp người: khoảng cách, tốc độ tay/chân, độ giật thân, vùng chạm đòn, các dấu hiệu bật.

    python scripts/debug_fight.py data/test_videos/fight5.mp4
    python scripts/debug_fight.py data/record-fight.mp4 --crop 198,84,1143,615 --start 2 --end 13

Dấu hiệu (cues): S = ra đòn nhanh, G = vật lộn, C = lao vào nhau, X = xô xát kéo dài.
rules_ok=False nghĩa là mô hình clip đang bác bỏ các luật S/G/C (xác suất < veto_threshold).
"""
import argparse
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch

if sys.platform == "win32":  # tránh treo khi import (xem main.py)
    import platform

    def _skip_wmi(*args, **kwargs):
        raise OSError("WMI query skipped")

    platform._wmi_query = _skip_wmi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import yaml  # noqa: E402

from src.core.pipeline import SurveillancePipeline  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--fps", type=float, default=None, help="Nhịp lấy mẫu (mặc định theo config)")
    parser.add_argument("--crop", default=None, help="x1,y1,x2,y2: chỉ lấy vùng này (vd. video quay màn hình)")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=float("inf"))
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cfg.setdefault("web", {})["snapshots_dir"] = os.path.join(os.environ.get("TEMP", "."), "debug_fight_snapshots")
    fps = args.fps or cfg["camera"].get("file_sample_fps", 10)
    crop = tuple(int(v) for v in args.crop.split(",")) if args.crop else None

    with patch("src.core.pipeline.DatabaseService") as db:
        db.return_value.count_total_events.return_value = 0
        pipe = SurveillancePipeline(cfg)
    alerts, now = [], {"t": 0.0}
    sink = Mock()
    sink.trigger_alert.side_effect = lambda **k: alerts.append((now["t"], k["event_type"], k.get("extra_info", "")))
    pipe.alert_manager = pipe.motion_fall_detector.alert_manager = pipe.fighting_detector.alert_manager = sink

    cap = cv2.VideoCapture(args.video)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    next_t, index = args.start, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = index / src_fps
        index += 1
        if t > args.end:
            break
        if t + 1e-9 < next_t:
            continue
        next_t += 1.0 / fps
        if crop:
            frame = frame[crop[1]:crop[3], crop[0]:crop[2]]
        now["t"] = t
        before = len(alerts)
        pipe.process_frame(frame, t)
        people = [tr.track_id for tr in pipe.track_manager.tracks.values() if tr.last_seen == t]
        p = pipe.last_fight_probability
        line = f"t={t:6.2f} nguoi={len(people)} p={'  -' if p is None else f'{p:.2f}'}"
        for (a, b), d in pipe.fighting_detector.debug_pairs.items():
            line += (f" | #{a}-#{b} d={float(d['d']):.2f} v={float(d['v_limb']):.2f} agit={float(d['agit']):.2f}"
                     f" zone={d['zone'] or '-'} xo_xat={d['scuffle']:.2f} rules_ok={d['rules_ok']} cues={d['cues'] or '-'}")
        for _, kind, info in alerts[before:]:
            line += f"   <== CANH BAO {kind}: {info}"
        print(line)
    print(f"\nTổng: {sum(1 for a in alerts if a[1] == 'FIGHTING')} cảnh báo đánh nhau")


if __name__ == "__main__":
    main()
