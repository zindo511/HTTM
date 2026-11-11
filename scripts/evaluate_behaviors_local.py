r"""Replay local videos through the real pipeline, without touching production events.

Run: .venv-test\Scripts\python.exe -m scripts.evaluate_behaviors_local
Pose caches allow repeatable detector tuning without repeating YOLO inference.
"""
import argparse
import copy
import json
import time
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO

from src.core.pipeline import SurveillancePipeline


def evaluate(source, config, out, target_fps):
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {source}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride = max(1, round(fps / target_fps))
    cache_path = out / (source.stem + '-poses.json')
    key = [source.stat().st_size, source.stat().st_mtime_ns, stride,
           config['system']['confidence_threshold']]
    cached = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    poses = cached.get('poses', {}) if cached.get('key') == key else {}
    model = None
    events, thumbs, transitions = [], [], []
    current = {'time': 0, 'frame': 0}

    def alert(**kwargs):
        event = {k: v for k, v in kwargs.items() if k != 'frame'}
        event['seconds'] = round(current['time'], 3)
        events.append(event)
        cv2.imwrite(str(out / f'{source.stem}-{len(events)}-{event["event_type"]}.jpg'), kwargs['frame'])

    cfg = copy.deepcopy(config)
    cfg['behaviors']['running']['enabled'] = False
    cfg['web']['snapshots_dir'] = str(out / 'snapshots')
    with patch('src.core.pipeline.DatabaseService') as db:
        db.return_value.count_total_events.side_effect = lambda: len(events)
        pipeline = SurveillancePipeline(cfg)
    sink = Mock()
    sink.trigger_alert.side_effect = alert
    pipeline.alert_manager = sink
    pipeline.motion_fall_detector.alert_manager = sink
    pipeline.fighting_detector.alert_manager = sink

    class CachedModel:
        def track(self, **kwargs):
            row = poses[str(current['frame'])]
            boxes = np.array(row['boxes'], dtype=np.float32).reshape(-1, 4)
            ids = np.array(row['ids'], dtype=np.float32)
            kpts = np.array(row['kpts'], dtype=np.float32).reshape(-1, 17, 3)
            return [Mock(boxes=Mock(xyxy=torch.from_numpy(boxes), id=torch.from_numpy(ids)),
                         keypoints=Mock(data=torch.from_numpy(kpts)))]

    pipeline.yolo_model = CachedModel()
    writer = None
    index = processed = 0
    previous_states = {}
    started = time.monotonic()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index % stride:
                index += 1
                continue
            current.update(time=index / fps, frame=index)
            if str(index) not in poses:
                if model is None:
                    model = YOLO(config['system']['model_pose_path'])
                r = model.track(frame, persist=True, classes=[0], tracker='bytetrack.yaml',
                                conf=config['system']['confidence_threshold'], verbose=False, device='cpu')[0]
                found = r.boxes is not None and r.boxes.id is not None
                poses[str(index)] = {
                    'boxes': r.boxes.xyxy.cpu().tolist() if found else [],
                    'ids': r.boxes.id.cpu().tolist() if found else [],
                    'kpts': r.keypoints.data.cpu().tolist() if found else [],
                }
            rendered = pipeline.process_frame(frame, index / fps)
            states = {str(tid): t.current_state for tid, t in pipeline.track_manager.tracks.items()
                      if t.last_seen == index / fps and t.current_state != 'NORMAL'}
            if states != previous_states:
                transitions.append({'seconds': round(index / fps, 3), 'states': states})
                previous_states = states
            if writer is None:
                h, w = rendered.shape[:2]
                writer = cv2.VideoWriter(str(out / (source.stem + '-annotated.mp4')),
                                         cv2.VideoWriter_fourcc(*'mp4v'), fps / stride, (w, h))
                if not writer.isOpened():
                    raise RuntimeError('Cannot create annotated video')
            writer.write(rendered)
            if processed % max(1, round(fps / stride * 3)) == 0 and len(thumbs) < 12:
                thumb = cv2.resize(rendered, (384, 216))
                cv2.putText(thumb, f'{index / fps:.1f}s', (5, 205), 0, .6, (255,255,255), 2)
                thumbs.append(thumb)
            processed += 1
            index += 1
            if processed % 150 == 0:
                print(f'{source.name}: {index}/{total} frames', flush=True)
    finally:
        cap.release()
        if writer:
            writer.release()
        cache_path.write_text(json.dumps({'key': key, 'poses': poses}))
    if thumbs:
        thumbs += [np.zeros_like(thumbs[0])] * ((-len(thumbs)) % 3)
        cv2.imwrite(str(out / (source.stem + '-contact.jpg')),
                    np.concatenate([np.concatenate(thumbs[i:i+3], axis=1)
                                    for i in range(0, len(thumbs), 3)], axis=0))
    return {'video': source.name, 'frames_processed': processed, 'source_fps': fps,
            'stride': stride, 'elapsed_seconds': round(time.monotonic()-started, 2),
            'events': events, 'state_transitions': transitions}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('videos', nargs='*', default=['fight1.mp4', 'fall_sample.mp4',
                                                    '01_video.mp4', '02_video.mp4', 'people_walking.mp4'])
    parser.add_argument('--fps', type=float, default=15)
    parser.add_argument('--output', type=Path, default=Path('data/behavior_evaluation'))
    parser.add_argument('--check', action='store_true', help='Assert expected events for the reviewed sample clips')
    args = parser.parse_args()
    torch.set_num_threads(2)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(Path('configs/config.yaml').read_text(encoding='utf-8'))
    report = []
    for name in args.videos:
        row = evaluate(Path('data/test_videos') / name, cfg, out, args.fps)
        report.append(row)
        (out / 'results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({k:v for k,v in row.items() if k != 'state_transitions'}, ensure_ascii=False), flush=True)
    if args.check:
        expected = {'fight1.mp4': {'FIGHTING'}, 'fall_sample.mp4': {'FALLING'},
                    '01_video.mp4': set(), '02_video.mp4': set(), 'people_walking.mp4': set()}
        checks = []
        for row in report:
            if row['video'] not in expected:
                raise ValueError(f"No reviewed expectation for {row['video']}")
            actual = {event['event_type'] for event in row['events']}
            checks.append({'video': row['video'], 'passed': actual == expected[row['video']],
                           'expected': sorted(expected[row['video']]), 'actual': sorted(actual)})
        (out / 'checks.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
        if not all(check['passed'] for check in checks):
            raise AssertionError(checks)
        print(f'{len(checks)} video checks passed', flush=True)


if __name__ == '__main__':
    main()
