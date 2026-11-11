r"""Replay cached YOLO poses through the behavior detectors (no video, no YOLO, no torch).

Use this to retune running / fighting / falling thresholds in seconds:
    python -m scripts.replay_pose_cache --check
    python -m scripts.replay_pose_cache fight1 --verbose

Pose caches are written by `scripts.evaluate_behaviors_local` into
`data/behavior_evaluation/<video>-poses.json`. Frame order mirrors
`SurveillancePipeline.process_frame` (motion/fall -> fighting -> fall latch).
"""
import argparse
import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import yaml

from src.behaviors.fall_run import MotionAndFallDetector
from src.behaviors.fighting import FightingDetector
from src.tracking.track_manager import TrackManager

# Source FPS of the reviewed clips (from data/behavior_evaluation/results.json).
SOURCE_FPS = {'fight1': 25.0, 'fall_sample': 23.98, '01_video': 32.0,
              '02_video': 32.0, 'people_walking': 12.0}
# Reviewed expectations: what a human sees in each clip.
EXPECTED = {'fight1': {'FIGHTING'}, 'fall_sample': {'FALLING'},
            '01_video': set(), '02_video': set(), 'people_walking': set()}


def replay(name, config, cache_dir, verbose=False):
    poses = json.loads((cache_dir / f'{name}-poses.json').read_text())['poses']
    fps = SOURCE_FPS[name]
    behav = config['behaviors']
    tracks = TrackManager(track_ttl=config.get('tracking', {}).get('track_ttl', 5.0))
    alerts = []
    sink = Mock()
    sink.trigger_alert.side_effect = lambda **kw: alerts.append(kw)
    motion = MotionAndFallDetector(behav.get('running', {}), behav.get('falling', {}),
                                   alert_manager=sink)
    fighting = FightingDetector(behav.get('fighting', {}), sink)

    events = []
    for index in sorted(poses, key=int):
        row, t = poses[index], int(index) / fps
        boxes = np.array(row['boxes'], dtype=np.float32).reshape(-1, 4)
        kpts = np.array(row['kpts'], dtype=np.float32).reshape(-1, 17, 3)
        active = []
        for i, (box, tid) in enumerate(zip(boxes, row['ids'])):
            box = tuple(float(v) for v in box)
            track = tracks.update_track(int(tid), box, ((box[0] + box[2]) / 2, box[3]),
                                        kpts[i] if len(kpts) > i else None, t)
            motion.process(track, t, None, [])
            active.append(track)
        for item in fighting.process_all(active, t, None)['new_alerts']:
            alerts.append({'event_type': 'FIGHTING', **item})
        for track in active:
            if track.fall_phase == 'FALL_LATCHED':
                track.current_state = 'FALLING_ALERT'
        tracks.purge_stale_tracks(t)
        for a in alerts:
            events.append({'event_type': a['event_type'], 'track_id': a['track_id'],
                           'seconds': round(t, 3), 'info': a.get('extra_info', '')})
            if verbose:
                print(f'  {t:7.2f}s  {a["event_type"]:<9} #{a["track_id"]}  {a.get("extra_info", "")}')
        alerts.clear()
    return events


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('videos', nargs='*', default=list(EXPECTED))
    parser.add_argument('--cache', type=Path, default=Path('data/behavior_evaluation'))
    parser.add_argument('--config', type=Path, default=Path('configs/config.yaml'))
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--check', action='store_true', help='Fail if events differ from EXPECTED')
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    failed = []
    for name in args.videos:
        name = Path(name).stem
        events = replay(name, config, args.cache, args.verbose)
        actual = {e['event_type'] for e in events}
        ok = actual == EXPECTED.get(name, actual)
        counts = {k: sum(e['event_type'] == k for e in events) for k in sorted(actual)}
        print(f'{"PASS" if ok else "FAIL"}  {name:<15} expected={sorted(EXPECTED.get(name, []))} got={counts}')
        if not ok:
            failed.append(name)
    if args.check and failed:
        raise SystemExit(f'Behavior replay failed: {failed}')


if __name__ == '__main__':
    main()
