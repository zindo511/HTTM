"""Synthetic COCO-17 skeletons for detector tests."""
import math
import numpy as np


def skeleton(cx, top, h, swing=0.0, lean_deg=0.0, conf=0.9):
    """Upright person centred at cx, head at `top`, height `h` pixels.

    `swing` in [-1, 1] moves wrists/elbows forward/back like an arm swing
    (wrist amplitude 0.25 BH). `lean_deg` tilts shoulders forward.
    """
    p = np.zeros((17, 3), dtype=np.float32)
    lean = math.tan(math.radians(lean_deg)) * 0.3 * h
    rows = {
        0: (0, .06), 1: (-.03, .05), 2: (.03, .05), 3: (-.06, .06), 4: (.06, .06),
        5: (-.12, .2), 6: (.12, .2), 11: (-.09, .52), 12: (.09, .52),
        13: (-.09, .74), 14: (.09, .74), 15: (-.09, .97), 16: (.09, .97),
    }
    for idx, (dx, dy) in rows.items():
        shift = lean if idx <= 6 else 0.0
        p[idx] = (cx + dx * h + shift, top + dy * h, conf)
    for side, (shoulder, elbow, wrist, sign) in enumerate(((5, 7, 9, 1), (6, 8, 10, -1))):
        sx, sy = p[shoulder, 0], p[shoulder, 1]
        s = swing * sign
        p[elbow] = (sx + 0.12 * h * s, sy + 0.15 * h, conf)
        p[wrist] = (sx + 0.25 * h * s, sy + 0.28 * h, conf)
    return p


def bbox(cx, top, h, w_ratio=0.4):
    w = h * w_ratio
    return (cx - w / 2, top, cx + w / 2, top + h)
