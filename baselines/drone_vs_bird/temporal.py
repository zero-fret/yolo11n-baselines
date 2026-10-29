"""
temporal.py — temporal consistency / missing-frame interpolation
(paper Fig. 4 / Stage 2).

Given a list of per-frame detection arrays (each np [n,6] = x1,y1,x2,y2,conf,cls),
this module interpolates detections into frames that came back empty.

Rules (from the paper):
  * within +/- window frames of a missing frame, match same-class detections
    by IoU >= 0.1;
  * linearly interpolate the box between the two endpoints;
  * interpolated box confidence = mean(endpoint confidences) * 0.96;
  * drop interpolated boxes near the image border or overlapping existing dets.

NOTE: this module is *stateless* — it receives an already-collected list of
per-frame detections. To wire it into your own video pipeline, collect one
detection array per frame (in chronological order) and pass the list to
`interpolate_sequence(frames, W, H)`.
"""

import numpy as np

WINDOW = 3
IOU_THR = 0.1
CONF_FACTOR = 0.96
BORDER = 5
EXIST_IOU = 0.3


def iou(a, b):
    """a [4], b [n,4] -> [n]."""
    x1 = np.maximum(a[0], b[:, 0])
    y1 = np.maximum(a[1], b[:, 1])
    x2 = np.minimum(a[2], b[:, 2])
    y2 = np.minimum(a[3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = max((a[2] - a[0]) * (a[3] - a[1]), 1e-9)
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa + ab - inter + 1e-9)


def match_pairs(a, b, iou_thr=IOU_THR):
    """Greedy same-class matching between two frames; returns [(ia, ib)]."""
    pairs = []
    used_b = set()
    for ia, da in enumerate(a):
        cand = [j for j, db in enumerate(b) if int(da[5]) == int(db[5])]
        best, bj = iou_thr, -1
        for j in cand:
            if j in used_b:
                continue
            v = float(iou(da[:4], b[j:j + 1, :4])[0])
            if v >= best:
                best, bj = v, j
        if bj >= 0:
            used_b.add(bj)
            pairs.append((ia, bj))
    return pairs


def interpolate_sequence(frames, W, H, window=WINDOW):
    """frames: list[np [n,6]] (in frame order). Mutates and returns frames."""
    T = len(frames)
    for t in range(T):
        if len(frames[t]):
            continue
        for k in range(1, window + 1):
            t0, t1 = t - k, t + k
            if t0 < 0 or t1 >= T:
                continue
            a, b = frames[t0], frames[t1]
            if not len(a) or not len(b):
                continue
            for ia, ib in match_pairs(a, b):
                f0 = k / (2 * k)
                box = a[ia, :4] * (1 - f0) + b[ib, :4] * f0
                conf = (a[ia, 4] + b[ib, 4]) / 2 * CONF_FACTOR
                cls = a[ia, 5]
                if (box[0] < BORDER or box[1] < BORDER
                        or box[2] > W - BORDER or box[3] > H - BORDER):
                    continue
                row = np.array([[box[0], box[1], box[2], box[3], conf, cls]],
                               np.float32)
                if len(frames[t]) and float(iou(box, frames[t][:, :4]).max()) > EXIST_IOU:
                    continue
                frames[t] = np.concatenate([frames[t], row], 0) \
                    if len(frames[t]) else row
    return frames
