"""
multiscale.py — multi-scale tiling inference (paper Fig. 4 / Stage 1).

  * full-frame resize to model input;
  * 4 corner crops at 55% coverage (overlapping), each resized to model input;
  * map every view's detections back to full-image coords;
  * cross-view global NMS at IoU=0.1.
"""

import cv2
import numpy as np
import torch

from common.infer import nms, xywh2xyxy

COVERAGE = 0.55


def make_views(frame, S):
    H, W = frame.shape[:2]
    views = []

    def to_tensor(crop):
        img = cv2.resize(crop, (S, S), interpolation=cv2.INTER_LINEAR)
        img = np.ascontiguousarray(img[:, :, ::-1]).transpose(2, 0, 1)
        return torch.from_numpy(img).float() / 255.0

    views.append((to_tensor(frame), {'kind': 'full', 'W': W, 'H': H}))

    sw, sh = int(round(W * COVERAGE)), int(round(H * COVERAGE))
    corners = [(0, 0), (W - sw, 0), (0, H - sh), (W - sw, H - sh)]
    for ox, oy in corners:
        crop = frame[oy:oy + sh, ox:ox + sw]
        views.append((to_tensor(crop),
                      {'kind': 'slice', 'ox': ox, 'oy': oy,
                       'sw': sw, 'sh': sh}))
    return views


def map_back(det, info, S):
    if not len(det):
        return det
    if info['kind'] == 'full':
        cw, ch, ox, oy = info['W'], info['H'], 0, 0
    else:
        cw, ch, ox, oy = info['sw'], info['sh'], info['ox'], info['oy']
    out = det.copy()
    out[:, [0, 2]] = out[:, [0, 2]] * (cw / S) + ox
    out[:, [1, 3]] = out[:, [1, 3]] * (ch / S) + oy
    return out


@torch.no_grad()
def detect_views(model, view_tensors, device, conf, iou, nc):
    x = view_tensors.to(device)
    out = model(x)
    z = []
    for i in range(x.shape[0]):
        boxes = xywh2xyxy(out[i, :4].transpose(0, 1))
        sc, cl = out[i, 4:4 + nc].max(0)
        keep = sc > conf
        boxes, sc, cl = boxes[keep], sc[keep], cl[keep]
        idx = nms(boxes, sc, cl, iou)
        if idx:
            d = torch.cat([boxes[idx], sc[idx, None],
                           cl[idx, None].float()], 1).cpu().numpy()
        else:
            d = np.zeros((0, 6), np.float32)
        z.append(d)
    return z


def detect_multiscale(model, frame, S=640, device='cuda', conf=0.25,
                      iou=0.45, nc=1, merge_iou=0.1):
    views = make_views(frame, S)
    vt = torch.stack([v[0] for v in views], 0)
    dets = detect_views(model, vt, device, conf, iou, nc)
    all_d = []
    for d, (_, info) in zip(dets, views):
        all_d.append(map_back(d, info, S))
    all_d = np.concatenate(all_d, 0) if any(len(d) for d in all_d) else \
        np.zeros((0, 6), np.float32)
    if not len(all_d):
        return all_d
    b = torch.from_numpy(all_d[:, :4])
    s = torch.from_numpy(all_d[:, 4])
    c = torch.from_numpy(all_d[:, 5])
    keep = nms(b, s, c, merge_iou)
    return all_d[np.asarray(keep, dtype=np.int64)]
