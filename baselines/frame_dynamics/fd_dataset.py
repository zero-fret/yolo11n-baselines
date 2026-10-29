"""
fd_dataset.py — Frame Dynamics dataset.

Paper (A Simple Detector with Frame Dynamics, CVPRW 2025) replaces the
3-channel RGB input with:
    ch0 = grayscale current frame I_t
    ch1 = |I_t - I_{t-1}|   (difference vs previous frame)
    ch2 = |I_t - I_{t-2}|   (difference vs frame two steps back)

Motion objects pop out on ch1/ch2; static background is suppressed. The
YOLO11n network itself is untouched (still 3-channel input) — so compute
cost is identical to vanilla YOLO11n.

================================================================
IMPORTANT — temporal frame loading
================================================================
Standard YOLO-format directories (images/{split}/*.jpg + labels/{split}/*.txt)
are *static-image* layouts — they do NOT encode which frame is the previous
one. To produce real frame-dynamics inputs you must plug in your own temporal
frame iterator.

This file ships TWO datasets:
  * `StaticFDDataset` — a degraded fallback that treats every image as an
    isolated frame (diff channels = 0, equivalent to grayscale-only input).
    It is intended for smoke tests and CI only.
  * `FrameDynamicsDataset` — the real thing. Its constructor takes a
    `frame_provider` callable that yields (cur_bgr, prev_bgr, prev2_bgr,
    targets) tuples; you provide this to match your own video storage.

TODO (for the user): implement `frame_provider(data_root, split, img_size)`
on your side. A minimal signature:

    def frame_provider(data_root, split, img_size):
        # yield tuples of (cur_bgr_uint8, prev_bgr_uint8, prev2_bgr_uint8,
        #                  targets_xyxy_px_array)
        ...

Then pass it in:
    loader = create_dataloader('data', 'train', 640, 16, augment=True,
                               frame_provider=your_provider_fn)
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from common.dataset import (img_list, img_to_label_path, load_label,
                            letterbox, augment_hsv, PAD_VALUE)


def _gray(bgr):
    """BGR -> grayscale float32."""
    return bgr.astype(np.float32).mean(axis=2)


def make_fd_image(cur_bgr, prev_bgr, prev2_bgr):
    """Stack three frames into the 3-channel frame-dynamics image (uint8 HxWx3)."""
    g_cur = _gray(cur_bgr)
    g_prev = _gray(prev_bgr)
    g_prev2 = _gray(prev2_bgr)
    diff1 = np.abs(g_cur - g_prev)
    diff2 = np.abs(g_cur - g_prev2)
    return np.stack([g_cur, diff1, diff2], axis=-1).astype(np.uint8)


# ------------------------------------------------------------------
# Static fallback: every image isolated, diff channels = 0.
# Only for smoke tests / users without a video storage.
# ------------------------------------------------------------------
class StaticFDDataset(Dataset):
    def __init__(self, data_root, split='train', img_size=640, augment=False):
        self.data_root = data_root
        self.images_root = os.path.join(data_root, 'images', split)
        self.labels_root = os.path.join(data_root, 'labels', split)
        self.img_size = img_size
        self.augment = augment
        self.files = img_list(self.images_root)
        if not self.files:
            raise FileNotFoundError(f'No images under {self.images_root}')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        ip = self.files[idx]
        img = cv2.imread(ip)
        if img is None:
            img = np.full((self.img_size, self.img_size, 3), PAD_VALUE, np.uint8)
            boxes = np.zeros((0, 5), np.float32)
        else:
            txt = img_to_label_path(ip, self.images_root, self.labels_root)
            boxes = load_label(txt)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Degenerate frame-dynamics: only ch0 = gray, ch1/ch2 = 0
            fd = cv2.merge([gray, np.zeros_like(gray), np.zeros_like(gray)])
            img = fd
            img, boxes, _, _ = letterbox(img, self.img_size, boxes)

        if self.augment:
            if np.random.random() < 0.5:
                img = img[:, ::-1, :]
                if len(boxes):
                    a, b = boxes[:, 1].copy(), boxes[:, 3].copy()
                    boxes[:, 1] = self.img_size - b
                    boxes[:, 3] = self.img_size - a

        img = np.ascontiguousarray(img.transpose(2, 0, 1))
        img_t = torch.from_numpy(img).float() / 255.0
        if len(boxes):
            targets = torch.from_numpy(boxes).float()
            keep = (targets[:, 3] > targets[:, 1] + 1e-3) & \
                   (targets[:, 4] > targets[:, 2] + 1e-3)
            targets = targets[keep]
        else:
            targets = torch.zeros((0, 5), dtype=torch.float32)
        return img_t, targets


# ------------------------------------------------------------------
# Real frame-dynamics dataset (uses user-provided frame_provider).
# ------------------------------------------------------------------
class FrameDynamicsDataset(Dataset):
    """Requires a `frame_provider(data_root, split, img_size)` callable that
    iterates over temporal frame triples and yields
    (cur_bgr_uint8, prev_bgr_uint8, prev2_bgr_uint8, targets_xyxy_px)."""

    def __init__(self, data_root, split='train', img_size=640, augment=False,
                 frame_provider=None):
        if frame_provider is None:
            raise ValueError(
                'frame_provider is required for FrameDynamicsDataset. '
                'Either implement one (see module docstring) or use '
                'StaticFDDataset for smoke tests.')
        self.img_size = img_size
        self.augment = augment
        self.frames = list(frame_provider(data_root, split, img_size))

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        cur, prev, prev2, boxes = self.frames[idx]
        fd = make_fd_image(cur, prev, prev2)
        fd, boxes, _, _ = letterbox(fd, self.img_size, boxes)

        if self.augment and np.random.random() < 0.5:
            fd = fd[:, ::-1, :]
            if len(boxes):
                a, b = boxes[:, 1].copy(), boxes[:, 3].copy()
                boxes[:, 1] = self.img_size - b
                boxes[:, 3] = self.img_size - a

        img_t = torch.from_numpy(np.ascontiguousarray(
            fd.transpose(2, 0, 1)).astype(np.float32) / 255.0)
        if len(boxes):
            targets = torch.from_numpy(boxes).float()
        else:
            targets = torch.zeros((0, 5), dtype=torch.float32)
        return img_t, targets


def collate_fd(batch):
    imgs, ts = zip(*batch)
    imgs = torch.stack(imgs, 0)
    out = []
    for b, t in enumerate(ts):
        if len(t):
            bi = torch.full((len(t), 1), b, dtype=torch.float32)
            out.append(torch.cat([bi, t], 1))
    targets = torch.cat(out, 0) if out else torch.zeros((0, 6), dtype=torch.float32)
    return imgs, targets


def create_dataloader(data_root, split='train', img_size=640, batch_size=8,
                     augment=False, shuffle=None, workers=2,
                     frame_provider=None):
    if frame_provider is None:
        ds = StaticFDDataset(data_root, split, img_size, augment)
    else:
        ds = FrameDynamicsDataset(data_root, split, img_size, augment,
                                  frame_provider=frame_provider)
    if shuffle is None:
        shuffle = augment
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, collate_fn=collate_fd,
                      drop_last=augment, pin_memory=False)
