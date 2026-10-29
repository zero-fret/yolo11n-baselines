"""
lfnet_dataset.py — RGB-T dataset where the thermal stream is synthesised
from the RGB image's grayscale (no real thermal data shipped).

Standard YOLO format on disk: images/{split}/*.jpg + labels/{split}/*.txt.
Outputs (xr, xt, targets):
  xr      visible-light [3,S,S]
  xt      pseudo-thermal [3,S,S] (grayscale broadcast to 3 channels)
  targets [N,5] = (cls, x1, y1, x2, y2) in letterbox pixel coords
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from common.dataset import (img_list, img_to_label_path, load_label,
                            letterbox, augment_hsv, PAD_VALUE)


def to_thermal(bgr):
    """BGR -> pseudo-thermal (grayscale broadcast to 3 channels)."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.merge([g, g, g])


class RGBTDataset(Dataset):
    def __init__(self, data_root, split='train', img_size=320, augment=False):
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
            thermal = img.copy()
            boxes = np.zeros((0, 5), np.float32)
        else:
            txt = img_to_label_path(ip, self.images_root, self.labels_root)
            boxes = load_label(txt)
            thermal = to_thermal(img)
            img, boxes, _, _ = letterbox(img, self.img_size, boxes)
            thermal, _, _, _ = letterbox(thermal, self.img_size)

        if self.augment:
            augment_hsv(img)
            if np.random.random() < 0.5:
                img = img[:, ::-1, :]
                thermal = thermal[:, ::-1, :]
                if len(boxes):
                    a, b = boxes[:, 1].copy(), boxes[:, 3].copy()
                    boxes[:, 1] = self.img_size - b
                    boxes[:, 3] = self.img_size - a

        # xr: RGB (BGR->RGB), xt: 3-channel grayscale (already BGR=RGB since gray)
        xr = torch.from_numpy(np.ascontiguousarray(
            img[:, :, ::-1]).transpose(2, 0, 1)).float() / 255.0
        xt = torch.from_numpy(np.ascontiguousarray(
            thermal).transpose(2, 0, 1)).float() / 255.0

        if len(boxes):
            t = torch.from_numpy(boxes).float()
            keep = (t[:, 3] > t[:, 1] + 1e-3) & (t[:, 4] > t[:, 2] + 1e-3)
            t = t[keep]
        else:
            t = torch.zeros((0, 5), dtype=torch.float32)
        return xr, xt, t


def collate_rgbt(batch):
    xr, xt, bs = zip(*batch)
    xr = torch.stack(xr, 0)
    xt = torch.stack(xt, 0)
    targets = []
    for b, t in enumerate(bs):
        if len(t):
            bi = torch.full((len(t), 1), b, dtype=torch.float32)
            targets.append(torch.cat([bi, t], 1))   # b,cls,x1,y1,x2,y2
    targets = torch.cat(targets, 0) if targets else torch.zeros((0, 6))
    return xr, xt, targets


def create_dataloader(data_root, split='train', img_size=320, batch_size=2,
                     augment=False, shuffle=None, workers=2):
    ds = RGBTDataset(data_root, split, img_size, augment)
    if shuffle is None:
        shuffle = augment
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, collate_fn=collate_rgbt,
                      drop_last=augment, pin_memory=False)
