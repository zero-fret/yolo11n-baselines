"""
copypaste.py — copy-paste augmentation (paper Sec. B / Fig. 5).

Builds a patch library from the standard YOLO training set (crop each labelled
box with a small margin), then during training randomly pastes perturbed
patches onto the image. Uses OpenCV/numpy (no Albumentations dependency).

This version is decoupled from any video format: it reads
<data_root>/images/train/*.jpg + <data_root>/labels/train/*.txt directly.
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from common.dataset import (img_list, img_to_label_path, load_label,
                            letterbox, augment_hsv, PAD_VALUE)


def build_patch_library(data_root, split='train', max_patches=200):
    """Crop labelled objects from the training set; return list of BGR patches."""
    images_root = os.path.join(data_root, 'images', split)
    labels_root = os.path.join(data_root, 'labels', split)
    files = img_list(images_root)
    step = max(1, len(files) // max_patches)
    patches = []
    for ip in files[::step]:
        img = cv2.imread(ip)
        if img is None:
            continue
        txt = img_to_label_path(ip, images_root, labels_root)
        boxes = load_label(txt)   # [N,5]=(cls,cx,cy,w,h) normalized
        if not len(boxes):
            continue
        H, W = img.shape[:2]
        for b in boxes:
            _, cx, cy, w, h = b
            cx, cy, w, h = cx * W, cy * H, w * W, h * H
            mx, my = w * 0.05, h * 0.05
            x1 = int(max(0, cx - w / 2 - mx))
            y1 = int(max(0, cy - h / 2 - my))
            x2 = int(min(W, cx + w / 2 + mx))
            y2 = int(min(H, cy + h / 2 + my))
            patch = img[y1:y2, x1:x2]
            if patch.shape[0] >= 4 and patch.shape[1] >= 4:
                patches.append(patch)
        if len(patches) >= max_patches:
            break
    return patches


def perturb_patch(patch):
    """Brightness / contrast / blur perturbation."""
    p = patch.astype(np.float32)
    alpha = np.random.uniform(0.7, 1.3)
    beta = np.random.uniform(-20, 20)
    p = np.clip(p * alpha + beta, 0, 255).astype(np.uint8)
    if np.random.random() < 0.3:
        k = int(np.random.choice((3, 5)))
        p = cv2.GaussianBlur(p, (k, k), 0)
    return p


class CopyPasteDataset(Dataset):
    def __init__(self, data_root, split='train', img_size=640, augment=True,
                 paste_prob=0.5, max_patches=200):
        self.data_root = data_root
        self.images_root = os.path.join(data_root, 'images', split)
        self.labels_root = os.path.join(data_root, 'labels', split)
        self.img_size = img_size
        self.augment = augment
        self.paste_prob = paste_prob
        self.files = img_list(self.images_root)
        self.patches = build_patch_library(data_root, split, max_patches)

    def __len__(self):
        return len(self.files)

    def _paste(self, img_u8, targets):
        """img_u8: letterbox BGR uint8 [S,S,3]; targets [n,5]=(cls,x1,y1,x2,y2)."""
        if not self.patches or np.random.random() > self.paste_prob:
            return img_u8, targets
        S = img_u8.shape[0]
        patch = perturb_patch(self.patches[np.random.randint(len(self.patches))])
        scale = np.random.uniform(0.03, 0.25)
        nw = max(4, int(round(S * scale * np.random.uniform(0.7, 1.4))))
        nh = max(4, int(round(patch.shape[0] * nw / patch.shape[1])))
        patch = cv2.resize(patch, (nw, nh))
        x1 = np.random.randint(0, max(1, S - nw))
        y1 = np.random.randint(0, max(1, S - nh))

        m = np.ones((nh, nw), np.float32)
        edge = 2
        m[:edge, :] *= np.linspace(0, 1, edge)[:, None]
        m[-edge:, :] *= np.linspace(1, 0, edge)[:, None]
        m[:, :edge] *= np.linspace(0, 1, edge)[None, :]
        m[:, -edge:] *= np.linspace(1, 0, edge)[None, :]
        roi = img_u8[y1:y1 + nh, x1:x1 + nw].astype(np.float32)
        blended = (patch.astype(np.float32) * m[..., None]
                   + roi * (1 - m[..., None])).astype(np.uint8)
        img_u8[y1:y1 + nh, x1:x1 + nw] = blended

        row = np.array([[0.0, x1, y1, x1 + nw, y1 + nh]], np.float32)
        targets = np.concatenate([targets, row], 0) if len(targets) else row
        return img_u8, targets

    def __getitem__(self, idx):
        ip = self.files[idx]
        img = cv2.imread(ip)
        if img is None:
            img = np.full((self.img_size, self.img_size, 3), PAD_VALUE, np.uint8)
            boxes = np.zeros((0, 5), np.float32)
        else:
            txt = img_to_label_path(ip, self.images_root, self.labels_root)
            boxes = load_label(txt)
            img, boxes, _, _ = letterbox(img, self.img_size, boxes)

        if self.augment:
            if np.random.random() < 0.5:
                img = img[:, ::-1, :]
                if len(boxes):
                    a, b = boxes[:, 1].copy(), boxes[:, 3].copy()
                    boxes[:, 1] = self.img_size - b
                    boxes[:, 3] = self.img_size - a
            augment_hsv(img)
            img, boxes = self._paste(img, boxes)

        img = np.ascontiguousarray(img[:, :, ::-1])
        img_t = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        if len(boxes):
            t = torch.from_numpy(np.asarray(boxes, np.float32))
            keep = (t[:, 3] > t[:, 1] + 1e-3) & (t[:, 4] > t[:, 2] + 1e-3)
            t = t[keep]
        else:
            t = torch.zeros((0, 5), dtype=torch.float32)
        return img_t, t


def collate_copypaste(batch):
    imgs, ts = zip(*batch)
    imgs = torch.stack(imgs, 0)
    out = []
    for b, t in enumerate(ts):
        if len(t):
            bi = torch.full((len(t), 1), b, dtype=torch.float32)
            out.append(torch.cat([bi, t], 1))
    targets = torch.cat(out, 0) if out else torch.zeros((0, 6), dtype=torch.float32)
    return imgs, targets


def create_copypaste_loader(data_root, split='train', img_size=640, batch_size=16,
                            augment=True, shuffle=None, workers=2,
                            paste_prob=0.5, max_patches=200):
    ds = CopyPasteDataset(data_root, split, img_size, augment, paste_prob, max_patches)
    if shuffle is None:
        shuffle = augment
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, collate_fn=collate_copypaste,
                      drop_last=augment, pin_memory=False)
