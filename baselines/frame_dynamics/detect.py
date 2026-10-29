"""
detect.py — Frame Dynamics inference.

For image directories: each image is treated as an isolated frame (diff = 0,
degenerate to grayscale-only input). This is a smoke-only mode.

For real video inference, wire your own frame iterator — see fd_dataset.py
docstring. The model itself is vanilla YOLO11n and the predict_fd() helper
below works on any 3-channel HxWx3 uint8 frame-dynamics image.

Usage:
  python -m baselines.frame_dynamics.detect --source <img_or_dir> \
      --weights weights/frame_dynamics_last.pt
"""

import os
import glob
import argparse

import cv2
import numpy as np
import torch

from yolocore.model import YOLO11n
from common.infer import postprocess, unletterbox, draw
from common.dataset import letterbox


def predict_fd(model, fd_img, img_size, device, conf, iou, nc):
    """fd_img: HxWx3 uint8 [gray, diff1, diff2]."""
    h0, w0 = fd_img.shape[:2]
    lb, _, ratio, (dw, dh) = letterbox(fd_img, img_size)
    x = np.ascontiguousarray(lb.transpose(2, 0, 1)).astype(np.float32) / 255.0
    x = torch.from_numpy(x)[None].to(device)
    with torch.no_grad():
        pred = model(x)
    det = postprocess(pred, conf, iou, nc)[0]
    return unletterbox(det, ratio, dw, dh, w0, h0)


def list_images(src):
    if os.path.isdir(src):
        out = []
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
            out += glob.glob(os.path.join(src, ext))
        return sorted(out)
    return [src]


def parse_opt():
    p = argparse.ArgumentParser(description='Frame Dynamics inference')
    p.add_argument('--source', required=True)
    p.add_argument('--weights', default='')
    p.add_argument('--img-size', type=int, default=640)
    p.add_argument('--conf', type=float, default=0.25)
    p.add_argument('--iou', type=float, default=0.45)
    p.add_argument('--device', default='')
    p.add_argument('--save-dir', default='')
    p.add_argument('--nc', type=int, default=80)
    return p.parse_opt()


def main():
    opt = parse_opt()
    device = torch.device(opt.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    model = YOLO11n(nc=opt.nc)
    if opt.weights and os.path.isfile(opt.weights):
        ck = torch.load(opt.weights, map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model'] if 'model' in ck else ck, strict=False)
        print('loaded weights:', opt.weights)
    else:
        print('WARNING: no valid --weights, random init')
    model.h23.stride = torch.tensor([8., 16., 32.])
    model.to(device).eval()

    if opt.save_dir:
        os.makedirs(opt.save_dir, exist_ok=True)

    for ip in list_images(opt.source):
        img = cv2.imread(ip)
        if img is None:
            continue
        # Single-image mode: no adjacent frames, diff channels = 0.
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        fd = cv2.merge([gray, np.zeros_like(gray), np.zeros_like(gray)])
        det = predict_fd(model, fd, opt.img_size, device, opt.conf, opt.iou, opt.nc)
        print(f'{os.path.basename(ip)}: {len(det)} detections')
        if opt.save_dir:
            vis = draw(img.copy(), det)
            cv2.imwrite(os.path.join(opt.save_dir, os.path.basename(ip)), vis)


if __name__ == '__main__':
    main()
