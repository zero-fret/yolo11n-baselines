"""
detect.py — Drone-vs-Bird multi-scale image inference.

Usage:
  python -m baselines.drone_vs_bird.detect --source <img_or_dir> \
      --weights weights/drone_vs_bird_last.pt
  Add --single-scale to fall back to full-image inference.
"""

import os
import glob
import argparse

import cv2
import torch

from yolocore.model import YOLO11n
from common.infer import predict_image, draw
from .multiscale import detect_multiscale


def load_model(weights, nc):
    model = YOLO11n(nc=nc)
    if weights and os.path.isfile(weights):
        ck = torch.load(weights, map_location='cpu', weights_only=False)
        sd = ck['model'] if 'model' in ck else ck
        missing, _ = model.load_state_dict(sd, strict=False)
        print('loaded weights:', weights)
    else:
        print('WARNING: no valid --weights, random init')
    return model


def list_images(src):
    if os.path.isdir(src):
        out = []
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
            out += glob.glob(os.path.join(src, ext))
        return sorted(out)
    return [src]


def parse_opt():
    p = argparse.ArgumentParser(description='Drone-vs-Bird multiscale inference')
    p.add_argument('--source', required=True)
    p.add_argument('--weights', default='')
    p.add_argument('--img-size', type=int, default=640)
    p.add_argument('--conf', type=float, default=0.25)
    p.add_argument('--iou', type=float, default=0.45)
    p.add_argument('--merge-iou', type=float, default=0.1)
    p.add_argument('--single-scale', action='store_true')
    p.add_argument('--device', default='')
    p.add_argument('--save-dir', default='')
    p.add_argument('--nc', type=int, default=80)
    return p.parse_args()


def main():
    opt = parse_opt()
    device = torch.device(opt.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    model = load_model(opt.weights, opt.nc).to(device).eval()
    if opt.save_dir:
        os.makedirs(opt.save_dir, exist_ok=True)

    for ip in list_images(opt.source):
        frame = cv2.imread(ip)
        if frame is None:
            continue
        if opt.single_scale:
            det = predict_image(model, frame, opt.img_size, device,
                                opt.conf, opt.iou, opt.nc)
        else:
            det = detect_multiscale(model, frame, opt.img_size, device,
                                    opt.conf, opt.iou, opt.nc, opt.merge_iou)
        print(f'{os.path.basename(ip)}: {len(det)} detections')
        if opt.save_dir:
            vis = draw(frame.copy(), det)
            cv2.imwrite(os.path.join(opt.save_dir, os.path.basename(ip)), vis)


if __name__ == '__main__':
    main()
