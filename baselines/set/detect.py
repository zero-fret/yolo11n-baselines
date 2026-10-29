"""
detect.py — SET inference. At inference time SET is a plain YOLO11n; the
training-only denoiser keys are silently ignored.
"""

import os
import glob
import argparse

import cv2
import torch

from yolocore.model import YOLO11n
from common.infer import predict_image, draw


def parse_opt():
    p = argparse.ArgumentParser(description='SET inference')
    p.add_argument('--source', required=True)
    p.add_argument('--weights', default='')
    p.add_argument('--img-size', type=int, default=640)
    p.add_argument('--conf', type=float, default=0.25)
    p.add_argument('--iou', type=float, default=0.45)
    p.add_argument('--device', default='')
    p.add_argument('--save-dir', default='')
    p.add_argument('--nc', type=int, default=80)
    return p.parse_args()


def list_images(src):
    if os.path.isdir(src):
        out = []
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
            out += glob.glob(os.path.join(src, ext))
        return sorted(out)
    return [src]


def main():
    opt = parse_opt()
    device = torch.device(opt.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    model = YOLO11n(nc=opt.nc)
    if opt.weights and os.path.isfile(opt.weights):
        ck = torch.load(opt.weights, map_location='cpu', weights_only=False)
        sd = ck['model'] if 'model' in ck else ck
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f'loaded weights: {opt.weights}')
        print(f'  (ignored {len(unexpected)} training-only keys)')
    else:
        print('WARNING: no valid --weights, random init')
    model.to(device).eval()

    if opt.save_dir:
        os.makedirs(opt.save_dir, exist_ok=True)
    for ip in list_images(opt.source):
        img = cv2.imread(ip)
        if img is None:
            continue
        det = predict_image(model, img, opt.img_size, device, opt.conf, opt.iou, opt.nc)
        print(f'{os.path.basename(ip)}: {len(det)} detections')
        if opt.save_dir:
            vis = draw(img.copy(), det)
            cv2.imwrite(os.path.join(opt.save_dir, os.path.basename(ip)), vis)


if __name__ == '__main__':
    main()
