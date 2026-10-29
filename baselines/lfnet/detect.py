"""
detect.py — LFBNet inference. At inference time the thermal branch is the
grayscale of the input image (pseudo-thermal, same as training).

Usage:
  python -m baselines.lfnet.detect --source <img_or_dir> --weights weights/lfnet_last.pt
"""

import os
import glob
import argparse

import cv2
import numpy as np
import torch

from .lfnet_model import LFBNet
from .lfnet_dataset import to_thermal
from common.dataset import letterbox
from common.infer import postprocess, unletterbox, draw


def predict_image(model, bgr_img, img_size=320, device='cpu', conf=0.25, iou=0.45, nc=80):
    h0, w0 = bgr_img.shape[:2]
    thermal = to_thermal(bgr_img)
    lb, _, ratio, (dw, dh) = letterbox(bgr_img, img_size)
    lt, _, _, _ = letterbox(thermal, img_size)

    xr = torch.from_numpy(np.ascontiguousarray(
        lb[:, :, ::-1]).transpose(2, 0, 1)).float()[None] / 255.0
    xt = torch.from_numpy(np.ascontiguousarray(
        lt).transpose(2, 0, 1)).float()[None] / 255.0
    xr, xt = xr.to(device), xt.to(device)
    with torch.no_grad():
        pred = model(xr, xt)
    det = postprocess(pred, conf, iou, nc=nc)[0]
    return unletterbox(det, ratio, dw, dh, w0, h0)


def list_images(src):
    if os.path.isdir(src):
        out = []
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
            out += glob.glob(os.path.join(src, ext))
        return sorted(out)
    return [src]


def parse_opt():
    p = argparse.ArgumentParser(description='LFBNet inference')
    p.add_argument('--source', required=True)
    p.add_argument('--weights', default='')
    p.add_argument('--img-size', type=int, default=320)
    p.add_argument('--conf', type=float, default=0.25)
    p.add_argument('--iou', type=float, default=0.45)
    p.add_argument('--device', default='')
    p.add_argument('--save-dir', default='')
    p.add_argument('--nc', type=int, default=80)
    return p.parse_args()


def main():
    opt = parse_opt()
    device = torch.device(opt.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    model = LFBNet(nc=opt.nc)
    if opt.weights and os.path.isfile(opt.weights):
        ck = torch.load(opt.weights, map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model'] if 'model' in ck else ck, strict=False)
        print('loaded weights:', opt.weights)
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
