"""
video_detect.py — Drone-vs-Bird video detection + temporal interpolation.

Pipeline: per-frame detection (default multi-scale) ->
          temporal.interpolate_sequence fills missing frames ->
          draw + write to video.

Usage:
  python -m baselines.drone_vs_bird.video_detect --source clip.mp4 \
      --weights weights/drone_vs_bird_last.pt --out out.mp4 --max-frames 30

NOTE on temporal data collection: this script already handles the video
decode / frame-stepping loop via cv2.VideoCapture. If you want to plug in
your own frame source (e.g. a custom dataset iterator, a real-time camera,
or a list of already-decoded frames), replace the `while True: cap.read()`
block with your own generator that yields BGR frames in chronological order,
then feed them into detect_multiscale() / predict_image() and pass the
resulting per-frame detections to temporal.interpolate_sequence().
"""

import os
import argparse

import cv2
import numpy as np
import torch

from yolocore.model import YOLO11n
from common.infer import predict_image, draw
from .multiscale import detect_multiscale
from .temporal import interpolate_sequence


def parse_opt():
    p = argparse.ArgumentParser(description='Drone-vs-Bird video detection')
    p.add_argument('--source', required=True)
    p.add_argument('--weights', default='')
    p.add_argument('--out', default='')
    p.add_argument('--img-size', type=int, default=640)
    p.add_argument('--conf', type=float, default=0.25)
    p.add_argument('--iou', type=float, default=0.45)
    p.add_argument('--merge-iou', type=float, default=0.1)
    p.add_argument('--single-scale', action='store_true')
    p.add_argument('--max-frames', type=int, default=0)
    p.add_argument('--device', default='')
    p.add_argument('--nc', type=int, default=80)
    return p.parse_opt()


def main():
    opt = parse_opt()
    device = torch.device(opt.device or ('cuda' if torch.cuda.is_available() else 'cpu'))

    model = YOLO11n(nc=opt.nc)
    if opt.weights and os.path.isfile(opt.weights):
        ck = torch.load(opt.weights, map_location='cpu', weights_only=False)
        sd = ck['model'] if 'model' in ck else ck
        model.load_state_dict(sd, strict=False)
        print('loaded weights:', opt.weights)
    else:
        print('WARNING: random init')
    model.to(device).eval()

    cap = cv2.VideoCapture(opt.source)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    writer = None
    if opt.out:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(opt.out, fourcc, fps, (W, H))

    frames_dets, raw_frames = [], []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if opt.single_scale:
            d = predict_image(model, frame, opt.img_size, device,
                              opt.conf, opt.iou, opt.nc)
        else:
            d = detect_multiscale(model, frame, opt.img_size, device,
                                  opt.conf, opt.iou, opt.nc, opt.merge_iou)
        frames_dets.append(np.asarray(d, np.float32) if len(d)
                           else np.zeros((0, 6), np.float32))
        raw_frames.append(frame)
        i += 1
        if opt.max_frames and i >= opt.max_frames:
            break
    cap.release()

    frames_dets = interpolate_sequence(frames_dets, W, H)
    for frame, d in zip(raw_frames, frames_dets):
        vis = draw(frame.copy(), d)
        if writer:
            writer.write(vis)
    if writer:
        writer.release()
    n_det = sum(len(d) for d in frames_dets)
    print(f'processed {i} frames, total detections after interpolation: {n_det}')
    if opt.out:
        print('saved:', opt.out)


if __name__ == '__main__':
    main()
