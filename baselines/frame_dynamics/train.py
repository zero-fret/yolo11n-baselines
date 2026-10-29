"""
train.py — Frame Dynamics training entry.

The model is vanilla YOLO11n — only the input is replaced by the 3-channel
frame-dynamics image. Same loss, same engine as YOLO11n-det.

Usage:
  python -m baselines.frame_dynamics.train --data data --nc 3 --img-size 640 --batch 8 --epochs 300
  python -m baselines.frame_dynamics.train --data data --nc 1 --img-size 320 --batch 2 --max-batches 2

NOTE: by default this uses the *static* fallback dataset (no temporal frames).
To use real frame-dynamics, pass your own `frame_provider` (see fd_dataset.py
docstring) by editing `create_dataloader(...)` below.
"""

import os
import argparse

import torch

from yolocore.model import YOLO11n
from yolocore.loss_utils import V8DetectionLoss, swap_train_head
from common.engine import train_model
from .fd_dataset import create_dataloader


def parse_opt():
    p = argparse.ArgumentParser(description='Frame Dynamics training')
    p.add_argument('--data', default='data')
    p.add_argument('--nc', type=int, default=80)
    p.add_argument('--img-size', type=int, default=640)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--lr0', type=float, default=0.01)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--device', default='')
    p.add_argument('--no-amp', action='store_true')
    p.add_argument('--optimizer', default='sgd', choices=['sgd', 'adam'])
    p.add_argument('--max-batches', type=int, default=0)
    p.add_argument('--tag', default='frame_dynamics')
    p.add_argument('--weights-dir', default='weights')
    p.add_argument('--resume', default='')
    return p.parse_args()


def main():
    opt = parse_opt()
    device = torch.device(opt.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f'=== Frame Dynamics train | device={device} | img={opt.img_size} batch={opt.batch} ===')

    model = YOLO11n(nc=opt.nc)
    swap_train_head(model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'parameters: {n_params:,}  (identical to vanilla YOLO11n)')

    # TODO(user): pass frame_provider=your_temporal_iterator to use real
    # frame-dynamics inputs. Without it, this uses the static fallback.
    train_loader = create_dataloader(opt.data, 'train', opt.img_size, opt.batch,
                                     augment=True, workers=opt.workers)
    val_loader = create_dataloader(opt.data, 'val', opt.img_size, opt.batch,
                                   augment=False, shuffle=False, workers=opt.workers)
    print(f'train samples: {len(train_loader.dataset)}  val samples: {len(val_loader.dataset)}')

    criterion = V8DetectionLoss(nc=opt.nc, strides=(8.0, 16.0, 32.0))
    last = train_model(model, train_loader, criterion, opt.epochs, device,
                       lr0=opt.lr0, amp=not opt.no_amp,
                       weights_dir=opt.weights_dir, tag=opt.tag,
                       max_batches=opt.max_batches, val_loader=val_loader,
                       resume=opt.resume, optimizer_type=opt.optimizer,
                       extra_state=lambda: {'nc': opt.nc, 'img_size': opt.img_size})
    print('last checkpoint:', last)


if __name__ == '__main__':
    main()
