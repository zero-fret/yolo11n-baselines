"""
train.py — Drone-vs-Bird training entry (YOLO11n + copy-paste augmentation).

Usage:
  python -m baselines.drone_vs_bird.train --data data --nc 3 --img-size 640 --batch 16 --epochs 300
  python -m baselines.drone_vs_bird.train --data data --nc 1 --img-size 320 --batch 2 --max-batches 2
"""

import os
import argparse

from yolocore.model import YOLO11n
from yolocore.loss_utils import V8DetectionLoss, swap_train_head
from common.dataset import create_dataloader
from common.engine import train_model
from .copypaste import create_copypaste_loader


def parse_opt():
    p = argparse.ArgumentParser(description='Drone-vs-Bird (YOLO11n) training')
    p.add_argument('--data', default='data')
    p.add_argument('--nc', type=int, default=80)
    p.add_argument('--img-size', type=int, default=640)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--lr0', type=float, default=0.01)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--device', default='')
    p.add_argument('--no-amp', action='store_true')
    p.add_argument('--no-paste', action='store_true',
                   help='disable copy-paste, fall back to plain YOLO aug')
    p.add_argument('--max-batches', type=int, default=0)
    p.add_argument('--tag', default='drone_vs_bird')
    p.add_argument('--weights-dir', default='weights')
    p.add_argument('--resume', default='')
    return p.parse_opt()


def main():
    opt = parse_opt()
    import torch
    device = torch.device(opt.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f'=== Drone-vs-Bird train | device={device} img={opt.img_size} '
          f'batch={opt.batch} copy-paste={not opt.no_paste} ===')

    model = YOLO11n(nc=opt.nc)
    swap_train_head(model)
    print(f'parameters: {sum(p.numel() for p in model.parameters()):,}')

    if opt.no_paste:
        train_loader = create_dataloader(opt.data, 'train', opt.img_size, opt.batch,
                                         augment=True, workers=opt.workers)
    else:
        train_loader = create_copypaste_loader(opt.data, 'train', opt.img_size,
                                                opt.batch, augment=True,
                                                workers=opt.workers)
    val_loader = create_dataloader(opt.data, 'val', opt.img_size, opt.batch,
                                   augment=False, shuffle=False, workers=opt.workers)
    print(f'train samples: {len(train_loader.dataset)}  val: {len(val_loader.dataset)}')

    criterion = V8DetectionLoss(nc=opt.nc, strides=(8.0, 16.0, 32.0))
    last = train_model(model, train_loader, criterion, opt.epochs, device,
                       lr0=opt.lr0, momentum=0.9, amp=not opt.no_amp,
                       weights_dir=opt.weights_dir, tag=opt.tag,
                       max_batches=opt.max_batches, val_loader=val_loader,
                       optimizer_type='sgd', resume=opt.resume,
                       extra_state=lambda: {'nc': opt.nc,
                                            'img_size': opt.img_size})
    print('last checkpoint:', last)


if __name__ == '__main__':
    main()
