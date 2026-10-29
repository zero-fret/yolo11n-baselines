"""
train.py — SET training entry (no AMP, custom loop because of autograd.grad).

Usage:
  python -m baselines.set.train --data data --nc 3 --img-size 640 --batch 8 --epochs 300
  python -m baselines.set.train --data data --nc 1 --img-size 320 --batch 2 --max-batches 2
"""

import os
import argparse
import math

import torch
from tqdm import tqdm

from yolocore.loss_utils import V8DetectionLoss
from common.dataset import create_dataloader
from .set_model import build_set
from .set_loss import set_step


def lr_at(step, total, lr0, lrf, warmup_steps):
    if step < warmup_steps:
        return lr0 * 0.1 + (lr0 - lr0 * 0.1) * (step / max(1, warmup_steps))
    prog = (step - warmup_steps) / max(1, total - warmup_steps)
    return lrf + 0.5 * (lr0 - lrf) * (1 + math.cos(math.pi * prog))


def train(model, loader, criterion, epochs, device, lr0=0.01, lrf=0.01,
          momentum=0.937, weight_decay=5e-4, warmup_frac=0.05,
          weights_dir='weights', tag='set', max_batches=0, nc=1):
    os.makedirs(weights_dir, exist_ok=True)
    model.to(device)

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if n.endswith('.bias') or 'bn.' in n else decay).append(p)
    optimizer = torch.optim.SGD(
        [{'params': decay, 'weight_decay': weight_decay},
         {'params': no_decay, 'weight_decay': 0.0}],
        lr=lr0, momentum=momentum, nesterov=True)

    steps_per_epoch = max_batches or len(loader)
    total_steps = epochs * steps_per_epoch
    warmup_steps = max(1, int(total_steps * warmup_frac))

    model.train()
    step = 0
    last_path = os.path.join(weights_dir, f'{tag}_last.pt')
    for epoch in range(epochs):
        bar = tqdm(loader, total=steps_per_epoch, ncols=100,
                   desc=f'epoch {epoch + 1}/{epochs}')
        running = 0.0
        for bi, (imgs, targets) in enumerate(bar):
            if max_batches and bi >= max_batches:
                break
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device)

            lr = lr_at(step, total_steps, lr0, lrf * lr0, warmup_steps)
            for g in optimizer.param_groups:
                g['lr'] = lr

            optimizer.zero_grad(set_to_none=True)
            total, (base_loss, set_loss) = set_step(model, imgs, targets, criterion)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            step += 1

            running = running * 0.9 + float(total) * 0.1 if bi else float(total)
            if bi % 10 == 0:
                bar.set_postfix(loss=f'{running:.4f}',
                                base=f'{float(base_loss):.3f}',
                                set=f'{float(set_loss):.3f}', lr=f'{lr:.4f}')

        ckpt = {'model': model.state_dict(), 'epoch': epoch,
                'nc': nc, 'img_size': int(imgs.shape[-1])}
        torch.save(ckpt, last_path)
    return last_path


def parse_opt():
    p = argparse.ArgumentParser(description='SET (YOLO11n) training')
    p.add_argument('--data', default='data')
    p.add_argument('--nc', type=int, default=80)
    p.add_argument('--img-size', type=int, default=640)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--lr0', type=float, default=0.01)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--device', default='')
    p.add_argument('--max-batches', type=int, default=0)
    p.add_argument('--tag', default='set')
    p.add_argument('--weights-dir', default='weights')
    return p.parse_args()


def main():
    opt = parse_opt()
    device = torch.device(opt.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f'=== SET train | device={device} | img={opt.img_size} batch={opt.batch} ===')

    model = build_set(nc=opt.nc, train=True)
    print(f'parameters: {sum(p.numel() for p in model.parameters()):,}')

    train_loader = create_dataloader(opt.data, 'train', opt.img_size, opt.batch,
                                     augment=True, workers=opt.workers)
    print(f'train samples: {len(train_loader.dataset)}')

    criterion = V8DetectionLoss(nc=opt.nc, strides=(8.0, 16.0, 32.0))
    last = train(model, train_loader, criterion, opt.epochs, device,
                 lr0=opt.lr0, weights_dir=opt.weights_dir, tag=opt.tag,
                 max_batches=opt.max_batches, nc=opt.nc)
    print('last checkpoint:', last)


if __name__ == '__main__':
    main()
