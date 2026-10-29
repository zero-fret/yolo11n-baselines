"""
set_loss.py — SET HBS + API training loss.

Adds SET's training-time augmentation on top of the standard YOLO11n loss:
  1. forward features -> raw predictions -> base loss
  2. compute d(base_loss)/d(feat), normalize, scale by reg_factor -> adversarial noise
  3. apply HBS denoising on background regions of P3/P4/P5
  4. second forward on (denoised feat + noise) -> set loss
  5. total = base_loss + scale * set_loss

Note: no AMP (autograd.grad + autocast don't mix well).
"""

import torch
import torch.nn.functional as F


def build_obj_mask(targets, B, H, W, device):
    """targets [N,6]=(b,cls,x1,y1,x2,y2) pixel -> [B,1,H,W] binary foreground mask."""
    mask = torch.zeros((B, 1, H, W), device=device)
    for b in range(B):
        m = targets[targets[:, 0] == b]
        for row in m:
            x1 = max(0, int(row[2].item()))
            y1 = max(0, int(row[3].item()))
            x2 = min(W, int(row[4].item()))
            y2 = min(H, int(row[5].item()))
            if x2 > x1 and y2 > y1:
                mask[b, 0, y1:y2, x1:x2] = 1.0
    return mask


def set_step(model, imgs, targets, criterion):
    """One SET forward+backward step. Returns (total_loss, (base_loss, set_loss))."""
    B, _, H, W = imgs.shape
    device = imgs.device

    feats = model.forward_features(imgs)
    preds = model.forward_head(feats)
    base_loss, _ = criterion(preds, targets)

    mask = build_obj_mask(targets, B, H, W, device)

    grads = torch.autograd.grad(base_loss, feats, retain_graph=True,
                               create_graph=False, allow_unused=True)
    x_set = []
    for i, f in enumerate(feats):
        fh, fw = f.shape[2], f.shape[3]
        bm = F.interpolate(mask, size=(fh, fw), mode='nearest')
        bg = 1.0 - bm

        dn = model.denoisers[i](f * bg)
        dn_feat = dn * bg + f * bm

        g = grads[i] if grads[i] is not None else torch.zeros_like(f)
        g = g / (g.norm() + 1e-12)
        noise = g * model.reg_factor_range[model.reg_factors[i]]

        x_set.append(dn_feat + noise)

    preds_set = model.forward_head(x_set)
    set_loss, _ = criterion(preds_set, targets)

    total = base_loss + model.scale * set_loss
    return total, (base_loss, set_loss)
