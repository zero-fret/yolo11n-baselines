"""
common/weights.py — robust weight loading.

Supports three sources:
  1. Self-trained checkpoint  (dict with 'model' = state_dict, 'nc', 'epoch', ...)
  2. Ultralytics official .pt  (dict with 'model' = a DetectionModule nn.Module)
  3. Raw state_dict dict       (just keys -> tensors)

The YOLO11n backbone in this repo is parametrically identical to ultralytics,
so official yolo11n.pt weights load directly (after unwrapping the module).
"""

import torch

from yolocore.model import YOLO11n


def _clean_state_dict(sd):
    """Strip common prefixes ('model.', 'model.model.') and unused keys."""
    out = {}
    for k, v in sd.items():
        nk = k
        for pre in ('model.model.', 'model.'):
            if nk.startswith(pre):
                nk = nk[len(pre):]
                break
        # Drop Detect head buffers that don't exist in our model
        if nk in ('anchors', 'strides'):
            continue
        out[nk] = v
    return out


def load_weights(path, nc=None, img_size=None, device='cpu'):
    """Load YOLO11n from any of the three supported checkpoint formats.

    Returns (model, metadata_dict). metadata includes 'epoch'/'nc'/'img_size'
    when available, else empty.
    """
    ck = torch.load(path, map_location=device, weights_only=False)
    meta = {}

    if isinstance(ck, dict) and 'model' in ck:
        m = ck['model']
        if isinstance(m, torch.nn.Module):
            # Ultralytics official checkpoint: unwrap the module
            sd = m.state_dict()
            meta['nc'] = int(getattr(m, 'nc', nc or 80))
        elif isinstance(m, dict):
            # Self-trained checkpoint
            sd = m
            meta['nc'] = int(ck.get('nc', nc or 80))
            meta['epoch'] = int(ck.get('epoch', -1))
            meta['img_size'] = int(ck.get('img_size', img_size or 640))
        else:
            raise ValueError(f'Unknown checkpoint model type: {type(m)}')
    elif isinstance(ck, dict):
        # Raw state_dict
        sd = ck
    else:
        raise ValueError(f'Unrecognized checkpoint format: {type(ck)}')

    nc = meta.get('nc', nc or 80)
    sd = _clean_state_dict(sd)

    model = YOLO11n(nc=nc)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # Anchor/stride buffers are expected missing (filled by forward)
    missing = [k for k in missing if k not in ('anchors', 'strides')]
    if missing:
        print(f'[weights] missing keys (first 5): {missing[:5]}')
    if unexpected:
        print(f'[weights] unexpected keys (first 5): {unexpected[:5]}')
    return model, meta
