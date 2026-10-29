# yolo11n-baselines

Four paper reproductions on top of a from-scratch YOLO11n backbone
(`yolocore/` + `common/` shared with the sibling repo `yolo11n-det/`).

All baselines ingest **standard YOLO-format datasets**
(`images/{split}/*.jpg` + `labels/{split}/*.txt`), ship GPLv3, and are
standalone — no dependency on any private video JSON format.

## Baselines

| Baseline | Paper | Year | What it changes vs. vanilla YOLO11n |
|---|---|---|---|
| `set/` | **SET** — Background smoothing + adversarial perturbation | CVPR 2025 | Training-time only (HBS on P3/P4/P5 + API along feature-gradient direction). Inference is identical to YOLO11n. |
| `lfnet/` | **LFBNet** — Local frequency bridge network (RGB-T) | CVPR 2026 | Dual-stream frequency-domain alignment on P4; thermal stream is synthesised from RGB grayscale (3-channel broadcast). |
| `drone_vs_bird/` | Drone-vs-Bird detection | — | Copy-paste augmentation + multi-scale tiling inference + temporal interpolation of missing frames. |
| `frame_dynamics/` | A Simple Detector with Frame Dynamics | CVPRW 2025 | Replace RGB input with `[gray_t, |I_t-I_{t-1}|, |I_t-I_{t-2}|]`. Model unchanged. |

## Layout

```
yolo11n-baselines/
├── yolocore/             # shared YOLO11n backbone + loss
├── common/               # shared dataset / engine / infer / weights
├── baselines/
│   ├── set/              # SET (CVPR 2025)
│   ├── lfnet/            # LFBNet (CVPR 2026)
│   ├── drone_vs_bird/    # copy-paste + multiscale + temporal
│   └── frame_dynamics/   # frame-dynamics input
├── LICENSE               # GPL-3.0-or-later
├── requirements.txt
└── README.md
```

## Train any baseline

```powershell
# Smoke (CPU, 2 batches)
python -m baselines.set.train   --data data --nc 1 --img-size 320 --batch 2 --max-batches 2
python -m baselines.lfnet.train  --data data --nc 1 --img-size 320 --batch 2 --max-batches 2
python -m baselines.drone_vs_bird.train --data data --nc 1 --img-size 320 --batch 2 --max-batches 2
python -m baselines.frame_dynamics.train --data data --nc 1 --img-size 320 --batch 2 --max-batches 2

# Full run
python -m baselines.set.train --data data --nc 3 --img-size 640 --batch 8 --epochs 300
```

Checkpoints are written to `weights/<tag>_last.pt` at the end of every epoch.

## Temporal data collection — user wiring required

Two baselines depend on consecutive video frames; this repo ships **no
video storage adapter**. Per user request, the temporal frame-loading
hooks are left as TODO comments for the user to implement:

- **`baselines/frame_dynamics/fd_dataset.py`** — `StaticFDDataset` is a
  smoke fallback (diff channels = 0). To use real frame-dynamics, implement
  a `frame_provider(data_root, split, img_size)` callable that yields
  `(cur_bgr, prev_bgr, prev2_bgr, targets_xyxy_px)` tuples, then pass it to
  `create_dataloader(..., frame_provider=your_fn)`.
- **`baselines/drone_vs_bird/video_detect.py`** — the video decode loop is
  ready (cv2.VideoCapture). To plug in a custom frame source (camera,
  dataset iterator, pre-decoded frames), replace the `while True: cap.read()`
  block with your own generator.

The interpolation logic itself (`drone_vs_bird/temporal.py`) is
self-contained: pass it a list of per-frame detection arrays and it
fills in missing detections.

## License

GPL-3.0-or-later. Each paper's name belongs to its respective authors; this
repository ships independent re-implementations only.
