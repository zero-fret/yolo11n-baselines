"""
lfnet_model.py — LFBNet (Local Frequency Bridge Network, CVPR 2026).

Original LFBNet is an RGB-T dual-stream network. Since this repo ships only
visible-light data, the thermal stream is synthesised from the grayscale of
the RGB image (3-channel broadcast). The frequency-domain alignment (LFCA /
FGSA) is placed on the P4 (stride-16) feature map with channel align_c=32 to
keep FLOPs on par with YOLO11n (~2.70M params, 1.67G FLOPs @ 320).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from yolocore.model import Conv, C3k2, SPPF, C2PSA, Concat, Detect


def wrap(phase):
    """Wrap phase difference to [-pi, pi]."""
    return torch.atan2(torch.sin(phase), torch.cos(phase))


def block_fft2(x, patch=8):
    B, C, H, W = x.shape
    nh = (H + patch - 1) // patch
    nw = (W + patch - 1) // patch
    Hp, Wp = nh * patch, nw * patch
    x = F.pad(x, (0, Wp - W, 0, Hp - H))
    x = x.view(B, C, nh, patch, nw, patch).permute(0, 1, 2, 4, 3, 5)
    Fq = torch.fft.fft2(x, dim=(-2, -1))
    return Fq, (H, W, nh, nw, patch)


def block_ifft2(Fq, meta):
    H, W, nh, nw, patch = meta
    x = torch.fft.ifft2(Fq, dim=(-2, -1)).real
    x = x.permute(0, 1, 2, 4, 3, 5).contiguous()
    x = x.view(x.shape[0], x.shape[1], nh * patch, nw * patch)
    return x[:, :, :H, :W]


def block_expand(x, meta):
    H, W, nh, nw, patch = meta
    x = x.permute(0, 1, 2, 4, 3, 5).contiguous()
    x = x.view(x.shape[0], x.shape[1], nh * patch, nw * patch)
    return x[:, :, :H, :W]


class SFE(nn.Module):
    """Spatial feature extractor (YOLO11n backbone, stride 8/16/32)."""

    def __init__(self):
        super().__init__()
        self.b0 = Conv(3, 16, 3, 2)
        self.b1 = Conv(16, 32, 3, 2)
        self.b2 = C3k2(32, 64, n=1, c3k=False, e=0.25)
        self.b3 = Conv(64, 64, 3, 2)
        self.b4 = C3k2(64, 128, n=1, c3k=False, e=0.25)
        self.b5 = Conv(128, 128, 3, 2)
        self.b6 = C3k2(128, 128, n=1, c3k=True)
        self.b7 = Conv(128, 256, 3, 2)
        self.b8 = C3k2(256, 256, n=1, c3k=True)
        self.b9 = SPPF(256, 256, k=5)
        self.b10 = C2PSA(256, 256, n=1)

    def forward(self, x):
        x0 = self.b0(x)
        x1 = self.b1(x0)
        x2 = self.b2(x1)
        x3 = self.b3(x2)
        x4 = self.b4(x3)
        x5 = self.b5(x4)
        x6 = self.b6(x5)
        x7 = self.b7(x6)
        x8 = self.b8(x7)
        x9 = self.b9(x8)
        x10 = self.b10(x9)
        return x4, x6, x10


class XAttn(nn.Module):
    """Gated cross-attention: F_align gates the spatial feature."""

    def __init__(self, c):
        super().__init__()
        self.q = nn.Conv2d(c, c, 1, bias=False)
        self.k = nn.Conv2d(c, c, 1, bias=False)
        self.v = nn.Conv2d(c, c, 1, bias=False)

    def forward(self, fm, f_align):
        q = self.q(fm)
        k = self.k(f_align)
        v = self.v(f_align)
        attn = torch.sigmoid((q * k).mean(dim=1, keepdim=True))
        return fm + attn * v


class LFCA(nn.Module):
    """Amplitude normalisation + alpha fusion, phase beta alignment, iFFT rebuild."""

    def __init__(self, in_ch=3, patch=8):
        super().__init__()
        self.patch = patch
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))

    def forward(self, xr, xt):
        Fr, meta = block_fft2(xr, self.patch)
        Ft, _ = block_fft2(xt, self.patch)
        Ar, Phr = Fr.abs(), Fr.angle()
        At, Pht = Ft.abs(), Ft.angle()

        Ar_n = Ar / (Ar.norm(dim=(-2, -1), keepdim=True) + 1e-6)
        At_n = At / (At.norm(dim=(-2, -1), keepdim=True) + 1e-6)
        Aa = torch.sigmoid(self.alpha) * Ar_n + (1 - torch.sigmoid(self.alpha)) * At_n

        dPhi = wrap(Phr - Pht)
        Pha = Pht + torch.sigmoid(self.beta) * dPhi

        Fa = torch.polar(Aa, Pha)
        F_align = block_ifft2(Fa, meta)

        m = lambda t: t.mean(dim=1, keepdim=True)
        dx = torch.sin(dPhi).mean(dim=1, keepdim=True)
        dy = torch.cos(dPhi).mean(dim=1, keepdim=True)
        sphi = dPhi.abs().mean(dim=1, keepdim=True)
        asum = m(Ar_n) + m(At_n) + 1e-6
        chf = m(Ar_n) / asum
        clf = m(At_n) / asum
        cross = (Fr * Ft.conj()).sum(dim=1).sum(dim=-1).sum(dim=-1)
        er = (Fr.abs() ** 2).sum(dim=1).sum(dim=-1).sum(dim=-1).sqrt()
        et = (Ft.abs() ** 2).sum(dim=1).sum(dim=-1).sum(dim=-1).sqrt()
        coh_patch = cross.abs() / (er * et + 1e-6)
        coh = coh_patch[:, None, :, :, None, None].expand_as(dx)
        g_patch = torch.cat([dx, dy, sphi, chf, clf, coh], dim=1)
        g_freq = block_expand(g_patch, meta)
        return F_align, g_freq


class FGSA(nn.Module):
    """Frequency-guided spatial alignment + symmetric cross-fusion."""

    def __init__(self, c=128):
        super().__init__()
        self.gate = nn.Sequential(Conv(2 * c, c, 1), nn.Sigmoid())
        self.offset_conv = nn.Sequential(
            Conv(c + c + c, c, 3), Conv(c, c, 3), nn.Conv2d(c, 2, 1))
        self.fuse1 = Conv(2 * c, c, 3)
        self.fuse2 = Conv(2 * c, c, 3)

    def forward(self, fr_x, ft_x, g_freq):
        g = self.gate(torch.cat([fr_x, ft_x], 1)) * g_freq
        offset = self.offset_conv(torch.cat([fr_x, ft_x, g], 1))
        ft_aligned = deform_bilinear(ft_x, offset)
        fr_aligned = deform_bilinear(fr_x, -offset)
        out = self.fuse1(torch.cat([fr_x, ft_aligned], 1)) + \
              self.fuse2(torch.cat([ft_x, fr_aligned], 1))
        return out


def deform_bilinear(x, offset):
    B, C, H, W = x.shape
    yy, xx = torch.meshgrid(torch.arange(H, device=x.device, dtype=x.dtype),
                            torch.arange(W, device=x.device, dtype=x.dtype),
                            indexing='ij')
    base = torch.stack([xx, yy], dim=0).unsqueeze(0)
    grid = base + offset
    gy = 2.0 * grid[:, 1] / max(H - 1, 1) - 1.0
    gx = 2.0 * grid[:, 0] / max(W - 1, 1) - 1.0
    grid_norm = torch.stack([gx, gy], dim=-1)
    return F.grid_sample(x, grid_norm, align_corners=True)


class LFBNet(nn.Module):
    """RGB-T dual-stream frequency bridge + YOLO11n detection head."""

    def __init__(self, nc=1, patch=8, align_c=32):
        super().__init__()
        self.sfe = SFE()
        self.align_c = align_c
        self.lfca = LFCA(in_ch=3, patch=patch)

        self.r_proj = Conv(128, align_c, 1)
        self.t_proj = Conv(128, align_c, 1)
        self.align_proj = Conv(3, align_c, 1)
        self.g_proj = Conv(6, align_c, 1)
        self.xattn_r = XAttn(align_c)
        self.xattn_t = XAttn(align_c)
        self.fgsa = FGSA(c=align_c)

        self.p4_out = Conv(align_c, 128, 1)
        self.p3_gate = Conv(128, 128, 1)

        # YOLO11n neck + head (channels unchanged)
        self.h11 = nn.Upsample(scale_factor=2, mode='nearest')
        self.h12 = Concat(1)
        self.h13 = C3k2(384, 128, n=1, c3k=False)
        self.h14 = nn.Upsample(scale_factor=2, mode='nearest')
        self.h15 = Concat(1)
        self.h16 = C3k2(256, 64, n=1, c3k=False)
        self.h17 = Conv(64, 64, 3, 2)
        self.h18 = Concat(1)
        self.h19 = C3k2(192, 128, n=1, c3k=False)
        self.h20 = Conv(128, 128, 3, 2)
        self.h21 = Concat(1)
        self.h22 = C3k2(384, 256, n=1, c3k=True)
        self.h23 = Detect(nc=nc, reg_max=16, ch=(64, 128, 256))
        self.h23.stride = torch.tensor([8.0, 16.0, 32.0])

    def forward(self, xr, xt=None):
        if xt is None:
            xt = xr
        p3, p4, p5 = self.sfe(xr)

        xr_s = F.interpolate(xr, scale_factor=1 / 16, mode='bilinear', align_corners=False)
        xt_s = F.interpolate(xt, scale_factor=1 / 16, mode='bilinear', align_corners=False)
        F_align, g_freq = self.lfca(xr_s, xt_s)
        fa = self.align_proj(F_align)
        gg = self.g_proj(g_freq)

        pr = self.xattn_r(self.r_proj(p4), fa)
        pt = self.xattn_t(self.t_proj(p4), fa)
        p4_aligned = self.fgsa(pr, pt, gg)
        p4 = self.p4_out(p4_aligned)

        p3 = p3 + F.interpolate(self.p3_gate(p4), scale_factor=2,
                                mode='bilinear', align_corners=False)

        x11 = self.h11(p5)
        x12 = self.h12([x11, p4])
        x13 = self.h13(x12)
        x14 = self.h14(x13)
        x15 = self.h15([x14, p3])
        x16 = self.h16(x15)
        x17 = self.h17(x16)
        x18 = self.h18([x17, x13])
        x19 = self.h19(x18)
        x20 = self.h20(x19)
        x21 = self.h21([x20, p5])
        x22 = self.h22(x21)
        return self.h23([x16, x19, x22])
