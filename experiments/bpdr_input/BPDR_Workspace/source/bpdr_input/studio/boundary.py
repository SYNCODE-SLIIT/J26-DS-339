"""Boundary Transformer inference helper for Vocal Rescue Studio.

Extracts boundary probability hints from audio and pitch evidence tracks,
and provides helpers to integrate these hints into studio sidecars.
"""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as tnf
import librosa
from scipy.signal import find_peaks

V1_CHECKPOINT = (
    Path(__file__).resolve().parents[3] / "results" / "studio" / "r1" / "boundary_transformer.pt"
)
V2_CHECKPOINT = (
    Path(__file__).resolve().parents[3] / "results" / "studio" / "r1" / "boundary_transformer_v2.pt"
)
V2_BPDR_CHECKPOINT = (
    Path(__file__).resolve().parents[3] / "results" / "studio" / "r1" / "boundary_transformer_v2_bpdr.pt"
)
V3_CHECKPOINT = (
    Path(__file__).resolve().parents[3] / "results" / "studio" / "r1" / "boundary_transformer_v3.pt"
)
DEFAULT_CHECKPOINT = V1_CHECKPOINT
CANONICAL_FRAME_SECONDS = 0.020
DEFAULT_THRESHOLD = 0.8
DEFAULT_EVAL_LOGIT_BIAS = 0.9


class FastTransformerBlock(nn.Module):
    """Pre-norm self-attention transformer block with linear projections."""
    def __init__(self, d=32, h=4):
        super().__init__()
        self.h = h
        self.n1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, x):
        B_, T, C = x.shape
        q, k, v = self.qkv(self.n1(x)).chunk(3, -1)
        f = lambda t: t.view(B_, T, self.h, C // self.h).transpose(1, 2)
        a = tnf.scaled_dot_product_attention(f(q), f(k), f(v)).transpose(1, 2).contiguous().view(B_, T, C)
        x = x + self.proj(a)
        return x + self.ff(self.n2(x))


class FastTNet(nn.Module):
    """Stride-4 dilated residual highway + multi-scale pitch step deltas + depthwise conv positional encoding + fast pre-norm attention."""
    def __init__(self, nin=5, d=32, nhead=4, nlayers=2, eval_logit_bias=DEFAULT_EVAL_LOGIT_BIAS):
        super().__init__()
        self.nin = nin
        self.d = d
        self.nhead = nhead
        self.nlayers = nlayers
        self.eval_logit_bias = float(eval_logit_bias)
        self.in_proj = nn.Conv1d(nin + 5, d, 3, padding=1)
        self.fine_convs = nn.ModuleList([nn.Conv1d(d, d, 5, padding=2 * k, dilation=k) for k in (1, 2, 4, 8)])
        self.down = nn.Conv1d(d, d, 4, stride=4, padding=1)
        self.pos_conv = nn.Conv1d(d, d, 7, padding=3, groups=d)
        self.enc = nn.Sequential(*[FastTransformerBlock(d, nhead) for _ in range(nlayers)])
        self.up = nn.ConvTranspose1d(d, d, 4, stride=4, padding=0)
        self.gate = nn.Parameter(torch.tensor(0.2))
        self.out = nn.Conv1d(d, 1, 1)

    def _deltas(self, x):
        p, ok = x[:, :, 0], x[:, :, 1]
        dp1 = torch.zeros_like(p)
        dp1[:, 1:] = (p[:, 1:] - p[:, :-1]).abs() * ok[:, 1:] * ok[:, :-1]
        dok = torch.zeros_like(ok)
        dok[:, 1:] = (ok[:, 1:] - ok[:, :-1]).abs()
        steps = [dp1.clamp(0, 2.0), dok]
        po = (p * ok)[:, None]
        ok1 = ok[:, None]
        for w in (3, 6, 10):
            pad_p = tnf.pad(po, (w, w))
            pad_ok = tnf.pad(ok1, (w, w))
            cs_p = pad_p.cumsum(-1)
            cs_ok = pad_ok.cumsum(-1)
            T = p.shape[1]
            lp = (cs_p[:, 0, w : w + T] - cs_p[:, 0, :T]) / (cs_ok[:, 0, w : w + T] - cs_ok[:, 0, :T]).clamp_min(1.0)
            rp = (cs_p[:, 0, w + 1 : w + 1 + T] - cs_p[:, 0, 1 : 1 + T]) / (cs_ok[:, 0, w + 1 : w + 1 + T] - cs_ok[:, 0, 1 : 1 + T]).clamp_min(1.0)
            valid = (pad_ok[:, 0, w - 1 : w - 1 + T] > 0.5) & (pad_ok[:, 0, w + 1 : w + 1 + T] > 0.5)
            steps.append(((rp - lp).abs() * valid.float()).clamp(0, 2.0))
        return torch.stack(steps, dim=-1)

    def forward(self, x):
        dx = self._deltas(x)
        z = self.in_proj(torch.cat([x, dx], dim=-1).transpose(1, 2))
        for c in self.fine_convs:
            z = z + tnf.gelu(c(z))
        zc = tnf.gelu(self.down(z))
        zc = zc + self.pos_conv(zc)
        zc = self.enc(zc.transpose(1, 2)).transpose(1, 2)
        zu = self.up(zc)
        zu = tnf.pad(zu, (0, max(0, x.shape[1] - zu.shape[-1])), mode="replicate")[..., : x.shape[1]]
        z = z + self.gate * tnf.gelu(zu)
        out = self.out(z)[:, 0]
        return out if self.training else out + self.eval_logit_bias


V5_CHECKPOINT = (
    Path(__file__).resolve().parents[3] / "results" / "studio" / "r1" / "boundary_transformer_v5.pt"
)
V5_DELTA_CHANNELS = 16


def _wsum(v, w, side):
    """Windowed sum of (B, T) over the w frames left of / right of / around each frame (zero padded)."""
    k = torch.zeros(1, 1, 2 * w + 1, dtype=v.dtype, device=v.device)
    if side == "left":
        k[..., :w] = 1.0
    elif side == "right":
        k[..., w + 1:] = 1.0
    else:
        k[...] = 1.0
    return tnf.conv1d(v[:, None], k, padding=w)[:, 0]


class FastTNetV5(FastTNet):
    """FastTNet v1 backbone plus singing-aware input deltas (signed multi-scale pitch steps, vibrato index,
    energy dip/step, flux prominence). Input stays the 5 canonical channels; the extra 16 are derived inside."""

    def __init__(self, nin=5, d=32, nhead=4, nlayers=2, eval_logit_bias=0.0):
        super().__init__(nin=nin, d=d, nhead=nhead, nlayers=nlayers, eval_logit_bias=eval_logit_bias)
        self.in_proj = nn.Conv1d(nin + V5_DELTA_CHANNELS, d, 3, padding=1)

    def _deltas(self, x):
        p, ok, conf, en, fl = x[:, :, 0], x[:, :, 1], x[:, :, 2], x[:, :, 3], x[:, :, 4]
        dp1 = torch.zeros_like(p)
        dp1[:, 1:] = (p[:, 1:] - p[:, :-1]).abs() * ok[:, 1:] * ok[:, :-1]
        dok = torch.zeros_like(ok)
        dok[:, 1:] = (ok[:, 1:] - ok[:, :-1]).abs()
        dc1 = torch.zeros_like(conf)
        dc1[:, 1:] = (conf[:, 1:] - conf[:, :-1]).abs()
        steps = [dp1.clamp(0, 2.0), dok, dc1.clamp(0, 1.0)]
        po = p * ok
        for w in (3, 6, 10):
            cl, cr = _wsum(ok, w, "left"), _wsum(ok, w, "right")
            lp = _wsum(po, w, "left") / cl.clamp_min(1.0)
            rp = _wsum(po, w, "right") / cr.clamp_min(1.0)
            valid = ((cl >= 1.0) & (cr >= 1.0)).float()
            sg = ((rp - lp) * valid).clamp(-2.0, 2.0)
            steps += [sg, sg.abs()]
        c = _wsum(ok, 5, "both")
        m = _wsum(po, 5, "both") / c.clamp_min(1.0)
        m2 = _wsum(po * po, 5, "both") / c.clamp_min(1.0)
        std = ((m2 - m * m).clamp_min(0.0).sqrt() * (c >= 3.0).float()).clamp(0, 1.0)
        steps += [std, (steps[6] - std).clamp(-1.0, 1.0)]
        edip = (en - tnf.max_pool1d(en[:, None], 9, 1, 4)[:, 0]).clamp(-3.0, 0.0)
        ed1 = torch.zeros_like(en)
        ed1[:, 1:] = (en[:, 1:] - en[:, :-1]).abs()
        erl = ((_wsum(en, 3, "right") - _wsum(en, 3, "left")) / 3.0).clamp(-3.0, 3.0)
        steps += [edip, ed1.clamp(0, 3.0), erl]
        fprom = (fl - tnf.avg_pool1d(fl[:, None], 9, 1, 4, count_include_pad=False)[:, 0]).clamp(-1.0, 1.0)
        fpk = (fl - tnf.max_pool1d(fl[:, None], 5, 1, 2)[:, 0]).clamp(-1.0, 0.0)
        steps += [fprom, fpk]
        return torch.stack(steps, dim=-1)


def refine_peak_times(probs, peak_idx, hop_s, subframe=False, shift_s=0.0):
    """Peak times in seconds. Optional 3-point parabolic sub-frame vertex and a constant shift (seconds)."""
    probs = np.asarray(probs, dtype=float)
    out = []
    for i in np.asarray(peak_idx, dtype=int):
        off = 0.0
        if subframe and 0 < i < len(probs) - 1:
            y0, y1, y2 = probs[i - 1], probs[i], probs[i + 1]
            den = y0 - 2.0 * y1 + y2
            if den < -1e-9:
                off = float(np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5))
        out.append((float(i) + off) * hop_s + float(shift_s))
    return out


class FastTransformerBlockV2(nn.Module):
    """Pre-norm multi-head self-attention block with dropout and GELU FFN."""

    def __init__(self, d=40, h=4, drop=0.1):
        super().__init__()
        self.h = h
        self.n1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.drop = nn.Dropout(drop)
        self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, 2 * d),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(2 * d, d),
        )

    def forward(self, x):
        B_, T, C = x.shape
        q, k, v = self.qkv(self.n1(x)).chunk(3, -1)
        f = lambda t: t.view(B_, T, self.h, C // self.h).transpose(1, 2)
        a = tnf.scaled_dot_product_attention(f(q), f(k), f(v))
        a = a.transpose(1, 2).contiguous().view(B_, T, C)
        x = x + self.drop(self.proj(a))
        return x + self.drop(self.ff(self.n2(x)))


class FastTNetV2(nn.Module):
    """Calibrated FastTNet v2 boundary detector (92.7k params, no eval logit bias)."""

    def __init__(self, nin=5, d=40, stride=2, nhead=4, nlayers=2, drop=0.1, eval_logit_offset=0.0):
        super().__init__()
        self.nin = nin
        self.d = d
        self.stride = stride
        self.nhead = nhead
        self.nlayers = nlayers
        self.drop_rate = drop
        self.eval_logit_offset = float(eval_logit_offset)

        self.in_proj = nn.Conv1d(nin + 7, d, 3, padding=1)
        self.in_norm = nn.GroupNorm(4, d)
        self.fine_convs = nn.ModuleList(
            [nn.Conv1d(d, d, 5, padding=2 * k, dilation=k) for k in (1, 2, 4, 8, 1, 2)]
        )
        self.fine_drop = nn.Dropout(drop)
        self.down = nn.Conv1d(d, d, stride + 2, stride=stride, padding=1)
        self.pos_conv = nn.Conv1d(d, d, 7, padding=3, groups=d)
        self.enc = nn.Sequential(*[FastTransformerBlockV2(d, nhead, drop) for _ in range(nlayers)])
        self.up_proj = nn.Conv1d(d, d, 3, padding=1)
        self.gate = nn.Parameter(torch.tensor(0.2))
        self.head = nn.Sequential(
            nn.Conv1d(d, d, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(d, 1, 1),
        )

    def _deltas(self, x):
        p, ok, conf = x[:, :, 0], x[:, :, 1], x[:, :, 2]
        dp1 = torch.zeros_like(p)
        dp1[:, 1:] = (p[:, 1:] - p[:, :-1]).abs() * ok[:, 1:] * ok[:, :-1]
        dok = torch.zeros_like(ok)
        dok[:, 1:] = (ok[:, 1:] - ok[:, :-1]).abs()
        dc1 = torch.zeros_like(conf)
        dc1[:, 1:] = (conf[:, 1:] - conf[:, :-1]).abs()
        steps = [dp1.clamp(0.0, 2.0), dok, dc1.clamp(0.0, 1.0)]

        po = (p * ok)[:, None]
        ok1 = ok[:, None]
        c1 = conf[:, None]
        T = p.shape[1]
        for w in (3, 6, 12):
            cs_p = tnf.pad(tnf.pad(po, (w, w)).cumsum(-1), (1, 0))
            cs_ok = tnf.pad(tnf.pad(ok1, (w, w)).cumsum(-1), (1, 0))
            cnt_l = cs_ok[:, 0, w : w + T] - cs_ok[:, 0, :T]
            cnt_r = cs_ok[:, 0, 2 * w : 2 * w + T] - cs_ok[:, 0, w : w + T]
            lp = (cs_p[:, 0, w : w + T] - cs_p[:, 0, :T]) / cnt_l.clamp_min(1.0)
            rp = (cs_p[:, 0, 2 * w : 2 * w + T] - cs_p[:, 0, w : w + T]) / cnt_r.clamp_min(1.0)
            valid = (cnt_l >= 1.0) & (cnt_r >= 1.0)
            steps.append(((rp - lp).abs() * valid.float()).clamp(0.0, 2.0))
            if w == 3:
                cs_c = tnf.pad(tnf.pad(c1, (w, w)).cumsum(-1), (1, 0))
                cl = (cs_c[:, 0, w : w + T] - cs_c[:, 0, :T]) / float(w)
                cr = (cs_c[:, 0, 2 * w : 2 * w + T] - cs_c[:, 0, w : w + T]) / float(w)
                steps.append((0.5 * (cl + cr) - conf).clamp(0.0, 1.0))
        return torch.stack(steps, dim=-1)

    def forward(self, x):
        dx = self._deltas(x)
        z = tnf.gelu(self.in_norm(self.in_proj(torch.cat([x, dx], dim=-1).transpose(1, 2))))
        for c in self.fine_convs:
            z = z + self.fine_drop(tnf.gelu(c(z)))
        zc = tnf.gelu(self.down(z))
        zc = zc + self.pos_conv(zc)
        zc = self.enc(zc.transpose(1, 2)).transpose(1, 2)
        zu = self.up_proj(tnf.interpolate(zc, size=x.shape[1], mode="linear", align_corners=False))
        z = z + self.gate * tnf.gelu(zu)
        out = self.head(z)[:, 0]
        if self.training or self.eval_logit_offset == 0.0:
            return out
        return out + self.eval_logit_offset


def _load_metadata_sidecar(path: Path):
    """Read optional JSON metadata stored alongside a boundary checkpoint."""
    candidates = [
        path.with_suffix(".json"),
        path.with_name(f"{path.stem}_meta.json"),
        path.with_name(f"{path.stem}.meta.json"),
    ]
    if path.stem.endswith("_bpdr"):
        base = path.stem[: -len("_bpdr")]
        candidates.extend([
            path.with_name(f"{base}_meta.json"),
            path.with_name(f"{base}.meta.json"),
        ])
    for cand in candidates:
        if cand.exists():
            try:
                data = json.loads(cand.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
            except Exception:
                pass
    return {}


def load_boundary_model(checkpoint_path=None, map_location="cpu"):
    """Load a FastTNet / FastTNetV2 boundary model checkpoint and calibration metadata."""
    path = Path(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT
    if not path.exists():
        raise FileNotFoundError(f"Boundary transformer checkpoint not found at: {path}")
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    meta = _load_metadata_sidecar(path)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt_meta = {k: v for k, v in ckpt.items() if k != "state_dict"}
        meta = {**meta, **ckpt_meta}
        state = ckpt["state_dict"]
    else:
        state = ckpt

    is_bpdr = path.stem.endswith("_bpdr")
    model_cls = str(meta.get("model_class", meta.get("arch", ""))).lower()
    is_v2 = (
        model_cls in ("fasttnetv2", "transformer_v2")
        or ("in_norm.weight" in state)
        or ("head.0.weight" in state)
    )

    in_weight = state.get("in_proj.weight")
    inferred_layers = sum(1 for k in state if k.startswith("enc.") and k.endswith(".n1.weight")) or 2
    nhead = int(meta.get("nhead", 4))
    nlayers = int(meta.get("nlayers", inferred_layers))

    if is_v2:
        inferred_nin = in_weight.shape[1] - 7 if in_weight is not None else 5
        nin_key = "bpdr_nin" if (is_bpdr and "bpdr_nin" in meta) else "nin"
        nin = int(meta.get(nin_key, inferred_nin))
        if in_weight is not None and in_weight.shape[1] - 7 != nin:
            nin = in_weight.shape[1] - 7
        d = int(meta.get("d", in_weight.shape[0] if in_weight is not None else 40))
        stride = int(meta.get("stride", 2))
        drop = float(meta.get("drop", 0.1))
        eval_offset = float(meta.get("eval_logit_offset", meta.get("eval_logit_bias", 0.0)))
        thr_key = "bpdr_calibrated_threshold" if (is_bpdr and "bpdr_calibrated_threshold" in meta) else "calibrated_threshold"
        threshold = float(meta.get(thr_key, meta.get("threshold", 0.65)))
        model_name = str(meta.get("model", meta.get("model_class", "FastTNetV2")))
        model = FastTNetV2(
            nin=nin,
            d=d,
            stride=stride,
            nhead=nhead,
            nlayers=nlayers,
            drop=drop,
            eval_logit_offset=eval_offset,
        )
    else:
        inferred_nin = in_weight.shape[1] - 5 if in_weight is not None else 5
        is_v5 = model_cls == "fasttnetv5" or (
            in_weight is not None and in_weight.shape[1] - V5_DELTA_CHANNELS in (5, 6)
        )
        if is_v5:
            inferred_nin = in_weight.shape[1] - V5_DELTA_CHANNELS
        nin_key = "bpdr_nin" if (is_bpdr and "bpdr_nin" in meta) else "nin"
        nin = int(meta.get(nin_key, inferred_nin))
        delta_ch = V5_DELTA_CHANNELS if is_v5 else 5
        if in_weight is not None and in_weight.shape[1] - delta_ch != nin:
            nin = in_weight.shape[1] - delta_ch
        d = int(meta.get("d", in_weight.shape[0] if in_weight is not None else 32))
        default_bias = 0.0 if is_v5 else DEFAULT_EVAL_LOGIT_BIAS
        eval_logit_bias = float(meta.get("eval_logit_offset", meta.get("eval_logit_bias", default_bias)))
        thr_key = "bpdr_calibrated_threshold" if (is_bpdr and "bpdr_calibrated_threshold" in meta) else "calibrated_threshold"
        threshold = float(meta.get(thr_key, meta.get("threshold", DEFAULT_THRESHOLD)))
        model_name = str(meta.get("model", meta.get("arch", "boundary_transformer")))
        model_ctor = FastTNetV5 if is_v5 else FastTNet
        model = model_ctor(nin=nin, d=d, nhead=nhead, nlayers=nlayers, eval_logit_bias=eval_logit_bias)

    model.load_state_dict(state)
    model.eval()
    model.default_threshold = threshold
    model.subframe_peaks = bool(meta.get("subframe_peaks", False))
    model.onset_shift_s = float(meta.get("onset_shift_s", 0.0))
    model.checkpoint_path = str(path)
    model.model_name = model_name
    model.metadata = meta
    return model


def canonical_hop_samples(sr=22050):
    """Canonical 20 ms hop in samples matching audio/analysis.py."""
    return max(1, int(round(sr * CANONICAL_FRAME_SECONDS)))


def align_track_to_20ms(trk, sr=22050):
    """Resample/align a pitch evidence track onto the canonical 20 ms (50 Hz) grid.

    Uses nearest native frame lookup (matching audio/analysis.py) so pitch values
    are never interpolated across unvoiced gaps.
    """
    src_t = np.asarray(trk["t"], dtype=np.float64)
    if len(src_t) == 0:
        raise ValueError("Pitch track is empty")
    if not np.all(np.isfinite(src_t)):
        raise ValueError("Pitch track timestamps must be finite")

    hop = canonical_hop_samples(sr)
    hop_s = hop / float(sr)
    k_start = int(round(float(src_t[0]) / hop_s))
    k_end = int(np.floor(float(src_t[-1]) / hop_s + 1e-6))
    if k_end < k_start:
        k_end = k_start
    t_grid = np.arange(k_start, k_end + 1, dtype=np.float64) * hop_s

    order = np.argsort(src_t) if np.any(np.diff(src_t) < 0) else None
    t_ref = src_t[order] if order is not None else src_t

    idx = np.searchsorted(t_ref, t_grid)
    idx = np.clip(idx, 0, len(t_ref) - 1)
    prev = np.maximum(idx - 1, 0)
    idx = np.where(np.abs(t_ref[prev] - t_grid) < np.abs(t_ref[idx] - t_grid), prev, idx)
    if order is not None:
        idx = order[idx]

    f0 = np.asarray(trk["f0"], dtype=np.float64)[idx]
    voiced = np.asarray(trk["voiced"], dtype=bool)[idx]
    conf_src = trk.get("conf")
    conf = (
        np.asarray(conf_src, dtype=np.float32)[idx]
        if conf_src is not None
        else np.ones(len(t_grid), dtype=np.float32)
    )
    return {
        "t": t_grid,
        "f0": f0,
        "voiced": voiced,
        "conf": conf,
        "hop_samples": hop,
        "hop_s": hop_s,
    }


def _extract_acoustic_cues(x, sr, t_grid, return_raw_onset=False):
    """Extract STFT log-energy (dB, z-scored) and 64-band log-mel positive flux matching analysis.py."""
    N = len(t_grid)
    if x is None or len(x) == 0 or N == 0:
        zeros = np.zeros(N, dtype=np.float32)
        if return_raw_onset:
            return zeros, zeros.copy(), np.zeros(N, dtype=np.float64)
        return zeros, zeros.copy()

    x_arr = np.asarray(x, dtype=np.float32)
    hop = canonical_hop_samples(sr)
    stft_mag = np.abs(librosa.stft(x_arr, n_fft=1024, hop_length=hop))
    spectrum = stft_mag ** 2
    mel = librosa.feature.melspectrogram(S=spectrum, sr=sr, n_mels=64)
    logmel = librosa.power_to_db(mel, ref=1.0, top_db=80).T
    rms = librosa.feature.rms(S=stft_mag, frame_length=1024)[0]
    logenergy_full = (20.0 * np.log10(np.maximum(rms, 1e-8))).astype(np.float32)
    flux = np.r_[0.0, np.maximum(np.diff(logmel, axis=0), 0.0).mean(axis=1)]
    onset_raw_full = flux / max(float(flux.max()), 1e-8)
    onset_full = onset_raw_full.astype(np.float32)

    times_stft = np.arange(len(logmel), dtype=np.float64) * hop / float(sr)
    logenergy_grid = np.interp(t_grid, times_stft, logenergy_full).astype(np.float32)
    onset = np.interp(t_grid, times_stft, onset_full).astype(np.float32)

    std = float(logenergy_grid.std())
    energy = ((logenergy_grid - float(logenergy_grid.mean())) / (std + 1e-6)).astype(np.float32)
    if return_raw_onset:
        onset_raw = np.interp(t_grid, times_stft, onset_raw_full)
        return energy, onset, onset_raw
    return energy, onset


def provisional_boundary_score(trk, x=None, sr=22050):
    """Compute analysis.py's provisional boundary score on the canonical 20 ms (50 Hz) grid."""
    aligned = align_track_to_20ms(trk, sr=sr)
    t_grid = aligned["t"]
    N = len(t_grid)
    f0 = aligned["f0"]
    voiced = aligned["voiced"] & np.isfinite(f0) & (f0 > 0)

    cents = np.full(N, np.nan, dtype=np.float64)
    cents[voiced] = 1200.0 * np.log2(f0[voiced] / 440.0)
    delta = np.zeros(N, dtype=np.float64)
    valid_delta = np.r_[False, voiced[1:] & voiced[:-1]]
    delta[valid_delta] = np.diff(cents)[valid_delta[1:]]

    _, _, onset_raw = _extract_acoustic_cues(x, sr, t_grid, return_raw_onset=True)
    transition = np.r_[0.0, voiced[1:] != voiced[:-1]].astype(np.float64)
    score = np.maximum.reduce([onset_raw, np.minimum(np.abs(delta) / 100.0, 1.0), transition]).astype(np.float32)
    return t_grid, score


def provisional_boundary_edges(trk, x=None, sr=22050, threshold=0.5, peak_distance=5, return_score=False):
    """Compute provisional boundary peak times (seconds) matching analysis.py + fit_r1_boundary.py."""
    t_grid, score = provisional_boundary_score(trk, x=x, sr=sr)
    pk, _ = find_peaks(score, height=threshold, distance=peak_distance)
    edges = t_grid[pk].astype(np.float64) if len(pk) > 0 else np.zeros(0, dtype=np.float64)
    if return_score:
        return edges, t_grid, score
    return edges


def extract_boundary_features(trk, x=None, sr=22050, boundary_edges=None, nin=5):
    """Build normalized 20 ms (50 Hz) input feature matrix matching training/analysis.py."""
    aligned = align_track_to_20ms(trk, sr=sr)
    t = aligned["t"]
    N = len(t)
    f0 = aligned["f0"]
    voiced = aligned["voiced"]
    ok = voiced & np.isfinite(f0) & (f0 > 0)

    # Pitch curve (cents relative to track median, scaled by 100 cents / semitone)
    if ok.sum() > 0:
        fine = 1200.0 * np.log2(np.maximum(f0[ok], 1e-6) / 440.0)
        med = np.median(fine)
        pitch = np.zeros(N, dtype=np.float32)
        pitch[ok] = (fine - med) / 100.0
    else:
        pitch = np.zeros(N, dtype=np.float32)

    ok_flag = ok.astype(np.float32)
    conf = aligned["conf"]
    energy, onset = _extract_acoustic_cues(x, sr, t)

    # Boundary track
    b_track = np.zeros(N, dtype=np.float32)
    if boundary_edges is not None and len(boundary_edges) > 0:
        t32 = t.astype(np.float32)
        for edge in boundary_edges:
            b_track = np.maximum(b_track, np.exp(-0.5 * ((t32 - float(edge)) / 0.04) ** 2).astype(np.float32))

    if nin == 3:
        cols = [pitch, ok_flag, conf]
    elif nin == 4:
        cols = [pitch, ok_flag, conf, b_track]
    elif nin == 6:
        cols = [pitch, ok_flag, conf, energy, onset, b_track]
    else:  # nin == 5 or default
        cols = [pitch, ok_flag, conf, energy, onset]

    return np.stack(cols, axis=-1).astype(np.float32)


def predict_boundary_probabilities(model, trk, x=None, sr=22050, boundary_edges=None, return_times=False):
    """Run model inference on the canonical 20 ms (50 Hz) grid."""
    model.eval()
    nin = getattr(model, "nin", 5)
    aligned = align_track_to_20ms(trk, sr=sr)
    if boundary_edges is None and nin in (4, 6):
        boundary_edges = provisional_boundary_edges(aligned, x=x, sr=sr)
    feats = extract_boundary_features(aligned, x=x, sr=sr, boundary_edges=boundary_edges, nin=nin)
    with torch.no_grad():
        inp = torch.from_numpy(feats[None, ...]).float()
        logits = model(inp)
        probs = torch.sigmoid(logits)[0].cpu().numpy().astype(float)
    if return_times:
        return aligned["t"], probs
    return probs


def extract_boundary_hints(
    trk,
    x=None,
    sr=22050,
    checkpoint_path=None,
    model=None,
    threshold=None,
    peak_distance=5,
    boundary_edges=None,
):
    """Extract canonical 20 ms boundary probability hints and picked peak locations."""
    if model is None:
        model = load_boundary_model(checkpoint_path)
    nin = getattr(model, "nin", 5)
    if boundary_edges is not None:
        boundary_edges_source = "provided"
    elif nin in (4, 6):
        boundary_edges = provisional_boundary_edges(trk, x=x, sr=sr)
        boundary_edges_source = "provisional_20ms"
    else:
        boundary_edges_source = "none"

    resolved_threshold = float(
        getattr(model, "default_threshold", DEFAULT_THRESHOLD) if threshold is None else threshold
    )
    t_grid, probs = predict_boundary_probabilities(
        model, trk, x=x, sr=sr, boundary_edges=boundary_edges, return_times=True
    )
    peak_idx, _ = find_peaks(probs, height=resolved_threshold, distance=peak_distance)
    hop_s = canonical_hop_samples(sr) / float(sr)
    subframe = bool(getattr(model, "subframe_peaks", False))
    shift_s = float(getattr(model, "onset_shift_s", 0.0))
    if subframe or shift_s != 0.0:
        peaks_s = refine_peak_times(probs, peak_idx, hop_s, subframe=subframe, shift_s=shift_s)
        peaks_s = [t0 for t0 in peaks_s if t0 >= 0.0]
    else:
        peaks_s = t_grid[peak_idx].tolist() if len(peak_idx) > 0 else []
    return {
        "times_s": np.round(t_grid, 4).tolist(),
        "probabilities": np.round(probs, 4).tolist(),
        "peaks_s": [round(float(p), 4) for p in peaks_s],
        "peak_indices": peak_idx.tolist(),
        "threshold": resolved_threshold,
        "subframe_peaks": subframe,
        "onset_shift_s": shift_s,
        "frame_hop_s": round(float(hop_s), 6),
        "boundary_edges_source": boundary_edges_source,
        "model": str(getattr(model, "model_name", "boundary_transformer")),
        "checkpoint": str(checkpoint_path or getattr(model, "checkpoint_path", DEFAULT_CHECKPOINT)),
    }


def write_boundary_hints_to_sidecar(sidecar_path, hints):
    """Attach boundary hints to an existing studio sidecar JSON."""
    p = Path(sidecar_path)
    if not p.exists():
        raise FileNotFoundError(f"Sidecar not found at: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    data["boundary_hints"] = hints
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return p


NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def midi_to_note_name(midi_num: int) -> str:
    """Convert integer MIDI note number to scientific pitch notation (e.g. 69 -> A4)."""
    m = int(round(midi_num))
    return f"{NOTE_NAMES[m % 12]}{(m // 12) - 1}"


def notes_from_track_with_boundary_hints(
    trk,
    boundary_hints,
    end_s: float,
    sr: int = 22050,
    min_frames: int = 4,
    split_unvoiced_gap_s: float = 0.10,
):
    """Form advisory segmented notes from a pitch track using FastTNet boundary peak hints.

    Matches the canonical 20 ms benchmark segmentation rule while also computing exact
    note duration (up to the last voiced frame in each segment) for hear-back synthesis.
    """
    aligned = align_track_to_20ms(trk, sr=sr)
    t = np.asarray(aligned["t"], dtype=float)
    f0 = np.asarray(aligned["f0"], dtype=float)
    voiced = np.asarray(aligned["voiced"], dtype=bool)
    conf = np.asarray(aligned["conf"], dtype=float)
    rel = voiced & np.isfinite(f0) & (f0 > 0)
    if len(t) == 0 or not rel.any():
        return []

    hop_s = float(aligned["hop_s"])
    fine = np.zeros(len(t), dtype=float)
    fine[rel] = 1200.0 * np.log2(np.maximum(f0[rel], 1e-6) / 440.0)

    peaks_s = boundary_hints.get("peaks_s", []) if isinstance(boundary_hints, dict) else list(boundary_hints)
    edges = np.sort(np.asarray(peaks_s, dtype=float))
    seg = np.digitize(t, edges)

    notes = []
    for s in np.unique(seg):
        idx = np.flatnonzero((seg == s) & rel)
        if len(idx) < min_frames:
            continue
        # Optionally split interior unvoiced silence gaps (>= split_unvoiced_gap_s) for clean audio playback
        if split_unvoiced_gap_s and split_unvoiced_gap_s > 0 and len(idx) > 1:
            cuts = np.flatnonzero(np.diff(t[idx]) > float(split_unvoiced_gap_s)) + 1
            sub_groups = np.split(idx, cuts)
        else:
            sub_groups = [idx]

        for g in sub_groups:
            if len(g) < min_frames:
                continue
            med_cents = float(np.median(fine[g]))
            midi_float = 69.0 + med_cents / 100.0
            midi_int = int(round(midi_float))
            if not (0 <= midi_int <= 127):
                continue
            onset_s = float(t[g[0]])
            stop_s = min(float(t[g[-1]]) + hop_s, float(end_s))
            dur_s = max(hop_s * len(g), stop_s - onset_s)
            dev_cents = float((midi_float - midi_int) * 100.0)
            vib_std = float(np.std(fine[g] - med_cents)) if len(g) >= 3 else 0.0
            notes.append({
                "pitch_midi": midi_int,
                "note_name": midi_to_note_name(midi_int),
                "onset_s": round(onset_s, 4),
                "duration_s": round(dur_s, 4),
                "median_hz": round(float(440.0 * (2.0 ** (med_cents / 1200.0))), 2),
                "deviation_cents": round(dev_cents, 2),
                "vibrato_std_cents": round(vib_std, 2),
                "mean_confidence": round(float(np.mean(conf[g])), 3),
                "n_frames": int(len(g)),
            })
    return notes


def analyze_boundary_spans(
    trk,
    boundary_hints,
    end_s: float,
    tuning_center_cents: float = None,
    sr: int = 22050,
    min_frames: int = 4,
):
    """Produce per-span boundary-disentangled pitch diagnostics and review-only repair candidates.

    Uses FastTNet boundaries to isolate individual sung note spans and measures each span's
    pitch center, tuning-calibrated deviation from nearest semitone, and vibrato stability.
    """
    notes = notes_from_track_with_boundary_hints(
        trk, boundary_hints, end_s=end_s, sr=sr, min_frames=min_frames, split_unvoiced_gap_s=0.10
    )
    tun = float(tuning_center_cents) if (tuning_center_cents is not None and np.isfinite(tuning_center_cents)) else 0.0
    spans = []
    for k, n in enumerate(notes, 1):
        raw_dev = float(n["deviation_cents"])
        # Wrap tuning-adjusted deviation into [-50, +50] cents around the singer's tuning grid
        adj_dev = float(((raw_dev - tun + 50.0) % 100.0) - 50.0)
        vib = float(n["vibrato_std_cents"])
        conf = float(n["mean_confidence"])
        dur = float(n["duration_s"])
        # A span is a review-gated repair candidate when it is sufficiently steady & voiced
        # but sits 18..45 cents off the singer's own tuning grid.
        is_candidate = (
            dur >= 0.12
            and conf >= 0.65
            and vib <= 45.0
            and 18.0 <= abs(adj_dev) <= 45.0
        )
        rec_shift = float(np.clip(-adj_dev, -35.0, 35.0)) if is_candidate else 0.0
        spans.append({
            "span_id": k,
            "start_s": n["onset_s"],
            "end_s": round(n["onset_s"] + dur, 4),
            "duration_s": dur,
            "pitch_midi": n["pitch_midi"],
            "note_name": n["note_name"],
            "median_hz": n["median_hz"],
            "raw_deviation_cents": round(raw_dev, 2),
            "tuning_adjusted_deviation_cents": round(adj_dev, 2),
            "vibrato_std_cents": round(vib, 2),
            "mean_confidence": conf,
            "review_repair_candidate": bool(is_candidate),
            "recommended_shift_cents": round(rec_shift, 2),
            "status": "REVIEW_CANDIDATE" if is_candidate else "IN_TUNE_PRESERVED",
        })
    return {
        "n_spans": len(spans),
        "n_review_candidates": sum(1 for s in spans if s["review_repair_candidate"]),
        "tuning_center_cents": round(tun, 2),
        "spans": spans,
    }

