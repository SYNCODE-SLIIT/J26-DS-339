"""Isolated FCPE measurement diagnostic; no model/auditor/runtime replacement."""
from math import gcd
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import torch
from scipy.signal import resample_poly

VERSION = 'fcpe-native-frame-diagnostic-1'


def align_native(times, frequencies, requested, minimum=50., maximum=1500.):
    """Nearest actual frame, earlier tie; never extrapolate or fill unvoiced gaps."""
    times = np.asarray(times, float); frequencies = np.asarray(frequencies, float)
    requested = np.asarray(requested, float)
    if (times.ndim != 1 or frequencies.shape != times.shape or len(times) == 0
            or np.any(np.diff(times) <= 0) or not np.isfinite(times).all()
            or requested.ndim != 1 or not np.isfinite(requested).all()):
        raise ValueError('Invalid native measurement grid')
    right = np.clip(np.searchsorted(times, requested), 0, len(times) - 1)
    left = np.maximum(right - 1, 0)
    indices = np.where(abs(requested - times[left]) <= abs(requested - times[right]) + 1e-12, left, right)
    values = frequencies[indices]
    valid = ((requested >= times[0]) & (requested <= times[-1])
             & (abs(requested - times[indices]) <= .005 + 1e-12)
             & np.isfinite(values) & (values >= minimum) & (values <= maximum))
    output = np.full(len(requested), np.nan)
    output[valid] = values[valid]
    return output, valid, indices


class FCPEMeasurement:
    def __init__(self, dependency_root):
        self.root = Path(dependency_root).resolve()
        receipt = json.loads((self.root / 'receipt.json').read_text())
        for path, digest in receipt['file_hashes'].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
        site = str(self.root / 'site_packages')
        if site not in sys.path:
            sys.path.insert(0, site)
        from torchfcpe import spawn_bundled_infer_model
        self.model = spawn_bundled_infer_model(device='cpu').eval()
        assert self.model.get_model_sr() == 16000 and self.model.get_hop_size() == 160
        config = self.model.get_mel_config()
        assert config['n_fft'] == config['win_size'] == 1024
        self.hop = 160; self.rate = 16000
        self.pad_left = (1024 - 160) // 2
        self.origin_seconds = (1024 / 2 - self.pad_left) / 16000
        self.checkpoint = self.root / 'site_packages/torchfcpe/assets/fcpe_c_v001.pt'

    @torch.inference_mode()
    def native(self, audio, sample_rate):
        audio = np.asarray(audio)
        if audio.ndim != 1 or len(audio) < sample_rate * .25 or not np.isfinite(audio).all():
            raise ValueError('Invalid mono diagnostic audio')
        common = gcd(int(sample_rate), self.rate)
        waveform = resample_poly(audio, self.rate // common, int(sample_rate) // common).astype(np.float32)
        tensor = torch.from_numpy(waveform)[None, :, None]
        # Use official mel/model methods, avoiding wrapper terminal-frame repeat,
        # output resizing, UV interpolation and maximum-frequency clamping.
        mel = self.model.wav2mel.mel_extractor(tensor)
        f0 = self.model.model.infer(mel, decoder='local_argmax', threshold=.006).flatten().cpu().numpy()
        pad_right = max((1024 - 160 + 1) // 2, 1024 - len(waveform) - self.pad_left)
        expected = (len(waveform) + self.pad_left + pad_right - 1024) // 160 + 1
        if len(f0) != expected or not np.isfinite(f0).all():
            raise ValueError('FCPE native-frame shape/nonfinite failure')
        times = self.origin_seconds + np.arange(len(f0)) * .01
        return times, f0, dict(resampled_samples=len(waveform), native_frames=len(f0),
                               native_hop_seconds=.01, native_origin_seconds=self.origin_seconds,
                               no_terminal_frame_repeat=True, no_frequency_clamp=True,
                               no_uv_interpolation=True)
