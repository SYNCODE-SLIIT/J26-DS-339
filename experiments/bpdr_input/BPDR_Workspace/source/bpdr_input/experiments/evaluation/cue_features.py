"""Runtime-only tuning-cue features derived from one view's own pitch track.

Nothing here reads a target, a private mask or a paired original. The cue is the
distance of each voiced frame from the view's own semitone grid (after removing
the view's overall tuning offset) plus short-window pitch steps. These are
BPDR-local derived inputs; stored feature archives and the shared pipeline are
never changed.
"""
import numpy as np

VERSION = 'tuning-cue-features-0.1'
PITCH, CONFIDENCE, VOICED = 64, 65, 66
HALF_WIDTHS = (7, 21, 63)
STEP_WIDTHS = (7, 21)
NAMES = (['grid_sin', 'grid_cos']
         + [f'{kind}_w{w}' for w in HALF_WIDTHS for kind in ('dev', 'resultant')]
         + [f'step_w{w}' for w in STEP_WIDTHS])
WIDTH = len(NAMES)


def _window_sum(x, left, right):
    """Sum of x[i-left : i+right+1] with zero padding, exact via cumulative sums."""
    c = np.concatenate([[0.], np.cumsum(x, dtype=np.float64)])
    n = len(x)
    lo = np.clip(np.arange(n)-left, 0, n)
    hi = np.clip(np.arange(n)+right+1, 0, n)
    return c[hi]-c[lo]


def tuning_offset(pitch, voiced):
    """Circular-mean offset of the voiced pitch track modulo 100 cents."""
    if voiced.sum() < 20:
        return 0.
    return float(np.angle(np.mean(np.exp(2j*np.pi*(pitch[voiced] % 100)/100)))*100/(2*np.pi))


def cue_features(raw):
    """raw: frames x 71 unnormalized archive features. Returns frames x WIDTH float32."""
    pitch = raw[:, PITCH].astype(np.float64)
    voiced = (raw[:, VOICED] > .5) & (pitch != 0)
    t = tuning_offset(pitch, voiced)
    theta = 2*np.pi*(pitch-t)/100
    z = np.where(voiced, np.exp(1j*theta), 0)
    out = [np.where(voiced, np.sin(theta), 0), np.where(voiced, np.cos(theta), 0)]
    v = voiced.astype(np.float64)
    for w in HALF_WIDTHS:
        count = _window_sum(v, w, w)
        s = _window_sum(z.real, w, w)+1j*_window_sum(z.imag, w, w)
        ok = count > 0
        dev = np.where(ok, np.angle(s)/np.pi, 0)          # [-1, 1] = +-50 cents
        res = np.where(ok, np.abs(s)/np.maximum(count, 1), 0)
        out += [np.where(voiced, dev, 0), np.where(voiced, res, 0)]
    pv = np.where(voiced, pitch, 0)
    for w in STEP_WIDTHS:
        lc, rc = _window_sum(v, w, -1), _window_sum(v, -1, w)
        lm = _window_sum(pv, w, -1)/np.maximum(lc, 1)
        rm = _window_sum(pv, -1, w)/np.maximum(rc, 1)
        step = np.where((lc > 0) & (rc > 0) & voiced, np.clip((rm-lm)/100, -2, 2), 0)
        out.append(step)
    result = np.column_stack(out).astype(np.float32)
    if result.shape != (len(raw), WIDTH) or not np.isfinite(result).all():
        raise ValueError('Invalid cue features')
    return result
