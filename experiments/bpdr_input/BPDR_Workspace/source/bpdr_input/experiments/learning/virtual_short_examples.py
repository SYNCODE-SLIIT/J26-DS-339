"""Training-only F0 counterfactuals; never physical examples or runtime inputs."""
import numpy as np


def make_pair(raw,*,left_seconds,width_seconds,cents,vibrato_amplitude_cents=0.,vibrato_hz=5.,phase=0.,drift_cents=0.):
    raw=np.asarray(raw,dtype=np.float32)
    if raw.ndim!=2 or raw.shape[1]!=71 or not np.isfinite(raw).all():raise ValueError('Invalid acoustic input')
    if width_seconds not in [.24,.28,.36,.50] or cents not in [-30.,-20.,20.,30.]:raise ValueError('Unregistered virtual intervention')
    if not 0<=left_seconds<left_seconds+width_seconds<=len(raw)*.02:raise ValueError('Virtual support outside recording')
    if not 0<=vibrato_amplitude_cents<=20 or not 4<=vibrato_hz<=7 or abs(drift_cents)>15:raise ValueError('Invalid shared expression augmentation')
    t=np.arange(len(raw))*.02;voiced=raw[:,66]>.5
    pulse=((t>=left_seconds)&(t<left_seconds+width_seconds)).astype(float)
    pulse=np.convolve(pulse,[.1,.2,.4,.2,.1],mode='same')
    deliberate=vibrato_amplitude_cents*np.sin(2*np.pi*vibrato_hz*t+phase)+drift_cents*np.linspace(-1,1,len(raw))**2
    a=raw.copy();c=raw.copy()
    for result,extra in [(a,deliberate),(c,deliberate+cents*pulse)]:
        result[:,64]=np.where(voiced,raw[:,64]+extra,0.)
        if voiced.any():result[voiced,64]-=np.median(result[voiced,64])
        result[:,67]=0.
        result[1:,67]=np.where(voiced[1:]&voiced[:-1],np.diff(result[:,64]),0.)
    positive=voiced&(pulse>=.999);zero=voiced&(pulse==0.)
    if not positive.any() or not zero.any():raise ValueError('Missing virtual supervised class')
    return a,c,dict(positive=positive,zero=zero,pulse=pulse,
                    target=np.where(positive,-cents,0.).astype(np.float32),physical_waveform=False)
