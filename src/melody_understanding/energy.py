"""Experimental short-window energy gate for long-window pYIN voicing."""
import numpy as np
import librosa


def gate_voicing(features, audio, mode='relative_20db'):
    """Mask pYIN voiced frames using 512-sample RMS; keep source timestamps.

    Thresholds are recording-relative and may reject quiet real vocal notes.
    This function never manufactures voiced frames or bridges existing rests.
    """
    if mode not in ['relative_20db','relative_10db','adaptive_noise2']:
        raise ValueError('Unknown energy gate mode')
    rms=librosa.feature.rms(y=audio.samples,frame_length=512,hop_length=256)[0]
    if len(rms)<len(features['times_s']):
        raise ValueError('Energy and pitch frames have incompatible lengths')
    rms=rms[:len(features['times_s'])]
    peak=float(rms.max())
    if mode=='relative_20db':threshold=peak*.1
    elif mode=='relative_10db':threshold=peak*10**(-10/20)
    else:threshold=max(peak*.05,2*float(np.percentile(rms,20)))
    mask=features['voiced'] & (rms>threshold)
    return {**features,'voiced':mask}, {'mode':mode,'rms_threshold':threshold,
        'energy_frame_length':512,'energy_hop_length':256,
        'removed_voiced_frames':int((features['voiced'] & ~mask).sum())}
