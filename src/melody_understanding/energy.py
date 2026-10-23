"""Experimental short-window energy gate for long-window pYIN voicing."""
import numpy as np
import librosa


# Filtering Existing Voiced Frames by Short-Window Energy
def gate_voicing(features, audio, mode='relative_20db'):
    """Mask pYIN voiced frames using 512-sample RMS; keep source timestamps.

    Thresholds are recording-relative and may reject quiet real vocal notes.
    This function never manufactures voiced frames or bridges existing rests.
    """
    # Reject unsupported threshold policies before measuring the signal.
    if mode not in ['relative_20db','relative_10db','adaptive_noise2']:
        raise ValueError('Unknown energy gate mode')
    # Measure local energy on the pitch hop grid and match the frame count.
    rms=librosa.feature.rms(y=audio.samples,frame_length=512,hop_length=256)[0]
    if len(rms)<len(features['times_s']):
        raise ValueError('Energy and pitch frames have incompatible lengths')
    rms=rms[:len(features['times_s'])]
    # Set a recording-relative threshold or estimate it from low-energy frames.
    peak=float(rms.max())
    if mode=='relative_20db':threshold=peak*.1
    elif mode=='relative_10db':threshold=peak*10**(-10/20)
    else:threshold=max(peak*.05,2*float(np.percentile(rms,20)))
    # Remove low-energy voicing decisions while retaining the original pitch values.
    mask=features['voiced'] & (rms>threshold)
    return {**features,'voiced':mask}, {'mode':mode,'rms_threshold':threshold,
        'energy_frame_length':512,'energy_hop_length':256,
        'removed_voiced_frames':int((features['voiced'] & ~mask).sum())}
