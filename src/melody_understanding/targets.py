"""Weak intended-score targets; never presented as performed-pitch truth."""
import numpy as np
import librosa


def score_targets(notes, times, duration_s, boundary_margin_s=.1):
    """Encode rest=0, pitch class=C..B as 1..12; mask uncertain boundaries."""
    if boundary_margin_s<0:
        raise ValueError('Boundary margin must be nonnegative')
    labels=np.zeros(len(times),dtype=np.int64)
    active=np.zeros(len(times),dtype=bool)
    keep=(times>=0)&(times<duration_s)
    boundaries=[0.,duration_s]
    for start,end,pitch in notes:
        if start<0 or end<=start or not 0<=pitch<=127:
            raise ValueError('Invalid MIDI note interval or pitch')
        mask=(times>=start)&(times<end)
        if (active&mask).any():raise ValueError('Overlapping reference notes')
        labels[mask]=int(pitch)%12+1
        active|=mask;boundaries.extend([start,end])
    for boundary in boundaries:keep&=np.abs(times-boundary)>boundary_margin_s
    return labels,keep


def make_score_features(audio,notes,boundary_margin_s=.1):
    """Extract fixed Mel frames with labels on the original recording timeline."""
    mel=librosa.feature.melspectrogram(y=audio.samples,sr=audio.sample_rate,
        n_fft=4096,hop_length=256,n_mels=80,power=2)
    mel=librosa.power_to_db(mel,ref=np.max).astype(np.float32)
    times=audio.offset_s+np.arange(mel.shape[1])*256/audio.sample_rate
    inside=times<audio.source_duration_s
    mel,times=mel[:,inside],times[inside]
    labels,keep=score_targets(notes,times,audio.source_duration_s,boundary_margin_s)
    return {'mel_db':mel,'times_s':times,'score_class':labels,'supervision_mask':keep}
