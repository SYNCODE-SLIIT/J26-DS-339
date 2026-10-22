"""Shared source-time contracts; no annotation targets enter inference."""
from dataclasses import dataclass, asdict
import hashlib
import math
import numpy as np

@dataclass(frozen=True)
class Settings:
    tonic_pc: int = 0
    bpm: float = 120.0
    meter: int = 4
    first_beat_s: float = 0.0
    genre: str = 'pop'
    bpdr_bypass: bool = True
    rahul_mode: str = 'major'

    def validate(self):
        if self.tonic_pc not in range(12) or self.meter not in (3,4):
            raise ValueError('Choose a valid tonic and 3/4 or 4/4 meter')
        if not math.isfinite(self.bpm) or not 40 <= self.bpm <= 240:
            raise ValueError('BPM must be between 40 and 240')
        if not math.isfinite(self.first_beat_s) or self.first_beat_s < 0:
            raise ValueError('First beat must be a finite nonnegative time')
        if self.rahul_mode != 'major':
            raise ValueError('This demo fixes Rahul mode to major')
        return asdict(self)

def sha(path):
    with open(path,'rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()

def beat_grid(duration, settings):
    settings.validate()
    step = 60/settings.bpm
    # Only complete beat intervals; never extend audio to fill a partial final beat.
    count = int(np.floor((duration-settings.first_beat_s+1e-8)/step))
    if count < settings.meter:
        raise ValueError('At least one full bar is required after the first-beat offset')
    return [{'start_s':settings.first_beat_s+i*step,'end_s':settings.first_beat_s+(i+1)*step,
             'beat_index':i,'bar_number':i//settings.meter,'beat_position':i%settings.meter+1}
            for i in range(count)]

def validate_notes(notes,duration):
    if not notes: raise ValueError('No voiced melody notes survived segmentation')
    for n in notes:
        if not all(math.isfinite(n[k]) for k in ('pitch_midi','onset_s','duration_s')):
            raise ValueError('Nonfinite melody event')
        if not (0<=n['pitch_midi']<=127 and n['onset_s']>=0 and n['duration_s']>0 and n['onset_s']+n['duration_s']<=duration+1e-6):
            raise ValueError('Melody event outside recording bounds')
