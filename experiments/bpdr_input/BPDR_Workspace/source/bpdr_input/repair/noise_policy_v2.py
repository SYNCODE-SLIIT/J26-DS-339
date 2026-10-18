"""Conservative colored-reference extension, awaiting independent-noise checks.

The original .25 flatness gate rejected steady recorded rain (.0108) despite
passing periodicity/stationarity. Relax only flatness to .01 for declared room
noise; reduce attenuation to 3 dB. Do not relax voice-preservation checks.
"""
from dataclasses import dataclass
from bpdr_input.repair.noise_preview import NoisePolicy


@dataclass(frozen=True)
class ColoredNoisePolicy(NoisePolicy):
    version: str='declared-colored-reference-preview-2'
    min_flatness: float=.01
    max_attenuation_db: float=3.
