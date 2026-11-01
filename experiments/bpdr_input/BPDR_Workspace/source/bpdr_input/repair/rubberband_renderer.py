"""Pinned offline Rubber Band R3 constant-step renderer, separate from live demo.

Use only an already permitted constant correction and unchanged support. The
same post-render and encoded-WAV audits still determine whether output survives.
No formant or perceptual-quality claim is made for this resynthesis.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import subprocess
import tempfile
import soundfile as sf

import numpy as np

from bpdr_input.repair.interventions import envelope
from bpdr_input.repair.repair_controller import check_source


WORKSPACE=Path(__file__).resolve().parents[3]
DEPENDENCY=WORKSPACE/'results/transformer_usability/dependencies/rubberband_4.0.0/extracted'
PINNED={'rubberband-r3.exe':'f4cd498f36e22e624268e3e5aa47803449cb786f7850b83ddd26324162fcdb8a',
        'sndfile.dll':'4e3bd2de8e1485110eaebef8e1239471f73d608773831c323bf528e05645655e'}


def verify_dependency(folder=DEPENDENCY):
    for name,digest in PINNED.items():
        if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Unpinned Rubber Band dependency')
    return folder/'rubberband-r3.exe'


@dataclass(frozen=True)
class RubberBandPolicy:
    version: str = 'rubberband-r3-constant-step-1'
    rubberband_version: str = '4.0.0'
    ramp_seconds: float = .08
    max_cents: float = 30.
    min_support_seconds: float = .86


def render_candidate(audio, sr, times, correction_cents, intervals, policy=RubberBandPolicy()):
    check_source(audio, sr)
    times, correction = np.asarray(times), np.asarray(correction_cents)
    expected = np.arange(len(audio)//441+1)*.02
    if (times.shape != expected.shape or not np.allclose(times, expected, atol=1e-12, rtol=0)
            or correction.shape != times.shape or not np.isfinite(correction).all()
            or np.max(abs(correction)) > policy.max_cents):
        raise ValueError('Invalid Rubber Band R3 grid or over-limit correction')
    sample_times = np.arange(len(audio))/sr
    covered = np.zeros(len(times), bool)
    all_blend = np.zeros(len(audio))
    supports = []
    previous = 0.
    for start, end in intervals:
        if (not np.isfinite(start+end) or start < previous or not 0 <= start < end <= len(audio)/sr
                or end-start < policy.min_support_seconds-1e-12):
            raise ValueError('Invalid or too-short Rubber Band R3 support')
        previous = end
        mask = (times >= start) & (times < end)
        values = correction[mask]
        if not len(values) or not np.all(values == values[0]):
            raise ValueError('Phase-vocoder condition requires a constant predicted plateau')
        covered |= mask
        blend = envelope(sample_times, (start, end), policy.ramp_seconds)
        if np.any((all_blend > 0) & (blend > 0)):
            raise ValueError('Overlapping local envelopes')
        all_blend += blend
        supports.append((float(values[0]), blend))
    if np.any((correction != 0) & ~covered):
        raise ValueError('Correction outside permitted support')
    diagnostics = dict(policy=asdict(policy), production_eligible=False,
                       scope='experimental_renderer_only', identity=not bool(np.any(correction)),
                       outside_samples_exact=True, same_sample_count=True,
                       perceptual_quality='not assessed; no formant-preservation claim')
    if not np.any(correction):
        return audio.copy(), diagnostics
    executable=verify_dependency()
    scratch=(DEPENDENCY.parent/'scratch').resolve();scratch.mkdir(exist_ok=True)
    if not scratch.is_relative_to(WORKSPACE.resolve()):raise ValueError('Invalid scratch directory')
    result = audio.astype(np.float64).copy()
    for cents, blend in supports:
        if cents == 0:
            continue
        with tempfile.TemporaryDirectory(prefix='r3_',dir=scratch) as tmp:
            folder=Path(tmp).resolve()
            if not folder.is_relative_to(scratch):raise ValueError('Invalid temporary directory')
            source,output=folder/'input.wav',folder/'output.wav'
            sf.write(source,audio,sr,subtype='FLOAT')
            command=[str(executable),'-3','--ignore-clipping','-q','-t','1','-p',format(cents/100.,'.17g'),str(source),str(output)]
            process=subprocess.run(command,capture_output=True,text=True,timeout=30,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
            if process.returncode!=0:raise ValueError('Rubber Band failed: '+process.stderr[-500:])
            full,output_sr=sf.read(output,dtype='float32')
            if output_sr!=sr:raise ValueError('Rubber Band changed sample rate')
        if full.shape != audio.shape or not np.isfinite(full).all():
            raise ValueError('Pitch-shift returned a different length or nonfinite audio')
        result += (full-audio)*blend
    result = result.astype(np.float32)
    if not np.isfinite(result).all() or np.max(abs(result)) >= 1.:
        raise ValueError('Nonfinite/clipped Rubber Band R3 output; no normalization')
    if not np.array_equal(result[all_blend == 0], audio[all_blend == 0]):
        raise ValueError('Samples outside support changed')
    diagnostics.update(engine="R3 offline",auto_gain_disabled=True,formant_preservation_requested=False,dependency_sha256=PINNED)
    return result, diagnostics
