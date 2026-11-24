"""Synthetic short-support measurement probe, never a model accuracy benchmark."""
from pathlib import Path
from dataclasses import asdict, replace
import hashlib
import json
import sys
import numpy as np
import soundfile as sf
import torch

WS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WS / 'source'))
from bpdr_input.repair.framewise_renderer import RendererPolicy, render_candidate
from bpdr_input.repair.interventions import AuditPolicy
from bpdr_input.repair.rule_correction import fine_hz
from bpdr_input.repair.centered_precision_audit import coarse_features
from bpdr_input.repair.framewise_audit import audit_curve

ROOT = WS / 'results/transformer_usability/2026-11-02_v38_short_phrase_probe'
SR = 22050


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def phrase(base, width):
    """Six-second harmonic fixture, ten changing notes, one central probe note."""
    y = np.zeros(6 * SR, np.float32)
    starts = [.2, .72, 1.24, 1.76, 2.40, 3.40, 3.92, 4.44, 4.96, 5.48]
    semitones = [0, 2, 4, 2, 0, 5, 4, 2, 0, -2]
    for index, (start, semitone) in enumerate(zip(starts, semitones)):
        duration = width + .12 if index == 4 else .42
        a, b = round(start*SR), round((start+duration)*SR)
        t = np.arange(b-a)/SR
        f = base * 2**(semitone/12)
        env = np.minimum(1, np.minimum(t/.01, (duration-t)/.01))
        wave = sum(np.sin(2*np.pi*k*f*t)/(k*k) for k in range(1, 9))
        y[a:b] = (.18*wave*env).astype(np.float32)
    # PCM24 source is the identity reference for encoded-output comparisons.
    return y, (2.46, 2.46+width)


def main():
    torch.set_num_threads(2)
    ROOT.mkdir(parents=True, exist_ok=True)
    default = RendererPolicy()
    exploratory = replace(default, version='short-support-probe-only-1', min_support_seconds=.24)
    declaration = dict(scope='synthetic renderer/measurement feasibility only',
        source_sha256=sha(__file__), frequencies_hz=[120, 240], support_seconds=[.24, .36, .50, .70],
        shifts_cents=[-30., 30.], duration_seconds=6,
        default_renderer=asdict(default), experimental_renderer=asdict(exploratory),
        audit_policy=asdict(AuditPolicy()),
        hypothesis='Shorter renderer supports may execute, but the unchanged audit may lack ten frames.',
        synthetic=True, real_human_validation=False, transformer_inference=False,
        accepted_audio=None, activation=False,
        note='Known request and support are fixtures, not inferred correction or musical intention. Only minimum renderer support changes; audit thresholds and ramp remain unchanged.')
    decl = ROOT/'declaration.json'
    declaration=json.loads(json.dumps(declaration))
    if decl.exists():
        previous=json.loads(decl.read_text(encoding='utf-8'))
        assert {k:v for k,v in previous.items() if k!='source_sha256'} == {k:v for k,v in declaration.items() if k!='source_sha256'}, 'Immutable protocol mismatch'
        if previous['source_sha256'] != declaration['source_sha256']:
            assert sha(ROOT/'initial_probe_source.py')==previous['source_sha256']
            write(ROOT/'resume_amendment.json',dict(original_declaration_sha256=sha(decl),
                original_source_sha256=previous['source_sha256'], resumed_source_sha256=sha(__file__),
                reason='Normalize JSON tuple/list representation for restart comparison. First process interrupted after four cases; same full sixteen-case protocol repeated. No numerical or policy change.'))
    else:
        write(decl, declaration)
    rows=[]
    for base in declaration['frequencies_hz']:
        for width in declaration['support_seconds']:
            folder=ROOT/f'{base}hz_{width:.2f}s'; folder.mkdir(exist_ok=True)
            audio, support=phrase(base,width)
            original=folder/'original.wav'; sf.write(original,audio,SR,subtype='PCM_24')
            audio,_=sf.read(original,dtype='float32')
            times=np.arange(len(audio)//441+1)*.02
            before=coarse_features(audio,SR,WS/'checkpoints')
            fine_before=fine_hz(audio,SR,times)
            reliable=before['voiced_mask'] & np.isfinite(fine_before) & (fine_before>0)
            identity,_=render_candidate(audio,SR,times,np.zeros(len(times)),[],exploratory)
            assert np.array_equal(audio,identity)
            for cents in declaration['shifts_cents']:
                curve=np.where((times>=support[0])&(times<support[1]),cents,0.)
                try:
                    render_candidate(audio,SR,times,curve,[support],default)
                except ValueError as exc:
                    default_rejection=str(exc)
                else:
                    raise AssertionError('Default policy unexpectedly accepted a short support')
                candidate,diagnostic=render_candidate(audio,SR,times,curve,[support],exploratory)
                path=folder/f'shift_{cents:+g}.wav';sf.write(path,candidate,SR,subtype='PCM_24')
                decoded,rate=sf.read(path,dtype='float32'); assert rate==SR and decoded.shape==audio.shape
                sample_times=np.arange(len(audio))/SR
                outside=(sample_times<=support[0])|(sample_times>=support[1])
                assert np.array_equal(decoded[outside],audio[outside])
                after=coarse_features(decoded,SR,WS/'checkpoints'); fine_after=fine_hz(decoded,SR,times)
                audit=audit_curve(before,after,fine_before,fine_after,reliable,[support],curve)
                row=dict(base_hz=base,width_seconds=width,cents=cents,
                    source_sha256=sha(original),output_sha256=sha(path),
                    default_rejection=default_rejection, sample_count=len(decoded),
                    outside_samples_exact=True,identity_fixture_exact=True,
                    renderer=diagnostic,audit=audit)
                rows.append(row);write(folder/f'result_{cents:+g}.json',row)
                print('SHORT_SUPPORT',base,width,cents,'frames',audit['common_frames'],
                      'audit',audit['passed'],'reasons',audit['reasons'],flush=True)
    write(ROOT/'results.json',dict(declaration_sha256=sha(decl),cases=rows,
        attempted=len(rows),audit_passed=sum(r['audit']['passed'] for r in rows),
        model_accuracy_assessed=False,real_human_validation=False,activation=False))
    print('SHORT_PHRASE_PROBE_COMPLETE',len(rows),flush=True)


if __name__=='__main__':main()
