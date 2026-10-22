"""Declared training-only renderer comparison; never adopts a renderer or creates pairs."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import uuid
import numpy as np
import soundfile as sf
from .analysis import extract
from .audit_diagnostics import failure_reasons, pitch_metrics
from .data_builder import validate_run, write_json
from .interventions import AuditPolicy, audit_shift, envelope, independent_pitch
from .pipeline import checksum

# Freeze this set before measuring. Active render_local is untouched.
CONDITIONS = {'fft2048_hop128': dict(n_fft=2048, hop_length=128),
              'fft1024_hop128': dict(n_fft=1024, hop_length=128),
              'fft1024_hop256': dict(n_fft=1024, hop_length=256)}


def candidate_audio(audio, sr, cents, support, settings, policy=AuditPolicy()):
    import librosa
    if cents == 0:
        return audio.copy()
    full = librosa.effects.pitch_shift(audio, sr=sr, n_steps=cents/100, scale=False, **settings)
    blend = envelope(np.arange(len(audio))/sr, support, policy.ramp_seconds)
    result = (audio*(1-blend)+full*blend).astype(np.float32)
    if result.shape != audio.shape or not np.isfinite(result).all() or abs(result).max() >= 1:
        raise ValueError('Candidate changed sample count, is non-finite or clips')
    return result


def run(diagnostic_path):
    import librosa
    import torch
    component = Path(__file__).resolve().parents[2]
    diagnostic_path = Path(diagnostic_path).resolve()
    if not diagnostic_path.is_relative_to(component):
        raise ValueError('Diagnostic must be inside BPDR')
    diagnostic = json.loads(diagnostic_path.read_text())
    prior_path = Path(diagnostic['source_evidence_path'])
    prior = json.loads(prior_path.read_text())
    policy = AuditPolicy()
    if diagnostic['run_status'] != 'complete' or prior['run_status'] != 'complete' or prior['policy'] != json.loads(json.dumps(asdict(policy))):
        raise ValueError('Completed unchanged-policy baseline required')
    if checksum(prior_path) != diagnostic['source_evidence_sha256']:
        raise ValueError('Baseline changed')
    prepared = Path(prior['preparation_path'])
    validate_run(prepared)
    records = {r['record_id']: r for r in json.loads((prepared/'manifest.json').read_text())['records']}
    probes = [p for p in prior['probes'] if p['shifts']]
    root = component/'runs/renderer_candidates'/uuid.uuid4().hex
    root.mkdir(parents=True)
    modules = ('renderer_candidates', 'audit_diagnostics', 'analysis', 'interventions')
    code = {name: checksum(Path(__file__).parent/(name+'.py')) for name in modules}
    snapshots = root/'source_code_at_run'
    snapshots.mkdir()
    for name in modules:
        (snapshots/(name+'.py')).write_bytes((Path(__file__).parent/(name+'.py')).read_bytes())
    result = dict(run_status='running', comparison_version='renderer-comparison-exploratory-0.1',
                  data_status='exploratory_not_research', training_ready_examples=0, conditions=CONDITIONS,
                  declared_record_ids=[p['record_id'] for p in probes], shifts_cents=policy.shifts_cents,
                  diagnostic_path=str(diagnostic_path), diagnostic_sha256=checksum(diagnostic_path),
                  source_evidence_path=str(prior_path), source_evidence_sha256=checksum(prior_path),
                  policy=asdict(policy), baseline_summary=diagnostic['summary'], code_sha256=code,
                  librosa_version=librosa.__version__, torch_version=torch.__version__, cpu_threads=torch.get_num_threads(),
                  variants=[], controls=[], adopted=False,
                  limitations=['Exploratory comparison on three already inspected training crops/two artist proxies',
                               'No human approvals, held-out validation, audible-quality test or production renderer',
                               'No automatic winner selection; unchanged active v0.3 renderer and thresholds'])
    def save(): write_json(root/'comparison.json', result)
    save()
    print('CANDIDATE_RUN:', root, flush=True)
    try:
        for probe in probes:
            record = records[probe['record_id']]
            if record['split'] != 'train' or record['training_eligible'] or checksum(record['accepted_audio_path']) != probe['accepted_sha256']:
                raise ValueError('Unsafe partition/eligibility or changed accepted source')
            audio, sr = sf.read(record['accepted_audio_path'], dtype='float32')
            before, _ = extract(audio, sr, component/'checkpoints')
            yin = independent_pitch(audio, sr)
            with np.load(record['screen_masks']['path']) as archive:
                reliable = archive['reliable_mask']
                if not np.array_equal(before['times_seconds'], archive['times_seconds']):
                    raise ValueError('Time grids differ')
            support = probe['geometry']['support_seconds']
            outside = envelope(np.arange(len(audio))/sr, support, policy.ramp_seconds) == 0
            for name, settings in CONDITIONS.items():
                identity = candidate_audio(audio, sr, 0., support, settings, policy)
                for cents in policy.shifts_cents:
                    print('CANDIDATE:', probe['record_id'], name, cents, flush=True)
                    shifted = candidate_audio(audio, sr, cents, support, settings, policy)
                    after, _ = extract(shifted, sr, component/'checkpoints')
                    measured, _, _ = audit_shift(before, after, yin, independent_pitch(shifted, sr), reliable, support, cents, policy)
                    measured.update(requested_cents=cents, identity_pass=bool(np.array_equal(identity,audio)),
                                    same_sample_count=len(shifted)==len(audio), outside_support_exact=bool(np.array_equal(audio[outside],shifted[outside])))
                    reasons = failure_reasons(measured, policy)
                    measured['passed'] = not reasons
                    result['variants'].append(dict(record_id=probe['record_id'], source_id=record['source_id'], group_id=record['group_id'],
                                                   condition=name, cents=cents, audit=measured, reasons=reasons,
                                                   generated_float32_waveform_sha256=hashlib.sha256(shifted.tobytes()).hexdigest()))
                    save()
        # Matched constructed controls; oracle in the preceding diagnostic is the reference.
        sr=22050
        time=np.arange(4*sr)/sr
        for profile, f0 in [('constant_440',np.full(len(time),440.)),('vibrato_220',220.*2**(35*np.sin(2*np.pi*5*time)/1200))]:
            phase=2*np.pi*np.cumsum(f0)/sr
            original=(.12*(np.sin(phase)+.4*np.sin(2*phase)+.2*np.sin(3*phase))).astype(np.float32)
            yin=independent_pitch(original,sr)
            mask=(np.arange(len(yin))*.02>=1.) & (np.arange(len(yin))*.02<=3.)
            for name, settings in CONDITIONS.items():
                for cents in policy.shifts_cents:
                    shifted=librosa.effects.pitch_shift(original,sr=sr,n_steps=cents/100,scale=False,**settings)
                    result['controls'].append(dict(profile=profile,condition=name,cents=cents,
                        fine=pitch_metrics(yin,independent_pitch(shifted,sr),mask,cents)))
        result['summary']={}
        for name in CONDITIONS:
            rows=[r for r in result['variants'] if r['condition']==name]
            passed=[r for r in rows if r['audit']['passed']]
            complete=[p for p in probes if sum(r['record_id']==p['record_id'] for r in passed)==len(policy.shifts_cents)]
            result['summary'][name]=dict(passing_variants=len(passed),tested_variants=len(rows),
                any_passing_crops=len({r['record_id'] for r in passed}),any_passing_artists=len({r['group_id'] for r in passed}),
                all_six_crops=len(complete),all_six_artists=len({p['group_id'] for p in complete}))
        validate_run(prepared)
        if checksum(diagnostic_path)!=result['diagnostic_sha256'] or checksum(prior_path)!=result['source_evidence_sha256'] or any(checksum(Path(__file__).parent/(n+'.py'))!=d for n,d in code.items()):
            raise ValueError('Evidence/code changed during comparison')
        result['run_status']='complete'
        save()
        print(json.dumps(result['summary'],indent=2),flush=True)
    except BaseException as exc:
        result.update(run_status='failed',failure=dict(type=type(exc).__name__,reason=str(exc)))
        save()
        raise
    return root


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('diagnostic',type=Path)
    args=parser.parse_args()
    run(args.diagnostic)


if __name__=='__main__': main()
