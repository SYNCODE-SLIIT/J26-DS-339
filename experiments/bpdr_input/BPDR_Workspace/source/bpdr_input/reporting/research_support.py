"""Read-only research evidence checks and explicit regression metrics.

Research failures are findings, not exceptions to hide. No fitting, downloads,
audio generation or model-policy mutation occurs in this module.
"""
from collections import Counter
from dataclasses import asdict
from importlib.metadata import version
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from bpdr_input import COMPONENT_ROOT

WORKSPACE = COMPONENT_ROOT / "BPDR_Workspace"


def read(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required research artifact unavailable: {path}. Restore it; research execution never downloads or trains.")
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def context():
    run = Path(read(WORKSPACE/"results/model_recovery/current.json")["path"]).resolve()
    if not run.is_relative_to((WORKSPACE/"results/model_recovery").resolve()):
        raise ValueError("Recovery pointer leaves workspace results")
    return dict(workspace=WORKSPACE, run=run, evidence=read(run/"evidence.json"),
                declaration=read(run/"declaration.json"), selection=read(run/"locked_selection.json"),
                manifest=read(WORKSPACE/"datasets/vocalset_pilot/manifest.json"),
                verification=read(run/"rendered_verification.json"),
                comparison=Path(read(WORKSPACE/"results/model_comparison/current.json")["path"]))


def fixed_metrics(prediction, target, mask, tolerance=10.):
    """Per-case correction error on an unchanged scoring denominator.

    Empty masks or missing predictions are errors, never silently removed.
    Singer aggregation belongs to the caller; frames are not independent people.
    """
    prediction, target, mask = np.asarray(prediction), np.asarray(target), np.asarray(mask)
    if prediction.ndim != 1 or prediction.shape != target.shape or mask.shape != target.shape or mask.dtype != bool:
        raise ValueError("Metric arrays must be aligned 1D arrays and an explicit boolean mask")
    if not mask.any() or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("A nonempty fixed mask and positive finite tolerance are required")
    if not np.isfinite(prediction[mask]).all() or not np.isfinite(target[mask]).all():
        raise ValueError("Missing scored pitch must be reported, not dropped from the denominator")
    error = np.abs(prediction[mask]-target[mask])
    return dict(frames=int(mask.sum()), mae_cents=float(error.mean()),
                within_tolerance_percent=100*float(np.mean(error <= tolerance)))


def paired_reduction(before, after):
    before, after = np.asarray(before, dtype=float), np.asarray(after, dtype=float)
    if before.ndim != 1 or before.shape != after.shape or not len(before) or not np.isfinite([before,after]).all() or before.mean() <= 0:
        raise ValueError("Paired finite nonempty errors with positive baseline are required")
    return dict(cases=len(before), before_mae_cents=float(before.mean()), after_mae_cents=float(after.mean()),
                mean_paired_improvement_cents=float((before-after).mean()),
                relative_reduction_percent=100*float(1-after.mean()/before.mean()))


def source_location(function):
    path = Path(inspect.getsourcefile(function)).resolve()
    lines, start = inspect.getsourcelines(function)
    return dict(path=str(path), relative_path=path.relative_to(WORKSPACE).as_posix(),
                function=function.__qualname__, line=start, lines=len(lines), sha256=sha(path),
                code="".join(lines))


def audit():
    """Verify locally available evidence, including independent waveform rescoring."""
    import torch
    from bpdr_input.modeling.recovery_model import load_locked_diagnostic
    from bpdr_input.repair.rule_correction import fine_pitch
    torch.set_num_threads(2)
    c=context(); run=c["run"]; evidence=c["evidence"]; records=c["manifest"]["records"]
    checks=[]; inventory=[]
    def check(name, ok, expected, actual, path, scope="engineering"):
        checks.append(dict(check=name,status="PASS" if ok else "FAIL",expected=str(expected),actual=str(actual),
                           evidence=str(path),scope=scope))
    def asset(kind,path,expected_hash=None):
        path=Path(path);found=path.is_file();digest=sha(path) if found else None
        valid=found and (expected_hash is None or digest==expected_hash)
        inventory.append(dict(item=kind,expected_location=str(path),found=found,valid=valid,
                              evidence=digest,validation="SHA-256 checked" if expected_hash else "Present/read below; no prior hash asserted"))
        check(kind+": available/integrity",valid,expected_hash or "present",digest,path)
    asset("Dataset manifest",WORKSPACE/"datasets/vocalset_pilot/manifest.json",c["declaration"]["manifest_sha256"])
    asset("Selected ExtraTrees checkpoint",evidence["checkpoint"],evidence["checkpoint_sha256"])
    asset("Training declaration",run/"declaration.json",evidence["declaration_sha256"])
    asset("Pilot waveform receipt",run/"rendered_verification.json",evidence["rendered_verification_sha256"])
    asset("Historical notebook source",archived_original())
    check("Original recording/singer counts",len(records)==24 and len({r['singer_id'] for r in records})==12,
          "24 recordings / 12 source singer IDs",f"{len(records)} / {len({r['singer_id'] for r in records})}",WORKSPACE/"datasets/vocalset_pilot/manifest.json")
    decoded={}; waveform_rows=[]
    for r in records:
        path=Path(r["path"]);asset("Original "+path.name,path,r["sha256"])
        x,sr=sf.read(path,dtype="float32",always_2d=True)
        check("Decoded nonempty finite "+path.name,x.size>0 and np.isfinite(x).all(),"finite nonempty",str(x.shape),path)
        digest=hashlib.sha256(str(sr).encode()+x.tobytes()).hexdigest()
        decoded.setdefault(digest,[]).append(r["sha256"])
        waveform_rows.append(dict(singer=r['singer_id'],split=r['split'],recording=path.name,seconds=len(x)/sr,
                                  sample_rate=sr,channels=x.shape[1],decoded_sha256=digest))
    check("Exact decoded-source duplicates",all(len(v)==1 for v in decoded.values()),"none",sum(len(v)>1 for v in decoded.values()),"24 original decoded waveforms")
    splits=("train","calibration","pilot_evaluation")
    groups={s:{r['singer_id'] for r in records if r['split']==s} for s in splits}
    for i,a in enumerate(splits):
        for b in splits[i+1:]:
            overlap=groups[a]&groups[b]
            check("Singer overlap: "+a+" / "+b,not overlap,"empty",sorted(overlap),run/"declaration.json")
    pairs={s:read(run/(s+"_pairs.json")) for s in splits}; coverage=[]
    for split,rows in pairs.items():
        valid=[p for p in rows if "excluded" not in p]
        keys=[(p['source']['sha256'],p['cents']) for p in rows]
        check("Unique descendants "+split,len(keys)==len(set(keys)),"unique source-hash/shift",len(keys),run/(split+"_pairs.json"))
        check("Descendants retain source split "+split,all(p['source']['split']==split for p in rows),split,"all source records checked",run/(split+"_pairs.json"))
        for p in valid:
            with np.load(p['target_path']) as z:
                mask=z['mask'].astype(bool);target=z['target'];times=z['times']
            with np.load(Path(p['path_A']).parent/f"features_{p['cents']:+g}.npz") as z:
                aligned=target.shape==mask.shape==times.shape==z['times_seconds'].shape and np.array_equal(times,z['times_seconds'])
            check("Target/frame/sign "+split+" "+str(keys[rows.index(p)]),
                  aligned and mask.any() and np.isfinite(target).all() and np.any(mask&(target!=0)) and np.allclose(target[mask&(target!=0)],-p['cents']),
                  "aligned finite nonempty mask; reverse sign",f"{len(times)} frames / {mask.sum()} eligible",p['target_path'])
        coverage.append(dict(split=split,attempted=len(rows),audited=len(valid),excluded=len(rows)-len(valid),
                             contributing_recordings=len({p['source']['sha256'] for p in valid}),
                             contributing_singers=len({p['source']['singer_id'] for p in valid})))
    check("Training contribution",(coverage[0]['audited'],coverage[0]['contributing_recordings'],coverage[0]['contributing_singers'])==(43,13,7),
          "43 pairs / 13 originals / 7 singers",coverage[0],run/"train_pairs.json")
    model,payload=load_locked_diagnostic(Path(evidence['checkpoint']));estimator=model['estimator']
    check("Saved model schema",estimator.n_estimators==96 and estimator.n_features_in_==31,"96 trees / 31 columns",f"{estimator.n_estimators} / {estimator.n_features_in_}",evidence['checkpoint'])
    probe=estimator.predict(np.zeros((4,31)))
    check("Checkpoint finite output shape",probe.shape==(4,) and np.isfinite(probe).all(),"four finite signed-cent predictions",probe.tolist(),evidence['checkpoint'])
    check("Recorded training rows",all(v['training_rows']==16842 for v in evidence['fit_receipts'].values()),16842,evidence['fit_receipts'],run/'evidence.json')
    cal=read(run/'calibration_all.json')
    check("27 configurations retained",len(cal)==27,27,len(cal),run/'calibration_all.json')
    check("Calibration research acceptance",any(r['summary']['passes'] for r in cal),"at least one full pass",sum(r['summary']['passes'] for r in cal),run/'calibration_all.json',"research")
    check("Two-singer pilot acceptance",evidence['outer_proposal_gate_passes'],True,evidence['outer_proposal_gate_passes'],run/'evidence.json',"research")
    scored=[]
    for recorded,pair in zip(c['verification']['shifts'],pairs['pilot_evaluation']):
        if 'excluded' in pair:continue
        waves=[];rates=[]
        for path in (pair['path_A'],pair['path_C'],recorded['output']):
            wave,sr=sf.read(path,dtype='float32');waves.append(wave);rates.append(sr)
        with np.load(pair['target_path']) as z:times=z['times'];mask=z['mask'].astype(bool)&(z['target']!=0)
        pitches=[fine_pitch(wave,sr,times) for wave,sr in zip(waves,rates)]
        before=fixed_metrics(pitches[1],pitches[0],mask)['mae_cents'];after=fixed_metrics(pitches[2],pitches[0],mask)['mae_cents']
        check("Delivered pitch score "+str(pair['cents'])+" "+Path(pair['path_A']).parent.name,
              np.allclose([before,after],[recorded['input_mae_cents'],recorded['output_mae_cents']],atol=1e-6),
              [recorded['input_mae_cents'],recorded['output_mae_cents']],[before,after],recorded['output'])
        check("Delivered waveform grid "+Path(recorded['directory']).name,len(set(rates))==1 and all(x.shape==waves[0].shape for x in waves),
              "same rate and sample count",[rates,[len(x) for x in waves]],recorded['output'])
        scored.append(dict(singer=recorded['singer'],recording=pair['path_A'],shift=pair['cents'],before=before,after=after,
                           improved=recorded['verified_improvement'],outcome=recorded['outcome'],scored_frames=int(mask.sum())))
    summary=paired_reduction([r['before'] for r in scored],[r['after'] for r in scored])
    for r in c['verification']['originals']:
        a,sr=sf.read(r['input'],dtype='float32');b,br=sf.read(r['output'],dtype='float32')
        check("Exact no-change preservation "+Path(r['directory']).name,sr==br and np.array_equal(a,b),"identical working samples",bool(np.array_equal(a,b)),r['output'])
    check("Pilot denominator",len(c['verification']['shifts'])==16 and len(scored)==4 and sum(r['improved'] for r in scored)==2,
          "16 attempts / 4 scored / 2 improved",[len(c['verification']['shifts']),len(scored),sum(r['improved'] for r in scored)],run/'rendered_verification.json')
    extraction_link=read(WORKSPACE/'results/model_recovery/learned_extractor_fixture_current.json')
    extraction=read(extraction_link['receipt'])
    asset('Accepted extractor WAV',Path(extraction_link['handoff_directory'])/'accepted.wav',extraction['accepted_sha256'])
    for item in extraction['artifacts']:asset('Extractor '+Path(item['path']).suffix,item['path'],item['sha256'])
    checks.extend([dict(check='Natural intended-note ground truth',status='BLOCKED',expected='independent intended-note labels',actual='not present for current pilot',evidence=str(run/'declaration.json'),scope='research'),
                   dict(check='Blinded listening judgments',status='BLOCKED',expected='independent completed listening study',actual='automated fixtures only; no human approval claimed',evidence=str(extraction_link['receipt']),scope='research'),
                   dict(check='Downstream extraction F1',status='BLOCKED',expected='matching reference notes for this exact WAV',actual='MIDI/CSV connection only; reference unavailable',evidence=str(extraction_link['receipt']),scope='research'),
                   dict(check='Population singer confidence interval',status='NOT APPLICABLE',expected='multiple scoreable independent pilot singers',actual='one scoreable pilot singer; no population CI reported',evidence=str(run/'rendered_verification.json'),scope='statistics'),
                   dict(check='Cross-file acoustic overlap beyond exact hashes',status='BLOCKED',expected='trusted recording-session identities or exhaustive similarity study',actual='exact decoded duplicates and source/singer partitions checked; near-duplicate/session overlap not established',evidence=str(run/'declaration.json'),scope='research')])
    return dict(checks=checks,inventory=inventory,coverage=coverage,waveforms=waveform_rows,scored=scored,pilot_summary=summary,
                status_counts=dict(Counter(r['status'] for r in checks)),
                assessment='Share with noted caveats: executed engineering demonstration; research validation incomplete',
                protected_hashes={str(run/'evidence.json'):sha(run/'evidence.json'),str(evidence['checkpoint']):sha(evidence['checkpoint']),
                                  str(archived_original()):sha(archived_original())})


def archived_original():
    history=read(WORKSPACE/'documentation/notebook_history.json')
    path=Path(history['archived_original_notebook'])
    if sha(path)!=history['archived_original_sha256']:
        raise ValueError('Archived original notebook changed')
    return path


def historical_assets():
    index=read(WORKSPACE/'notebooks/evidence_index.json')
    return dict(release_manifest=index['evidence']['presentation_manifest'],
                release_manifest_sha256=index['current_gate']['presentation_manifest_sha256'],
                adaptation_evidence=index['evidence']['presentation_adapt'])
