"""Decision-level evaluation of the rule-based correction on the frozen development variants.

Thresholds are the fixed RulePolicy values (not tuned on these results). For each development record:
  original (A waveform, nothing injected): does the rule edit acceptable singing? (harmful-edit risk)
  detuned  (C waveform, known injected shift on the audited correction frames):
      is the shifted note edited, in which direction, and what residual remains against the original
      performance (fine pitch of C + rule correction - fine pitch of A) on the audited correction frames?
Uses the stored PESTO voicing/pitch for reliability and fresh YIN on the existing waveforms. No rendering here.
"""
from bpdr_input import COMPONENT_ROOT, module_source
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import uuid
import numpy as np
import soundfile
from bpdr_input.repair.rule_correction import RulePolicy, decide, fine_pitch, reliable_frames
from dataclasses import asdict

COMPONENT = COMPONENT_ROOT
DEV = COMPONENT/'runs/support_development_packaging/779ea64608384daa9014865187c365b4/combined_catalog.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stored_features(path):
    with np.load(path) as z:
        x = z['boundary_features_71']; times = z['times_seconds']
    voiced = (x[:, 66] > .5) & (x[:, 64] != 0)
    return dict(times_seconds=times, voiced_mask=voiced, f0_hz=np.where(voiced, 440*2**(x[:, 64]/1200), np.nan))


def analyse(wav, archive, policy):
    audio, sr = soundfile.read(wav, dtype='float32')
    f = stored_features(archive)
    cents = fine_pitch(audio, sr, f['times_seconds'])
    mask = reliable_frames(f, cents, policy)
    corr, _, offset, notes = decide(cents, mask, policy)
    return cents, mask, corr, notes


def run():
    policy = RulePolicy()
    out = COMPONENT/'runs/rule_evaluation'/uuid.uuid4().hex; out.mkdir(parents=True)
    (out/'declaration.json').write_text(json.dumps(dict(declared_utc=datetime.now(timezone.utc).isoformat(), policy=asdict(policy),
        development_catalog_sha256=sha(DEV), source_sha256={n: sha(module_source(n+'.py')) for n in ('rule_evaluation', 'rule_correction')},
        note='Thresholds fixed before this evaluation; no tuning on its results.'), indent=2), encoding='utf-8')
    records = json.loads(DEV.read_text())['records']
    originals, detuned, seen_a, seen_c = [], [], set(), set()
    for r in records:
        d = Path(r['input_directory'])
        key_a, key_c = (r['dataset'], r['phrase_id']), (r['dataset'], r['phrase_id'], r['cents'])
        if key_a in seen_a and key_c in seen_c:
            continue
        ca, ma, corr_a, notes_a = analyse(r['waveform_paths']['A'], d/'A_input.npz', policy)
        if key_a not in seen_a:
            seen_a.add(key_a)
            edits = [n for n in notes_a if n['action'] == 'CORRECT']
            originals.append(dict(dataset=r['dataset'], phrase_id=r['phrase_id'], notes=len(notes_a),
                                  edited_notes=len(edits), edited=bool(edits), mean_abs_correction=float(np.mean([abs(n['correction_cents']) for n in edits])) if edits else 0.,
                                  edited_fraction_of_reliable=float((corr_a != 0).sum()/max(ma.sum(), 1))))
        if key_c in seen_c:
            continue
        seen_c.add(key_c)
        cc, mc, corr_c, notes_c = analyse(r['waveform_paths']['C'], d/'C_input.npz', policy)
        with np.load(r['target_path']) as z:
            P = z['common_reliable_mask'].astype(bool) & (z['correction_target_cents'] != 0)
            target = z['correction_target_cents']
        both = P & ma & mc
        edited_p = both & (corr_c != 0)
        residual_before = cc[both]-ca[both]
        residual_after = cc[both]+corr_c[both]-ca[both]
        detuned.append(dict(dataset=r['dataset'], phrase_id=r['phrase_id'], injected_cents=r['cents'], correction_frames=int(both.sum()),
                            shifted_region_edited=bool(edited_p.any()),
                            edit_direction_correct=(bool(np.all(np.sign(corr_c[edited_p]) == np.sign(target[edited_p]))) if edited_p.any() else None),
                            error_before_cents=float(np.mean(np.abs(residual_before))) if both.any() else None,
                            error_after_cents=float(np.mean(np.abs(residual_after))) if both.any() else None,
                            edits_outside_shifted_region=int(((corr_c != 0) & ~P & mc).sum())))
    summary = {}
    for ds in ('jamendo', 'vocadito'):
        o = [x for x in originals if x['dataset'] == ds]; c = [x for x in detuned if x['dataset'] == ds and x['correction_frames']]
        e = [x for x in c if x['shifted_region_edited']]
        summary[ds] = dict(
            originals=len(o), originals_with_any_edit=sum(x['edited'] for x in o),
            original_edit_rate=float(np.mean([x['edited'] for x in o])) if o else None,
            original_edited_notes=sum(x['edited_notes'] for x in o),
            detuned=len(c), detuned_shift_edited=len(e), detuned_edit_rate=float(len(e)/len(c)) if c else None,
            correct_direction=sum(bool(x['edit_direction_correct']) for x in e),
            mean_error_before_cents=float(np.mean([x['error_before_cents'] for x in c])) if c else None,
            mean_error_after_cents=float(np.mean([x['error_after_cents'] for x in c])) if c else None,
            mean_error_before_edited=float(np.mean([x['error_before_cents'] for x in e])) if e else None,
            mean_error_after_edited=float(np.mean([x['error_after_cents'] for x in e])) if e else None,
            by_injected={str(k): dict(n=len(v), edited=sum(x['shifted_region_edited'] for x in v),
                                      error_before=float(np.mean([x['error_before_cents'] for x in v])), error_after=float(np.mean([x['error_after_cents'] for x in v])))
                         for k, v in sorted(_group(c, 'injected_cents').items())})
    evidence = dict(declaration_sha256=sha(out/'declaration.json'), summary=summary, originals=originals, detuned=detuned)
    (out/'evidence.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    print('RULE_EVALUATION_RUN:', out); print(json.dumps(summary, indent=1))


def _group(rows, key):
    g = defaultdict(list)
    for r in rows:
        g[r[key]].append(r)
    return g


if __name__ == '__main__':
    run()
