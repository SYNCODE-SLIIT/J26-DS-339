"""Model-first improvement and repeatable, reference-controlled accuracy checks.

All objective selection uses grouped folds of the existing training partition.
Previously inspected calibration is reported only after endpoints are locked.
Nothing activates a model, changes permission limits, or receives references at
the callable predictor interface.
"""
import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import sys
import time
import uuid

import numpy as np
import soundfile as sf
import torch

from bpdr_input import COMPONENT_ROOT
from bpdr_input.experiments.learning import model_recovery as recovery
from bpdr_input.modeling.context_transformer import (ContextConfig, ContextTransformer,
    load_checkpoint, predict_features, save_checkpoint, window_at)
from bpdr_input.modeling.correction_accuracy import (ARMS, FOLDS, assert_independent,
    improvement_loss, prediction_row, summarize)
from bpdr_input.modeling.local_context_pilot import context_features
from bpdr_input.modeling.recovery_model import scope_mask
from bpdr_input.reporting.bpdr_showcase import correction_from_prediction
from bpdr_input.repair.repair_controller import plan_fixture, transact_fixture

WORKSPACE = COMPONENT_ROOT/'BPDR_Workspace'
ROOT = WORKSPACE/'results/transformer_improvement'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n',encoding='utf-8')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def group(row):
    return row.get('group',row['source']['singer_id'])


def prepared(rows):
    pairs = recovery.cached_pairs(rows)
    cache = {}
    for p in pairs:
        if 'excluded' in p:
            continue
        n = len(p['A']['features']['times_seconds'])
        if (p['target'].shape!=(n,) or p['mask'].shape!=(n,) or not np.isfinite(p['target']).all()
                or len(p['C']['features']['times_seconds'])!=n or not p['audit']['passed']):
            raise ValueError('Invalid or unaudited supervised pair')
        p['context'] = {}
        for view in ('A','C'):
            key = p['path_'+view]
            if key not in cache:
                cache[key] = [context_features(raw) for raw in p[view]['views']]
            p['context'][view] = cache[key]
    return pairs


def input_hashes(rows):
    paths = set()
    for p in rows:
        original = Path(p['source']['path'])
        if sha(original)!=p['source']['sha256']:
            raise ValueError('Original source recording changed')
        paths.add(original)
        if 'excluded' in p:
            continue
        folder = Path(p['path_A']).parent
        paths.update([Path(p['path_A']),Path(p['path_C']),Path(p['target_path']),
            folder/'original_features.npz',folder/f"features_{p['cents']:+g}.npz"])
    return {str(p):sha(p) for p in sorted(paths)}


def fit(pairs, arm, seed, updates, folder, declaration_hash):
    valid = [p for p in pairs if 'excluded' not in p]
    groups = sorted({p['group'] for p in valid})
    grouped = {g:[p for p in valid if p['group']==g] for g in groups}
    torch.manual_seed(seed)
    rng, original_rng = np.random.default_rng(seed), np.random.default_rng(seed+10000)
    model = ContextTransformer(ContextConfig(window=81))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0005, weight_decay=.001)
    continuity = arm in ('continuity','combined')
    preservation = arm in ('preservation','combined')
    offsets = (-1,0,1) if continuity else (0,)
    center_index = len(offsets)//2
    original_sources = list({p['path_A']:p for p in valid}.values())
    original_groups = {g:[p for p in original_sources if p['group']==g] for g in groups}
    history = []
    started = time.perf_counter()
    model.train()
    for step in range(updates):
        windows, padding, targets, neighbour_targets, neighbour_valid = [], [], [], [], []
        original_windows, original_padding = [], []
        for _ in range(8):
            g = groups[int(rng.integers(len(groups)))]
            p = grouped[g][int(rng.integers(len(grouped[g])))]
            plateau = p['mask'] & (p['target'] != 0)
            outside = p['mask'] & (p['target'] == 0)
            indices = np.flatnonzero(plateau if rng.random()<.7 or not outside.any() else outside)
            center = int(rng.choice(indices))
            alternate = int(rng.integers(1,3))
            targets.append(float(p['target'][center])/30.)
            nt, nv = [], []
            for offset in offsets:
                actual = int(np.clip(center+offset,0,len(p['mask'])-1))
                nt.append(float(p['target'][actual])/30.)
                nv.append(bool(actual==center+offset and p['mask'][actual]))
                for view,boundary in (('A',0),('A',alternate),('C',0),('C',alternate)):
                    x,pad,_ = window_at(p['context'][view][boundary],actual,81)
                    windows.append(x); padding.append(pad)
            neighbour_targets.append(nt); neighbour_valid.append(nv)
            if preservation:
                # Separate RNG keeps paired examples/schedule matched across arms.
                op = original_groups[g][int(original_rng.integers(len(original_groups[g])))]
                oc = int(original_rng.choice(np.flatnonzero(op['A']['reliable'])))
                for boundary in (0,alternate):
                    x,pad,_ = window_at(op['context']['A'][boundary],oc,81)
                    original_windows.append(x); original_padding.append(pad)
        x = torch.from_numpy(np.stack(windows))
        pad = torch.from_numpy(np.stack(padding))
        pred = (model(x,pad)[:,40]/30.).reshape(8,len(offsets),4)
        original_pred = None
        if preservation:
            original_pred = model(torch.from_numpy(np.stack(original_windows)),
                                  torch.from_numpy(np.stack(original_padding)))[:,40]/30.
        loss,parts = improvement_loss(pred[:,center_index],torch.tensor(targets),arm=arm,
            response_active=step>=updates//4, neighbour_predictions=pred if continuity else None,
            neighbour_targets=torch.tensor(neighbour_targets) if continuity else None,
            neighbour_valid=torch.tensor(neighbour_valid) if continuity else None,
            original_predictions=original_pred)
        if not torch.isfinite(loss):
            raise ValueError('Non-finite objective')
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        if step%50==0 or step==updates-1:
            history.append(dict(update=step+1,loss=float(loss.detach()),
                                **{k:float(v.detach()) for k,v in parts.items()}))
    folder.mkdir(parents=True,exist_ok=True)
    metadata = dict(arm=arm,condition='bpdr_engineering_squared_response',seed=seed,
        updates=updates,training_groups=groups,
        training_sources=sorted({p['source']['sha256'] for p in valid}),
        declaration_sha256=declaration_hash,eligible_for_demo=False,research_validation_passed=False)
    save_checkpoint(folder/'checkpoint.pt',model.eval(),metadata)
    restored,payload = load_checkpoint(folder/'checkpoint.pt')
    probe = valid[0]['context']['A'][0][:7]
    np.testing.assert_allclose(predict_features(model,probe),predict_features(restored,probe),atol=1e-6)
    receipt = dict(**metadata,checkpoint_sha256=sha(folder/'checkpoint.pt'),
        model=model.describe(),fit_seconds=time.perf_counter()-started,reload_verified=True,
        history=history)
    write(folder/'fit_receipt.json',receipt)
    return restored,payload,receipt


def predictions(model,pair,view):
    return np.stack([predict_features(model,x,batch_size=128) for x in pair['context'][view]])


def legacy_proposal(pair,view,pred):
    analysis = pair[view]
    allowed,metadata = scope_mask(analysis['views'][0])
    operating = pred.copy(); operating[:,~allowed] = 0.
    reliable = analysis['views'][0][:,66]>.5
    disagreement = np.max(np.abs(operating[1:]-operating[0]),axis=0)
    curve = correction_from_prediction(operating[0],operating[1],reliable,threshold=3.)
    plan = plan_fixture(pair['audio_'+view],22050,analysis['features']['times_seconds'],
        curve,operating[1],disagreement,reliable,analysis['protected'])
    return curve,plan,allowed


def rendered_output(pair,view,curve,plan,folder):
    """No clean reference enters permission/render/technical audit callbacks."""
    from bpdr_input.audio.analysis import extract
    from bpdr_input.repair.framewise_renderer import render_candidate
    from bpdr_input.repair.framewise_audit import audit_curve
    from bpdr_input.repair.rule_correction import fine_hz
    from bpdr_input.repair.learned_runtime import audit_delivered_wav
    audio = pair['audio_'+view]
    analysis = pair[view]
    times = analysis['features']['times_seconds']
    def renderer(a,sr,intervals):
        return render_candidate(a,sr,times,curve,intervals)[0]
    def auditor(a,b,sr,intervals):
        after = extract(b,sr,COMPONENT_ROOT/'checkpoints')[0]
        return audit_curve(analysis['features'],after,fine_hz(a,sr,times),fine_hz(b,sr,times),
                           analysis['reliable'],intervals,curve)
    output,transaction = transact_fixture(audio,22050,plan,renderer,auditor)
    if transaction['outcome']=='VERIFIED_FIXTURE_CANDIDATE':
        folder.mkdir(parents=True,exist_ok=True)
        path = folder/(view+'_candidate.wav')
        sf.write(path,output,22050,subtype='PCM_24')
        try:
            transaction['delivered_wav_audit'] = audit_delivered_wav(path,audio,22050,plan['intervals_seconds'],auditor)
            output,_ = sf.read(path,dtype='float32')
        except Exception as exc:
            transaction.update(outcome='ROLLED_BACK',rollback=True,delivered_error=str(exc))
            output = audio.copy()
    return output,transaction


def evaluate(model,payload,pairs,folder,*,render=False,seed=0):
    folder.mkdir(parents=True,exist_ok=True)
    cache,render_cache,arrays,rows = {},{},{},[]
    for index,p in enumerate(pairs):
        if 'excluded' in p:
            rows.append(dict(group=group(p),seed=seed,cents=float(p['cents']),
                             source_sha256=p['source']['sha256'],excluded=p['excluded']))
            continue
        pp = {}
        for view in ('A','C'):
            key = p['path_'+view]
            if key not in cache:
                cache[key] = predictions(model,p,view)
            pp[view] = cache[key]
            arrays[f'{index}_{view}'] = pp[view]
        row = prediction_row(p,pp['A'],pp['C'])
        row['seed'] = seed
        row['trace'] = {}
        curves = {}
        for view in ('A','C'):
            curve,plan,allowed = legacy_proposal(p,view,pp[view])
            curves[view] = curve if plan['permitted'] else np.zeros(len(curve))
            row['trace'][view] = dict(scope_eligible_frames=int(allowed.sum()),
                scope_eligible_scored_frames=int((allowed&p['mask']&(p['target']!=0)).sum()),
                permission=plan,render_attempted=False,
                outcome='IDENTITY' if plan['identity'] else ('PERMITTED_NOT_RENDERED' if plan['permitted'] else 'WITHHELD'))
            arrays[f'{index}_requested_{view}'] = curve
            arrays[f'{index}_permitted_{view}'] = curves[view]
            if render:
                key = p['path_'+view]
                if key not in render_cache:
                    render_cache[key] = rendered_output(p,view,curve,plan,folder/'audio'/f'case_{index:02d}')
                output,transaction = render_cache[key]
                row['trace'][view].update(render_attempted=bool(plan['permitted'] and not plan['identity']),
                    outcome=transaction['outcome'],transaction=transaction)
                from bpdr_input.repair.rule_correction import fine_pitch
                if np.array_equal(output,p['audio_'+view]):
                    output_fine = p[view]['fine']
                else:
                    output_fine = fine_pitch(output,22050,p[view]['features']['times_seconds'])
                scored = p['mask']&(p['target']!=0)
                if view=='C':
                    before_error = np.abs(p['C']['fine'][scored]-p['A']['fine'][scored])
                    after_error = np.abs(output_fine[scored]-p['A']['fine'][scored])
                    if not np.isfinite(before_error).all() or not np.isfinite(after_error).all():
                        raise ValueError('Invalid delivered-waveform reference score')
                    row.update(delivered_before_mae_cents=float(before_error.mean()),
                        delivered_after_mae_cents=float(after_error.mean()),
                        delivered_verified_improvement=bool(transaction['outcome']=='VERIFIED_FIXTURE_CANDIDATE'
                            and after_error.mean()<before_error.mean()-1e-6))
                else:
                    reliable = p['A']['reliable']
                    row['delivered_original_abs_cents'] = float(np.abs(output_fine[reliable]-p['A']['fine'][reliable]).mean())
        plateau = p['mask']&(p['target']!=0)
        row.update(permitted_mae_cents=float(np.abs(curves['C'][plateau]-p['target'][plateau]).mean()),
            permitted_original_abs_cents=float(np.abs(curves['A'][p['A']['reliable']]).mean()),
            permitted_nonidentity=bool(np.any(curves['C'])))
        rows.append(row)
    result = summarize(rows)
    result.update(rows=rows,rendered_waveforms_evaluated=render,
        metric_scope='Controlled correction on previously declared public-data sources; not natural intended-note accuracy',
        eligibility_policy='Unmodified historical long-held scope, largest >=3c span and fixture controller',
        reference_not_supplied_to_predictor=True,research_validation_passed=False,demo_activated=False)
    np.savez_compressed(folder/'predictions.npz',**arrays)
    result['predictions_sha256'] = sha(folder/'predictions.npz')
    write(folder/'accuracy.json',result)
    write_report(folder,result,'Transformer accuracy check')
    return result


def write_report(folder,result,title):
    s = result['summary']
    lines = [f'# {title}', '', '**Development evidence; no model activation.**', '',
        'Accuracy within a cents tolerance refers to scored frames, averaged by singer. It is not a percentage of correct singing.', '',
        '| Metric | Value |','|---|---:|']
    for name in ('raw_mae_cents','raw_rmse_cents','within_5c_percent','within_10c_percent',
                 'original_abs_cents','outside_abs_cents','response_gain','permitted_mae_cents',
                 'delivered_before_mae_cents','delivered_after_mae_cents','singers','original_recordings',
                 'attempted_case_evaluations','scoreable_case_evaluations','excluded_case_evaluations'):
        if name in s and s[name] is not None:
            value = f'{s[name]:.3f}' if isinstance(s[name],float) else str(s[name])
            lines.append(f'| {name} | {value} |')
    lines += ['',f"Prediction preservation/response gate: **{result['prediction_gate_passed']}**.",
        f"Actual encoded-WAV evaluation performed: **{result.get('rendered_waveforms_evaluated',False)}**.",
        'A prediction gate is not permission, delivered-audio approval, listener approval or research adoption.',
        'Read accuracy.json for per-singer/direction results, every exclusion and every processing trace.',
        'Small-cohort singer bootstrap intervals are descriptive development uncertainty; seeds are reported separately from singers.']
    (folder/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    metrics = ''.join(f'<tr><td>{html.escape(k)}</td><td>{html.escape(str(v))}</td></tr>' for k,v in s.items())
    prediction_link = '<a href="predictions.npz">Saved predictions</a>' if (folder/'predictions.npz').exists() else '<a href="cross_validation.json">Cross-validation details</a>'
    body = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>'+html.escape(title)+
        '</title><style>body{font:16px system-ui;max-width:1050px;margin:48px auto;padding:0 24px;color:#183036}'
        'table{border-collapse:collapse;width:100%}td,th{padding:9px;border-bottom:1px solid #d9e2e5;text-align:left}'
        'h1{font-size:32px}p{line-height:1.6}.note{padding:16px;background:#eef4f4}</style><h1>'+html.escape(title)+
        '</h1><p class="note">Controlled, development-only correction accuracy. Tolerance percentages are scored-frame metrics, '
        'not general singing correctness. Training and evaluation singers must be disjoint.</p><table>'+metrics+
        '</table><p>Prediction gate: <strong>'+str(result['prediction_gate_passed'])+'</strong>. WAV evaluation: '+
        str(result.get('rendered_waveforms_evaluated',False))+'. No demo activation or independent human approval.</p>'+
        '<p><a href="accuracy.json">Full evidence</a> · '+prediction_link+' · '
        '<a href="cases.csv">Per-case measurements</a></p></html>')
    (folder/'index.html').write_text(body,encoding='utf-8')
    fields = sorted({k for r in result.get('rows',[]) for k in r})
    with (folder/'cases.csv').open('w',newline='',encoding='utf-8') as stream:
        writer = csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for row in result.get('rows',[]):
            writer.writerow({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in row.items()})


def protected_files():
    paths = [WORKSPACE/'selected_model/bpdr.pt',WORKSPACE/'selected_model/model.json',
        WORKSPACE/'results/notebook_training/current.json',
        WORKSPACE/'exports/selected_run_inference/selected_model.joblib',
        WORKSPACE/'exports/selected_run_inference/manifest.json',
        WORKSPACE/'notebooks/04_model_training_and_evaluation.ipynb']
    return {str(p):sha(p) for p in paths}


def run_training(updates=300):
    if updates<1:
        raise ValueError('Updates must be positive')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    run = ROOT/uuid.uuid4().hex
    run.mkdir(parents=True)
    data = Path(read(WORKSPACE/'results/model_recovery/current.json')['path'])
    train_rows = read(data/'train_pairs.json')
    calibration_manifest_rows = read(data/'calibration_pairs.json')
    groups = {group(p) for p in train_rows if 'excluded' not in p}
    if groups != {g for fold in FOLDS for g in fold}:
        raise ValueError('Training singer manifest changed; redeclare folds')
    protected = protected_files()
    sources = [Path(__file__),WORKSPACE/'source/bpdr_input/modeling/correction_accuracy.py',
        WORKSPACE/'source/bpdr_input/modeling/context_transformer.py',
        WORKSPACE/'source/bpdr_input/modeling/local_context_pilot.py']
    declaration = dict(version='model-first-grouped-improvement-1',created_at=datetime.now(timezone.utc).isoformat(),
        data_run=str(data),arms=list(ARMS),seeds=[339,340,341],folds=[list(x) for x in FOLDS],
        updates=updates,config=asdict(ContextConfig(window=81)),
        optimizer=dict(name='AdamW',lr=.0005,weight_decay=.001,gradient_clip=1.),
        objective='Existing squared engineering anchor/response; preservation x2 plus broader original anchors; derivative SmoothL1 .1 on labelled neighbours; over-30c squared penalty .1',
        budgets='Same update count and paired sampling; continuity uses adjacent windows and preservation adds original-only supervision; compute/label counts differ and are reported',
        selection='Among arms passing all singer prediction preservation/response checks, lowest singer-balanced raw MAE; otherwise lowest .35 repair + .50 original + .15 outside MAE, diagnostic only; tie uses declared arm order',
        evaluation='Four grouped training-side folds and three seeds; calibration access only after selected arm and all full-fit seed checkpoints are locked',
        default_checkpoint='Seed 339 chosen in advance, never best calibration seed',
        independence='All variants of fitted singers/sources excluded from that fold validation; old calibration is previously inspected development, not final test',
        runtime='Unmodified legacy permission/rendering; no reference/target/support in predictor',
        limitations=['Small seven-singer development cohort','No new natural intended-note labels or fresh final singers',
                     'Does not prove benefit of BPDR response terms versus matched neural controls'],
        input_hashes={**{str(data/n):sha(data/n) for n in ('train_pairs.json','calibration_pairs.json','declaration.json')},
                      **input_hashes(train_rows+calibration_manifest_rows)},
        source_hashes={str(p):sha(p) for p in sources},protected_files=protected)
    write(run/'declaration.json',declaration)
    declaration_hash = sha(run/'declaration.json')
    print('RUN',run,flush=True)
    pairs = prepared(train_rows)
    arm_rows = {arm:[] for arm in ARMS}
    fits = []
    for fold_index,fold in enumerate(FOLDS):
        train = [p for p in pairs if group(p) not in fold]
        validation = [p for p in pairs if group(p) in fold]
        assert_independent({group(p) for p in train},{p['source']['sha256'] for p in train},validation)
        for seed in declaration['seeds']:
            for arm in ARMS:
                folder = run/'cross_validation'/f'fold_{fold_index+1}'/f'seed_{seed}'/arm
                started = time.perf_counter()
                model,payload,receipt = fit(train,arm,seed,updates,folder,declaration_hash)
                result = evaluate(model,payload,validation,folder/'validation',seed=seed)
                arm_rows[arm].extend(result['rows'])
                fits.append(dict(fold=fold_index+1,arm=arm,seed=seed,fit=receipt,
                                 validation_summary=result['summary']))
                print('CV',fold_index+1,seed,arm,json.dumps(dict(mae=result['summary']['raw_mae_cents'],
                    original=result['summary']['original_abs_cents'],within10=result['summary']['within_10c_percent'],
                    seconds=time.perf_counter()-started)),flush=True)
                write(run/'progress.json',dict(completed_fits=len(fits),planned_fits=len(FOLDS)*3*len(ARMS),last=fits[-1]))
    # Keep fully unscorable training singers in every arm's attempt denominator.
    uncovered = [p for p in train_rows if group(p) not in {g for f in FOLDS for g in f}]
    for arm in ARMS:
        for seed in declaration['seeds']:
            arm_rows[arm].extend(dict(group=group(p),seed=seed,cents=p['cents'],
                source_sha256=p['source']['sha256'],excluded=p['excluded']) for p in uncovered)
    summaries = {arm:summarize(rows) for arm,rows in arm_rows.items()}
    qualified = [a for a in ARMS if summaries[a]['prediction_gate_passed']]
    selected = min(qualified or list(ARMS),key=lambda a:(summaries[a]['summary']['raw_mae_cents']
        if qualified else summaries[a]['summary']['balanced_error_cents'],list(ARMS).index(a)))
    selection = dict(arm=selected,prediction_gate_passed=bool(qualified),
        reason='All-singer prediction gate then correction error' if qualified else 'No arm passed every singer gate; diagnostic balanced-error selection only',
        development_results=summaries,calibration_used_for_selection=False)
    write(run/'cross_validation.json',dict(**selection,rows=arm_rows,fits=fits))
    final_models = {}
    for seed in declaration['seeds']:
        final_models[seed] = fit(pairs,selected,seed,updates,run/'final'/f'seed_{seed}',declaration_hash)
    write(run/'locked_selection.json',dict(**selection,default_seed=339,
        checkpoints={str(seed):record[2]['checkpoint_sha256'] for seed,record in final_models.items()}))
    print('LOCKED',selected,'gate',bool(qualified),flush=True)
    calibration_rows = read(data/'calibration_pairs.json')
    for model,payload,receipt in final_models.values():
        assert_independent(payload['training_groups'],payload['training_sources'],calibration_rows)
    calibration = prepared(calibration_rows)
    calibration_results = {}
    for seed,(model,payload,receipt) in final_models.items():
        calibration_results[str(seed)] = evaluate(model,payload,calibration,
            run/'calibration'/f'seed_{seed}',render=seed==339,seed=seed)
        print('CALIBRATION',seed,json.dumps(calibration_results[str(seed)]['summary']),flush=True)
    assert all(sha(p)==v for p,v in protected.items())
    assert all(sha(p)==v for p,v in declaration['input_hashes'].items())
    assert all(sha(p)==v for p,v in declaration['source_hashes'].items())
    evidence = dict(declaration_sha256=declaration_hash,selected_arm=selected,
        cross_validation={a:dict(summary=v['summary'],by_singer=v['by_singer'],
                                prediction_gate_passed=v['prediction_gate_passed']) for a,v in summaries.items()},
        calibration={s:dict(summary=v['summary'],by_singer=v['by_singer'],prediction_gate_passed=v['prediction_gate_passed'])
                     for s,v in calibration_results.items()},
        default_checkpoint=str(run/'final/seed_339/checkpoint.pt'),
        default_checkpoint_sha256=sha(run/'final/seed_339/checkpoint.pt'),
        full_declared_matrix_completed=True,protected_artifacts_unchanged=True,
        previously_inspected_development_only=True,fresh_final_test_accessed=False,
        demo_activated=False,research_validation_passed=False)
    write(run/'evidence.json',evidence)
    root_result = dict(**summaries[selected],rows=arm_rows[selected],rendered_waveforms_evaluated=False)
    write(run/'accuracy.json',root_result)
    write_report(run,root_result,'Transformer improvement: singer-held-out development')
    # The root report compares every declared arm, including failures.
    table = ['','## All declared objectives','','| Arm | MAE | Within 10c % | Original change | Outside change | Prediction gate |',
             '|---|---:|---:|---:|---:|---|']
    for arm,res in summaries.items():
        s=res['summary']; table.append(f"| {arm} | {s['raw_mae_cents']:.3f} | {s['within_10c_percent']:.1f} | {s['original_abs_cents']:.3f} | {s['outside_abs_cents']:.3f} | {res['prediction_gate_passed']} |")
    with (run/'REPORT.md').open('a',encoding='utf-8') as stream:
        stream.write('\n'.join(table)+'\n\nSelected from training-side folds: '+selected+'. Calibration is reported separately, never used for selection.\n')
    try:
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        fig=Figure(figsize=(11,4)); FigureCanvasAgg(fig); axes=fig.subplots(1,2)
        for ax,key,title in zip(axes,('raw_mae_cents','original_abs_cents'),('Held-out correction MAE (cents)','Original correction request (cents)')):
            ax.bar(list(ARMS),[summaries[a]['summary'][key] for a in ARMS],color='#327f86')
            ax.set_title(title); ax.tick_params(axis='x',rotation=25); ax.set_ylim(bottom=0)
        axes[1].axhline(3,color='#a24d35',linestyle='--',label='3c preservation criterion'); axes[1].legend()
        fig.tight_layout(); fig.savefig(run/'comparison.png',dpi=150)
    except ImportError:
        pass
    comparison = '<h2>All declared objectives</h2><table><tr><th>Arm</th><th>MAE (cents)</th><th>Within 10c</th><th>Original request (cents)</th><th>Prediction gate</th></tr>'
    for arm,res in summaries.items():
        s=res['summary']
        comparison+=f"<tr><td>{arm}</td><td>{s['raw_mae_cents']:.3f}</td><td>{s['within_10c_percent']:.1f}%</td><td>{s['original_abs_cents']:.3f}</td><td>{res['prediction_gate_passed']}</td></tr>"
    comparison+='</table><p>Every row uses singer-held-out development folds and three seeds. Calibration was not used for selection.</p>'
    if (run/'comparison.png').exists():
        comparison+='<img src="comparison.png" alt="Correction error and original-preservation comparison" style="width:100%">'
    comparison+='<p>Locked calibration checks: '+ ' · '.join(f'<a href="calibration/seed_{seed}/index.html">Seed {seed}</a>' for seed in declaration['seeds'])+'</p>'
    page=run/'index.html'; page.write_text(page.read_text(encoding='utf-8').replace('</html>',comparison+'</html>'),encoding='utf-8')
    write(ROOT/'latest_experiment.json',dict(path=str(run),evidence_sha256=sha(run/'evidence.json')))
    print('COMPLETE',run,flush=True)
    return run


def run_evaluation(checkpoint,manifest,output_root=None,render=False):
    torch.set_num_threads(2)
    model,payload = load_checkpoint(checkpoint)
    if not payload.get('training_groups') or not payload.get('training_sources'):
        raise ValueError('Checkpoint lacks training lineage; use an improvement-run checkpoint for independent evaluation')
    rows = read(manifest)
    assert_independent(payload['training_groups'],payload['training_sources'],rows)
    hashes = input_hashes(rows)
    folder = Path(output_root or ROOT/'accuracy_checks')/uuid.uuid4().hex
    folder.mkdir(parents=True)
    before = sha(checkpoint)
    write(folder/'evaluation_declaration.json',dict(checkpoint=str(Path(checkpoint).resolve()),
        checkpoint_sha256=before,pairs_manifest=str(Path(manifest).resolve()),pairs_manifest_sha256=sha(manifest),
        training_groups=payload['training_groups'],render=render,
        purpose='Reference-controlled accuracy check; caller must document whether singers/material were previously inspected',
        input_hashes=hashes,source_sha256=sha(__file__)))
    result = evaluate(model,payload,prepared(rows),folder,render=render,seed=payload['seed'])
    assert before==sha(checkpoint)
    assert all(sha(p)==v for p,v in hashes.items())
    print(json.dumps(result['summary'],indent=2)); print('ACCURACY_REPORT',folder,flush=True)
    return folder


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    train=commands.add_parser('train'); train.add_argument('--updates',type=int,default=300)
    check=commands.add_parser('evaluate'); check.add_argument('--checkpoint',type=Path,required=True)
    check.add_argument('--pairs-manifest',type=Path,required=True); check.add_argument('--output-root',type=Path)
    check.add_argument('--render',action='store_true')
    args=parser.parse_args(argv)
    if args.command=='train':
        run_training(args.updates)
    else:
        run_evaluation(args.checkpoint,args.pairs_manifest,args.output_root,args.render)


if __name__=='__main__':
    main()
