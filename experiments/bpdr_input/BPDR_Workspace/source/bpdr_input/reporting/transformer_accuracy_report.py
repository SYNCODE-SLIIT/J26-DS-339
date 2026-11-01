"""Make denominators and prediction-vs-delivery evidence explicit in reports.

This adds reporting only; it does not recalculate model selection, change a
threshold or change any measured prediction in accuracy.json.
"""
from collections import Counter
import hashlib
import html
import json
from pathlib import Path


MARKER='<!-- ACCURACY_EVIDENCE_SUMMARY -->'


def evidence_counts(result):
    rows=result['rows']
    valid=[r for r in rows if 'excluded' not in r]
    rendered=bool(result.get('rendered_waveforms_evaluated'))
    attempted=set(); encoded=set(); failures=set(); reasons=Counter()
    for row in valid:
        for view,trace in row.get('trace',{}).items():
            identity=(view,trace['permission']['source_sha256'])
            if trace.get('render_attempted'):
                attempted.add(identity)
            transaction=trace.get('transaction',{})
            delivered=transaction.get('delivered_wav_audit',{})
            if delivered.get('passed'):
                encoded.add(delivered['sha256'])
            if transaction.get('outcome')=='ROLLED_BACK':
                failures.add(identity)
        reasons.update(row.get('trace',{}).get('C',{}).get('permission',{}).get('reasons',[]))
    improved=sum(bool(r.get('delivered_verified_improvement')) for r in valid) if rendered else None
    return dict(attempted_case_evaluations=len(rows),scoreable_case_evaluations=len(valid),
        unscoreable_case_evaluations=len(rows)-len(valid),
        render_pipeline_requested=rendered,unique_render_attempts=len(attempted),
        unique_encoded_verified_wavs=len(encoded),unique_render_rollbacks=len(failures),
        delivered_improved_case_evaluations=improved,
        verified_improvement_percent_all_attempts=(100*improved/len(rows) if rendered and rows else None),
        verified_improvement_percent_scoreable=(100*improved/len(valid) if rendered and valid else None),
        detuned_permission_reasons=dict(reasons),
        prediction_gate_passed=bool(result['prediction_gate_passed']),
        independent_listening='not assessed',general_singing_accuracy='not established',demo_activated=False)


def complete_report(folder):
    folder=Path(folder)
    path=folder/'accuracy.json'
    result=json.loads(path.read_text(encoding='utf-8'))
    summary=evidence_counts(result)
    # In CV, count repeated seed evaluations explicitly, alongside actual data.
    declaration_path=folder/'declaration.json'
    evaluation_path=folder/'evaluation_declaration.json'
    if declaration_path.exists():
        declaration=json.loads(declaration_path.read_text(encoding='utf-8'))
        manifest=Path(declaration['data_run'])/'train_pairs.json' if 'data_run' in declaration else folder/'train_pairs.json'
        summary['dataset_role']='singer-held-out cross-validation inside historical training data'
    elif evaluation_path.exists():
        declaration=json.loads(evaluation_path.read_text(encoding='utf-8'))
        manifest=Path(declaration['pairs_manifest'])
        summary['dataset_role']=declaration.get('dataset_role',declaration.get('role','specified independent-in-fit pairs; freshness must be established separately'))
    else:
        manifest=None
        summary['dataset_role']=('singer-held-out training-side fold' if folder.name=='validation' else
            'previously inspected calibration development' if folder.parent.name=='calibration' else
            'reporting role not supplied; inspect the run declaration')
    if manifest is not None:
        pairs=json.loads(manifest.read_text(encoding='utf-8'))
        summary['manifest_attempted_interventions']=len(pairs)
        summary['manifest_scoreable_interventions']=sum('excluded' not in p for p in pairs)
        summary['manifest_source_recordings']=len({p['source']['sha256'] for p in pairs})
    summary['accuracy_json_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    summary['reporting_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (folder/'verification_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    rows=''.join('<tr><td>'+html.escape(k)+'</td><td>'+html.escape(str(v))+'</td></tr>'
                 for k,v in summary.items() if not isinstance(v,dict) and not k.endswith('sha256'))
    block=(MARKER+'<h2>Evidence and actual delivery</h2><p>Repeated seed evaluations are not additional '
        'recordings. A prediction pass is not a verified audio repair. All-attempt improvement includes '
        'construction exclusions and withheld cases in its denominator.</p><table>'+rows+'</table>'
        '<p><a href="verification_summary.json">Verification/denominator summary</a></p>')
    index=folder/'index.html'
    if index.exists():
        page=index.read_text(encoding='utf-8')
        if MARKER in page:
            page=page.split(MARKER,1)[0]+'</html>'
        index.write_text(page.replace('</html>',block+'</html>'),encoding='utf-8')
    report=folder/'REPORT.md'
    if report.exists():
        text=report.read_text(encoding='utf-8').split(MARKER,1)[0]
        items=['',MARKER,'## Evidence and actual delivery','',
            '| Check | Result |','|---|---|']
        items.extend(f'| {k} | {v} |' for k,v in summary.items()
                     if not isinstance(v,dict) and not k.endswith('sha256'))
        report.write_text(text+'\n'.join(items)+'\n',encoding='utf-8')
    return summary
