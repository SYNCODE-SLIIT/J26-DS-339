"""Build a readable status report from measured files; never changes selection."""
import argparse
import html
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from bpdr_input.experiments.learning import context_improvement as ex
from bpdr_input.modeling.correction_accuracy import summarize
from bpdr_input.reporting.transformer_accuracy_report import complete_report


def report(root):
    root=Path(root).resolve();selection=ex.read(root/'locked_selection.json');arm=selection['arm'];rows=[]
    declaration=ex.read(root/'declaration.json');cv_root=Path(declaration.get('upstream_model_run',root))
    fold_files=sorted((cv_root/'cross_validation').glob(f'fold_*/{arm}/validation/accuracy.json'))
    for p in fold_files:
        for row in ex.read(p)['rows']:
            rows.append({k:v for k,v in row.items() if k!='trace' and not k.startswith(('permitted_','delivered_'))})
    result=summarize(rows);result.update(rows=rows,rendered_waveforms_evaluated=False)
    ex.write(root/'accuracy.json',result);ex.write_report(root,result,'Held-voice Transformer: grouped training-side evidence')
    complete_report(root)
    stages=[]
    for name,folder in [('Development','development'),('Reserved final','final_evaluation')]:
        target=root/folder
        if not (target/'evidence.json').exists():
            stages.append(dict(name=name,status='Not evaluated',folder=folder));continue
        evidence=ex.read(target/'evidence.json');measurement=ex.read(target/'accuracy.json')['summary'];gate=evidence['gates']
        controls=ex.read(target/'original_controls.json')
        stages.append(dict(name=name,status='Pass' if gate['passed'] else 'Fail',folder=folder,
            attempted=measurement['attempted_case_evaluations'],scoreable=measurement['scoreable_case_evaluations'],
            attempted_singers=len(gate['by_singer']),scoreable_singers=gate.get('scoreable_singers',sum(v['scoreable']>0 for v in gate['by_singer'].values())),
            raw_mae_cents=measurement['raw_mae_cents'],within_10c_percent=measurement['within_10c_percent'],
            before_cents=measurement['delivered_before_mae_cents'],after_cents=measurement['delivered_after_mae_cents'],
            improved=sum(x['improved'] for x in gate['by_singer'].values()),original_controls=len(controls),
            original_identity=sum(bool(x.get('delivered_identity')) for x in controls),per_singer=gate['by_singer']))
    active=ex.WORKSPACE/'results/transformer_usability/active_model.json'
    activated=active.exists() and Path(ex.read(active)['run_root'])==root
    evidence=dict(run=str(root),selected_arm=arm,cross_validation=result['summary'],cross_validation_prediction_gate=result['prediction_gate_passed'],
        cross_validation_source=str(cv_root),source_fold_receipts=[str(p) for p in fold_files],
        stages=stages,activated_for_scoped_preview=activated,research_validation_passed=False,independent_listening='not assessed')
    ex.write(root/'usability_summary.json',evidence)
    lines=['','## Actual audio and project availability','',f'Scoped preview activated: **{activated}**. Independent listening: **not assessed**.',
        '','| Stage | Scoped actual-WAV gate | Scored / attempted | Scoreable / attempted singers | Improved cases | Original identity | Delivered error, cents |','|---|---|---:|---:|---:|---:|---|']
    block='<h2>Actual audio and project availability</h2><p>Scoped preview activated: <strong>'+str(activated)+'</strong>. Independent listening and general natural-intention accuracy remain unestablished. Unscoreable singers remain unassessed, not validated; all attempts remain in cohort yield and every original must be preserved.</p><table><tr><th>Stage</th><th>Scoped gate</th><th>Scored / attempted</th><th>Scoreable / attempted singers</th><th>Verified improvements</th><th>Original identity</th><th>Delivered error</th></tr>'
    for stage in stages:
        if 'attempted' not in stage:
            lines.append(f'| {stage["name"]} | Not evaluated | - | - | - | - | - |')
            block+='<tr><td>'+stage['name']+'</td><td>Not evaluated</td></tr>';continue
        delivered=f'{stage["before_cents"]:.3f} -> {stage["after_cents"]:.3f}'
        count=f'{stage["scoreable"]} / {stage["attempted"]}'
        originals=f'{stage["original_identity"]} / {stage["original_controls"]}'
        singers=f'{stage["scoreable_singers"]} / {stage["attempted_singers"]}'
        lines.append(f'| [{stage["name"]}]({stage["folder"]}/index.html) | {stage["status"]} | {count} | {singers} | {stage["improved"]} | {originals} | {delivered} |')
        block+=f'<tr><td><a href="{stage["folder"]}/index.html">{stage["name"]}</a></td><td>{stage["status"]}</td><td>{count}</td><td>{singers}</td><td>{stage["improved"]}</td><td>{originals}</td><td>{delivered}</td></tr>'
    block+='</table><p>Every stage retains creation exclusions and withheld outputs. Technical file verification is different from reduced reference-controlled error. Final singer material is not accessed until locked development gates pass.</p>'
    block+='<p><a href="locked_selection.json">Model selection</a> | <a href="declaration.json">Prospective declaration</a> | <a href="usability_summary.json">Complete status and per-singer results</a></p>'
    page=root/'index.html';page.write_text(page.read_text(encoding='utf-8').replace('</html>',block+'</html>'),encoding='utf-8')
    with (root/'REPORT.md').open('a',encoding='utf-8') as stream:stream.write('\n'.join(lines)+'\n')
    print('USABILITY_REPORT',root/'index.html',flush=True)
    return evidence


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-root',type=Path,required=True)
    report(p.parse_args().run_root)
