"""Export verified notebook results with neutral navigation; preserve all outputs."""
from pathlib import Path
import hashlib
import json
import os
import sys

import nbformat
from nbconvert import HTMLExporter

ROOT = next(p for p in Path(__file__).resolve().parents if p.name == 'bpdr_input' and (p/'pyproject.toml').is_file())
sys.path.insert(0, str(ROOT / 'BPDR_Workspace/source'))
from bpdr_input.saved_results import pitch_example, noise_example


def sha(p):
    with p.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(p, text):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding='utf-8')


def main():
    final = ROOT / 'BPDR_Workspace/results'
    current = json.loads((ROOT / 'runs/notebooks/latest.json').read_text())
    run = Path(current['path'])
    verification = json.loads((run / 'verification.json').read_text())
    provenance = json.loads((run/'provenance.json').read_text())
    assert sha(run/'verification.json') == provenance['verification_sha256']
    assert sha(ROOT/'notebooks/evidence_index.json') == provenance['evidence_index_sha256']
    declared = json.loads((ROOT/'notebooks/evidence_index.json').read_text())['maintained_notebooks']
    assert [row['notebook'] for row in verification] == declared
    assert all(r['errors'] == 0 for r in verification)
    artifacts = json.loads((final / 'CURRENT_ARTIFACTS.json').read_text())
    artifacts['organized_at'] = '2026-10-18'
    artifacts['purpose'] = 'Neutral current dataset, method, input/output and evaluation walkthrough'
    artifacts['notebooks'] = []
    for row in verification:
        assert sha(ROOT / 'notebooks' / row['notebook']) == row['source_sha256']
        assert sha(Path(row['executed_copy'])) == row['executed_sha256']
        notebook = nbformat.read(row['executed_copy'], as_version=4)
        outputs = [o for c in notebook.cells for o in c.get('outputs', [])]
        assert not any(o.output_type == 'error' for o in outputs)
        # Input code is available in the maintained source and full execution.
        # All markdown and result outputs are retained in this reading view.
        rendered, _ = HTMLExporter(exclude_input=True, exclude_input_prompt=True,
            exclude_output_prompt=True).from_notebook_node(notebook)
        for name in declared:
            rendered = rendered.replace('href="'+name+'"', 'href="'+Path(name).stem+'.html"')
        target = final / 'reports' / (Path(row['notebook']).stem + '.html')
        write(target, rendered)
        artifacts['notebooks'].append(dict(source='BPDR_Workspace/notebooks/' + row['notebook'],
            source_sha256=row['source_sha256'], executed_html=target.relative_to(ROOT).as_posix(),
            html_sha256=sha(target), full_executed_notebook=row['executed_copy'],
            full_executed_html=row['html_export'], outputs_preserved=len(outputs)))
    pitch, noise = pitch_example(), noise_example()
    def serialize(result):
        return {k: v.relative_to(ROOT).as_posix() if isinstance(v, Path) else v for k, v in result.items()}
    artifacts['examples'] = [serialize(pitch), serialize(noise)]
    artifacts['source_execution'] = str(run)
    artifacts['full_output_view'] = True
    artifacts['code_inputs_available_in_sources'] = True
    notebook_repair = ROOT/'runs/notebook_repair/current.json'
    if notebook_repair.exists():
        artifacts['notebook_maintenance_record'] = str(Path(json.loads(notebook_repair.read_text())['path']).relative_to(ROOT))
    audit = Path(json.loads((ROOT / 'runs/finalization/current.json').read_text())['path'])
    artifacts['finalization_record'] = audit.relative_to(ROOT).as_posix()
    artifacts.setdefault('previous_organization_recovery', artifacts['recovery'])
    artifacts['recovery'] = (audit / 'before_changes.zip').relative_to(ROOT).as_posix()
    catalog_path = final / 'CODE_CATALOG.json'
    catalog = json.loads(catalog_path.read_text(encoding='utf-8'))
    for entry in catalog:
        entry['sha256'] = sha(ROOT / entry['path'])
    catalog_path.write_text(json.dumps(catalog, indent=2) + '\n', encoding='utf-8')
    (final / 'CURRENT_ARTIFACTS.json').write_text(json.dumps(artifacts, indent=2) + '\n', encoding='utf-8')
    write(final / 'README.md', '''# Current implementation and results

Open **[index.html](index.html)**. Start the local system with `Run-System.ps1` in this folder, then use http://127.0.0.1:8501.

| Item | Purpose |
| --- | --- |
| [Notebook 1](reports/01_input_validation_and_analysis.html) | Original Vocadito coverage, data inspection, acoustic features, technical checks and unchanged acceptance |
| [Notebook 2](reports/02_data_review_and_paired_experiments.html) | Dataset roles, preparation, four-view Transformer, real-data adaptation, input/output and all measured results |
| [Notebook 3: Transformer training](reports/03_transformer_training.html) | Actual partitions, four-view features, checkpoint forward pass, losses, optimizer schedules and all-seed learning curves |
| [Notebook 4: Transformer evaluation](reports/04_transformer_evaluation.html) | Recomputed prediction metrics, matched controls, real-source gates, uncertainty and playable comparisons |
| [Implementation map](CODE_MAP.md) | Current application, inference, model, training, data helpers and retained experiments |
| [Source application](../source/bpdr_input/application/app.py) | Recording/upload interface and verified recorded comparisons |
| [Result reader](../source/bpdr_input/application/saved_results.py) | Checks saved input/output identity and computes paired noise improvement |
| [Validation guide](VALIDATION.md) | Metric meanings, implemented requirements, best practices and remaining validation |
| [Current artifacts](CURRENT_ARTIFACTS.json) | Pinned files, checksums and example provenance |

## Dataset and example identity

Vocadito is the proposed original dataset; Jamendo contributes separated vocals and predicted metadata. The selected Transformer uses synthetic pretraining followed by Vocadito/Jamendo adaptation with synthetic replay. HumTrans is an additional real-humming source; its later learning candidate was not selected.

The local pitch-error comparison is a generated phrase with a known injected shift. The noise comparison uses real HumTrans audio mixed with recorded rain. These are useful checked examples, not general real-singer accuracy. The notebook retains the full evaluation, unsuccessful seeds/candidates and relevant validation limits.

## Reading views and full record

The report views omit executable cell inputs for readability and retain every markdown cell and result output. Four maintained notebook sources remain in [notebooks/](../notebooks/). For a focused walkthrough use 01 → 03 → 04; 02 keeps the detailed study record. Full executed copies, code and historical outputs remain under `runs/notebooks/`; no result or log was rewritten to improve the reported performance.

Original accepted downstream audio remains unchanged. Verified pitch/noise candidates are separate playable/downloadable previews. No reference audio is supplied to live inference. Input/output must be compared at the same playback volume.

## Use and maintain

Run `../Run-Notebooks.ps1` to edit sources using the installed local Python. `-Check` checks the environment; `-Execute` runs all four notebooks. `Open-Notebooks.ps1` opens saved reading views. After a deliberate source change, execute the declared notebook suite and then run `python -P -B BPDR_Workspace/results/build_views.py` in the existing environment. This refreshes views and their checksum manifest without fitting models or changing research decisions.

Complete notes and proposal: [docs/](../documentation/README.md). Experiments/logs: [runs/](../../research_archive/evidence/README.md). Historical module names and compatibility launchers are preserved for recovery and reproducibility; they are not the current entry point.
''')
    write(final / 'VALIDATION.md', f'''# Method, output and validation

## Problem and research objective

Short solo singing/humming contains uncertain note boundaries and locally changing pitch. BPDR asks whether a correction model can respond to a local pitch perturbation while remaining insensitive to boundary metadata changes, preserving the rest of the voice. This is distinct from forcing every note to a musical grid or inferring all intended notes from unrestricted humming.

## Executable workflow

1. Inspect original Vocadito metadata, audio/annotation presence and recording durations. Keep source hashes and distinguish performed references from ideal musical intention.
2. Use artist/singer-group partitions before cropping/augmentation. Jamendo annotations are model estimates; artist IDs are proxies.
3. Construct A/B/C/D views: original or locally detuned audio crossed with nominal or altered boundaries.
4. Apply a four-layer Transformer to 70 acoustic features plus boundary metadata. Compare BPDR with augmentation-only and frame-only conditions on matched schedules.
5. Validate live audio, derive features/boundaries, infer correction, apply permission/render/remeasurement/rollback checks. Optional noise cleaning requires a declared noise-only interval and re-analysis.
6. Compare input with a verified separate candidate. Retain the original accepted package and report unsupported/withheld outcomes.

## Metric meanings and recorded examples

| Measure | Interpretation | Current example |
| --- | --- | --- |
| Local pitch error in cents | Difference from a known original on the injected local region; not a whole-user accuracy percentage | Generated-reference input {abs(pitch['before_cents']):.2f} -> output {abs(pitch['after_cents']):.2f} cents |
| Paired response gain | How much the prediction changes opposite to the known added detuning; ideal 1, constant-zero 0 | Full condition/seed/dataset tables in notebook 2 |
| Original/outside alteration | Preservation of unshifted regions; lower is preferable | Reported alongside correction results, not dropped when a fit improves |
| Added-noise SNR gain | Reduction of error against the recording before known added noise | Real-HumTrans/rain saved runtime output {noise['snr_gain_db']:.2f} dB |
| Noise-only prefix attenuation | Change within the declared noise interval | {noise['reference_attenuation_db']:.2f} dB |
| Measured pitch change during cleaning | Numerical preservation diagnostic; not intended-note or listening approval | Median {noise['median_pitch_change_cents']:.2f} cents |

No universal classification-style model-accuracy percentage has been established. A one-example error reduction is not a held-out participant success rate. These numbers preserve the source identities: generated pitch-control example and a constructed mixture of real recordings.

## Design and best practices

Source retention; accepted-only handoff; explicit technical states; fixed engineering limits; masks for unreliable/unvoiced frames; training-only normalization; grouped partitions; independent input/reference roles; verified checkpoints; feature re-analysis; rollback; reproducible logs; retained failed comparisons; candidate checksums before playback; and equal-gain notebook playback.

## User requirements and achievable benefits

Record/upload a short solo voice without knowing musical notation. Receive clear feedback, measured features and optional bounded processing previews. Compare/download the original and candidate. A novice should be able to submit input; whether the repair reliably improves novice recordings still requires evaluation. The prototype may support later melody/harmony extraction, but a commercial benefit or transcription gain is not yet measured.

## Completion and risk

The prototype, preparation paths, controlled learned response, optional cleaning and recorded validation are executable. Reliable general real-voice repair, a consistent BPDR advantage, independent musical review, novice/listener studies and downstream benefit remain open. The newer real-input and humming models failed their checks; the selected mixed-data checkpoint remains unchanged. Do not unlock a larger fit or force edits merely to make an output look different.

The notebooks cover problem definition, original data, design, executable implementation, training, evaluation and measured examples. A working walkthrough does not establish every research objective or predict assessment marks.
''')
    write(final / 'index.html', '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>BPDR · Implementation and results</title><style>
:root{color-scheme:light;--ink:#183d3c;--paper:#f6f2e9;--accent:#a44324;--line:#c7c9bb}*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:17px/1.6 'Segoe UI',sans-serif}main{max-width:1120px;margin:auto;padding:42px 38px}.eyebrow,.number,footer{font:12px/1.6 Consolas,monospace;letter-spacing:.12em;text-transform:uppercase}.eyebrow,.number{color:var(--accent)}h1{font:normal clamp(43px,6vw,72px)/1.06 Georgia,serif;letter-spacing:-.04em;margin:20px 0}h2{font:normal 29px/1.2 Georgia,serif;margin:10px 0}header{border-bottom:2px solid var(--ink);padding-bottom:26px}.intro{max-width:740px;font-size:20px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:0 42px}article{padding:27px 0;border-bottom:1px solid var(--line)}a{color:var(--ink);text-underline-offset:4px}article a{display:block;width:fit-content;margin:10px 0}a:hover{color:var(--accent)}a:focus-visible{outline:3px solid var(--accent);outline-offset:4px}.note{padding:18px 22px;border-left:4px solid var(--accent);background:#eee8dc;margin:30px 0}code{font:14px Consolas,monospace;overflow-wrap:anywhere}footer{padding-top:18px;border-top:1px solid var(--line);letter-spacing:.03em}@media(max-width:700px){main{padding:28px 22px}.grid{grid-template-columns:1fr}}
</style></head><body><main><header><div class="eyebrow">Vocal input processing / Research prototype</div><h1>From a recording<br>to a measured result.</h1><p class="intro">Explore the original data, follow the four-view BPDR method, and compare input with a verified processing preview.</p></header><section class="grid" aria-label="Implementation and results">
<article><span class="number">01 / Data</span><h2>Inspect the original recordings</h2><p>Vocadito coverage, duration, metadata, measured pitch, confidence and technical acceptance. Jamendo's role and source distinctions follow in the method notebook.</p><a href="reports/01_input_validation_and_analysis.html">Dataset analysis and input validation</a></article>
<article><span class="number">02 / Method</span><h2>Follow the model and evaluation</h2><p>Four views, compact Transformer, losses, grouped data, matched controls and the complete measured results.</p><a href="reports/02_data_review_and_paired_experiments.html">Method, input/output and results</a></article>
<article><span class="number">03 / Training</span><h2>Understand the Transformer fit</h2><p>Actual training partitions, four aligned views, checkpoint forward pass, loss terms, sampling schedules and every saved learning curve.</p><a href="reports/03_transformer_training.html">Transformer training walkthrough</a></article>
<article><span class="number">04 / Evaluation</span><h2>Check the measured evidence</h2><p>Recomputed per-frame metrics, matched controls, all seeds, real-source transfer, preservation and playable before/after comparisons.</p><a href="reports/04_transformer_evaluation.html">Transformer evaluation and input/output</a></article>
<article><span class="number">05 / System</span><h2>Listen before and after</h2><p>Run <code>BPDR_Workspace/Run-System.ps1</code>, then use the recording/upload workflow and the <strong>Input / output</strong> comparisons.</p><a href="http://127.0.0.1:8501">Open the running local system ↗</a><a href="VALIDATION.md">Understand each metric and example</a></article>
<article><span class="number">06 / Implementation</span><h2>Find the maintained code</h2><p>One implementation and four notebook sources. Current entry points are clearly identified; historical studies remain recoverable.</p><a href="CODE_MAP.md">Implementation map</a><a href="../source/bpdr_input/application/app.py">Application entry point</a><a href="../notebooks/">Editable notebook sources</a></article>
</section><aside class="note"><strong>Source identity and validation scope.</strong> The selected model uses synthetic pretraining and real Vocadito/Jamendo adaptation. The local pitch comparison has a known generated reference; the noise comparison uses real HumTrans audio with added recorded rain. Broader real-voice repair remains unproven. Originals are retained and processing candidates remain separate.</aside>
<footer><a href="README.md">Usage guide</a> · <a href="CURRENT_ARTIFACTS.json">Artifact checksums</a> · <a href="../documentation/README.md">Notes and proposal</a> · <a href="../../research_archive/evidence/README.md">Full experiment record</a></footer></main></body></html>''')
    assert all(row['outputs_preserved'] > 0 for row in artifacts['notebooks'])
    print(json.dumps(dict(status='views_exported', full_outputs_preserved=True,
        source_execution=str(run), notebooks=artifacts['notebooks']), indent=2))


if __name__ == '__main__':
    main()
