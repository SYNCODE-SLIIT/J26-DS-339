"""Fast panel walkthrough backed by verified, packaged saved evidence."""
import json
from pathlib import Path

import streamlit as st

from bpdr_input import COMPONENT_ROOT


def render_panel():
    from bpdr_input.reporting.panel_demo import load_verified_manifest

    root = COMPONENT_ROOT / 'BPDR_Workspace/results/panel_demo'
    st.subheader('Panel demonstration')
    st.write('Compare a known local pitch repair and conservative noise reduction. Each recording identifies its source and the limits of the evidence.')
    try:
        manifest = load_verified_manifest(root / 'manifest.json')
    except (OSError, ValueError, KeyError) as exc:
        st.error('The panel evidence package is unavailable or changed. Run Run-Panel-Demo.ps1 -Build before presenting.')
        st.caption(str(exc))
        return
    cases = manifest['cases']
    chosen = st.selectbox('Demonstration case', range(len(cases)), format_func=lambda index: cases[index]['title'])
    case = cases[chosen]
    st.write(case.get('source_kind', ''))
    if case.get('summary'):
        st.write(case['summary'])
    if case.get('method'):
        st.caption('Method: ' + case['method'])
    st.caption('Saved outcome: ' + case.get('outcome', 'Recorded engineering check'))
    metrics = case.get('metrics', {})
    if 'before_median_cents' in metrics:
        before, after = st.columns(2)
        before.metric('Input local pitch error', f"{abs(metrics['before_median_cents']):.2f} cents")
        after.metric('Output local pitch error', f"{abs(metrics['after_median_cents']):.2f} cents")
        st.caption('Absolute median residual in the known shifted region, relative to the constructed original.')
    elif 'snr_gain_db' in metrics:
        gain, pitch = st.columns(2)
        gain.metric('Paired SNR improvement', f"{metrics['snr_gain_db']:.2f} dB")
        pitch.metric('Median pitch movement', f"{metrics['median_pitch_change_cents']:.2f} cents")
    columns = st.columns(2)
    for column, role, label in zip(columns, ('input', 'output'), ('Input recording', 'Selected output / retained audio')):
        item = case.get(role)
        with column:
            st.write(label)
            if item:
                path = root / item['path']
                st.audio(str(path))
                st.download_button('Download ' + role, path.read_bytes(), path.name, 'audio/wav', key='panel_' + str(chosen) + role)
            else:
                st.info('No accepted output. Inspect the reason and request another recording or fix the file.')
    if metrics:
        with st.expander('All measured values'):
            st.dataframe([dict(measure=name.replace('_', ' '), value=str(value)) for name, value in metrics.items()], hide_index=True, width='stretch')
    plot = case.get('plot')
    if plot:
        st.image(str(root / (plot['path'] if isinstance(plot, dict) else plot)))
    if case.get('reference'):
        with st.expander('Known reference for this controlled example'):
            st.audio(str(root / case['reference']['path']))
            st.caption('A reference belongs to this controlled evaluation. A live free-humming user does not supply an ideal target melody.')
    if case.get('handoff_status'):
        st.write('Recorded downstream status:', case['handoff_status'])
    if case.get('handoff_status') in ('ACCEPT_UNCHANGED', 'ACCEPT_PREPARED'):
        st.caption('Prepared audio is a separate engineering preview. The current downstream handoff retains the original working copy. Musical approval and natural real-singer repair remain unvalidated.')
    else:
        st.caption('This case has no accepted audio. Nothing is handed to the next component. The status describes the recording or processing problem, not the singer\'s musical ability.')
    with st.expander('Research question and four-view evidence'):
        st.write('Does BPDR keep its correction stable when only note boundaries change, while responding to a known physical pitch change?')
        st.table([
            {'view': 'A', 'waveform': 'Original', 'boundaries': 'Nominal'},
            {'view': 'B', 'waveform': 'Same as A', 'boundaries': 'Split, merge or jitter'},
            {'view': 'C', 'waveform': 'Known local detuning', 'boundaries': 'Nominal'},
            {'view': 'D', 'waveform': 'Same as C', 'boundaries': 'Split, merge or jitter'},
        ])
        st.write('A/B and C/D test boundary stability. A/C and B/D test the compensating pitch response. A model that always predicts zero cannot pass the latter check.')
        if manifest.get('four_view', {}).get('plot'):
            st.image(str(root / manifest['four_view']['plot']))
        st.json(manifest.get('four_view', {}))
        st.warning('A BPDR advantage over matched controls, general real-voice repair, and validated melodic-usability decisions remain open research requirements.')
    with st.expander('Next-component evidence and remaining integration'):
        st.json(manifest.get('downstream_check', {}))
        st.caption('A frozen extractor consistency check is separate from completed team integration or improved melody-extraction accuracy.')
    current = root / 'rehearsal_current.json'
    if current.exists():
        try:
            summary_path = Path(json.loads(current.read_text())['path']) / 'summary.json'
            if not summary_path.resolve().is_relative_to(root.resolve()):
                raise ValueError('Rehearsal path leaves the panel package')
            summary = json.loads(summary_path.read_text())
            with st.expander('Fresh execution rehearsal'):
                st.write('All declared cases passed:', summary['passed'])
                st.dataframe(summary['cases'], hide_index=True, width='stretch')
                st.caption('Inference and rendering only. No training or replacement of model weights.')
        except (OSError, ValueError, KeyError) as exc:
            st.warning('Rehearsal receipt could not be read: ' + str(exc))
    with st.expander('Evidence provenance and limitations'):
        st.json(case)
        st.json(case.get('limitations', []))
        st.json(manifest.get('limitations', []))
    notebook = COMPONENT_ROOT / 'BPDR_Workspace/notebooks/05_panel_demonstration.ipynb'
    if notebook.exists():
        st.download_button('Download executed panel notebook', notebook.read_bytes(), notebook.name, 'application/x-ipynb+json')
