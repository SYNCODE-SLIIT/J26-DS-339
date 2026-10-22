"""Minimal working demo: choose input, run a method, compare actual output."""
import json
from pathlib import Path
import uuid

import numpy as np
import soundfile as sf
import streamlit as st

from bpdr_input import COMPONENT_ROOT
from bpdr_input.application.saved_results import noise_example, paired_snr_gain, pitch_example
from bpdr_input.repair.demo_runtime import METHODS, available_methods, run, selected_output
from bpdr_input.repair.demo_review import review, export_reviewed
from bpdr_input.application.demo_examples import examples
from bpdr_input.repair.learned_runtime import METHOD_NAME as LEARNED_METHOD

WORKSPACE = COMPONENT_ROOT / 'BPDR_Workspace'


def main():
    st.set_page_config(page_title='BPDR working demo', page_icon='🎙', layout='wide')
    st.title('Vocal input demo')
    st.write('Check a vocal idea, preserve or improve it, and review the audio for the next component.')
    st.caption('Originals are preserved. Acoustic usability checks and pitch edits are provisional; the system does not claim to recover unknown musical intention.')
    methods = available_methods()
    source_mode = st.radio('Input', ['Try an example', 'Upload audio', 'Record a voice'], horizontal=True)
    source, reference, known_case, score_path = None, None, None, None
    payload, suffix = None, '.wav'
    suggested = 0
    intent = 'singing_humming_chant'
    if source_mode == 'Try an example':
        options = examples()
        choice = st.selectbox('Recording', range(len(options)), format_func=lambda index: options[index]['label'])
        record = options[choice]
        source, reference, known_case = record['input'], record['reference'], record['id']
        score_path = record.get('score_mask_path')
        suggested = methods.index(record['method'])
        st.caption(record['source_kind'])
    elif source_mode == 'Upload audio':
        uploaded = st.file_uploader('Upload solo singing or humming (WAV / FLAC)', type=['wav', 'flac'])
        if uploaded:
            payload, suffix = uploaded.getvalue(), Path(uploaded.name).suffix.lower()
    else:
        uploaded = st.audio_input('Record 0.25–20 seconds of solo singing or humming')
        if uploaded:
            payload = uploaded.getvalue()
    if source_mode != 'Try an example':
        intent_labels = ['I sang, hummed or chanted a musical idea', 'I read or spoke without an intended tune', 'I am not sure']
        intent_choice = st.selectbox('What did you intend?', intent_labels, index=2)
        intent = ('singing_humming_chant', 'reading_spoken_text', 'unsure')[intent_labels.index(intent_choice)]
    method = st.selectbox('Processing method', methods, index=suggested, key='method_' + str(known_case))
    if method == METHODS[0]:
        st.caption('Historical experimental Transformer. Controlled generated-voice results do not establish reliable repair of real singing; inspect its actual output and checks.')
    if method == LEARNED_METHOD:
        st.caption('Learned from a small subset of real VocalSet singers. Intended scope: small local pitch deviations inside long stable notes; this is not a general singing-correctness model.')
        st.warning('Experimental model: broader calibration and singer-coverage criteria failed. The selected listening example demonstrates a verified improvement; the complete pilot includes unsuccessful and unscorable attempts. Listen before choosing any output.')
    if method == METHODS[1]:
        st.caption('This nonlearning baseline estimates the singer’s tuning offset and considers small corrections on stable held notes. It does not infer a key or intended melody.')
    declared_noise = known_case == 'noise'
    if method == METHODS[2] and not declared_noise:
        declared_noise = st.checkbox('The first 0.7 seconds contain only steady room noise, with no voice, breath, speech or music.')
    if source:
        st.audio(str(source))
    elif payload:
        st.audio(payload)
    disabled = not (source or payload or known_case == 'silence') or (method == METHODS[2] and not declared_noise)
    if st.button('Process recording', type='primary', disabled=disabled):
        for key in ('demo_result', 'demo_export', 'demo_extraction'):
            st.session_state.pop(key, None)
        if payload is not None or known_case == 'silence':
            folder = WORKSPACE / 'results/demo_sessions/inputs' / uuid.uuid4().hex
            folder.mkdir(parents=True)
            source = folder / ('input' + suffix)
            if payload is not None:
                source.write_bytes(payload)
            else:
                sf.write(source, np.zeros(22050, dtype=np.float32), 22050, subtype='PCM_24')
        with st.spinner('Processing the recording and checking the candidate…'):
            try:
                effective_method = METHODS[3] if intent == 'reading_spoken_text' else method
                directory, report = run(source, effective_method, declared_noise)
                acoustic_review = review(directory, report, intent)
                st.session_state['demo_result'] = dict(directory=str(directory), report=report,
                    method=effective_method, known_case=known_case, reference=str(reference) if reference else None,
                    review=acoustic_review, score_mask_path=score_path)
                st.session_state.pop('demo_export', None)
                st.session_state.pop('demo_extraction', None)
            except Exception as exc:
                st.error('Processing stopped: ' + str(exc))
    if 'demo_result' not in st.session_state:
        st.info('Press Process recording to run the selected method. No training is required.')
        return
    result = st.session_state['demo_result']
    report, directory = result['report'], Path(result['directory'])
    if 'review' not in result:
        result['review'] = review(directory, report, 'unsure')
        st.session_state['demo_result'] = result
    acoustic_review = result['review']
    st.divider()
    st.subheader('Input and output')
    st.caption('Completed method: ' + result['method'] + '. Recording: ' + Path(report['source']).name)
    try:
        output, label = selected_output(report)
    except (OSError, ValueError) as exc:
        st.error(str(exc))
        return
    left, right = st.columns(2)
    originals = list(directory.glob('original.*'))
    if originals:
        left.write('Input'); left.audio(str(originals[0]))
    right.write(label)
    if output:
        right.audio(str(output))
        right.download_button('Download comparison audio', output.read_bytes(), output.name, 'audio/wav')
        if label == 'Original retained':
            st.info('No supported edit was applied. The original working copy is retained; this is not proof that every sung note matches the intended melody.')
        else:
            st.success('A processed preview passed its numerical checks. Play both recordings at the same volume.')
            st.caption('This preview remains separate from the original downstream working copy.')
    else:
        st.warning(report['handoff'].get('user_message') or 'No usable audio was accepted. Provide another recording or inspect the file.')
    assessed_role = 'preview' if label == 'Processed preview' else 'original'
    assessed = acoustic_review.get(assessed_role, acoustic_review['original'])
    st.subheader('Can this vocal idea go forward?')
    if assessed['decision'] == 'PASS_EVIDENCE':
        st.info(assessed['feedback'])
    elif assessed['decision'] in ('NEED_CLEARER_INPUT', 'NEED_INTENT_CONFIRMATION'):
        st.warning(assessed['feedback'])
    else:
        st.info(assessed['feedback'])
    if assessed['decision'] == 'NEED_INTENT_CONFIRMATION':
        if st.button('I intended a melody: reassess the vocal evidence'):
            result['review'] = review(directory, report, 'singing_humming_chant')
            st.session_state['demo_result'] = result
            st.rerun()
    if output and result['reference'] and label == 'Processed preview':
        if result['known_case'] == 'noise' and report.get('noise_candidate'):
            baseline, sr = sf.read(result['reference'], dtype='float64')
            before, before_sr = sf.read(originals[0], dtype='float64')
            after, after_sr = sf.read(output, dtype='float64')
            if sr == before_sr == after_sr:
                st.metric('Paired SNR improvement', f'{paired_snr_gain(baseline, before, after):.2f} dB')
        elif result['known_case'] == 'pitch':
            from bpdr_input.repair.rule_correction import fine_pitch
            baseline, sr = sf.read(result['reference'], dtype='float32')
            before, _ = sf.read(originals[0], dtype='float32')
            after, _ = sf.read(output, dtype='float32')
            times = np.arange(len(baseline) // 441 + 1) * .02
            # The controlled reference is used for scoring after processing only.
            with np.load(Path(report['source']).parent / 'views.npz') as target:
                mask = target['mask'].astype(bool) & np.isclose(np.abs(target['correction_target_cents']), 20.)
            ref_pitch = fine_pitch(baseline, sr, times)
            residual_before = fine_pitch(before, sr, times) - ref_pitch
            residual_after = fine_pitch(after, sr, times) - ref_pitch
            mask = mask & np.isfinite(residual_before) & np.isfinite(residual_after)
            if mask.any():
                a, b = st.columns(2)
                a.metric('Input local pitch error', f'{abs(np.median(residual_before[mask])):.2f} cents')
                b.metric('Output local pitch error', f'{abs(np.median(residual_after[mask])):.2f} cents')
                st.line_chart({'Input residual (cents)': residual_before[mask], 'Output residual (cents)': residual_after[mask]})
        elif result['known_case'] == 'learned_pitch' and result.get('score_mask_path'):
            from bpdr_input.repair.rule_correction import fine_pitch
            baseline, sr = sf.read(result['reference'], dtype='float32')
            before, before_sr = sf.read(originals[0], dtype='float32')
            after, after_sr = sf.read(output, dtype='float32')
            times = np.arange(len(baseline) // 441 + 1) * .02
            with np.load(result['score_mask_path']) as targets:
                if 'scoring_mask' in targets.files:
                    mask = targets['scoring_mask'].astype(bool)
                elif 'mask' in targets.files and 'target' in targets.files:
                    mask = targets['mask'].astype(bool) & (targets['target'] != 0)
                else:
                    mask = targets['common_reliable_mask'].astype(bool) & (targets['correction_target_cents'] != 0)
            if sr == before_sr == after_sr and len(before) == len(after) == len(baseline) and mask.shape == times.shape:
                reference_pitch = fine_pitch(baseline, sr, times)
                before_error = fine_pitch(before, sr, times) - reference_pitch
                after_error = fine_pitch(after, sr, times) - reference_pitch
                if not np.isfinite(before_error[mask]).all() or not np.isfinite(after_error[mask]).all():
                    st.warning('Some declared score frames have unmeasurable pitch. A complete before/after error score is unavailable.')
                elif mask.any():
                    a, b = st.columns(2)
                    a.metric('Input local absolute pitch error', f'{np.mean(np.abs(before_error[mask])):.2f} cents')
                    b.metric('Output local absolute pitch error', f'{np.mean(np.abs(after_error[mask])):.2f} cents')
                    st.line_chart({'Input residual (cents)': before_error[mask], 'Output residual (cents)': after_error[mask]},
                                  x_label='Scored frame index', y_label='Residual relative to the unshifted performance (cents)')
                    st.caption('The unshifted real recording and known intervention mask are used for this score only, after audio-only inference.')
    if report.get('model_metrics'):
        with st.expander('What this learned model achieved on separate singers'):
            metrics = report['model_metrics']
            if 'error_reduction_fraction' in metrics:
                scored_singers = metrics.get('singers_with_valid_shift_labels', metrics['heldout_singers'])
                st.write(f"{metrics['model_type']}: {100*metrics['error_reduction_fraction']:.1f}% lower measured pitch error across {metrics['audited_shifts']} audited controlled shifts from {scored_singers} singer(s) absent from fitting, including withheld outputs. {metrics['heldout_singers']} singers were attempted.")
                st.write(f"{metrics['verified_improvements']}/{metrics['audited_shifts']} improved, verified previews; {metrics['originals_preserved']}/{metrics['originals_tested']} unshifted recordings preserved. {metrics['attempted_shifts']-metrics['audited_shifts']} attempted interventions failed their creation audit and have no valid labels.")
            st.json(metrics)
            st.caption(report.get('model_scope', 'Scoped pilot result; not general natural-singing accuracy.'))
    allowed_roles = [role for role in ('original', 'preview') if acoustic_review.get(role, {}).get('decision') == 'PASS_EVIDENCE']
    if allowed_roles:
        st.subheader('Choose audio for the next component')
        chosen_role = st.radio('Version to hand off', allowed_roles,
                               format_func=lambda role: 'Original working copy' if role == 'original' else 'Verified prepared preview',
                               key='export_role_' + directory.name)
        reviewed_hash = acoustic_review[chosen_role]['waveform_sha256']
        confirmed = st.checkbox('I listened to this version and it preserves the musical idea I intended.', key='confirm_' + directory.name + '_' + chosen_role + '_' + reviewed_hash)
        if st.button('Create accepted demo package', disabled=not confirmed):
            try:
                exported, handoff = export_reviewed(directory, report, acoustic_review, use_preview=chosen_role == 'preview', confirmed=confirmed)
                st.session_state['demo_export'] = dict(directory=str(exported), handoff=handoff)
                st.session_state.pop('demo_extraction', None)
            except (OSError, ValueError) as exc:
                st.error('Export stopped: ' + str(exc))
    if 'demo_export' in st.session_state:
        exported = st.session_state['demo_export']
        package_root = Path(exported['directory'])
        st.success('Created a separate user-reviewed demo handoff: ' + exported['handoff']['status'])
        st.download_button('Download accepted WAV and metadata', (package_root / 'accepted_package.zip').read_bytes(), 'bpdr_accepted_package.zip', 'application/zip')
        st.caption('This export uses a provisional demo policy. The original shared handoff and research approval remain unchanged.')
        if st.button('Try the accepted audio with the melody extractor'):
            from bpdr_input.repair.demo_extraction import run as extract_demo
            with st.spinner('Running the existing melody extractor on this accepted waveform…'):
                try:
                    st.session_state['demo_extraction'] = extract_demo(package_root)
                except (OSError, ValueError) as exc:
                    st.error('Extractor stopped: ' + str(exc))
    if 'demo_extraction' in st.session_state:
        extracted = st.session_state['demo_extraction']
        if extracted['success']:
            st.success('The next-component extractor consumed the reviewed WAV and produced melody files.')
            for item in extracted['artifacts']:
                path = Path(item['path'])
                st.download_button('Download ' + path.name, path.read_bytes(), path.name, key='melody_' + path.suffix)
        else:
            st.warning('The extractor did not complete. The accepted audio is preserved; inspect the processing details.')
    with st.expander('Processing details'):
        st.write('Outcome:', report['outcome'])
        st.write('Technical input status:', report['handoff']['status'])
        st.json(report)
        st.json(acoustic_review)
        if 'demo_extraction' in st.session_state:
            st.json(st.session_state['demo_extraction'])
    st.caption('Supported input: one voice, 0.25–20 seconds. Ordinary reading may contain too little melody; the current system does not invent missing musical intention.')


if __name__ == '__main__':
    main()
