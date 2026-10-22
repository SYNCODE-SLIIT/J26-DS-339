"""Local vocal-processing interface. Start with Run-System.ps1."""
from bpdr_input import COMPONENT_ROOT
import json
from pathlib import Path
import uuid

import numpy as np
import streamlit as st

COMPONENT = COMPONENT_ROOT


def main():
    st.set_page_config(page_title='BPDR | Vocal input laboratory', page_icon='🎙', layout='wide')
    st.markdown('''<style>
    .stApp {background:#f7f4ed;color:#182f32}
    h1,h2,h3 {font-family:Georgia,serif!important;letter-spacing:-.025em}
    h1 {font-size:2.6rem!important} .block-container {max-width:1180px;padding-top:1.5rem}
    [data-testid="stMetric"] {border-top:3px solid #b55732;padding-top:14px}
    .eyebrow {font-family:monospace;letter-spacing:.16em;color:#9b482b}
    </style>''', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">VOCAL INPUT / BOUNDARY-PITCH DISENTANGLED REPAIR</div>', unsafe_allow_html=True)
    st.title('BPDR vocal input demo')
    st.caption('Experimental Transformer with synthetic pretraining and Vocadito/Jamendo adaptation. Real-voice pitch repair and a BPDR advantage remain unproven. Originals are retained.')
    panel, live, outputs, evidence, method = st.tabs(['Panel demo', 'Process audio', 'Input / output', 'Datasets & results', 'Method'])
    with panel:
        from bpdr_input.application.panel_ui import render_panel
        render_panel()
    with live:
        st.subheader('One voice. One short phrase.')
        st.write('Record 0.25–20 seconds of solo singing or humming, without accompaniment. Use a comfortable volume and leave natural pauses. WAV and FLAC uploads are supported.')
        st.caption('Files stay in this BPDR component on this computer. They are retained for this local run; they are not automatically added to training or research. Session paths are shown below for removal by the owner.')
        mode = st.radio('Input source', ['Upload audio', 'Record with microphone', 'Use an example recording'], horizontal=True)
        source = None; payload = None; suffix = '.wav'
        if mode == 'Upload audio':
            uploaded = st.file_uploader('Choose a vocal recording', type=['wav', 'flac'])
            if uploaded is not None:
                payload = uploaded.getvalue(); suffix = Path(uploaded.name).suffix.lower()
        elif mode == 'Record with microphone':
            uploaded = st.audio_input('Record a solo vocal phrase')
            if uploaded is not None:
                payload = uploaded.getvalue()
        else:
            root = COMPONENT/'runs/demo_inputs/488d391b4d824368933103c13f24dac6'
            options = sorted(root.glob('*.wav'))
            demo_labels={}
            study = COMPONENT/'runs/controlled_study/current.json'
            if study.exists():
                catalog=json.loads((Path(json.loads(study.read_text())['path'])/'catalog.json').read_text())['records']
                example=next(r for r in catalog if r['split']=='heldout' and abs(r['cents'])==20 and r['corruption']=='split')
                options = [Path(example['path']).parent/'C.wav', Path(example['path']).parent/'A.wav']+options
            noise_study=COMPONENT/'runs/noise_study/current.json'
            if noise_study.exists():
                nr=json.loads((Path(json.loads(noise_study.read_text())['path'])/'evidence.json').read_text())
                control=next((r for r in nr['rows'] if r['passes'] and r['source']['dataset']=='vocadito'),None)
                if control:
                    p=Path(control['directory'])/'noisy.wav';options.append(p)
                    demo_labels[str(p)]='Real Vocadito source · known added noise · room-noise prefix'
            environment=COMPONENT/'runs/environmental_noise_study/current.json'
            if environment.exists():
                er=json.loads((Path(json.loads(environment.read_text())['path'])/'evidence.json').read_text())
                control=next((r for r in er['rows'] if r['passes'] and r['dataset']=='humtrans' and r['noise']=='rain' and r['snr_db']==20),None)
                if control:
                    p=Path(control['directory'])/'noisy.wav';options.append(p)
                    demo_labels[str(p)]='Real HumTrans humming + recorded rain · controlled mixture'
            if options:
                source = st.selectbox('Example recording (source is identified)', options,
                    format_func=lambda p: demo_labels.get(str(p),{'C.wav':'Synthetic held-out phrase · local 20-cent detuning','A.wav':'Synthetic held-out phrase · unchanged original'}.get(p.name,p.name)))
        if payload is not None:
            st.audio(payload)
        elif source:
            st.audio(str(source))
        render = st.checkbox('Try a separate experimental rendered candidate', value=False,
                               help='Permission and verification may withhold or roll back the candidate. It never replaces the downstream audio.')
        noise = st.checkbox('Try room-noise reduction: I confirm the first 0.7 seconds contain only steady room noise',value=False,
            help='No humming, speech, breath or music in that interval. The preview is withheld if screening or voice-preservation checks fail. This is not an automatic speech/noise detector.')
        if st.button('Analyse recording', type='primary', disabled=source is None and payload is None):
            if payload is not None:
                folder = COMPONENT/'runs/ui_uploads'/uuid.uuid4().hex; folder.mkdir(parents=True)
                source = folder/('recording'+suffix); source.write_bytes(payload)
            with st.spinner('Checking the recording and measuring model output locally…'):
                if noise:
                    from bpdr_input.repair.noise_preview import run
                    from bpdr_input.repair.noise_policy_v2 import ColoredNoisePolicy
                    directory, report = run(source, (0.,.7), render_preview=render,noise_policy=ColoredNoisePolicy())
                else:
                    from bpdr_input.repair.model_runtime import run
                    directory, report = run(source, render_preview=render)
            st.session_state['last_run'] = (str(directory), report)
        if 'last_run' in st.session_state:
            location, r = st.session_state['last_run']; folder = Path(location)
            st.subheader('Last completed analysis · '+r['handoff']['status'].replace('_',' ').title())
            st.caption('Analysed recording: '+Path(r['source']).name)
            st.write(r['handoff']['user_message'] or 'The accepted working copy keeps the original pitch and timing.')
            st.write('Experimental model outcome:', r['outcome'])
            if r.get('reason'):
                st.warning(r['reason'])
            cols = st.columns(2)
            originals = list(folder.glob('original.*'))
            if originals:
                cols[0].write('Original recording'); cols[0].audio(str(originals[0]))
            accepted = r['handoff'].get('accepted_audio')
            preview = r.get('noise_candidate') or r.get('candidate')
            if preview:
                cols[1].write('Processed preview — separate from the retained working copy')
                cols[1].audio(preview['path'])
            if accepted:
                if not preview:
                    cols[1].write('Retained working copy — no verified change requested'); cols[1].audio(accepted['path'])
                st.download_button('Download retained working copy', Path(accepted['path']).read_bytes(), 'bpdr_accepted.wav', 'audio/wav')
            if r.get('candidate') and not r.get('noise_candidate'):
                st.caption('Pitch candidate passed numerical checks; independent musical quality review remains pending.')
                st.download_button('Download separate pitch-repair preview',Path(r['candidate']['path']).read_bytes(),'bpdr_pitch_preview.wav','audio/wav')
            if r.get('noise_candidate'):
                st.download_button('Download separate noise-reduction preview',Path(r['noise_candidate']['path']).read_bytes(),'bpdr_noise_preview.wav','audio/wav')
                st.caption('Model inference uses recalculated features from this preview. Downstream accepted audio remains the original working copy.')
            if r.get('noise'):
                if not r.get('noise_candidate'):
                    st.warning('Noise cleaning was withheld because the reference or voice-preservation checks did not pass. Your original working copy remains available.')
                with st.expander('Noise-reference screening and preservation checks'):
                    st.json(r['noise'])
            predictions = folder/'model_predictions.npz'
            if predictions.exists():
                from matplotlib.figure import Figure
                with np.load(predictions) as z:
                    fig = Figure(figsize=(10,3)); ax=fig.subplots()
                    for i,label in enumerate(['Nominal boundaries','Jittered boundaries','Removed boundaries']):
                        ax.plot(z['times'], z['predictions'][i], label=label, lw=1)
                    ax.set(xlabel='Time (seconds)', ylabel='Predicted correction (cents)', title='Model requests — these are not musical reference labels')
                    ax.legend();st.pyplot(fig)
            st.download_button('Download diagnostics and hand-off JSON', json.dumps(r,indent=2), 'bpdr_diagnostics.json', 'application/json')
            st.caption('Local run: '+location)
            with st.expander('Reasons, proposed intervals and operation log'):
                st.json(r)
    with outputs:
        from bpdr_input.application.saved_results import pitch_example, noise_example
        st.subheader('Compare the same recording before and after processing')
        st.write('These saved examples come from completed, checked runs. Play both at the same volume. The processing output is a separate candidate; the accepted downstream package retains the original working copy.')
        for label, result in [('Local pitch correction', pitch_example()), ('Room-noise reduction', noise_example())]:
            st.markdown('### '+label)
            st.write(result['kind'])
            before, after = st.columns(2)
            before.write('Input recording'); before.audio(str(result['input']))
            after.write('Processed preview'); after.audio(str(result['output']))
            if 'before_cents' in result:
                x, y = st.columns(2)
                x.metric('Input local pitch error', f"{abs(result['before_cents']):.2f} cents")
                y.metric('Output local pitch error', f"{abs(result['after_cents']):.2f} cents")
            else:
                x, y, z = st.columns(3)
                x.metric('Added-noise SNR improvement', f"{result['snr_gain_db']:.2f} dB")
                y.metric('Noise-only prefix attenuation', f"{result['reference_attenuation_db']:.2f} dB")
                z.metric('Median measured pitch change', f"{result['median_pitch_change_cents']:.2f} cents")
            st.caption(result['scope'])
            st.download_button('Download '+label.lower()+' preview', result['output'].read_bytes(),
                'bpdr_'+('pitch' if 'before_cents' in result else 'noise')+'_preview.wav', 'audio/wav')
    with evidence:
        st.subheader('What was actually measured')
        root = COMPONENT/'runs/controlled_study'
        from bpdr_input.experiments.evaluation.transfer_study import frozen_runs, read
        rows=[]
        for seed,path in frozen_runs():
            for condition,result in read(path/'evidence.json')['results'].items():
                s=result['heldout'];rows.append(dict(seed=seed,model=condition,error_cents=s['correction_mae'],response_gain=s['median_response_gain'],original_change_cents=s['unshifted_alteration']))
        st.write('Synthetic evaluation: 72 variants, six excluded synthetic voices; no-correction error 19.31 cents. All seeds are shown.')
        st.dataframe(rows, hide_index=True, width='stretch')
        transfer=COMPONENT/'runs/presentation_study/transfer_current.json'
        if transfer.exists():
            run=Path(read(transfer)['path']);data=read(run/'evidence.json');rows=[]
            for seed,conditions in data['results'].items():
                for condition,result in conditions.items():
                    for ds,s in result['by_dataset'].items():
                        rows.append(dict(seed=seed,model=condition,dataset=ds,error_cents=s['correction_mae'],response_gain=s['median_response_gain'],original_change_cents=s['unshifted_alteration']))
            st.write('Frozen synthetic models on real development recordings: 285 Jamendo and 19 Vocadito variants. These are controlled shifts of real audio, not a final musical-quality test.')
            st.dataframe(rows,hide_index=True,width='stretch')
        adaptation=COMPONENT/'runs/presentation_study/adapt_current.json'
        if adaptation.exists():
            data=read(Path(read(adaptation)['path'])/'evidence.json');rows=[]
            for condition,result in data['results'].items():
                for ds,s in result['by_dataset'].items():
                    rows.append(dict(model=condition,dataset=ds,error_cents=s['correction_mae'],response_gain=s['median_response_gain'],original_change_cents=s['unshifted_alteration'],gate_passes=result['engineering_gate_passes']))
            st.write('Each model trained for 300 additional updates using real Vocadito/Jamendo training pairs mixed with synthetic pairs. All outcomes are retained. The mixed-data BPDR is the experimental processing model after synthetic-retention and audio checks; real-singer repair remains unvalidated. Synthetic reference weights are preserved.')
            st.dataframe(rows,hide_index=True,width='stretch')
        noise_pointer=COMPONENT/'runs/noise_study/current.json'
        if noise_pointer.exists():
            noise_data=read(Path(read(noise_pointer)['path'])/'evidence.json')
            st.write('Noise-cleaning controls: known added noise on six real training excerpts and two generated phrases. This is a separate preservation gate, not successful real pitch correction.')
            st.dataframe([dict(dataset=k,**v) for k,v in noise_data['summary'].items()],hide_index=True,width='stretch')
            st.caption('All failures are retained. These are correlated controls, not 32 independent users. No naturally noisy paired evaluation or listening approval.')
        environment=COMPONENT/'runs/environmental_noise_study/current.json'
        if environment.exists():
            er=read(Path(read(environment)['path'])/'evidence.json')
            st.write('Additional fixed-policy checks: field-recorded noise mixed with real Vocadito, Jamendo and HumTrans audio. The original failed noise policy is retained separately.')
            st.dataframe([dict(noise=k,**v) for k,v in er['summary'].items()],hide_index=True,width='stretch')
            st.caption('Breathing controls had no eligible declared noise reference: identity is expected, not successful breath detection. Recorded-noise mixtures are not naturally co-recorded novice-user validation.')
        humming=COMPONENT/'runs/humming_experiment/current.json'
        if humming.exists():
            hr=Path(read(humming)['path']);he=read(hr/'evidence.json')
            st.write('Humming-data suitability: all initial recordings and the prospectively selected long-note extension are retained.')
            st.dataframe([he['summary']],hide_index=True,width='stretch')
            if (hr/'frozen_evaluation/evidence.json').exists():
                result=read(hr/'frozen_evaluation/evidence.json')
                st.caption('Frozen selected-model probe: '+str(result['variants'])+' variants; the small selected subset cannot establish general humming accuracy.')
        learning=COMPONENT/'runs/real_learning_v2/current.json'
        if learning.exists():
            lr=Path(read(learning)['path'])
            if (lr/'pilot_current.json').exists():
                result=read(Path(read(lr/'pilot_current.json')['path'])/'evidence.json')
                st.write('A4-safe real-input learning check: eight fixed training variants, 200 updates. Larger fitting stayed locked because this training gate failed.')
                st.dataframe([dict(dataset=k,**v) for k,v in result['results']['augmentation_only']['by_dataset'].items()],hide_index=True,width='stretch')
        hum_fit=COMPONENT/'runs/humming_learning/current.json'
        if hum_fit.exists():
            hf=read(Path(read(hum_fit)['path'])/'evidence.json')
            st.write('Separate humming-learning attempt: 12 real-humming training variants mixed with synthetic data, 200 updates. The selected model was retained.')
            st.dataframe([dict(split=k,**v['all']) for k,v in hf['results'].items()],hide_index=True,width='stretch')
            st.caption('The six humming development variants come from one prospectively selected recording/person. New humming weights failed preservation/generalisation checks and are diagnostic only.')
        example=COMPONENT/'runs/demo/bpdr_explainer'
        if (example/'example_result.json').exists():
            st.subheader('Recorded end-to-end synthetic correction')
            c=st.columns(3)
            for column,name,label in zip(c,['example_original.wav','example_detuned.wav','example_bpdr_corrected.wav'],['Constructed reference','Known local detuning','Verified candidate']):
                column.write(label)
                if (example/name).exists():column.audio(str(example/name))
            st.caption('Recorded fixture with nominal/corrupted boundary metadata; separate from the audio-only live inference tab.')
    with method:
        st.subheader('A / B / C / D')
        st.write('A: original audio and nominal boundaries. B: same audio, changed boundaries. C: locally detuned audio and nominal boundaries. D: same detuned audio, changed boundaries.')
        st.write('The supervised anchor preserves originals and learns the known compensating correction. Boundary loss asks for the same correction after metadata changes. Counter-detuning loss asks the correction to change opposite to the injected pitch shift. Smoothness discourages abrupt correction curves.')
        st.write('Three matched four-layer Transformer encoders: augmentation-only (71 features), frame-only (70), and BPDR (71). Runtime uses one BPDR checkpoint and compares alternate boundary views; the comparison models are offline controls.')
        st.warning('Further validation: reliable real-voice pitch correction, independent musical review, novice-user and listening studies, and downstream benefit. Numerical candidate checks do not establish these outcomes.')
        st.write('Vocadito supplies human-performed pitch/note references, not guaranteed ideal intonation. Jamendo supplies separated vocals and model predictions. Neither is promoted to human musical approval.')
        st.write('Optional noise reduction requires a declared room-noise-only prefix, uses a limited smoothed spectral mask with original phase, and checks measured pitch and voiced-frame preservation. It does not remove overlapping speech or prove real-singer pitch correction.')


if __name__ == '__main__':
    main()
