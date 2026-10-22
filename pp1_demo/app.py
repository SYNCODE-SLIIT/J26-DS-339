"""Run: uv run --no-sync streamlit run pp1_demo/app.py"""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import json
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import plotly.graph_objects as go
import streamlit as st
from pp1_demo.config import APP,RUNTIME,VOCAB,PC_NAMES
from pp1_demo.contracts import Settings,sha
from pp1_demo.orchestrator import run_pipeline

st.set_page_config(page_title='Harmony Lab · J26-DS-339',page_icon='🎵',layout='wide')
st.markdown('''<style>
.block-container{padding-top:2rem;max-width:1450px} h1{letter-spacing:-1.4px}
[data-testid="stMetric"]{background:#132b3a12;border:1px solid #7b99ab40;border-radius:12px;padding:14px}
.small-label{font-size:12px;letter-spacing:2px;color:#4b8e9a;font-weight:700}
</style>''',unsafe_allow_html=True)
st.markdown('<div class="small-label">J26-DS-339 / PP1 RESEARCH DEMONSTRATION</div>',unsafe_allow_html=True)
st.title('Harmony Lab')
st.caption('Listen to the input. Follow the melody. Inspect the chord predictions.')
st.info('Research prototype · BPDR bypass is explicit · Rahul mode is fixed to major · Final audio is synthesized instrumental accompaniment, not a generated harmony voice.')


def line_plot(x,y,title,ylabel):
    f=go.Figure(go.Scatter(x=x,y=y,mode='lines',line={'color':'#23a6a0','width':1}))
    f.update_layout(title=title,height=260,margin=dict(l=20,r=20,t=45,b=30),xaxis_title='Time from original audio start (s)',yaxis_title=ylabel)
    return f


def notes_plot(notes):
    f=go.Figure()
    for n in notes:
        f.add_trace(go.Scatter(x=[n['onset_s'],n['onset_s']+n['duration_s']],y=[n['pitch_midi']]*2,mode='lines',line=dict(width=6,color='#24a69d'),showlegend=False,hovertemplate=f"{librosa.midi_to_note(n['pitch_midi'])}<br>%{{x:.3f}}s<extra></extra>"))
    f.update_layout(title='Extracted melody notes',height=300,xaxis_title='Time (s)',yaxis_title='MIDI pitch')
    return f


def chord_plot(rows,title):
    f=go.Figure()
    for i,c in enumerate(rows):
        f.add_trace(go.Bar(x=[c['end_s']-c['start_s']],base=[c['start_s']],y=[title],orientation='h',text=[c['label']],textposition='inside',name=c['label'],showlegend=False,marker_color=['#298b96','#645ca7','#bc7b37'][i%3],hovertemplate=c['label']+'<extra></extra>'))
    f.update_layout(title=title,barmode='overlay',height=180,xaxis_title='Time (s)',margin=dict(l=10,r=10,t=50,b=30))
    return f


def show_audio(folder,key,label,run):
    if key in run['artifacts']:
        st.markdown('**'+label+'**');st.audio(str(folder/run['artifacts'][key]))


def content(owner,section):
    st.markdown((APP/'panels'/owner/(section+'.md')).read_text())

with st.sidebar:
    st.header('Demonstration controls')
    mode=st.radio('Execution',['Live recording','Saved run'])
    st.caption('Saved mode replays the original run and settings; it does not run models again.')
    vocab=json.loads((VOCAB/'feature_vocabularies.json').read_text())
    tonic=st.selectbox('Major reference tonic (user supplied)',PC_NAMES)
    bpm=st.number_input('BPM (user supplied)',40.,240.,120.,1.)
    meter=st.selectbox('Meter (user supplied)',[4,3],format_func=lambda n:f'{n}/4')
    first=st.number_input('First beat offset in seconds',0.,19.,0.,.01)
    genre=st.selectbox('Genre (user supplied)',vocab['genres'],index=vocab['genres'].index('pop'))
    st.text_input('Rahul mode','major — demo constant',disabled=True)
    bypass=st.checkbox('Bypass unavailable BPDR; use original audio',value=True)
    st.caption('No model repair is claimed while bypassed.')
    settings=Settings(PC_NAMES.index(tonic),bpm,meter,first,genre,bypass)
    st.divider();st.markdown('**Customize your panel**')
    st.caption('Edit Markdown files under pp1_demo/panels/<name>/. Runtime results are generated separately.')

if mode=='Live recording':
    st.subheader('01 / Input recording')
    uploaded=st.file_uploader('Upload a short solo vocal WAV or MP3 (0.25–20 seconds)',type=['wav','mp3'])
    example=ROOT/'datasets/humtrans/downloads/wav_data_sync_with_midi/F01_0001_0001_1_D.wav'
    source=None
    if uploaded:
        if uploaded.size>25*1024*1024:
            st.error('Maximum upload size is 25 MB.');st.stop()
        import hashlib
        target=RUNTIME/'uploads'/hashlib.sha256(uploaded.getvalue()).hexdigest()
        target.mkdir(parents=True,exist_ok=True)
        source=target/('upload'+Path(uploaded.name).suffix.lower())
        source.write_bytes(uploaded.getvalue())
        st.caption('Selected upload: '+uploaded.name)
    elif example.exists():
        source=example;st.caption('Local rehearsal example: '+example.name+' · dataset recording, not a new performance')
    if source:
        st.audio(str(source))
        try:
            info=sf.info(source);st.caption(f'{info.duration:.2f} seconds · {info.samplerate:,} Hz · {info.channels} channel(s)')
        except Exception as exc:st.error('Audio could not be decoded: '+str(exc));st.stop()
        if st.button('Run visual pipeline',type='primary',use_container_width=True):
            progress=st.empty()
            try:
                with st.spinner('Processing the recording…'):
                    folder,run=run_pipeline(source,settings,lambda name,status:progress.info(f'{name}: {status}'))
                st.session_state['run_folder']=str(folder)
                st.session_state['run_audio_hash']=sha(source)
            except Exception as exc:st.error(str(exc))
            progress.empty()
else:
    paths=sorted((RUNTIME/'runs').glob('*/manifest.json'),key=lambda p:p.stat().st_mtime,reverse=True)
    if paths:
        selection=st.selectbox('Saved execution',paths,format_func=lambda p:p.parent.name)
        st.session_state['run_folder']=str(selection.parent)
    else:st.warning('No saved runs yet. Run a recording first.')

folder=Path(st.session_state['run_folder']) if st.session_state.get('run_folder') else None
if folder and (folder/'manifest.json').exists():
    run=json.loads((folder/'manifest.json').read_text())
    damaged=[name for name,digest in run.get('artifact_hashes',{}).items() if not (folder/name).is_file() or sha(folder/name)!=digest]
    if sha(folder/run['input_file'])!=run['audio_sha256']:damaged.append('input')
    if damaged:st.error('Saved artifacts changed or missing: '+', '.join(damaged));st.stop()
    st.subheader('02 / Pipeline results')
    st.caption(f"{'Saved result' if mode=='Saved run' else 'Last executed result'} · {run['id']} · {run['source_name']} · {run['created_utc']}")
    if mode=='Live recording' and (run['settings']!=settings.validate() or (source and sha(source)!=run['audio_sha256'])):
        st.warning('The controls or recording differ from this result. Run again to apply the new selection.')
    with st.expander('Settings and provenance for this result'):
        st.json({'settings':run['settings'],'provenance':run['provenance']})
    cols=st.columns(4)
    for col,name in zip(cols,['Praveen','Vishwa','Pasan','Rahul']):
        status=run['stages'].get(name,{'status':'Not run'})
        col.metric(name,status['status']);col.caption(status.get('reason',status.get('error',f"{status.get('seconds',0):.2f} seconds")))
    if run.get('error'):st.error(run['error'])
    for warning in run['warnings']:st.warning(warning)
else:
    run=None
    st.subheader('02 / Pipeline')
    st.markdown('**Praveen: input preparation → Vishwa: melody → Pasan: basic chords → Rahul: refinement**')
    st.caption('Run a recording to populate real input/output graphs. Member explanations are available below.')

st.subheader('03 / Member research panels')
member_tabs=st.tabs(['Vishwa · melody understanding','Praveen · input','Pasan · harmony','Rahul · refinement'])
for tab,owner in zip(member_tabs,['vishwa','praveen','pasan','rahul']):
    with tab:
        tabs=st.tabs(['Input & output','Processing steps','Visual evidence','Decisions','Completed / missing / next'])
        with tabs[0]:
            if not run:st.info('Run a recording to see actual model outputs.')
            elif owner=='vishwa' and 'vishwa' in run:
                result=run['vishwa'];notes=result['notes']
                a,b,c=st.columns(3);a.metric('Extracted notes',len(notes));b.metric('Processing rate','22,050 Hz');c.metric('Method','pYIN')
                show_audio(folder,'prepared','Prepared mono recording',run)
                show_audio(folder,'melody','Synthesized extracted melody',run)
                st.plotly_chart(notes_plot(notes),use_container_width=True)
                st.dataframe(pd.DataFrame(notes),use_container_width=True,hide_index=True)
                with st.expander('Preprocessing and limitations'):st.json(result['preprocessing']);st.write(result['warnings'])
            elif owner=='praveen':
                st.warning('BPDR unavailable: required local model/data assets are missing. The pipeline explicitly bypasses this stage.')
                st.audio(str(folder/run['input_file']));st.caption('Original audio forwarded unchanged; no BPDR model inference.')
            elif owner in run:
                st.plotly_chart(chord_plot(run[owner]['chords'],owner.title()+' chords'),use_container_width=True)
                st.dataframe(pd.DataFrame(run[owner]['chords']),use_container_width=True,hide_index=True)
                if owner=='rahul':st.info('Mode: major — temporary demo assumption');st.json(run[owner]['unknown_inputs'])
                if owner=='pasan':st.caption(run[owner]['bridge']);st.caption('Genre is not an input to this selected Pasan model.')
            else:st.info('This stage has not produced outputs.')
        with tabs[1]:content(owner,'steps')
        with tabs[2]:
            content(owner,'evidence')
            if owner=='vishwa':
                st.bar_chart(pd.DataFrame({'Method':['Majority','pYIN','CNN','CNN–Transformer'],'Accuracy (%)':[12.10,48.55,42.53,39.45]}).set_index('Method'))
                if run and 'vishwa' in run:
                    with np.load(folder/'features.npz',allow_pickle=False) as z:features={k:z[k] for k in z.files}
                    raw,sr=sf.read(folder/run['input_file'],always_2d=True);hop=max(1,len(raw)//5000)
                    st.plotly_chart(line_plot(np.arange(0,len(raw),hop)/sr,raw.mean(axis=1)[::hop],'Original waveform','Amplitude'),use_container_width=True)
                    prepared,psr=sf.read(folder/'prepared.wav');hop=max(1,len(prepared)//5000)
                    st.plotly_chart(line_plot(np.arange(0,len(prepared),hop)/psr,prepared[::hop],'Prepared waveform','Amplitude'),use_container_width=True)
                    fig=go.Figure(go.Heatmap(z=features['mel_db'],x=features['times_s'],y=librosa.mel_frequencies(n_mels=80,fmax=11025),colorscale='Magma',colorbar_title='dB'))
                    fig.update_layout(title='Mel spectrogram',height=300,xaxis_title='Time (s)',yaxis_title='Mel band centre (Hz)');st.plotly_chart(fig,use_container_width=True)
                    f0=np.where(features['voiced'],features['f0_hz'],np.nan)
                    st.plotly_chart(line_plot(features['times_s'],librosa.hz_to_midi(f0),'Voiced pitch contour','MIDI pitch'),use_container_width=True)
                    st.bar_chart(pd.Series(run['vishwa']['harmony_features']['pitch_class_duration_distribution'],index=PC_NAMES,name='Duration share'))
                    st.caption('Pitch-class duration describes the extracted melody; it is not a key prediction.')
                    for name,pilot in run.get('pilot_models',{}).items():
                        fig=go.Figure(go.Heatmap(z=np.array(pilot['probabilities']).T,x=features['times_s'],y=pilot['classes'],zmin=0,zmax=1,colorbar_title='Probability'))
                        fig.update_layout(title=name+' · live weak-score pilot probabilities',height=300,xaxis_title='Time (s)');st.plotly_chart(fig,use_container_width=True)
                    st.caption('Pilot predictions have no octave or validated note boundaries. They do not drive downstream notes.')
        with tabs[3]:content(owner,'decisions')
        with tabs[4]:content(owner,'roadmap')

if run:
    st.subheader('04 / Listen and export')
    for key,label in [('basic','Pasan: basic chord instrument preview'),('refined','Rahul: refined chord instrument preview'),('combined','Prepared voice + refined accompaniment')]:show_audio(folder,key,label,run)
    st.caption('Instrument previews preserve predicted chord labels; unsupported/reserved labels remain visible and are silent in rendering.')
    st.download_button('Download run JSON',(folder/'manifest.json').read_bytes(),file_name='harmony_run_'+run['id']+'.json',mime='application/json')
    cols=st.columns(3)
    for col,key in zip(cols,['melody_midi','basic_midi','refined_midi']):
        if key in run['artifacts']:col.download_button('Download '+key.replace('_',' '),(folder/run['artifacts'][key]).read_bytes(),file_name=run['artifacts'][key],mime='audio/midi')
    if 'combined' in run['artifacts']:st.download_button('Download combined WAV',(folder/run['artifacts']['combined']).read_bytes(),file_name='combined.wav',mime='audio/wav')
