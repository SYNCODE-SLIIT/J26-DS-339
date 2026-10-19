"""Build the complete readable BPDR/ExtraTrees walkthrough from source evidence.

This builder writes notebook source only. Execution verifies recorded work and
runs one bounded inference; it never retrains or silently changes old results.
"""
from pathlib import Path
from textwrap import dedent

import nbformat

WORKSPACE = Path(__file__).resolve().parents[1]


def build():
    cells = []

    def md(value):
        cells.append(nbformat.v4.new_markdown_cell(dedent(value).strip()))

    def code(value):
        cells.append(nbformat.v4.new_code_cell(dedent(value).strip()))

    md('''
    # Vocal input model: the complete data-to-output notebook

    **Read this notebook in order.** It explains the problem, why we tried a different model, what data was actually usable, how labels and features were constructed, how training and selection worked, what the measurements mean, and how an actual waveform reaches the next component.

    **Current status:** an optional **ExtraTrees Regressor with 96 trees** produces measured improvements on selected real-voice controlled examples. The declared broader validation criteria **failed**. The historical BPDR Transformer remains a separate experimental option. This notebook does not claim that ExtraTrees is a proven replacement for the research model, that every voice can be repaired, or that 46.6% error reduction is general singing accuracy.

    The most useful distinction is between **a model predicting a correction**, **a controller allowing an edit**, **a rendered WAV passing numerical checks**, and **a person confirming the intended musical idea**. Those are four different decisions.

    ## Reading route

    | Sections | What you will understand |
    |---|---|
    | [1](#s01)-[3](#s03) | Goal, terminology, research gap and model-change history |
    | [4](#s04)-[7](#s07) | Original data, integrity checks, cleaning, splits, usable coverage and targets |
    | [8](#s08)-[10](#s10) | All 31 features, learning algorithms, training rows, weights and saved fit |
    | [11](#s11)-[13](#s13) | Metric definitions, each model's accuracy on common examples, original selection and failed criteria |
    | [14](#s14)-[17](#s17) | All delivered results, fresh inference, actual tree decisions, waveform audit and playable output |
    | [18](#s18)-[20](#s20) | Export, actual downstream MIDI/CSV, user flow, reproducibility and remaining work |

    **Execution:** choose the component's **BPDR local Python** kernel and Run All. Required originals, weights and recorded experiments must already exist locally. This notebook makes no downloads, fits no models, constructs no new training interventions, and changes no model-selection criteria. It reads the completed training evidence and runs one short saved-weight inference. Large assets and executed outputs remain outside Git; notebook source is versioned.
    ''')

    md('''
    <a id="s01"></a>
    ## 1. Start with the actual product goal

    The component should **preserve a usable vocal idea, make only justified improvements, and request another recording when the musical information cannot be recovered reliably**.

    A useful input can be ordinary singing, humming, repeated notes, chant, or speech-like singing with recoverable melodic information. Professional technique and conformity to an assumed C/D scale are not requirements. Ordinary reading without an intended tune does not tell us the melody the person wanted; the interface asks for a sung or hummed take instead of inventing that intention.

    The current model solves a narrower measurable task: predict a small local correction inside sufficiently stable sustained vocals. A clean performance and a known artificial detuning give us a recovery target. They do not label every naturally sung note as correct or wrong.

    We separate technical file acceptance, provisional melodic evidence, optional preparation, and selected-waveform handoff. A blocked correction can still leave a usable original.
    ''')
    code('''
    import os, sys, json, hashlib, zlib, inspect
    from pathlib import Path
    from collections import Counter
    from dataclasses import asdict
    from importlib.metadata import version
    start = Path(os.environ.get('BPDR_COMPONENT_ROOT', Path.cwd())).resolve()
    candidates = [start, *start.parents, start/'experiments/bpdr_input', start/'J26-DS-339/experiments/bpdr_input']
    COMPONENT = next((p for p in candidates if p.name == 'bpdr_input' and (p/'pyproject.toml').is_file()), None)
    if COMPONENT is None:
        raise RuntimeError('Open this notebook inside BPDR_Workspace with the BPDR local Python kernel.')
    WORKSPACE = COMPONENT/'BPDR_Workspace'
    sys.path.insert(0, str(WORKSPACE/'source'))
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import soundfile as sf
    import torch
    import nbformat
    from IPython.display import display, Markdown, HTML, Audio, Code
    torch.set_num_threads(2)
    plt.rcParams.update({'figure.dpi': 115, 'axes.spines.top': False, 'axes.spines.right': False})
    COLORS = {'input':'#9b645e', 'output':'#357985', 'neutral':'#6a7280'}

    def read_json(path):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f'Required local evidence is missing: {path}. Restore it; this notebook does not download or retrain.')
        return json.loads(path.read_text(encoding='utf-8'))

    def sha(path):
        with Path(path).open('rb') as stream:
            return hashlib.file_digest(stream, 'sha256').hexdigest()

    def source_code(obj):
        display(Code(inspect.getsource(obj), language='python'))

    DATASET = WORKSPACE/'datasets/vocalset_pilot'
    MANIFEST = read_json(DATASET/'manifest.json')
    DATA_DECLARATION = read_json(DATASET/'declaration.json')
    RUN = Path(read_json(WORKSPACE/'results/model_recovery/current.json')['path']).resolve()
    assert RUN.is_relative_to((WORKSPACE/'results/model_recovery').resolve())
    EVIDENCE = read_json(RUN/'evidence.json')
    DECLARATION = read_json(RUN/'declaration.json')
    LOCK = read_json(RUN/'locked_selection.json')
    CARD = read_json(WORKSPACE/'results/model_recovery/demo_model.json')
    CHECKPOINT = Path(EVIDENCE['checkpoint'])
    assert sha(CHECKPOINT) == EVIDENCE['checkpoint_sha256'] == CARD['checkpoint_sha256']
    assert sha(RUN/'declaration.json') == EVIDENCE['declaration_sha256']
    assert sha(DATASET/'manifest.json') == DECLARATION['manifest_sha256']
    print('Python:', sys.executable)
    print('Workspace:', WORKSPACE)
    print('Recorded run:', RUN.name)
    display(pd.DataFrame([{'package': p, 'current_verification_version': version(p)} for p in
        ['numpy','scipy','pandas','torch','scikit-learn','joblib','librosa','pesto-pitch','praat-parselmouth','nbformat']]))
    print('These are current verification-environment versions; a complete training-time package freeze was not stored in this run.')
    ''')
    md('''
    <a id="s02"></a>
    ## 2. Terms and the complete processing route

    | Term | Meaning here |
    |---|---|
    | F0 / performed pitch | The estimated fundamental frequency of the recording, not the singer's private intention |
    | Cent | One hundredth of a semitone; 100 cents is one semitone |
    | Frame | One feature observation every 20 milliseconds; many frames from one singer are dependent |
    | Original / working copy | Immutable downloaded WAV / a separately resampled 22,050 Hz derivative |
    | Intervention | A deliberately added known local pitch shift used to create measurable recovery labels |
    | Target | Desired reverse correction on the audited plateau; zero on eligible unchanged regions |
    | Calibration | Separate singers used to choose the model and operating settings; not probability calibration |
    | Gain / deadband | Multiply a predicted correction / withhold predictions smaller than a threshold |
    | MAE | Mean absolute error in cents; lower is better |
    | Withheld / rollback | Do not render an unsupported proposal / discard a failed rendered candidate |
    | Generalization | Working on people and musical material absent from fitting, with trustworthy labels |

    Corrections are signed. If a training intervention raises a region by +30 cents, its intended recovery target is -30 cents. A zero target means preserve that performance. Converting PCM16 to PCM24 does not add information or make the singing better.
    ''')
    code('''
    frequency = 440.
    shifted = frequency * 2**(30/1200)
    recovered = shifted * 2**(-30/1200)
    display(pd.DataFrame([{'step':'Illustrative unshifted frequency','Hz':frequency},
        {'step':'After +30 cents','Hz':shifted},{'step':'After -30-cent correction','Hz':recovered}]))
    assert np.isclose(recovered, frequency)
    fig, ax = plt.subplots(figsize=(12,3.3)); ax.set_axis_off()
    stages = ['Preserve original','Check file + vocal evidence','Measure pitch + context',
              'Predict signed correction','Permission + render + audit','Review / export / extractor']
    for i, label in enumerate(stages):
        x = .09 + (i%3)*.40; y = .77 if i<3 else .20
        ax.text(x,y,label,ha='center',va='center',fontsize=10,
                bbox=dict(boxstyle='round,pad=.6',facecolor='#eef4f3',edgecolor=COLORS['output']),transform=ax.transAxes)
    for a,b in [((.18,.77),(.36,.77)),((.59,.77),(.75,.77)),((.89,.64),(.09,.34)),((.18,.20),(.36,.20)),((.59,.20),(.75,.20))]:
        ax.annotate('',xy=b,xytext=a,xycoords='axes fraction',arrowprops=dict(arrowstyle='->',color='#59666b'))
    ax.set_title('Training happens separately; a user upload runs saved-weight inference only')
    plt.tight_layout(); plt.show()
    ''')
    md('''
    <a id="s03"></a>
    ## 3. Why try a different model? What happened to the Transformer?

    The original BPDR research question is whether pitch-repair predictions can respond to real pitch changes while remaining stable when only provisional note-boundary metadata changes. Its four views are A/B (same original audio, different boundaries) and C/D (same detuned audio, different boundaries). The historical Transformer implements the proposed sequence model and paired losses.

    Historical experiments produced controlled generated-voice demonstrations, but did not establish reliable repair of natural real singing. In the saved adapted BPDR Transformer run, 304 real-source development variants had mean correction error **20.318 cents**, compared with **20.329 cents** for zero correction, and median paired response gain about **0.00021** (ideal response 1). That setup largely preserved audio without responding usefully to the known shifts. The code below verifies the release/model/evidence chain and reproduces its per-dataset summaries from saved rows.

    Early pitch quantization also obscured small shifts. Fine YIN features improved observability; later setups still did not establish reliable transfer. Short reliable regions and the renderer's minimum support further limited the earlier real-data pilot. These are observed failures of specific setups, not proof that a Transformer can never work.

    **Why the demo model changed:** the saved Transformer predictions barely responded to known pitch changes, so they supplied little useful repair beyond retaining the input. We therefore evaluated explicit local pitch-context features with ExtraTrees and histogram gradient boosting for the stable-vocal recovery task. ExtraTrees ranked highest under the declared calibration rule and later produced two measured delivered-waveform improvements. The reason for choosing the experimental demo path is measured correction behavior and preservation checks, not a claim about dataset size. Fast fitting and inspectable trees help implementation, but they are not accuracy evidence.

    **Important research boundary:** the current ExtraTrees input excludes boundary metadata. Zero response to changed boundary metadata is therefore expected by construction; it is not proof of learned BPDR disentanglement. Section [12B](#model-accuracy-comparison) now evaluates all four checkpoints on the same VocalSet calibration examples. Their training histories and feature interfaces differ, so this is a comparison of available systems, not a matched architecture-training experiment or an independent final test. The historical Transformer remains available as an experimental option.

    The table below checks actual saved identities. The earlier compact neural pilot is an MLP, not another Transformer.
    ''')
    code('''
    from bpdr_input.reporting.bpdr_showcase import load_model as load_transformer
    transformer_card = read_json(WORKSPACE/'selected_model/model.json')
    transformer_path = WORKSPACE/'selected_model'/transformer_card['checkpoint']
    assert sha(transformer_path) == transformer_card['sha256']
    historical_transformer, _, transformer_width = load_transformer(transformer_path)
    transformer_description = historical_transformer.describe()
    small_run = Path(read_json(WORKSPACE/'results/small_real_pilot/current.json')['path'])
    small_evidence = read_json(small_run/'evidence.json')
    historical_rows = []
    for condition, row in small_evidence['results'].items():
        for dataset, result in row['development']['by_dataset'].items():
            historical_rows.append(dict(model='Earlier LocalContextMLP',condition=condition,dataset=dataset,
                correction_MAE_cents=result['correction_mae'],unchanged_baseline_cents=result['zero_mae'],
                original_prediction_abs_cents=result['unshifted_alteration'],criteria_passed=result['passes']))
    display(pd.DataFrame([dict(model='Historical BPDR Transformer', **transformer_description)]))
    historical_index=read_json(WORKSPACE/'notebooks/evidence_index.json')
    historical_manifest_path=COMPONENT/historical_index['evidence']['presentation_manifest']
    historical_manifest=read_json(historical_manifest_path)
    assert sha(historical_manifest_path)==historical_index['current_gate']['presentation_manifest_sha256']
    historical_release_card_path=historical_manifest_path.parent/'model_card.json'
    assert sha(historical_release_card_path)==historical_manifest['model_card_sha256']
    assert read_json(historical_release_card_path)['checkpoint_sha256']==sha(transformer_path)
    adapted_path=COMPONENT/historical_index['evidence']['presentation_adapt']
    adapted=read_json(adapted_path)
    assert sha(adapted_path) in historical_manifest['source_hashes'].values()
    adapted_rows=read_json(adapted_path.parent/'bpdr/rows.json')
    assert len(adapted_rows)==adapted['results']['bpdr']['all']['records']==304
    transformer_results=[]
    for dataset,result in adapted['results']['bpdr']['by_dataset'].items():
        rows=[r for r in adapted_rows if r['dataset']==dataset]
        for metric in ['correction_mae','zero_mae','unshifted_alteration']:
            np.testing.assert_allclose(np.mean([r[metric] for r in rows]),result[metric],atol=1e-6)
        np.testing.assert_allclose(np.median([r['response_gain'] for r in rows]),result['median_response_gain'])
        transformer_results.append(dict(model='Historical adapted BPDR Transformer',dataset=dataset,variants=result['records'],
            raw_correction_MAE=result['correction_mae'],zero_baseline_MAE=result['zero_mae'],
            median_response_gain=result['median_response_gain'],original_abs_proposal=result['unshifted_alteration']))
    display(pd.DataFrame(transformer_results).round(6))
    print('Historical adapted Transformer criteria passed:',adapted['results']['bpdr']['engineering_gate_passes'])
    print('Earlier compact MLP pilot, shown separately:')
    display(pd.DataFrame(historical_rows).round(4))
    print('Transformer training:', transformer_card['training'])
    print('Historical research adoption:', transformer_card['research_model_adopted'])
    print('MLP eligible conditions:', small_evidence['eligible_conditions'])
    print('These historical datasets/endpoints differ from the revised VocalSet pilot. Do not rank them in one accuracy leaderboard.')
    del historical_transformer
    ''')
    md('''
    <a id="s04"></a>
    ## 4. Where the data came from and exactly what was selected

    **VocalSet 1.0**, Julia Wilkins, Prem Seetharaman, Alison Wahl and Bryan Pardo (2018), is the source of the new human recordings. Source and license: [official record](https://zenodo.org/records/1193957), [DOI](https://doi.org/10.5281/zenodo.1193957), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

    Before acoustic analysis, the bounded download selected vowels **a/e**, straight long-tone exercises, female1-female6 and male1-male6. IDs 1-4 were assigned to fitting, 5 to calibration, and 6 to the pilot, within each named source group. These are source singer IDs, not inferred demographic or performance-quality labels. No entire dataset was downloaded or used for fitting.

    The full archive is about 2 GB. HTTP range extraction downloaded selected entries, verified entry CRC32 and saved SHA-256 and decoded metadata. The full-archive MD5 is a declared source value and was **not verified**, because the full archive was not downloaded. Local hashes detect later changes; they do not establish musical correctness. All 24 originals remain immutable, even when later preparation fails.

    Sustained professional exercises make this a narrow controlled-recovery dataset. They do not represent free novice melodies, ordinary reading, natural intended-note errors, accompaniment or overlapping singers.
    ''')
    code('''
    assert MANIFEST['status'] == 'COMPLETE'
    assert sha(DATASET/'declaration.json') == MANIFEST['declaration_sha256']
    assert DATA_DECLARATION['license'] == 'CC-BY-4.0'
    assert not DATA_DECLARATION['full_archive_md5_verified']
    source_rows=[]
    for r in MANIFEST['records']:
        path=Path(r['path']).resolve()
        assert path.is_relative_to(DATASET.resolve())
        assert sha(path)==r['sha256'] and path.stat().st_size==r['bytes']
        assert format(zlib.crc32(path.read_bytes()) & 0xffffffff,'08x')==r['archive_crc32']
        assert r['archive_crc32_verified'] and not r['intervention_applied']
        info=sf.info(path)
        assert (info.frames,info.samplerate,info.channels)==(r['sample_count'],r['sample_rate_hz'],r['channels'])
        audio,rate=sf.read(path,dtype='float32')
        source_rows.append(dict(split=r['split'],singer=r['singer_id'],vowel=r['vowel'],
            source_file=path.name,duration_seconds=info.duration,rate_Hz=rate,channels=info.channels,
            finite=bool(np.isfinite(audio).all()),peak=float(np.max(np.abs(audio))),
            RMS=float(np.sqrt(np.mean(audio.astype(np.float64)**2))),
            clipped_fraction=float(np.mean(np.abs(audio)>=.999)),sha256=r['sha256']))
    SOURCES=pd.DataFrame(source_rows)
    assert len(SOURCES)==24 and SOURCES.singer.nunique()==12 and SOURCES.sha256.nunique()==24
    display(SOURCES.drop(columns='sha256').round(5))
    print((DATASET/'ATTRIBUTION.txt').read_text(encoding='utf-8'))
    print('Verified 24 source hashes, 24 archive-entry CRCs, decoded metadata and source uniqueness.')
    ''')
    md('''
    <a id="s05"></a>
    ## 5. Data cleaning and working copies: what changed and what did not

    | Step | Actual behavior and purpose |
    |---|---|
    | Integrity | Reject changed source bytes; preserve originals |
    | Decode | Read finite samples; retain strongest channel if multiple exist; all selected originals are mono |
    | Duration | Revised training preparation needs at least 5 seconds and at most 20; all selected files already fit the upper limit |
    | Resample | 44,100 Hz to 22,050 Hz with `resample_poly`; keep a separate full-length working copy |
    | Level | Reject peak clipping during training preparation; no normalization rescue, silence trimming or denoising |
    | Encode | Working copies use WAV PCM24; increasing bit depth does not recover missing information |
    | Pitch evidence | Frozen PESTO confidence >=0.8, finite YIN pitch, and agreement within 50 cents determine reliable observations |
    | Context eligibility | To construct a controlled intervention, require a measured original stable voiced run >=1.36 seconds and detrended spread <=20 cents |
    | Failed context | Keep the original and all four excluded signed attempt records; do not manufacture labels |

    **Technical runtime limits differ from training-preparation limits:** a user's phrase may be 0.25-20 seconds. Runtime also checks silence and excessive clipping. A file passing those technical checks is not automatically musically usable. These checks are separate from the stable-note eligibility required for this particular pitch model.

    The next cell checks saved working copies. Original and resampled samples are not expected to be byte-identical. The correct invariants are unchanged source hashes, known processing, finite mono data, correct sample rate, retained duration and separate working files.
    ''')
    code('''
    SPLITS=['train','calibration','pilot_evaluation']
    SOURCE_RECORDS={s:read_json(RUN/(s+'_sources.json')) for s in SPLITS}
    PAIRS={s:read_json(RUN/(s+'_pairs.json')) for s in SPLITS}
    working_rows=[]
    for split in SPLITS:
        for r in SOURCE_RECORDS[split]:
            if 'working_audio' not in r:
                working_rows.append(dict(split=split,singer=r['source']['singer_id'],exclusion=r.get('excluded')))
                continue
            path=Path(r['working_audio']); assert sha(path)==r['working_sha256']
            audio,sr=sf.read(path,dtype='float32'); original_info=sf.info(r['source']['path'])
            assert sr==22050 and audio.ndim==1 and np.isfinite(audio).all() and np.max(np.abs(audio))<1
            assert r['crop_offset_seconds']==0
            assert abs(len(audio)/sr-original_info.duration)<=1/sr+1e-8
            working_rows.append(dict(split=split,singer=r['source']['singer_id'],vowel=r['source']['vowel'],
                working_rate_Hz=sr,working_seconds=len(audio)/sr,source_seconds=original_info.duration,
                subtype=sf.info(path).subtype,context_exclusion=r.get('excluded','none')))
    WORKING=pd.DataFrame(working_rows)
    display(WORKING.round(6))
    assert SOURCES.finite.all()
    fig,axes=plt.subplots(1,2,figsize=(12,3.6))
    axes[0].bar(np.arange(len(SOURCES)),SOURCES.duration_seconds,color=COLORS['output'])
    axes[0].set(xlabel='Source recording index',ylabel='Seconds',title='Every original remains within the 20-second runtime limit')
    axes[0].axhline(20,linestyle='--',color=COLORS['input'])
    axes[1].scatter(SOURCES.RMS,SOURCES.peak,color=COLORS['output'])
    axes[1].set(xlabel='Original RMS amplitude',ylabel='Original peak amplitude',title='Source levels are observed, not normalized into a pass')
    axes[1].set_ylim(0,1.02);plt.tight_layout();plt.show()
    ''')
    md('''
    <a id="s06"></a>
    ## 6. Splits and usable coverage: people, recordings and frames are different

    Every recording and all its modified versions stay with its singer's split. This avoids fitting one vowel from a singer and evaluating another vowel from the same singer as if it were independent. The split was declared before acoustic screening.

    Assigned training contained **16 recordings / 8 singers**. Valid training pairs actually came from **13 recordings / 7 singers**. The other originals remain recorded exclusions. Likewise, both pilot singers were attempted, but only one supplied valid controlled-shift labels.

    The original pilot singer IDs were inspected during the first failed placement attempt, then reused after a preparation correction. They remain absent from model fitting, but these results are **development evidence, not an untouched final test**. Frames and signed shifts increase observations, not the number of independent people.
    ''')
    code('''
    groups={s:set(SOURCES.loc[SOURCES.split==s,'singer']) for s in SPLITS}
    for i,a in enumerate(SPLITS):
        for b in SPLITS[i+1:]:
            assert not groups[a]&groups[b]
            assert not set(SOURCES.loc[SOURCES.split==a,'sha256'])&set(SOURCES.loc[SOURCES.split==b,'sha256'])
    coverage=[];excluded_rows=[]
    for split in SPLITS:
        pairs=PAIRS[split]; valid=[p for p in pairs if 'excluded' not in p]
        assert len(pairs)==4*len(SOURCE_RECORDS[split])
        keys=[(p['source']['sha256'],p['cents']) for p in pairs]
        assert len(set(keys))==len(keys)
        coverage.append(dict(split=split,assigned_recordings=len(SOURCE_RECORDS[split]),assigned_singers=len(groups[split]),
            recordings_with_valid_pairs=len({p['source']['sha256'] for p in valid}),
            singers_with_valid_pairs=len({p['source']['singer_id'] for p in valid}),
            signed_attempts=len(pairs),audited_pairs=len(valid),excluded_attempts=len(pairs)-len(valid)))
        for reason,count in Counter(p['excluded'] for p in pairs if 'excluded' in p).items():
            excluded_rows.append(dict(split=split,reason=reason,attempts=count))
    COVERAGE=pd.DataFrame(coverage).set_index('split').loc[SPLITS]
    display(COVERAGE);display(pd.DataFrame(excluded_rows))
    assert (COVERAGE.loc['train',['recordings_with_valid_pairs','singers_with_valid_pairs','audited_pairs']].to_numpy()==[13,7,43]).all()
    assert COVERAGE.loc['pilot_evaluation','audited_pairs']==4
    fig,ax=plt.subplots(figsize=(9,3.7));x=np.arange(3)
    ax.bar(x,COVERAGE.audited_pairs,label='Audited, usable labels',color=COLORS['output'])
    ax.bar(x,COVERAGE.excluded_attempts,bottom=COVERAGE.audited_pairs,label='Retained exclusions',color=COLORS['input'])
    ax.set_xticks(x,SPLITS);ax.set(ylabel='Signed intervention attempts',title='Coverage before model performance is measured')
    ax.legend();plt.tight_layout();plt.show()
    ''')
    md('''
    <a id="s07"></a>
    ## 7. Where the labels came from, and why the first preparation failed

    There are no trustworthy natural intended-note labels in this subset. To make a measurable engineering task, we start from a real recording A and add a known local -30/-20/+20/+30-cent shift, producing C. A's target is zero; C's recovery target is the negative of the known shift on audited plateau frames. Eligible unchanged frames have zero target. Ramps and analysis guards are excluded from supervision.

    **First attempt:** an eight-second center crop and randomly placed 1.15-1.35-second interventions sometimes put shifts into pauses or outside stable voice. That run remains saved and failed.

    **Revised attempt:** retain the full source; find the longest original reliable stable run; place a hash-varied 0.96-1.12-second intervention with at least 0.20-second measured flanks. The original is consulted to construct supervised examples, which is legitimate. Its support, clean pitch, singer identity and target are never supplied to inference.

    The corruption renderer uses librosa pitch shifting with local crossfades. Its realized movement is independently measured with YIN, with coarse PESTO consistency, coverage and outside-drift checks. Passing known-shift creation is required before a target can be used. Failed examples are retained, not relabelled as successes. This construction audit is separate from auditing a model's later repaired output.

    Using a known corruption does not prove that the model detects natural mistakes rather than properties of that corruption process. Natural errors, another intervention renderer and independent listening remain important future checks.
    ''')
    code('''
    FIRST_RUN=WORKSPACE/'results/model_recovery/14a10a583e40497ea2514fd3570a7f79'
    first_evidence=read_json(FIRST_RUN/'evidence.json')
    first_pairs=read_json(FIRST_RUN/'train_pairs.json')
    comparison=pd.DataFrame([
        dict(attempt='First: arbitrary crop positions',train_attempts=len(first_pairs),
             train_audited=sum('excluded' not in p for p in first_pairs),calibration_passed=first_evidence['selection']['calibration_gate_passes']),
        dict(attempt='Revised: measured stable voice',train_attempts=len(PAIRS['train']),
             train_audited=int(COVERAGE.loc['train','audited_pairs']),calibration_passed=LOCK['calibration_gate_passes'])])
    display(comparison)
    TRAIN_PAIR=next(p for p in PAIRS['train'] if 'excluded' not in p)
    with np.load(TRAIN_PAIR['target_path']) as z:
        train_times=z['times'];train_mask=z['mask'].astype(bool);train_target=z['target']
    TRAIN_A,sr=sf.read(TRAIN_PAIR['path_A'],dtype='float32')
    TRAIN_C,_=sf.read(TRAIN_PAIR['path_C'],dtype='float32')
    assert sr==22050 and TRAIN_A.shape==TRAIN_C.shape
    from bpdr_input.repair.rule_correction import fine_pitch
    train_pitch_a=fine_pitch(TRAIN_A,sr,train_times);train_pitch_c=fine_pitch(TRAIN_C,sr,train_times)
    fig,axes=plt.subplots(2,1,figsize=(12,5),sharex=True)
    axes[0].plot(train_times,train_pitch_c-train_pitch_a,color=COLORS['input'],label='Measured C minus A')
    axes[0].axvspan(*TRAIN_PAIR['support_seconds'],alpha=.15,color=COLORS['output'])
    axes[0].set(ylabel='Added pitch change (cents)',title='One audited training intervention on a real voice');axes[0].legend()
    axes[1].plot(train_times[train_mask],train_target[train_mask],'.',color=COLORS['output'])
    axes[1].set(xlabel='Seconds',ylabel='Recovery target (cents)',title='Fixed audited labels: reverse shift on plateau, zero outside; gaps/guards excluded')
    plt.tight_layout();plt.show()
    display(pd.DataFrame([{'injected_cents':TRAIN_PAIR['cents'],'plateau_target_cents':float(np.median(train_target[train_mask & (train_target!=0)])),
        'scored_plateau_frames':int((train_mask & (train_target!=0)).sum()),'unchanged_supervision_frames':int((train_mask & (train_target==0)).sum())}]))
    assert TRAIN_PAIR['audit']['passed']
    display(Markdown('**Original real training voice:**'));display(Audio(filename=TRAIN_PAIR['path_A'],normalize=False))
    display(Markdown('**Same voice with known training corruption:**'));display(Audio(filename=TRAIN_PAIR['path_C'],normalize=False))
    ''')

    md('''
    <a id="s08"></a>
    ## 8. Audio features: exactly what the model sees

    The acoustic analyzer first creates 64 log-mel measurements plus performed pitch, confidence, voicing, pitch change, energy and onset: 70 columns, or 71 with a provisional boundary score. PESTO is frozen; it is not retrained in this pilot. Independent YIN supplies fine pitch. Runtime rejects pitch observations when trackers disagree by more than 50 cents, then centers pitch relative to the current phrase.

    **The tree model does not consume all 71 columns.** `context_features(..., include_boundary=False)` derives **31** values from this input's own pitch, confidence, voicing, energy and onset. It does not consume the 64 log-mel columns directly. Local windows are 11/31/61/121 frames (0.22/0.62/1.22/2.42 seconds). Future as well as past frames are used, so this is phrase-level processing, not a demonstrated zero-latency streaming model.

    The circular chromatic residual uses all twelve semitones with the current performance's tuning offset. It is supporting context, not a C-major/D-major prediction or intended-note reference. Missing pitch is interpolated only for context statistics; explicit voicing and coverage features remain, and unreliable output edits are withheld. A4 equals zero cents and must not be mistaken for silence.

    This cell prints every feature in model order and checks the actual 31-column matrix. Three timing-only boundary views produce identical tree features because boundary metadata is deliberately excluded. That identity is not learned boundary robustness.
    ''')
    code('''
    from bpdr_input.modeling.local_context_pilot import context_features, WINDOW_FRAMES
    from bpdr_input.repair.model_runtime import runtime_views
    with np.load(Path(TRAIN_PAIR['path_A']).parent/'original_features.npz') as z:
        TRAIN_FEATURES={k:z[k] for k in z.files}
    raw_views,raw_reliable,raw_protected,adapter_metadata=runtime_views(TRAIN_FEATURES,train_pitch_a)
    feature_rows=[
        ('phrase_relative_pitch','cents / 1200, clipped','Position relative to this phrase median'),
        ('chromatic_residual','cents / 30','Deviation from this performance circular semitone grid'),
        ('residual_sine','unitless','Circular encoding of the residual'),
        ('residual_cosine','unitless','Circular encoding without a discontinuity at wraparound'),
        ('pitch_confidence','0-1','Frozen tracker confidence; not probability of singing correctly'),
        ('voiced','0 or 1','A reliable performed pitch is present'),
        ('pitch_delta','cents / 30, clipped','Neighboring fine-pitch change'),
        ('relative_energy','dB / 20, clipped','Energy relative to this phrase'),
        ('onset','0-1','Measured onset score')]
    for width in WINDOW_FRAMES:
        seconds=width*.02
        feature_rows.extend([
            (f'pitch_minus_median_{width}', 'cents / 30, clipped',f'Pitch deviation from {seconds:.2f}s local median'),
            (f'residual_minus_median_{width}','cents / 30',f'Circular residual deviation in {seconds:.2f}s'),
            (f'residual_mean_{width}','cents / 30',f'Mean local residual in {seconds:.2f}s'),
            (f'residual_spread_{width}','cents / 30',f'Local residual variation in {seconds:.2f}s'),
            (f'voiced_coverage_{width}','0-1',f'Fraction of voiced observations in {seconds:.2f}s')])
    feature_rows += [('endpoint_contour_residual','cents / 30, clipped','Difference from an own-audio endpoint contour'),
                     ('absolute_pitch_delta','abs(cents) / 100, clipped','Magnitude of neighboring pitch movement')]
    FEATURE_TABLE=pd.DataFrame(feature_rows,columns=['feature','scale','meaning'])
    FEATURE_TABLE.index.name='model_column';FEATURE_NAMES=FEATURE_TABLE.feature.tolist()
    X_EXAMPLE=context_features(raw_views[0],include_boundary=False)
    assert X_EXAMPLE.shape==(len(train_times),31) and np.isfinite(X_EXAMPLE).all()
    for alternative in raw_views[1:]:
        np.testing.assert_array_equal(X_EXAMPLE,context_features(alternative,include_boundary=False))
    display(FEATURE_TABLE)
    display(pd.DataFrame(X_EXAMPLE[:5],columns=FEATURE_NAMES).round(4))
    print('Actual source of the 31-column interface:')
    source_code(context_features)
    ''')
    md('''
    <a id="s09"></a>
    ## 9. How the candidate models learn

    | Candidate | What was fitted | Purpose and limitations |
    |---|---|---|
    | ExtraTrees Regressor | 96 randomized decision trees, maximum depth 8, minimum 12 samples per leaf; weighted squared error | Fast bounded regression over the 31 context features |
    | Histogram Gradient Boosting | Up to 120 iterations, 15 leaf nodes, depth 5, learning rate 0.08; fixed settings | Another bounded tree-based alternative on the same new training rows |
    | Earlier LocalContextMLP | Previously trained checkpoint, frozen here | Diagnostic comparator trained on different data; not a newly matched retraining |
    | Historical BPDR Transformer | Preserved previous checkpoint | Not fitted or compared head-to-head on this new split |

    A decision tree asks successive questions about feature values, then returns the correction learned in its leaf. ExtraTrees averages the predictions of its 96 trees. In this fitted model `bootstrap=False`, so the training rows are not bootstrap-resampled; randomized split thresholds provide tree variation. Source: [official ExtraTrees documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html).

    Gradient boosting adds trees sequentially, using histogram-based training; it is a separate algorithm, not a Transformer. Source: [official histogram gradient boosting documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html).

    The regression target is **signed cents**, not a binary correct/wrong class. ExtraTrees training minimizes weighted squared error; we evaluate using absolute errors, response, preservation and coverage. There is no meaningful classification confusion matrix or universal accuracy percentage for this target. Tree count is not an epoch count. A training-fit score alone cannot establish unseen-singer performance.
    ''')
    code('''
    from bpdr_input.modeling.recovery_model import load_locked_diagnostic
    MODEL,PAYLOAD=load_locked_diagnostic(CHECKPOINT)
    ESTIMATOR=MODEL['estimator']
    assert MODEL['kind']=='extra_trees' and ESTIMATOR.n_estimators==96 and ESTIMATOR.n_features_in_==31
    params=ESTIMATOR.get_params()
    display(pd.DataFrame([{'parameter':k,'saved_model_value':params[k]} for k in
        ['n_estimators','max_depth','min_samples_leaf','criterion','bootstrap','max_features','random_state','n_jobs']]))
    display(pd.DataFrame([dict(model=name,**settings) for name,settings in DECLARATION['models'].items()]))
    print('Saved model type:',PAYLOAD['model_type'])
    print('Saved operating settings:',PAYLOAD['operating_policy'])
    print('Checkpoint SHA-256:',sha(CHECKPOINT))
    print('Loading this diagnostic checkpoint does not assert that research gates passed.')
    ''')
    md('''
    <a id="s10"></a>
    ## 10. Actual training rows, weights, settings and fit receipts

    Training uses only audited training pairs. Each contributes three supervised regions:

    | Region | Target | Share of its singer-balanced pair weight |
    |---|---|---|
    | Original A, eligible mask | Zero | 0.50 |
    | Detuned C, audited plateau | Reverse the known shift | 0.35 |
    | Detuned C, eligible outside region | Zero | 0.15 |

    Each represented training singer receives equal total weight; its weight is divided among its valid pairs and then among frames in each region. This reduces dominance by longer recordings, singers with more passing pairs and numerous unchanged frames. The same original A can occur in multiple paired rows. Those are repeated supervised observations, not new independent recordings.

    The saved fit used **16,842 frame rows**, derived from **43 valid pairs, 13 original recordings and 7 singers**. Fixed settings and seed 339 were declared. No validation or pilot singer was used to fit these new tree estimators. Feature scaling is fixed in code rather than learned from evaluation data.

    Two distinct new model configurations were tried. A duplicate-key reporting error interrupted the first summary after fitting; the same two deterministic configurations were fitted again from cached prepared data. No additional hyperparameter search or new labels were introduced by that repair. The technical receipt is displayed below.

    The following cell reconstructs training arrays from already saved features and labels **without fitting**. It checks row counts, finite values and weights, then computes a descriptive training-fit MAE for the saved selected model. That newly recomputed training score is not used for selection and is not validation accuracy.
    ''')
    code('''
    from bpdr_input.experiments.learning import model_recovery as recovery_experiment
    cached_train=recovery_experiment.cached_pairs(PAIRS['train'])
    X_TRAIN,Y_TRAIN,W_TRAIN=recovery_experiment.training_arrays(cached_train)
    assert X_TRAIN.shape==(16842,31) and len(Y_TRAIN)==len(W_TRAIN)==16842
    assert np.isfinite(X_TRAIN).all() and np.isfinite(Y_TRAIN).all() and np.isfinite(W_TRAIN).all()
    assert (W_TRAIN>0).all() and np.isclose(W_TRAIN.sum(),1.)
    label_rows=[]
    for value in np.unique(Y_TRAIN):
        match=Y_TRAIN==value
        label_rows.append(dict(target_cents=float(value),frame_rows=int(match.sum()),training_weight=float(W_TRAIN[match].sum())))
    display(pd.DataFrame(label_rows))
    display(pd.DataFrame(EVIDENCE['fit_receipts']).T)
    print('Mechanical resume receipt:',read_json(RUN/'technical_resume.json'))
    expected_sources={Path(p).name:digest for p,digest in DECLARATION['source_hashes'].items()}
    snapshot_rows=[]
    for original,snapshot in [('model_recovery.py','executed_model_recovery.py'),('recovery_model.py','executed_recovery_model.py')]:
        matched=sha(RUN/snapshot)==expected_sources[original]
        assert matched
        snapshot_rows.append(dict(original_module=original,frozen_executing_snapshot=snapshot,declared_hash_matches=matched))
    assert sha(WORKSPACE/'source/bpdr_input/modeling/local_context_pilot.py')==expected_sources['local_context_pilot.py']
    assert sha(RUN/'technical_resume_model_recovery.py')==read_json(RUN/'technical_resume.json')['new_source_sha256']
    display(pd.DataFrame(snapshot_rows))
    print('The current 31-feature generator matches its training declaration. Frozen execution snapshots and the mechanical-resume source are hash-verified; later loader safeguards do not rewrite those snapshots.')
    training_prediction=ESTIMATOR.predict(X_TRAIN)
    training_fit_mae=float(np.average(np.abs(training_prediction-Y_TRAIN),weights=W_TRAIN))
    training_zero_mae=float(np.average(np.abs(Y_TRAIN),weights=W_TRAIN))
    display(pd.DataFrame([dict(stage='Recomputed weighted TRAINING diagnostic only',
        unchanged_baseline_MAE_cents=training_zero_mae,selected_model_MAE_cents=training_fit_mae)]).round(4))
    print('Actual row/weight assembly function (displayed, not fitted):')
    source_code(recovery_experiment.training_arrays)
    run_source=inspect.getsource(recovery_experiment.run)
    fit_fragment=run_source[run_source.index('    with threadpool_limits'):run_source.index('    previous =')]
    display(Code(fit_fragment,language='python'))
    print('The displayed .fit() code belongs to the completed experiment. It is NOT executed by this notebook.')
    ''')
    md('''
    <a id="s11"></a>
    ## 11. What does "accuracy" mean for this project?

    For a fixed set of scored frames, MAE is `mean(abs(prediction - target))`, in cents. Lower is better. For a delivered waveform, we instead compare measured output pitch against the unshifted real reference on the same fixed frames. Source: [official MAE definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html).

    | Measurement | Definition | What it can establish |
    |---|---|---|
    | Zero / unchanged baseline MAE | Error if no correction is made | A model must do more than predict zero |
    | Raw correction MAE | Predicted cents versus known reverse shift | Learning of the controlled target before permission/rendering |
    | Controlled correction accuracy within 10 cents | Percentage of the fixed shifted-plateau frames with `abs(prediction - target) <= 10` | A plainly defined tolerance score for each model on the same examples; not natural singing correctness |
    | Permitted correction MAE | Error of allowed curve; zero curve when withheld | Whether operating settings permit useful proposals; still not delivered WAV quality |
    | Delivered local MAE | Fine pitch of actual WAV versus unshifted reference | Measured realized recovery on the fixed controlled region |
    | Error reduction | `1 - output_MAE / input_MAE` | Relative improvement on those specific measured cases |
    | Original alteration | Absolute raw/permitted proposals on unchanged A | Unnecessary prediction/edit risk; not the same as exact waveform preservation |
    | Response gain | Mean C-minus-A correction divided by known reverse shift | Whether predictions respond in the intended direction |
    | Coverage | Permitted or verified cases divided by a stated attempted/audited count | How often the model can act; a model that rejects everything is not useful repair |

    **Aggregation matters.** Calibration first averages pair metrics within each singer, then gives each singer equal weight. Its selection metric averages each singer's individual error-reduction ratio. It is not necessarily `1 - aggregate_MAE / aggregate_baseline_MAE`. The development WAV result currently has one scoreable singer and four equally weighted case MAEs. These are not pooled-frame averages.

    All withheld and rolled-back **audited** cases remain in the delivered denominator with the retained input. An intervention that failed construction has no valid recovery label: it is reported as unscorable, not silently removed from attempted coverage or counted as a correct prediction. Output cannot drop difficult scored frames; missing delivered pitch prevents a verified gain.

    A 46.6% reduction in measured error is not 46.6% of people singing correctly. Two improvements in four scoreable examples are a conditional demo count, not a population success estimate.
    ''')
    code('''
    toy_target=np.array([-30.,-30.,0.,0.])
    toy_prediction=np.array([-24.,-27.,0.,2.])
    toy_mae=float(np.abs(toy_prediction-toy_target).mean())
    toy_baseline=float(np.abs(toy_target).mean())
    display(pd.DataFrame(dict(target_cents=toy_target,predicted_cents=toy_prediction,
                             absolute_error=np.abs(toy_prediction-toy_target))))
    print(f'Illustrative MAE: {toy_mae:.2f} cents; unchanged baseline: {toy_baseline:.2f}; error reduction: {100*(1-toy_mae/toy_baseline):.2f}%')
    print('These four toy numbers explain the calculation; they are not project results.')
    ''')
    md('''
    <a id="s12"></a>
    ## 12. Was ExtraTrees chosen because it had better accuracy?

    **It was chosen by the declared calibration ranking as the best diagnostic fallback. It was not selected as a model that had passed validation.**

    The experiment tried gains 0.50/0.75/1.00 and deadbands 3/5/8 cents: nine operating points per model. The two new fits plus the frozen MLP give 27 calibration rows. The rule first selects among complete candidates passing all practical criteria. If none pass, it selects the highest singer-balanced permitted error reduction **for diagnostic use only**. Tie-breakers prefer lower raw original alteration, then the model name. Selection was locked before this run prepared pilot audio.

    None of the 27 rows passed. ExtraTrees at gain 1.0 and deadband 3 cents ranked highest. Its singer-balanced permitted correction gain was about **13.93%**, versus **13.37%** for the best gradient-boosting setting and **11.85%** for the best frozen MLP setting. The margin over gradient boosting is small; no uncertainty estimate or matched independent test establishes a meaningful superiority claim. Raw prediction errors were also lower for ExtraTrees, but raw MAE was not the ranking's primary metric.

    The frozen MLP was trained on other data, and the historical Transformer was not newly trained on this exact split. Consequently this table cannot establish that ExtraTrees is generally more accurate than either neural architecture. Section 12B adds common-example inference for all four models, including the Transformer. It is supplementary evidence collected after selection; the later listening example and this new comparison did not drive the already locked model choice.
    ''')
    code('''
    CALIBRATION=read_json(RUN/'calibration_all.json')
    assert len(CALIBRATION)==27 and not any(r['summary']['passes'] for r in CALIBRATION)
    rank_key=lambda r:(r['summary'].get('permitted_correction_gain',-1),
                       -r['summary'].get('raw_original_abs_mean',100),r['model'])
    replayed_choice=max(CALIBRATION,key=rank_key)
    assert replayed_choice['model']==LOCK['model']=='extra_trees'
    assert replayed_choice['operating_policy']==LOCK['operating_policy']==PAYLOAD['operating_policy']
    assert not LOCK['calibration_gate_passes'] and LOCK['locked_before_outer_preparation']
    best=[];all_settings=[]
    for name in sorted({r['model'] for r in CALIBRATION}):
        row=max((r for r in CALIBRATION if r['model']==name),key=rank_key)
        s=row['summary']
        best.append(dict(model=name,**row['operating_policy'],unchanged_baseline_MAE=s['zero_mae'],
            raw_correction_MAE=s['raw_correction_mae'],permitted_correction_MAE=s['permitted_correction_mae'],
            singer_mean_gain_percent=100*s['permitted_correction_gain'],raw_original_abs_cents=s['raw_original_abs_mean'],
            permitted_original_abs_cents=s['permitted_original_abs_mean'],all_attempt_coverage_percent=100*s['all_attempt_permitted_coverage'],
            passed=s['passes']))
    for r in CALIBRATION:
        s=r['summary'];all_settings.append(dict(model=r['model'],**r['operating_policy'],
            mean_singer_gain_percent=100*s.get('permitted_correction_gain',0),raw_MAE=s.get('raw_correction_mae'),
            permitted_MAE=s.get('permitted_correction_mae'),raw_original_abs=s.get('raw_original_abs_mean'),passed=s['passes']))
    BEST=pd.DataFrame(best).sort_values('singer_mean_gain_percent',ascending=False)
    display(BEST.round(4))
    display(HTML('<details><summary>Show all 27 calibration operating points</summary>'+pd.DataFrame(all_settings).round(4).to_html(index=False)+'</details>'))
    group_scores=LOCK['summary']['by_group']
    mean_group_ratios=np.mean([1-g['permitted_correction_mae']/g['zero_mae'] for g in group_scores.values()])
    ratio_of_group_means=1-LOCK['summary']['permitted_correction_mae']/LOCK['summary']['zero_mae']
    np.testing.assert_allclose(mean_group_ratios,LOCK['summary']['permitted_correction_gain'])
    print(f'Selection metric = mean of singer gains: {100*mean_group_ratios:.4f}%; ratio of aggregated MAEs: {100*ratio_of_group_means:.4f}% (different aggregation).')
    fig,ax=plt.subplots(figsize=(10,3.8));x=np.arange(len(BEST))
    ax.bar(x-.18,BEST.raw_correction_MAE,.36,label='Raw prediction MAE',color=COLORS['neutral'])
    ax.bar(x+.18,BEST.permitted_correction_MAE,.36,label='Permitted curve MAE (withholds included)',color=COLORS['output'])
    ax.axhline(LOCK['summary']['zero_mae'],linestyle='--',label='Unchanged singer-mean baseline',color=COLORS['input'])
    ax.set_xticks(x,BEST.model);ax.set(ylabel='Calibration MAE (cents)',title='Best declared operating point for each candidate; ALL criteria failed')
    ax.legend(fontsize=8);plt.tight_layout();plt.show()
    print('Actual selection fragment from the experiment:')
    selection_fragment=run_source[run_source.index('    passing ='):run_source.index('    checkpoint =')]
    display(Code(selection_fragment,language='python'))
    ''')
    md('''
    <a id="model-accuracy-comparison"></a>
    ### 12B. Each model's accuracy on the same recorded examples

    This supplementary comparison asks a concrete question: **given the same recorded input, which available checkpoint predicts the known local correction most accurately?** It contains the historical BPDR Transformer, frozen LocalContextMLP, histogram gradient boosting, ExtraTrees, and a zero-correction baseline.

    **What is held constant:** the 16 existing female5/male5 calibration attempts; the same 10 audited interventions from three recordings; the same fixed scored frames and signed targets; equal weighting of cases within singer and then of the two singers. Six failed intervention constructions remain listed as unscorable. No new intervention, pilot evaluation, final-test access, or model selection is performed here.

    **What is measured:** raw correction MAE in cents (lower is better) and **accuracy within 5 / 10 cents** (higher is better). For example, predicting -24 cents for a -30-cent target has 6 cents error and passes the 10-cent tolerance but fails the 5-cent tolerance. Tolerances were specified before this supplementary comparison was executed; they are descriptive engineering choices, not validated musical acceptability limits. The aggregation is a mean of per-case frame percentages within each singer, then a mean of the singer percentages. It is not a count of independent people or songs.

    The second stage applies one **shared** policy (gain 1, deadband 3 cents), own-audio held-note scope and permission controller to every model. Withheld cases retain a zero curve and stay in the denominator. Its permitted-curve scores are not rendered-WAV accuracy. This shared held-note wrapper is imposed for this comparison; it is not a replay of every model's historical live controller.

    **Training provenance matters:** both tree models used the revised VocalSet training singers. The neural models retain their historical weights, feature interfaces and Transformer normalizer; neither was fine-tuned here. All receive the same runtime-fine-v2 input observations, with their documented model-specific feature transforms. Thus this compares current checkpoint configurations on common examples, not equally trained architectures. Calibration examples were already used for tree-model choice, so these are development scores, not unbiased final-test estimates.

    The original HGB estimator was not serialized. A separate comparison tool repeated its one deterministic fit using only the original cached training rows, weights, seed and settings, then checked that its existing calibration MAEs and original-alteration scores reproduced. That reconstructed checkpoint and fit receipt are stored separately. **Run All performs no fitting**: it verifies the frozen comparison inputs/predictions and recomputes the following table. Original training evidence, chosen model and failed validation flags remain unchanged.
    ''')
    code('''
    comparison_pointer=read_json(WORKSPACE/'results/model_comparison/current.json')
    COMPARISON_RUN=Path(comparison_pointer['path']).resolve()
    assert COMPARISON_RUN.is_relative_to((WORKSPACE/'results/model_comparison').resolve())
    COMPARISON=read_json(COMPARISON_RUN/'evidence.json')
    assert sha(COMPARISON_RUN/'evidence.json')==comparison_pointer['evidence_sha256']
    assert sha(COMPARISON_RUN/'declaration.json')==COMPARISON['declaration_sha256']
    assert COMPARISON['protocol']==read_json(COMPARISON_RUN/'declaration.json')
    assert COMPARISON['protocol']['recovery_run']==str(RUN)
    assert COMPARISON['protocol']['shared_policy']==dict(gain=1.,deadband_cents=3.)
    assert COMPARISON['reconstructed_original_metrics_match'] and COMPARISON['original_artifacts_unchanged']
    assert sha(COMPARISON_RUN/'predictions.npz')==COMPARISON['predictions_sha256']
    assert sha(COMPARISON_RUN/'reconstructed_hgb.joblib')==COMPARISON['checkpoint_hashes']['reconstructed_hgb']
    assert sha(WORKSPACE/'tools/compare_recovery_models.py')==COMPARISON['protocol']['source_sha256']
    for path,digest in {**COMPARISON['input_hashes'],**COMPARISON['protocol']['protected_artifacts']}.items():
        assert sha(path)==digest, f'Changed comparison source: {path}'
    sys.path.insert(0,str(WORKSPACE/'tools'))
    from compare_recovery_models import score_case as rescore_frozen_case, aggregate as aggregate_comparison
    comparison_pairs=recovery_experiment.cached_pairs(PAIRS['calibration'])
    model_names={'unchanged_baseline':'Unchanged baseline', 'historical_transformer':'Historical BPDR Transformer',
        'old_mlp':'Frozen LocalContextMLP', 'hist_gradient_boosting':'Histogram Gradient Boosting', 'extra_trees':'ExtraTrees (demo)'}
    comparison_rows=[];comparison_by_singer=[]
    with np.load(COMPARISON_RUN/'predictions.npz') as frozen:
        for name,label in model_names.items():
            case_rows=[]
            for index,pair in enumerate(comparison_pairs):
                if 'excluded' in pair:
                    case_rows.append(dict(singer=pair['source']['singer_id'],cents=pair['cents'],excluded=pair['excluded']))
                    continue
                predictions={v:frozen[f'{name}__{index}__{v}'] for v in ('A','C')}
                assert all(p.shape==(3,len(pair['target'])) and np.isfinite(p).all() for p in predictions.values())
                row,curves=rescore_frozen_case(pair,predictions)
                for v in ('A','C'):
                    np.testing.assert_allclose(curves[v],frozen[f'{name}__{index}__permitted_{v}'])
                case_rows.append(row)
            summary,by_singer=aggregate_comparison(case_rows)
            for metric,value in summary.items():
                np.testing.assert_allclose(value,COMPARISON['results'][name]['summary'][metric],atol=1e-8)
            assert (summary['attempted'],summary['audited'],summary['singers'],summary['original_recordings'])==(16,10,2,3)
            comparison_rows.append(dict(model=label,raw_MAE_cents=summary['raw_mae_cents'],
                accuracy_within_5c_percent=summary['raw_within_5c_percent'],accuracy_within_10c_percent=summary['raw_within_10c_percent'],
                original_raw_proposal_cents=summary['original_raw_abs_mean_cents'],
                permitted_MAE_cents=summary['permitted_mae_cents'],permitted_accuracy_within_10c_percent=summary['permitted_within_10c_percent'],
                original_edited_frames_percent=summary['original_edited_frames_percent'],
                permitted_coverage_all_16_attempts_percent=summary['coverage_all_attempts_percent']))
            for singer,result in by_singer.items():
                comparison_by_singer.append(dict(model=label,singer=singer,**result))
    MODEL_ACCURACY=pd.DataFrame(comparison_rows)
    display(MODEL_ACCURACY.round(3))
    display(HTML('<details><summary>Show per-singer results and denominators</summary>'+pd.DataFrame(comparison_by_singer).round(4).to_html(index=False)+'</details>'))
    print('HGB reconstruction fit seconds (separate completed tool run):',round(COMPARISON['reconstruction_fit_seconds'],3))
    print('Original calibration scores reproduced; existing evidence/model-selection artifacts are hash-unchanged.')
    print("Common policy = gain 1, deadband 3 cents. The earlier table uses each model's original best policy; those permitted scores can differ.")
    fig,axes=plt.subplots(1,2,figsize=(14,4.8))
    labels=MODEL_ACCURACY.model.str.replace('Historical BPDR ','').str.replace('Histogram Gradient Boosting','Gradient Boosting')
    positions=np.arange(len(labels))
    bar_colors=[COLORS['output'] if 'ExtraTrees' in label else COLORS['neutral'] for label in labels]
    axes[0].barh(positions,MODEL_ACCURACY.raw_MAE_cents,color=bar_colors)
    axes[0].set_yticks(positions,labels);axes[0].invert_yaxis()
    axes[0].set(xlabel='Raw correction MAE (cents); lower is better',title='Same 10 audited interventions / two calibration singers')
    axes[1].barh(positions,MODEL_ACCURACY.accuracy_within_10c_percent,color=bar_colors)
    axes[1].set_yticks(positions,labels);axes[1].invert_yaxis();axes[1].set_xlim(0,100)
    axes[1].set(xlabel='Mean frame accuracy within 10 cents (%)',title='Same fixed target frames; higher is better')
    for ax,values in [(axes[0],MODEL_ACCURACY.raw_MAE_cents),(axes[1],MODEL_ACCURACY.accuracy_within_10c_percent)]:
        for i,value in enumerate(values):ax.text(value+.4,i,f'{value:.2f}',va='center',fontsize=9)
    plt.tight_layout();plt.show()
    tree_score=MODEL_ACCURACY.loc[MODEL_ACCURACY.model=='ExtraTrees (demo)'].iloc[0]
    transformer_score=MODEL_ACCURACY.loc[MODEL_ACCURACY.model=='Historical BPDR Transformer'].iloc[0]
    display(Markdown(f"**Measured reason for the experimental demo choice:** ExtraTrees has {tree_score.raw_MAE_cents:.3f} cents raw MAE and {tree_score.accuracy_within_10c_percent:.2f}% accuracy within 10 cents on these common calibration examples, versus {transformer_score.raw_MAE_cents:.3f} cents and {transformer_score.accuracy_within_10c_percent:.2f}% for the historical Transformer checkpoint. ExtraTrees has the lowest raw MAE and highest 10-cent tolerance score in this comparison. Its permitted accuracy is only {tree_score.permitted_accuracy_within_10c_percent:.2f}%, so useful raw predictions do not guarantee an allowed or beneficial waveform edit. The full validation criteria still fail. This supplementary result supports the current diagnostic choice; it did not determine the earlier locked selection."))
    print('Read accuracy, MAE, original alteration and coverage together; this is not natural intended-note accuracy or a final test.')
    ''')
    md('''
    <a id="s13"></a>
    ## 13. Which validation criteria failed, and why is the option still visible?

    Before evaluation, the practical criteria required both singers to supply labels and each singer to meet response, error reduction, original-preservation and coverage conditions. Actual rendered improvements for each singer and unchanged original anchors were additional requirements. These are provisional engineering criteria, not calibrated musical standards.

    Calibration failed particularly because male5 had no permitted repair response or coverage, while female5's raw original proposals exceeded the three-cent mean limit. In the pilot, female6 had no valid intervention labels, so two-singer performance could not be established. Favorable male6 results cannot fill that missing evidence.

    The app therefore uses a **separate experimental preview card**, not a passed-research-model card. It exposes a post-hoc selected verified example, explicitly reports failed broader criteria, leaves the historical model-adoption flags false, preserves originals, and requires user review before a separate export. It does not quietly turn the failed calibration result into a pass.

    The table recomputes each available criterion from saved per-singer metrics. Missing pilot data is shown as **no valid labels**, never as a pass or as zero error.
    ''')
    code('''
    criterion_rows=[]
    for stage,summary in [('calibration',LOCK['summary']),('pilot',EVIDENCE['outer'])]:
        all_singers=sorted(groups['calibration' if stage=='calibration' else 'pilot_evaluation'])
        for singer in all_singers:
            g=summary.get('by_group',{}).get(singer)
            if g is None:
                criterion_rows.append(dict(stage=stage,singer=singer,criterion='Measurable controlled labels',required='present',observed='no valid labels',passes=False))
                continue
            checks=[('Permitted correction gain','> 0',g['permitted_correction_gain'],g['permitted_correction_gain']>0),
                ('Response gain','>= 0.10',g['permitted_response_gain'],g['permitted_response_gain']>=.1),
                ('Raw original mean absolute proposal','<= 3 cents',g['raw_original_abs_mean'],g['raw_original_abs_mean']<=3),
                ('Permitted original mean absolute edit','<= 1 cent',g['permitted_original_abs_mean'],g['permitted_original_abs_mean']<=1),
                ('Edited original frame fraction','<= 0.05',g['original_edit_frame_fraction'],g['original_edit_frame_fraction']<=.05),
                ('Permitted coverage over ALL attempts','>= 0.25',g['all_attempt_permitted_coverage'],g['all_attempt_permitted_coverage']>=.25)]
            for name,required,observed,passed in checks:
                criterion_rows.append(dict(stage=stage,singer=singer,criterion=name,required=required,observed=observed,passes=passed))
    CRITERIA=pd.DataFrame(criterion_rows)
    display(CRITERIA)
    assert not EVIDENCE['outer_proposal_gate_passes'] and not EVIDENCE['rendered_demo_gate_passes'] and not EVIDENCE['model_adopted']
    assert CARD['validation_status']=='exploratory_preview_failed_generalization_gate'
    assert CARD['research_gates_passed'] is False and CARD['requires_user_review']
    display(pd.DataFrame([dict(calibration_passed=LOCK['calibration_gate_passes'],
        broader_pilot_passed=EVIDENCE['outer_proposal_gate_passes'],rendered_generalization_passed=EVIDENCE['rendered_demo_gate_passes'],
        research_model_adopted=EVIDENCE['model_adopted'],experimental_option_available=CARD['eligible_for_demo'])]))
    ''')

    md('''
    <a id="s14"></a>
    ## 14. Complete delivered results: successes, withholding and missing labels

    The locked ExtraTrees candidate was replayed through the actual runtime, controller, renderer and encoded-file audit. Four unique unshifted working recordings were separately tested. Every audited detuned case was scored using its delivered output, including retained inputs when an edit was withheld.

    The outcomes are: **16 attempted interventions; 12 unscorable; 4 scoreable from one singer; 3 numerically verified previews; 2 local improvements**. One +20-cent case was withheld. One -30-cent case edited another region without reducing the known target-region error. Both measured improvements were on +30-cent interventions.

    This last failure is instructive: an output can pass the audit of the movement the model requested while failing to improve the actual recovery target. A renderer audit cannot infer musical intention. Numerical validity and beneficial correction must be evaluated separately.

    Mean delivered local MAE across all four scoreable cases was **27.442 -> 14.663 cents**, a **46.6% relative error reduction**. That conditional estimate comes from two recordings by one singer. It is not an accuracy estimate across the two assigned pilot singers or the general user population. Report 12 unscorable attempts beside it, and keep the two-singer criterion failed.
    ''')
    code('''
    VERIFICATION_PATH=RUN/'rendered_verification.json'
    assert sha(VERIFICATION_PATH)==EVIDENCE['rendered_verification_sha256']==CARD['verification_sha256']
    VERIFIED=read_json(VERIFICATION_PATH)
    assert VERIFIED['checkpoint_sha256']==sha(CHECKPOINT)
    assert VERIFIED['declaration_sha256']==EVIDENCE['declaration_sha256']
    assert len(VERIFIED['shifts'])==16 and len(VERIFIED['originals'])==4
    result_rows=[]
    for recorded,pair in zip(VERIFIED['shifts'],PAIRS['pilot_evaluation']):
        assert recorded['singer']==pair['source']['singer_id'] and recorded['cents']==pair['cents']
        row=dict(singer=recorded['singer'],vowel=pair['source']['vowel'],injected_cents=pair['cents'])
        if 'excluded' in recorded:
            row.update(outcome='unscorable',reason=recorded['excluded'],input_local_MAE=None,output_local_MAE=None,improved=False)
        else:
            row.update(outcome=recorded['outcome'],reason='improved' if recorded['verified_improvement'] else 'no demonstrated target improvement',
                input_local_MAE=recorded['input_mae_cents'],output_local_MAE=recorded['output_mae_cents'],
                improved=recorded['verified_improvement'])
        result_rows.append(row)
    ALL_RESULTS=pd.DataFrame(result_rows)
    display(ALL_RESULTS.round(4))
    VALID=[r for r in VERIFIED['shifts'] if 'excluded' not in r]
    assert len(VALID)==4 and all(r['missing_output_scored_frames']==0 for r in VALID)
    input_mean=float(np.mean([r['input_mae_cents'] for r in VALID]))
    output_mean=float(np.mean([r['output_mae_cents'] for r in VALID]))
    improvement=1-output_mean/input_mean
    np.testing.assert_allclose([input_mean,output_mean,improvement],
        [CARD['metrics']['input_mae_cents'],CARD['metrics']['output_mae_cents'],CARD['metrics']['error_reduction_fraction']])
    original_rows=[]
    for r in VERIFIED['originals']:
        a,sr=sf.read(r['input'],dtype='float32');b,br=sf.read(r['output'],dtype='float32')
        assert r['preserved'] and sr==br and np.array_equal(a,b)
        original_rows.append(dict(singer=r['singer'],file=Path(r['input']).parent.name,outcome=r['outcome'],exact_working_samples_preserved=True))
    display(pd.DataFrame(original_rows))
    display(pd.DataFrame(VERIFIED['by_singer']).T)
    stages=pd.DataFrame([
        dict(stage='Raw predicted correction vs known target',baseline_MAE=EVIDENCE['outer']['zero_mae'],MAE=EVIDENCE['outer']['raw_correction_mae']),
        dict(stage='Permitted curve vs known target; withholds included',baseline_MAE=EVIDENCE['outer']['zero_mae'],MAE=EVIDENCE['outer']['permitted_correction_mae']),
        dict(stage='Actual WAV vs measured original; fixed local frames',baseline_MAE=input_mean,MAE=output_mean)])
    display(stages.round(4))
    print('The known intervention target and the realized measured pitch shift are close but not identical. Keep stage and reference definitions attached to each MAE.')
    counts=[12,1,1,2];names=['Unscorable intervention creation','Audited but withheld','Preview without target improvement','Verified local improvement']
    assert sum(counts)==16
    fig,axes=plt.subplots(1,2,figsize=(13,4))
    axes[0].barh(names,counts,color=[COLORS['neutral'],COLORS['input'],COLORS['input'],COLORS['output']])
    axes[0].set(xlabel='Mutually exclusive outcomes of 16 attempts',title='Coverage and failures stay visible')
    x=np.arange(4)
    axes[1].bar(x-.18,[r['input_mae_cents'] for r in VALID],.36,label='Input baseline',color=COLORS['input'])
    axes[1].bar(x+.18,[r['output_mae_cents'] for r in VALID],.36,label='Delivered output',color=COLORS['output'])
    axes[1].set_xticks(x,[f"{Path(r['input']).parent.name} / {r['cents']:+g}c" for r in VALID],rotation=15)
    axes[1].set(ylabel='Local MAE (cents)',title='All four scoreable cases: male6, two recordings');axes[1].legend()
    plt.tight_layout();plt.show()
    ''')
    md('''
    <a id="s15"></a>
    ## 15. Fresh notebook inference using the saved model

    The next cell runs the same public learned-model adapter as the app on the selected example. This is a new bounded inference, not fitting. It passes **only the shifted input file** to the model runtime. The known original and target mask are read later for evaluation.

    Selection is explicit: the example is the first numerically verified improvement in the locked source/shift order, chosen **after** observing the diagnostic pilot. It is a selected listening demonstration. Its behavior must not be substituted for the full table above.

    A technically accepted original is retained. Any prepared candidate remains a separate file. The public loader verifies the checkpoint, declaration, locked settings, verification receipt and experimental status; it does not change failed research gates into passes.
    ''')
    code('''
    SELECTED=next(r for r in VALID if r['verified_improvement'])
    SELECTED_PAIR=next(p for p in PAIRS['pilot_evaluation'] if p.get('path_C')==SELECTED['input'])
    assert CARD['demo_cases'][0]['input']==SELECTED['input']
    from bpdr_input.repair.learned_runtime import run as learned_inference, load_model as load_demo_model
    from bpdr_input.repair.demo_runtime import selected_output
    before_model_sha=sha(CHECKPOINT);before_evidence_sha=sha(RUN/'evidence.json')
    FRESH_DIRECTORY,FRESH_REPORT=learned_inference(SELECTED['input'],output_root=WORKSPACE/'results/notebook_inference')
    FRESH_OUTPUT,output_label=selected_output(FRESH_REPORT)
    assert FRESH_REPORT['outcome']=='VERIFIED_FIXTURE_CANDIDATE' and output_label=='Processed preview',FRESH_REPORT
    assert FRESH_REPORT['research_gates_passed'] is False
    assert sha(CHECKPOINT)==before_model_sha and sha(RUN/'evidence.json')==before_evidence_sha
    assert sha(FRESH_OUTPUT)==FRESH_REPORT['candidate']['sha256']
    INPUT_AUDIO,input_sr=sf.read(FRESH_REPORT['handoff']['accepted_audio']['path'],dtype='float32')
    OUTPUT_AUDIO,output_sr=sf.read(FRESH_OUTPUT,dtype='float32')
    REFERENCE_AUDIO,ref_sr=sf.read(SELECTED['reference'],dtype='float32')
    assert input_sr==output_sr==ref_sr==22050 and INPUT_AUDIO.shape==OUTPUT_AUDIO.shape==REFERENCE_AUDIO.shape
    with np.load(SELECTED['target_path']) as z:
        SCORE_TIMES=z['times'];FIXED_MASK=z['mask'].astype(bool)&(z['target']!=0)
    reference_pitch=fine_pitch(REFERENCE_AUDIO,ref_sr,SCORE_TIMES)
    fresh_input_error=fine_pitch(INPUT_AUDIO,input_sr,SCORE_TIMES)-reference_pitch
    fresh_output_error=fine_pitch(OUTPUT_AUDIO,output_sr,SCORE_TIMES)-reference_pitch
    assert np.isfinite(fresh_input_error[FIXED_MASK]).all() and np.isfinite(fresh_output_error[FIXED_MASK]).all()
    FRESH_INPUT_MAE=float(np.abs(fresh_input_error[FIXED_MASK]).mean())
    FRESH_OUTPUT_MAE=float(np.abs(fresh_output_error[FIXED_MASK]).mean())
    np.testing.assert_allclose([FRESH_INPUT_MAE,FRESH_OUTPUT_MAE],[SELECTED['input_mae_cents'],SELECTED['output_mae_cents']],atol=.01)
    display(pd.DataFrame([dict(input_local_MAE_cents=FRESH_INPUT_MAE,output_local_MAE_cents=FRESH_OUTPUT_MAE,
        local_error_reduction_percent=100*(1-FRESH_OUTPUT_MAE/FRESH_INPUT_MAE),scored_frames=int(FIXED_MASK.sum()),
        full_recording_seconds=len(INPUT_AUDIO)/input_sr,outcome=FRESH_REPORT['outcome'])]).round(4))
    print('Fresh run folder:',FRESH_DIRECTORY)
    print('Fresh delivered WAV:',FRESH_OUTPUT)
    print('Clean reference and private mask were used only AFTER the runtime call.')
    ''')
    md('''
    <a id="s16"></a>
    ## 16. Follow one frame through the actual 96-tree model

    We can inspect the exact feature questions a fitted tree asks. The middle scored frame is chosen for explanation after inference; it does not tell the model where to correct. The first tree follows its learned thresholds and returns one signed correction. The forest averages all 96 tree predictions. Operating gain is then applied once, and short/unsupported proposals may still be withheld.

    A tree split uses a scaled feature value from the dictionary in section 8, not a named music key. The spread across tree outputs below is descriptive model variation; the implementation does not treat it as a calibrated probability of correct singing.

    Current alternative boundary views yield identical tree predictions because the boundary column is not used. Their agreement cannot certify that the predicted correction matches musical intention.
    ''')
    code('''
    from bpdr_input.modeling.recovery_model import propose
    with np.load(FRESH_DIRECTORY/'features.npz') as z:
        FRESH_FEATURES={k:z[k] for k in z.files}
    fine_input=fine_pitch(INPUT_AUDIO,input_sr,FRESH_FEATURES['times_seconds'])
    FRESH_VIEWS,FRESH_RELIABLE,FRESH_PROTECTED,FRESH_ADAPTER=runtime_views(FRESH_FEATURES,fine_input)
    X_FRESH=context_features(FRESH_VIEWS[0],include_boundary=False)
    proposal=propose(FRESH_VIEWS,MODEL,PAYLOAD)
    with np.load(FRESH_DIRECTORY/'model_predictions.npz') as z:
        np.testing.assert_allclose(z['predictions'],proposal['operating_predictions'],atol=1e-6)
        REQUESTED_CURVE=z['proposed_correction'].copy()
    frame=int(np.flatnonzero(FIXED_MASK)[int(FIXED_MASK.sum())//2])
    tree_outputs=np.array([tree.predict(X_FRESH[frame:frame+1])[0] for tree in ESTIMATOR.estimators_])
    forest_output=float(ESTIMATOR.predict(X_FRESH[frame:frame+1])[0])
    np.testing.assert_allclose(tree_outputs.mean(),forest_output)
    display(pd.DataFrame([dict(frame=frame,time_seconds=SCORE_TIMES[frame],first_tree_cents=tree_outputs[0],
        mean_of_96_trees_cents=forest_output,operating_gain=PAYLOAD['operating_policy']['gain'],
        final_requested_cents=REQUESTED_CURVE[frame])]).round(5))
    tree=ESTIMATOR.estimators_[0];node=0;path_rows=[]
    while tree.tree_.feature[node]>=0:
        column=int(tree.tree_.feature[node]);threshold=float(tree.tree_.threshold[node]);value=float(X_FRESH[frame,column])
        left=value<=threshold
        path_rows.append(dict(node=node,feature=FEATURE_NAMES[column],observed_scaled_value=value,
            question=f'<= {threshold:.5f}',answer=bool(left)))
        node=int(tree.tree_.children_left[node] if left else tree.tree_.children_right[node])
    display(pd.DataFrame(path_rows))
    print('Reached leaf:',node,'; learned correction:',float(tree.tree_.value[node].ravel()[0]),'cents; training rows reaching leaf:',int(tree.tree_.n_node_samples[node]))
    fig,ax=plt.subplots(figsize=(10,3.5));ax.hist(tree_outputs,bins=15,color=COLORS['output'])
    ax.axvline(forest_output,color=COLORS['input'],linestyle='--',label='Forest average')
    ax.set(xlabel='Individual tree prediction (cents)',ylabel='Trees',title='Actual 96-tree outputs for one frame; not calibrated correctness probabilities')
    ax.legend();plt.tight_layout();plt.show()
    ''')
    md('''
    <a id="s17"></a>
    ## 17. Permission, rendering, remeasurement and the final audible WAV

    The controller can reject a predicted curve for excessive correction (>30 cents), changed-boundary disagreement (>5 cents), unreliable voicing, protected transitions, insufficient continuous support (<0.86 seconds), or too much affected audio (>25%). It does not silently clip an over-limit prediction into a pass. The largest above-deadband contiguous proposal is considered; short fragmented suggestions can therefore be withheld.

    A permitted request is rendered using a pinned Praat/Parselmouth pitch tier. Measured frequencies are scaled by `2 ** (correction_cents / 1200)`, with local crossfades. The duration tier is unchanged. Samples outside the permitted support are copied exactly. The renderer is separate from both the frozen pitch analyzer and the learned tree model.

    The result is checked for sample grid, clipping, realized pitch movement, direction, coverage and outside drift. It is then encoded as PCM24, reloaded from the exact saved WAV, and audited again. A failure retains the original. These are numerical safeguards, not independent musical approval.

    Listen at the same volume. The shift is small and local; most of the full recording should sound unchanged. The reference is the original performed vocal, not a perfected or independently annotated intended melody.
    ''')
    code('''
    plan=FRESH_REPORT['permission'];transaction=FRESH_REPORT['transaction']
    delivered=transaction['delivered_wav_audit'];audit=delivered['acoustic_audit']
    assert plan['permitted'] and delivered['passed'] and audit['passed']
    assert delivered['sha256']==sha(FRESH_OUTPUT) and delivered['outside_support_unchanged']
    inside=np.zeros(len(INPUT_AUDIO),dtype=bool);sample_times=np.arange(len(INPUT_AUDIO))/input_sr
    for left,right in plan['intervals_seconds']:
        inside|=(sample_times>=left)&(sample_times<right)
    assert np.array_equal(INPUT_AUDIO[~inside],OUTPUT_AUDIO[~inside])
    display(pd.DataFrame([dict(check='Controller magnitude limit',observed=plan['requested_max_abs_cents'],limit=plan['policy']['max_correction_cents']),
        dict(check='Affected sample fraction',observed=plan['affected_fraction'],limit=plan['policy']['max_affected_fraction']),
        dict(check='Delivered fine-pitch median movement error',observed=audit['fine']['median_error_cents'],limit=audit['base_limits']['median_error_tolerance_cents']),
        dict(check='Delivered fine-pitch p90 movement error',observed=audit['fine']['p90_error_cents'],limit=audit['base_limits']['p90_error_tolerance_cents'])]))
    print('Permitted intervals (seconds):',plan['intervals_seconds'])
    print('Encoded subtype:',delivered['subtype'],'; sample count:',delivered['sample_count'])
    print('Outside-support samples exactly unchanged:',bool(np.array_equal(INPUT_AUDIO[~inside],OUTPUT_AUDIO[~inside])))
    fig,axes=plt.subplots(2,1,figsize=(12,5.3))
    axes[0].plot(SCORE_TIMES[FIXED_MASK],fresh_input_error[FIXED_MASK],color=COLORS['input'],label='Input local residual')
    axes[0].plot(SCORE_TIMES[FIXED_MASK],fresh_output_error[FIXED_MASK],color=COLORS['output'],label='Output local residual')
    axes[0].axhline(0,color='#333333',linewidth=.7)
    axes[0].set(xlabel='Recording time (seconds)',ylabel='Cents relative to unshifted voice',title='Selected measured local improvement; same fixed scoring frames');axes[0].legend()
    axes[1].plot(FRESH_FEATURES['times_seconds'],REQUESTED_CURVE,color=COLORS['output'])
    axes[1].set(xlabel='Recording time (seconds)',ylabel='Requested correction (cents)',title='Full-phrase requested curve: edit remains local')
    plt.tight_layout();plt.show()
    for label,wave in [('Input: real voice with a known added +30-cent shift',INPUT_AUDIO),
                       ('Actual fresh learned-model output',OUTPUT_AUDIO),
                       ('Original unshifted performance: evaluation reference only',REFERENCE_AUDIO)]:
        display(Markdown('**'+label+'**'));display(Audio(wave,rate=input_sr,normalize=False))
    ''')

    md('''
    <a id="s18"></a>
    ## 18. From the final WAV to an accepted package and actual MIDI/CSV

    Pitch recovery alone does not decide whether a phrase is a useful musical idea. A separate **provisional, rule-based acoustic assessment** checks agreeing pitch evidence. It is not a trained speech/singing or musical-correctness classifier. It uses the person's declared intent and does not require several notes, scale membership, a high whole-recording voiced fraction or absence of expressive slides.

    In the real app, a person selects the original or the verified preview, listens, and explicitly confirms that it preserves the intended idea. Export copies that exact waveform into a separate package, rechecks its hash and includes operations, interval metadata, policy/model identity and assessment. It does not overwrite the shared original handoff or assert independent research approval.

    For this reproducible notebook we exercise export as an **`automated_test_fixture`**. No human listening or approval is claimed. This verifies mechanics, not musical judgment. The previously executed GAME connection used the same prepared WAV bytes; below we verify those identities and read its actual MIDI/CSV outputs. No new broad extractor run or training occurs in this cell.

    GAME is a separately pinned existing melody extractor, not our ExtraTrees repair model. Successful MIDI/CSV creation proves an executable handoff. It does not prove the notes match intention, improve extractor F1 or satisfy downstream harmony requirements.
    ''')
    code('''
    from bpdr_input.audio.usability import UsabilityPolicy
    from bpdr_input.repair.demo_review import review,export_reviewed
    ACOUSTIC_REVIEW=review(FRESH_DIRECTORY,FRESH_REPORT,'singing_humming_chant')
    display(pd.DataFrame([asdict(UsabilityPolicy())]).T.rename(columns={0:'provisional_setting'}))
    display(pd.DataFrame([dict(version=role,decision=value['decision'],feedback=value['feedback'])
        for role,value in ACOUSTIC_REVIEW.items() if role in ('original','preview')]))
    assert ACOUSTIC_REVIEW['preview']['decision']=='PASS_EVIDENCE'
    PACKAGE_DIRECTORY,PACKAGE=export_reviewed(FRESH_DIRECTORY,FRESH_REPORT,ACOUSTIC_REVIEW,
        use_preview=True,confirmed=True,confirmation_kind='automated_test_fixture')
    PACKAGE_DIAGNOSTICS=read_json(PACKAGE_DIRECTORY/'diagnostics.json')
    assert PACKAGE['status']=='ACCEPT_PREPARED'
    assert not PACKAGE_DIAGNOSTICS['user_confirmed'] and not PACKAGE_DIAGNOSTICS['research_approval']
    assert sha(PACKAGE_DIRECTORY/'accepted.wav')==sha(FRESH_OUTPUT)==PACKAGE['accepted_audio']['checksum']
    display(pd.DataFrame([dict(status=PACKAGE['status'],policy=PACKAGE['policy_version'],
        exact_candidate_bytes_exported=True,human_listening_performed=False,research_approval=False,
        output_waveform_sha256=PACKAGE['accepted_audio']['checksum'])]))
    print('Notebook integration-fixture package:',PACKAGE_DIRECTORY)
    import zipfile
    with zipfile.ZipFile(PACKAGE_DIRECTORY/'accepted_package.zip') as package_zip:
        assert set(package_zip.namelist())=={'accepted.wav','handoff.json','diagnostics.json'}
        print('Downloadable package contents:',package_zip.namelist())
    ''')
    code('''
    EXTRACTION_LINK=read_json(WORKSPACE/'results/model_recovery/learned_extractor_fixture_current.json')
    EXTRACTION=read_json(EXTRACTION_LINK['receipt'])
    assert EXTRACTION['success'] and EXTRACTION['returncode']==0
    assert EXTRACTION['accepted_sha256']==sha(FRESH_OUTPUT)
    original_fixture=Path(EXTRACTION_LINK['handoff_directory'])
    assert sha(original_fixture/'accepted.wav')==EXTRACTION['accepted_sha256']
    assert read_json(original_fixture/'diagnostics.json')['confirmation_kind']=='automated_test_fixture'
    for item in EXTRACTION['artifacts']:
        assert sha(item['path'])==item['sha256']
    csv_path=next(Path(r['path']) for r in EXTRACTION['artifacts'] if Path(r['path']).suffix=='.csv')
    midi_path=next(Path(r['path']) for r in EXTRACTION['artifacts'] if Path(r['path']).suffix=='.mid')
    EVENTS=pd.read_csv(csv_path)
    assert {'onset','offset','pitch'}<=set(EVENTS.columns)
    assert np.isfinite(EVENTS[['onset','offset','pitch']].to_numpy()).all()
    assert (EVENTS.offset>EVENTS.onset).all()
    display(EVENTS)
    print('Exact prepared WAV -> GAME success:',EXTRACTION['success'])
    print('Recorded extractor seconds:',EXTRACTION['elapsed_seconds'])
    print('CSV:',csv_path)
    print('MIDI:',midi_path)
    print('GAME checkpoint SHA-256:',EXTRACTION['model_sha256'])
    print('These pitch numbers are estimated MIDI-unit pitches, including fractions; they are not human ground truth.')
    fig,ax=plt.subplots(figsize=(11,3.4))
    for row in EVENTS.itertuples():
        ax.plot([row.onset,row.offset],[row.pitch,row.pitch],linewidth=6,solid_capstyle='round',color=COLORS['output'])
    ax.set(xlabel='Seconds in accepted WAV',ylabel='Estimated MIDI pitch',title='Actual downstream GAME note estimates from the prepared demonstration waveform')
    plt.tight_layout();plt.show()
    ''')
    md('''
    <a id="s19"></a>
    ## 19. Use the demo, or process another recording in this notebook

    From `BPDR_Workspace`, run `./Run-Demo.ps1` (or `./Run-Demo.ps1 -Port 8503`). Choose the **experimental learned repair** example, press **Process recording**, compare the two audio players and inspect the full-pilot metrics. Select a version and confirm only after actually listening. The model runs saved-weight inference; the app does not fit a model during the presentation.

    For a personal WAV/FLAC or microphone recording, declare whether it contains an intended sung/hummed/chanted idea, ordinary reading without a tune, or uncertain intent. Explicit reading is sent through preserve-only analysis and receives guidance to sing or hum. Uncertain intent needs confirmation. Insufficient acoustic evidence blocks handoff and requests a clearer take. These are conservative demo behaviors with unvalidated thresholds, not reliable automatic intention recognition.

    | State | Meaning |
    |---|---|
    | ACCEPT_UNCHANGED | Technical acceptance of an original working copy, or selection of a reviewed usable original |
    | ACCEPT_PREPARED | Separate selected prepared package; acoustic checks plus explicit user review in the app |
    | RERECORD_REQUIRED | Technical input conditions fail, such as silence; ask for another usable recording |
    | TECHNICAL_FAILURE | Processing failed; this is not a judgment of singing skill |
    | NEED_CLEARER_INPUT | Separate acoustic assessment lacks enough reliable evidence; block export |
    | NEED_INTENT_CONFIRMATION | Ask whether the person intended a melodic idea |

    The optional cell below is disabled by default. Set `CUSTOM_AUDIO` to an existing local file to run the same demo methods. With no trusted reference, it cannot report whether a natural note was "correct", a pitch-error accuracy percentage or recovery of the person's intended melody. It does not automatically export or claim a human reviewed the result.
    ''')
    code('''
    CUSTOM_AUDIO=None  # Example: Path(r'C:\\path\\to\\your_solo_voice.wav')
    CUSTOM_INTENT='unsure'  # singing_humming_chant / reading_spoken_text / unsure
    if CUSTOM_AUDIO is None:
        print('Optional custom recording skipped. Set CUSTOM_AUDIO and your true intent, then run this cell.')
    else:
        from bpdr_input.repair.demo_runtime import run as demo_process,METHODS,selected_output
        from bpdr_input.repair.learned_runtime import METHOD_NAME
        method=METHODS[3] if CUSTOM_INTENT=='reading_spoken_text' else METHOD_NAME
        directory,report=demo_process(Path(CUSTOM_AUDIO),method=method)
        assessment=review(directory,report,CUSTOM_INTENT)
        output,label=selected_output(report)
        print('Processing outcome:',report['outcome'],'; technical status:',report['handoff']['status'])
        print('Acoustic/intent assessment:',assessment)
        if output:
            display(Markdown('**'+label+'; no unknown-intention accuracy score is available**'))
            display(Audio(filename=str(output),normalize=False))
        print('No automatic export or human approval was performed by this cell.')
    ''')
    md('''
    <a id="s20"></a>
    ## 20. Reproducibility, conclusions, limitations and next research steps

    **What is established:** the source files are preserved; bounded tree training completed; the declared model selection can be replayed; the selected model produces two measured local improvements in the four scoreable pilot interventions; four unshifted working recordings were preserved; exact prepared audio can be exported and consumed by the existing extractor. A new notebook inference reproduces the selected input/output measurements.

    **What remains unestablished:** natural intended-note accuracy, useful repair across unfamiliar singers/melodies, recovery of an intended tune from ordinary reading, absence of audible artifacts, meaningful superiority over the historical Transformer, benefit to downstream note F1, population-calibrated usability, and learned BPDR boundary/pitch disentanglement.

    | Limitation | Consequence / next useful step |
    |---|---|
    | Only one pilot singer has valid shift labels; many exclusions | Expand reviewed usable coverage and retain failure denominators |
    | Revised protocol reuses development singer IDs | Reserve genuinely untouched singers and melodies for a later final test |
    | Clean target means preserve performed audio, not ideal intention | Obtain guide/reference melodies and independent performance/intention annotations where justified |
    | Known renderer-generated shifts | Test other corruption methods and natural novice errors; check for renderer artifacts learned by the model |
    | Only +30-cent cases improved here | Report directional coverage and failure cases; do not advertise all signed small errors as solved |
    | YIN is used in features and fine scoring; PESTO is coarse | Add independent annotations/listening; estimator agreement is not ground truth |
    | Slight calibration ranking margin; all criteria failed | No general architecture superiority claim; use matched training/compute/splits for a later comparison |
    | Boundary metadata excluded by trees | Treat this as an engineering alternative; the original BPDR research question still needs its own matched study |
    | Professional sustained vowels, not diverse user phrases | Evaluate repeated notes, humming, expressive singing, pauses, speech-like melody, ordinary reading, noise and overlap |
    | Acoustic rules and optional user selection are provisional | Review usable/unusable labels, acceptance coverage, harmful edits, unnecessary rerecording and listening preferences |

    A future downstream evaluation should compare original and prepared audio through the same frozen extractor, with trusted note/timing references and comparable acceptance coverage. MIDI production alone is not such an evaluation. Further training should follow an explicit revised protocol, not repeated fitting until one example looks successful.

    ### Where each step lives

    | Purpose | Workspace source/evidence |
    |---|---|
    | Official bounded source intake | `source/bpdr_input/preparation/sources/vocalset_subset.py`; `datasets/vocalset_pilot/manifest.json` |
    | Preparation, weights, fixed fits and selection | `source/bpdr_input/experiments/learning/model_recovery.py`; run declaration/selection/evidence |
    | 31 features | `source/bpdr_input/modeling/local_context_pilot.py` |
    | Model proposal and guarded loading | `source/bpdr_input/modeling/recovery_model.py` |
    | Exact live model adapter | `source/bpdr_input/repair/learned_runtime.py` |
    | Permission, resynthesis and audits | `repair_controller.py`, `framewise_renderer.py`, `framewise_audit.py` under `source/bpdr_input/repair` |
    | All-case delivered verification | `source/bpdr_input/experiments/learning/verify_model_recovery.py`; `rendered_verification.json` |
    | Usability, export and next component | `audio/usability.py`, `repair/demo_review.py`, `repair/demo_extraction.py` |
    | User interface | `source/bpdr_input/application/demo_app.py` |
    | Earlier studies | Notebook 02 (history), 03 (Transformer training), 04 (Transformer evaluation), 06 (earlier real-data MLP pilot) |
    | Common-example accuracy comparison | `tools/compare_recovery_models.py`; `results/model_comparison/current.json` and its frozen predictions, deterministic HGB reconstruction and integrity receipts |

    **Explicit commands, not executed by Run All:** `python -B -m bpdr_input.experiments.learning.model_recovery` starts a new declared training experiment; `tools/execute_recovery_notebook.py` verifies and exports this notebook; `tools/check_recovery_demo.py` checks the live UI; `tools/prepare_notebooks_for_git.py` preserves executed copies and clears source outputs. Use the component `.venv` and `PYTHONPATH=BPDR_Workspace/source`. Never overwrite old failed evidence to present a successful result.

    ### Reference material

    - [VocalSet official source and attribution](https://zenodo.org/records/1193957) and [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
    - [ExtraTreesRegressor algorithm and prediction definition](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html).
    - [Histogram Gradient Boosting Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html).
    - [Mean absolute error](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html).
    - Project implementation, frozen declarations, source snapshots, model hash and recorded measurements checked throughout this notebook.
    ''')
    code('''
    QA=dict(source_recordings_verified=len(SOURCES),unique_source_hashes=int(SOURCES.sha256.nunique()),
        training_originals_with_valid_labels=int(COVERAGE.loc['train','recordings_with_valid_pairs']),
        training_singers_with_valid_labels=int(COVERAGE.loc['train','singers_with_valid_pairs']),training_rows=len(Y_TRAIN),
        calibration_selection_replayed=True,all_27_calibration_rows_failed=not any(r['summary']['passes'] for r in CALIBRATION),
        common_example_models_compared=len(MODEL_ACCURACY)-1,comparison_accuracy_tolerance_cents=10,
        comparison_original_evidence_unchanged=COMPARISON['original_artifacts_unchanged'],
        pilot_attempts=len(VERIFIED['shifts']),pilot_audited=len(VALID),verified_improvements=sum(r['verified_improvement'] for r in VALID),
        exact_originals_preserved=len(original_rows),fresh_inference_reproduced=True,encoded_WAV_verified=delivered['passed'],
        downstream_note_events=len(EVENTS),downstream_exact_waveform_verified=EXTRACTION['accepted_sha256']==sha(FRESH_OUTPUT),
        human_musical_judgment=False,research_model_adopted=EVIDENCE['model_adopted'],
        confidence_assessment='Share with noted caveats: narrow development demonstration; broader validation failed')
    assert QA['unique_source_hashes']==24 and QA['pilot_attempts']==16 and QA['pilot_audited']==4
    assert QA['verified_improvements']==2 and QA['exact_originals_preserved']==4
    assert sha(CHECKPOINT)==before_model_sha and sha(RUN/'evidence.json')==before_evidence_sha
    display(pd.DataFrame([{'check':k,'observed':str(v)} for k,v in QA.items()]))
    print('Training, download and intervention-generation calls executed by this notebook: 0')
    print('One saved-weight model inference and one explicitly labelled automated export fixture were executed.')
    print('The downstream CSV/MIDI are verified recorded outputs; the extractor was not rerun by Run All.')
    ''')

    return cells


def main():
    notebook = nbformat.v4.new_notebook(cells=build())
    notebook.metadata.update(kernelspec=dict(name='bpdr-local',display_name='BPDR local Python',language='python'),
        language_info=dict(name='python'),bpdr_verification=dict(minimum_plots=10,minimum_audio=5))
    target = WORKSPACE/'notebooks/07_learned_stable_note_recovery.ipynb'
    nbformat.validate(notebook)
    nbformat.write(notebook, target)
    print(target)


if __name__ == '__main__':
    main()
