"""Gate E input alignment and a locked exploratory comparison; never trains a model."""
from collections import defaultdict
import json
from pathlib import Path
import shutil
import uuid

import numpy as np

from .laboratory import validate_pair_record
from .shared_exports import digest
from .shared_pair_audit import game_boundary_track


STATUS = 'exploratory_not_research'


def checked_training_records(records):
    """Check the complete selection before any input file is opened."""
    if not records:
        raise ValueError('Empty training selection')
    keys = []
    for r in records:
        if (r['split'] != 'train' or not r['audit_passed'] or not r['exploratory']
                or r['training_eligible'] is not False):
            raise ValueError('Only audited exploratory frozen-training pairs are allowed')
        keys.append((r['dataset'], r['phrase_id'], r['cents'], r['corruption']))
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate pair identity')
    return records


def balanced_weights(records):
    """Equal dataset, group, recording, phrase, shift, then corruption mass.

    This is an explicit engineering mixture, not an estimate of population frequency.
    Artist groups remain proxies. Each recording's many variants share its mass.
    """
    checked_training_records(records)
    paths = [(r['dataset'], r['group_id'], r['source_id'], r['phrase_id'],
              str(r['cents']), r['corruption']) for r in records]
    children = defaultdict(set)
    for path in paths:
        for depth, child in enumerate(path):
            children[path[:depth]].add(child)
    weights = []
    for path in paths:
        weight = 1.
        for depth in range(len(path)):
            weight /= len(children[path[:depth]])
        weights.append(weight)
    result = np.asarray(weights, dtype=np.float64)
    if not np.isclose(result.sum(), 1.):
        raise ValueError('Non-unit sampling mass')
    return result


def fit_streaming_normalizer(records, weights):
    """Training-only weighted frame moments; at most one view archive in memory.

    Each pair has its declared mass, divided equally over four views and real
    frames. Padding never enters the moments; natural unvoiced context does.
    """
    checked_training_records(records)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (len(records),) or not np.isfinite(weights).all() or np.any(weights <= 0) or not np.isclose(weights.sum(), 1.):
        raise ValueError('Invalid normalization weights')
    mean = np.zeros(71, dtype=np.float64)
    second = mean.copy()
    frames = 0
    for r, weight in zip(records, weights):
        for view in 'ABCD':
            with np.load(Path(r['input_directory'])/(view+'_input.npz'), allow_pickle=False) as z:
                x = z['boundary_features_71'].astype(np.float64)
            if x.ndim != 2 or x.shape[1] != 71 or not len(x) or not np.isfinite(x).all():
                raise ValueError('Invalid normalization input')
            mean += weight/4 * x.mean(axis=0)
            second += weight/4 * np.square(x).mean(axis=0)
            frames += len(x)
    scale = np.sqrt(np.maximum(second-mean*mean, 0.))
    scale[scale < 1e-6] = 1.
    return dict(mean=mean.astype(np.float32), scale=scale.astype(np.float32), real_view_frames=frames)


def verify_record(r, cache=None):
    """No extraction or audit rerun: authenticate recorded source artifacts."""
    checked_training_records([r])
    cache = {} if cache is None else cache
    def check(path, expected):
        path = str(path)
        if path not in cache:
            cache[path] = digest(path)
        if cache[path] != expected:
            raise ValueError('Changed source artifact: '+path)
    for path, key in [('manifest_path', 'manifest_sha256'), ('target_path', 'targets_sha256'),
                      ('audit_path', 'audit_sha256'), ('audit_policy_path', 'audit_policy_sha256')]:
        check(r[path], r[key])
    if 'review_file_path' in r:
        check(r['review_file_path'], r['review_file_sha256'])
    for view in 'ABCD':
        check(r['waveform_paths'][view], r['waveform_sha256'][view])
        check(Path(r['input_directory'])/(view+'_input.npz'), r['input_sha256'][view])
    audit = json.loads(Path(r['audit_path']).read_text())
    policy = json.loads(Path(r['audit_policy_path']).read_text())
    if not audit['passed'] or audit['policy'] != policy:
        raise ValueError('Source audit or policy is invalid')
    if 'requested_cents' in audit and audit['requested_cents'] != r['cents']:
        raise ValueError('Source audit shift differs')
    validate_pair_record(r)
    with np.load(r['target_path'], allow_pickle=False) as z:
        common, plateau, zero = (z[k] for k in ('common_reliable_mask', 'detuned_plateau_mask', 'zero_anchor_mask'))
        if (not plateau.any() or not zero.any() or np.any(plateau & zero)
                or not np.array_equal(common, plateau | zero)
                or np.any(z['correction_target_cents'][~common] != 0)):
            raise ValueError('Invalid private target partition')


def align_record(record, folder):
    """Change only boundary column 71 in NEW files; keep source timing/70 features."""
    folder = Path(folder)
    folder.mkdir(parents=True)
    derived = dict(record)
    derived.update(source_input_directory=record['input_directory'], source_input_sha256=record['input_sha256'],
                   input_directory=str(folder), boundary_encoding='gaussian_max_sigma_0.04_seconds',
                   data_status=STATUS, training_eligible=False,
                   boundary_provenance='Timing endpoints from saved pair; shared Gaussian encoding; pitches excluded')
    changed = 0
    for view in 'ABCD':
        with np.load(Path(record['input_directory'])/(view+'_input.npz'), allow_pickle=False) as z:
            values = {k: z[k] for k in z.files}
        edges = record['nominal_boundary_seconds' if view in 'AC' else 'corrupted_boundary_seconds']
        score = game_boundary_track(values['times_seconds'], edges)
        changed += int(not np.array_equal(values['boundary_features_71'][:, 70], score))
        values['boundary_features_71'] = np.column_stack([values['frame_features_70'], score]).astype(np.float32)
        np.savez_compressed(folder/(view+'_input.npz'), **values)
    derived['input_sha256'] = {v: digest(folder/(v+'_input.npz')) for v in 'ABCD'}
    validate_pair_record(derived)
    return derived, changed


def comparison_protocol():
    return dict(version='gate-e-exploratory-comparison-1', data_status=STATUS, research_ready_examples=0,
        conditions=dict(augmentation_only=71, frame_only=70, bpdr=71),
        model=dict(hidden=128, layers=4, heads=4, feed_forward=512, dropout=.1),
        fit=dict(seed=339, updates=300, warmup=100, batch_size=8, learning_rate=.001,
                 optimizer='AdamW', gradient_clip_norm=1., checkpoint_every=25),
        loss_weights=dict(boundary=1., detuning=1., smoothness=.1),
        loss_reduction='Per-example losses first, then equal batch mean; never global masked-frame mean across examples',
        sampling='Equal dataset -> group -> recording -> phrase -> shift -> corruption; with replacement; common saved schedule',
        normalization='Same saved 71-column training-only weighted real-frame moments; frame-only takes first 70',
        initialization='Same seed and common-shape tensors; frame-only projection uses first 70 columns of 71-column initialization',
        sequence_context='Full real phrase, padded only within each small batch; no chunking or hidden crop change',
        augmentation='Existing audited shifts and corruptions only; no new audio, denoising or added noise',
        evaluation_batch_size=1,
        checkpoint_selection=dict(partition='development', direction='minimize', tie_break='earliest update',
            criterion='Dataset/group/recording/phrase/shift/corruption macro MAE on A/B zero and C/D injected correction, equal view mean',
            candidates='Saved updates 25,50,...,300; same checkpoints and prepared development examples for every condition',
            no_development_pairs='Stop selection and report unavailable; never substitute training fit or heldout accuracy'),
        development_preparation=dict(partitions='Existing frozen partitions only; no source reassignment',
            selection='All screen-eligible development exports and Vocadito development phrases; existing suitability policy unchanged',
            intervention='Existing v0.3 fixed supports/masks, all six shifts and three corruptions; retain every failure',
            boundary='Same Gaussian encoding; human-note and predicted-GAME timing metrics reported separately',
            no_change='Report A/B correction magnitude on joint voiced real frames, independent of injected-detuning masks; acceptable-singing claim requires future review',
            exclusions='Do not inspect PP1 evaluation or final-holdout acoustics or outcomes'),
        reporting='Per-dataset and per-group results plus explicitly equal-dataset macro; artist proxies are not verified singers',
        accuracy_scope='Relative synthetic-offset recovery; no claim of original intonation correctness or delivered musical benefit',
        training_updates_executed=0, development_processing_executed=False)


def run():
    root = Path(__file__).resolve().parents[2]
    index = json.loads((root/'notebooks/evidence_index.json').read_text())['evidence']
    sources = [('vocadito', root/index['exploratory']/'summary.json'),
               ('jamendo', root/index['shared_pair_summary'])]
    records = []
    for dataset, path in sources:
        for original in json.loads(path.read_text())['examples']:
            r = dict(original)
            r.update(dataset=dataset, group_id=dataset+':'+str(r['singer_id']),
                     group_kind='annotated_singer_id' if dataset == 'vocadito' else 'artist_proxy_singer_unknown',
                     reference_type='human_annotation' if dataset == 'vocadito' else 'model_estimate',
                     source_summary_path=str(path), source_summary_sha256=digest(path))
            records.append(r)
    checked_training_records(records)
    weights = balanced_weights(records)
    destination = root/'runs/comparison_preparation'/uuid.uuid4().hex
    destination.mkdir(parents=True)
    def write(name, value):
        (destination/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    prior = root/index['shared_pair_audit']
    previous = json.loads((prior.parent/'declaration.json').read_text())
    frozen = {str(path): digest(path) for _, path in sources}
    frozen.update(previous['frozen_inputs'])
    for name in ['comparison_preparation', 'examples', 'laboratory', 'model', 'losses', 'shared_pair_audit']:
        p = Path(__file__).with_name(name+'.py')
        frozen[str(p)] = digest(p)
        copied = destination/'source_code_at_run'; copied.mkdir(exist_ok=True)
        shutil.copy2(p, copied/p.name)
    pipeline = previous['pipeline_files']
    write('declaration.json', dict(frozen_inputs=frozen, pipeline_files=pipeline,
        ordered_source_pairs=[dict(dataset=r['dataset'], phrase_id=r['phrase_id'], cents=r['cents'], corruption=r['corruption']) for r in records],
        action='New derived views and training-only normalizer; no audio rendering, feature extraction or learner training',
        original_reviews_preserved=True, research_ready_examples=0))
    protocol = comparison_protocol()
    write('locked_protocol.json', protocol)
    print('PREPARATION_RUN:', destination, flush=True)
    cache = {}; derived = []; changed = defaultdict(int)
    for i, r in enumerate(records):
        verify_record(r, cache)
        folder = destination/'inputs'/r['dataset']/r['phrase_id']/f"shift_{r['cents']:+g}"/r['corruption']
        d, n = align_record(r, folder)
        derived.append(d); changed[r['dataset']] += n
        if (i+1) % 100 == 0:
            print('ALIGNED:', i+1, '/', len(records), flush=True)
    normalizer = fit_streaming_normalizer(derived, weights)
    frames = normalizer.pop('real_view_frames')
    np.savez_compressed(destination/'normalizer.npz', **normalizer)
    schedule = np.random.default_rng(protocol['fit']['seed']).choice(len(derived),
        size=(protocol['fit']['updates'], protocol['fit']['batch_size']), p=weights)
    np.savez_compressed(destination/'sampling_schedule.npz', indices=schedule, pair_probabilities=weights)
    write('summary.json', dict(mode=STATUS, examples=derived, research_ready_examples=0, training_eligible_pairs=0, run_status='complete'))
    for p, h in {**frozen, **pipeline, **cache}.items():
        if digest(p) != h:
            raise ValueError('Frozen artifact changed: '+p)
    artifacts = {str(p.relative_to(destination)): digest(p) for p in destination.rglob('*') if p.is_file()}
    write('evidence.json', dict(status='complete', data_status=STATUS, research_ready_examples=0, training_updates=0,
        engineering_pair_examples=len(derived), changed_boundary_views=dict(changed), real_view_frames=frames,
        datasets={dataset: dict(pairs=sum(r['dataset']==dataset for r in records),
            recordings=len({r['source_id'] for r in records if r['dataset']==dataset}),
            groups=len({r['group_id'] for r in records if r['dataset']==dataset}),
            probability_mass=float(sum(w for r,w in zip(records,weights) if r['dataset']==dataset))) for dataset,_ in sources},
        original_artifacts_verified=cache, artifacts=artifacts, pipeline_files_unchanged=len(pipeline),
        no_development_processing=True, no_model_training=True, no_new_audio=True))
    write('completion.json', dict(evidence_sha256=digest(destination/'evidence.json'), status='complete'))
    (root/'runs/comparison_preparation_current.json').write_text(json.dumps(dict(path=str(destination)),indent=2)+'\n',encoding='utf-8')
    print('PREPARATION_COMPLETE:', len(derived), dict(changed), flush=True)
    return destination


if __name__ == '__main__':
    run()
