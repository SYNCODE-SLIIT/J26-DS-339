"""Create separate pending reviewer sheets without changing original reviews or eligibility."""
from bpdr_input import COMPONENT_ROOT
import argparse
import csv
from datetime import datetime, timezone
from html import escape
import json
import os
from pathlib import Path
from urllib.parse import quote
import uuid
from bpdr_input.preparation.sources.data_builder import validate_run, write_json
from bpdr_input.preparation.sources.dataset import REVIEW_FIELDS
from bpdr_input.audio.pipeline import checksum

EXTRA_FIELDS = ['multiple_voices', 'accompaniment_leakage', 'unclear_transitions', 'edge_or_context_concerns']


def candidates(manifest):
    rows = [r for r in manifest['records'] if r['screen']['status'] == 'review_candidate']
    if len({r['record_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate review crop IDs')
    for row in rows:
        if row['split'] != 'train' or row['training_eligible'] or 'accepted_audio_path' not in row:
            raise ValueError('Only accepted exploratory training candidates can enter this bundle')
    return rows


def create(prepared):
    prepared = Path(prepared).resolve()
    component = COMPONENT_ROOT
    if not prepared.is_relative_to(component):
        raise ValueError('Preparation must be inside BPDR')
    verified = validate_run(prepared)
    manifest_path = prepared/'manifest.json'
    original_reviews = prepared/'reviews.csv'
    before = {str(p): checksum(p) for p in (manifest_path, original_reviews)}
    rows = candidates(json.loads(manifest_path.read_text()))
    root = component/'runs/review_bundles'/uuid.uuid4().hex
    root.mkdir(parents=True)
    fields = REVIEW_FIELDS+EXTRA_FIELDS
    for number in (1, 2):
        with (root/f'reviewer_{number}.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow(dict(phrase_id=row['record_id'], crop_sha256=row['sha256'],
                                     decision='pending', approved_intervals_json='[]'))
    instructions = '''# Independent reference review

Two different reviewers with relevant musical experience must complete one sheet each, independently. These are pending templates, not approvals or a blind effect study. Do not inspect the other reviewer's decisions before submitting your own.

1. Listen to the entire unmodified crop, preferably with headphones and consistent playback volume. Decide whether this is an acceptable single-voice reference for preservation, considering tuning stability, leakage, backing voices, unclear transitions and phrase/crop edges. Do not infer an intended tune or snap expressive singing to a scale.
2. Fill your actual reviewer ID/name and musical experience. Choose approve, reject or leave pending with a reason if a decision requires clarification. Approve only after listening; an automatic screen is not approval.
3. For an approval, enter stable agreed intervals in crop-relative seconds as JSON, for example [[1.2,2.4],[4.0,5.3]]. This is a format example, not suggested intervals. Keep endpoints within the displayed crop duration. Exclude ambiguous transitions, uncertain edges, leakage and unstable regions. These intervals are not automatically detuning supports.
4. Record a reason, the actual review timestamp in UTC (YYYY-MM-DDTHH:MM:SSZ), and observations in the four extra columns. Keep the crop ID and audio hash unchanged. Reject unsuitable material rather than forcing intervals to satisfy geometry.
5. Return both separate sheets through the owner's workflow. Disagreement needs resolution; approval requires two distinct actual identities, complete evidence and the intersection of valid intervals. Model estimates remain model_estimate even after an audio reference is approved.

Do not edit the original preparation reviews or manifest. This bundle has no automatic approval/import/training path. Approved intervals still need integrity checks and realized-shift audits; research eligibility remains zero. No sheet is transmitted to anyone by this tool.
'''
    (root/'instructions.md').write_text(instructions, encoding='utf-8')
    cards = []
    for row in rows:
        audio = Path(row['accepted_audio_path']).resolve()
        if not audio.is_relative_to(component) or checksum(audio) != row['sha256']:
            raise ValueError('Review audio leaves component or changed')
        url = quote(Path(os.path.relpath(audio, root)).as_posix(), safe='/')
        cards.append('<article><h2>'+escape(row['record_id'])+'</h2><p>'+escape(row['dataset'])+' / '+escape(row['reference_kind'])+
                     ' / '+escape(row['group_id'])+'</p><audio controls preload="none" src="'+url+'"></audio><p>Crop duration: '+
                     str(round(row['duration_seconds'], 3))+' seconds. Original offset: '+str(row['source_map']['original_offset_seconds'])+
                     ' seconds. No musical repair.</p><p>Enter intervals relative to this crop, starting at 0 seconds.</p><details><summary>Accepted audio identity</summary><code>'+row['sha256']+'</code></details></article>')
    html = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>BPDR independent review</title>
<style>body{font:17px/1.6 system-ui;max-width:950px;margin:35px auto;padding:0 22px;background:#f4f6f8;color:#17212c}article{background:white;border:1px solid #d7dde3;border-radius:10px;padding:22px;margin:20px 0}audio{width:100%}code{overflow-wrap:anywhere}a{color:#145e99}.notice{background:#fff5dd;border-left:4px solid #ad6d00;padding:14px}</style>
<h1>Independent musical reference review</h1><p class="notice">These crops passed a provisional automatic screen. All decisions remain pending and zero examples are research-ready. Reviewers supply actual independent decisions; this page cannot approve data.</p>
<p><a href="instructions.md">Review instructions</a> · <a href="reviewer_1.csv">First reviewer's blank sheet</a> · <a href="reviewer_2.csv">Second reviewer's blank sheet</a></p>'''
    (root/'listen.html').write_text(html+''.join(cards)+'</html>', encoding='utf-8')
    if any(checksum(path) != digest for path,digest in before.items()):
        raise ValueError('Original manifest/reviews changed during packaging')
    package = dict(schema_version='independent-review-bundle-0.1', run_status='complete',
                   created_at_utc=datetime.now(timezone.utc).isoformat(), data_status='exploratory_not_research',
                   preparation_path=str(prepared), source_hashes=before, validation=verified,
                   candidates=[dict(record_id=r['record_id'], dataset=r['dataset'], group_id=r['group_id'], reference_kind=r['reference_kind'],
                                    audio_path=r['accepted_audio_path'], audio_sha256=r['sha256'], duration_seconds=r['duration_seconds']) for r in rows],
                   pending_review_rows=2*len(rows), training_ready_examples=0,
                   artifacts=[dict(path=str(p), sha256=checksum(p)) for p in root.iterdir() if p.is_file()],
                   importer_status='No automatic importer; original eligibility gates remain required')
    write_json(root/'manifest.json', package)
    print('REVIEW_BUNDLE:', root, flush=True)
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('preparation', type=Path)
    args = parser.parse_args()
    create(args.preparation)


if __name__ == '__main__':
    main()
