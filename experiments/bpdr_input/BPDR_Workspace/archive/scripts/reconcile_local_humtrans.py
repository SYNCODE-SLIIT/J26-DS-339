"""Reconcile relocated pre-existing HumTrans intake, without decoding any audio.

Retain historical roles and prior use. Never relabel known people as unseen.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path

import audit_humtrans_metadata as metadata

WS = Path(__file__).resolve().parents[1]
DATA = WS / 'datasets/humtrans'
ROOT = WS / 'results/transformer_usability/2026-11-02_v49_local_humtrans'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def relocate(old):
    marker = '/data/humtrans/'
    normalized = old.replace('\\', '/')
    if marker not in normalized:
        raise ValueError('Unexpected historical intake path')
    target = (DATA / normalized.split(marker, 1)[1]).resolve()
    target.relative_to(DATA.resolve())
    if not target.is_file():
        raise FileNotFoundError(target)
    return target


def main():
    declaration = read(DATA / 'intake_declaration.json')
    receipt = read(DATA / 'intake_receipt.json')
    repository = read(DATA / 'repository_metadata.json')
    if repository['sha'] != declaration['revision'] or declaration['revision'] != metadata.REVISION:
        raise ValueError('Historical repository revision mismatch')
    if sha(DATA / 'README.md') != receipt['license_card_sha256']:
        raise ValueError('License card hash mismatch')
    if b'license: cc-by-nc-4.0' not in (DATA / 'README.md').read_bytes():
        raise ValueError('Noncommercial license missing')
    if sha(DATA / 'intake_declaration.json') != receipt['declaration_sha256']:
        raise ValueError('Historical intake declaration hash mismatch')
    if sha(DATA / 'all_midi.zip') != receipt['midi_zip_sha256']:
        raise ValueError('Offline reference archive hash mismatch')
    extension = DATA / 'extensions/f80ddac6a5c54e869f5fb559acd74029'
    ext_declaration = read(extension / 'declaration.json')
    ext_receipt = read(extension / 'extension_only_receipt.json')
    if ext_declaration['original_intake_sha256'] != sha(DATA / 'intake_receipt.json'):
        raise ValueError('Extension parent receipt mismatch')
    expected = {r['key']: r for r in declaration['selection'] + ext_declaration['selection']}
    supplied = receipt['records'] + ext_receipt['records']
    if len(supplied) != len(expected) or {r['key'] for r in supplied} != set(expected):
        raise ValueError('Missing or duplicate historical recording')
    rows = []
    for row in supplied:
        declared = expected[row['key']]
        for key in ('key', 'split', 'person_id', 'composition_id', 'official_split'):
            if row[key] != declared[key]:
                raise ValueError('Historical source role differs from declaration')
        if declaration['person_roles'][row['person_id']] != row['split']:
            raise ValueError('Historical person role conflict')
        target = relocate(row['path'])
        midi = relocate(row['midi_path'])
        if sha(target) != row['sha256'] or sha(midi) != row['midi_sha256']:
            raise ValueError('Relocated source hash differs')
        rows.append(dict(key=row['key'], person_id=row['person_id'], composition_id=row['composition_id'],
                         split=row['split'], path=str(target), sha256=row['sha256'],
                         midi_path=str(midi), midi_sha256=row['midi_sha256'],
                         prior_use='Prior engineering training/development; not fresh confirmation',
                         selection='initial lexicographic' if row['key'] in {r['key'] for r in declaration['selection']} else 'score-selected long-note development extension'))
    train = [r for r in rows if r['split'] == 'train']
    dev = [r for r in rows if r['split'] == 'development']
    for field in ('person_id', 'composition_id'):
        if {r[field] for r in train} & {r[field] for r in dev}:
            raise ValueError('Historical training/development overlap')
    official = metadata.audit(read(DATA / 'train_valid_test_keys.json'), reserve=False)
    official_train = set(read(DATA / 'train_valid_test_keys.json')['TRAIN'])
    if not all(r['key'] in official_train for r in rows):
        raise ValueError('Historical official split mismatch')
    ROOT.mkdir(parents=True, exist_ok=True)
    report = dict(version='local-humtrans-reconciliation-1', source_sha256=sha(__file__),
                  inputs_sha256={str(p): sha(p) for p in [DATA / 'intake_declaration.json', DATA / 'intake_receipt.json',
                      DATA / 'README.md', DATA / 'repository_metadata.json', DATA / 'train_valid_test_keys.json',
                      extension / 'declaration.json', extension / 'extension_only_receipt.json']},
                  records=rows, counts=dict(Counter(r['split'] for r in rows)),
                  historical_person_roles=declaration['person_roles'], official_audit=official,
                  provisional_v48_roles_superseded=True, source_path_remap_only=True,
                  audio_decoded=False, reference_notes_decoded=False, new_model_trained=False,
                  human_reference_approved=False, research_training_eligible=False,
                  production_eligible=False, research_validation_passed=False)
    (ROOT / 'reconciliation.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print('LOCAL_HUMTRANS_HASH_ROLE_RECONCILIATION_PASS', len(rows), report['counts'], flush=True)
    print('OFFICIAL_SPLIT_OVERLAP', official['official_overlap'], flush=True)


if __name__ == '__main__':
    main()
