"""Acquire pinned small metadata only; audit official leakage and reserve grouped roles.

Never downloads audio/MIDI or interprets dataset labels as performed-pitch truth.
An offline --metadata path is accepted for reproducible audits after acquisition.
"""
import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import re

WS = Path(__file__).resolve().parents[1]
ROOT = WS / 'results/transformer_usability/2026-11-02_v48_humtrans_metadata'
REVISION = 'c3c8504280cbdef0f00bb8119f16f034bea2bd9f'
BASE = f'https://huggingface.co/datasets/dadinghh2/HumTrans/resolve/{REVISION}/'
PATTERN = re.compile(r'^(F0[1-5]|M0[1-5])_(\d{4})_(\d{4})_([12])(?:_(U|D|DD|DDD))?$')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write(path, value):
    data = json.dumps(value, indent=2, allow_nan=False) + '\n'
    if path.exists() and path.read_text(encoding='utf-8') != data:
        raise ValueError('Refuse to overwrite an existing reservation; create a new version')
    path.write_text(data, encoding='utf-8')


def acquire(name, limit):
    import requests
    # Strictly bounded, including when a remote endpoint ignores size hints.
    with requests.get(BASE + name, stream=True, timeout=(10, 30)) as response:
        response.raise_for_status()
        if int(response.headers.get('Content-Length', 0)) > limit:
            raise ValueError('Metadata response exceeds byte budget')
        data = bytearray()
        for chunk in response.iter_content(16384):
            data.extend(chunk)
            if len(data) > limit:
                raise ValueError('Metadata response exceeds byte budget')
        return bytes(data)


def audit(splits, reserve=True):
    if set(splits) != {'TRAIN', 'VALID', 'TEST'}:
        raise ValueError('Unexpected official split schema')
    parsed = {}
    all_keys = set()
    for split, keys in splits.items():
        if not isinstance(keys, list) or not keys:
            raise ValueError('Each official split must be a nonempty list')
        parsed[split] = []
        for key in keys:
            if not isinstance(key, str) or not (match := PATTERN.fullmatch(key)):
                raise ValueError(f'Unrecognized key: {key!r}')
            if key in all_keys:
                raise ValueError(f'Duplicate recording key: {key}')
            all_keys.add(key)
            person, music, segment, repetition, octave = match.groups()
            parsed[split].append(dict(key=key, person=person, music=music,
                                      segment=segment, repetition=repetition, octave=octave))
    overlap = {}
    for left, right in itertools.combinations(sorted(parsed), 2):
        overlap[f'{left}/{right}'] = {}
        for name, fields in [('people', ('person',)), ('compositions', ('music',)),
                             ('repeated_segment_families', ('person', 'music', 'segment'))]:
            sets = [{tuple(r[f] for f in fields) for r in parsed[s]} for s in (left, right)]
            overlap[f'{left}/{right}'][name] = len(sets[0] & sets[1])
    inventory = dict(official_counts={s: len(rows) for s, rows in parsed.items()},
                     official_people={s: dict(Counter(r['person'] for r in rows)) for s, rows in parsed.items()},
                     official_overlap=overlap)
    if not reserve:
        return inventory
    people = sorted({r['person'] for rows in parsed.values() for r in rows})
    expected = [f'{prefix}0{i}' for prefix in ('F', 'M') for i in range(1, 6)]
    if people != expected:
        raise ValueError('Participant inventory differs; explicitly review reservation policy')
    # Fixed identity roles before audio inspection. Both singer and composition
    # must agree with a role; conflicting cross-cells are excluded, not reassigned.
    roles = {'train': {'F01', 'F02', 'F03', 'M01', 'M02', 'M03'},
             'development': {'F04', 'M04'}, 'confirmation': {'F05', 'M05'}}
    pool = {role: [] for role in roles}
    excluded = 0
    for official, rows in parsed.items():
        for row in rows:
            bucket = int(digest(('BPDR-HumTrans-339:' + row['music']).encode())[:8], 16) % 100
            role = 'train' if bucket < 80 else ('development' if bucket < 90 else 'confirmation')
            if row['person'] not in roles[role]:
                excluded += 1
                continue
            pool[role].append(dict(row, official_split=official))
    # Reserve IDs only. Later audio acquisition must use this frozen list, without
    # selecting/replacing examples based on pitch quality or correction outcome.
    selection = {}
    for role, persons in roles.items():
        selected = []
        for person in sorted(persons):
            candidates = sorted((r for r in pool[role] if r['person'] == person),
                                key=lambda r: digest(('BPDR-pilot-339:' + r['key']).encode()))
            seen_music = set()
            for row in candidates:
                if row['music'] in seen_music:
                    continue
                seen_music.add(row['music'])
                selected.append(row)
                if len(seen_music) == (8 if role == 'train' else 4):
                    break
            if len(seen_music) != (8 if role == 'train' else 4):
                raise ValueError(f'Insufficient unique compositions for {role}/{person}')
        selection[role] = selected
    for left, right in itertools.combinations(selection, 2):
        for field in ('person', 'music'):
            if {r[field] for r in selection[left]} & {r[field] for r in selection[right]}:
                raise AssertionError('Grouped holdout leakage')
    return dict(official_counts={s: len(rows) for s, rows in parsed.items()},
                official_people={s: dict(Counter(r['person'] for r in rows)) for s, rows in parsed.items()},
                official_overlap=overlap, reservation=selection,
                grouped_pool_counts={role: len(rows) for role, rows in pool.items()},
                cross_cell_exclusions=excluded,
                split_policy='New grouped BPDR protocol, not the official benchmark split; all official roles remain recorded.',
                audio_acquired=False, audio_analyzed=False, model_trained=False,
                production_eligible=False, research_validation_passed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', type=Path)
    args = parser.parse_args()
    if (WS / 'datasets/humtrans/intake_declaration.json').exists():
        raise ValueError('Existing HumTrans roles take precedence; use reconcile_local_humtrans.py, not a new reservation')
    ROOT.mkdir(parents=True, exist_ok=True)
    if args.metadata:
        data = args.metadata.read_bytes()
        origin = str(args.metadata.resolve())
    else:
        readme = acquire('README.md', 16384)
        if b'license: cc-by-nc-4.0' not in readme:
            raise ValueError('Expected academic noncommercial license missing; stop')
        data = acquire('train_valid_test_keys.json', 1048576)
        (ROOT / 'README.upstream.md').write_bytes(readme)
        origin = BASE + 'train_valid_test_keys.json'
    result = audit(json.loads(data))
    result.update(dataset_revision=REVISION, metadata_origin=origin,
                  metadata_sha256=digest(data), auditor_sha256=digest(Path(__file__).read_bytes()))
    (ROOT / 'train_valid_test_keys.json').write_bytes(data)
    write(ROOT / 'audit_and_reservation.json', result)
    print('HUMTRANS_METADATA_AUDIT_COMPLETE', result['official_counts'],
          {k: len(v) for k, v in result['reservation'].items()}, flush=True)


if __name__ == '__main__':
    main()
