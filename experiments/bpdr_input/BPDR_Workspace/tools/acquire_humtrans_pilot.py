"""Acquire only reserved training/development WAV bytes, never confirmation.

Requires a successful real pinned-metadata audit and upstream license snapshot.
No waveform decoding, feature extraction, MIDI access or fitting occurs here.
"""
import hashlib
import json
from pathlib import Path, PurePosixPath
import zlib

import audit_humtrans_metadata as metadata
from bounded_remote_zip import RangeArchive, open_archive, read_reserved_member

ROOT = metadata.ROOT
BUDGET = 128 * 1024 * 1024


def sha(data):
    return hashlib.sha256(data).hexdigest()


def reservation():
    if ROOT == metadata.ROOT and (metadata.WS / 'datasets/humtrans/intake_declaration.json').exists():
        raise ValueError('Existing HumTrans roles take precedence; provisional pilot acquisition is disabled')
    audit_path = ROOT / 'audit_and_reservation.json'
    raw = (ROOT / 'train_valid_test_keys.json').read_bytes()
    audit_bytes = audit_path.read_bytes()
    saved = json.loads(audit_bytes)
    license_bytes = (ROOT / 'README.upstream.md').read_bytes()
    if b'license: cc-by-nc-4.0' not in license_bytes:
        raise ValueError('Pinned upstream license snapshot missing')
    if saved['dataset_revision'] != metadata.REVISION or saved['metadata_sha256'] != sha(raw):
        raise ValueError('Metadata identity mismatch')
    if saved['auditor_sha256'] != sha(Path(metadata.__file__).read_bytes()):
        raise ValueError('Auditor changed after reservation; version and review explicitly')
    replay = metadata.audit(json.loads(raw))
    for key, value in replay.items():
        if saved[key] != value:
            raise ValueError(f'Reservation audit replay differs: {key}')
    # Confirmation is intentionally absent from the output selection.
    selected = [(role, row) for role in ('train', 'development') for row in saved['reservation'][role]]
    if len(selected) != 56 or {row['person'] for _, row in selected} & {'F05', 'M05'}:
        raise ValueError('Unexpected pilot selection or confirmation access')
    return sha(audit_bytes), sha(license_bytes), selected


def persist(value):
    temporary = ROOT / 'pilot_acquisition_receipt.pending.json'
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(ROOT / 'pilot_acquisition_receipt.json')


def main():
    audit_hash, license_hash, selected = reservation()  # before any network call
    receipt_path = ROOT / 'pilot_acquisition_receipt.json'
    prior = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    if prior and prior['reservation_sha256'] != audit_hash:
        raise ValueError('Existing acquisition belongs to a different reservation')
    prior_bytes = prior.get('network_bytes', 0)
    receipt = dict(status='IN_PROGRESS', reservation_sha256=audit_hash,
                   upstream_readme_sha256=license_hash, dataset_revision=metadata.REVISION,
                   source_sha256=sha(Path(__file__).read_bytes()),
                   range_reader_sha256=sha((ROOT.parents[2] / 'tools/bounded_remote_zip.py').read_bytes()),
                   records=[], network_bytes=prior_bytes, requests=prior.get('requests', []),
                   audio_decoded=False, confirmation_audio_acquired=False,
                   model_trained=False, production_eligible=False, research_validation_passed=False)
    source = None
    try:
        source = RangeArchive(metadata.BASE + 'all_wav.zip', budget=BUDGET - prior_bytes)
        with open_archive(source) as archive:
            by_stem = {}
            for entry in archive.infolist():
                path = PurePosixPath(entry.filename)
                if path.suffix.lower() == '.wav':
                    by_stem.setdefault(path.stem, []).append(entry)
            # Validate all reserved names and sizes before any selected audio read.
            entries = []
            for role, row in selected:
                matches = by_stem.get(row['key'], [])
                if len(matches) != 1:
                    raise ValueError(f'Reserved entry missing/ambiguous: {row["key"]}')
                entry = matches[0]
                if entry.file_size > 16 * 1024 * 1024 or entry.compress_size > 8 * 1024 * 1024:
                    raise ValueError('Reserved entry exceeds limit; no favorable replacement')
                entries.append((role, row, entry))
            for role, row, entry in entries:
                target = ROOT / 'private_audio' / role / (row['key'] + '.wav')
                if target.exists():
                    data = target.read_bytes()
                    if len(data) != entry.file_size or zlib.crc32(data) != entry.CRC:
                        raise ValueError('Existing file differs from pinned archive entry')
                    reused = True
                else:
                    data = read_reserved_member(archive, entry.filename)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(data)
                    reused = False
                receipt['records'].append(dict(role=role, key=row['key'], entry=entry.filename,
                                                bytes=len(data), crc32=entry.CRC, sha256=sha(data),
                                                reused_crc_checked_file=reused, path=str(target.resolve())))
                receipt['network_bytes'] = prior_bytes + source.transferred
                receipt['requests'] = prior.get('requests', []) + source.requests
                persist(receipt)
                print('HUMTRANS_RESERVED_WAV_BYTES', role, row['key'], len(data), flush=True)
        receipt['status'] = 'COMPLETE'
    except Exception as error:
        receipt['status'] = 'FAILED'
        receipt['failure_type'] = type(error).__name__
        if source is not None:
            receipt['network_bytes'] = prior_bytes + source.transferred
            receipt['requests'] = prior.get('requests', []) + source.requests
        persist(receipt)
        raise
    finally:
        if source is not None:
            source.close()
    persist(receipt)
    print('HUMTRANS_RESERVED_PILOT_ACQUIRED', len(receipt['records']), receipt['network_bytes'], flush=True)


if __name__ == '__main__':
    main()
