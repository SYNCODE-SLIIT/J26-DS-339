"""Current review metadata around the unchanged historical export contract."""

import io
import json
import zipfile

from bpdr_input.repair.demo_review import export_reviewed as legacy_export


def export_reviewed(directory, report, assessment, *, use_preview=False,
                    confirmed=False, confirmation_kind='user'):
    summary = report.get('delivery_summary')
    if summary:
        item = (report.get('noise_candidate') or report.get('candidate')) if use_preview else report['handoff']['accepted_audio']
        expected = (item.get('sha256') or item.get('checksum')) if item else None
        if summary.get('selected_role') != ('preview' if use_preview else 'original') or summary.get('selected_sha256') != expected:
            raise ValueError('The waveform summary belongs to a different selection.')
    destination, handoff = legacy_export(directory, report, assessment,
        use_preview=use_preview, confirmed=confirmed, confirmation_kind=confirmation_kind)
    path = destination / 'diagnostics.json'
    diagnostics = json.loads(path.read_text(encoding='utf-8'))
    diagnostics['user_protection'] = report.get('user_protection')
    diagnostics['local_selection'] = report.get('local_selection')
    diagnostics['recording_feedback'] = report.get('recording_feedback')
    diagnostics['delivery_summary'] = summary
    path.write_text(json.dumps(diagnostics, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    portable = dict(handoff, accepted_audio=dict(handoff['accepted_audio'], path='accepted.wav'))
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.write(destination / 'accepted.wav', 'accepted.wav')
        archive.writestr('handoff.json', json.dumps(portable, indent=2))
        archive.write(path, 'diagnostics.json')
    (destination / 'accepted_package.zip').write_bytes(buffer.getvalue())
    return destination, handoff
