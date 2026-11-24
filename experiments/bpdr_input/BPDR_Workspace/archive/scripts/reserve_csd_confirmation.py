"""Choose independent confirmation from published score metadata, no audio."""
import io
import json
from pathlib import Path
import re
import struct
import sys

import mido

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE/'source'))
from bpdr_input.preparation.sources import vocalset_subset as z

ROOT = WORKSPACE/'results/transformer_usability/confirmation_sources_2026-10-29'


def main():
    declaration = json.loads((ROOT/'csd_confirmation_declaration.json').read_text())
    rows = json.loads((ROOT/'csd_archive_directory.json').read_text())
    record = json.loads((ROOT/'csd_official_record.json').read_text())
    archive = record['files'][0]
    assert record['metadata']['license']['id'] == 'cc-by-nc-sa-4.0'
    z.EXPECTED_ARCHIVE_SIZE = archive['size']
    z.ARCHIVE = 'https://zenodo.org/records/4916302/files/CSD.zip?download=1'
    z.MAX_BYTES = 2*1024*1024
    selected_mid = sorted([r for r in rows if re.fullmatch(r'CSD/english/mid/en\d{3}a.mid', r['name'])], key=lambda r:r['name'])
    assert len(selected_mid) == 50
    assert all(r['decoded_bytes'] < 128*1024 for r in selected_mid)
    # Fetch a single contiguous MIDI-only extent. Reject any interleaved audio
    # rather than accidentally accessing it before the development gate.
    start = min(r['local_header'] for r in selected_mid)
    last = max(selected_mid, key=lambda r:r['local_header'])
    source = z.BoundedSource()
    header = source.part(last['local_header'], last['local_header']+29)
    fields = struct.unpack('<4s5H3L2H', header)
    assert fields[0] == b'PK\x03\x04'
    end = last['local_header']+30+fields[-2]+fields[-1]+last['compressed_bytes']-1
    assert end-start < 1024*1024
    assert all(r['name'].endswith('.mid') for r in rows
               if start <= r['local_header'] <= end and r['compressed_bytes'] > 0)
    block = source.part(start, end)
    class CachedMidiExtent:
        def part(self, a, b):
            if not start <= a <= b <= end:
                raise ValueError('Requested data outside MIDI-only extent')
            return block[a-start:b-start+1]
    cache = CachedMidiExtent()
    metadata, chosen = [], []
    out = ROOT/'csd_score_metadata';out.mkdir(exist_ok=True)
    for row in selected_mid:
        payload = z.member(cache, row)
        (out/Path(row['name']).name).write_bytes(payload)
        clock, active, notes = 0., {}, []
        for message in mido.MidiFile(file=io.BytesIO(payload)):
            clock += message.time
            if message.type == 'note_on' and message.velocity > 0:
                active[(message.channel, message.note)] = clock
            elif message.type == 'note_off' or message.type == 'note_on' and message.velocity == 0:
                key = (message.channel, message.note)
                if key in active:
                    begin = active.pop(key)
                    notes.append(dict(start=begin, end=clock, duration=clock-begin))
        qualifying = [n for n in notes if n['duration'] >= 1.8 and n['start'] >= .4 and n['end'] <= 19.6]
        item = dict(name=row['name'], sha256=z.sha256(payload), crc32_verified=True,
                    qualifying_score_notes=qualifying, qualifies=bool(qualifying))
        metadata.append(item)
        if qualifying and len(chosen) < 3:
            wav_name = row['name'].replace('/mid/', '/wav/').replace('.mid', '.wav')
            wav = next(r for r in rows if r['name'] == wav_name)
            chosen.append(dict(singer_id=declaration['singer_id'], dataset='CSD1.1',
                               source_entry=wav, score_metadata=item, crop_seconds=[0.,20.],
                               role='reserved_confirmation_only', audio_accessed=False))
    receipt = dict(selected=chosen, selected_count=len(chosen), required=3,
                   source_rule_satisfied=len(chosen)==3, metadata_rows=metadata,
                   declaration_sha256=z.sha256((ROOT/'csd_confirmation_declaration.json').read_bytes()),
                   archive_declared_md5=archive['checksum'], full_archive_md5_verified=False,
                   network_bytes=source.bytes, requests=source.requests,
                   audio_downloaded=False, audio_analyzed=False, model_inference=False)
    (ROOT/'csd_reservation.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print('CSD_METADATA_RESERVATION', len(chosen), [r['source_entry']['name'] for r in chosen],
          'metadata transfer',source.bytes,'audio not accessed',flush=True)


if __name__ == '__main__':main()
