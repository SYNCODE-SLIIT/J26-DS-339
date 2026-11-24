"""Prospective vowel-i supplement for every fixed development/final singer."""
from datetime import datetime,timezone
import io
import json
from pathlib import Path
import sys
import soundfile as sf
WORKSPACE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WORKSPACE/'source'))
from bpdr_input.preparation.sources.vocalset_subset import BoundedSource,member,sha256


if __name__=='__main__':
    root=WORKSPACE/'datasets/vocalset_evaluation_i_2026-10-28';root.mkdir(exist_ok=True)
    directory=json.loads((WORKSPACE/'results/transformer_usability/archive_directory.json').read_text())
    selected=[]
    for gender in ('female','male'):
        for number in (5,6,8,9):
            singer=f'{gender}{number}';matches=[r for r in directory if r['name'].startswith(f'FULL/{singer}/long_tones/straight/') and r['name'].endswith('_i.wav')]
            if len(matches)!=1:raise ValueError('Predetermined source unavailable: '+singer)
            selected.append(dict(matches[0],singer_id=singer,vowel='i',split='historical_development' if number<=6 else 'final_reserved'))
    def write(path,obj):path.write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8')
    if not (root/'declaration.json').exists():write(root/'declaration.json',dict(created_at=datetime.now(timezone.utc).isoformat(),reason='Include vowels a/e/i as specified in the master protocol; supplement every fixed singer before new evaluation target creation, no source replacement',selected=selected,license='CC-BY-4.0',doi='10.5281/zenodo.1193957',source_outcomes_used=False,maximum_transfer_bytes=80*1024*1024))
    manifest=json.loads((root/'manifest.json').read_text()) if (root/'manifest.json').exists() else dict(records=[],status='IN_PROGRESS')
    source=BoundedSource()
    for row in selected:
        if any(r['source_entry']==row['name'] for r in manifest['records']):continue
        payload=member(source,row);info=sf.info(io.BytesIO(payload));path=root/row['singer_id']/Path(row['name']).name;path.parent.mkdir(exist_ok=True);path.write_bytes(payload)
        manifest['records'].append(dict(source_entry=row['name'],path=str(path),sha256=sha256(payload),singer_id=row['singer_id'],split=row['split'],vowel='i',bytes=len(payload),duration_seconds=info.duration,archive_crc32_verified=True))
        write(root/'manifest.json',manifest);print('EVALUATION_VOWEL_I',row['singer_id'],flush=True)
    manifest.update(status='COMPLETE',transferred_bytes=source.bytes,requests=source.requests);write(root/'manifest.json',manifest)
    (root/'ATTRIBUTION.txt').write_text('VocalSet 1.0, Wilkins, Seetharaman, Wahl, Pardo (2018). https://doi.org/10.5281/zenodo.1193957\nCC BY 4.0; source archive identity and ZIP CRC checked by bounded reader.\n',encoding='utf-8')
