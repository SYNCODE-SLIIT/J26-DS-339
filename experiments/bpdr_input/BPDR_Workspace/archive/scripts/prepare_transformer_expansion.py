"""Prospective, bounded source expansion; final singer audio is never analyzed."""
import argparse
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import sys
import soundfile as sf

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WORKSPACE/'source'))
from bpdr_input.preparation.sources.vocalset_subset import BoundedSource, member, sha256, EXPECTED_ARCHIVE_SIZE, EXPECTED_ARCHIVE_MD5

ROOT=WORKSPACE/'results/transformer_usability'
DATA=WORKSPACE/'datasets/vocalset_expansion_2026-10-28'


def write(path,obj):
    path.write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8')


def main():
    DATA.mkdir(exist_ok=True)
    directory=json.loads((ROOT/'archive_directory.json').read_text())
    selected=[]
    for gender in ('female','male'):
        for number in (1,2,3,4,7,8,9):
            singer=f'{gender}{number}'
            split='train' if number in (1,2,3,4,7) else 'development' if number==8 else 'final_reserved'
            vowels=('i','o','u') if number<=4 else ('a','e','i','o','u') if number==7 else ('a','e')
            for vowel in vowels:
                matches=sorted([r for r in directory if r['name'].startswith(f'FULL/{singer}/long_tones/straight/') and r['name'].endswith(f'_{vowel}.wav')],key=lambda r:r['name'])
                if not matches: raise ValueError(f'Missing predetermined source {singer}/{vowel}')
                selected.append(dict(matches[0],singer_id=singer,split=split,vowel=vowel))
    assert len(selected)==42
    declaration=DATA/'declaration.json'
    if not declaration.exists():
        write(declaration,dict(created_at=datetime.now(timezone.utc).isoformat(),
            dataset='VocalSet 1.0',doi='10.5281/zenodo.1193957',license='CC-BY-4.0',
            archive_bytes=EXPECTED_ARCHIVE_SIZE,archive_md5=EXPECTED_ARCHIVE_MD5,
            maximum_transfer_bytes=80*1024*1024,selected=selected,
            prior_use='Existing IDs 1-4 remain training; existing 5/6 remain historical development only. Local manifests contained no IDs 7-9 before reservation.',
            split_rule='IDs 1-4 additional i/o/u and ID7 all vowels train; ID8 a/e development; ID9 a/e reserved final; never replace an excluded source',
            final_access_rule='Download/CRC/hash/duration inventory only; no acoustic extraction, target creation or inference until development gates pass and all endpoints are locked',
            scope='Straight sustained vowels; recovery toward original performance, not intended-note truth'))
    else:
        assert json.loads(declaration.read_text())['selected']==selected
    receipt_path=DATA/'manifest.json'
    receipt=json.loads(receipt_path.read_text()) if receipt_path.exists() else dict(status='IN_PROGRESS',records=[])
    saved={r['source_entry']:r for r in receipt['records']}
    source=BoundedSource()
    metadata,raw=source.metadata()
    archive=next(x for x in metadata['files'] if x['key']=='VocalSet.zip')
    assert archive['size']==EXPECTED_ARCHIVE_SIZE and archive['checksum']=='md5:'+EXPECTED_ARCHIVE_MD5
    assert metadata['metadata']['license']['id']=='cc-by-4.0'
    (DATA/'official_record.json').write_bytes(raw)
    for i,row in enumerate(selected,1):
        if row['name'] in saved:
            assert sha256(Path(saved[row['name']]['path']).read_bytes())==saved[row['name']]['sha256']
            continue
        payload=member(source,row)
        info=sf.info(io.BytesIO(payload))
        path=DATA/'originals'/row['split']/row['singer_id']/Path(row['name']).name
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(payload)
        receipt['records'].append(dict(source_entry=row['name'],singer_id=row['singer_id'],split=row['split'],vowel=row['vowel'],path=str(path),sha256=sha256(payload),bytes=len(payload),duration_seconds=info.duration,archive_crc32_verified=True))
        receipt.update(bytes_this_session=source.bytes,requests_this_session=source.requests)
        write(receipt_path,receipt)
        print('DOWNLOADED',i,'/42',row['singer_id'],row['vowel'],row['split'],flush=True)
    receipt.update(status='COMPLETE',recordings=len(receipt['records']),declaration_sha256=sha256(declaration.read_bytes()))
    write(receipt_path,receipt)
    (DATA/'ATTRIBUTION.txt').write_text('VocalSet 1.0, Julia Wilkins, Prem Seetharaman, Alison Wahl and Bryan Pardo (2018).\nhttps://doi.org/10.5281/zenodo.1193957\nCC BY 4.0. Originals retained; controlled derived interventions are separate.\n',encoding='utf-8')
    print('COMPLETE',DATA,flush=True)


if __name__=='__main__': main()
