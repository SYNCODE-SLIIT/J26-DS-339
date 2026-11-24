"""Acquire only prospectively reserved public confirmation recordings, no acoustic analysis."""
import argparse,json,re,stat,struct,zlib
from datetime import datetime,timezone
from pathlib import Path
import sys
WORKSPACE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WORKSPACE/'source'))
from bpdr_input.preparation.sources import vocalset_subset as zip_source

ROOT=WORKSPACE/'results/transformer_usability/2026-11-01_v37_fresh_confirmation'
JSUT_URL='https://ss-takashi.sakura.ne.jp/corpus/jsut-song_ver1.zip'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return zip_source.sha256(Path(path).read_bytes())
def write(path,value):zip_source.write_json(Path(path),value)


def check_lock():
    lock=read(ROOT/'pipeline_lock.json');cohort=read(ROOT/'cohort_reservation.json');closure=read(ROOT/'component_source_lock.json')
    assert cohort['pipeline_lock_sha256']==sha(ROOT/'pipeline_lock.json')==closure['pipeline_lock_sha256']
    assert closure['cohort_reservation_sha256']==sha(ROOT/'cohort_reservation.json')
    assert sha(lock['model_checkpoint'])==lock['model_checkpoint_sha256']
    for kind in (lock['source_and_input_hashes'],closure['python_source_hashes']):
        for p,h in kind.items():assert sha(p)==h,p
    dev=ROOT.parent/'2026-11-01_v36_centered_development/integrity_receipt.json'
    assert sha(dev)==lock['known_development_integrity_sha256'] and read(dev)['qualified_on_known_scoreable_scope']
    return cohort


class Source(zip_source.BoundedSource):
    def __init__(self,blob=None):
        super().__init__();self.blob=blob;self.blocks={};self.capture=True
    def part(self,start,end):
        if self.blob is None:data=super().part(start,end)
        else:
            assert 0<=start<=end<len(self.blob);data=self.blob[start:end+1]
            self.requests.append(dict(start=start,end=end,bytes=len(data),from_private_zip_cache=True))
        if self.capture:self.blocks[(start,end)]=data
        return data


def jsut_source():
    zip_source.ARCHIVE=JSUT_URL;zip_source.MAX_BYTES=128*1024*1024
    source=Source()
    with source.session.get(JSUT_URL,headers={'Range':'bytes=-65536'},stream=True,timeout=(10,30)) as response:
        response.raise_for_status();assert response.headers.get('Content-Encoding','identity')=='identity'
        if response.status_code==206:
            match=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',response.headers.get('Content-Range',''));assert match
            start,end,size=map(int,match.groups());assert start==size-65536 and end==size-1 and 65536<size<=128*1024*1024
            data=source._read(response,65536);assert len(data)==65536
            source.requests.append(dict(start=start,end=end,bytes=len(data),status=206,etag=response.headers.get('ETag')))
        elif response.status_code==200:
            length=response.headers.get('Content-Length');assert length and 0<int(length)<=128*1024*1024
            size=int(length);parts=[];received=0
            for chunk in response.iter_content(1024*1024):
                received+=len(chunk);assert received<=size and received<=128*1024*1024;parts.append(chunk)
                if received//(8*1024*1024)!=(received-len(chunk))//(8*1024*1024):print('JSUT_PRIVATE_ZIP_BYTES',received,flush=True)
            assert received==size;source.blob=b''.join(parts);source.bytes=received
            (ROOT/'raw_private').mkdir(exist_ok=True);archive=ROOT/'raw_private/jsut-song_ver1.zip'
            if archive.exists():assert archive.read_bytes()==source.blob
            else:archive.write_bytes(source.blob)
            source.requests.append(dict(status=200,bytes=received,bounded_full_archive_fallback=True,archive_sha256=sha(archive),etag=response.headers.get('ETag')))
        else:raise ValueError('Unexpected archive response')
    zip_source.EXPECTED_ARCHIVE_SIZE=size
    return source,size


def select_jsut(source,size):
    path=ROOT/'jsut_file_selection.json'
    rows,receipt=zip_source.entries(source)
    directory=source.blocks[(source.requests[-1]['start'],source.requests[-1]['end'])];attrs=[];at=0
    while at<len(directory):
        f=struct.unpack_from('<4s6H3L5H2L',directory,at);assert f[0]==b'PK\x01\x02';attrs.append(f[15]);at+=46+f[10]+f[11]+f[12]
    assert len(attrs)==len(rows)
    wavs=[]
    for row,attr in zip(rows,attrs,strict=True):
        parts=Path(row['name']).parts;mode=(attr>>16)&0xffff;kind=stat.S_IFMT(mode)
        if row['name'].lower().endswith('.wav') and not any(p.startswith('.') or p=='__MACOSX' for p in parts) and kind in (0,stat.S_IFREG):wavs.append(row)
    wavs=sorted(wavs,key=lambda r:r['name']);assert len(wavs)==27,('Expected27 nonhidden regular WAV entries; do not substitute',len(wavs))
    selected=wavs[:3]
    for row in selected:assert 0<row['compressed_bytes']<=16*1024*1024 and 0<row['decoded_bytes']<=32*1024*1024
    result=dict(archive=JSUT_URL,archive_size=size,directory_receipt=receipt,selected=selected,selection_before_any_waveform_decode=True,cohort_reservation_sha256=sha(ROOT/'cohort_reservation.json'))
    if path.exists():assert read(path)==result
    else:write(path,result)
    source.blocks.clear();source.capture=False
    return selected


def acquire(dataset,cohort):
    records=[]
    if dataset=='jsut':
        source,size=jsut_source();selected=select_jsut(source,size);items=[('female_jsut_song_1',row) for row in selected]
    else:
        source=Source();zip_source.MAX_BYTES=80*1024*1024
        if dataset=='vocalset':
            zip_source.ARCHIVE='https://zenodo.org/records/1193957/files/VocalSet.zip?download=1';zip_source.EXPECTED_ARCHIVE_SIZE=2077087366
            items=[(s['singer_id'],row) for s in cohort['sources'] if s['dataset']=='VocalSet1.0' for row in s['entries']]
        else:
            meta=read(ROOT.parent/'confirmation_sources_2026-10-29/csd_official_record.json')
            zip_source.ARCHIVE='https://zenodo.org/records/4916302/files/CSD.zip?download=1';zip_source.EXPECTED_ARCHIVE_SIZE=meta['files'][0]['size']
            items=[('female_csd_1',row['source_entry']) for s in cohort['sources'] if s['dataset']=='CSD1.1' for row in s['entries']]
        source.capture=False
    for singer,row in items:
        assert 0<row['compressed_bytes']<=16*1024*1024 and 0<row['decoded_bytes']<=32*1024*1024
        out=ROOT/'raw_private'/singer;out.mkdir(parents=True,exist_ok=True);path=out/(Path(row['name']).name)
        if path.exists():
            payload=path.read_bytes();assert len(payload)==row['decoded_bytes'] and zlib.crc32(payload)&0xffffffff==row['crc32']
        else:
            payload=zip_source.member(source,row);assert payload[:4]==b'RIFF' and payload[8:12]==b'WAVE';path.write_bytes(payload)
        records.append(dict(singer_id=singer,dataset={'jsut':'JSUT-song','vocalset':'VocalSet1.0','csd':'CSD1.1'}[dataset],path=str(path.resolve()),sha256=sha(path),archive_member=row['name'],archive_entry=row,role='fresh_confirmation_only',training_permitted=False,audio_acoustically_analyzed=False,public_audio_export_allowed=False if dataset=='jsut' else None))
        print('FRESH_SOURCE_ACQUIRED',singer,row['name'],len(payload),flush=True)
    write(ROOT/(dataset+'_acquisition_receipt.json'),dict(records=records,network_bytes=source.bytes,requests=source.requests,cohort_sha256=sha(ROOT/'cohort_reservation.json'),fetch_source_sha256=sha(Path(__file__)),audio_acoustically_analyzed=False,training_permitted=False,research_validation_passed=False))
    return records


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('dataset',choices=['jsut','vocalset','csd','all']);args=parser.parse_args();cohort=check_lock()
    declaration=ROOT/'acquisition_declaration.json'
    if not declaration.exists():write(declaration,dict(created_at=datetime.now(timezone.utc).isoformat(),fetch_source_sha256=sha(Path(__file__)),pipeline_lock_sha256=sha(ROOT/'pipeline_lock.json'),cohort_sha256=sha(ROOT/'cohort_reservation.json'),source_closure_sha256=sha(ROOT/'component_source_lock.json'),protocol='Private inbound research data only. Prefer exact ZIP ranges. JSUT only may use a bounded <=128MiB full ZIP if publisher ignores ranges; no waveform decoded before first3 filename selection. VocalSet/CSD use exact bounded ranges only. Immutable CRC/SHA records, no extraction by archive paths, no replacements or acoustic/model/label analysis.',research_validation_passed=False))
    d=read(declaration);assert d['fetch_source_sha256']==sha(Path(__file__))
    for dataset in (['jsut','vocalset','csd'] if args.dataset=='all' else [args.dataset]):acquire(dataset,cohort)
    receipts=[ROOT/(name+'_acquisition_receipt.json') for name in ('vocalset','csd','jsut')]
    if all(p.exists() for p in receipts):
        records=[row for p in receipts for row in read(p)['records']];assert len(records)==12 and len({r['singer_id'] for r in records})==4
        for row in records:assert sha(row['path'])==row['sha256']
        write(ROOT/'raw_source_manifest.json',dict(records=records,pipeline_lock_sha256=sha(ROOT/'pipeline_lock.json'),cohort_sha256=sha(ROOT/'cohort_reservation.json'),source_file_count=12,source_performer_count=4,acoustic_analysis_started=False,research_validation_passed=False))
        print('FRESH12_SOURCES_READY_NO_ACOUSTIC_ANALYSIS',flush=True)
    check_lock()

if __name__=='__main__':main()
