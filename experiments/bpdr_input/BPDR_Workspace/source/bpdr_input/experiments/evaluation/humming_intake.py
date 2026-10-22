"""Bounded research-only HumTrans intake; no full 14.7-GB archive download.

Uses standard HTTP byte ranges and ZIP CRC verification. Downloads data only,
never executes supplied code. License/source metadata and raw bytes are kept.
"""
from collections import Counter
import io
import json
from pathlib import Path
import re
import time
import zipfile
from bpdr_input.experiments.evaluation.transfer_study import COMPONENT, read, write, sha

ROOT=COMPONENT/'data/humtrans'
REVISION='c3c8504280cbdef0f00bb8119f16f034bea2bd9f'
BASE=f'https://huggingface.co/datasets/dadinghh2/HumTrans/resolve/{REVISION}/'


def session():
    import requests
    from urllib3.util.retry import Retry
    from requests.adapters import HTTPAdapter
    s=requests.Session();s.headers.update({'User-Agent':'BPDR-local-research-intake/1.0','Accept-Encoding':'identity'})
    s.mount('https://',HTTPAdapter(max_retries=Retry(total=3,backoff_factor=1,status_forcelist=[429,500,502,503,504])))
    return s


def download(s,url,path,limit):
    with s.get(url,stream=True,timeout=45) as r:
        r.raise_for_status();parts=[];n=0
        for block in r.iter_content(65536):
            n+=len(block)
            if n>limit:raise ValueError('Download exceeded the declared byte limit')
            parts.append(block)
    Path(path).write_bytes(b''.join(parts));return n


class RangeReader(io.RawIOBase):
    def __init__(self,s,url,max_bytes=128*1024*1024):
        self.session=s;self.url=url;self.position=0;self.transferred=0;self.max_bytes=max_bytes
        body,total=self.fetch(0,1);self.length=total

    def fetch(self,start,count):
        if count<0 or count>16*1024*1024 or self.transferred+count>self.max_bytes:
            raise ValueError('Range exceeds research download budget')
        with self.session.get(self.url,headers={'Range':f'bytes={start}-{start+count-1}'},stream=True,timeout=60) as r:
            if r.status_code!=206:raise ValueError(f'Byte-range server required (received HTTP {r.status_code}); full download refused')
            m=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',r.headers.get('Content-Range',''))
            if not m or int(m[1])!=start or int(m[2])!=start+count-1:raise ValueError('Mismatched HTTP Content-Range')
            body=r.raw.read(count+1)
            if len(body)!=count:raise ValueError('Truncated or oversized byte range')
            self.transferred+=count
            return body,int(m[3])

    def seekable(self):return True
    def readable(self):return True
    def tell(self):return self.position
    def seek(self,offset,whence=0):
        pos=offset if whence==0 else self.position+offset if whence==1 else self.length+offset
        if pos<0:raise ValueError('Negative seek')
        self.position=pos;return pos
    def read(self,size=-1):
        n=self.length-self.position if size is None or size<0 else min(size,self.length-self.position)
        if n<=0:return b''
        body,total=self.fetch(self.position,n)
        if total!=self.length:raise ValueError('Remote archive length changed')
        self.position+=n;return body


def identity(key):
    parts=Path(key).stem.split('_')
    if len(parts)<4 or not re.fullmatch('[FM][0-9]+',parts[0]):raise ValueError('Unexpected HumTrans key')
    return parts[0],parts[1]


def select(keys):
    # Official train recordings only. All official valid/test audio stays unseen.
    people=sorted({identity(k)[0] for k in keys})
    if len(people)<10:raise ValueError('Expected at least ten participant IDs')
    roles={p:'train' if i<6 else 'development' if i<8 else 'reserved_unseen' for i,p in enumerate(people)}
    chosen=[];used=set()
    for person in people[:8]:
        for key in sorted(keys):
            p,song=identity(key)
            if p!=person or song in used:continue
            chosen.append(dict(key=Path(key).stem,person_id=p,composition_id=song,split=roles[p],official_split='train'))
            used.add(song)
            if sum(r['person_id']==person for r in chosen)==2:break
    if len(chosen)!=16:raise ValueError('Could not select 16 distinct compositions')
    return chosen,roles


def main():
    ROOT.mkdir(parents=True,exist_ok=True);s=session()
    for name,limit in [('README.md',20000),('train_valid_test_keys.json',1000000)]:
        p=ROOT/name
        if not p.exists():download(s,BASE+name,p,limit)
    splits=read(ROOT/'train_valid_test_keys.json')
    # Dataset split formats are checked before downloading any audio.
    print('Official split sizes',{k:len(v) for k,v in splits.items()},flush=True)
    train_key=next(k for k in splits if k.lower() in ['train','training'])
    selected,roles=select(splits[train_key])
    overlap={}
    for k,values in splits.items():overlap[k]=sorted({identity(v)[0] for v in values})
    declaration=dict(version='humtrans-small-intake-1',revision=REVISION,license='CC-BY-NC-4.0 per retained dataset card',
        source='https://huggingface.co/datasets/dadinghh2/HumTrans',selection=selected,person_roles=roles,
        source_official_singers_by_split=overlap,selection_rule='First two lexicographic official-train keys per first eight persons, distinct compositions globally; first six train, next two development; last two unseen',
        max_transferred_audio_archive_bytes=128*1024*1024,source_sha256=sha(__file__),
        labels='Score-derived MIDI, not performed F0 or approved musical repair reference',
        status='exploratory_noncommercial_research',training_eligible=False,final_test_audio_accessed=False)
    write(ROOT/'intake_declaration.json',declaration)
    midizip=ROOT/'all_midi.zip'
    if not midizip.exists():download(s,BASE+'all_midi.zip',midizip,8*1024*1024)
    with zipfile.ZipFile(midizip) as z:
        names={Path(n).stem:n for n in z.namelist() if n.lower().endswith(('.mid','.midi'))}
        for r in selected:
            key=r['key'];name=names.get(key)
            if name is None:raise ValueError('No corresponding MIDI for selected audio')
            if z.getinfo(name).file_size>100000:raise ValueError('Unexpected MIDI size')
            folder=ROOT/'subset'/r['split']/key;folder.mkdir(parents=True,exist_ok=True)
            (folder/'reference.mid').write_bytes(z.read(name))
    reader=RangeReader(s,BASE+'all_wav.zip')
    with zipfile.ZipFile(reader) as z:
        names={Path(n).stem:n for n in z.namelist() if n.lower().endswith('.wav')}
        for r in selected:
            key=r['key'];info=z.getinfo(names[key])
            if info.file_size>16*1024*1024 or info.compress_size>16*1024*1024:raise ValueError('Selected WAV exceeds limit')
            p=ROOT/'subset'/r['split']/key/'original.wav'
            if not p.exists():p.write_bytes(z.read(info))
            r.update(path=str(p),sha256=sha(p),midi_path=str(p.with_name('reference.mid')),midi_sha256=sha(p.with_name('reference.mid')))
            print(r['split'],key,'bytes',p.stat().st_size,flush=True)
    write(ROOT/'intake_receipt.json',dict(declaration_sha256=sha(ROOT/'intake_declaration.json'),records=selected,
        archive_total_bytes=reader.length,archive_bytes_transferred=reader.transferred,
        official_split_singer_counts={k:len(v) for k,v in overlap.items()},source_splits_overlap=bool(set(overlap[train_key]) & set().union(*(set(v) for k,v in overlap.items() if k!=train_key))),
        component_singer_overlap=False,component_composition_overlap=False,final_test_audio_accessed=False,
        license_card_sha256=sha(ROOT/'README.md'),midi_zip_sha256=sha(midizip)))
    print('INTAKE_COMPLETE',len(selected),'range bytes',reader.transferred,flush=True)


if __name__=='__main__':main()
