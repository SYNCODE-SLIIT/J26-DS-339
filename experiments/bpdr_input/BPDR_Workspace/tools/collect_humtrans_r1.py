"""Collect HumTrans humming audio for the R1 note-centre estimator (declared 2026-11-13, before any download).
Source: https://huggingface.co/datasets/dadinghh2/HumTrans (CC-BY-NC-4.0, research use). Official TRAIN keys only, plain
recordings only (no _U/_D/_DD/_DDD pitch-shifted copies). Roles follow the existing intake: F01-F05, M01 = training (120 each,
evenly spaced over the sorted keys, or all if fewer); M02 = development (100, evaluation only). M03, M04, M05, VALID, TEST are not touched.
Only the selected zip members are fetched with HTTP range requests; the 14.7 GB archive is never downloaded in full."""
import io, json, re, hashlib, sys, zipfile
from pathlib import Path
import numpy as np, requests
WS = Path(__file__).resolve().parents[1]; OUT = WS / 'datasets/humtrans_r1'
URL = 'https://huggingface.co/datasets/dadinghh2/HumTrans/resolve/main/all_wav.zip'
KEYS = WS / 'datasets/humtrans/train_valid_test_keys.json'
ROLES = {**{p: ('train', 120) for p in ('F01', 'F02', 'F03', 'F04', 'F05', 'M01')}, 'M02': ('development', 100)}

class RangeFile(io.RawIOBase):
    def __init__(self, url, block=2 << 20):
        self.s = requests.Session(); self.url = url; self.block = block; self.pos = 0; self.cache = (None, b'')
        r = self._retry(lambda: self.s.head(url, allow_redirects=True, timeout=60)); self.size = int(r.headers.get('X-Linked-Size') or r.headers['Content-Length']); self.final = r.url; self.bytes = 0
    @staticmethod
    def _retry(fn, tries=8):
        import time
        for k in range(tries):
            try: return fn()
            except requests.RequestException:
                if k == tries - 1: raise
                time.sleep(min(60, 2 ** k))
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off; return self.pos
    def _get(self, a, b):
        for attempt in range(8):
            try:
                r = self.s.get(self.final, headers={'Range': f'bytes={a}-{b}'}, timeout=120)
                if r.status_code == 206: self.bytes += len(r.content); return r.content
                if r.status_code in (403, 401): self.final = self._retry(lambda: self.s.head(self.url, allow_redirects=True, timeout=60)).url
            except requests.RequestException:
                import time; time.sleep(2 ** attempt)
        raise RuntimeError(f'range {a}-{b} failed')
    def read(self, n=-1):
        if n < 0: n = self.size - self.pos
        n = min(n, self.size - self.pos); out = b''
        while n > 0:
            start, buf = self.cache
            if start is None or not (start <= self.pos < start + len(buf)):
                a = self.pos; b = min(self.size - 1, a + max(self.block, n) - 1); self.cache = (a, self._get(a, b)); start, buf = self.cache
            take = buf[self.pos - start: self.pos - start + n]; out += take; self.pos += len(take); n -= len(take)
        return out
    def readinto(self, b): d = self.read(len(b)); b[:len(d)] = d; return len(d)

def select():
    keys = json.load(open(KEYS))['TRAIN']; plain = sorted(k for k in keys if not re.search(r'_(U|D|DD|DDD)$', k)); sel = []
    for p, (role, cap) in ROLES.items():
        mine = [k for k in plain if k.startswith(p + '_')]
        idx = np.unique(np.linspace(0, len(mine) - 1, min(cap, len(mine))).round().astype(int)) if mine else []
        sel += [dict(key=mine[i], person=p, role=role) for i in idx]
    return sel

def main():
    OUT.mkdir(exist_ok=True); sel = select(); decl = dict(declared='2026-11-13 before download', source=URL, rule=__doc__, selected=len(sel),
        per_person={p: sum(s['person'] == p for s in sel) for p in ROLES}, not_touched=['M03', 'M04', 'M05', 'VALID', 'TEST', 'variants _U/_D/_DD/_DDD'])
    (OUT / 'declaration.json').write_text(json.dumps(decl, indent=1) + '\n'); print(decl['per_person'], flush=True)
    rf = RangeFile(URL); zf = zipfile.ZipFile(rf); names = {Path(n).stem: n for n in zf.namelist() if n.lower().endswith('.wav')}
    print('archive entries', len(names), 'size', rf.size, flush=True); man = []
    for i, s in enumerate(sel):
        n = names.get(s['key'])
        if n is None: s['missing'] = True; man.append(s); continue
        dest = OUT / 'audio' / f"{s['key']}.wav"; dest.parent.mkdir(exist_ok=True)
        if not dest.exists(): dest.write_bytes(zf.read(n))
        s['sha256'] = hashlib.sha256(dest.read_bytes()).hexdigest(); s['bytes'] = dest.stat().st_size; man.append(s)
        if i % 25 == 0: print(i, len(sel), round(rf.bytes / 1e6), 'MB fetched', flush=True); (OUT / 'manifest.json').write_text(json.dumps(man, indent=1))
    (OUT / 'manifest.json').write_text(json.dumps(man, indent=1)); print('done', sum(1 for m in man if 'sha256' in m), 'files', round(rf.bytes / 1e6), 'MB', flush=True)
if __name__ == '__main__': main()
