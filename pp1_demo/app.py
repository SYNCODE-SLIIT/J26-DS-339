"""Local research website. Start: .venv/bin/python -B -m pp1_demo.app"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pp1_demo.config import APP,RUNTIME,VOCAB,PASAN,RAHUL,BPDR_EXPORT,configure_runtime
configure_runtime()
import argparse
from functools import partial
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
import mimetypes
import re
import threading
from urllib.parse import unquote,urlparse

from pp1_demo.dataset import catalogue,context,read,track_preview
from pp1_demo.orchestrator import new_run,execute

LOCK=threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def json(self,value,status=200):
        raw=json.dumps(value,allow_nan=False).encode()
        self.send_response(status);self.send_header('Content-Type','application/json')
        self.send_header('Content-Length',str(len(raw)));self.send_header('Cache-Control','no-store')
        self.end_headers();self.wfile.write(raw)

    def file(self,path,download=False):
        path=Path(path)
        if not path.is_file():return self.json({'error':'File not found'},404)
        total=path.stat().st_size;start,end=0,total-1;status=200
        ranged=self.headers.get('Range')
        if ranged:
            m=re.fullmatch(r'bytes=(\d+)-(\d*)',ranged)
            if not m:return self.json({'error':'Unsupported byte range'},416)
            start=int(m[1]);end=min(int(m[2]) if m[2] else end,end)
            if start>end or start>=total:return self.json({'error':'Range outside file'},416)
            status=206
        self.send_response(status)
        self.send_header('Content-Type',mimetypes.guess_type(path.name)[0] or 'application/octet-stream')
        self.send_header('Content-Length',str(end-start+1));self.send_header('Accept-Ranges','bytes')
        self.send_header('X-Content-Type-Options','nosniff')
        if status==206:self.send_header('Content-Range',f'bytes {start}-{end}/{total}')
        if download:self.send_header('Content-Disposition',f'attachment; filename="{path.name}"')
        self.end_headers()
        try:
            with path.open('rb') as stream:
                stream.seek(start);remaining=end-start+1
                while remaining:
                    block=stream.read(min(65536,remaining))
                    if not block:break
                    self.wfile.write(block);remaining-=len(block)
        except (BrokenPipeError,ConnectionResetError):pass

    def do_GET(self):
        try:
            route=unquote(urlparse(self.path).path)
            if route=='/api/catalog':
                return self.json({'tracks':catalogue(),'genres':read(VOCAB/'feature_vocabularies.json')['genres'],
                                  'models':{'bpdr':(BPDR_EXPORT/'selected_model.joblib').exists(),
                                            'vishwa':all((ROOT/'experiments/vishwa/checkpoints/weak_score_pilot'/name).is_file() for name in ['cnn.pt','cnn_transformer.pt','train_normalization.npz','manifest.json']),'pasan':PASAN.exists(),'rahul':RAHUL.exists()}})
            if route=='/api/panels/vishwa':
                return self.json({name:(APP/'panels/vishwa'/(name+'.md')).read_text(encoding='utf-8') for name in ['steps','evidence','decisions','roadmap']})
            if route=='/api/health':return self.json({'status':'ok','application':'pp1_demo'})
            if route.startswith('/api/tracks/'):
                tid=route.removeprefix('/api/tracks/')
                if tid not in {c['id'] for c in catalogue()}:return self.json({'error':'Unknown recording'},404)
                return self.json(track_preview(tid))
            if route.startswith('/api/audio/'):
                tid=route.removeprefix('/api/audio/')
                if tid not in {c['id'] for c in catalogue()}:return self.json({'error':'Unknown recording'},404)
                return self.file(context(tid)['audio_path'])
            m=re.fullmatch(r'/api/runs/(r[0-9a-f]{12})',route)
            if m:
                p=RUNTIME/'runs'/m[1]/'manifest.json'
                return self.json(read(p)) if p.exists() else self.json({'error':'Run not found'},404)
            m=re.fullmatch(r'/api/artifacts/(r[0-9a-f]{12})/(.+)',route)
            if m:
                folder=(RUNTIME/'runs'/m[1]).resolve();p=(folder/m[2]).resolve()
                if not p.is_relative_to(folder):return self.json({'error':'Invalid artifact path'},403)
                manifest=read(folder/'manifest.json')
                allowed=set(manifest['artifacts'].values())|{'manifest.json'}
                if m[2] not in allowed:return self.json({'error':'Artifact is not public'},403)
                return self.file(p,download=m[2].endswith(('.json','.mid')))
            assets={'/':'index.html','/style.css':'style.css','/app.js':'app.js'}
            if route in assets:return self.file(APP/'web'/assets[route])
            return self.json({'error':'Not found'},404)
        except (OSError,ValueError,KeyError) as exc:self.json({'error':str(exc)},400)

    def do_POST(self):
        if self.path!='/api/runs':return self.json({'error':'Not found'},404)
        origin=self.headers.get('Origin')
        if origin and urlparse(origin).netloc!=self.headers.get('Host'):
            return self.json({'error':'Cross-origin requests are not allowed'},403)
        if not LOCK.acquire(blocking=False):return self.json({'error':'A pipeline is already running. Please wait.'},409)
        try:
            length=int(self.headers.get('Content-Length','0'))
            if not 0<length<=4096:raise ValueError('Invalid request length')
            data=json.loads(self.rfile.read(length))
            if not isinstance(data,dict):raise ValueError('Expected a JSON object')
            tid=data.get('track_id')
            if tid not in {c['id'] for c in catalogue()}:raise ValueError('Select a recording from the library')
            genre=data.get('genre')
            if genre is not None and genre not in read(VOCAB/'feature_vocabularies.json')['genres']:raise ValueError('Unknown genre')
            folder,run=new_run(tid,genre)
            def worker():
                try:execute(folder,run,handoff_pause_s=1.35)
                finally:LOCK.release()
            threading.Thread(target=worker,daemon=True).start()
            self.json({'id':run['id']},202)
        except (ValueError,KeyError,OSError,TypeError) as exc:
            LOCK.release();self.json({'error':str(exc)},400)

    def log_message(self,format,*args):
        if not self.path.startswith('/api/runs/'):
            super().log_message(format,*args)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--port',type=int,default=8765)
    args=parser.parse_args()
    catalogue()  # Fail early with an actionable missing-dataset message.
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    print(f'Harmony Lab is ready at http://127.0.0.1:{args.port}',flush=True)
    print('Dataset/model files are read-only. All results are stored in pp1_demo/runtime/.',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()


if __name__=='__main__':main()
