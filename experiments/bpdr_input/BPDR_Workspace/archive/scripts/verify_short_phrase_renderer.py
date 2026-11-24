"""Independent decoded-file and frozen-estimator replay of the short-support probe."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import soundfile as sf
import torch

WS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WS/'source'))
sys.path.insert(0,str(WS/'tools'))
from bpdr_input.repair.centered_precision_audit import coarse_features
from bpdr_input.repair.rule_correction import fine_hz
from bpdr_input.repair.framewise_audit import audit_curve
from acquire_fresh_confirmation import check_lock

ROOT=WS/'results/transformer_usability/2026-11-02_v38_short_phrase_probe'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    torch.set_num_threads(2)
    check_lock()  # The prior confirmation's complete runtime closure must still match.
    result=read(ROOT/'results.json');declaration=read(ROOT/'declaration.json')
    assert result['declaration_sha256']==sha(ROOT/'declaration.json')
    amendment=read(ROOT/'resume_amendment.json')
    assert amendment['original_source_sha256']==sha(ROOT/'initial_probe_source.py')==declaration['source_sha256']
    assert amendment['resumed_source_sha256']==sha(WS/'tools/probe_short_phrase_renderer.py')
    expected={(f,w,c) for f in [120,240] for w in [.24,.36,.50,.70] for c in [-30.,30.]}
    assert {(r['base_hz'],r['width_seconds'],r['cents']) for r in result['cases']}==expected
    assert len(result['cases'])==16
    cache={};summary=[]
    for row in result['cases']:
        base,width,cents=row['base_hz'],row['width_seconds'],row['cents']
        folder=ROOT/f'{base}hz_{width:.2f}s'
        original=folder/'original.wav';output=folder/f'shift_{cents:+g}.wav'
        assert sha(original)==row['source_sha256'] and sha(output)==row['output_sha256']
        audio,sr=sf.read(original,dtype='float32');decoded,rate=sf.read(output,dtype='float32')
        assert sr==rate==22050 and len(audio)==len(decoded)==132300
        assert np.isfinite(decoded).all() and np.max(abs(decoded))<1
        support=(2.46,2.46+width);t=np.arange(len(audio))/sr
        outside=(t<=support[0])|(t>=support[1])
        assert np.array_equal(audio[outside],decoded[outside])
        assert np.any(audio[~outside]!=decoded[~outside]), 'Nonzero fixture must actually alter samples'
        times=np.arange(len(audio)//441+1)*.02
        if original not in cache:
            cache[original]=(coarse_features(audio,sr,WS/'checkpoints'),fine_hz(audio,sr,times))
        before,yb=cache[original]
        after=coarse_features(decoded,sr,WS/'checkpoints');ya=fine_hz(decoded,sr,times)
        reliable=before['voiced_mask']&np.isfinite(yb)&(yb>0)
        curve=np.where((times>=support[0])&(times<support[1]),cents,0.)
        audit=audit_curve(before,after,yb,ya,reliable,[support],curve)
        assert json.loads(json.dumps(audit,allow_nan=False))==row['audit'],(base,width,cents)
        summary.append(dict(base_hz=base,width_seconds=width,cents=cents,
                            passed=audit['passed'],common_frames=audit['common_frames']))
    assert sum(r['passed'] for r in summary)==4==result['audit_passed']
    receipt=dict(decoded_files_checked=16,source_files_checked=8,decoded_audits_replayed=16,
        outside_sample_preservation_passed=True,unchanged_v37_runtime_closure=True,
        results_sha256=sha(ROOT/'results.json'),verifier_source_sha256=sha(__file__),
        synthetic_only=True,model_accuracy_assessed=False,real_voice_usability_assessed=False,
        activation=False,summary=summary)
    (ROOT/'integrity_receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('SHORT_PHRASE_INDEPENDENT_REPLAY_PASS',16,flush=True)


if __name__=='__main__':main()
