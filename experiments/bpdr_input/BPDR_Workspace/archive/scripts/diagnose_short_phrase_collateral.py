"""Post-hoc preservation check on known-zero reference frames, separate from gain scoring."""
from pathlib import Path
import sys,json
import numpy as np
import soundfile as sf

WS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WS/'source'));sys.path.insert(0,str(WS/'tools'))
import compare_short_phrase_models as study
from probe_short_phrase_renderer import sha
from bpdr_input.repair.rule_correction import fine_pitch

ROOT=WS/'results/transformer_usability/2026-11-02_v42_short_delivery'


def main():
    result=study.read(ROOT/'results.json');rows=study.read(study.DATA/'train_pairs.json')
    originals={s['source_id']:s for s in study.read(study.DATA/'train_originals.json')};arms={}
    for arm,content in result['arms'].items():
        cases=[]
        for r in content['records']:
            if r['kind']!='corrupted' or not r['verified_output']:continue
            source=rows[r['pair_index']];original=originals[source['source_id']]
            with np.load(source['target_path']) as z:zero=z['mask']&(z['target']==0);times=z['times'].copy()
            with np.load(original['analysis_A']) as z:reference=z['fine'].copy()
            with np.load(source['analysis_C']) as z:before=z['fine'].copy()
            output,sr=sf.read(r['verified_output'],dtype='float32');assert sr==22050
            after=fine_pitch(output,sr,times)
            b=float(np.mean(abs(before[zero]-reference[zero])))
            a=float(np.mean(abs(after[zero]-reference[zero])))
            cases.append(dict(id=r['id'],group=r['group'],zero_frames=int(zero.sum()),
                before_zero_mae_cents=b,after_zero_mae_cents=a,
                known_zero_error_increased=bool(a>b+1e-6),positive_plateau_improved=r['verified_improvement'],
                output_sha256=sha(r['verified_output'])))
        arms[arm]=dict(cases=cases,verified_corrupted_outputs=len(cases),
            outputs_increasing_known_zero_error=sum(c['known_zero_error_increased'] for c in cases))
        print('KNOWN_ZERO_REFERENCE_DRIFT',arm,arms[arm]['outputs_increasing_known_zero_error'],'/',len(cases),flush=True)
    study.write(ROOT/'reference_preservation_diagnosis.json',dict(script_sha256=sha(__file__),
        locked_results_sha256=sha(ROOT/'results.json'),arms=arms,
        interpretation='Post-hoc measured collateral drift on predeclared known-zero frames. Does not alter the locked positive-plateau gain metric or classify unknown human intention. Technical adherence to a predicted correction is not proof that the prediction should have been applied.',
        new_fits=0,production_eligible=False))


if __name__=='__main__':main()
