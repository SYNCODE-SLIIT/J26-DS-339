"""Independent staged-array, target and source-group checks, no audio/model mutation."""
from pathlib import Path
import sys,json
import numpy as np

WS=Path(__file__).resolve().parents[1];sys.path.insert(0,str(WS/'tools'))
import build_native_short_pairs as run


def main():
    root=run.ROOT;d=run.read(root/'declaration.json');rows=run.read(root/'train_pairs.json')
    originals=run.read(root/'train_originals.json');cal=run.read(run.CAL/'results.json')
    assert d['script_sha256']==run.sha(WS/'tools/build_native_short_pairs.py')
    assert d['calibration_receipt_sha256']==run.sha(run.CAL/'integrity_receipt.json')
    assert len(rows)==336 and len(originals)==21
    for p,h in run.read(root/'input_hashes.json').items():assert run.sha(p)==h,p
    by_id={s['source_id']:s for s in originals};valid=0;arrays=0
    for index,row in enumerate(rows):
        ref=cal['cases'][index];source=by_id[row['source_id']]
        assert row['group']==source['group'] and row['split']=='train' and source['source']['split']=='train'
        assert row['calibration_index']==index and row['source_id']==ref['source_id']
        if 'excluded' in row:
            assert 'excluded' in ref or not ref['three_tracker']['passed'] or not ref['observed_context']['both_sides_at_least_three']
            continue
        assert ref['three_tracker']['passed'] and row['path_C']==ref['output']
        context,labels=run.expected(ref,source)
        for path,expected in [(row['analysis_C'],context),(row['target_path'],labels)]:
            with np.load(path) as z:
                assert set(z.files)==set(expected)
                for key,value in expected.items():np.testing.assert_array_equal(z[key],value);arrays+=1
        assert labels['plateau'].sum()>=3 and not np.any(labels['plateau']&labels['known_zero'])
        valid+=1
    assert valid==115
    run.write(root/'integrity_receipt.json',dict(staged_pairs_verified=valid,staged_arrays_verified=arrays,
        all_attempts_retained=len(rows),training_groups_only=True,
        declaration_sha256=run.sha(root/'declaration.json'),train_pairs_sha256=run.sha(root/'train_pairs.json'),
        train_originals_sha256=run.sha(root/'train_originals.json'),input_hashes_sha256=run.sha(root/'input_hashes.json'),
        verifier_source_sha256=run.sha(__file__),research_training_eligible=False,production_eligible=False))
    print('NATIVE_SHORT_PAIR_REPLAY_PASS',valid,arrays,flush=True)


if __name__=='__main__':main()
