"""Verify fresh source preprocessing and diagnose reference-region geometry before inference."""
import io
from math import gcd
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from scipy.ndimage import median_filter
from scipy.signal import resample_poly
import prepare_fresh_confirmation as prep
from bpdr_input.repair.held_voice_policy import eligibility,spans
from bpdr_input.repair.rule_correction import segment_notes,RulePolicy


def tested_region(raw,a,b,spec):
    a+=int(spec['trim_frames']);b-=int(spec['trim_frames']);seconds=(b-a)*.02
    required=max(spec['minimum_width']+2*spec['context_seconds'],spec['minimum_width']/spec['region_fraction'])
    if seconds<required:return dict(start_frame=int(a),stop_frame=int(b),duration_seconds=seconds,required_seconds=required,eligible=False,reason='DURATION')
    y=median_filter(raw[a:b,64],size=spec['quality_median_frames'],mode='nearest');x=np.arange(len(y))
    spread=float(np.std(y-np.polyval(np.polyfit(x,y,1),x)))
    return dict(start_frame=int(a),stop_frame=int(b),duration_seconds=seconds,detrended_spread_cents=spread,eligible=spread<=10.,reason='ELIGIBLE' if spread<=10. else 'SPREAD')


def main():
    torch.set_num_threads(2);d,rows=prep.declare();ex=prep.ex;root=prep.PREP;manifest=ex.read(root/'prepared_manifest.json')
    assert manifest['attempted']==48 and manifest['scoreable']==0 and manifest['excluded']==48
    assert manifest['pairs_sha256']==ex.sha(root/'pairs.json') and manifest['sources_sha256']==ex.sha(root/'original_sources.json')
    sources=ex.read(root/'original_sources.json');pairs=ex.read(root/'pairs.json');assert len(sources)==12 and len(pairs)==48
    for p,h in ex.read(root/'input_hashes.json').items():assert ex.sha(p)==h,p
    reports=[];feature_arrays=0
    for index,(row,source) in enumerate(zip(rows,sources,strict=True),1):
        assert ex.sha(row['path'])==row['sha256']==source['source']['sha256']
        audio,sr=sf.read(row['path'],dtype='float32',always_2d=True);channel=int(np.argmax(np.mean(audio.astype(np.float64)**2,axis=0)))
        wave=audio[:min(len(audio),round(20*sr)),channel];divisor=gcd(sr,22050)
        if sr!=22050:wave=resample_poly(wave,22050//divisor,sr//divisor).astype(np.float32)
        stream=io.BytesIO();sf.write(stream,np.asarray(wave,np.float32),22050,format='WAV',subtype='PCM_24');recreated,_=sf.read(io.BytesIO(stream.getvalue()),dtype='float32')
        working,rate=sf.read(source['working_audio'],dtype='float32');assert rate==22050 and channel==source['source_channel'];np.testing.assert_array_equal(recreated,working)
        assert ex.sha(source['working_audio'])==source['working_sha256']
        measured=prep.recovery.measured(working)
        with np.load(Path(source['working_audio']).parent/'original_features.npz') as z:
            assert set(z.files)==set(measured['features'])
            for key in z.files:np.testing.assert_array_equal(z[key],measured['features'][key]);feature_arrays+=1
        raw=measured['views'][0];allowed,reliable,_=eligibility(raw,region_split_cents=70.)
        runs=spans(allowed&reliable);regions=segment_notes(raw[:,64],reliable,RulePolicy(note_split_cents=70.))
        original=[tested_region(raw,a,b,d['spec']) for a,b in runs]
        separate=[tested_region(raw,a,b,d['spec']) for a,b in regions if np.all(allowed[a:b]&reliable[a:b])]
        assert not any(r['eligible'] for r in original)
        subset=pairs[(index-1)*4:index*4];assert all(p['excluded']=='NO_STABLE_HELD_REGION_WITH_TWO_SIDED_CONTEXT' for p in subset)
        overlaps=[int(sum(a<hi and b>lo for a,b in regions)) for lo,hi in runs]
        report=dict(index=index,singer_id=row['singer_id'],archive_member=row['archive_member'],working_frames=len(raw),reliable_frames=int(reliable.sum()),allowed_frames=int(allowed.sum()),acoustic_region_count=len(regions),allowed_boolean_run_count=len(runs),acoustic_regions_per_boolean_run=overlaps,original_span_candidates=original,separate_region_candidates=separate,separate_eligible_regions=sum(r['eligible'] for r in separate),correction_inference_performed=False,diagnostic_only=True)
        reports.append(report);print('FRESH_PREP_VERIFIED',index,row['singer_id'],'runs',len(runs),'separate_eligible',report['separate_eligible_regions'],flush=True)
    prep.declare();receipt=dict(source_preprocessings_recreated=12,original_feature_arrays_reinferred=feature_arrays,attempted=48,scoreable=0,excluded=48,original_zero_reference_result_reproduced=True,geometry_reports=reports,originals_with_joined_region_runs=sum(any(n>1 for n in r['acoustic_regions_per_boolean_run']) for r in reports),originals_with_separate_eligible_regions=sum(r['separate_eligible_regions']>0 for r in reports),prepared_manifest_sha256=ex.sha(root/'prepared_manifest.json'),verifier_sha256=ex.sha(Path(__file__)),correction_inference_started=False,diagnostic_only=True,research_validation_passed=False)
    ex.write(root/'integrity_receipt.json',receipt);print('FRESH_PREP_ZERO_REFERENCES_VERIFIED',receipt['originals_with_joined_region_runs'],receipt['originals_with_separate_eligible_regions'],flush=True)

if __name__=='__main__':main()
