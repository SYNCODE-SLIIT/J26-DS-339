"""Measure V64 supervision geometry without creating labels, fits or interventions.

Reliability here is A/C runtime availability, not new performed-pitch/intention gold.
Use this diagnostic together with verified delivered failure traces, not alone.
"""
from collections import defaultdict
import numpy as np
import prepare_reliable_training_regions as common

DATA=common.WS/'results/transformer_usability/2026-11-04_v64_reliable_melodic_pairs'


def aggregate(rows):
    keys=['registered_positive_frames','reliable_injected_frames','unlabelled_reliable_injected_frames','unlabelled_reliable_near_outside_frames','registered_known_zero_frames']
    counts={key:sum(r[key] for r in rows) for key in keys}
    return dict(pairs=len(rows),**counts,positive_label_fraction_of_reliable_injection=counts['registered_positive_frames']/counts['reliable_injected_frames'] if counts['reliable_injected_frames'] else None,
        pairs_with_unlabelled_near_outside_frames=sum(r['unlabelled_reliable_near_outside_frames']>0 for r in rows))


def main():
    common.check_lock();proof=common.read(DATA/'integrity_receipt.json')
    assert proof['train_pairs_sha256']==common.sha(DATA/'train_pairs.json')
    assert proof['train_originals_sha256']==common.sha(DATA/'train_originals.json')
    pairs=common.read(DATA/'train_pairs.json');originals={r['source_id']:r for r in common.read(DATA/'train_originals.json')};source_masks={};rows=[]
    for index,pair in enumerate(pairs):
        if 'excluded' in pair:continue
        assert pair['split']=='train';source=originals[pair['source_id']]
        if pair['source_id'] not in source_masks:
            with np.load(source['analysis_A']) as z:source_masks[pair['source_id']]=z['reliable'].copy()
        with np.load(pair['analysis_C']) as z:times=z['times_seconds'].copy();reliable=source_masks[pair['source_id']]&z['reliable']
        with np.load(pair['target_path']) as z:positive=z['plateau'].copy();zero=z['known_zero'].copy();mask=z['mask'].copy();target=z['target'].copy()
        np.testing.assert_array_equal(mask,positive|zero);assert not np.any(positive&zero)
        np.testing.assert_array_equal(target,np.where(positive,-pair['cents'],0.).astype(np.float32))
        start,end=pair['support_seconds'];inside=(times>=start)&(times<end)
        near_outside=((times<start)&(times>=start-.07-1e-12))|((times>=end)&(times<=end+.07+1e-12))
        assert not np.any(positive&~inside) and not np.any(zero&inside)
        assert np.all(~positive|reliable)
        rows.append(dict(id='C_'+str(index),source_id=pair['source_id'],group=pair['group'],width_seconds=pair['width_seconds'],
            registered_positive_frames=int(positive.sum()),reliable_injected_frames=int(np.sum(inside&reliable)),
            unlabelled_reliable_injected_frames=int(np.sum(inside&reliable&~mask)),
            unlabelled_reliable_near_outside_frames=int(np.sum(near_outside&reliable&~mask)),registered_known_zero_frames=int(zero.sum())))
    assert len(rows)==proof['staged_pairs_verified']==445
    summary=dict(all=aggregate(rows),by_width={str(w):aggregate([r for r in rows if r['width_seconds']==w]) for w in [.24,.28,.36,.50]},
        by_group={g:aggregate([r for r in rows if r['group']==g]) for g in sorted({r['group'] for r in rows})})
    common.write(DATA/'supervision_geometry_report.json',dict(summary=summary,rows=rows,source_sha256=common.sha(__file__),
        data_integrity_sha256=common.sha(DATA/'integrity_receipt.json'),
        scope='Count existing labels and reliable frame centers only. Unlabelled frames are not automatically valid new observed-F0 targets; future waveform-command anchors require explicit semantics and proof. No interpolation, new target, waveform, fit or runtime change.',
        new_model_fits=0,nontraining_audio_accessed=False,production_eligible=False,research_validation_passed=False))
    print('SHORT_SUPERVISION_GEOMETRY',summary,flush=True)


if __name__=='__main__':main()
