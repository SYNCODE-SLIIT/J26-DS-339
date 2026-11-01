"""Apply the plan's PER REPRESENTED SINGER criteria without hiding exclusions.

Construction-unscoreable singers are unassessed, never validated. Their sources
remain in cohort yield and all originals must be preserved. A scoped preview
requires >=2 scoreable independent singers and both dataset voice categories.
"""
import numpy as np


def assess(result,controls):
    rows=result['rows'];groups=sorted({r['group'] for r in rows});by_group={}
    represented=[];directions=set();improved_total=0
    for group in groups:
        attempted=[r for r in rows if r['group']==group];valid=[r for r in attempted if 'excluded' not in r]
        originals=[r for r in controls if r['group']==group]
        requests=[r['raw_original_abs_cents'] for r in originals if r.get('raw_original_abs_cents') is not None]
        identity=bool(originals) and all(r.get('delivered_identity',False) for r in originals)
        improved=sum(bool(r.get('delivered_verified_improvement')) for r in valid);improved_total+=improved
        before=float(np.mean([r['delivered_before_mae_cents'] for r in valid])) if valid else None
        after=float(np.mean([r['delivered_after_mae_cents'] for r in valid])) if valid else None
        change=1-after/before if before else None
        raw=float(np.mean(requests)) if requests else None
        passes=bool(identity and raw is not None and raw<=3. and improved/len(attempted)>=.25 and change is not None and change>=.25) if valid else None
        if valid:
            represented.append(group)
            directions.update(int(np.sign(r['cents'])) for r in valid if r.get('delivered_verified_improvement'))
        by_group[group]=dict(attempted=len(attempted),scoreable=len(valid),improved=improved,
            all_attempt_coverage=improved/len(attempted),delivered_error_reduction=change,original_controls=len(originals),
            originals_identity=identity,raw_original_mean_cents=raw,passed=passes,
            assessment='assessed' if valid else 'unassessed_no_valid_reference_cases')
    categories={'female' if g.startswith('female') else 'male' if g.startswith('male') else 'unknown' for g in represented}
    sufficient=len(represented)>=2 and categories>={'female','male'}
    original_identity=bool(controls) and all(c.get('delivered_identity',False) for c in controls)
    passed=bool(sufficient and original_identity and all(by_group[g]['passed'] for g in represented) and directions=={-1,1})
    return dict(version='represented-singer-criteria-1',passed=passed,by_singer=by_group,attempted_singers=len(groups),
        scoreable_singers=len(represented),unassessed_singers=[g for g in groups if g not in represented],
        original_controls_all_identity=original_identity,sufficient_scoreable_representation=sufficient,
        cohort_attempted=len(rows),cohort_scoreable=sum('excluded' not in r for r in rows),cohort_improved=improved_total,
        cohort_all_attempt_coverage=improved_total/len(rows) if rows else 0.,improved_directions=sorted(directions),
        independent_listening='not assessed',research_validation_passed=False,
        meaning='Usability criteria apply only to represented, reference-scoreable held-voice scope. No general accuracy claim for unassessed singers or unsupported recordings.')
