"""Validate human review records; never infer human decisions from algorithms."""
from pathlib import Path
import json
from datetime import datetime
ROOT=Path(__file__).resolve().parents[1]


def validate_reviews(document,expected):
    """Reject incomplete completed records and return honest review counts."""
    rows=document['cases'];keys=[r['key'] for r in rows]
    if len(keys)!=len(set(keys)) or set(keys)!=set(expected):raise ValueError('Review case membership mismatch')
    completed=0;unresolved=0
    for row in rows:
        if row['status']=='pending':continue
        if row['status']!='reviewed':raise ValueError('Unknown review status')
        for field in ['reviewer','reviewed_at','evidence']:
            if not isinstance(row.get(field),str) or not row[field].strip():raise ValueError(f'Missing {field}')
        datetime.fromisoformat(row['reviewed_at'])
        if row.get('listened_original') is not True or row.get('listened_references') is not True:raise ValueError('Listening confirmation missing')
        if row.get('confidence') not in ['low','medium','high']:raise ValueError('Invalid confidence')
        if row.get('outcome') not in ['consistent_with_original','possible_score_performance_octave_difference','possible_estimator_error','timing_or_mixed_issue','unresolved']:raise ValueError('Invalid outcome')
        if row.get('suggested_shift_semitones') not in [None,-24,-12,0,12]:raise ValueError('Unsupported hypothesis')
        completed+=1;unresolved+=row['outcome']=='unresolved'
    return {'total':len(rows),'reviewed':completed,'pending':len(rows)-completed,'reviewed_but_unresolved':unresolved,'annotation_changes_authorized':False}


def main():
    """Print review status; completion never automatically authorizes relabeling."""
    folder=ROOT/'experiments/vishwa/results'
    expected=[r['key'] for r in json.loads((folder/'listening_review_evidence.json').read_text())['cases']]
    print(json.dumps(validate_reviews(json.loads((folder/'listening_review_decisions.json').read_text()),expected),indent=2))
if __name__=='__main__':main()
