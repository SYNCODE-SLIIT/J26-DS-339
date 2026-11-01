"""Prospective feasible-context data; all original source attempts retained."""
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
import sys
WORKSPACE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WORKSPACE/'source'));sys.path.insert(0,str(WORKSPACE/'tools'))
from bpdr_input.experiments.learning import context_improvement as ex
from prepare_recoverable_step_pairs import worker
ROOT=WORKSPACE/'results/transformer_usability/2026-10-28_v6'

if __name__=='__main__':
    ROOT.mkdir(exist_ok=True)
    sources=ex.read(ROOT.parent/'2026-10-28_long_v2/data_declaration.json')['source_records']
    spec=dict(root=str(ROOT),minimum_width=1.16,maximum_width=1.40,context_seconds=.26,region_fraction=.70,trim_frames=3,quality_median_frames=9,version='recoverable-v6')
    if not (ROOT/'data_declaration.json').exists():ex.write(ROOT/'data_declaration.json',dict(created_at=datetime.now(timezone.utc).isoformat(),
        previous_condition='V5 stricter .52s flanks produced only44 scoreable/200 attempts; retained, no model fitted',
        reason='Declare a feasible local deviation with >=.26s unshifted context on each side, exceeding the independent pitch-analysis window; retain all sources, no outcome-based replacements',
        source_records=sources,spec=spec,scope='Stable held acoustic region, actual deviation1.16-1.40s, <=70% region, <=24% recording; two-sided measured context',
        quality='180ms median pitch used only for source-context stability; no waveform smoothing. Detrended spread<=10c; source reliability and physical intervention audits unchanged',
        permission_limits_changed=False,final_singers_accessed=False,source_hashes={str(p):ex.sha(p) for p in [Path(__file__),WORKSPACE/'tools/prepare_recoverable_step_pairs.py',WORKSPACE/'source/bpdr_input/modeling/step_context.py',WORKSPACE/'source/bpdr_input/repair/held_voice_policy.py']}))
    rows=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        for i,result in enumerate(pool.map(worker,[(i,s,spec) for i,s in enumerate(sources,1)]),1):
            rows.extend(result['pairs']);ex.write(ROOT/'preparation_progress.json',dict(completed_sources=i,planned_sources=len(sources),attempted=len(rows),scoreable=sum('excluded' not in r for r in rows)))
            print('STEP_CONTEXT_SOURCE',i,'/',len(sources),'valid',sum('excluded' not in r for r in rows),'/',len(rows),flush=True)
    ex.write(ROOT/'train_pairs.json',rows);print('STEP_CONTEXT_DATA_READY',flush=True)
