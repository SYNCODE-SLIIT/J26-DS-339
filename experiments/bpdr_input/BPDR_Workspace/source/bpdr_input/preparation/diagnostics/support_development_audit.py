"""Frozen inventory for completion of development support audits; never trains."""
from copy import deepcopy
from bpdr_input.preparation.diagnostics.support_acoustic_pilot import PHRASES, SHIFTS, pilot_sources


def completion_inventory(geometry, pilot):
    dev=[r for r in geometry['phrases'] if r['split']=='development']
    if len(dev)!=14 or len({r['phrase_id'] for r in dev})!=14:
        raise ValueError('Expected all fourteen unique frozen development phrases')
    if any(r['training_eligible'] or r['evidence_level']!='cached_final_reliable_mask_exact_replay' for r in dev):
        raise ValueError('Expected unchanged development final-mask evidence')
    pilot_sources(geometry)
    if pilot['status']!='complete' or pilot['new_pairs'] or pilot['new_optimizer_updates'] or pilot['research_ready_examples']:
        raise ValueError('Expected completed exploratory pilot without packaging or fitting')
    keys={(pid,c) for pid in PHRASES for c in SHIFTS}
    attempts=pilot['attempts']
    if len(attempts)!=4 or {(a['phrase_id'],a['cents']) for a in attempts}!=keys or any(a['status']!='measured' for a in attempts):
        raise ValueError('Reuse exactly the four measured pilot attempts, including failures')
    by_id={r['phrase_id']:r for r in dev}
    if len(pilot['sources'])!=2 or {s['phrase_id'] for s in pilot['sources']}!=set(PHRASES):
        raise ValueError('Pilot identity inventory changed')
    for source in pilot['sources']:
        row=by_id[source['phrase_id']]
        if source['support_seconds']!=row['candidate']['support_seconds'] or not source['identity_pass']:
            raise ValueError('Pilot support or identity differs from frozen geometry')
    reused={(a['phrase_id'],a['cents']):a for a in attempts};inventory=[]
    for row in dev:
        for cents in SHIFTS:
            key=(row['phrase_id'],cents)
            if key in reused:
                item=deepcopy(reused[key]);item['origin']='verified_pilot_reuse'
            elif row['candidate'] is None:
                item=dict(phrase_id=row['phrase_id'],cents=cents,status='not_attempted_geometry_refused',passed=False,
                    reason='No candidate under the unchanged final mask and fixed geometry rules',origin='geometry_refusal')
            else:
                item=dict(phrase_id=row['phrase_id'],cents=cents,status='not_started',passed=False,origin='new_completion')
            inventory.append(item)
    selected=[r for r in dev if r['candidate'] is not None and r['phrase_id'] not in PHRASES]
    if len(selected)!=10 or sum(r['candidate'] is None for r in dev)!=2:
        raise ValueError('Frozen ten-candidate / two-refusal inventory changed')
    return dev,selected,inventory


def completion_summary(attempts, sources):
    expected={(s['phrase_id'],c) for s in sources for c in SHIFTS}
    if len(attempts)!=len(expected) or {(a['phrase_id'],a['cents']) for a in attempts}!=expected:
        raise ValueError('Missing, duplicated or undeclared audit outcome')
    terminal={'measured','not_attempted_geometry_refused','not_attempted_identity_or_source_refused','execution_refused'}
    if any(a['status'] not in terminal or (a['passed'] and a['status']!='measured') for a in attempts):
        raise ValueError('Unfinished or improperly promoted audit outcome')
    passed=[a for a in attempts if a['passed']];ids={a['phrase_id'] for a in passed}
    both=[s for s in sources if all(a['passed'] for a in attempts if a['phrase_id']==s['phrase_id'])]
    return dict(development_phrases=len(sources),prospective_keys=len(expected),
        measured_shifts=sum(a['status']=='measured' for a in attempts),passed_shifts=len(passed),
        failed_measured_shifts=sum(a['status']=='measured' and not a['passed'] for a in attempts),
        geometry_refused_shifts=sum(a['status']=='not_attempted_geometry_refused' for a in attempts),
        reused_measured_shifts=sum(a['origin']=='verified_pilot_reuse' and a['status']=='measured' for a in attempts),
        newly_measured_shifts=sum(a['origin']=='new_completion' and a['status']=='measured' for a in attempts),
        phrases_with_any_pass=len(ids),recordings_with_any_pass=len({s['source_id'] for s in sources if s['phrase_id'] in ids}),
        singers_with_any_pass=len({s['singer_id'] for s in sources if s['phrase_id'] in ids}),
        phrases_with_both_directions_pass=len(both),recordings_with_both_directions_pass=len({s['source_id'] for s in both}),
        singers_with_both_directions_pass=len({s['singer_id'] for s in both}),new_pairs=0,new_optimizer_updates=0,research_ready_examples=0)
