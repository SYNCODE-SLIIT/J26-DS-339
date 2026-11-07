"""Explain missing input-only pulse evidence; references used only for diagnosis."""
from pathlib import Path
from collections import defaultdict
import sys,json
import numpy as np
from scipy.ndimage import median_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

WS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WS/'source'));sys.path.insert(0,str(WS/'tools'))
from probe_short_phrase_renderer import sha
from bpdr_input.modeling.short_event_evidence import _regions

DATA=WS/'results/transformer_usability/2026-11-02_v40_short_phrase_data'
ROOT=WS/'results/transformer_usability/2026-11-02_v43_short_event_evidence'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def main():
    rows=read(DATA/'train_pairs.json');originals={s['source_id']:s for s in read(DATA/'train_originals.json')}
    reports=[];examples={}
    for index,row in enumerate(rows):
        if 'excluded' in row:continue
        source=originals[row['source_id']];a,b=source['longest_region_frames'];start,end=row['support_seconds']
        with np.load(source['analysis_A']) as z:fa=z['fine'].copy();ra=z['reliable'].copy()
        with np.load(row['analysis_C']) as z:fc=z['fine'].copy();rc=z['reliable'].copy();raw=z['raw_views'][0].copy();times=z['times_seconds'].copy()
        region=(times>=a*.02)&(times<b*.02)
        left=region&(times<start-.07)&rc;right=region&(times>end+.07)&rc
        missing=region&~rc;changes=np.diff(np.r_[False,missing,False].astype(int))
        gaps=[int(y-x) for x,y in zip(np.flatnonzero(changes==1),np.flatnonzero(changes==-1))]
        voiced=raw[:,66]>.5;x=np.arange(len(raw));pitch=np.interp(x,x[voiced],raw[voiced,64]) if voiced.any() else np.zeros(len(raw))
        smooth=median_filter(pitch,size=3,mode='nearest');context=voiced.copy()
        d=np.diff(np.r_[False,~voiced,False].astype(int))
        for j,k in zip(np.flatnonzero(d==1),np.flatnonzero(d==-1)):
            if j>0 and k<len(raw) and k-j<=2:context[j:k]=True
        detected=_regions(smooth,context);center=(start+end)/2
        center_regions=[(j,k) for j,k in detected if j*.02<=center<k*.02]
        maximum=max(((k-j)*.02 for j,k in center_regions),default=0.)
        usable=np.flatnonzero(region&rc);fit=None
        if len(usable)>=6:
            t=(times[usable]-center);pulse=((times>=start)&(times<end)).astype(float)
            pulse=np.convolve(pulse,[.1,.2,.4,.2,.1],mode='same')[usable]
            base=np.column_stack([np.ones(len(t)),t]);null=fc[usable]-base@np.linalg.lstsq(base,fc[usable],rcond=None)[0]
            design=np.column_stack([base,pulse]);coef=np.linalg.lstsq(design,fc[usable],rcond=None)[0]
            residual=fc[usable]-design@coef
            fit=dict(level_cents=float(coef[2]),gain=float(1-np.sum(residual**2)/np.sum(null**2)) if np.sum(null**2)>1e-8 else 0.,residual_cents=float(np.std(residual)))
        report=dict(pair_index=index,group=row['group'],source_id=row['source_id'],width_seconds=row['width_seconds'],cents=row['cents'],
            original_region_seconds=(b-a)*.02,corrupted_region_containing_center_seconds=maximum,
            reliable_left_anchor_frames=int(left.sum()),reliable_right_anchor_frames=int(right.sum()),
            unreliable_frames_in_source_region=int(missing.sum()),maximum_unreliable_gap_frames=max(gaps,default=0),
            known_support_oracle_fit_NOT_RUNTIME=fit)
        reports.append(report)
        # Fixed deterministic display: smallest positive-shift index per singer.
        if row['cents']==30. and row['group'] not in examples:
            examples[row['group']]=(row,source,times,fa,fc,ra,rc,raw,report)
    fig,axes=plt.subplots(6,2,figsize=(15,17));fig.patch.set_facecolor('#f8fafc')
    for axisrow,(group,values) in zip(axes,sorted(examples.items()),strict=True):
        row,source,times,fa,fc,ra,rc,raw,report=values;a,b=source['longest_region_frames'];start,end=row['support_seconds']
        local=(times>=max(0,a*.02-.16))&(times<=min(times[-1],b*.02+.16));reference=float(np.median(fa[a:b]))
        ax=axisrow[0];ax.plot(times[local],fa[local]-reference,color='#64748b',label='Original fine pitch',lw=1)
        ax.plot(times[local],fc[local]-reference,color='#cbd5e1',label='C fine pitch, including withheld context',lw=1)
        trusted=local&rc;ax.plot(times[trusted],fc[trusted]-reference,'.',color='#0369a1',ms=3,label='Pitch retained for model context')
        ax.axvspan(start,end,color='#f59e0b',alpha=.15,label='Known intervention (diagnosis only)')
        ax.set_title(group+' / '+row['source_id']+' / '+str(row['width_seconds'])+' s',loc='left',fontsize=10,fontweight='bold')
        ax.set_ylabel('Cents relative to original local median',fontsize=8)
        with np.load(ROOT/f'C_{report["pair_index"]}.npz') as z:cue=z['new'][:,0]*30
        ax2=axisrow[1];ax2.plot(times[local],cue[local],color='#0f766e',label='Input-only short-event cue')
        ax2.axhline(-row['cents'],color='#c2410c',ls='--',lw=.8,label='Nominal correction (not input)')
        ax2.axvspan(start,end,color='#f59e0b',alpha=.15)
        ax2.text(.02,.05,f"Trusted side observations: {report['reliable_left_anchor_frames']} / {report['reliable_right_anchor_frames']}\nC region at center: {report['corrupted_region_containing_center_seconds']:.2f}s; source region: {report['original_region_seconds']:.2f}s",transform=ax2.transAxes,fontsize=8,bbox=dict(facecolor='white',edgecolor='none',alpha=.85))
        ax2.set_ylabel('Measured cue cents; no permission',fontsize=8)
        for axy in axisrow:
            axy.set_xlim(times[local][0],times[local][-1]);axy.set_xlabel('Seconds',fontsize=8);axy.grid(alpha=.16);axy.tick_params(labelsize=8)
            axy.spines[['top','right']].set_visible(False)
    axes[0,0].legend(fontsize=7,loc='upper left');axes[0,1].legend(fontsize=7,loc='upper right')
    fig.suptitle('Why short-event evidence misses the measured deviations',x=.05,ha='left',fontweight='bold',fontsize=17)
    fig.text(.05,.953,'One predetermined positive-shift case per scoreable singer shown; all44 cases are retained in the JSON. References/supports appear only in this diagnostic.',fontsize=9)
    fig.tight_layout(rect=[.02,.02,.99,.94]);plot=ROOT/'source_context_diagnosis.png';fig.savefig(plot,dpi=140);plt.close(fig)
    output=dict(script_sha256=sha(__file__),diagnostic_only=True,research_validation_passed=False,
        all44_cases=reports,plot_sha256=sha(plot),
        missing_both_three_frame_anchors=sum(r['reliable_left_anchor_frames']<3 or r['reliable_right_anchor_frames']<3 for r in reports),
        corrupted_center_region_shorter_than_width=sum(r['corrupted_region_containing_center_seconds']<r['width_seconds'] for r in reports))
    (ROOT/'context_diagnosis.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('SHORT_CONTEXT_DIAGNOSIS_COMPLETE',output['missing_both_three_frame_anchors'],output['corrupted_center_region_shorter_than_width'],flush=True)


if __name__=='__main__':main()
