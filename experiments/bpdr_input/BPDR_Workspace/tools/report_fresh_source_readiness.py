"""Source-readiness profile for the locked confirmation pack; no corrector predictions."""
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import median_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import prepare_fresh_confirmation as prep
from bpdr_input.repair.held_voice_policy import eligibility,spans
from bpdr_input.repair.rule_correction import segment_notes,RulePolicy


def main():
    d,_=prep.declare();ex=prep.ex;root=prep.PREP;sources=ex.read(root/'original_sources.json');reports=[]
    fig,axes=plt.subplots(4,3,figsize=(17,12));fig.patch.set_facecolor('#f8fafc')
    for source,ax in zip(sources,axes.flat,strict=True):
        audio,analysis=prep.recovery.from_cached(source['working_audio'],Path(source['working_audio']).parent/'original_features.npz',{})
        raw=analysis['views'][0];times=analysis['features']['times_seconds'];allowed,reliable,_=eligibility(raw,region_split_cents=70.)
        regions=segment_notes(raw[:,64],reliable,RulePolicy(note_split_cents=70.));long=[]
        for a,b in regions:
            if (b-a)*.02<1.8-1e-12:continue
            y=raw[a:b,64];x=np.arange(len(y));rawspread=float(np.std(y-np.polyval(np.polyfit(x,y,1),x)))
            y=median_filter(raw[a+3:b-3,64],size=9,mode='nearest');x=np.arange(len(y));smoothspread=float(np.std(y-np.polyval(np.polyfit(x,y,1),x)))
            long.append(dict(start_seconds=float(a*.02),end_seconds=float(b*.02),raw_detrended_spread_cents=rawspread,median9_interior_detrended_spread_cents=smoothspread,raw20_gate=rawspread<=20.,median10_gate=smoothspread<=10.))
        r=dict(singer_id=source['source']['singer_id'],archive_member=source['source']['archive_member'],duration_seconds=len(audio)/22050,frames=len(raw),reliable_frames=int(reliable.sum()),allowed_frames=int(allowed.sum()),max_reliable_run_seconds=max(((b-a)*.02 for a,b in spans(reliable)),default=0.),max_acoustic_region_seconds=max(((b-a)*.02 for a,b in regions),default=0.),long_regions=long,long_regions_passing_raw20=sum(x['raw20_gate'] for x in long),long_regions_passing_both_spreads=sum(x['raw20_gate'] and x['median10_gate'] for x in long),correction_transformer_inference=False)
        reports.append(r)
        fine=np.where(reliable,raw[:,64],np.nan);smooth=median_filter(raw[:,64],size=9,mode='nearest');smooth=np.where(reliable,smooth,np.nan)
        ax.plot(times,fine,color='#94a3b8',lw=.65,label='Fine pitch');ax.plot(times,smooth,color='#0369a1',lw=1,label='180 ms median')
        for a,b in spans(allowed&reliable):ax.axvspan(a*.02,b*.02,color='#16a34a',alpha=.15)
        ax.set_xlim(0,len(audio)/22050);ax.set_title(source['source']['singer_id']+' · '+Path(source['source']['archive_member']).name,fontsize=10,loc='left',fontweight='bold')
        ax.text(.02,.03,f"Long regions: {len(long)} | pass raw spread20c: {r['long_regions_passing_raw20']}",transform=ax.transAxes,fontsize=8,bbox=dict(facecolor='white',alpha=.9,edgecolor='none'))
        ax.set_xlabel('Time (seconds)',fontsize=8);ax.set_ylabel('Pitch cents relative to own median',fontsize=8);ax.grid(alpha=.16);ax.tick_params(labelsize=8);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Fresh recordings: readiness for the frozen held-region protocol',fontsize=17,fontweight='bold',x=.05,ha='left')
    fig.text(.05,.935,'All 12 predetermined recordings shown. Green = existing input eligibility. No correction-Transformer output is plotted.',fontsize=10,color='#475569')
    fig.text(.05,.015,'Reference construction requires at least 1.80 s source regions, raw detrended spread ≤20c and 180 ms-median interior spread ≤10c. Fixed first ≤20 s crops.',fontsize=9,color='#475569')
    fig.tight_layout(rect=[.025,.04,.995,.92]);path=root/'source_readiness.png';fig.savefig(path,dpi=160,facecolor=fig.get_facecolor());plt.close(fig)
    ex.write(root/'source_readiness.json',dict(sources=reports,source_count=12,original_raw20_failures_among_long_regions=sum(sum(not x['raw20_gate'] for x in r['long_regions']) for r in reports),source_plot_sha256=ex.sha(path),reporter_source_sha256=ex.sha(Path(__file__)),correction_transformer_inference_started=False,diagnostic_only=True,research_validation_passed=False))
    print('READINESS_PROFILE_COMPLETE',[(r['singer_id'],len(r['long_regions']),r['long_regions_passing_raw20'],r['long_regions_passing_both_spreads']) for r in reports],flush=True)

if __name__=='__main__':main()
