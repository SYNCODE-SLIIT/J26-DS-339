"""Reader-facing evidence snapshot, generated only after the model comparison exists."""
from pathlib import Path
import json,hashlib
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,KeepTogether

WS=Path(__file__).resolve().parents[1]
BASE=WS/'results/transformer_usability'
OUT=WS/'output/pdf/BPDR_Current_System_and_Transformer_Progress_2026-11-02.pdf'
NAVY=colors.HexColor('#13283f');TEAL=colors.HexColor('#007e83');INK=colors.HexColor('#25384a')
MUTED=colors.HexColor('#596d7e');PALE=colors.HexColor('#eff5f8');WHITE=colors.white
CONTENT_WIDTH=A4[0]-112
STYLES={
 'title':ParagraphStyle('title',fontName='Helvetica-Bold',fontSize=28,leading=32,textColor=NAVY,spaceAfter=16),
 'h1':ParagraphStyle('h1',fontName='Helvetica-Bold',fontSize=22,leading=26,textColor=NAVY,spaceAfter=14),
 'h2':ParagraphStyle('h2',fontName='Helvetica-Bold',fontSize=12,leading=16,textColor=TEAL,spaceBefore=14,spaceAfter=7),
 'body':ParagraphStyle('body',fontName='Helvetica',fontSize=10,leading=15,textColor=INK,spaceAfter=8),
 'small':ParagraphStyle('small',fontName='Helvetica',fontSize=8.5,leading=12,textColor=MUTED,spaceAfter=6),
 'cell':ParagraphStyle('cell',fontName='Helvetica',fontSize=9,leading=12,textColor=INK),
 'head':ParagraphStyle('head',fontName='Helvetica-Bold',fontSize=8.5,leading=11,textColor=WHITE),
 'number':ParagraphStyle('number',fontName='Helvetica-Bold',fontSize=28,leading=32,textColor=TEAL),
 'kicker':ParagraphStyle('kicker',fontName='Helvetica-Bold',fontSize=9,leading=12,textColor=TEAL,spaceAfter=9),
}

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def p(text,style='body'):return Paragraph(text,STYLES[style])
def table(headers,rows,widths):
    data=[[p(escape(str(x)),'head') for x in headers]]+[[p(escape(str(x)),'cell') for x in row] for row in rows]
    t=Table(data,colWidths=[w*CONTENT_WIDTH/sum(widths) for w in widths],hAlign='LEFT',repeatRows=1)
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),NAVY),('VALIGN',(0,0),(-1,-1),'TOP'),
        ('LEFTPADDING',(0,0),(-1,-1),9),('RIGHTPADDING',(0,0),(-1,-1),9),
        ('TOPPADDING',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),8),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[WHITE,PALE]),('LINEBELOW',(0,0),(-1,0),1,TEAL)]))
    return t
def notice(text):
    t=Table([[p(text)]],colWidths=[CONTENT_WIDTH],hAlign='LEFT');t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),PALE),
       ('BOX',(0,0),(-1,-1),.5,colors.HexColor('#d4e1e8')),('LEFTPADDING',(0,0),(-1,-1),13),
       ('TOPPADDING',(0,0),(-1,-1),11),('BOTTOMPADDING',(0,0),(-1,-1),5)]));return t
def section(kicker,title):return [p(kicker.upper(),'kicker'),p(title,'h1')]
def footer(canvas,doc):
    canvas.saveState();canvas.setStrokeColor(colors.HexColor('#d4e1e8'));canvas.line(50,46,545,46)
    canvas.setFont('Helvetica',8);canvas.setFillColor(MUTED)
    canvas.drawString(50,31,'J26-DS-339 | BPDR input-preparation component | 2 November 2026')
    canvas.drawRightString(545,31,str(doc.page));canvas.restoreState()


def main():
    model_root=BASE/'2026-11-02_v41_short_phrase_models'
    verified=(model_root/'results.json').exists() and (model_root/'integrity_receipt.json').exists()
    comparison=read(model_root/'results.json') if verified else None
    if verified:
        receipt=read(model_root/'integrity_receipt.json')
        assert receipt['checkpoints_reloaded']==12 and receipt['results_sha256']==sha(model_root/'results.json')
    delivery_path=BASE/'2026-11-02_v42_short_delivery/results.json'
    delivery=read(delivery_path) if delivery_path.exists() else None
    if delivery:
        receipt=read(BASE/'2026-11-02_v42_short_delivery/integrity_receipt.json')
        assert receipt['results_sha256']==sha(delivery_path)
    rejection=''
    if delivery:
        s=delivery['arms']['short_transformer']['summary']
        rejection=f" The latest revised candidate is rejected: {s['verified_improvements']} target-plateau gains, {s['verified_harms']} target-plateau harms and failure to preserve every original control."
    names={'frozen_joint':'Current frozen Transformer','joint_retrained':'Retrained old representation',
           'short_transformer':'Revised short-phrase Transformer','short_extra_trees':'Matched ExtraTrees'}
    story=[p('RESEARCH &amp; ENGINEERING PROGRESS','kicker'),p('BPDR current system<br/>and Transformer improvement','title'),
        p('A source-backed snapshot of what works, what changed, and what remains unproved.'),
        p('Component owner: Liyanage L. N. P. / IT23141506. Canonical implementation: BPDR_Workspace.','small'),
        notice('<b>Current position:</b> the goal is a 5-15-second melodic hum with short notes. The revised model is experimental.'+rejection+' No general usable-accuracy claim, research-validation pass or Transformer activation is asserted.'),Spacer(1,16)]
    metric=Table([[p('44','number'),p('6','number'),p('336','number')],
                  [p('Audited short-deviation examples','small'),p('Singers with valid examples','small'),p('Planned construction attempts','small')]],colWidths=[CONTENT_WIDTH/3]*3,hAlign='LEFT')
    story.extend([metric,p('What has been completed','h2'),
        p('The project has a technical vocal-input preparation path, measured acoustic features, bounded experimental repairs, rollback, encoded-WAV checks and traceable handoff machinery. The existing shared demo selection remains ExtraTrees; this work has not replaced it.'),
        p('The latest iteration adds short-support measurement experiments, a training-only melodic dataset, a genuine Transformer with dense short-event context, and a matched singer-separated comparison. Tests and reproducible receipts support these engineering statements.'),
        p('Research question','h2'),p('Does boundary-robust training with counter-detuning and preservation anchors improve local audio repair under segmentation errors, without more harmful changes at matched acceptance coverage? The full research comparison still needs matched augmentation-only and frame-only controls; this engineering iteration does not establish that advantage.'),
        p('What is still needed','h2'),p('Reliable support for the shortest notes, adequate coverage across unfamiliar voices, verified beneficial delivered audio, unchanged original controls, independent reference/listening review and a qualified user-facing model. A low prediction error alone does not complete these requirements.'),PageBreak()])
    story+=section('01 / System','What the component does')
    story+=[p('BPDR prepares a short solo vocal idea for later musical analysis. Its output is an accepted waveform with status and provenance, or an actionable reason to keep or record the input again.'),
        table(['Stage','Current behavior'],[
          ['Recording and technical checks','Validate format, duration, finite samples and clipping; preserve the source and map the working copy.'],
          ['Acoustic analysis','Measure pitch, voicing, energy and provisional boundary evidence from the input waveform.'],
          ['Learned proposal','Predict a relative local correction. The new Transformer is experimental and does not grant export permission.'],
          ['Permission and preservation','Bound correction size and affected duration; veto unreliable, protected or disagreeing proposals.'],
          ['Rendering and verification','Praat local rendering; check in-memory and decoded PCM24 audio; roll back failures.'],
          ['Review and handoff','Accepted-only package machinery exists. The proposed individual repair/protection interface remains planned.']],[125,370]),
        p('Ownership boundaries','h2'),table(['BPDR owns','Other components own'],[
          ['Audio validation, local waveform repair, protected time ranges, verification and provenance','Final note transcription and durations; beats and tempo; key/mode; genre; harmony and chords']],[247.5,247.5]),
        p('Internal acoustic regions are time-based evidence. They are not a second public melody or note representation. No new key, chord or beat feature is introduced here.','small'),PageBreak()]
    story+=section('02 / Diagnosis','Why this has been difficult')
    story+=[p('The earlier benchmark was built around long held tones. The intended product now needs ordinary melodic phrases, so several non-model constraints excluded short notes before learning could help.'),
        table(['Old constraint','Consequence'],[
          ['Stable acoustic region >=1.10 s','Many short-note inputs are ineligible before prediction.'],
          ['Minimum repair support 0.86 s','Short proposed edits cannot reach the old renderer/controller.'],
          ['80 ms ramp +100 ms analysis guard per side','360 ms disappears before the audit counts interior frames.'],
          ['Ten common 20 ms frames','A short edit can execute yet remain unmeasurable under this protocol.'],
          ['Old step feature searches widths >=0.86 s','Its engineered evidence does not describe short deviations.']],[235,260]),
        p('Fresh feasibility check','h2'),p('V37 retained twelve predetermined source files and 48 attempted deviations. All 48 were excluded before any correction-Transformer inference because no source region met the held-tone reference geometry. This is an unassessed accuracy result, not 0% model accuracy.'),
        p('A measured timing improvement','h2'),p('The installed PESTO acoustic window extends at most 66.440 ms per side; YIN extends 46.440 ms. A 70 ms guard was tested without changing pitch-error, direction, coverage or ten-frame limits.'),
        table(['Real-voice timing probe','Passes / planned requests'],[['Baseline timing','8 /96'],['70 ms guard','17 /96'],['70 ms guard +20 ms ramp','23 /96']],[330,165]),
        p('These are forward-shift measurement passes, not trained correction accuracy. The 0.24 s gap and all-zero JSUT results remain unresolved. Runtime defaults were not changed.','small'),PageBreak()]
    story+=section('03 / New iteration','Melodic data and short context')
    story+=[p('The existing Vocadito singer assignments were retained. Only the training partition was acoustically accessed in this iteration; no human annotation array was loaded into training or inference.'),
        table(['Existing partition','Recordings','Singers'],[['Training','21','14'],['Development','8','6'],['PP1 evaluation','8','6'],['Final holdout','3','3']],[275,110,110]),
        p('Construction accounting','h2'),table(['Disposition','Attempts'],[
          ['Valid audited examples','44'],['Insufficient single reliable region/context','176'],
          ['Physical intervention audit failed','84'],['Source preprocessing refused','32'],['Total planned attempts','336']],[385,110]),
        p('Nineteen working originals were created. Two sources were refused because resampling produced peaks above 1.0 (1.00920 and 1.00328); no gain normalization rescued them. The 44 valid examples cover 0.36, 0.50 and 0.70 s, from six singers. No 0.24 s example passed.','small'),
        p('What changed in the Transformer','h2'),p('The new model keeps a genuine two-layer, four-head, 32-wide Transformer. It uses 81 context tokens spanning four seconds, with dense 20 ms samples near the event and sparser context farther away. Eight continuous acoustic channels include relative pitch, local change, confidence, voicing, energy and short-scale pitch contrasts.'),
        p('There are no intended notes, key labels, clean references, source IDs or known edit supports in the inference interface. Original recordings supply preservation training examples. Five feature/interface checks and four policy checks passed.'),PageBreak()]
    story+=section('04 / Model evidence','Singer-separated prediction results')
    rows=[]
    for arm in ['frozen_joint','joint_retrained','short_transformer','short_extra_trees']:
        if comparison:
            s=comparison['arms'][arm]
            rows.append([names[arm],f"{s['positive_singer_macro_mae_cents']:.2f}",f"{s['original_singer_macro_mae_cents']:.2f}",'Pass' if s['raw_gate_passed'] else 'Fail'])
        else:rows.append([names[arm],'Unassessed','Unassessed','Pending'])
    status=('All 12 checkpoints and their predictions were independently reloaded and checked.' if comparison else
            f"Fitting is running: {len(list(model_root.glob('fold_*/*/fit_receipt.json')))}/12 fit receipts exist. Final prediction results and independent checkpoint replay are not yet available; no accuracy value is asserted.")
    story+=[p('Four grouped folds keep fitted singers and recordings out of their scoring fold. Each neural fit has 800 updates; ExtraTrees consumes the exact same sampled rows as the revised Transformer. '+status),
        table(['Model','Correction MAE (c)','Original MAE (c)','Raw gate'],rows,[225,95,95,80]),
        p('Lower is better; 100 cents equals one semitone. Correction MAE is singer-macro error on audited nonzero plateaus; original MAE measures raw predicted changes on reliable original frames. These are different quantities. The raw gate additionally checks each represented group and signed response; averages alone cannot pass it.','small'),
        p('Revised Transformer: contributing singers','h2')]
    per=[]
    for g in (comparison['arms']['short_transformer']['groups'] if comparison else []):
        if g['positive_mae_cents'] is not None:
            per.append([g['group'],str(g['scoreable_pairs']),f"{g['positive_mae_cents']:.2f}",f"{g['original_mae_cents']:.2f}" if g['original_mae_cents'] is not None else 'Unassessed',f"{g['signed_response']:.2f}"])
    if per:story.append(table(['Singer','Valid pairs','Correction MAE','Original MAE','Response'],per,[65,85,120,120,105]))
    else:story.append(p('Valid positive examples come from S5, S7, S22, S23, S28 and S29. Per-singer results will remain unassessed until the full comparison and replay finish. The eight other attempted training singers have no valid positive example under this construction protocol.'))
    story+=[
        notice('This is controlled-recovery engineering evidence from a small, previously inspected training corpus. It does not establish natural intended-pitch accuracy, unfamiliar-user usability or the BPDR research advantage.'),
        p('The frozen model has a different training history. Old-versus-revised Transformer also changes representation and token density. Only revised Transformer versus matched ExtraTrees uses identical input tokens; no claim of optimal tuning or equal compute is made.','small')]
    generalization=BASE/'2026-11-02_v41_short_phrase_models/generalization_diagnosis/results.json'
    if generalization.exists():
        diagnosis=read(generalization)
        fit=[f['training']['positive_singer_macro_mae_cents'] for f in diagnosis['folds']]
        held=[f['heldout']['positive_singer_macro_mae_cents'] for f in diagnosis['folds']]
        story.append(p(f"The fixed revised checkpoints score {min(fit):.1f}-{max(fit):.1f} cents on fitted singers versus {min(held):.1f}-{max(held):.1f} on held-out singers. This demonstrates a transfer gap; it does not identify a single cause or prove that another architecture would solve the task.",'small'))
    story.append(PageBreak())
    story+=section('05 / Delivered audio','The WAV is the decisive output')
    if delivery:
        rows=[]
        for arm in ['frozen_joint','joint_retrained','short_transformer','short_extra_trees']:
            s=delivery['arms'][arm]['summary']
            changed=sum(r['kind']=='original' and not r['waveform_unchanged'] for r in delivery['arms'][arm]['records'])
            rows.append([names[arm],f"{s['verified_improvements']}/336",str(s['verified_harms']),f"{s['singer_macro_after_mae_cents']:.2f}",f'{changed}/19'])
        story+=[table(['Model','Verified gains','Target harms','Remaining MAE (c)','Changed originals'],rows,[190,75,70,85,75]),
            p('The same 336 planned attempts, 44 scoreable cases and 19 valid original controls are retained per arm. The 292 construction exclusions and two refused original sources are explicit. Target harms are measured on the same known-corruption plateau as target gains; they do not exhaust collateral or musical harms. Counts above are not population accuracy. All arms use identical permission, renderer and audit rules.','small'),
            notice('<b>Reject the revised candidate.</b> Its four measured gains do not outweigh four target harms and two changed original controls. Overall remaining error is worse than keeping the input. Technical fidelity to a predicted correction does not prove that the correction was appropriate.')]
        collateral=BASE/'2026-11-02_v42_short_delivery/reference_preservation_diagnosis.json'
        if collateral.exists():
            c=read(collateral)['arms']['short_transformer']
            story.append(p(f"A separate post-hoc reference diagnostic finds increased pitch error on known-unchanged frames in {c['outputs_increasing_known_zero_error']}/{c['verified_corrupted_outputs']} verified corrupted-input outputs. This collateral measure is separate from the locked target-plateau gain count.",'small'))
    else:
        story+=[notice('<b>Delivery qualification pending.</b> No result from prediction alone is presented as a verified repair. The prospective V42 protocol is locked; output bytes and independent replay are required before filling in delivered performance.'),Spacer(1,10)]
    story+=[p('Fixed evaluation path','h2'),p('Predict from input acoustics only, create a bounded proposal, veto protected or unreliable intervals, render the permitted candidate, audit actual pitch movement, encode PCM24, decode and repeat the integrity/acoustic checks. Only then score against the separate reference.'),
        table(['Requirement','What must be retained'],[
          ['Preservation','Original controls and exact samples outside/protected intervals.'],
          ['Safety bounds','Maximum 30-cent correction, maximum 25% edited duration; no clipping or duration padding.'],
          ['Acoustic verification','Fine and coarse pitch error, direction and coverage checks; explicit rollback.'],
          ['Reference scoring','The same predeclared frames before and after, with no easier output-dependent denominator.'],
          ['Human suitability','User intention and independent listening remain separate, currently incomplete evidence.']],[145,350]),
        p('The short-support controller is an experimental opt-in module. It has not replaced the shared demo or made an accepted-user export. Boundary-view agreement is invariance by construction, not calibrated model confidence.'),PageBreak()]
    story+=section('06 / Next decisions','Finish usability before expanding features')
    story+=[table(['Priority','Completion evidence needed'],[
          ['1. Short-phrase model','Better held-singer delivered audio, adequate duration/source coverage, unchanged originals and retained failures.'],
          ['2. Shortest-note measurement','Resolve the 0.24 s audit gap without hiding failures or treating overlapping frames as independent evidence.'],
          ['3. Independent confirmation','Freeze data recipe, model, permission and renderer; reserve unfamiliar sources before acoustic inspection.'],
          ['4. Musical review','Two independent reviewers and intention-aware recordings; do not invent approvals or unknown target notes.'],
          ['5. Presentation feature','Protected regions, per-repair A/B review and Accept/Keep controls, then final combined-WAV re-audit. Still planned.']],[145,350]),
        p('Recording and review pilot','h2'),p('The collection plan proposes eight adults with six 5-15-second phrases each, four development participants and four untouched participants, plus two musical reviewers. Guided phrases need approved stimuli/reference alignment; free ideas may have unknown intended pitch. Recruitment, consent, authorization and ratings have not been completed or claimed.'),
        p('Evidence map','h2'),p('V37: fresh reference feasibility and exclusions. V38-V39: short-support rendering/measurement. V40: source reservation, 336 attempts, labels and independent data replay. V41: model declarations, folds, sample traces, checkpoints and raw prediction replay. V42: prospective delivered-WAV comparison. All are under results/transformer_usability/ in the canonical workspace.','small'),
        p('Sources and companion plans','h2'),
        p('<link href="https://zenodo.org/records/5578807" color="#007e83">Bittner et al. (2021), Vocadito dataset, DOI 10.5281/zenodo.5578807</link>. The official mirdata loader identifies CC BY 4.0 licensing. Annotations describe performed pitch; they are not unknown natural intention.','small'),
        p('<link href="https://github.com/SonyCSLParis/pesto" color="#007e83">Sony CSL Paris, PESTO official implementation</link>; installed code and weights are hashed in the experiment declarations. <link href="https://arxiv.org/abs/2511.20006v3" color="#007e83">Kim et al. (2026), BERT-APC v3</link> motivates expression preservation and delivered-audio assessment; its symbolic musical inference is outside BPDR ownership.','small'),
        p('Companion documents: BPDR_Master_Plan.md; Short_Phrase_Transformer_Plan_2026-11-02.md; Short_Support_Timing_Study_2026-11-02.md; Review_and_Collection_Plan_2026-11-01.md. A completed engineering experiment is not a research-validation pass.','small')]
    OUT.parent.mkdir(parents=True,exist_ok=True)
    SimpleDocTemplate(str(OUT),pagesize=A4,rightMargin=50,leftMargin=50,topMargin=45,bottomMargin=62,
        title='BPDR Current System and Transformer Progress',author='BPDR / IT23141506').build(story,onFirstPage=footer,onLaterPages=footer)
    print(OUT)


if __name__=='__main__':main()
