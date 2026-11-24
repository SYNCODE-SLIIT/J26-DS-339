"""Evaluate an independently partitioned reference-controlled pairs manifest."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from bpdr_input.experiments.learning.context_improvement import ROOT, read, run_evaluation, sha
from bpdr_input.reporting.transformer_accuracy_report import complete_report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    choice=parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--latest',action='store_true',help='Check the default seed from the latest complete experiment')
    choice.add_argument('--checkpoint',type=Path)
    parser.add_argument('--pairs-manifest',type=Path)
    parser.add_argument('--output-root',type=Path)
    parser.add_argument('--render',action='store_true')
    args=parser.parse_args()
    candidate_pointer=ROOT.parent/'transformer_usability/latest_candidate.json'
    if args.latest and candidate_pointer.exists():
        if args.pairs_manifest:parser.error('--latest selects the candidate recorded evaluation manifest')
        current=read(candidate_pointer);run=Path(current['path']).resolve()
        if not run.is_relative_to((ROOT.parent/'transformer_usability').resolve()):raise SystemExit('Candidate pointer leaves the component experiment root')
        checks=[(run/'final/seed_339/checkpoint.pt',current['checkpoint_sha256']),(run/'locked_policy.json',current['policy_sha256']),
                (run/'final_evaluation/evidence.json',current['verification_sha256'])]
        if any(sha(file)!=expected for file,expected in checks):raise SystemExit('Latest candidate identity or verification changed')
        sys.path.insert(0,str(Path(__file__).resolve().parent))
        from check_held_voice_accuracy import check
        print('LATEST_CANDIDATE_STATUS',current['status'],'DEMO_ACTIVATED',current['demo_activated'])
        check(run,render=args.render,output_root=args.output_root)
        raise SystemExit(0)
    if args.latest:
        from bpdr_input.repair.held_voice_runtime import CARD
        if CARD.exists():
            if args.pairs_manifest:parser.error('--latest uses the activated model evaluation manifest')
            sys.path.insert(0,str(Path(__file__).resolve().parent))
            from check_held_voice_accuracy import check
            check(latest=True,render=args.render,output_root=args.output_root)
            raise SystemExit(0)
    if args.latest:
        if args.pairs_manifest:
            parser.error('--latest already selects the declared calibration manifest')
        pointer=ROOT/'latest_experiment.json'
        if not pointer.exists():
            raise SystemExit('No completed improvement experiment yet. Explicit checkpoints can still be evaluated.')
        current=read(pointer)
        run=Path(current['path'])
        if sha(run/'evidence.json')!=current['evidence_sha256']:
            raise SystemExit('Experiment evidence hash mismatch')
        evidence=read(run/'evidence.json')
        checkpoint=evidence['default_checkpoint']
        if sha(checkpoint)!=evidence['default_checkpoint_sha256']:
            raise SystemExit('Selected checkpoint hash mismatch')
        manifest=Path(read(run/'declaration.json')['data_run'])/'calibration_pairs.json'
    else:
        if not args.pairs_manifest:
            parser.error('--checkpoint requires --pairs-manifest')
        checkpoint,manifest=args.checkpoint,args.pairs_manifest
    folder=run_evaluation(checkpoint,manifest,args.output_root,args.render)
    summary=complete_report(folder)
    print('ENCODED_VERIFIED_WAVS',summary['unique_encoded_verified_wavs'])
    print('IMPROVEMENT_PERCENT_ALL_ATTEMPTS',summary['verified_improvement_percent_all_attempts'])
