"""Sequential declared construction, fixed-budget learning, output and diagnosis checks.

Invoke only after V64 construction exits successfully. Each phase leaves a durable
status; nonzero exits stop before later phases. Delivered evaluation is separate.
"""
from pathlib import Path
import json
import argparse
import subprocess
import sys
from datetime import datetime,timezone

WS=Path(__file__).resolve().parents[1]
STATUS=WS/'results/transformer_usability/preservation_pipeline_status.json'
PHASES=['verify_reliable_melodic_pairs.py','report_reliable_melodic_coverage.py','compare_preservation_sampling.py','verify_preservation_sampling.py','report_preservation_raw.py','evaluate_preservation_delivery.py','verify_preservation_delivery.py','report_preservation_delivery.py','trace_preservation_delivery.py','diagnose_preservation_sampling.py','verify_preservation_diagnosis.py','compare_tail_preservation.py','verify_tail_preservation.py','report_tail_raw.py','evaluate_tail_delivery.py','verify_tail_delivery.py','report_tail_delivery.py','compare_original_coverage.py','verify_original_coverage.py','report_coverage_raw.py','evaluate_coverage_delivery.py','verify_coverage_delivery.py','report_coverage_delivery.py']


def save(phase,status,code=None):
    STATUS.write_text(json.dumps(dict(phase=phase,status=status,exit_code=code,updated_utc=datetime.now(timezone.utc).isoformat(),
        delivered_audio_qualification=False,production_eligible=False),indent=2)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--from-phase',choices=PHASES,default=PHASES[0]);args=parser.parse_args()
    source=WS/'results/transformer_usability/2026-11-04_v64_reliable_melodic_pairs/summary.json'
    assert source.exists(),'Physical construction has not completed'
    for phase in PHASES[PHASES.index(args.from_phase):]:
        save(phase,'running');print('PRESERVATION_PIPELINE_PHASE',phase,flush=True)
        code=subprocess.call([sys.executable,'-X','utf8','-B',str(WS/'tools'/phase)],cwd=WS)
        if code:
            save(phase,'failed',code);raise SystemExit(code)
        save(phase,'complete',0)
    save('declared_preservation_pipeline','complete',0)


if __name__=='__main__':main()
