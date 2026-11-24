"""Run the declared model-first improvement experiment or evaluate a checkpoint."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from bpdr_input.experiments.learning.context_improvement import ROOT, main, read
from bpdr_input.reporting.transformer_accuracy_report import complete_report

if __name__=='__main__':
    main()
    if len(sys.argv)>1 and sys.argv[1]=='train':
        run=Path(read(ROOT/'latest_experiment.json')['path'])
        complete_report(run)
        for folder in (run/'calibration').glob('seed_*'):
            complete_report(folder)
