"""Explicit JSON-representation amendment; frozen diagnostic/auditor arithmetic unchanged."""
import argparse
import json
from pathlib import Path
from datetime import datetime,timezone
import diagnose_render_disagreement as diagnostic


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['diagnose','verify']);args=parser.parse_args()
    d=diagnostic.declare();ex=diagnostic.study.ex;root=diagnostic.ROOT
    path=root/'serialization_amendment.json'
    if not path.exists():
        assert args.stage=='diagnose' and not (root/'results.json').exists()
        ex.write(path,dict(created_at=datetime.now(timezone.utc).isoformat(),original_declaration_sha256=ex.sha(root/'declaration.json'),wrapper_source_sha256=ex.sha(Path(__file__)),original_failure=dict(session=96590,exit_code=1,case_index=9,arm='reference',stage='audit_object_equality',cause='AuditPolicy shifts_cents tuple in memory differs from JSON-decoded list; printed numerical audit values match. No completed candidate receipt or result file.'),amendment='JSON round-trip the audit dictionary only before comparing/persisting, matching existing JSON representation. No change to values, numerical precision, masks, curves, waveform, thresholds or passed/reasons calculation.',model_retrained=False,selection_eligible=False))
    amendment=ex.read(path)
    assert amendment['wrapper_source_sha256']==ex.sha(Path(__file__))
    assert amendment['original_declaration_sha256']==ex.sha(root/'declaration.json')
    if args.stage=='diagnose':module=diagnostic
    else:
        import verify_render_disagreement as verification
        module=verification
    original=module.audit_curve
    module.audit_curve=lambda *a,**kw:json.loads(json.dumps(original(*a,**kw),allow_nan=False))
    try:module.main()
    finally:module.audit_curve=original
    out=root/('results.json' if args.stage=='diagnose' else 'integrity_receipt.json')
    receipt=ex.read(out);receipt['serialization_amendment_sha256']=ex.sha(path);ex.write(out,receipt)

if __name__=='__main__':main()
