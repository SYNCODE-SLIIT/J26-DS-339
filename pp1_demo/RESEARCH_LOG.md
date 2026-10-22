# PP1 application decisions and execution log

## 2026-10-22 — M0: integration setup

- Use a separate `pp1_demo/` application and existing project uv environment. Add only missing UI packages; preserve root dependency manifests and teammates' source.
- User authorized implementation, evaluation and incremental local commits. No push requested.
- Use Pasan's temporal weighted best checkpoint, not the function3 checkpoint. Decode its root/family heads and retain original architecture.
- Rahul receives mode `major` with `demo_constant` provenance on every run.
- BPDR reproduction depends on missing local manifests, historical model and audited pairs. Execute the notebook setup to record the real failure. Never manufacture training evidence or silently replace its model.
- Store uploaded recordings, executed notebooks and generated media only under ignored `runtime/`.
