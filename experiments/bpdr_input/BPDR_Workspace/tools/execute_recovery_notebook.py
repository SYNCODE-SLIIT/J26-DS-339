"""Execute the recorded recovery walkthrough; never fit or regenerate evidence."""
import asyncio
import json
import os
from pathlib import Path
import sys
import time

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

WORKSPACE = Path(__file__).resolve().parents[1]
COMPONENT = WORKSPACE.parent
sys.path.insert(0, str(WORKSPACE / "source"))
from bpdr_input.audio.pipeline import checksum


def main():
    if sys.platform == "win32" and hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    pointer_path = WORKSPACE / "results/model_recovery/current.json"
    if not pointer_path.is_file():
        raise RuntimeError("Wait for the declared recovery experiment to finish; this runner performs no training.")
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    folder = Path(pointer["path"]).resolve()
    if not folder.is_relative_to((WORKSPACE / "results/model_recovery").resolve()):
        raise ValueError("Recovery completion pointer leaves the component's results directory")
    evidence_path = folder / "evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if evidence.get("outer_proposal_gate_passes") and not (folder / "rendered_verification.json").is_file():
        raise RuntimeError("Actual waveform verification is still pending; do not present proposal metrics as delivered recovery.")
    evidence_before = checksum(evidence_path)
    runtime = WORKSPACE / "results/recovery_notebook_runtime"
    kernel = runtime / "kernels/bpdr-local"
    kernel.mkdir(parents=True, exist_ok=True)
    temporary = runtime / "temporary"
    temporary.mkdir(parents=True, exist_ok=True)
    os.environ["TEMP"] = os.environ["TMP"] = str(temporary)
    (kernel / "kernel.json").write_text(json.dumps(dict(
        argv=[sys.executable, "-P", "-B", "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        display_name="BPDR local Python", language="python",
        env=dict(PYTHONPATH=str(WORKSPACE / "source"), PYTHONDONTWRITEBYTECODE="1",
                 BPDR_COMPONENT_ROOT=str(COMPONENT), TEMP=str(temporary), TMP=str(temporary)))), encoding="utf-8")
    for key, directory in [("JUPYTER_PATH", runtime), ("JUPYTER_RUNTIME_DIR", runtime / "runtime"),
                           ("JUPYTER_CONFIG_DIR", runtime / "config"), ("IPYTHONDIR", runtime / "ipython")]:
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(directory)
    path = WORKSPACE / "notebooks/07_learned_stable_note_recovery.ipynb"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    started = time.perf_counter()
    client = NotebookClient(notebook, kernel_name="bpdr-local", timeout=180, startup_timeout=40,
                            resources={"metadata": {"path": str(path.parent)}})
    client.on_cell_start = lambda cell, cell_index: print("Executing cell", cell_index, flush=True)
    try:
        client.execute()
        if checksum(evidence_path) != evidence_before:
            raise RuntimeError("Recovery evidence changed while the notebook was running; execute again after it is finalized.")
    except Exception:
        nbformat.write(notebook, WORKSPACE / "results/07_recovery_failed.ipynb")
        raise
    outputs = [output for cell in notebook.cells for output in cell.get("outputs", [])]
    plots = sum("image/png" in output.get("data", {}) for output in outputs)
    audio = sum("<audio" in output.get("data", {}).get("text/html", "") for output in outputs)
    errors = sum(output.get("output_type") == "error" for output in outputs)
    required = notebook.metadata.get("bpdr_verification", {})
    if errors or plots < required.get("minimum_plots", 0) or audio < required.get("minimum_audio", 0):
        raise RuntimeError(f"Incomplete notebook outputs: errors={errors}, plots={plots}, audio={audio}")
    nbformat.write(notebook, path)
    html, _ = HTMLExporter().from_notebook_node(notebook)
    html_path = path.with_suffix(".html")
    html_path.write_text(html, encoding="utf-8")
    receipt = dict(notebook=str(path), notebook_sha256=checksum(path), html_export=str(html_path),
                   html_sha256=checksum(html_path), recovery_evidence=str(evidence_path),
                   recovery_evidence_sha256=evidence_before,
                   code_cells=sum(cell.cell_type == "code" for cell in notebook.cells), errors=errors,
                   plot_outputs=plots, audio_players=audio,
                   elapsed_seconds=round(time.perf_counter() - started, 3),
                   scope="Fresh kernel; recorded source, model, proposal and waveform evidence verification; no training or intervention generation")
    (WORKSPACE / "results/recovery_notebook_execution.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
