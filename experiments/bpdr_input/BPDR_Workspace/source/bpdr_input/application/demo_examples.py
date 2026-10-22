"""Small explicit example selection; real recordings precede generated controls."""
import json
from pathlib import Path

from bpdr_input import COMPONENT_ROOT
from bpdr_input.application.saved_results import noise_example, pitch_example


def examples():
    noise, pitch = noise_example(), pitch_example()
    original = COMPONENT_ROOT / 'BPDR_Workspace/datasets/humtrans/subset/train/F01_0001_0001_1_D/original.wav'
    shifted = COMPONENT_ROOT / 'runs/humming_experiment/b30e0327c70b4dbf90dd917219dcd8d3/M03_0056_0001_1_DD/shift_+20/C.wav'
    records = [
        dict(id='real_original', label='Real humming: original recording', input=original,
             reference=None, source_kind='Original HumTrans recording. No injected pitch error and no ideal intended-note reference.', method='Conservative stable-note baseline'),
        dict(id='noise', label='Real humming: added rain', input=noise['input'], reference=noise['reference'],
             source_kind='Real HumTrans humming with known added recorded rain and a known 0.7-second noise-only prefix.', method='Noise reduction only'),
        dict(id='real_shift', label='Real humming: controlled pitch shift', input=shifted, reference=shifted.parent.parent / 'A.wav',
             source_kind='Real HumTrans development recording with a known injected +20-cent shift. This is not a natural-error or population-accuracy example.', method='BPDR Transformer'),
        dict(id='pitch', label='Generated pitch control', input=pitch['input'], reference=pitch['reference'],
             source_kind='Generated held-out phrase with known local detuning. Kept as a secondary engineering control.', method='BPDR Transformer'),
        dict(id='silence', label='Silence: rerecording check', input=None, reference=None,
             source_kind='Constructed silence used only to check failure handling.', method='Conservative stable-note baseline'),
    ]
    from bpdr_input.repair.learned_runtime import METHOD_NAME, available, load_model
    if available():
        _, _, card = load_model()
        recovered = []
        for item in card.get('demo_cases', []):
            source = Path(item['input']).resolve()
            if not source.is_relative_to((COMPONENT_ROOT / 'BPDR_Workspace').resolve()):
                raise ValueError('Learned demo source leaves BPDR_Workspace.')
            recovered.append(dict(item, input=source,
                                  reference=Path(item['reference']) if item.get('reference') else None,
                                  method=METHOD_NAME))
        records = recovered + records
    # A failed/missing optional dataset never hides the working upload flow.
    return [record for record in records if record['input'] is None or Path(record['input']).exists()]
