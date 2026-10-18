# Method, output and validation

## Problem and research objective

Short solo singing/humming contains uncertain note boundaries and locally changing pitch. BPDR asks whether a correction model can respond to a local pitch perturbation while remaining insensitive to boundary metadata changes, preserving the rest of the voice. This is distinct from forcing every note to a musical grid or inferring all intended notes from unrestricted humming.

## Executable workflow

1. Inspect original Vocadito metadata, audio/annotation presence and recording durations. Keep source hashes and distinguish performed references from ideal musical intention.
2. Use artist/singer-group partitions before cropping/augmentation. Jamendo annotations are model estimates; artist IDs are proxies.
3. Construct A/B/C/D views: original or locally detuned audio crossed with nominal or altered boundaries.
4. Apply a four-layer Transformer to 70 acoustic features plus boundary metadata. Compare BPDR with augmentation-only and frame-only conditions on matched schedules.
5. Validate live audio, derive features/boundaries, infer correction, apply permission/render/remeasurement/rollback checks. Optional noise cleaning requires a declared noise-only interval and re-analysis.
6. Compare input with a verified separate candidate. Retain the original accepted package and report unsupported/withheld outcomes.

## Metric meanings and recorded examples

| Measure | Interpretation | Current example |
| --- | --- | --- |
| Local pitch error in cents | Difference from a known original on the injected local region; not a whole-user accuracy percentage | Generated-reference input 19.80 -> output 1.48 cents |
| Paired response gain | How much the prediction changes opposite to the known added detuning; ideal 1, constant-zero 0 | Full condition/seed/dataset tables in notebook 2 |
| Original/outside alteration | Preservation of unshifted regions; lower is preferable | Reported alongside correction results, not dropped when a fit improves |
| Added-noise SNR gain | Reduction of error against the recording before known added noise | Real-HumTrans/rain saved runtime output 2.46 dB |
| Noise-only prefix attenuation | Change within the declared noise interval | 2.96 dB |
| Measured pitch change during cleaning | Numerical preservation diagnostic; not intended-note or listening approval | Median 0.08 cents |

No universal classification-style model-accuracy percentage has been established. A one-example error reduction is not a held-out participant success rate. These numbers preserve the source identities: generated pitch-control example and a constructed mixture of real recordings.

## Design and best practices

Source retention; accepted-only handoff; explicit technical states; fixed engineering limits; masks for unreliable/unvoiced frames; training-only normalization; grouped partitions; independent input/reference roles; verified checkpoints; feature re-analysis; rollback; reproducible logs; retained failed comparisons; candidate checksums before playback; and equal-gain notebook playback.

## User requirements and achievable benefits

Record/upload a short solo voice without knowing musical notation. Receive clear feedback, measured features and optional bounded processing previews. Compare/download the original and candidate. A novice should be able to submit input; whether the repair reliably improves novice recordings still requires evaluation. The prototype may support later melody/harmony extraction, but a commercial benefit or transcription gain is not yet measured.

## Completion and risk

The prototype, preparation paths, controlled learned response, optional cleaning and recorded validation are executable. Reliable general real-voice repair, a consistent BPDR advantage, independent musical review, novice/listener studies and downstream benefit remain open. The newer real-input and humming models failed their checks; the selected mixed-data checkpoint remains unchanged. Do not unlock a larger fit or force edits merely to make an output look different.

The supplied assessment sheet values problem definition/proof of concept (10%), relevant knowledge/technologies (25%), implementation including design/completion/standards/requirements/risk (40%), communication (15%) and achievable business potential (10%). The two notebooks and live comparisons provide a clear implementation walkthrough; this mapping does not predict marks or assert that every research objective is validated.
