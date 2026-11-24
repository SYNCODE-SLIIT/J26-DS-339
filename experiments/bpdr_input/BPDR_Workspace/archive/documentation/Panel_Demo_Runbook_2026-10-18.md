# BPDR panel demonstration

**Current priority: working demo.** Run `Run-Demo.ps1` for the short input/process/output interface. It supports the existing BPDR Transformer, a conservative stable-note baseline and declared-reference noise reduction. Start with **Known pitch error**, then **Real humming with added rain**. Upload and microphone input are available. The longer study interface is still accessible through `Run-System.ps1`; the evidence route below describes that interface. Video and additional presentation work are deferred.

This package demonstrates existing implemented behavior and saved measurements. It does not claim completed natural-user validation, reliable real-singer pitch repair, a BPDR advantage, or a universal decision about musical correctness.

## Open and rehearse

From `BPDR_Workspace`, run:

```powershell
.\Run-Panel-Demo.ps1 -Check
.\Run-Panel-Demo.ps1
```

Open the displayed localhost address, choose **Known pitch error**, and press **Process recording**. Repeat with **Real humming with added rain**, which selects noise reduction. Keep playback volume unchanged between inputs and outputs. The small pitch difference is better explained with the measured curve than exaggerated as an obvious listening effect. The first **Panel demo** tab in the longer `Run-System.ps1` interface provides saved-case replay.

`-Offline` opens `results/panel_demo/index.html`, with packaged audio and figures and no model loading. This is a saved evidence replay, not a live inference claim. `-Build -Check` rebuilds and verifies the package from existing evidence, without fitting models. `-Rehearse -Check` reruns the existing pitch, noise, no-change and silence paths, retaining every outcome under `results/panel_demo/rehearsal`.

Open `notebooks/05_panel_demonstration.ipynb` using `Run-Notebooks.ps1`. Notebook 03 contains the training explanation, notebook 04 the broader evaluation, and notebook 02 the full historical study. The short notebook is the panel route; the historical failures remain available.

## Two-minute individual route

The supplied template requires an individual opening and two content slides. Use optional appendix slides only for questions.

| Time | Action and explanation |
| --- | --- |
| 0:00-0:15 | State the input problem: novice singing can contain recoverable melody, recording problems and uncertain boundaries. The component preserves usable material and proposes limited preparation. |
| 0:15-0:35 | Explain the research gap. A/B use identical audio with different boundaries; C/D use identical detuned audio with different boundaries. Boundary stability must coexist with a corrective acoustic response. |
| 0:35-1:05 | Function 1: play the known detuned generated phrase and the verified pitch candidate. Point to the measured local error reduction, source label and rendering audit. This is a controlled generated example. |
| 1:05-1:30 | Function 2: play real HumTrans humming mixed with recorded rain and its cleaned preview. Point to paired SNR gain and pitch preservation. The noise-only prefix is known for this fixture. |
| 1:30-1:45 | Show no-change preservation and the silence rerecording result. Rejection of an edit does not establish that a singer is wrong. |
| 1:45-2:00 | State the limits: real-pitch learning attempts failed their gates; prepared handoff remains off; natural novice-user evaluation and melody-extraction benefit remain open. |

Live inference takes additional time. Rehearse it before the panel and use the saved comparison during the two-minute slot unless the panel permits a longer demonstration. Label saved and live execution accurately.

## Research interpretation

The Transformer outputs a correction in cents, not a correct/wrong singing label. Four-view training teaches a known compensating response and robustness to boundary metadata. It does not recover arbitrary intention from ordinary speech. The component still needs a validated melodic-usability gate. Chant-like or speech-like singing must not be rejected solely by its delivery style. No user key entry is required, and no unfamiliar note is automatically forced into a scale.

The synthetic, real-recording controlled-mixture, real-data learning-failure and technical-fixture evidence must remain distinct. A valid source pitch annotation is not an ideal intended-note label. An engineering renderer pass is not independent musical approval. Do not claim improved extraction from an unchanged frozen-extractor consistency result.

## Handoff and protection

The two successful output previews remain separate candidates. The current accepted downstream waveform is the original working copy. A failed candidate cannot replace it. Current statuses are `ACCEPT_UNCHANGED`, `ACCEPT_PREPARED` (defined contract, not enabled by this package), `RERECORD_REQUIRED`, and `TECHNICAL_FAILURE`. Only accepted packages contain an accepted path.

The fixed GAME check included in the evidence is a bounded saved extractor check. Full integration with the team's actual next component and improvements against independent note references are separate requirements.

## Assessment preparation still requiring human evidence

The supplied rubric allocates 10% to the gap/solution, 25% to specialized knowledge, 40% to implementation, 15% to communication and 10% to commercialization. The working package supports explanation and implementation evidence; it cannot supply missing human records.

| Item | What to provide before presentation |
| --- | --- |
| Personal photo | Replace the marked photo slot in the individual title slide. |
| Supervisor/user feedback | Add dated actual comments, resulting changes and unresolved points. No feedback is invented. |
| Diary and Planner | Bring the actual individual diary and project planning records. |
| Repository contribution | Show relevant commits and code, distinguishing prior work from current uncommitted changes. |
| Team integration | Demonstrate the actual accepted-audio consumer with its owner. This package alone is not full-team integration proof. |
| Team video | Prepare the required 1-2 minute overall problem/solution video with all components represented accurately. The offline evidence page is a fallback for this component, not that team video. |
| Commercialization | Complete the team-level market/persona, uniqueness and cost-recovery section using agreed facts. |
| AI-use log | Record AI assistance with code, evidence assembly, notebook and slides; retain student verification and responsibility. |

## Current work and earlier evidence

This pass reuses the selected mixed-data checkpoint and existing datasets. It performs no optimizer updates, new separation, dataset expansion or policy adoption. The fresh rehearsal receipt records elapsed times and preserved hashes. Input/output differences are numerical evidence for the named cases, not population accuracy.

Relevant existing component commits observed during this pass include `7968e9c` (Transformer and controlled study), `ec0f64a` (PP1 validation/rule demo), and `1e7eb44` (Gate E diagnosis). Inspect the live repository history before presenting ownership claims. Current package additions are uncommitted unless the owner commits them later.
