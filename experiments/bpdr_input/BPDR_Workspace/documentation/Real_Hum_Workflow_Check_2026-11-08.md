# Real-hum workflow: local engineering check

8 November 2026. One already opened training-role HumTrans phrase,
F01_0001_0001_1_D (8 seconds), its previously constructed rain derivative
(8.8 seconds), and synthetic silence. No new participants, guide input,
musical judgments or fresh confirmation data.

## Result

The new review UI completes four declared routes. A fresh process reproduces
their consequential results exactly: eight route executions and six real
frozen-pYIN consumer calls across the two processes. Session 21082 completed
both phases, exit 0.

| Route | Result | Selected waveform reaches consumer? |
|---|---|---|
| WAV upload of the real training hum | Format-only ACCEPT_UNCHANGED; no pitch edit; 8-second timeline retained | Exact selected/consumer hash |
| Simulated recording-widget bytes from the same hum | Same unchanged working waveform | Exact selected/consumer hash |
| Controlled rain derivative | Spectral preview passes existing preservation checks; protection veto and undo work; ACCEPT_PREPARED fixture exported | Prepared hash differs from original and matches consumer |
| Constructed silence | RERECORD_REQUIRED, useful rerecord feedback, accepted_audio=None | No export or consumer control available |

The original 44.1 kHz PCM16 input is preserved and converted to an 8-second
22.05 kHz PCM24 working copy. No score, key, guide melody or guide MIDI is
provided to processing. The recording-widget route injects existing WAV bytes
through the widget boundary; no microphone, browser permission or live capture
is tested.

The automated test replaces confirmation metadata with
`confirmation_kind='automated_test_fixture'` and `user_confirmed=False`.
It explicitly enables fixture consumption only inside the test harness.
The ordinary UI still requires user confirmation. No generated artifact
claims that a person listened or approved the musical result.

## Controlled rain measurements

The fixed conservative spectral preview gives **2.45525 dB paired
clean-reference SNR improvement** on this 8.8-second digital mixture.
The clean reference is used for post-processing measurement; it does not
enter the processing operator. The operator uses the already declared
voice-free noise prefix and its existing screening/preservation policy.

The encoded-output preservation audit passes on 295 reliable frames:

- Median measured pitch change: 0.08422 cents.
- 95th-percentile pitch change: 0.60326 cents.
- Voiced-frame loss fraction: 0.00339.
- 95th-percentile voice attenuation: 0.48026 dB.
- Noise-reference attenuation: 2.95876 dB.

These are numerical results on one fixed development example. They do not
establish perceived quality, intended-note preservation, note-transcription
accuracy or general cleanup qualification. The separate 189-case neural
comparison remains a negative qualification result for always-on neural
cleanup; this spectral example does not replace it.

## Evidence and reproduction

Tool: `tools/check_real_hum_workflow.py`.
Declaration, current-source snapshots, results and replay receipt:
`results/demo_sessions/2026-11-08_real_hum_workflow/`.
The receipt matches the result hash. All three exported fixture ZIPs and
consumer artifacts were independently checked against their stored hashes;
fixture confirmation labels are intact.

Use the component Python with `PYTHONPATH=source;tools` and run the tool with
`--mode verify` to reproduce the declared routes. Streamlit AppTest needs a
working Windows asyncio loopback socket; sandboxed execution can block before
app startup. This is local testing and sends no recordings to another service.

The execution records an Intel64 Family 6 Model 170 Stepping 4 processor,
18 logical CPUs and two PyTorch threads. Training session 36082 was concurrent.
Stored times include UI automation and child startup; they are not isolated
cold/warm deployment latency benchmarks.

## What remains unproven

There is one corpus phrase and its derivative, not multiple independent users.
The recorded-widget test is not live microphone validation. Human listening,
performer intention feedback, naturally co-recorded noise, unfamiliar-user
confirmation, qualified short-hum pitch repair and complete shared PP1
orchestration remain open. This check advances engineering evidence without
claiming the full component or its mandatory research function is complete.
