# R1 End-to-End & Architectural Statistical Analysis Report
**Source Data**: `e2e_fit_results.json` (FINAL complete results) + `transformer_v2_results.json` + `transformer_v3_results.json` + `transformer_v4_results.json` + `studio_vocadito_benchmark.json`
**Methodology**: Singer-macro (14 Vocadito singers), 95% bootstrap confidence intervals (1,000 resamples, seed 339); `dev_M02` uses clip-level (64 clips) 95% bootstrap CIs.
**Fit Health**: 18 fits evaluated, 100.0% healthy.

### Part 1: Vocadito Note F1 & Onset F1 (Model vs Extractor)

**Declared bar target: TCN full-feature `bpdr` arm on clean Note F1 (also reporting FastTNet Transformer full-feature vs Extractor across all arms and conditions).**

| Arch | Metric | Arm | Condition | Model (Mean [95% CI]) | Extractor (Mean [95% CI]) | Diff (Model - Ext) [95% CI] | Win (CI > 0)? |
|---|---|---|---|---|---|---|---|
| `tcn` | Note F1 | `frame_only` | clean | 0.522 [0.449, 0.595] | 0.562 [0.489, 0.635] | **-0.040** [-0.067, -0.011] | NO |
| `tcn` | Note F1 | `frame_only` | split | 0.522 [0.449, 0.595] | 0.559 [0.488, 0.629] | **-0.037** [-0.063, -0.009] | NO |
| `tcn` | Note F1 | `frame_only` | merge | 0.522 [0.449, 0.595] | 0.554 [0.486, 0.617] | **-0.032** [-0.055, -0.005] | NO |
| `tcn` | Note F1 | `frame_only` | jitter | 0.522 [0.449, 0.595] | 0.479 [0.406, 0.552] | **+0.043** [+0.015, +0.073] | YES |
| `tcn` | Note F1 | `augmentation_only` | clean | 0.548 [0.484, 0.614] | 0.562 [0.489, 0.635] | **-0.014** [-0.044, +0.016] | NO |
| `tcn` | Note F1 | `augmentation_only` | split | 0.547 [0.484, 0.614] | 0.559 [0.488, 0.629] | **-0.011** [-0.039, +0.017] | NO |
| `tcn` | Note F1 | `augmentation_only` | merge | 0.546 [0.478, 0.616] | 0.554 [0.486, 0.617] | **-0.008** [-0.033, +0.019] | NO |
| `tcn` | Note F1 | `augmentation_only` | jitter | 0.537 [0.460, 0.611] | 0.479 [0.406, 0.552] | **+0.057** [+0.024, +0.094] | YES |
| `tcn` | Note F1 | `bpdr` | clean | 0.535 [0.460, 0.609] | 0.562 [0.489, 0.635] | **-0.027** [-0.057, +0.005] | NO |
| `tcn` | Note F1 | `bpdr` | split | 0.537 [0.464, 0.609] | 0.559 [0.488, 0.629] | **-0.022** [-0.050, +0.008] | NO |
| `tcn` | Note F1 | `bpdr` | merge | 0.535 [0.460, 0.609] | 0.554 [0.486, 0.617] | **-0.019** [-0.042, +0.008] | NO |
| `tcn` | Note F1 | `bpdr` | jitter | 0.529 [0.456, 0.604] | 0.479 [0.406, 0.552] | **+0.050** [+0.018, +0.084] | YES |
| `tcn` | Onset F1 | `frame_only` | clean | 0.680 [0.607, 0.755] | 0.601 [0.556, 0.643] | **+0.079** [+0.027, +0.138] | YES |
| `tcn` | Onset F1 | `frame_only` | split | 0.680 [0.607, 0.755] | 0.596 [0.555, 0.635] | **+0.084** [+0.033, +0.142] | YES |
| `tcn` | Onset F1 | `frame_only` | merge | 0.680 [0.607, 0.755] | 0.594 [0.551, 0.635] | **+0.086** [+0.028, +0.150] | YES |
| `tcn` | Onset F1 | `frame_only` | jitter | 0.680 [0.607, 0.755] | 0.538 [0.504, 0.570] | **+0.142** [+0.081, +0.203] | YES |
| `tcn` | Onset F1 | `augmentation_only` | clean | 0.698 [0.626, 0.764] | 0.601 [0.556, 0.643] | **+0.097** [+0.052, +0.145] | YES |
| `tcn` | Onset F1 | `augmentation_only` | split | 0.697 [0.625, 0.762] | 0.596 [0.555, 0.635] | **+0.101** [+0.057, +0.147] | YES |
| `tcn` | Onset F1 | `augmentation_only` | merge | 0.697 [0.625, 0.763] | 0.594 [0.551, 0.635] | **+0.103** [+0.052, +0.156] | YES |
| `tcn` | Onset F1 | `augmentation_only` | jitter | 0.679 [0.604, 0.751] | 0.538 [0.504, 0.570] | **+0.140** [+0.082, +0.200] | YES |
| `tcn` | Onset F1 | `bpdr` | clean | 0.688 [0.614, 0.757] | 0.601 [0.556, 0.643] | **+0.087** [+0.038, +0.140] | YES |
| `tcn` | Onset F1 | `bpdr` | split | 0.690 [0.615, 0.758] | 0.596 [0.555, 0.635] | **+0.094** [+0.047, +0.145] | YES |
| `tcn` | Onset F1 | `bpdr` | merge | 0.688 [0.614, 0.757] | 0.594 [0.551, 0.635] | **+0.094** [+0.041, +0.151] | YES |
| `tcn` | Onset F1 | `bpdr` | jitter | 0.685 [0.609, 0.756] | 0.538 [0.504, 0.570] | **+0.147** [+0.089, +0.206] | YES |
| `transformer` | Note F1 | `frame_only` | clean | 0.601 [0.538, 0.665] | 0.562 [0.489, 0.635] | **+0.039** [+0.006, +0.072] | YES |
| `transformer` | Note F1 | `frame_only` | split | 0.601 [0.538, 0.665] | 0.559 [0.488, 0.629] | **+0.042** [+0.011, +0.073] | YES |
| `transformer` | Note F1 | `frame_only` | merge | 0.601 [0.538, 0.665] | 0.554 [0.486, 0.617] | **+0.047** [+0.019, +0.078] | YES |
| `transformer` | Note F1 | `frame_only` | jitter | 0.601 [0.538, 0.665] | 0.479 [0.406, 0.552] | **+0.122** [+0.083, +0.158] | YES |
| `transformer` | Note F1 | `augmentation_only` | clean | 0.568 [0.498, 0.635] | 0.562 [0.489, 0.635] | **+0.006** [-0.018, +0.028] | NO |
| `transformer` | Note F1 | `augmentation_only` | split | 0.568 [0.499, 0.636] | 0.559 [0.488, 0.629] | **+0.010** [-0.012, +0.031] | NO |
| `transformer` | Note F1 | `augmentation_only` | merge | 0.566 [0.498, 0.632] | 0.554 [0.486, 0.617] | **+0.012** [-0.006, +0.033] | NO |
| `transformer` | Note F1 | `augmentation_only` | jitter | 0.568 [0.500, 0.634] | 0.479 [0.406, 0.552] | **+0.089** [+0.057, +0.120] | YES |
| `transformer` | Note F1 | `bpdr` | clean | 0.577 [0.510, 0.639] | 0.562 [0.489, 0.635] | **+0.015** [-0.007, +0.034] | NO |
| `transformer` | Note F1 | `bpdr` | split | 0.580 [0.514, 0.642] | 0.559 [0.488, 0.629] | **+0.021** [-0.001, +0.039] | NO |
| `transformer` | Note F1 | `bpdr` | merge | 0.576 [0.510, 0.638] | 0.554 [0.486, 0.617] | **+0.022** [+0.005, +0.039] | YES |
| `transformer` | Note F1 | `bpdr` | jitter | 0.582 [0.517, 0.643] | 0.479 [0.406, 0.552] | **+0.103** [+0.071, +0.135] | YES |
| `transformer` | Onset F1 | `frame_only` | clean | 0.719 [0.646, 0.784] | 0.601 [0.556, 0.643] | **+0.118** [+0.067, +0.170] | YES |
| `transformer` | Onset F1 | `frame_only` | split | 0.719 [0.646, 0.784] | 0.596 [0.555, 0.635] | **+0.123** [+0.072, +0.174] | YES |
| `transformer` | Onset F1 | `frame_only` | merge | 0.719 [0.646, 0.784] | 0.594 [0.551, 0.635] | **+0.125** [+0.070, +0.181] | YES |
| `transformer` | Onset F1 | `frame_only` | jitter | 0.719 [0.646, 0.784] | 0.538 [0.504, 0.570] | **+0.180** [+0.117, +0.240] | YES |
| `transformer` | Onset F1 | `augmentation_only` | clean | 0.712 [0.637, 0.780] | 0.601 [0.556, 0.643] | **+0.111** [+0.061, +0.163] | YES |
| `transformer` | Onset F1 | `augmentation_only` | split | 0.711 [0.636, 0.779] | 0.596 [0.555, 0.635] | **+0.115** [+0.066, +0.166] | YES |
| `transformer` | Onset F1 | `augmentation_only` | merge | 0.709 [0.634, 0.776] | 0.594 [0.551, 0.635] | **+0.115** [+0.061, +0.170] | YES |
| `transformer` | Onset F1 | `augmentation_only` | jitter | 0.710 [0.635, 0.775] | 0.538 [0.504, 0.570] | **+0.172** [+0.111, +0.229] | YES |
| `transformer` | Onset F1 | `bpdr` | clean | 0.715 [0.639, 0.781] | 0.601 [0.556, 0.643] | **+0.114** [+0.063, +0.165] | YES |
| `transformer` | Onset F1 | `bpdr` | split | 0.715 [0.640, 0.782] | 0.596 [0.555, 0.635] | **+0.119** [+0.068, +0.169] | YES |
| `transformer` | Onset F1 | `bpdr` | merge | 0.714 [0.638, 0.780] | 0.594 [0.551, 0.635] | **+0.120** [+0.067, +0.176] | YES |
| `transformer` | Onset F1 | `bpdr` | jitter | 0.715 [0.642, 0.780] | 0.538 [0.504, 0.570] | **+0.177** [+0.113, +0.238] | YES |

**Corruption Drop Analysis (Clean - Mean(Corruptions)):**

| Arch | Metric | Arm | Clean | Split | Merge | Jitter | Mean Drop |
|---|---|---|---|---|---|---|---|
| `tcn` | Note F1 | `frame_only` | 0.522 | 0.522 | 0.522 | 0.522 | +0.000 |
| `tcn` | Note F1 | `augmentation_only` | 0.548 | 0.547 | 0.546 | 0.537 | +0.005 |
| `tcn` | Note F1 | `bpdr` | 0.535 | 0.537 | 0.535 | 0.529 | +0.002 |
| `tcn` | Onset F1 | `frame_only` | 0.680 | 0.680 | 0.680 | 0.680 | +0.000 |
| `tcn` | Onset F1 | `augmentation_only` | 0.698 | 0.697 | 0.697 | 0.679 | +0.007 |
| `tcn` | Onset F1 | `bpdr` | 0.688 | 0.690 | 0.688 | 0.685 | +0.001 |
| `transformer` | Note F1 | `frame_only` | 0.601 | 0.601 | 0.601 | 0.601 | +0.000 |
| `transformer` | Note F1 | `augmentation_only` | 0.568 | 0.568 | 0.566 | 0.568 | +0.000 |
| `transformer` | Note F1 | `bpdr` | 0.577 | 0.580 | 0.576 | 0.582 | -0.002 |
| `transformer` | Onset F1 | `frame_only` | 0.719 | 0.719 | 0.719 | 0.719 | +0.000 |
| `transformer` | Onset F1 | `augmentation_only` | 0.712 | 0.711 | 0.709 | 0.710 | +0.002 |
| `transformer` | Onset F1 | `bpdr` | 0.715 | 0.715 | 0.714 | 0.715 | -0.000 |
| `extractor` | Note F1 | `baseline` | 0.562 | 0.559 | 0.554 | 0.479 | +0.032 |
| `extractor` | Onset F1 | `baseline` | 0.601 | 0.596 | 0.594 | 0.538 | +0.025 |

### Part 2: Transformer vs TCN (Onset F1 & Note F1)

**Adoption criterion: Vocadito Onset F1 at 50 ms exceeds TCN for same arm with 95% CI excluding 0.**

| Metric | Feat | Arm | Cond | Transformer (Mean [CI]) | TCN (Mean [CI]) | Diff (Trans - TCN) [95% CI] | Adopt? |
|---|---|---|---|---|---|---|---|
| Onset F1 | `full` | `frame_only` | clean | 0.719 [0.646, 0.784] | 0.680 [0.607, 0.755] | **+0.039** [+0.012, +0.062] | ADOPT |
| Onset F1 | `full` | `frame_only` | split | 0.719 [0.646, 0.784] | 0.680 [0.607, 0.755] | **+0.039** [+0.012, +0.062] | ADOPT |
| Onset F1 | `full` | `frame_only` | merge | 0.719 [0.646, 0.784] | 0.680 [0.607, 0.755] | **+0.039** [+0.012, +0.062] | ADOPT |
| Onset F1 | `full` | `frame_only` | jitter | 0.719 [0.646, 0.784] | 0.680 [0.607, 0.755] | **+0.039** [+0.012, +0.062] | ADOPT |
| Onset F1 | `full` | `augmentation_only` | clean | 0.712 [0.637, 0.780] | 0.698 [0.626, 0.764] | **+0.014** [-0.008, +0.037] | DID NOT HELP |
| Onset F1 | `full` | `augmentation_only` | split | 0.711 [0.636, 0.779] | 0.697 [0.625, 0.762] | **+0.014** [-0.008, +0.035] | DID NOT HELP |
| Onset F1 | `full` | `augmentation_only` | merge | 0.709 [0.634, 0.776] | 0.697 [0.625, 0.763] | **+0.012** [-0.010, +0.034] | DID NOT HELP |
| Onset F1 | `full` | `augmentation_only` | jitter | 0.710 [0.635, 0.775] | 0.679 [0.604, 0.751] | **+0.031** [+0.004, +0.055] | ADOPT |
| Onset F1 | `full` | `bpdr` | clean | 0.715 [0.639, 0.781] | 0.688 [0.614, 0.757] | **+0.026** [+0.006, +0.047] | ADOPT |
| Onset F1 | `full` | `bpdr` | split | 0.715 [0.640, 0.782] | 0.690 [0.615, 0.758] | **+0.025** [+0.006, +0.045] | ADOPT |
| Onset F1 | `full` | `bpdr` | merge | 0.714 [0.638, 0.780] | 0.688 [0.614, 0.757] | **+0.026** [+0.007, +0.046] | ADOPT |
| Onset F1 | `full` | `bpdr` | jitter | 0.715 [0.642, 0.780] | 0.685 [0.609, 0.756] | **+0.030** [+0.006, +0.054] | ADOPT |
| Onset F1 | `lean` | `frame_only` | clean | 0.713 [0.648, 0.770] | 0.687 [0.616, 0.758] | **+0.026** [+0.001, +0.052] | ADOPT |
| Onset F1 | `lean` | `frame_only` | split | 0.713 [0.648, 0.770] | 0.687 [0.616, 0.758] | **+0.026** [+0.001, +0.052] | ADOPT |
| Onset F1 | `lean` | `frame_only` | merge | 0.713 [0.648, 0.770] | 0.687 [0.616, 0.758] | **+0.026** [+0.001, +0.052] | ADOPT |
| Onset F1 | `lean` | `frame_only` | jitter | 0.713 [0.648, 0.770] | 0.687 [0.616, 0.758] | **+0.026** [+0.001, +0.052] | ADOPT |
| Onset F1 | `lean` | `augmentation_only` | clean | 0.705 [0.626, 0.773] | 0.692 [0.621, 0.756] | **+0.013** [-0.013, +0.037] | DID NOT HELP |
| Onset F1 | `lean` | `augmentation_only` | split | 0.704 [0.626, 0.772] | 0.693 [0.622, 0.756] | **+0.011** [-0.014, +0.035] | DID NOT HELP |
| Onset F1 | `lean` | `augmentation_only` | merge | 0.707 [0.629, 0.775] | 0.691 [0.620, 0.755] | **+0.016** [-0.012, +0.040] | DID NOT HELP |
| Onset F1 | `lean` | `augmentation_only` | jitter | 0.708 [0.629, 0.777] | 0.686 [0.614, 0.753] | **+0.022** [-0.006, +0.044] | DID NOT HELP |
| Onset F1 | `lean` | `bpdr` | clean | 0.706 [0.625, 0.777] | 0.688 [0.617, 0.754] | **+0.018** [-0.007, +0.043] | DID NOT HELP |
| Onset F1 | `lean` | `bpdr` | split | 0.710 [0.632, 0.779] | 0.688 [0.619, 0.755] | **+0.022** [-0.002, +0.045] | DID NOT HELP |
| Onset F1 | `lean` | `bpdr` | merge | 0.706 [0.625, 0.778] | 0.688 [0.619, 0.755] | **+0.018** [-0.008, +0.042] | DID NOT HELP |
| Onset F1 | `lean` | `bpdr` | jitter | 0.705 [0.624, 0.776] | 0.684 [0.613, 0.753] | **+0.020** [-0.003, +0.042] | DID NOT HELP |
| Note F1 | `full` | `frame_only` | clean | 0.601 [0.538, 0.665] | 0.522 [0.449, 0.595] | **+0.079** [+0.043, +0.116] | ADOPT |
| Note F1 | `full` | `frame_only` | split | 0.601 [0.538, 0.665] | 0.522 [0.449, 0.595] | **+0.079** [+0.043, +0.116] | ADOPT |
| Note F1 | `full` | `frame_only` | merge | 0.601 [0.538, 0.665] | 0.522 [0.449, 0.595] | **+0.079** [+0.043, +0.116] | ADOPT |
| Note F1 | `full` | `frame_only` | jitter | 0.601 [0.538, 0.665] | 0.522 [0.449, 0.595] | **+0.079** [+0.043, +0.116] | ADOPT |
| Note F1 | `full` | `augmentation_only` | clean | 0.568 [0.498, 0.635] | 0.548 [0.484, 0.614] | **+0.019** [-0.009, +0.044] | DID NOT HELP |
| Note F1 | `full` | `augmentation_only` | split | 0.568 [0.499, 0.636] | 0.547 [0.484, 0.614] | **+0.021** [-0.006, +0.046] | DID NOT HELP |
| Note F1 | `full` | `augmentation_only` | merge | 0.566 [0.498, 0.632] | 0.546 [0.478, 0.616] | **+0.020** [-0.010, +0.046] | DID NOT HELP |
| Note F1 | `full` | `augmentation_only` | jitter | 0.568 [0.500, 0.634] | 0.537 [0.460, 0.611] | **+0.032** [-0.001, +0.064] | DID NOT HELP |
| Note F1 | `full` | `bpdr` | clean | 0.577 [0.510, 0.639] | 0.535 [0.460, 0.609] | **+0.042** [+0.008, +0.074] | ADOPT |
| Note F1 | `full` | `bpdr` | split | 0.580 [0.514, 0.642] | 0.537 [0.464, 0.609] | **+0.043** [+0.010, +0.073] | ADOPT |
| Note F1 | `full` | `bpdr` | merge | 0.576 [0.510, 0.638] | 0.535 [0.460, 0.609] | **+0.041** [+0.007, +0.071] | ADOPT |
| Note F1 | `full` | `bpdr` | jitter | 0.582 [0.517, 0.643] | 0.529 [0.456, 0.604] | **+0.053** [+0.021, +0.087] | ADOPT |
| Note F1 | `lean` | `frame_only` | clean | 0.573 [0.518, 0.625] | 0.538 [0.476, 0.605] | **+0.034** [+0.002, +0.065] | ADOPT |
| Note F1 | `lean` | `frame_only` | split | 0.573 [0.518, 0.625] | 0.538 [0.476, 0.605] | **+0.034** [+0.002, +0.065] | ADOPT |
| Note F1 | `lean` | `frame_only` | merge | 0.573 [0.518, 0.625] | 0.538 [0.476, 0.605] | **+0.034** [+0.002, +0.065] | ADOPT |
| Note F1 | `lean` | `frame_only` | jitter | 0.573 [0.518, 0.625] | 0.538 [0.476, 0.605] | **+0.034** [+0.002, +0.065] | ADOPT |
| Note F1 | `lean` | `augmentation_only` | clean | 0.565 [0.492, 0.632] | 0.546 [0.485, 0.606] | **+0.019** [-0.007, +0.041] | DID NOT HELP |
| Note F1 | `lean` | `augmentation_only` | split | 0.565 [0.492, 0.635] | 0.546 [0.484, 0.606] | **+0.019** [-0.006, +0.042] | DID NOT HELP |
| Note F1 | `lean` | `augmentation_only` | merge | 0.566 [0.493, 0.636] | 0.547 [0.485, 0.607] | **+0.020** [-0.005, +0.042] | DID NOT HELP |
| Note F1 | `lean` | `augmentation_only` | jitter | 0.564 [0.491, 0.628] | 0.543 [0.469, 0.614] | **+0.021** [-0.004, +0.044] | DID NOT HELP |
| Note F1 | `lean` | `bpdr` | clean | 0.570 [0.492, 0.642] | 0.537 [0.474, 0.598] | **+0.032** [+0.009, +0.059] | ADOPT |
| Note F1 | `lean` | `bpdr` | split | 0.571 [0.492, 0.643] | 0.538 [0.475, 0.598] | **+0.034** [+0.010, +0.060] | ADOPT |
| Note F1 | `lean` | `bpdr` | merge | 0.569 [0.492, 0.642] | 0.539 [0.476, 0.598] | **+0.031** [+0.007, +0.058] | ADOPT |
| Note F1 | `lean` | `bpdr` | jitter | 0.567 [0.489, 0.639] | 0.530 [0.463, 0.593] | **+0.036** [+0.017, +0.058] | ADOPT |

### Part 3: BPDR Loss vs Controls on Lean Features (Corruption Average)

**Bar: WIN if bpdr beats BOTH frame_only and augmentation_only on corruption-average onset F1 with 95% CI > 0.**

| Architecture | Arm | Split | Merge | Jitter | Corruption Mean (Mean [CI]) | Diff vs bpdr [95% CI] | Win (CI > 0)? |
|---|---|---|---|---|---|---|---|
| `tcn` | `bpdr` (target) | 0.688 | 0.688 | 0.684 | 0.687 [0.617, 0.754] | - | baseline |
| `tcn` | `frame_only` | 0.687 | 0.687 | 0.687 | 0.687 [0.616, 0.758] | **-0.000** [-0.015, +0.013] | NO |
| `tcn` | `augmentation_only` | 0.693 | 0.691 | 0.686 | 0.690 [0.619, 0.755] | **-0.003** [-0.011, +0.006] | NO |
| `transformer` | `bpdr` (target) | 0.710 | 0.706 | 0.705 | 0.707 [0.627, 0.778] | - | baseline |
| `transformer` | `frame_only` | 0.713 | 0.713 | 0.713 | 0.713 [0.648, 0.770] | **-0.006** [-0.033, +0.021] | NO |
| `transformer` | `augmentation_only` | 0.704 | 0.707 | 0.708 | 0.706 [0.627, 0.775] | **+0.000** [-0.010, +0.012] | NO |

### Development Split (`dev_M02`): Boundary Onset F1 (64 Clips)

**Note**: `dev_M02` is a single development singer (64 clips, alignment-derived labels); intervals below are clip-level 95% bootstrap CIs (1,000 resamples).

| Arch | Feat | Arm | Clean (Mean [95% CI]) | Split | Merge | Jitter | Diff vs Ext (Clean) [95% CI] |
|---|---|---|---|---|---|---|---|
| `tcn` | `full` | `frame_only` | 0.780 [0.752, 0.809] | 0.780 | 0.780 | 0.780 | **+0.232** [+0.207, +0.258] |
| `tcn` | `full` | `augmentation_only` | 0.781 [0.754, 0.807] | 0.781 | 0.781 | 0.784 | **+0.233** [+0.210, +0.258] |
| `tcn` | `full` | `bpdr` | 0.782 [0.754, 0.809] | 0.782 | 0.782 | 0.781 | **+0.234** [+0.210, +0.259] |
| `tcn` | `lean` | `frame_only` | 0.781 [0.752, 0.808] | 0.781 | 0.781 | 0.781 | **+0.233** [+0.209, +0.259] |
| `tcn` | `lean` | `augmentation_only` | 0.783 [0.756, 0.810] | 0.782 | 0.782 | 0.784 | **+0.235** [+0.212, +0.259] |
| `tcn` | `lean` | `bpdr` | 0.781 [0.754, 0.808] | 0.782 | 0.782 | 0.782 | **+0.234** [+0.210, +0.258] |
| `transformer` | `full` | `frame_only` | 0.754 [0.724, 0.782] | 0.754 | 0.754 | 0.754 | **+0.206** [+0.183, +0.231] |
| `transformer` | `full` | `augmentation_only` | 0.762 [0.734, 0.789] | 0.760 | 0.762 | 0.759 | **+0.214** [+0.193, +0.237] |
| `transformer` | `full` | `bpdr` | 0.767 [0.738, 0.796] | 0.767 | 0.767 | 0.767 | **+0.220** [+0.196, +0.243] |
| `transformer` | `lean` | `frame_only` | 0.741 [0.714, 0.767] | 0.741 | 0.741 | 0.741 | **+0.193** [+0.171, +0.217] |
| `transformer` | `lean` | `augmentation_only` | 0.765 [0.736, 0.793] | 0.764 | 0.766 | 0.764 | **+0.217** [+0.194, +0.241] |
| `transformer` | `lean` | `bpdr` | 0.763 [0.734, 0.790] | 0.763 | 0.763 | 0.764 | **+0.215** [+0.192, +0.238] |
| `extractor` | `-` | `baseline` | 0.548 [0.529, 0.564] | 0.537 | 0.541 | 0.526 | - |

### Fit Threshold & Seed Coverage Summary

| Arch | Feat | Arm | Seeds Run | Fits | Healthy | Thresholds (`thr`) |
|---|---|---|---|---|---|---|
| `tcn` | `full` | `augmentation_only` | `339, 340` | 2 | 2/2 | `0.8` |
| `tcn` | `full` | `bpdr` | `339, 340` | 2 | 2/2 | `0.8` |
| `tcn` | `full` | `frame_only` | `339, 340` | 2 | 2/2 | `0.8` |
| `tcn` | `lean` | `augmentation_only` | `339, 340` | 2 | 2/2 | `0.8` |
| `tcn` | `lean` | `bpdr` | `339, 340` | 2 | 2/2 | `0.8` |
| `tcn` | `lean` | `frame_only` | `339, 340` | 2 | 2/2 | `0.8` |
| `transformer` | `full` | `augmentation_only` | `339` | 1 | 1/1 | `0.8` |
| `transformer` | `full` | `bpdr` | `339` | 1 | 1/1 | `0.8` |
| `transformer` | `full` | `frame_only` | `339` | 1 | 1/1 | `0.8` |
| `transformer` | `lean` | `augmentation_only` | `339` | 1 | 1/1 | `0.8` |
| `transformer` | `lean` | `bpdr` | `339` | 1 | 1/1 | `0.8` |
| `transformer` | `lean` | `frame_only` | `339` | 1 | 1/1 | `0.8` |

- **Seed coverage**: `tcn`: 2 seed(s) ([339, 340]); `transformer`: 1 seed(s) ([339]).
- **Threshold saturation**: All 18/18 fits hit `thr = 0.8` (the upper ceiling of the candidate threshold grid), indicating uncalibrated/shifted eval logits.

## Transformer v2 Statistical Comparison (vs v1, TCN, and Extractor)

**Methodology**: Vocadito uses singer-macro (14 singers) 95% bootstrap CIs (1,000 resamples); `dev_M02` uses clip-level (64 clips) 95% bootstrap CIs.

### Vocadito: Transformer v2 vs Baselines (Note F1 & Onset F1)

| Metric | Feat | Arm | Cond | Trans v2 (Mean [CI]) | Baseline | Baseline (Mean [CI]) | Diff (v2 - Base) [95% CI] | Win (CI > 0)? |
|---|---|---|---|---|---|---|---|---|
| Note F1 | `full` | `augmentation_only` | clean | 0.563 [0.501, 0.622] | `transformer_v1` | 0.568 [0.498, 0.635] | **-0.005** [-0.033, +0.026] | NO |
| Note F1 | `full` | `augmentation_only` | clean | 0.563 [0.501, 0.622] | `tcn` | 0.548 [0.484, 0.614] | **+0.015** [-0.017, +0.047] | NO |
| Note F1 | `full` | `augmentation_only` | clean | 0.563 [0.501, 0.622] | `extractor` | 0.562 [0.489, 0.635] | **+0.001** [-0.038, +0.041] | NO |
| Note F1 | `full` | `augmentation_only` | split | 0.564 [0.502, 0.623] | `transformer_v1` | 0.568 [0.499, 0.636] | **-0.005** [-0.032, +0.025] | NO |
| Note F1 | `full` | `augmentation_only` | split | 0.564 [0.502, 0.623] | `tcn` | 0.547 [0.484, 0.614] | **+0.016** [-0.015, +0.048] | NO |
| Note F1 | `full` | `augmentation_only` | split | 0.564 [0.502, 0.623] | `extractor` | 0.559 [0.488, 0.629] | **+0.005** [-0.032, +0.044] | NO |
| Note F1 | `full` | `augmentation_only` | merge | 0.563 [0.502, 0.622] | `transformer_v1` | 0.566 [0.498, 0.632] | **-0.003** [-0.030, +0.027] | NO |
| Note F1 | `full` | `augmentation_only` | merge | 0.563 [0.502, 0.622] | `tcn` | 0.546 [0.478, 0.616] | **+0.017** [-0.018, +0.056] | NO |
| Note F1 | `full` | `augmentation_only` | merge | 0.563 [0.502, 0.622] | `extractor` | 0.554 [0.486, 0.617] | **+0.009** [-0.026, +0.049] | NO |
| Note F1 | `full` | `augmentation_only` | jitter | 0.562 [0.499, 0.624] | `transformer_v1` | 0.568 [0.500, 0.634] | **-0.006** [-0.038, +0.032] | NO |
| Note F1 | `full` | `augmentation_only` | jitter | 0.562 [0.499, 0.624] | `tcn` | 0.537 [0.460, 0.611] | **+0.026** [-0.020, +0.080] | NO |
| Note F1 | `full` | `augmentation_only` | jitter | 0.562 [0.499, 0.624] | `extractor` | 0.479 [0.406, 0.552] | **+0.083** [+0.039, +0.131] | YES |
| Note F1 | `full` | `bpdr` | clean | 0.577 [0.522, 0.633] | `transformer_v1` | 0.577 [0.510, 0.639] | **-0.000** [-0.031, +0.030] | NO |
| Note F1 | `full` | `bpdr` | clean | 0.577 [0.522, 0.633] | `tcn` | 0.535 [0.460, 0.609] | **+0.042** [-0.000, +0.083] | NO |
| Note F1 | `full` | `bpdr` | clean | 0.577 [0.522, 0.633] | `extractor` | 0.562 [0.489, 0.635] | **+0.015** [-0.028, +0.058] | NO |
| Note F1 | `full` | `bpdr` | split | 0.576 [0.522, 0.633] | `transformer_v1` | 0.580 [0.514, 0.642] | **-0.004** [-0.033, +0.027] | NO |
| Note F1 | `full` | `bpdr` | split | 0.576 [0.522, 0.633] | `tcn` | 0.537 [0.464, 0.609] | **+0.040** [-0.000, +0.080] | NO |
| Note F1 | `full` | `bpdr` | split | 0.576 [0.522, 0.633] | `extractor` | 0.559 [0.488, 0.629] | **+0.018** [-0.023, +0.060] | NO |
| Note F1 | `full` | `bpdr` | merge | 0.577 [0.523, 0.634] | `transformer_v1` | 0.576 [0.510, 0.638] | **+0.001** [-0.029, +0.031] | NO |
| Note F1 | `full` | `bpdr` | merge | 0.577 [0.523, 0.634] | `tcn` | 0.535 [0.460, 0.609] | **+0.042** [-0.000, +0.083] | NO |
| Note F1 | `full` | `bpdr` | merge | 0.577 [0.523, 0.634] | `extractor` | 0.554 [0.486, 0.617] | **+0.023** [-0.015, +0.062] | NO |
| Note F1 | `full` | `bpdr` | jitter | 0.576 [0.522, 0.632] | `transformer_v1` | 0.582 [0.517, 0.643] | **-0.006** [-0.033, +0.020] | NO |
| Note F1 | `full` | `bpdr` | jitter | 0.576 [0.522, 0.632] | `tcn` | 0.529 [0.456, 0.604] | **+0.047** [+0.007, +0.088] | YES |
| Note F1 | `full` | `bpdr` | jitter | 0.576 [0.522, 0.632] | `extractor` | 0.479 [0.406, 0.552] | **+0.096** [+0.056, +0.137] | YES |
| Note F1 | `full` | `frame_only` | clean | 0.563 [0.502, 0.622] | `transformer_v1` | 0.601 [0.538, 0.665] | **-0.038** [-0.060, -0.014] | NO |
| Note F1 | `full` | `frame_only` | clean | 0.563 [0.502, 0.622] | `tcn` | 0.522 [0.449, 0.595] | **+0.041** [+0.001, +0.082] | YES |
| Note F1 | `full` | `frame_only` | clean | 0.563 [0.502, 0.622] | `extractor` | 0.562 [0.489, 0.635] | **+0.001** [-0.037, +0.042] | NO |
| Note F1 | `full` | `frame_only` | split | 0.563 [0.502, 0.622] | `transformer_v1` | 0.601 [0.538, 0.665] | **-0.038** [-0.060, -0.014] | NO |
| Note F1 | `full` | `frame_only` | split | 0.563 [0.502, 0.622] | `tcn` | 0.522 [0.449, 0.595] | **+0.041** [+0.001, +0.082] | YES |
| Note F1 | `full` | `frame_only` | split | 0.563 [0.502, 0.622] | `extractor` | 0.559 [0.488, 0.629] | **+0.005** [-0.033, +0.045] | NO |
| Note F1 | `full` | `frame_only` | merge | 0.563 [0.502, 0.622] | `transformer_v1` | 0.601 [0.538, 0.665] | **-0.038** [-0.060, -0.014] | NO |
| Note F1 | `full` | `frame_only` | merge | 0.563 [0.502, 0.622] | `tcn` | 0.522 [0.449, 0.595] | **+0.041** [+0.001, +0.082] | YES |
| Note F1 | `full` | `frame_only` | merge | 0.563 [0.502, 0.622] | `extractor` | 0.554 [0.486, 0.617] | **+0.010** [-0.026, +0.049] | NO |
| Note F1 | `full` | `frame_only` | jitter | 0.563 [0.502, 0.622] | `transformer_v1` | 0.601 [0.538, 0.665] | **-0.038** [-0.060, -0.014] | NO |
| Note F1 | `full` | `frame_only` | jitter | 0.563 [0.502, 0.622] | `tcn` | 0.522 [0.449, 0.595] | **+0.041** [+0.001, +0.082] | YES |
| Note F1 | `full` | `frame_only` | jitter | 0.563 [0.502, 0.622] | `extractor` | 0.479 [0.406, 0.552] | **+0.084** [+0.044, +0.129] | YES |
| Note F1 | `lean` | `augmentation_only` | clean | 0.566 [0.503, 0.626] | `transformer_v1` | 0.565 [0.492, 0.632] | **+0.001** [-0.036, +0.044] | NO |
| Note F1 | `lean` | `augmentation_only` | clean | 0.566 [0.503, 0.626] | `tcn` | 0.546 [0.485, 0.606] | **+0.020** [-0.014, +0.053] | NO |
| Note F1 | `lean` | `augmentation_only` | clean | 0.566 [0.503, 0.626] | `extractor` | 0.562 [0.489, 0.635] | **+0.004** [-0.041, +0.048] | NO |
| Note F1 | `lean` | `augmentation_only` | split | 0.565 [0.502, 0.626] | `transformer_v1` | 0.565 [0.492, 0.635] | **-0.001** [-0.038, +0.042] | NO |
| Note F1 | `lean` | `augmentation_only` | split | 0.565 [0.502, 0.626] | `tcn` | 0.546 [0.484, 0.606] | **+0.019** [-0.015, +0.052] | NO |
| Note F1 | `lean` | `augmentation_only` | split | 0.565 [0.502, 0.626] | `extractor` | 0.559 [0.488, 0.629] | **+0.006** [-0.036, +0.050] | NO |
| Note F1 | `lean` | `augmentation_only` | merge | 0.565 [0.503, 0.625] | `transformer_v1` | 0.566 [0.493, 0.636] | **-0.002** [-0.042, +0.043] | NO |
| Note F1 | `lean` | `augmentation_only` | merge | 0.565 [0.503, 0.625] | `tcn` | 0.547 [0.485, 0.607] | **+0.018** [-0.016, +0.053] | NO |
| Note F1 | `lean` | `augmentation_only` | merge | 0.565 [0.503, 0.625] | `extractor` | 0.554 [0.486, 0.617] | **+0.011** [-0.030, +0.055] | NO |
| Note F1 | `lean` | `augmentation_only` | jitter | 0.566 [0.502, 0.629] | `transformer_v1` | 0.564 [0.491, 0.628] | **+0.003** [-0.031, +0.042] | NO |
| Note F1 | `lean` | `augmentation_only` | jitter | 0.566 [0.502, 0.629] | `tcn` | 0.543 [0.469, 0.614] | **+0.023** [-0.022, +0.070] | NO |
| Note F1 | `lean` | `augmentation_only` | jitter | 0.566 [0.502, 0.629] | `extractor` | 0.479 [0.406, 0.552] | **+0.087** [+0.041, +0.138] | YES |
| Note F1 | `lean` | `bpdr` | clean | 0.557 [0.497, 0.610] | `transformer_v1` | 0.570 [0.492, 0.642] | **-0.013** [-0.056, +0.034] | NO |
| Note F1 | `lean` | `bpdr` | clean | 0.557 [0.497, 0.610] | `tcn` | 0.537 [0.474, 0.598] | **+0.019** [-0.011, +0.051] | NO |
| Note F1 | `lean` | `bpdr` | clean | 0.557 [0.497, 0.610] | `extractor` | 0.562 [0.489, 0.635] | **-0.006** [-0.045, +0.037] | NO |
| Note F1 | `lean` | `bpdr` | split | 0.557 [0.498, 0.611] | `transformer_v1` | 0.571 [0.492, 0.643] | **-0.015** [-0.058, +0.033] | NO |
| Note F1 | `lean` | `bpdr` | split | 0.557 [0.498, 0.611] | `tcn` | 0.538 [0.475, 0.598] | **+0.019** [-0.011, +0.051] | NO |
| Note F1 | `lean` | `bpdr` | split | 0.557 [0.498, 0.611] | `extractor` | 0.559 [0.488, 0.629] | **-0.002** [-0.040, +0.040] | NO |
| Note F1 | `lean` | `bpdr` | merge | 0.558 [0.498, 0.610] | `transformer_v1` | 0.569 [0.492, 0.642] | **-0.012** [-0.054, +0.034] | NO |
| Note F1 | `lean` | `bpdr` | merge | 0.558 [0.498, 0.610] | `tcn` | 0.539 [0.476, 0.598] | **+0.019** [-0.010, +0.051] | NO |
| Note F1 | `lean` | `bpdr` | merge | 0.558 [0.498, 0.610] | `extractor` | 0.554 [0.486, 0.617] | **+0.004** [-0.032, +0.044] | NO |
| Note F1 | `lean` | `bpdr` | jitter | 0.556 [0.495, 0.610] | `transformer_v1` | 0.567 [0.489, 0.639] | **-0.011** [-0.050, +0.033] | NO |
| Note F1 | `lean` | `bpdr` | jitter | 0.556 [0.495, 0.610] | `tcn` | 0.530 [0.463, 0.593] | **+0.025** [-0.005, +0.061] | NO |
| Note F1 | `lean` | `bpdr` | jitter | 0.556 [0.495, 0.610] | `extractor` | 0.479 [0.406, 0.552] | **+0.077** [+0.034, +0.124] | YES |
| Note F1 | `lean` | `frame_only` | clean | 0.552 [0.488, 0.616] | `transformer_v1` | 0.573 [0.518, 0.625] | **-0.020** [-0.056, +0.021] | NO |
| Note F1 | `lean` | `frame_only` | clean | 0.552 [0.488, 0.616] | `tcn` | 0.538 [0.476, 0.605] | **+0.014** [-0.034, +0.056] | NO |
| Note F1 | `lean` | `frame_only` | clean | 0.552 [0.488, 0.616] | `extractor` | 0.562 [0.489, 0.635] | **-0.010** [-0.067, +0.045] | NO |
| Note F1 | `lean` | `frame_only` | split | 0.552 [0.488, 0.616] | `transformer_v1` | 0.573 [0.518, 0.625] | **-0.020** [-0.056, +0.021] | NO |
| Note F1 | `lean` | `frame_only` | split | 0.552 [0.488, 0.616] | `tcn` | 0.538 [0.476, 0.605] | **+0.014** [-0.034, +0.056] | NO |
| Note F1 | `lean` | `frame_only` | split | 0.552 [0.488, 0.616] | `extractor` | 0.559 [0.488, 0.629] | **-0.006** [-0.063, +0.047] | NO |
| Note F1 | `lean` | `frame_only` | merge | 0.552 [0.488, 0.616] | `transformer_v1` | 0.573 [0.518, 0.625] | **-0.020** [-0.056, +0.021] | NO |
| Note F1 | `lean` | `frame_only` | merge | 0.552 [0.488, 0.616] | `tcn` | 0.538 [0.476, 0.605] | **+0.014** [-0.034, +0.056] | NO |
| Note F1 | `lean` | `frame_only` | merge | 0.552 [0.488, 0.616] | `extractor` | 0.554 [0.486, 0.617] | **-0.001** [-0.054, +0.051] | NO |
| Note F1 | `lean` | `frame_only` | jitter | 0.552 [0.488, 0.616] | `transformer_v1` | 0.573 [0.518, 0.625] | **-0.020** [-0.056, +0.021] | NO |
| Note F1 | `lean` | `frame_only` | jitter | 0.552 [0.488, 0.616] | `tcn` | 0.538 [0.476, 0.605] | **+0.014** [-0.034, +0.056] | NO |
| Note F1 | `lean` | `frame_only` | jitter | 0.552 [0.488, 0.616] | `extractor` | 0.479 [0.406, 0.552] | **+0.073** [+0.015, +0.127] | YES |
| Onset F1 | `full` | `augmentation_only` | clean | 0.708 [0.634, 0.774] | `transformer_v1` | 0.712 [0.637, 0.780] | **-0.005** [-0.026, +0.015] | NO |
| Onset F1 | `full` | `augmentation_only` | clean | 0.708 [0.634, 0.774] | `tcn` | 0.698 [0.626, 0.764] | **+0.009** [-0.016, +0.030] | NO |
| Onset F1 | `full` | `augmentation_only` | clean | 0.708 [0.634, 0.774] | `extractor` | 0.601 [0.556, 0.643] | **+0.107** [+0.055, +0.157] | YES |
| Onset F1 | `full` | `augmentation_only` | split | 0.709 [0.635, 0.774] | `transformer_v1` | 0.711 [0.636, 0.779] | **-0.003** [-0.023, +0.017] | NO |
| Onset F1 | `full` | `augmentation_only` | split | 0.709 [0.635, 0.774] | `tcn` | 0.697 [0.625, 0.762] | **+0.011** [-0.015, +0.032] | NO |
| Onset F1 | `full` | `augmentation_only` | split | 0.709 [0.635, 0.774] | `extractor` | 0.596 [0.555, 0.635] | **+0.113** [+0.061, +0.162] | YES |
| Onset F1 | `full` | `augmentation_only` | merge | 0.707 [0.634, 0.773] | `transformer_v1` | 0.709 [0.634, 0.776] | **-0.002** [-0.020, +0.017] | NO |
| Onset F1 | `full` | `augmentation_only` | merge | 0.707 [0.634, 0.773] | `tcn` | 0.697 [0.625, 0.763] | **+0.010** [-0.016, +0.032] | NO |
| Onset F1 | `full` | `augmentation_only` | merge | 0.707 [0.634, 0.773] | `extractor` | 0.594 [0.551, 0.635] | **+0.113** [+0.057, +0.168] | YES |
| Onset F1 | `full` | `augmentation_only` | jitter | 0.706 [0.634, 0.773] | `transformer_v1` | 0.710 [0.635, 0.775] | **-0.004** [-0.027, +0.020] | NO |
| Onset F1 | `full` | `augmentation_only` | jitter | 0.706 [0.634, 0.773] | `tcn` | 0.679 [0.604, 0.751] | **+0.028** [-0.003, +0.056] | NO |
| Onset F1 | `full` | `augmentation_only` | jitter | 0.706 [0.634, 0.773] | `extractor` | 0.538 [0.504, 0.570] | **+0.168** [+0.102, +0.230] | YES |
| Onset F1 | `full` | `bpdr` | clean | 0.724 [0.656, 0.786] | `transformer_v1` | 0.715 [0.639, 0.781] | **+0.010** [-0.018, +0.035] | NO |
| Onset F1 | `full` | `bpdr` | clean | 0.724 [0.656, 0.786] | `tcn` | 0.688 [0.614, 0.757] | **+0.036** [+0.012, +0.058] | YES |
| Onset F1 | `full` | `bpdr` | clean | 0.724 [0.656, 0.786] | `extractor` | 0.601 [0.556, 0.643] | **+0.123** [+0.076, +0.170] | YES |
| Onset F1 | `full` | `bpdr` | split | 0.724 [0.656, 0.786] | `transformer_v1` | 0.715 [0.640, 0.782] | **+0.009** [-0.018, +0.035] | NO |
| Onset F1 | `full` | `bpdr` | split | 0.724 [0.656, 0.786] | `tcn` | 0.690 [0.615, 0.758] | **+0.035** [+0.011, +0.056] | YES |
| Onset F1 | `full` | `bpdr` | split | 0.724 [0.656, 0.786] | `extractor` | 0.596 [0.555, 0.635] | **+0.128** [+0.082, +0.174] | YES |
| Onset F1 | `full` | `bpdr` | merge | 0.724 [0.656, 0.786] | `transformer_v1` | 0.714 [0.638, 0.780] | **+0.010** [-0.017, +0.034] | NO |
| Onset F1 | `full` | `bpdr` | merge | 0.724 [0.656, 0.786] | `tcn` | 0.688 [0.614, 0.757] | **+0.036** [+0.012, +0.057] | YES |
| Onset F1 | `full` | `bpdr` | merge | 0.724 [0.656, 0.786] | `extractor` | 0.594 [0.551, 0.635] | **+0.130** [+0.079, +0.181] | YES |
| Onset F1 | `full` | `bpdr` | jitter | 0.722 [0.654, 0.784] | `transformer_v1` | 0.715 [0.642, 0.780] | **+0.007** [-0.021, +0.033] | NO |
| Onset F1 | `full` | `bpdr` | jitter | 0.722 [0.654, 0.784] | `tcn` | 0.685 [0.609, 0.756] | **+0.037** [+0.010, +0.061] | YES |
| Onset F1 | `full` | `bpdr` | jitter | 0.722 [0.654, 0.784] | `extractor` | 0.538 [0.504, 0.570] | **+0.184** [+0.125, +0.237] | YES |
| Onset F1 | `full` | `frame_only` | clean | 0.699 [0.631, 0.761] | `transformer_v1` | 0.719 [0.646, 0.784] | **-0.020** [-0.038, -0.001] | NO |
| Onset F1 | `full` | `frame_only` | clean | 0.699 [0.631, 0.761] | `tcn` | 0.680 [0.607, 0.755] | **+0.019** [-0.013, +0.049] | NO |
| Onset F1 | `full` | `frame_only` | clean | 0.699 [0.631, 0.761] | `extractor` | 0.601 [0.556, 0.643] | **+0.098** [+0.048, +0.145] | YES |
| Onset F1 | `full` | `frame_only` | split | 0.699 [0.631, 0.761] | `transformer_v1` | 0.719 [0.646, 0.784] | **-0.020** [-0.038, -0.001] | NO |
| Onset F1 | `full` | `frame_only` | split | 0.699 [0.631, 0.761] | `tcn` | 0.680 [0.607, 0.755] | **+0.019** [-0.013, +0.049] | NO |
| Onset F1 | `full` | `frame_only` | split | 0.699 [0.631, 0.761] | `extractor` | 0.596 [0.555, 0.635] | **+0.103** [+0.055, +0.149] | YES |
| Onset F1 | `full` | `frame_only` | merge | 0.699 [0.631, 0.761] | `transformer_v1` | 0.719 [0.646, 0.784] | **-0.020** [-0.038, -0.001] | NO |
| Onset F1 | `full` | `frame_only` | merge | 0.699 [0.631, 0.761] | `tcn` | 0.680 [0.607, 0.755] | **+0.019** [-0.013, +0.049] | NO |
| Onset F1 | `full` | `frame_only` | merge | 0.699 [0.631, 0.761] | `extractor` | 0.594 [0.551, 0.635] | **+0.105** [+0.052, +0.156] | YES |
| Onset F1 | `full` | `frame_only` | jitter | 0.699 [0.631, 0.761] | `transformer_v1` | 0.719 [0.646, 0.784] | **-0.020** [-0.038, -0.001] | NO |
| Onset F1 | `full` | `frame_only` | jitter | 0.699 [0.631, 0.761] | `tcn` | 0.680 [0.607, 0.755] | **+0.019** [-0.013, +0.049] | NO |
| Onset F1 | `full` | `frame_only` | jitter | 0.699 [0.631, 0.761] | `extractor` | 0.538 [0.504, 0.570] | **+0.161** [+0.098, +0.219] | YES |
| Onset F1 | `lean` | `augmentation_only` | clean | 0.704 [0.630, 0.766] | `transformer_v1` | 0.705 [0.626, 0.773] | **-0.001** [-0.030, +0.034] | NO |
| Onset F1 | `lean` | `augmentation_only` | clean | 0.704 [0.630, 0.766] | `tcn` | 0.692 [0.621, 0.756] | **+0.011** [-0.011, +0.033] | NO |
| Onset F1 | `lean` | `augmentation_only` | clean | 0.704 [0.630, 0.766] | `extractor` | 0.601 [0.556, 0.643] | **+0.103** [+0.051, +0.154] | YES |
| Onset F1 | `lean` | `augmentation_only` | split | 0.706 [0.633, 0.767] | `transformer_v1` | 0.704 [0.626, 0.772] | **+0.002** [-0.030, +0.039] | NO |
| Onset F1 | `lean` | `augmentation_only` | split | 0.706 [0.633, 0.767] | `tcn` | 0.693 [0.622, 0.756] | **+0.013** [-0.010, +0.036] | NO |
| Onset F1 | `lean` | `augmentation_only` | split | 0.706 [0.633, 0.767] | `extractor` | 0.596 [0.555, 0.635] | **+0.110** [+0.058, +0.160] | YES |
| Onset F1 | `lean` | `augmentation_only` | merge | 0.703 [0.630, 0.765] | `transformer_v1` | 0.707 [0.629, 0.775] | **-0.004** [-0.036, +0.032] | NO |
| Onset F1 | `lean` | `augmentation_only` | merge | 0.703 [0.630, 0.765] | `tcn` | 0.691 [0.620, 0.755] | **+0.011** [-0.012, +0.034] | NO |
| Onset F1 | `lean` | `augmentation_only` | merge | 0.703 [0.630, 0.765] | `extractor` | 0.594 [0.551, 0.635] | **+0.109** [+0.051, +0.163] | YES |
| Onset F1 | `lean` | `augmentation_only` | jitter | 0.701 [0.630, 0.762] | `transformer_v1` | 0.708 [0.629, 0.777] | **-0.007** [-0.037, +0.028] | NO |
| Onset F1 | `lean` | `augmentation_only` | jitter | 0.701 [0.630, 0.762] | `tcn` | 0.686 [0.614, 0.753] | **+0.014** [-0.013, +0.043] | NO |
| Onset F1 | `lean` | `augmentation_only` | jitter | 0.701 [0.630, 0.762] | `extractor` | 0.538 [0.504, 0.570] | **+0.162** [+0.095, +0.223] | YES |
| Onset F1 | `lean` | `bpdr` | clean | 0.701 [0.630, 0.767] | `transformer_v1` | 0.706 [0.625, 0.777] | **-0.005** [-0.034, +0.030] | NO |
| Onset F1 | `lean` | `bpdr` | clean | 0.701 [0.630, 0.767] | `tcn` | 0.688 [0.617, 0.754] | **+0.014** [-0.011, +0.038] | NO |
| Onset F1 | `lean` | `bpdr` | clean | 0.701 [0.630, 0.767] | `extractor` | 0.601 [0.556, 0.643] | **+0.100** [+0.048, +0.152] | YES |
| Onset F1 | `lean` | `bpdr` | split | 0.702 [0.630, 0.766] | `transformer_v1` | 0.710 [0.632, 0.779] | **-0.008** [-0.037, +0.022] | NO |
| Onset F1 | `lean` | `bpdr` | split | 0.702 [0.630, 0.766] | `tcn` | 0.688 [0.619, 0.755] | **+0.014** [-0.012, +0.038] | NO |
| Onset F1 | `lean` | `bpdr` | split | 0.702 [0.630, 0.766] | `extractor` | 0.596 [0.555, 0.635] | **+0.106** [+0.054, +0.155] | YES |
| Onset F1 | `lean` | `bpdr` | merge | 0.702 [0.631, 0.767] | `transformer_v1` | 0.706 [0.625, 0.778] | **-0.003** [-0.033, +0.030] | NO |
| Onset F1 | `lean` | `bpdr` | merge | 0.702 [0.631, 0.767] | `tcn` | 0.688 [0.619, 0.755] | **+0.014** [-0.011, +0.038] | NO |
| Onset F1 | `lean` | `bpdr` | merge | 0.702 [0.631, 0.767] | `extractor` | 0.594 [0.551, 0.635] | **+0.108** [+0.050, +0.163] | YES |
| Onset F1 | `lean` | `bpdr` | jitter | 0.701 [0.629, 0.767] | `transformer_v1` | 0.705 [0.624, 0.776] | **-0.004** [-0.031, +0.029] | NO |
| Onset F1 | `lean` | `bpdr` | jitter | 0.701 [0.629, 0.767] | `tcn` | 0.684 [0.613, 0.753] | **+0.017** [-0.009, +0.042] | NO |
| Onset F1 | `lean` | `bpdr` | jitter | 0.701 [0.629, 0.767] | `extractor` | 0.538 [0.504, 0.570] | **+0.163** [+0.096, +0.224] | YES |
| Onset F1 | `lean` | `frame_only` | clean | 0.690 [0.607, 0.757] | `transformer_v1` | 0.713 [0.648, 0.770] | **-0.024** [-0.046, -0.003] | NO |
| Onset F1 | `lean` | `frame_only` | clean | 0.690 [0.607, 0.757] | `tcn` | 0.687 [0.616, 0.758] | **+0.003** [-0.036, +0.039] | NO |
| Onset F1 | `lean` | `frame_only` | clean | 0.690 [0.607, 0.757] | `extractor` | 0.601 [0.556, 0.643] | **+0.089** [+0.030, +0.147] | YES |
| Onset F1 | `lean` | `frame_only` | split | 0.690 [0.607, 0.757] | `transformer_v1` | 0.713 [0.648, 0.770] | **-0.024** [-0.046, -0.003] | NO |
| Onset F1 | `lean` | `frame_only` | split | 0.690 [0.607, 0.757] | `tcn` | 0.687 [0.616, 0.758] | **+0.003** [-0.036, +0.039] | NO |
| Onset F1 | `lean` | `frame_only` | split | 0.690 [0.607, 0.757] | `extractor` | 0.596 [0.555, 0.635] | **+0.094** [+0.037, +0.149] | YES |
| Onset F1 | `lean` | `frame_only` | merge | 0.690 [0.607, 0.757] | `transformer_v1` | 0.713 [0.648, 0.770] | **-0.024** [-0.046, -0.003] | NO |
| Onset F1 | `lean` | `frame_only` | merge | 0.690 [0.607, 0.757] | `tcn` | 0.687 [0.616, 0.758] | **+0.003** [-0.036, +0.039] | NO |
| Onset F1 | `lean` | `frame_only` | merge | 0.690 [0.607, 0.757] | `extractor` | 0.594 [0.551, 0.635] | **+0.096** [+0.033, +0.157] | YES |
| Onset F1 | `lean` | `frame_only` | jitter | 0.690 [0.607, 0.757] | `transformer_v1` | 0.713 [0.648, 0.770] | **-0.024** [-0.046, -0.003] | NO |
| Onset F1 | `lean` | `frame_only` | jitter | 0.690 [0.607, 0.757] | `tcn` | 0.687 [0.616, 0.758] | **+0.003** [-0.036, +0.039] | NO |
| Onset F1 | `lean` | `frame_only` | jitter | 0.690 [0.607, 0.757] | `extractor` | 0.538 [0.504, 0.570] | **+0.151** [+0.076, +0.217] | YES |

### `dev_M02`: Transformer v2 vs Baselines (Boundary Onset F1, 64 Clips)

| Feat | Arm | Cond | Trans v2 (Mean [CI]) | Baseline | Baseline (Mean [CI]) | Diff (v2 - Base) [95% CI] | Win (CI > 0)? |
|---|---|---|---|---|---|---|---|
| `full` | `augmentation_only` | clean | 0.812 [0.785, 0.837] | `transformer_v1` | 0.762 [0.734, 0.789] | **+0.050** [+0.036, +0.063] | YES |
| `full` | `augmentation_only` | clean | 0.812 [0.785, 0.837] | `tcn` | 0.781 [0.754, 0.807] | **+0.031** [+0.022, +0.039] | YES |
| `full` | `augmentation_only` | clean | 0.812 [0.785, 0.837] | `extractor` | 0.548 [0.529, 0.564] | **+0.264** [+0.242, +0.288] | YES |
| `full` | `augmentation_only` | split | 0.812 [0.785, 0.837] | `transformer_v1` | 0.760 [0.732, 0.785] | **+0.052** [+0.038, +0.065] | YES |
| `full` | `augmentation_only` | split | 0.812 [0.785, 0.837] | `tcn` | 0.781 [0.754, 0.808] | **+0.031** [+0.022, +0.039] | YES |
| `full` | `augmentation_only` | split | 0.812 [0.785, 0.837] | `extractor` | 0.537 [0.519, 0.554] | **+0.274** [+0.252, +0.299] | YES |
| `full` | `augmentation_only` | merge | 0.811 [0.784, 0.837] | `transformer_v1` | 0.762 [0.735, 0.790] | **+0.049** [+0.036, +0.062] | YES |
| `full` | `augmentation_only` | merge | 0.811 [0.784, 0.837] | `tcn` | 0.781 [0.754, 0.808] | **+0.030** [+0.022, +0.039] | YES |
| `full` | `augmentation_only` | merge | 0.811 [0.784, 0.837] | `extractor` | 0.541 [0.523, 0.559] | **+0.270** [+0.247, +0.296] | YES |
| `full` | `augmentation_only` | jitter | 0.812 [0.785, 0.838] | `transformer_v1` | 0.759 [0.730, 0.785] | **+0.054** [+0.040, +0.067] | YES |
| `full` | `augmentation_only` | jitter | 0.812 [0.785, 0.838] | `tcn` | 0.784 [0.757, 0.811] | **+0.028** [+0.018, +0.038] | YES |
| `full` | `augmentation_only` | jitter | 0.812 [0.785, 0.838] | `extractor` | 0.526 [0.503, 0.545] | **+0.286** [+0.263, +0.310] | YES |
| `full` | `bpdr` | clean | 0.812 [0.786, 0.838] | `transformer_v1` | 0.767 [0.738, 0.796] | **+0.045** [+0.031, +0.058] | YES |
| `full` | `bpdr` | clean | 0.812 [0.786, 0.838] | `tcn` | 0.782 [0.754, 0.809] | **+0.030** [+0.021, +0.039] | YES |
| `full` | `bpdr` | clean | 0.812 [0.786, 0.838] | `extractor` | 0.548 [0.529, 0.564] | **+0.264** [+0.242, +0.289] | YES |
| `full` | `bpdr` | split | 0.812 [0.786, 0.838] | `transformer_v1` | 0.767 [0.738, 0.795] | **+0.045** [+0.031, +0.058] | YES |
| `full` | `bpdr` | split | 0.812 [0.786, 0.838] | `tcn` | 0.782 [0.754, 0.809] | **+0.030** [+0.021, +0.039] | YES |
| `full` | `bpdr` | split | 0.812 [0.786, 0.838] | `extractor` | 0.537 [0.519, 0.554] | **+0.275** [+0.252, +0.298] | YES |
| `full` | `bpdr` | merge | 0.812 [0.786, 0.838] | `transformer_v1` | 0.767 [0.737, 0.795] | **+0.045** [+0.031, +0.058] | YES |
| `full` | `bpdr` | merge | 0.812 [0.786, 0.838] | `tcn` | 0.782 [0.754, 0.809] | **+0.030** [+0.020, +0.039] | YES |
| `full` | `bpdr` | merge | 0.812 [0.786, 0.838] | `extractor` | 0.541 [0.523, 0.559] | **+0.271** [+0.248, +0.296] | YES |
| `full` | `bpdr` | jitter | 0.812 [0.787, 0.838] | `transformer_v1` | 0.767 [0.737, 0.795] | **+0.045** [+0.032, +0.058] | YES |
| `full` | `bpdr` | jitter | 0.812 [0.787, 0.838] | `tcn` | 0.781 [0.753, 0.808] | **+0.031** [+0.021, +0.040] | YES |
| `full` | `bpdr` | jitter | 0.812 [0.787, 0.838] | `extractor` | 0.526 [0.503, 0.545] | **+0.286** [+0.264, +0.311] | YES |
| `full` | `frame_only` | clean | 0.807 [0.779, 0.834] | `transformer_v1` | 0.754 [0.724, 0.782] | **+0.053** [+0.042, +0.066] | YES |
| `full` | `frame_only` | clean | 0.807 [0.779, 0.834] | `tcn` | 0.780 [0.752, 0.809] | **+0.027** [+0.017, +0.039] | YES |
| `full` | `frame_only` | clean | 0.807 [0.779, 0.834] | `extractor` | 0.548 [0.529, 0.564] | **+0.259** [+0.236, +0.286] | YES |
| `full` | `frame_only` | split | 0.807 [0.779, 0.834] | `transformer_v1` | 0.754 [0.724, 0.782] | **+0.053** [+0.042, +0.066] | YES |
| `full` | `frame_only` | split | 0.807 [0.779, 0.834] | `tcn` | 0.780 [0.752, 0.809] | **+0.027** [+0.017, +0.039] | YES |
| `full` | `frame_only` | split | 0.807 [0.779, 0.834] | `extractor` | 0.537 [0.519, 0.554] | **+0.270** [+0.246, +0.296] | YES |
| `full` | `frame_only` | merge | 0.807 [0.779, 0.834] | `transformer_v1` | 0.754 [0.724, 0.782] | **+0.053** [+0.042, +0.066] | YES |
| `full` | `frame_only` | merge | 0.807 [0.779, 0.834] | `tcn` | 0.780 [0.752, 0.809] | **+0.027** [+0.017, +0.039] | YES |
| `full` | `frame_only` | merge | 0.807 [0.779, 0.834] | `extractor` | 0.541 [0.523, 0.559] | **+0.266** [+0.242, +0.295] | YES |
| `full` | `frame_only` | jitter | 0.807 [0.779, 0.834] | `transformer_v1` | 0.754 [0.724, 0.782] | **+0.053** [+0.042, +0.066] | YES |
| `full` | `frame_only` | jitter | 0.807 [0.779, 0.834] | `tcn` | 0.780 [0.752, 0.809] | **+0.027** [+0.017, +0.039] | YES |
| `full` | `frame_only` | jitter | 0.807 [0.779, 0.834] | `extractor` | 0.526 [0.503, 0.545] | **+0.281** [+0.257, +0.307] | YES |
| `lean` | `augmentation_only` | clean | 0.807 [0.780, 0.834] | `transformer_v1` | 0.765 [0.736, 0.793] | **+0.042** [+0.030, +0.054] | YES |
| `lean` | `augmentation_only` | clean | 0.807 [0.780, 0.834] | `tcn` | 0.783 [0.756, 0.810] | **+0.024** [+0.014, +0.036] | YES |
| `lean` | `augmentation_only` | clean | 0.807 [0.780, 0.834] | `extractor` | 0.548 [0.529, 0.564] | **+0.259** [+0.237, +0.283] | YES |
| `lean` | `augmentation_only` | split | 0.807 [0.780, 0.833] | `transformer_v1` | 0.764 [0.734, 0.791] | **+0.043** [+0.032, +0.056] | YES |
| `lean` | `augmentation_only` | split | 0.807 [0.780, 0.833] | `tcn` | 0.782 [0.755, 0.809] | **+0.025** [+0.015, +0.037] | YES |
| `lean` | `augmentation_only` | split | 0.807 [0.780, 0.833] | `extractor` | 0.537 [0.519, 0.554] | **+0.270** [+0.248, +0.293] | YES |
| `lean` | `augmentation_only` | merge | 0.808 [0.780, 0.834] | `transformer_v1` | 0.766 [0.736, 0.794] | **+0.042** [+0.031, +0.054] | YES |
| `lean` | `augmentation_only` | merge | 0.808 [0.780, 0.834] | `tcn` | 0.782 [0.754, 0.809] | **+0.026** [+0.015, +0.037] | YES |
| `lean` | `augmentation_only` | merge | 0.808 [0.780, 0.834] | `extractor` | 0.541 [0.523, 0.559] | **+0.267** [+0.244, +0.292] | YES |
| `lean` | `augmentation_only` | jitter | 0.807 [0.780, 0.833] | `transformer_v1` | 0.764 [0.735, 0.791] | **+0.044** [+0.033, +0.055] | YES |
| `lean` | `augmentation_only` | jitter | 0.807 [0.780, 0.833] | `tcn` | 0.784 [0.756, 0.811] | **+0.024** [+0.013, +0.035] | YES |
| `lean` | `augmentation_only` | jitter | 0.807 [0.780, 0.833] | `extractor` | 0.526 [0.503, 0.545] | **+0.281** [+0.258, +0.304] | YES |
| `lean` | `bpdr` | clean | 0.814 [0.788, 0.840] | `transformer_v1` | 0.763 [0.734, 0.790] | **+0.051** [+0.038, +0.065] | YES |
| `lean` | `bpdr` | clean | 0.814 [0.788, 0.840] | `tcn` | 0.781 [0.754, 0.808] | **+0.032** [+0.023, +0.042] | YES |
| `lean` | `bpdr` | clean | 0.814 [0.788, 0.840] | `extractor` | 0.548 [0.529, 0.564] | **+0.266** [+0.242, +0.290] | YES |
| `lean` | `bpdr` | split | 0.814 [0.788, 0.840] | `transformer_v1` | 0.763 [0.734, 0.790] | **+0.051** [+0.038, +0.065] | YES |
| `lean` | `bpdr` | split | 0.814 [0.788, 0.840] | `tcn` | 0.782 [0.754, 0.809] | **+0.032** [+0.022, +0.041] | YES |
| `lean` | `bpdr` | split | 0.814 [0.788, 0.840] | `extractor` | 0.537 [0.519, 0.554] | **+0.276** [+0.253, +0.301] | YES |
| `lean` | `bpdr` | merge | 0.813 [0.787, 0.839] | `transformer_v1` | 0.763 [0.734, 0.790] | **+0.050** [+0.038, +0.064] | YES |
| `lean` | `bpdr` | merge | 0.813 [0.787, 0.839] | `tcn` | 0.782 [0.754, 0.808] | **+0.031** [+0.022, +0.040] | YES |
| `lean` | `bpdr` | merge | 0.813 [0.787, 0.839] | `extractor` | 0.541 [0.523, 0.559] | **+0.272** [+0.248, +0.298] | YES |
| `lean` | `bpdr` | jitter | 0.815 [0.789, 0.841] | `transformer_v1` | 0.764 [0.735, 0.791] | **+0.051** [+0.037, +0.065] | YES |
| `lean` | `bpdr` | jitter | 0.815 [0.789, 0.841] | `tcn` | 0.782 [0.754, 0.809] | **+0.033** [+0.023, +0.043] | YES |
| `lean` | `bpdr` | jitter | 0.815 [0.789, 0.841] | `extractor` | 0.526 [0.503, 0.545] | **+0.289** [+0.265, +0.314] | YES |
| `lean` | `frame_only` | clean | 0.811 [0.785, 0.836] | `transformer_v1` | 0.741 [0.714, 0.767] | **+0.070** [+0.057, +0.083] | YES |
| `lean` | `frame_only` | clean | 0.811 [0.785, 0.836] | `tcn` | 0.781 [0.752, 0.808] | **+0.030** [+0.019, +0.042] | YES |
| `lean` | `frame_only` | clean | 0.811 [0.785, 0.836] | `extractor` | 0.548 [0.529, 0.564] | **+0.264** [+0.241, +0.288] | YES |
| `lean` | `frame_only` | split | 0.811 [0.785, 0.836] | `transformer_v1` | 0.741 [0.714, 0.767] | **+0.070** [+0.057, +0.083] | YES |
| `lean` | `frame_only` | split | 0.811 [0.785, 0.836] | `tcn` | 0.781 [0.752, 0.808] | **+0.030** [+0.019, +0.042] | YES |
| `lean` | `frame_only` | split | 0.811 [0.785, 0.836] | `extractor` | 0.537 [0.519, 0.554] | **+0.274** [+0.251, +0.298] | YES |
| `lean` | `frame_only` | merge | 0.811 [0.785, 0.836] | `transformer_v1` | 0.741 [0.714, 0.767] | **+0.070** [+0.057, +0.083] | YES |
| `lean` | `frame_only` | merge | 0.811 [0.785, 0.836] | `tcn` | 0.781 [0.752, 0.808] | **+0.030** [+0.019, +0.042] | YES |
| `lean` | `frame_only` | merge | 0.811 [0.785, 0.836] | `extractor` | 0.541 [0.523, 0.559] | **+0.270** [+0.247, +0.296] | YES |
| `lean` | `frame_only` | jitter | 0.811 [0.785, 0.836] | `transformer_v1` | 0.741 [0.714, 0.767] | **+0.070** [+0.057, +0.083] | YES |
| `lean` | `frame_only` | jitter | 0.811 [0.785, 0.836] | `tcn` | 0.781 [0.752, 0.808] | **+0.030** [+0.019, +0.042] | YES |
| `lean` | `frame_only` | jitter | 0.811 [0.785, 0.836] | `extractor` | 0.526 [0.503, 0.545] | **+0.285** [+0.262, +0.310] | YES |

### Transformer v2 Fit Threshold & Seed Coverage

| Arch | Feat | Arm | Seeds Run | Fits | Healthy | Thresholds (`thr`) |
|---|---|---|---|---|---|---|
| `transformer_v2` | `full` | `augmentation_only` | `339, 340` | 2 | 2/2 | `0.7, 0.75` |
| `transformer_v2` | `full` | `bpdr` | `339, 340` | 2 | 2/2 | `0.6` |
| `transformer_v2` | `full` | `frame_only` | `339, 340` | 2 | 2/2 | `0.65, 0.7` |
| `transformer_v2` | `lean` | `augmentation_only` | `339, 340` | 2 | 2/2 | `0.6, 0.7` |
| `transformer_v2` | `lean` | `bpdr` | `339, 340` | 2 | 2/2 | `0.55, 0.65` |
| `transformer_v2` | `lean` | `frame_only` | `339, 340` | 2 | 2/2 | `0.65, 0.7` |

- **Seed coverage**: `transformer_v2`: 2 seed(s) ([339, 340]).
- **Threshold distribution**: Observed `thr` values across 12 fits: `0.55, 0.6, 0.65, 0.7, 0.75`.

## Transformer v3 Statistical Comparison (vs v1, transformer_v2, TCN, and Extractor)

**Methodology**: Vocadito uses singer-macro (14 singers) 95% bootstrap CIs (1,000 resamples); `dev_M02` uses clip-level (64 clips) 95% bootstrap CIs.

### Vocadito: Transformer v3 vs Baselines (Note F1 & Onset F1)

| Metric | Feat | Arm | Cond | Trans v3 (Mean [CI]) | Baseline | Baseline (Mean [CI]) | Diff (v3 - Base) [95% CI] | Win (CI > 0)? |
|---|---|---|---|---|---|---|---|---|
| Note F1 | `full` | `augmentation_only` | clean | 0.528 [0.458, 0.595] | `transformer_v1` | 0.568 [0.498, 0.635] | **-0.040** [-0.061, -0.019] | NO |
| Note F1 | `full` | `augmentation_only` | clean | 0.528 [0.458, 0.595] | `transformer_v2` | 0.563 [0.501, 0.622] | **-0.035** [-0.074, -0.001] | NO |
| Note F1 | `full` | `augmentation_only` | clean | 0.528 [0.458, 0.595] | `tcn` | 0.548 [0.484, 0.614] | **-0.021** [-0.045, +0.004] | NO |
| Note F1 | `full` | `augmentation_only` | clean | 0.528 [0.458, 0.595] | `extractor` | 0.562 [0.489, 0.635] | **-0.034** [-0.064, -0.004] | NO |
| Note F1 | `full` | `augmentation_only` | split | 0.529 [0.458, 0.596] | `transformer_v1` | 0.568 [0.499, 0.636] | **-0.039** [-0.060, -0.019] | NO |
| Note F1 | `full` | `augmentation_only` | split | 0.529 [0.458, 0.596] | `transformer_v2` | 0.564 [0.502, 0.623] | **-0.035** [-0.073, -0.000] | NO |
| Note F1 | `full` | `augmentation_only` | split | 0.529 [0.458, 0.596] | `tcn` | 0.547 [0.484, 0.614] | **-0.018** [-0.043, +0.006] | NO |
| Note F1 | `full` | `augmentation_only` | split | 0.529 [0.458, 0.596] | `extractor` | 0.559 [0.488, 0.629] | **-0.030** [-0.058, -0.000] | NO |
| Note F1 | `full` | `augmentation_only` | merge | 0.525 [0.457, 0.591] | `transformer_v1` | 0.566 [0.498, 0.632] | **-0.041** [-0.064, -0.019] | NO |
| Note F1 | `full` | `augmentation_only` | merge | 0.525 [0.457, 0.591] | `transformer_v2` | 0.563 [0.502, 0.622] | **-0.038** [-0.076, -0.004] | NO |
| Note F1 | `full` | `augmentation_only` | merge | 0.525 [0.457, 0.591] | `tcn` | 0.546 [0.478, 0.616] | **-0.021** [-0.048, +0.004] | NO |
| Note F1 | `full` | `augmentation_only` | merge | 0.525 [0.457, 0.591] | `extractor` | 0.554 [0.486, 0.617] | **-0.028** [-0.058, +0.001] | NO |
| Note F1 | `full` | `augmentation_only` | jitter | 0.525 [0.454, 0.592] | `transformer_v1` | 0.568 [0.500, 0.634] | **-0.043** [-0.065, -0.021] | NO |
| Note F1 | `full` | `augmentation_only` | jitter | 0.525 [0.454, 0.592] | `transformer_v2` | 0.562 [0.499, 0.624] | **-0.037** [-0.077, -0.003] | NO |
| Note F1 | `full` | `augmentation_only` | jitter | 0.525 [0.454, 0.592] | `tcn` | 0.537 [0.460, 0.611] | **-0.012** [-0.035, +0.010] | NO |
| Note F1 | `full` | `augmentation_only` | jitter | 0.525 [0.454, 0.592] | `extractor` | 0.479 [0.406, 0.552] | **+0.046** [+0.025, +0.072] | YES |
| Note F1 | `full` | `bpdr` | clean | 0.531 [0.463, 0.597] | `transformer_v1` | 0.577 [0.510, 0.639] | **-0.046** [-0.071, -0.019] | NO |
| Note F1 | `full` | `bpdr` | clean | 0.531 [0.463, 0.597] | `transformer_v2` | 0.577 [0.522, 0.633] | **-0.045** [-0.078, -0.013] | NO |
| Note F1 | `full` | `bpdr` | clean | 0.531 [0.463, 0.597] | `tcn` | 0.535 [0.460, 0.609] | **-0.004** [-0.024, +0.016] | NO |
| Note F1 | `full` | `bpdr` | clean | 0.531 [0.463, 0.597] | `extractor` | 0.562 [0.489, 0.635] | **-0.031** [-0.055, -0.004] | NO |
| Note F1 | `full` | `bpdr` | split | 0.531 [0.463, 0.598] | `transformer_v1` | 0.580 [0.514, 0.642] | **-0.049** [-0.073, -0.022] | NO |
| Note F1 | `full` | `bpdr` | split | 0.531 [0.463, 0.598] | `transformer_v2` | 0.576 [0.522, 0.633] | **-0.045** [-0.078, -0.013] | NO |
| Note F1 | `full` | `bpdr` | split | 0.531 [0.463, 0.598] | `tcn` | 0.537 [0.464, 0.609] | **-0.006** [-0.025, +0.013] | NO |
| Note F1 | `full` | `bpdr` | split | 0.531 [0.463, 0.598] | `extractor` | 0.559 [0.488, 0.629] | **-0.027** [-0.052, -0.000] | NO |
| Note F1 | `full` | `bpdr` | merge | 0.529 [0.463, 0.594] | `transformer_v1` | 0.576 [0.510, 0.638] | **-0.046** [-0.071, -0.020] | NO |
| Note F1 | `full` | `bpdr` | merge | 0.529 [0.463, 0.594] | `transformer_v2` | 0.577 [0.523, 0.634] | **-0.047** [-0.080, -0.018] | NO |
| Note F1 | `full` | `bpdr` | merge | 0.529 [0.463, 0.594] | `tcn` | 0.535 [0.460, 0.609] | **-0.006** [-0.025, +0.012] | NO |
| Note F1 | `full` | `bpdr` | merge | 0.529 [0.463, 0.594] | `extractor` | 0.554 [0.486, 0.617] | **-0.024** [-0.046, +0.000] | NO |
| Note F1 | `full` | `bpdr` | jitter | 0.529 [0.459, 0.597] | `transformer_v1` | 0.582 [0.517, 0.643] | **-0.053** [-0.080, -0.024] | NO |
| Note F1 | `full` | `bpdr` | jitter | 0.529 [0.459, 0.597] | `transformer_v2` | 0.576 [0.522, 0.632] | **-0.047** [-0.077, -0.014] | NO |
| Note F1 | `full` | `bpdr` | jitter | 0.529 [0.459, 0.597] | `tcn` | 0.529 [0.456, 0.604] | **+0.000** [-0.023, +0.023] | NO |
| Note F1 | `full` | `bpdr` | jitter | 0.529 [0.459, 0.597] | `extractor` | 0.479 [0.406, 0.552] | **+0.050** [+0.025, +0.079] | YES |
| Note F1 | `full` | `frame_only` | clean | 0.537 [0.463, 0.604] | `transformer_v1` | 0.601 [0.538, 0.665] | **-0.064** [-0.098, -0.034] | NO |
| Note F1 | `full` | `frame_only` | clean | 0.537 [0.463, 0.604] | `transformer_v2` | 0.563 [0.502, 0.622] | **-0.026** [-0.066, +0.009] | NO |
| Note F1 | `full` | `frame_only` | clean | 0.537 [0.463, 0.604] | `tcn` | 0.522 [0.449, 0.595] | **+0.015** [-0.002, +0.030] | NO |
| Note F1 | `full` | `frame_only` | clean | 0.537 [0.463, 0.604] | `extractor` | 0.562 [0.489, 0.635] | **-0.025** [-0.054, +0.004] | NO |
| Note F1 | `full` | `frame_only` | split | 0.537 [0.463, 0.604] | `transformer_v1` | 0.601 [0.538, 0.665] | **-0.064** [-0.098, -0.034] | NO |
| Note F1 | `full` | `frame_only` | split | 0.537 [0.463, 0.604] | `transformer_v2` | 0.563 [0.502, 0.622] | **-0.026** [-0.066, +0.009] | NO |
| Note F1 | `full` | `frame_only` | split | 0.537 [0.463, 0.604] | `tcn` | 0.522 [0.449, 0.595] | **+0.015** [-0.002, +0.030] | NO |
| Note F1 | `full` | `frame_only` | split | 0.537 [0.463, 0.604] | `extractor` | 0.559 [0.488, 0.629] | **-0.022** [-0.049, +0.008] | NO |
| Note F1 | `full` | `frame_only` | merge | 0.537 [0.463, 0.604] | `transformer_v1` | 0.601 [0.538, 0.665] | **-0.064** [-0.098, -0.034] | NO |
| Note F1 | `full` | `frame_only` | merge | 0.537 [0.463, 0.604] | `transformer_v2` | 0.563 [0.502, 0.622] | **-0.026** [-0.066, +0.009] | NO |
| Note F1 | `full` | `frame_only` | merge | 0.537 [0.463, 0.604] | `tcn` | 0.522 [0.449, 0.595] | **+0.015** [-0.002, +0.030] | NO |
| Note F1 | `full` | `frame_only` | merge | 0.537 [0.463, 0.604] | `extractor` | 0.554 [0.486, 0.617] | **-0.017** [-0.039, +0.009] | NO |
| Note F1 | `full` | `frame_only` | jitter | 0.537 [0.463, 0.604] | `transformer_v1` | 0.601 [0.538, 0.665] | **-0.064** [-0.098, -0.034] | NO |
| Note F1 | `full` | `frame_only` | jitter | 0.537 [0.463, 0.604] | `transformer_v2` | 0.563 [0.502, 0.622] | **-0.026** [-0.066, +0.009] | NO |
| Note F1 | `full` | `frame_only` | jitter | 0.537 [0.463, 0.604] | `tcn` | 0.522 [0.449, 0.595] | **+0.015** [-0.002, +0.030] | NO |
| Note F1 | `full` | `frame_only` | jitter | 0.537 [0.463, 0.604] | `extractor` | 0.479 [0.406, 0.552] | **+0.058** [+0.032, +0.090] | YES |
| Note F1 | `lean` | `augmentation_only` | clean | 0.514 [0.448, 0.573] | `transformer_v1` | 0.565 [0.492, 0.632] | **-0.052** [-0.072, -0.027] | NO |
| Note F1 | `lean` | `augmentation_only` | clean | 0.514 [0.448, 0.573] | `transformer_v2` | 0.566 [0.503, 0.626] | **-0.052** [-0.091, -0.015] | NO |
| Note F1 | `lean` | `augmentation_only` | clean | 0.514 [0.448, 0.573] | `tcn` | 0.546 [0.485, 0.606] | **-0.033** [-0.055, -0.012] | NO |
| Note F1 | `lean` | `augmentation_only` | clean | 0.514 [0.448, 0.573] | `extractor` | 0.562 [0.489, 0.635] | **-0.049** [-0.078, -0.018] | NO |
| Note F1 | `lean` | `augmentation_only` | split | 0.516 [0.451, 0.577] | `transformer_v1` | 0.565 [0.492, 0.635] | **-0.049** [-0.071, -0.025] | NO |
| Note F1 | `lean` | `augmentation_only` | split | 0.516 [0.451, 0.577] | `transformer_v2` | 0.565 [0.502, 0.626] | **-0.048** [-0.087, -0.013] | NO |
| Note F1 | `lean` | `augmentation_only` | split | 0.516 [0.451, 0.577] | `tcn` | 0.546 [0.484, 0.606] | **-0.030** [-0.054, -0.006] | NO |
| Note F1 | `lean` | `augmentation_only` | split | 0.516 [0.451, 0.577] | `extractor` | 0.559 [0.488, 0.629] | **-0.043** [-0.073, -0.012] | NO |
| Note F1 | `lean` | `augmentation_only` | merge | 0.515 [0.449, 0.574] | `transformer_v1` | 0.566 [0.493, 0.636] | **-0.052** [-0.073, -0.027] | NO |
| Note F1 | `lean` | `augmentation_only` | merge | 0.515 [0.449, 0.574] | `transformer_v2` | 0.565 [0.503, 0.625] | **-0.050** [-0.089, -0.014] | NO |
| Note F1 | `lean` | `augmentation_only` | merge | 0.515 [0.449, 0.574] | `tcn` | 0.547 [0.485, 0.607] | **-0.032** [-0.055, -0.011] | NO |
| Note F1 | `lean` | `augmentation_only` | merge | 0.515 [0.449, 0.574] | `extractor` | 0.554 [0.486, 0.617] | **-0.039** [-0.065, -0.013] | NO |
| Note F1 | `lean` | `augmentation_only` | jitter | 0.519 [0.455, 0.580] | `transformer_v1` | 0.564 [0.491, 0.628] | **-0.045** [-0.065, -0.021] | NO |
| Note F1 | `lean` | `augmentation_only` | jitter | 0.519 [0.455, 0.580] | `transformer_v2` | 0.566 [0.502, 0.629] | **-0.048** [-0.089, -0.011] | NO |
| Note F1 | `lean` | `augmentation_only` | jitter | 0.519 [0.455, 0.580] | `tcn` | 0.543 [0.469, 0.614] | **-0.024** [-0.044, -0.005] | NO |
| Note F1 | `lean` | `augmentation_only` | jitter | 0.519 [0.455, 0.580] | `extractor` | 0.479 [0.406, 0.552] | **+0.039** [+0.013, +0.070] | YES |
| Note F1 | `lean` | `bpdr` | clean | 0.520 [0.455, 0.583] | `transformer_v1` | 0.570 [0.492, 0.642] | **-0.049** [-0.069, -0.028] | NO |
| Note F1 | `lean` | `bpdr` | clean | 0.520 [0.455, 0.583] | `transformer_v2` | 0.557 [0.497, 0.610] | **-0.036** [-0.076, -0.003] | NO |
| Note F1 | `lean` | `bpdr` | clean | 0.520 [0.455, 0.583] | `tcn` | 0.537 [0.474, 0.598] | **-0.017** [-0.037, +0.002] | NO |
| Note F1 | `lean` | `bpdr` | clean | 0.520 [0.455, 0.583] | `extractor` | 0.562 [0.489, 0.635] | **-0.042** [-0.070, -0.015] | NO |
| Note F1 | `lean` | `bpdr` | split | 0.521 [0.455, 0.584] | `transformer_v1` | 0.571 [0.492, 0.643] | **-0.051** [-0.071, -0.029] | NO |
| Note F1 | `lean` | `bpdr` | split | 0.521 [0.455, 0.584] | `transformer_v2` | 0.557 [0.498, 0.611] | **-0.036** [-0.077, -0.002] | NO |
| Note F1 | `lean` | `bpdr` | split | 0.521 [0.455, 0.584] | `tcn` | 0.538 [0.475, 0.598] | **-0.017** [-0.037, +0.002] | NO |
| Note F1 | `lean` | `bpdr` | split | 0.521 [0.455, 0.584] | `extractor` | 0.559 [0.488, 0.629] | **-0.038** [-0.065, -0.011] | NO |
| Note F1 | `lean` | `bpdr` | merge | 0.520 [0.455, 0.583] | `transformer_v1` | 0.569 [0.492, 0.642] | **-0.049** [-0.068, -0.028] | NO |
| Note F1 | `lean` | `bpdr` | merge | 0.520 [0.455, 0.583] | `transformer_v2` | 0.558 [0.498, 0.610] | **-0.037** [-0.076, -0.004] | NO |
| Note F1 | `lean` | `bpdr` | merge | 0.520 [0.455, 0.583] | `tcn` | 0.539 [0.476, 0.598] | **-0.018** [-0.038, +0.001] | NO |
| Note F1 | `lean` | `bpdr` | merge | 0.520 [0.455, 0.583] | `extractor` | 0.554 [0.486, 0.617] | **-0.033** [-0.058, -0.008] | NO |
| Note F1 | `lean` | `bpdr` | jitter | 0.521 [0.455, 0.584] | `transformer_v1` | 0.567 [0.489, 0.639] | **-0.046** [-0.067, -0.026] | NO |
| Note F1 | `lean` | `bpdr` | jitter | 0.521 [0.455, 0.584] | `transformer_v2` | 0.556 [0.495, 0.610] | **-0.035** [-0.075, -0.002] | NO |
| Note F1 | `lean` | `bpdr` | jitter | 0.521 [0.455, 0.584] | `tcn` | 0.530 [0.463, 0.593] | **-0.009** [-0.027, +0.007] | NO |
| Note F1 | `lean` | `bpdr` | jitter | 0.521 [0.455, 0.584] | `extractor` | 0.479 [0.406, 0.552] | **+0.042** [+0.013, +0.074] | YES |
| Note F1 | `lean` | `frame_only` | clean | 0.508 [0.442, 0.572] | `transformer_v1` | 0.573 [0.518, 0.625] | **-0.065** [-0.105, -0.024] | NO |
| Note F1 | `lean` | `frame_only` | clean | 0.508 [0.442, 0.572] | `transformer_v2` | 0.552 [0.488, 0.616] | **-0.044** [-0.091, +0.002] | NO |
| Note F1 | `lean` | `frame_only` | clean | 0.508 [0.442, 0.572] | `tcn` | 0.538 [0.476, 0.605] | **-0.030** [-0.059, +0.000] | NO |
| Note F1 | `lean` | `frame_only` | clean | 0.508 [0.442, 0.572] | `extractor` | 0.562 [0.489, 0.635] | **-0.054** [-0.087, -0.019] | NO |
| Note F1 | `lean` | `frame_only` | split | 0.508 [0.442, 0.572] | `transformer_v1` | 0.573 [0.518, 0.625] | **-0.065** [-0.105, -0.024] | NO |
| Note F1 | `lean` | `frame_only` | split | 0.508 [0.442, 0.572] | `transformer_v2` | 0.552 [0.488, 0.616] | **-0.044** [-0.091, +0.002] | NO |
| Note F1 | `lean` | `frame_only` | split | 0.508 [0.442, 0.572] | `tcn` | 0.538 [0.476, 0.605] | **-0.030** [-0.059, +0.000] | NO |
| Note F1 | `lean` | `frame_only` | split | 0.508 [0.442, 0.572] | `extractor` | 0.559 [0.488, 0.629] | **-0.051** [-0.083, -0.016] | NO |
| Note F1 | `lean` | `frame_only` | merge | 0.508 [0.442, 0.572] | `transformer_v1` | 0.573 [0.518, 0.625] | **-0.065** [-0.105, -0.024] | NO |
| Note F1 | `lean` | `frame_only` | merge | 0.508 [0.442, 0.572] | `transformer_v2` | 0.552 [0.488, 0.616] | **-0.044** [-0.091, +0.002] | NO |
| Note F1 | `lean` | `frame_only` | merge | 0.508 [0.442, 0.572] | `tcn` | 0.538 [0.476, 0.605] | **-0.030** [-0.059, +0.000] | NO |
| Note F1 | `lean` | `frame_only` | merge | 0.508 [0.442, 0.572] | `extractor` | 0.554 [0.486, 0.617] | **-0.046** [-0.076, -0.014] | NO |
| Note F1 | `lean` | `frame_only` | jitter | 0.508 [0.442, 0.572] | `transformer_v1` | 0.573 [0.518, 0.625] | **-0.065** [-0.105, -0.024] | NO |
| Note F1 | `lean` | `frame_only` | jitter | 0.508 [0.442, 0.572] | `transformer_v2` | 0.552 [0.488, 0.616] | **-0.044** [-0.091, +0.002] | NO |
| Note F1 | `lean` | `frame_only` | jitter | 0.508 [0.442, 0.572] | `tcn` | 0.538 [0.476, 0.605] | **-0.030** [-0.059, +0.000] | NO |
| Note F1 | `lean` | `frame_only` | jitter | 0.508 [0.442, 0.572] | `extractor` | 0.479 [0.406, 0.552] | **+0.029** [-0.000, +0.065] | NO |
| Onset F1 | `full` | `augmentation_only` | clean | 0.680 [0.613, 0.749] | `transformer_v1` | 0.712 [0.637, 0.780] | **-0.032** [-0.051, -0.013] | NO |
| Onset F1 | `full` | `augmentation_only` | clean | 0.680 [0.613, 0.749] | `transformer_v2` | 0.708 [0.634, 0.774] | **-0.027** [-0.049, -0.001] | NO |
| Onset F1 | `full` | `augmentation_only` | clean | 0.680 [0.613, 0.749] | `tcn` | 0.698 [0.626, 0.764] | **-0.018** [-0.033, -0.004] | NO |
| Onset F1 | `full` | `augmentation_only` | clean | 0.680 [0.613, 0.749] | `extractor` | 0.601 [0.556, 0.643] | **+0.079** [+0.031, +0.131] | YES |
| Onset F1 | `full` | `augmentation_only` | split | 0.684 [0.617, 0.751] | `transformer_v1` | 0.711 [0.636, 0.779] | **-0.027** [-0.046, -0.008] | NO |
| Onset F1 | `full` | `augmentation_only` | split | 0.684 [0.617, 0.751] | `transformer_v2` | 0.709 [0.635, 0.774] | **-0.025** [-0.046, +0.001] | NO |
| Onset F1 | `full` | `augmentation_only` | split | 0.684 [0.617, 0.751] | `tcn` | 0.697 [0.625, 0.762] | **-0.014** [-0.028, -0.001] | NO |
| Onset F1 | `full` | `augmentation_only` | split | 0.684 [0.617, 0.751] | `extractor` | 0.596 [0.555, 0.635] | **+0.088** [+0.040, +0.139] | YES |
| Onset F1 | `full` | `augmentation_only` | merge | 0.677 [0.609, 0.745] | `transformer_v1` | 0.709 [0.634, 0.776] | **-0.032** [-0.054, -0.011] | NO |
| Onset F1 | `full` | `augmentation_only` | merge | 0.677 [0.609, 0.745] | `transformer_v2` | 0.707 [0.634, 0.773] | **-0.030** [-0.052, -0.004] | NO |
| Onset F1 | `full` | `augmentation_only` | merge | 0.677 [0.609, 0.745] | `tcn` | 0.697 [0.625, 0.763] | **-0.020** [-0.035, -0.006] | NO |
| Onset F1 | `full` | `augmentation_only` | merge | 0.677 [0.609, 0.745] | `extractor` | 0.594 [0.551, 0.635] | **+0.083** [+0.030, +0.140] | YES |
| Onset F1 | `full` | `augmentation_only` | jitter | 0.679 [0.612, 0.747] | `transformer_v1` | 0.710 [0.635, 0.775] | **-0.031** [-0.052, -0.010] | NO |
| Onset F1 | `full` | `augmentation_only` | jitter | 0.679 [0.612, 0.747] | `transformer_v2` | 0.706 [0.634, 0.773] | **-0.027** [-0.050, -0.001] | NO |
| Onset F1 | `full` | `augmentation_only` | jitter | 0.679 [0.612, 0.747] | `tcn` | 0.679 [0.604, 0.751] | **+0.000** [-0.014, +0.015] | NO |
| Onset F1 | `full` | `augmentation_only` | jitter | 0.679 [0.612, 0.747] | `extractor` | 0.538 [0.504, 0.570] | **+0.141** [+0.084, +0.198] | YES |
| Onset F1 | `full` | `bpdr` | clean | 0.680 [0.609, 0.747] | `transformer_v1` | 0.715 [0.639, 0.781] | **-0.035** [-0.050, -0.018] | NO |
| Onset F1 | `full` | `bpdr` | clean | 0.680 [0.609, 0.747] | `transformer_v2` | 0.724 [0.656, 0.786] | **-0.044** [-0.071, -0.017] | NO |
| Onset F1 | `full` | `bpdr` | clean | 0.680 [0.609, 0.747] | `tcn` | 0.688 [0.614, 0.757] | **-0.008** [-0.021, +0.004] | NO |
| Onset F1 | `full` | `bpdr` | clean | 0.680 [0.609, 0.747] | `extractor` | 0.601 [0.556, 0.643] | **+0.079** [+0.028, +0.132] | YES |
| Onset F1 | `full` | `bpdr` | split | 0.681 [0.611, 0.748] | `transformer_v1` | 0.715 [0.640, 0.782] | **-0.033** [-0.049, -0.018] | NO |
| Onset F1 | `full` | `bpdr` | split | 0.681 [0.611, 0.748] | `transformer_v2` | 0.724 [0.656, 0.786] | **-0.043** [-0.068, -0.016] | NO |
| Onset F1 | `full` | `bpdr` | split | 0.681 [0.611, 0.748] | `tcn` | 0.690 [0.615, 0.758] | **-0.008** [-0.021, +0.004] | NO |
| Onset F1 | `full` | `bpdr` | split | 0.681 [0.611, 0.748] | `extractor` | 0.596 [0.555, 0.635] | **+0.085** [+0.037, +0.138] | YES |
| Onset F1 | `full` | `bpdr` | merge | 0.679 [0.608, 0.747] | `transformer_v1` | 0.714 [0.638, 0.780] | **-0.035** [-0.051, -0.019] | NO |
| Onset F1 | `full` | `bpdr` | merge | 0.679 [0.608, 0.747] | `transformer_v2` | 0.724 [0.656, 0.786] | **-0.045** [-0.071, -0.019] | NO |
| Onset F1 | `full` | `bpdr` | merge | 0.679 [0.608, 0.747] | `tcn` | 0.688 [0.614, 0.757] | **-0.009** [-0.021, +0.002] | NO |
| Onset F1 | `full` | `bpdr` | merge | 0.679 [0.608, 0.747] | `extractor` | 0.594 [0.551, 0.635] | **+0.085** [+0.030, +0.144] | YES |
| Onset F1 | `full` | `bpdr` | jitter | 0.679 [0.606, 0.750] | `transformer_v1` | 0.715 [0.642, 0.780] | **-0.036** [-0.054, -0.016] | NO |
| Onset F1 | `full` | `bpdr` | jitter | 0.679 [0.606, 0.750] | `transformer_v2` | 0.722 [0.654, 0.784] | **-0.043** [-0.070, -0.014] | NO |
| Onset F1 | `full` | `bpdr` | jitter | 0.679 [0.606, 0.750] | `tcn` | 0.685 [0.609, 0.756] | **-0.006** [-0.019, +0.007] | NO |
| Onset F1 | `full` | `bpdr` | jitter | 0.679 [0.606, 0.750] | `extractor` | 0.538 [0.504, 0.570] | **+0.141** [+0.079, +0.202] | YES |
| Onset F1 | `full` | `frame_only` | clean | 0.693 [0.619, 0.760] | `transformer_v1` | 0.719 [0.646, 0.784] | **-0.026** [-0.045, -0.004] | NO |
| Onset F1 | `full` | `frame_only` | clean | 0.693 [0.619, 0.760] | `transformer_v2` | 0.699 [0.631, 0.761] | **-0.006** [-0.031, +0.020] | NO |
| Onset F1 | `full` | `frame_only` | clean | 0.693 [0.619, 0.760] | `tcn` | 0.680 [0.607, 0.755] | **+0.012** [-0.002, +0.025] | NO |
| Onset F1 | `full` | `frame_only` | clean | 0.693 [0.619, 0.760] | `extractor` | 0.601 [0.556, 0.643] | **+0.092** [+0.040, +0.146] | YES |
| Onset F1 | `full` | `frame_only` | split | 0.693 [0.619, 0.760] | `transformer_v1` | 0.719 [0.646, 0.784] | **-0.026** [-0.045, -0.004] | NO |
| Onset F1 | `full` | `frame_only` | split | 0.693 [0.619, 0.760] | `transformer_v2` | 0.699 [0.631, 0.761] | **-0.006** [-0.031, +0.020] | NO |
| Onset F1 | `full` | `frame_only` | split | 0.693 [0.619, 0.760] | `tcn` | 0.680 [0.607, 0.755] | **+0.012** [-0.002, +0.025] | NO |
| Onset F1 | `full` | `frame_only` | split | 0.693 [0.619, 0.760] | `extractor` | 0.596 [0.555, 0.635] | **+0.097** [+0.047, +0.151] | YES |
| Onset F1 | `full` | `frame_only` | merge | 0.693 [0.619, 0.760] | `transformer_v1` | 0.719 [0.646, 0.784] | **-0.026** [-0.045, -0.004] | NO |
| Onset F1 | `full` | `frame_only` | merge | 0.693 [0.619, 0.760] | `transformer_v2` | 0.699 [0.631, 0.761] | **-0.006** [-0.031, +0.020] | NO |
| Onset F1 | `full` | `frame_only` | merge | 0.693 [0.619, 0.760] | `tcn` | 0.680 [0.607, 0.755] | **+0.012** [-0.002, +0.025] | NO |
| Onset F1 | `full` | `frame_only` | merge | 0.693 [0.619, 0.760] | `extractor` | 0.594 [0.551, 0.635] | **+0.099** [+0.043, +0.159] | YES |
| Onset F1 | `full` | `frame_only` | jitter | 0.693 [0.619, 0.760] | `transformer_v1` | 0.719 [0.646, 0.784] | **-0.026** [-0.045, -0.004] | NO |
| Onset F1 | `full` | `frame_only` | jitter | 0.693 [0.619, 0.760] | `transformer_v2` | 0.699 [0.631, 0.761] | **-0.006** [-0.031, +0.020] | NO |
| Onset F1 | `full` | `frame_only` | jitter | 0.693 [0.619, 0.760] | `tcn` | 0.680 [0.607, 0.755] | **+0.012** [-0.002, +0.025] | NO |
| Onset F1 | `full` | `frame_only` | jitter | 0.693 [0.619, 0.760] | `extractor` | 0.538 [0.504, 0.570] | **+0.154** [+0.094, +0.212] | YES |
| Onset F1 | `lean` | `augmentation_only` | clean | 0.670 [0.600, 0.737] | `transformer_v1` | 0.705 [0.626, 0.773] | **-0.035** [-0.060, -0.006] | NO |
| Onset F1 | `lean` | `augmentation_only` | clean | 0.670 [0.600, 0.737] | `transformer_v2` | 0.704 [0.630, 0.766] | **-0.034** [-0.063, -0.007] | NO |
| Onset F1 | `lean` | `augmentation_only` | clean | 0.670 [0.600, 0.737] | `tcn` | 0.692 [0.621, 0.756] | **-0.023** [-0.038, -0.008] | NO |
| Onset F1 | `lean` | `augmentation_only` | clean | 0.670 [0.600, 0.737] | `extractor` | 0.601 [0.556, 0.643] | **+0.069** [+0.017, +0.124] | YES |
| Onset F1 | `lean` | `augmentation_only` | split | 0.673 [0.603, 0.741] | `transformer_v1` | 0.704 [0.626, 0.772] | **-0.031** [-0.056, -0.000] | NO |
| Onset F1 | `lean` | `augmentation_only` | split | 0.673 [0.603, 0.741] | `transformer_v2` | 0.706 [0.633, 0.767] | **-0.032** [-0.061, -0.005] | NO |
| Onset F1 | `lean` | `augmentation_only` | split | 0.673 [0.603, 0.741] | `tcn` | 0.693 [0.622, 0.756] | **-0.020** [-0.037, -0.004] | NO |
| Onset F1 | `lean` | `augmentation_only` | split | 0.673 [0.603, 0.741] | `extractor` | 0.596 [0.555, 0.635] | **+0.077** [+0.026, +0.132] | YES |
| Onset F1 | `lean` | `augmentation_only` | merge | 0.670 [0.598, 0.737] | `transformer_v1` | 0.707 [0.629, 0.775] | **-0.037** [-0.062, -0.006] | NO |
| Onset F1 | `lean` | `augmentation_only` | merge | 0.670 [0.598, 0.737] | `transformer_v2` | 0.703 [0.630, 0.765] | **-0.033** [-0.062, -0.005] | NO |
| Onset F1 | `lean` | `augmentation_only` | merge | 0.670 [0.598, 0.737] | `tcn` | 0.691 [0.620, 0.755] | **-0.021** [-0.037, -0.007] | NO |
| Onset F1 | `lean` | `augmentation_only` | merge | 0.670 [0.598, 0.737] | `extractor` | 0.594 [0.551, 0.635] | **+0.076** [+0.018, +0.136] | YES |
| Onset F1 | `lean` | `augmentation_only` | jitter | 0.670 [0.600, 0.737] | `transformer_v1` | 0.708 [0.629, 0.777] | **-0.038** [-0.064, -0.007] | NO |
| Onset F1 | `lean` | `augmentation_only` | jitter | 0.670 [0.600, 0.737] | `transformer_v2` | 0.701 [0.630, 0.762] | **-0.031** [-0.061, -0.004] | NO |
| Onset F1 | `lean` | `augmentation_only` | jitter | 0.670 [0.600, 0.737] | `tcn` | 0.686 [0.614, 0.753] | **-0.016** [-0.033, -0.001] | NO |
| Onset F1 | `lean` | `augmentation_only` | jitter | 0.670 [0.600, 0.737] | `extractor` | 0.538 [0.504, 0.570] | **+0.132** [+0.072, +0.190] | YES |
| Onset F1 | `lean` | `bpdr` | clean | 0.683 [0.613, 0.749] | `transformer_v1` | 0.706 [0.625, 0.777] | **-0.023** [-0.043, -0.002] | NO |
| Onset F1 | `lean` | `bpdr` | clean | 0.683 [0.613, 0.749] | `transformer_v2` | 0.701 [0.630, 0.767] | **-0.018** [-0.043, +0.007] | NO |
| Onset F1 | `lean` | `bpdr` | clean | 0.683 [0.613, 0.749] | `tcn` | 0.688 [0.617, 0.754] | **-0.005** [-0.014, +0.004] | NO |
| Onset F1 | `lean` | `bpdr` | clean | 0.683 [0.613, 0.749] | `extractor` | 0.601 [0.556, 0.643] | **+0.082** [+0.033, +0.135] | YES |
| Onset F1 | `lean` | `bpdr` | split | 0.685 [0.615, 0.749] | `transformer_v1` | 0.710 [0.632, 0.779] | **-0.025** [-0.045, -0.004] | NO |
| Onset F1 | `lean` | `bpdr` | split | 0.685 [0.615, 0.749] | `transformer_v2` | 0.702 [0.630, 0.766] | **-0.017** [-0.041, +0.008] | NO |
| Onset F1 | `lean` | `bpdr` | split | 0.685 [0.615, 0.749] | `tcn` | 0.688 [0.619, 0.755] | **-0.003** [-0.012, +0.005] | NO |
| Onset F1 | `lean` | `bpdr` | split | 0.685 [0.615, 0.749] | `extractor` | 0.596 [0.555, 0.635] | **+0.089** [+0.042, +0.140] | YES |
| Onset F1 | `lean` | `bpdr` | merge | 0.683 [0.613, 0.749] | `transformer_v1` | 0.706 [0.625, 0.778] | **-0.023** [-0.042, -0.002] | NO |
| Onset F1 | `lean` | `bpdr` | merge | 0.683 [0.613, 0.749] | `transformer_v2` | 0.702 [0.631, 0.767] | **-0.019** [-0.044, +0.006] | NO |
| Onset F1 | `lean` | `bpdr` | merge | 0.683 [0.613, 0.749] | `tcn` | 0.688 [0.619, 0.755] | **-0.005** [-0.015, +0.004] | NO |
| Onset F1 | `lean` | `bpdr` | merge | 0.683 [0.613, 0.749] | `extractor` | 0.594 [0.551, 0.635] | **+0.089** [+0.035, +0.146] | YES |
| Onset F1 | `lean` | `bpdr` | jitter | 0.683 [0.614, 0.749] | `transformer_v1` | 0.705 [0.624, 0.776] | **-0.022** [-0.041, -0.000] | NO |
| Onset F1 | `lean` | `bpdr` | jitter | 0.683 [0.614, 0.749] | `transformer_v2` | 0.701 [0.629, 0.767] | **-0.018** [-0.042, +0.007] | NO |
| Onset F1 | `lean` | `bpdr` | jitter | 0.683 [0.614, 0.749] | `tcn` | 0.684 [0.613, 0.753] | **-0.001** [-0.011, +0.007] | NO |
| Onset F1 | `lean` | `bpdr` | jitter | 0.683 [0.614, 0.749] | `extractor` | 0.538 [0.504, 0.570] | **+0.145** [+0.087, +0.201] | YES |
| Onset F1 | `lean` | `frame_only` | clean | 0.667 [0.600, 0.733] | `transformer_v1` | 0.713 [0.648, 0.770] | **-0.046** [-0.067, -0.020] | NO |
| Onset F1 | `lean` | `frame_only` | clean | 0.667 [0.600, 0.733] | `transformer_v2` | 0.690 [0.607, 0.757] | **-0.022** [-0.056, +0.017] | NO |
| Onset F1 | `lean` | `frame_only` | clean | 0.667 [0.600, 0.733] | `tcn` | 0.687 [0.616, 0.758] | **-0.020** [-0.037, -0.001] | NO |
| Onset F1 | `lean` | `frame_only` | clean | 0.667 [0.600, 0.733] | `extractor` | 0.601 [0.556, 0.643] | **+0.066** [+0.016, +0.122] | YES |
| Onset F1 | `lean` | `frame_only` | split | 0.667 [0.600, 0.733] | `transformer_v1` | 0.713 [0.648, 0.770] | **-0.046** [-0.067, -0.020] | NO |
| Onset F1 | `lean` | `frame_only` | split | 0.667 [0.600, 0.733] | `transformer_v2` | 0.690 [0.607, 0.757] | **-0.022** [-0.056, +0.017] | NO |
| Onset F1 | `lean` | `frame_only` | split | 0.667 [0.600, 0.733] | `tcn` | 0.687 [0.616, 0.758] | **-0.020** [-0.037, -0.001] | NO |
| Onset F1 | `lean` | `frame_only` | split | 0.667 [0.600, 0.733] | `extractor` | 0.596 [0.555, 0.635] | **+0.071** [+0.023, +0.125] | YES |
| Onset F1 | `lean` | `frame_only` | merge | 0.667 [0.600, 0.733] | `transformer_v1` | 0.713 [0.648, 0.770] | **-0.046** [-0.067, -0.020] | NO |
| Onset F1 | `lean` | `frame_only` | merge | 0.667 [0.600, 0.733] | `transformer_v2` | 0.690 [0.607, 0.757] | **-0.022** [-0.056, +0.017] | NO |
| Onset F1 | `lean` | `frame_only` | merge | 0.667 [0.600, 0.733] | `tcn` | 0.687 [0.616, 0.758] | **-0.020** [-0.037, -0.001] | NO |
| Onset F1 | `lean` | `frame_only` | merge | 0.667 [0.600, 0.733] | `extractor` | 0.594 [0.551, 0.635] | **+0.074** [+0.018, +0.134] | YES |
| Onset F1 | `lean` | `frame_only` | jitter | 0.667 [0.600, 0.733] | `transformer_v1` | 0.713 [0.648, 0.770] | **-0.046** [-0.067, -0.020] | NO |
| Onset F1 | `lean` | `frame_only` | jitter | 0.667 [0.600, 0.733] | `transformer_v2` | 0.690 [0.607, 0.757] | **-0.022** [-0.056, +0.017] | NO |
| Onset F1 | `lean` | `frame_only` | jitter | 0.667 [0.600, 0.733] | `tcn` | 0.687 [0.616, 0.758] | **-0.020** [-0.037, -0.001] | NO |
| Onset F1 | `lean` | `frame_only` | jitter | 0.667 [0.600, 0.733] | `extractor` | 0.538 [0.504, 0.570] | **+0.129** [+0.070, +0.188] | YES |

### `dev_M02`: Transformer v3 vs Baselines (Boundary Onset F1, 64 Clips)

| Feat | Arm | Cond | Trans v3 (Mean [CI]) | Baseline | Baseline (Mean [CI]) | Diff (v3 - Base) [95% CI] | Win (CI > 0)? |
|---|---|---|---|---|---|---|---|
| `full` | `augmentation_only` | clean | 0.792 [0.765, 0.819] | `transformer_v1` | 0.762 [0.734, 0.789] | **+0.030** [+0.018, +0.042] | YES |
| `full` | `augmentation_only` | clean | 0.792 [0.765, 0.819] | `transformer_v2` | 0.812 [0.785, 0.837] | **-0.019** [-0.028, -0.011] | NO |
| `full` | `augmentation_only` | clean | 0.792 [0.765, 0.819] | `tcn` | 0.781 [0.754, 0.807] | **+0.011** [+0.003, +0.020] | YES |
| `full` | `augmentation_only` | clean | 0.792 [0.765, 0.819] | `extractor` | 0.548 [0.529, 0.564] | **+0.244** [+0.220, +0.270] | YES |
| `full` | `augmentation_only` | split | 0.793 [0.766, 0.821] | `transformer_v1` | 0.760 [0.732, 0.785] | **+0.034** [+0.021, +0.046] | YES |
| `full` | `augmentation_only` | split | 0.793 [0.766, 0.821] | `transformer_v2` | 0.812 [0.785, 0.837] | **-0.019** [-0.027, -0.011] | NO |
| `full` | `augmentation_only` | split | 0.793 [0.766, 0.821] | `tcn` | 0.781 [0.754, 0.808] | **+0.013** [+0.004, +0.021] | YES |
| `full` | `augmentation_only` | split | 0.793 [0.766, 0.821] | `extractor` | 0.537 [0.519, 0.554] | **+0.256** [+0.232, +0.281] | YES |
| `full` | `augmentation_only` | merge | 0.793 [0.766, 0.820] | `transformer_v1` | 0.762 [0.735, 0.790] | **+0.031** [+0.018, +0.042] | YES |
| `full` | `augmentation_only` | merge | 0.793 [0.766, 0.820] | `transformer_v2` | 0.811 [0.784, 0.837] | **-0.018** [-0.027, -0.010] | NO |
| `full` | `augmentation_only` | merge | 0.793 [0.766, 0.820] | `tcn` | 0.781 [0.754, 0.808] | **+0.012** [+0.004, +0.020] | YES |
| `full` | `augmentation_only` | merge | 0.793 [0.766, 0.820] | `extractor` | 0.541 [0.523, 0.559] | **+0.252** [+0.227, +0.278] | YES |
| `full` | `augmentation_only` | jitter | 0.790 [0.763, 0.818] | `transformer_v1` | 0.759 [0.730, 0.785] | **+0.032** [+0.021, +0.044] | YES |
| `full` | `augmentation_only` | jitter | 0.790 [0.763, 0.818] | `transformer_v2` | 0.812 [0.785, 0.838] | **-0.022** [-0.030, -0.013] | NO |
| `full` | `augmentation_only` | jitter | 0.790 [0.763, 0.818] | `tcn` | 0.784 [0.757, 0.811] | **+0.007** [-0.002, +0.015] | NO |
| `full` | `augmentation_only` | jitter | 0.790 [0.763, 0.818] | `extractor` | 0.526 [0.503, 0.545] | **+0.264** [+0.241, +0.289] | YES |
| `full` | `bpdr` | clean | 0.794 [0.766, 0.822] | `transformer_v1` | 0.767 [0.738, 0.796] | **+0.026** [+0.012, +0.040] | YES |
| `full` | `bpdr` | clean | 0.794 [0.766, 0.822] | `transformer_v2` | 0.812 [0.786, 0.838] | **-0.018** [-0.029, -0.008] | NO |
| `full` | `bpdr` | clean | 0.794 [0.766, 0.822] | `tcn` | 0.782 [0.754, 0.809] | **+0.012** [+0.003, +0.020] | YES |
| `full` | `bpdr` | clean | 0.794 [0.766, 0.822] | `extractor` | 0.548 [0.529, 0.564] | **+0.246** [+0.222, +0.271] | YES |
| `full` | `bpdr` | split | 0.794 [0.766, 0.822] | `transformer_v1` | 0.767 [0.738, 0.795] | **+0.026** [+0.013, +0.040] | YES |
| `full` | `bpdr` | split | 0.794 [0.766, 0.822] | `transformer_v2` | 0.812 [0.786, 0.838] | **-0.018** [-0.029, -0.008] | NO |
| `full` | `bpdr` | split | 0.794 [0.766, 0.822] | `tcn` | 0.782 [0.754, 0.809] | **+0.012** [+0.003, +0.020] | YES |
| `full` | `bpdr` | split | 0.794 [0.766, 0.822] | `extractor` | 0.537 [0.519, 0.554] | **+0.256** [+0.233, +0.282] | YES |
| `full` | `bpdr` | merge | 0.793 [0.766, 0.822] | `transformer_v1` | 0.767 [0.737, 0.795] | **+0.026** [+0.012, +0.040] | YES |
| `full` | `bpdr` | merge | 0.793 [0.766, 0.822] | `transformer_v2` | 0.812 [0.786, 0.838] | **-0.018** [-0.030, -0.008] | NO |
| `full` | `bpdr` | merge | 0.793 [0.766, 0.822] | `tcn` | 0.782 [0.754, 0.809] | **+0.012** [+0.003, +0.020] | YES |
| `full` | `bpdr` | merge | 0.793 [0.766, 0.822] | `extractor` | 0.541 [0.523, 0.559] | **+0.252** [+0.227, +0.280] | YES |
| `full` | `bpdr` | jitter | 0.794 [0.767, 0.822] | `transformer_v1` | 0.767 [0.737, 0.795] | **+0.028** [+0.015, +0.041] | YES |
| `full` | `bpdr` | jitter | 0.794 [0.767, 0.822] | `transformer_v2` | 0.812 [0.787, 0.838] | **-0.018** [-0.029, -0.008] | NO |
| `full` | `bpdr` | jitter | 0.794 [0.767, 0.822] | `tcn` | 0.781 [0.753, 0.808] | **+0.013** [+0.005, +0.022] | YES |
| `full` | `bpdr` | jitter | 0.794 [0.767, 0.822] | `extractor` | 0.526 [0.503, 0.545] | **+0.268** [+0.244, +0.294] | YES |
| `full` | `frame_only` | clean | 0.788 [0.762, 0.815] | `transformer_v1` | 0.754 [0.724, 0.782] | **+0.035** [+0.023, +0.047] | YES |
| `full` | `frame_only` | clean | 0.788 [0.762, 0.815] | `transformer_v2` | 0.807 [0.779, 0.834] | **-0.019** [-0.030, -0.008] | NO |
| `full` | `frame_only` | clean | 0.788 [0.762, 0.815] | `tcn` | 0.780 [0.752, 0.809] | **+0.008** [-0.001, +0.018] | NO |
| `full` | `frame_only` | clean | 0.788 [0.762, 0.815] | `extractor` | 0.548 [0.529, 0.564] | **+0.241** [+0.216, +0.266] | YES |
| `full` | `frame_only` | split | 0.788 [0.762, 0.815] | `transformer_v1` | 0.754 [0.724, 0.782] | **+0.035** [+0.023, +0.047] | YES |
| `full` | `frame_only` | split | 0.788 [0.762, 0.815] | `transformer_v2` | 0.807 [0.779, 0.834] | **-0.019** [-0.030, -0.008] | NO |
| `full` | `frame_only` | split | 0.788 [0.762, 0.815] | `tcn` | 0.780 [0.752, 0.809] | **+0.008** [-0.001, +0.018] | NO |
| `full` | `frame_only` | split | 0.788 [0.762, 0.815] | `extractor` | 0.537 [0.519, 0.554] | **+0.251** [+0.227, +0.276] | YES |
| `full` | `frame_only` | merge | 0.788 [0.762, 0.815] | `transformer_v1` | 0.754 [0.724, 0.782] | **+0.035** [+0.023, +0.047] | YES |
| `full` | `frame_only` | merge | 0.788 [0.762, 0.815] | `transformer_v2` | 0.807 [0.779, 0.834] | **-0.019** [-0.030, -0.008] | NO |
| `full` | `frame_only` | merge | 0.788 [0.762, 0.815] | `tcn` | 0.780 [0.752, 0.809] | **+0.008** [-0.001, +0.018] | NO |
| `full` | `frame_only` | merge | 0.788 [0.762, 0.815] | `extractor` | 0.541 [0.523, 0.559] | **+0.247** [+0.224, +0.274] | YES |
| `full` | `frame_only` | jitter | 0.788 [0.762, 0.815] | `transformer_v1` | 0.754 [0.724, 0.782] | **+0.035** [+0.023, +0.047] | YES |
| `full` | `frame_only` | jitter | 0.788 [0.762, 0.815] | `transformer_v2` | 0.807 [0.779, 0.834] | **-0.019** [-0.030, -0.008] | NO |
| `full` | `frame_only` | jitter | 0.788 [0.762, 0.815] | `tcn` | 0.780 [0.752, 0.809] | **+0.008** [-0.001, +0.018] | NO |
| `full` | `frame_only` | jitter | 0.788 [0.762, 0.815] | `extractor` | 0.526 [0.503, 0.545] | **+0.262** [+0.239, +0.288] | YES |
| `lean` | `augmentation_only` | clean | 0.792 [0.765, 0.819] | `transformer_v1` | 0.765 [0.736, 0.793] | **+0.027** [+0.017, +0.036] | YES |
| `lean` | `augmentation_only` | clean | 0.792 [0.765, 0.819] | `transformer_v2` | 0.807 [0.780, 0.834] | **-0.015** [-0.025, -0.005] | NO |
| `lean` | `augmentation_only` | clean | 0.792 [0.765, 0.819] | `tcn` | 0.783 [0.756, 0.810] | **+0.009** [+0.002, +0.018] | YES |
| `lean` | `augmentation_only` | clean | 0.792 [0.765, 0.819] | `extractor` | 0.548 [0.529, 0.564] | **+0.244** [+0.220, +0.269] | YES |
| `lean` | `augmentation_only` | split | 0.792 [0.765, 0.819] | `transformer_v1` | 0.764 [0.734, 0.791] | **+0.028** [+0.018, +0.038] | YES |
| `lean` | `augmentation_only` | split | 0.792 [0.765, 0.819] | `transformer_v2` | 0.807 [0.780, 0.833] | **-0.015** [-0.025, -0.006] | NO |
| `lean` | `augmentation_only` | split | 0.792 [0.765, 0.819] | `tcn` | 0.782 [0.755, 0.809] | **+0.010** [+0.002, +0.019] | YES |
| `lean` | `augmentation_only` | split | 0.792 [0.765, 0.819] | `extractor` | 0.537 [0.519, 0.554] | **+0.254** [+0.229, +0.279] | YES |
| `lean` | `augmentation_only` | merge | 0.792 [0.764, 0.819] | `transformer_v1` | 0.766 [0.736, 0.794] | **+0.026** [+0.016, +0.035] | YES |
| `lean` | `augmentation_only` | merge | 0.792 [0.764, 0.819] | `transformer_v2` | 0.808 [0.780, 0.834] | **-0.016** [-0.027, -0.006] | NO |
| `lean` | `augmentation_only` | merge | 0.792 [0.764, 0.819] | `tcn` | 0.782 [0.754, 0.809] | **+0.010** [+0.002, +0.018] | YES |
| `lean` | `augmentation_only` | merge | 0.792 [0.764, 0.819] | `extractor` | 0.541 [0.523, 0.559] | **+0.251** [+0.226, +0.279] | YES |
| `lean` | `augmentation_only` | jitter | 0.792 [0.764, 0.821] | `transformer_v1` | 0.764 [0.735, 0.791] | **+0.028** [+0.019, +0.038] | YES |
| `lean` | `augmentation_only` | jitter | 0.792 [0.764, 0.821] | `transformer_v2` | 0.807 [0.780, 0.833] | **-0.016** [-0.026, -0.005] | NO |
| `lean` | `augmentation_only` | jitter | 0.792 [0.764, 0.821] | `tcn` | 0.784 [0.756, 0.811] | **+0.008** [-0.001, +0.018] | NO |
| `lean` | `augmentation_only` | jitter | 0.792 [0.764, 0.821] | `extractor` | 0.526 [0.503, 0.545] | **+0.266** [+0.241, +0.292] | YES |
| `lean` | `bpdr` | clean | 0.791 [0.764, 0.819] | `transformer_v1` | 0.763 [0.734, 0.790] | **+0.028** [+0.018, +0.040] | YES |
| `lean` | `bpdr` | clean | 0.791 [0.764, 0.819] | `transformer_v2` | 0.814 [0.788, 0.840] | **-0.023** [-0.033, -0.013] | NO |
| `lean` | `bpdr` | clean | 0.791 [0.764, 0.819] | `tcn` | 0.781 [0.754, 0.808] | **+0.010** [+0.002, +0.017] | YES |
| `lean` | `bpdr` | clean | 0.791 [0.764, 0.819] | `extractor` | 0.548 [0.529, 0.564] | **+0.243** [+0.220, +0.267] | YES |
| `lean` | `bpdr` | split | 0.791 [0.764, 0.819] | `transformer_v1` | 0.763 [0.734, 0.790] | **+0.028** [+0.018, +0.040] | YES |
| `lean` | `bpdr` | split | 0.791 [0.764, 0.819] | `transformer_v2` | 0.814 [0.788, 0.840] | **-0.023** [-0.033, -0.013] | NO |
| `lean` | `bpdr` | split | 0.791 [0.764, 0.819] | `tcn` | 0.782 [0.754, 0.809] | **+0.010** [+0.002, +0.017] | YES |
| `lean` | `bpdr` | split | 0.791 [0.764, 0.819] | `extractor` | 0.537 [0.519, 0.554] | **+0.254** [+0.230, +0.278] | YES |
| `lean` | `bpdr` | merge | 0.792 [0.764, 0.819] | `transformer_v1` | 0.763 [0.734, 0.790] | **+0.029** [+0.018, +0.041] | YES |
| `lean` | `bpdr` | merge | 0.792 [0.764, 0.819] | `transformer_v2` | 0.813 [0.787, 0.839] | **-0.021** [-0.032, -0.012] | NO |
| `lean` | `bpdr` | merge | 0.792 [0.764, 0.819] | `tcn` | 0.782 [0.754, 0.808] | **+0.010** [+0.002, +0.017] | YES |
| `lean` | `bpdr` | merge | 0.792 [0.764, 0.819] | `extractor` | 0.541 [0.523, 0.559] | **+0.251** [+0.225, +0.277] | YES |
| `lean` | `bpdr` | jitter | 0.792 [0.764, 0.820] | `transformer_v1` | 0.764 [0.735, 0.791] | **+0.027** [+0.017, +0.038] | YES |
| `lean` | `bpdr` | jitter | 0.792 [0.764, 0.820] | `transformer_v2` | 0.815 [0.789, 0.841] | **-0.023** [-0.035, -0.013] | NO |
| `lean` | `bpdr` | jitter | 0.792 [0.764, 0.820] | `tcn` | 0.782 [0.754, 0.809] | **+0.009** [+0.002, +0.017] | YES |
| `lean` | `bpdr` | jitter | 0.792 [0.764, 0.820] | `extractor` | 0.526 [0.503, 0.545] | **+0.265** [+0.241, +0.290] | YES |
| `lean` | `frame_only` | clean | 0.797 [0.769, 0.825] | `transformer_v1` | 0.741 [0.714, 0.767] | **+0.056** [+0.044, +0.067] | YES |
| `lean` | `frame_only` | clean | 0.797 [0.769, 0.825] | `transformer_v2` | 0.811 [0.785, 0.836] | **-0.015** [-0.025, -0.005] | NO |
| `lean` | `frame_only` | clean | 0.797 [0.769, 0.825] | `tcn` | 0.781 [0.752, 0.808] | **+0.016** [+0.008, +0.024] | YES |
| `lean` | `frame_only` | clean | 0.797 [0.769, 0.825] | `extractor` | 0.548 [0.529, 0.564] | **+0.249** [+0.223, +0.276] | YES |
| `lean` | `frame_only` | split | 0.797 [0.769, 0.825] | `transformer_v1` | 0.741 [0.714, 0.767] | **+0.056** [+0.044, +0.067] | YES |
| `lean` | `frame_only` | split | 0.797 [0.769, 0.825] | `transformer_v2` | 0.811 [0.785, 0.836] | **-0.015** [-0.025, -0.005] | NO |
| `lean` | `frame_only` | split | 0.797 [0.769, 0.825] | `tcn` | 0.781 [0.752, 0.808] | **+0.016** [+0.008, +0.024] | YES |
| `lean` | `frame_only` | split | 0.797 [0.769, 0.825] | `extractor` | 0.537 [0.519, 0.554] | **+0.259** [+0.234, +0.286] | YES |
| `lean` | `frame_only` | merge | 0.797 [0.769, 0.825] | `transformer_v1` | 0.741 [0.714, 0.767] | **+0.056** [+0.044, +0.067] | YES |
| `lean` | `frame_only` | merge | 0.797 [0.769, 0.825] | `transformer_v2` | 0.811 [0.785, 0.836] | **-0.015** [-0.025, -0.005] | NO |
| `lean` | `frame_only` | merge | 0.797 [0.769, 0.825] | `tcn` | 0.781 [0.752, 0.808] | **+0.016** [+0.008, +0.024] | YES |
| `lean` | `frame_only` | merge | 0.797 [0.769, 0.825] | `extractor` | 0.541 [0.523, 0.559] | **+0.256** [+0.230, +0.284] | YES |
| `lean` | `frame_only` | jitter | 0.797 [0.769, 0.825] | `transformer_v1` | 0.741 [0.714, 0.767] | **+0.056** [+0.044, +0.067] | YES |
| `lean` | `frame_only` | jitter | 0.797 [0.769, 0.825] | `transformer_v2` | 0.811 [0.785, 0.836] | **-0.015** [-0.025, -0.005] | NO |
| `lean` | `frame_only` | jitter | 0.797 [0.769, 0.825] | `tcn` | 0.781 [0.752, 0.808] | **+0.016** [+0.008, +0.024] | YES |
| `lean` | `frame_only` | jitter | 0.797 [0.769, 0.825] | `extractor` | 0.526 [0.503, 0.545] | **+0.271** [+0.246, +0.297] | YES |

### Transformer v3 Fit Threshold & Seed Coverage

| Arch | Feat | Arm | Seeds Run | Fits | Healthy | Thresholds (`thr`) |
|---|---|---|---|---|---|---|
| `transformer_v3` | `full` | `augmentation_only` | `339, 340` | 2 | 2/2 | `0.6, 0.8` |
| `transformer_v3` | `full` | `bpdr` | `339, 340` | 2 | 2/2 | `0.6, 0.8` |
| `transformer_v3` | `full` | `frame_only` | `339, 340` | 2 | 2/2 | `0.65, 0.8` |
| `transformer_v3` | `lean` | `augmentation_only` | `339, 340` | 2 | 2/2 | `0.7, 0.85` |
| `transformer_v3` | `lean` | `bpdr` | `339, 340` | 2 | 2/2 | `0.7, 0.75` |
| `transformer_v3` | `lean` | `frame_only` | `339, 340` | 2 | 2/2 | `0.8` |

- **Seed coverage**: `transformer_v3`: 2 seed(s) ([339, 340]).
- **Threshold distribution**: Observed `thr` values across 12 fits: `0.6, 0.65, 0.7, 0.75, 0.8, 0.85`.

- **v3 Adoption Bar Verdict**: **DO NOT ADOPT v3 (keep V1_CHECKPOINT as Studio default)**.

## Transformer v1 Replication (Seed 340) Statistical Comparison (vs v1, TCN, and Extractor)

**Methodology**: Vocadito uses singer-macro (14 singers) 95% bootstrap CIs (1,000 resamples); `dev_M02` uses clip-level (64 clips) 95% bootstrap CIs.

### Vocadito: Transformer v1 Replication (Seed 340) vs Baselines (Note F1 & Onset F1)

| Metric | Feat | Arm | Cond | Trans v1_340 (Mean [CI]) | Baseline | Baseline (Mean [CI]) | Diff (v1_340 - Base) [95% CI] | Win (CI > 0)? |
|---|---|---|---|---|---|---|---|---|
| Note F1 | `full` | `bpdr` | clean | 0.561 [0.502, 0.619] | `transformer_v1` | 0.577 [0.510, 0.639] | **-0.016** [-0.038, +0.007] | NO |
| Note F1 | `full` | `bpdr` | clean | 0.561 [0.502, 0.619] | `tcn` | 0.535 [0.460, 0.609] | **+0.026** [-0.002, +0.057] | NO |
| Note F1 | `full` | `bpdr` | clean | 0.561 [0.502, 0.619] | `extractor` | 0.562 [0.489, 0.635] | **-0.001** [-0.027, +0.024] | NO |
| Note F1 | `full` | `bpdr` | split | 0.561 [0.502, 0.620] | `transformer_v1` | 0.580 [0.514, 0.642] | **-0.019** [-0.040, +0.003] | NO |
| Note F1 | `full` | `bpdr` | split | 0.561 [0.502, 0.620] | `tcn` | 0.537 [0.464, 0.609] | **+0.024** [-0.002, +0.053] | NO |
| Note F1 | `full` | `bpdr` | split | 0.561 [0.502, 0.620] | `extractor` | 0.559 [0.488, 0.629] | **+0.003** [-0.018, +0.025] | NO |
| Note F1 | `full` | `bpdr` | merge | 0.560 [0.501, 0.619] | `transformer_v1` | 0.576 [0.510, 0.638] | **-0.016** [-0.037, +0.007] | NO |
| Note F1 | `full` | `bpdr` | merge | 0.560 [0.501, 0.619] | `tcn` | 0.535 [0.460, 0.609] | **+0.025** [-0.003, +0.056] | NO |
| Note F1 | `full` | `bpdr` | merge | 0.560 [0.501, 0.619] | `extractor` | 0.554 [0.486, 0.617] | **+0.007** [-0.013, +0.028] | NO |
| Note F1 | `full` | `bpdr` | jitter | 0.563 [0.503, 0.621] | `transformer_v1` | 0.582 [0.517, 0.643] | **-0.019** [-0.037, +0.001] | NO |
| Note F1 | `full` | `bpdr` | jitter | 0.563 [0.503, 0.621] | `tcn` | 0.529 [0.456, 0.604] | **+0.034** [+0.007, +0.067] | YES |
| Note F1 | `full` | `bpdr` | jitter | 0.563 [0.503, 0.621] | `extractor` | 0.479 [0.406, 0.552] | **+0.084** [+0.049, +0.117] | YES |
| Note F1 | `full` | `frame_only` | clean | 0.579 [0.516, 0.646] | `transformer_v1` | 0.601 [0.538, 0.665] | **-0.022** [-0.035, -0.009] | NO |
| Note F1 | `full` | `frame_only` | clean | 0.579 [0.516, 0.646] | `tcn` | 0.522 [0.449, 0.595] | **+0.057** [+0.027, +0.087] | YES |
| Note F1 | `full` | `frame_only` | clean | 0.579 [0.516, 0.646] | `extractor` | 0.562 [0.489, 0.635] | **+0.017** [-0.014, +0.047] | NO |
| Note F1 | `full` | `frame_only` | split | 0.579 [0.516, 0.646] | `transformer_v1` | 0.601 [0.538, 0.665] | **-0.022** [-0.035, -0.009] | NO |
| Note F1 | `full` | `frame_only` | split | 0.579 [0.516, 0.646] | `tcn` | 0.522 [0.449, 0.595] | **+0.057** [+0.027, +0.087] | YES |
| Note F1 | `full` | `frame_only` | split | 0.579 [0.516, 0.646] | `extractor` | 0.559 [0.488, 0.629] | **+0.020** [-0.007, +0.049] | NO |
| Note F1 | `full` | `frame_only` | merge | 0.579 [0.516, 0.646] | `transformer_v1` | 0.601 [0.538, 0.665] | **-0.022** [-0.035, -0.009] | NO |
| Note F1 | `full` | `frame_only` | merge | 0.579 [0.516, 0.646] | `tcn` | 0.522 [0.449, 0.595] | **+0.057** [+0.027, +0.087] | YES |
| Note F1 | `full` | `frame_only` | merge | 0.579 [0.516, 0.646] | `extractor` | 0.554 [0.486, 0.617] | **+0.025** [+0.002, +0.053] | YES |
| Note F1 | `full` | `frame_only` | jitter | 0.579 [0.516, 0.646] | `transformer_v1` | 0.601 [0.538, 0.665] | **-0.022** [-0.035, -0.009] | NO |
| Note F1 | `full` | `frame_only` | jitter | 0.579 [0.516, 0.646] | `tcn` | 0.522 [0.449, 0.595] | **+0.057** [+0.027, +0.087] | YES |
| Note F1 | `full` | `frame_only` | jitter | 0.579 [0.516, 0.646] | `extractor` | 0.479 [0.406, 0.552] | **+0.100** [+0.067, +0.129] | YES |
| Onset F1 | `full` | `bpdr` | clean | 0.692 [0.624, 0.753] | `transformer_v1` | 0.715 [0.639, 0.781] | **-0.023** [-0.047, +0.004] | NO |
| Onset F1 | `full` | `bpdr` | clean | 0.692 [0.624, 0.753] | `tcn` | 0.688 [0.614, 0.757] | **+0.004** [-0.019, +0.027] | NO |
| Onset F1 | `full` | `bpdr` | clean | 0.692 [0.624, 0.753] | `extractor` | 0.601 [0.556, 0.643] | **+0.091** [+0.047, +0.134] | YES |
| Onset F1 | `full` | `bpdr` | split | 0.693 [0.625, 0.753] | `transformer_v1` | 0.715 [0.640, 0.782] | **-0.022** [-0.047, +0.004] | NO |
| Onset F1 | `full` | `bpdr` | split | 0.693 [0.625, 0.753] | `tcn` | 0.690 [0.615, 0.758] | **+0.003** [-0.019, +0.024] | NO |
| Onset F1 | `full` | `bpdr` | split | 0.693 [0.625, 0.753] | `extractor` | 0.596 [0.555, 0.635] | **+0.097** [+0.056, +0.138] | YES |
| Onset F1 | `full` | `bpdr` | merge | 0.693 [0.625, 0.753] | `transformer_v1` | 0.714 [0.638, 0.780] | **-0.021** [-0.044, +0.003] | NO |
| Onset F1 | `full` | `bpdr` | merge | 0.693 [0.625, 0.753] | `tcn` | 0.688 [0.614, 0.757] | **+0.005** [-0.019, +0.028] | NO |
| Onset F1 | `full` | `bpdr` | merge | 0.693 [0.625, 0.753] | `extractor` | 0.594 [0.551, 0.635] | **+0.100** [+0.053, +0.145] | YES |
| Onset F1 | `full` | `bpdr` | jitter | 0.694 [0.626, 0.753] | `transformer_v1` | 0.715 [0.642, 0.780] | **-0.021** [-0.044, +0.004] | NO |
| Onset F1 | `full` | `bpdr` | jitter | 0.694 [0.626, 0.753] | `tcn` | 0.685 [0.609, 0.756] | **+0.009** [-0.015, +0.033] | NO |
| Onset F1 | `full` | `bpdr` | jitter | 0.694 [0.626, 0.753] | `extractor` | 0.538 [0.504, 0.570] | **+0.156** [+0.100, +0.207] | YES |
| Onset F1 | `full` | `frame_only` | clean | 0.714 [0.645, 0.779] | `transformer_v1` | 0.719 [0.646, 0.784] | **-0.005** [-0.018, +0.008] | NO |
| Onset F1 | `full` | `frame_only` | clean | 0.714 [0.645, 0.779] | `tcn` | 0.680 [0.607, 0.755] | **+0.034** [+0.005, +0.058] | YES |
| Onset F1 | `full` | `frame_only` | clean | 0.714 [0.645, 0.779] | `extractor` | 0.601 [0.556, 0.643] | **+0.113** [+0.064, +0.163] | YES |
| Onset F1 | `full` | `frame_only` | split | 0.714 [0.645, 0.779] | `transformer_v1` | 0.719 [0.646, 0.784] | **-0.005** [-0.018, +0.008] | NO |
| Onset F1 | `full` | `frame_only` | split | 0.714 [0.645, 0.779] | `tcn` | 0.680 [0.607, 0.755] | **+0.034** [+0.005, +0.058] | YES |
| Onset F1 | `full` | `frame_only` | split | 0.714 [0.645, 0.779] | `extractor` | 0.596 [0.555, 0.635] | **+0.118** [+0.071, +0.165] | YES |
| Onset F1 | `full` | `frame_only` | merge | 0.714 [0.645, 0.779] | `transformer_v1` | 0.719 [0.646, 0.784] | **-0.005** [-0.018, +0.008] | NO |
| Onset F1 | `full` | `frame_only` | merge | 0.714 [0.645, 0.779] | `tcn` | 0.680 [0.607, 0.755] | **+0.034** [+0.005, +0.058] | YES |
| Onset F1 | `full` | `frame_only` | merge | 0.714 [0.645, 0.779] | `extractor` | 0.594 [0.551, 0.635] | **+0.120** [+0.068, +0.173] | YES |
| Onset F1 | `full` | `frame_only` | jitter | 0.714 [0.645, 0.779] | `transformer_v1` | 0.719 [0.646, 0.784] | **-0.005** [-0.018, +0.008] | NO |
| Onset F1 | `full` | `frame_only` | jitter | 0.714 [0.645, 0.779] | `tcn` | 0.680 [0.607, 0.755] | **+0.034** [+0.005, +0.058] | YES |
| Onset F1 | `full` | `frame_only` | jitter | 0.714 [0.645, 0.779] | `extractor` | 0.538 [0.504, 0.570] | **+0.175** [+0.114, +0.235] | YES |

### `dev_M02`: Transformer v1 Replication (Seed 340) vs Baselines (Boundary Onset F1, 64 Clips)

| Feat | Arm | Cond | Trans v1_340 (Mean [CI]) | Baseline | Baseline (Mean [CI]) | Diff (v1_340 - Base) [95% CI] | Win (CI > 0)? |
|---|---|---|---|---|---|---|---|
| `full` | `bpdr` | clean | 0.766 [0.739, 0.794] | `transformer_v1` | 0.767 [0.738, 0.796] | **-0.001** [-0.010, +0.009] | NO |
| `full` | `bpdr` | clean | 0.766 [0.739, 0.794] | `tcn` | 0.782 [0.754, 0.809] | **-0.015** [-0.025, -0.005] | NO |
| `full` | `bpdr` | clean | 0.766 [0.739, 0.794] | `extractor` | 0.548 [0.529, 0.564] | **+0.219** [+0.197, +0.241] | YES |
| `full` | `bpdr` | split | 0.766 [0.739, 0.794] | `transformer_v1` | 0.767 [0.738, 0.795] | **-0.001** [-0.010, +0.009] | NO |
| `full` | `bpdr` | split | 0.766 [0.739, 0.794] | `tcn` | 0.782 [0.754, 0.809] | **-0.015** [-0.025, -0.005] | NO |
| `full` | `bpdr` | split | 0.766 [0.739, 0.794] | `extractor` | 0.537 [0.519, 0.554] | **+0.229** [+0.208, +0.252] | YES |
| `full` | `bpdr` | merge | 0.767 [0.739, 0.794] | `transformer_v1` | 0.767 [0.737, 0.795] | **-0.000** [-0.010, +0.009] | NO |
| `full` | `bpdr` | merge | 0.767 [0.739, 0.794] | `tcn` | 0.782 [0.754, 0.809] | **-0.015** [-0.025, -0.005] | NO |
| `full` | `bpdr` | merge | 0.767 [0.739, 0.794] | `extractor` | 0.541 [0.523, 0.559] | **+0.226** [+0.204, +0.251] | YES |
| `full` | `bpdr` | jitter | 0.766 [0.738, 0.793] | `transformer_v1` | 0.767 [0.737, 0.795] | **-0.001** [-0.010, +0.008] | NO |
| `full` | `bpdr` | jitter | 0.766 [0.738, 0.793] | `tcn` | 0.781 [0.753, 0.808] | **-0.015** [-0.025, -0.005] | NO |
| `full` | `bpdr` | jitter | 0.766 [0.738, 0.793] | `extractor` | 0.526 [0.503, 0.545] | **+0.239** [+0.218, +0.263] | YES |
| `full` | `frame_only` | clean | 0.758 [0.730, 0.785] | `transformer_v1` | 0.754 [0.724, 0.782] | **+0.004** [-0.005, +0.014] | NO |
| `full` | `frame_only` | clean | 0.758 [0.730, 0.785] | `tcn` | 0.780 [0.752, 0.809] | **-0.022** [-0.034, -0.010] | NO |
| `full` | `frame_only` | clean | 0.758 [0.730, 0.785] | `extractor` | 0.548 [0.529, 0.564] | **+0.210** [+0.189, +0.234] | YES |
| `full` | `frame_only` | split | 0.758 [0.730, 0.785] | `transformer_v1` | 0.754 [0.724, 0.782] | **+0.004** [-0.005, +0.014] | NO |
| `full` | `frame_only` | split | 0.758 [0.730, 0.785] | `tcn` | 0.780 [0.752, 0.809] | **-0.022** [-0.034, -0.010] | NO |
| `full` | `frame_only` | split | 0.758 [0.730, 0.785] | `extractor` | 0.537 [0.519, 0.554] | **+0.221** [+0.200, +0.245] | YES |
| `full` | `frame_only` | merge | 0.758 [0.730, 0.785] | `transformer_v1` | 0.754 [0.724, 0.782] | **+0.004** [-0.005, +0.014] | NO |
| `full` | `frame_only` | merge | 0.758 [0.730, 0.785] | `tcn` | 0.780 [0.752, 0.809] | **-0.022** [-0.034, -0.010] | NO |
| `full` | `frame_only` | merge | 0.758 [0.730, 0.785] | `extractor` | 0.541 [0.523, 0.559] | **+0.217** [+0.194, +0.244] | YES |
| `full` | `frame_only` | jitter | 0.758 [0.730, 0.785] | `transformer_v1` | 0.754 [0.724, 0.782] | **+0.004** [-0.005, +0.014] | NO |
| `full` | `frame_only` | jitter | 0.758 [0.730, 0.785] | `tcn` | 0.780 [0.752, 0.809] | **-0.022** [-0.034, -0.010] | NO |
| `full` | `frame_only` | jitter | 0.758 [0.730, 0.785] | `extractor` | 0.526 [0.503, 0.545] | **+0.232** [+0.209, +0.258] | YES |

### Transformer v1 Replication (Seed 340) Fit Threshold & Seed Coverage

| Arch | Feat | Arm | Seeds Run | Fits | Healthy | Thresholds (`thr`) |
|---|---|---|---|---|---|---|
| `v1_rep_340` | `full` | `bpdr` | `340` | 1 | 1/1 | `0.8` |
| `v1_rep_340` | `full` | `frame_only` | `340` | 1 | 1/1 | `0.8` |

- **Seed coverage**: `v1_rep_340`: 1 seed(s) ([340]).
- **Threshold saturation**: All 2/2 fits hit `thr = 0.8` (the upper ceiling of the candidate threshold grid), indicating uncalibrated/shifted eval logits.

## Transformer v4 (POS_WEIGHT=10.0, offset=0.0) Statistical Comparison (vs v1, transformer_v1_2seed, transformer_v2, TCN, and Extractor)

**Methodology**: Vocadito uses singer-macro (14 singers) 95% bootstrap CIs (1,000 resamples); `dev_M02` uses clip-level (64 clips) 95% bootstrap CIs.

### Vocadito: Transformer v4 (POS_WEIGHT=10.0, offset=0.0) vs Baselines (Note F1 & Onset F1)

| Metric | Feat | Arm | Cond | Trans v4 (Mean [CI]) | Baseline | Baseline (Mean [CI]) | Diff (v4 - Base) [95% CI] | Win (CI > 0)? |
|---|---|---|---|---|---|---|---|---|
| Note F1 | `full` | `bpdr` | clean | 0.515 [0.449, 0.581] | `transformer_v1` | 0.577 [0.510, 0.639] | **-0.062** [-0.101, -0.024] | NO |
| Note F1 | `full` | `bpdr` | clean | 0.515 [0.449, 0.581] | `transformer_v1_2seed` | 0.569 [0.507, 0.629] | **-0.054** [-0.091, -0.019] | NO |
| Note F1 | `full` | `bpdr` | clean | 0.515 [0.449, 0.581] | `transformer_v2` | 0.577 [0.522, 0.633] | **-0.062** [-0.095, -0.030] | NO |
| Note F1 | `full` | `bpdr` | clean | 0.515 [0.449, 0.581] | `tcn` | 0.535 [0.460, 0.609] | **-0.020** [-0.049, +0.005] | NO |
| Note F1 | `full` | `bpdr` | clean | 0.515 [0.449, 0.581] | `extractor` | 0.562 [0.489, 0.635] | **-0.047** [-0.089, -0.008] | NO |
| Note F1 | `full` | `bpdr` | split | 0.515 [0.449, 0.581] | `transformer_v1` | 0.580 [0.514, 0.642] | **-0.065** [-0.102, -0.028] | NO |
| Note F1 | `full` | `bpdr` | split | 0.515 [0.449, 0.581] | `transformer_v1_2seed` | 0.571 [0.509, 0.631] | **-0.055** [-0.091, -0.021] | NO |
| Note F1 | `full` | `bpdr` | split | 0.515 [0.449, 0.581] | `transformer_v2` | 0.576 [0.522, 0.633] | **-0.061** [-0.095, -0.030] | NO |
| Note F1 | `full` | `bpdr` | split | 0.515 [0.449, 0.581] | `tcn` | 0.537 [0.464, 0.609] | **-0.021** [-0.051, +0.003] | NO |
| Note F1 | `full` | `bpdr` | split | 0.515 [0.449, 0.581] | `extractor` | 0.559 [0.488, 0.629] | **-0.043** [-0.085, -0.004] | NO |
| Note F1 | `full` | `bpdr` | merge | 0.515 [0.449, 0.581] | `transformer_v1` | 0.576 [0.510, 0.638] | **-0.061** [-0.099, -0.024] | NO |
| Note F1 | `full` | `bpdr` | merge | 0.515 [0.449, 0.581] | `transformer_v1_2seed` | 0.568 [0.506, 0.629] | **-0.053** [-0.090, -0.018] | NO |
| Note F1 | `full` | `bpdr` | merge | 0.515 [0.449, 0.581] | `transformer_v2` | 0.577 [0.523, 0.634] | **-0.062** [-0.096, -0.031] | NO |
| Note F1 | `full` | `bpdr` | merge | 0.515 [0.449, 0.581] | `tcn` | 0.535 [0.460, 0.609] | **-0.020** [-0.049, +0.005] | NO |
| Note F1 | `full` | `bpdr` | merge | 0.515 [0.449, 0.581] | `extractor` | 0.554 [0.486, 0.617] | **-0.039** [-0.079, -0.001] | NO |
| Note F1 | `full` | `bpdr` | jitter | 0.517 [0.449, 0.582] | `transformer_v1` | 0.582 [0.517, 0.643] | **-0.065** [-0.099, -0.031] | NO |
| Note F1 | `full` | `bpdr` | jitter | 0.517 [0.449, 0.582] | `transformer_v1_2seed` | 0.572 [0.511, 0.632] | **-0.055** [-0.088, -0.023] | NO |
| Note F1 | `full` | `bpdr` | jitter | 0.517 [0.449, 0.582] | `transformer_v2` | 0.576 [0.522, 0.632] | **-0.059** [-0.091, -0.028] | NO |
| Note F1 | `full` | `bpdr` | jitter | 0.517 [0.449, 0.582] | `tcn` | 0.529 [0.456, 0.604] | **-0.012** [-0.035, +0.009] | NO |
| Note F1 | `full` | `bpdr` | jitter | 0.517 [0.449, 0.582] | `extractor` | 0.479 [0.406, 0.552] | **+0.038** [+0.007, +0.074] | YES |
| Note F1 | `full` | `frame_only` | clean | 0.544 [0.474, 0.608] | `transformer_v1` | 0.601 [0.538, 0.665] | **-0.057** [-0.089, -0.028] | NO |
| Note F1 | `full` | `frame_only` | clean | 0.544 [0.474, 0.608] | `transformer_v1_2seed` | 0.590 [0.528, 0.654] | **-0.046** [-0.075, -0.019] | NO |
| Note F1 | `full` | `frame_only` | clean | 0.544 [0.474, 0.608] | `transformer_v2` | 0.563 [0.502, 0.622] | **-0.019** [-0.059, +0.014] | NO |
| Note F1 | `full` | `frame_only` | clean | 0.544 [0.474, 0.608] | `tcn` | 0.522 [0.449, 0.595] | **+0.022** [+0.002, +0.040] | YES |
| Note F1 | `full` | `frame_only` | clean | 0.544 [0.474, 0.608] | `extractor` | 0.562 [0.489, 0.635] | **-0.018** [-0.047, +0.010] | NO |
| Note F1 | `full` | `frame_only` | split | 0.544 [0.474, 0.608] | `transformer_v1` | 0.601 [0.538, 0.665] | **-0.057** [-0.089, -0.028] | NO |
| Note F1 | `full` | `frame_only` | split | 0.544 [0.474, 0.608] | `transformer_v1_2seed` | 0.590 [0.528, 0.654] | **-0.046** [-0.075, -0.019] | NO |
| Note F1 | `full` | `frame_only` | split | 0.544 [0.474, 0.608] | `transformer_v2` | 0.563 [0.502, 0.622] | **-0.019** [-0.059, +0.014] | NO |
| Note F1 | `full` | `frame_only` | split | 0.544 [0.474, 0.608] | `tcn` | 0.522 [0.449, 0.595] | **+0.022** [+0.002, +0.040] | YES |
| Note F1 | `full` | `frame_only` | split | 0.544 [0.474, 0.608] | `extractor` | 0.559 [0.488, 0.629] | **-0.015** [-0.044, +0.013] | NO |
| Note F1 | `full` | `frame_only` | merge | 0.544 [0.474, 0.608] | `transformer_v1` | 0.601 [0.538, 0.665] | **-0.057** [-0.089, -0.028] | NO |
| Note F1 | `full` | `frame_only` | merge | 0.544 [0.474, 0.608] | `transformer_v1_2seed` | 0.590 [0.528, 0.654] | **-0.046** [-0.075, -0.019] | NO |
| Note F1 | `full` | `frame_only` | merge | 0.544 [0.474, 0.608] | `transformer_v2` | 0.563 [0.502, 0.622] | **-0.019** [-0.059, +0.014] | NO |
| Note F1 | `full` | `frame_only` | merge | 0.544 [0.474, 0.608] | `tcn` | 0.522 [0.449, 0.595] | **+0.022** [+0.002, +0.040] | YES |
| Note F1 | `full` | `frame_only` | merge | 0.544 [0.474, 0.608] | `extractor` | 0.554 [0.486, 0.617] | **-0.010** [-0.034, +0.015] | NO |
| Note F1 | `full` | `frame_only` | jitter | 0.544 [0.474, 0.608] | `transformer_v1` | 0.601 [0.538, 0.665] | **-0.057** [-0.089, -0.028] | NO |
| Note F1 | `full` | `frame_only` | jitter | 0.544 [0.474, 0.608] | `transformer_v1_2seed` | 0.590 [0.528, 0.654] | **-0.046** [-0.075, -0.019] | NO |
| Note F1 | `full` | `frame_only` | jitter | 0.544 [0.474, 0.608] | `transformer_v2` | 0.563 [0.502, 0.622] | **-0.019** [-0.059, +0.014] | NO |
| Note F1 | `full` | `frame_only` | jitter | 0.544 [0.474, 0.608] | `tcn` | 0.522 [0.449, 0.595] | **+0.022** [+0.002, +0.040] | YES |
| Note F1 | `full` | `frame_only` | jitter | 0.544 [0.474, 0.608] | `extractor` | 0.479 [0.406, 0.552] | **+0.065** [+0.036, +0.098] | YES |
| Onset F1 | `full` | `bpdr` | clean | 0.662 [0.590, 0.733] | `transformer_v1` | 0.715 [0.639, 0.781] | **-0.052** [-0.079, -0.028] | NO |
| Onset F1 | `full` | `bpdr` | clean | 0.662 [0.590, 0.733] | `transformer_v1_2seed` | 0.703 [0.630, 0.765] | **-0.041** [-0.064, -0.014] | NO |
| Onset F1 | `full` | `bpdr` | clean | 0.662 [0.590, 0.733] | `transformer_v2` | 0.724 [0.656, 0.786] | **-0.062** [-0.089, -0.031] | NO |
| Onset F1 | `full` | `bpdr` | clean | 0.662 [0.590, 0.733] | `tcn` | 0.688 [0.614, 0.757] | **-0.026** [-0.043, -0.010] | NO |
| Onset F1 | `full` | `bpdr` | clean | 0.662 [0.590, 0.733] | `extractor` | 0.601 [0.556, 0.643] | **+0.061** [+0.008, +0.121] | YES |
| Onset F1 | `full` | `bpdr` | split | 0.662 [0.590, 0.734] | `transformer_v1` | 0.715 [0.640, 0.782] | **-0.052** [-0.079, -0.028] | NO |
| Onset F1 | `full` | `bpdr` | split | 0.662 [0.590, 0.734] | `transformer_v1_2seed` | 0.704 [0.631, 0.765] | **-0.041** [-0.064, -0.015] | NO |
| Onset F1 | `full` | `bpdr` | split | 0.662 [0.590, 0.734] | `transformer_v2` | 0.724 [0.656, 0.786] | **-0.062** [-0.089, -0.031] | NO |
| Onset F1 | `full` | `bpdr` | split | 0.662 [0.590, 0.734] | `tcn` | 0.690 [0.615, 0.758] | **-0.027** [-0.043, -0.011] | NO |
| Onset F1 | `full` | `bpdr` | split | 0.662 [0.590, 0.734] | `extractor` | 0.596 [0.555, 0.635] | **+0.067** [+0.015, +0.122] | YES |
| Onset F1 | `full` | `bpdr` | merge | 0.662 [0.590, 0.733] | `transformer_v1` | 0.714 [0.638, 0.780] | **-0.052** [-0.078, -0.028] | NO |
| Onset F1 | `full` | `bpdr` | merge | 0.662 [0.590, 0.733] | `transformer_v1_2seed` | 0.704 [0.631, 0.765] | **-0.042** [-0.066, -0.014] | NO |
| Onset F1 | `full` | `bpdr` | merge | 0.662 [0.590, 0.733] | `transformer_v2` | 0.724 [0.656, 0.786] | **-0.062** [-0.089, -0.031] | NO |
| Onset F1 | `full` | `bpdr` | merge | 0.662 [0.590, 0.733] | `tcn` | 0.688 [0.614, 0.757] | **-0.026** [-0.043, -0.010] | NO |
| Onset F1 | `full` | `bpdr` | merge | 0.662 [0.590, 0.733] | `extractor` | 0.594 [0.551, 0.635] | **+0.068** [+0.010, +0.132] | YES |
| Onset F1 | `full` | `bpdr` | jitter | 0.663 [0.589, 0.734] | `transformer_v1` | 0.715 [0.642, 0.780] | **-0.052** [-0.075, -0.028] | NO |
| Onset F1 | `full` | `bpdr` | jitter | 0.663 [0.589, 0.734] | `transformer_v1_2seed` | 0.704 [0.633, 0.765] | **-0.042** [-0.064, -0.015] | NO |
| Onset F1 | `full` | `bpdr` | jitter | 0.663 [0.589, 0.734] | `transformer_v2` | 0.722 [0.654, 0.784] | **-0.059** [-0.086, -0.028] | NO |
| Onset F1 | `full` | `bpdr` | jitter | 0.663 [0.589, 0.734] | `tcn` | 0.685 [0.609, 0.756] | **-0.022** [-0.038, -0.006] | NO |
| Onset F1 | `full` | `bpdr` | jitter | 0.663 [0.589, 0.734] | `extractor` | 0.538 [0.504, 0.570] | **+0.124** [+0.060, +0.186] | YES |
| Onset F1 | `full` | `frame_only` | clean | 0.698 [0.625, 0.765] | `transformer_v1` | 0.719 [0.646, 0.784] | **-0.021** [-0.038, -0.000] | NO |
| Onset F1 | `full` | `frame_only` | clean | 0.698 [0.625, 0.765] | `transformer_v1_2seed` | 0.716 [0.645, 0.781] | **-0.018** [-0.034, +0.003] | NO |
| Onset F1 | `full` | `frame_only` | clean | 0.698 [0.625, 0.765] | `transformer_v2` | 0.699 [0.631, 0.761] | **-0.001** [-0.023, +0.022] | NO |
| Onset F1 | `full` | `frame_only` | clean | 0.698 [0.625, 0.765] | `tcn` | 0.680 [0.607, 0.755] | **+0.018** [+0.006, +0.030] | YES |
| Onset F1 | `full` | `frame_only` | clean | 0.698 [0.625, 0.765] | `extractor` | 0.601 [0.556, 0.643] | **+0.097** [+0.048, +0.150] | YES |
| Onset F1 | `full` | `frame_only` | split | 0.698 [0.625, 0.765] | `transformer_v1` | 0.719 [0.646, 0.784] | **-0.021** [-0.038, -0.000] | NO |
| Onset F1 | `full` | `frame_only` | split | 0.698 [0.625, 0.765] | `transformer_v1_2seed` | 0.716 [0.645, 0.781] | **-0.018** [-0.034, +0.003] | NO |
| Onset F1 | `full` | `frame_only` | split | 0.698 [0.625, 0.765] | `transformer_v2` | 0.699 [0.631, 0.761] | **-0.001** [-0.023, +0.022] | NO |
| Onset F1 | `full` | `frame_only` | split | 0.698 [0.625, 0.765] | `tcn` | 0.680 [0.607, 0.755] | **+0.018** [+0.006, +0.030] | YES |
| Onset F1 | `full` | `frame_only` | split | 0.698 [0.625, 0.765] | `extractor` | 0.596 [0.555, 0.635] | **+0.102** [+0.055, +0.153] | YES |
| Onset F1 | `full` | `frame_only` | merge | 0.698 [0.625, 0.765] | `transformer_v1` | 0.719 [0.646, 0.784] | **-0.021** [-0.038, -0.000] | NO |
| Onset F1 | `full` | `frame_only` | merge | 0.698 [0.625, 0.765] | `transformer_v1_2seed` | 0.716 [0.645, 0.781] | **-0.018** [-0.034, +0.003] | NO |
| Onset F1 | `full` | `frame_only` | merge | 0.698 [0.625, 0.765] | `transformer_v2` | 0.699 [0.631, 0.761] | **-0.001** [-0.023, +0.022] | NO |
| Onset F1 | `full` | `frame_only` | merge | 0.698 [0.625, 0.765] | `tcn` | 0.680 [0.607, 0.755] | **+0.018** [+0.006, +0.030] | YES |
| Onset F1 | `full` | `frame_only` | merge | 0.698 [0.625, 0.765] | `extractor` | 0.594 [0.551, 0.635] | **+0.104** [+0.050, +0.161] | YES |
| Onset F1 | `full` | `frame_only` | jitter | 0.698 [0.625, 0.765] | `transformer_v1` | 0.719 [0.646, 0.784] | **-0.021** [-0.038, -0.000] | NO |
| Onset F1 | `full` | `frame_only` | jitter | 0.698 [0.625, 0.765] | `transformer_v1_2seed` | 0.716 [0.645, 0.781] | **-0.018** [-0.034, +0.003] | NO |
| Onset F1 | `full` | `frame_only` | jitter | 0.698 [0.625, 0.765] | `transformer_v2` | 0.699 [0.631, 0.761] | **-0.001** [-0.023, +0.022] | NO |
| Onset F1 | `full` | `frame_only` | jitter | 0.698 [0.625, 0.765] | `tcn` | 0.680 [0.607, 0.755] | **+0.018** [+0.006, +0.030] | YES |
| Onset F1 | `full` | `frame_only` | jitter | 0.698 [0.625, 0.765] | `extractor` | 0.538 [0.504, 0.570] | **+0.160** [+0.100, +0.217] | YES |

### `dev_M02`: Transformer v4 (POS_WEIGHT=10.0, offset=0.0) vs Baselines (Boundary Onset F1, 64 Clips)

| Feat | Arm | Cond | Trans v4 (Mean [CI]) | Baseline | Baseline (Mean [CI]) | Diff (v4 - Base) [95% CI] | Win (CI > 0)? |
|---|---|---|---|---|---|---|---|
| `full` | `bpdr` | clean | 0.795 [0.769, 0.822] | `transformer_v1` | 0.767 [0.738, 0.796] | **+0.028** [+0.014, +0.042] | YES |
| `full` | `bpdr` | clean | 0.795 [0.769, 0.822] | `transformer_v1_2seed` | 0.767 [0.737, 0.794] | **+0.028** [+0.016, +0.042] | YES |
| `full` | `bpdr` | clean | 0.795 [0.769, 0.822] | `transformer_v2` | 0.812 [0.786, 0.838] | **-0.017** [-0.027, -0.008] | NO |
| `full` | `bpdr` | clean | 0.795 [0.769, 0.822] | `tcn` | 0.782 [0.754, 0.809] | **+0.014** [+0.004, +0.023] | YES |
| `full` | `bpdr` | clean | 0.795 [0.769, 0.822] | `extractor` | 0.548 [0.529, 0.564] | **+0.248** [+0.223, +0.273] | YES |
| `full` | `bpdr` | split | 0.795 [0.769, 0.822] | `transformer_v1` | 0.767 [0.738, 0.795] | **+0.028** [+0.014, +0.043] | YES |
| `full` | `bpdr` | split | 0.795 [0.769, 0.822] | `transformer_v1_2seed` | 0.767 [0.737, 0.794] | **+0.029** [+0.016, +0.042] | YES |
| `full` | `bpdr` | split | 0.795 [0.769, 0.822] | `transformer_v2` | 0.812 [0.786, 0.838] | **-0.017** [-0.027, -0.008] | NO |
| `full` | `bpdr` | split | 0.795 [0.769, 0.822] | `tcn` | 0.782 [0.754, 0.809] | **+0.014** [+0.004, +0.023] | YES |
| `full` | `bpdr` | split | 0.795 [0.769, 0.822] | `extractor` | 0.537 [0.519, 0.554] | **+0.258** [+0.234, +0.284] | YES |
| `full` | `bpdr` | merge | 0.796 [0.770, 0.822] | `transformer_v1` | 0.767 [0.737, 0.795] | **+0.029** [+0.015, +0.044] | YES |
| `full` | `bpdr` | merge | 0.796 [0.770, 0.822] | `transformer_v1_2seed` | 0.767 [0.738, 0.794] | **+0.029** [+0.016, +0.042] | YES |
| `full` | `bpdr` | merge | 0.796 [0.770, 0.822] | `transformer_v2` | 0.812 [0.786, 0.838] | **-0.016** [-0.026, -0.008] | NO |
| `full` | `bpdr` | merge | 0.796 [0.770, 0.822] | `tcn` | 0.782 [0.754, 0.809] | **+0.014** [+0.005, +0.023] | YES |
| `full` | `bpdr` | merge | 0.796 [0.770, 0.822] | `extractor` | 0.541 [0.523, 0.559] | **+0.255** [+0.230, +0.282] | YES |
| `full` | `bpdr` | jitter | 0.796 [0.770, 0.823] | `transformer_v1` | 0.767 [0.737, 0.795] | **+0.029** [+0.016, +0.043] | YES |
| `full` | `bpdr` | jitter | 0.796 [0.770, 0.823] | `transformer_v1_2seed` | 0.766 [0.737, 0.793] | **+0.030** [+0.017, +0.043] | YES |
| `full` | `bpdr` | jitter | 0.796 [0.770, 0.823] | `transformer_v2` | 0.812 [0.787, 0.838] | **-0.016** [-0.025, -0.008] | NO |
| `full` | `bpdr` | jitter | 0.796 [0.770, 0.823] | `tcn` | 0.781 [0.753, 0.808] | **+0.015** [+0.005, +0.025] | YES |
| `full` | `bpdr` | jitter | 0.796 [0.770, 0.823] | `extractor` | 0.526 [0.503, 0.545] | **+0.270** [+0.245, +0.297] | YES |
| `full` | `frame_only` | clean | 0.788 [0.761, 0.815] | `transformer_v1` | 0.754 [0.724, 0.782] | **+0.034** [+0.022, +0.047] | YES |
| `full` | `frame_only` | clean | 0.788 [0.761, 0.815] | `transformer_v1_2seed` | 0.756 [0.728, 0.783] | **+0.032** [+0.021, +0.043] | YES |
| `full` | `frame_only` | clean | 0.788 [0.761, 0.815] | `transformer_v2` | 0.807 [0.779, 0.834] | **-0.019** [-0.031, -0.009] | NO |
| `full` | `frame_only` | clean | 0.788 [0.761, 0.815] | `tcn` | 0.780 [0.752, 0.809] | **+0.008** [-0.002, +0.018] | NO |
| `full` | `frame_only` | clean | 0.788 [0.761, 0.815] | `extractor` | 0.548 [0.529, 0.564] | **+0.240** [+0.216, +0.265] | YES |
| `full` | `frame_only` | split | 0.788 [0.761, 0.815] | `transformer_v1` | 0.754 [0.724, 0.782] | **+0.034** [+0.022, +0.047] | YES |
| `full` | `frame_only` | split | 0.788 [0.761, 0.815] | `transformer_v1_2seed` | 0.756 [0.728, 0.783] | **+0.032** [+0.021, +0.043] | YES |
| `full` | `frame_only` | split | 0.788 [0.761, 0.815] | `transformer_v2` | 0.807 [0.779, 0.834] | **-0.019** [-0.031, -0.009] | NO |
| `full` | `frame_only` | split | 0.788 [0.761, 0.815] | `tcn` | 0.780 [0.752, 0.809] | **+0.008** [-0.002, +0.018] | NO |
| `full` | `frame_only` | split | 0.788 [0.761, 0.815] | `extractor` | 0.537 [0.519, 0.554] | **+0.251** [+0.227, +0.276] | YES |
| `full` | `frame_only` | merge | 0.788 [0.761, 0.815] | `transformer_v1` | 0.754 [0.724, 0.782] | **+0.034** [+0.022, +0.047] | YES |
| `full` | `frame_only` | merge | 0.788 [0.761, 0.815] | `transformer_v1_2seed` | 0.756 [0.728, 0.783] | **+0.032** [+0.021, +0.043] | YES |
| `full` | `frame_only` | merge | 0.788 [0.761, 0.815] | `transformer_v2` | 0.807 [0.779, 0.834] | **-0.019** [-0.031, -0.009] | NO |
| `full` | `frame_only` | merge | 0.788 [0.761, 0.815] | `tcn` | 0.780 [0.752, 0.809] | **+0.008** [-0.002, +0.018] | NO |
| `full` | `frame_only` | merge | 0.788 [0.761, 0.815] | `extractor` | 0.541 [0.523, 0.559] | **+0.247** [+0.223, +0.273] | YES |
| `full` | `frame_only` | jitter | 0.788 [0.761, 0.815] | `transformer_v1` | 0.754 [0.724, 0.782] | **+0.034** [+0.022, +0.047] | YES |
| `full` | `frame_only` | jitter | 0.788 [0.761, 0.815] | `transformer_v1_2seed` | 0.756 [0.728, 0.783] | **+0.032** [+0.021, +0.043] | YES |
| `full` | `frame_only` | jitter | 0.788 [0.761, 0.815] | `transformer_v2` | 0.807 [0.779, 0.834] | **-0.019** [-0.031, -0.009] | NO |
| `full` | `frame_only` | jitter | 0.788 [0.761, 0.815] | `tcn` | 0.780 [0.752, 0.809] | **+0.008** [-0.002, +0.018] | NO |
| `full` | `frame_only` | jitter | 0.788 [0.761, 0.815] | `extractor` | 0.526 [0.503, 0.545] | **+0.262** [+0.237, +0.288] | YES |

### Transformer v4 (POS_WEIGHT=10.0, offset=0.0) Fit Threshold & Seed Coverage

| Arch | Feat | Arm | Seeds Run | Fits | Healthy | Thresholds (`thr`) |
|---|---|---|---|---|---|---|
| `transformer_v4` | `full` | `bpdr` | `339, 340` | 2 | 2/2 | `0.9` |
| `transformer_v4` | `full` | `frame_only` | `339, 340` | 2 | 2/2 | `0.8, 0.85` |

- **Seed coverage**: `transformer_v4`: 2 seed(s) ([339, 340]).
- **Threshold distribution**: Observed `thr` values across 4 fits: `0.8, 0.85, 0.9`.

## Studio End-to-End Vocadito Benchmark (39 Windows, 14 Singers)

**Pipeline**: `evidence.load_mono` (10 s window crop) -> `clean.spectral_gate` + `evidence.pesto_track` + `clean.choose` -> `boundary.extract_boundary_hints` (20 ms grid) -> `make_notes` / `evidence.notes_from_track`; 14-singer singer-macro 95% bootstrap CIs (1,000 resamples, seed 339).

| Method | nin | Thr | Note F1 (Mean [95% CI]) | Onset F1 (Mean [95% CI]) | Diff vs `pyin_4096` (Note F1) | Diff vs `pesto_notes` (Note F1) | Diff vs `studio_v1` (Note F1) |
|---|---|---|---|---|---|---|---|
| `pyin_4096` | - | - | 0.301 [0.256, 0.352] | 0.419 [0.360, 0.483] | - | **-0.185** [-0.239, -0.131] | **-0.330** [-0.395, -0.257] |
| `pyin_4096_prepared` | - | - | 0.301 [0.256, 0.352] | 0.419 [0.360, 0.483] | **+0.000** [+0.000, +0.000] | **-0.185** [-0.239, -0.131] | **-0.330** [-0.395, -0.257] |
| `pesto_notes_from_track` | - | - | 0.486 [0.431, 0.546] | 0.617 [0.572, 0.664] | **+0.185** [+0.131, +0.239] | - | **-0.145** [-0.201, -0.087] |
| `studio_v1` | 5 | 0.80 | 0.631 [0.585, 0.676] | 0.767 [0.732, 0.799] | **+0.330** [+0.257, +0.395] | **+0.145** [+0.087, +0.201] | - |
| `studio_v2` | 5 | 0.65 | 0.591 [0.534, 0.643] | 0.728 [0.689, 0.766] | **+0.289** [+0.209, +0.367] | **+0.104** [+0.032, +0.176] | **-0.040** [-0.072, -0.010] |
| `studio_v2_bpdr` | 6 | 0.60 | 0.617 [0.561, 0.673] | 0.746 [0.706, 0.787] | **+0.315** [+0.233, +0.394] | **+0.130** [+0.057, +0.200] | **-0.014** [-0.036, +0.010] |
| `studio_v4` | 5 | 0.85 | 0.576 [0.529, 0.628] | 0.736 [0.702, 0.770] | **+0.275** [+0.200, +0.349] | **+0.090** [+0.032, +0.146] | **-0.055** [-0.096, -0.012] |
| `studio_v4_bpdr` | 6 | 0.90 | 0.534 [0.494, 0.579] | 0.705 [0.674, 0.736] | **+0.233** [+0.170, +0.298] | **+0.048** [-0.010, +0.095] | **-0.097** [-0.132, -0.061] |
| `studio_v5` | 5 | 0.85 | 0.576 [0.526, 0.627] | 0.731 [0.699, 0.763] | **+0.274** [+0.209, +0.341] | **+0.089** [+0.050, +0.130] | **-0.055** [-0.092, -0.021] |
| `studio_v1_pre_parity_100hz_thr05` | - | - | 0.587 [0.541, 0.638] | 0.751 [0.719, 0.785] | **+0.285** [+0.232, +0.336] | **+0.100** [+0.081, +0.121] | **-0.044** [-0.083, -0.003] |

## Summary of Declared Experimental Bars
1. **Part 1 (Vocadito Note F1 Model vs Extractor)**: **FAIL / NO WIN (TCN declared target)** — On clean audio, TCN full-feature BPDR Note F1 (0.535) does not beat the Extractor (0.562; diff -0.027 [-0.057, +0.005], interval includes 0), though TCN beats the Extractor on Onset F1 (+0.087 [+0.038, +0.140]) and jitter Note F1 (+0.050 [+0.018, +0.084]). However, **FastTNet Transformer (`full`, `frame_only`) beats the Extractor on clean Note F1** (0.601 vs 0.562; diff +0.039 [+0.006, +0.072]), with 95% CI > 0 across all 4 conditions (clean/split/merge/jitter).
2. **Part 2 (Transformer vs TCN Onset & Note F1)**: **ADOPT TRANSFORMER**.
3. **Part 3 (BPDR Loss on Lean Features Across Corruptions)**: **BPDR loss terms do not statistically improve onset F1 over controls when the track is needed**.
4. **Development (`dev_M02`) & Fit Coverage**: On `dev_M02` (64 clips), TCN `full frame_only` clean Onset F1 is 0.780 vs FastTNet Transformer v1 0.754. TCN ran 2 seeds (`339, 340`, 12 fits) vs Transformer v1's 1 seed (`339`, 6 fits); all 18/18 fits hit `thr = 0.8`.
5. **Transformer v2 Comparison**: 12 fits (100.0% healthy, seeds `[339, 340]`), thresholds `0.55, 0.6, 0.65, 0.7, 0.75` (no ceiling saturation at 0.8). On `dev_M02`, v2 `full bpdr` achieves 0.812 Onset F1 (vs v1 +0.045 [+0.031, +0.058], vs TCN +0.030 [+0.021, +0.039]). On Vocadito, v2 `full bpdr` reaches 0.724 Onset F1 (vs v1 +0.010 [-0.018, +0.035], vs TCN +0.036 [+0.012, +0.058]) and 0.577 Note F1 (vs v1 -0.000 [-0.031, +0.030]). However, v2 `full frame_only` Note F1 (0.563) trails v1 (0.601; diff -0.038 [-0.060, -0.014]), indicating stronger HumTrans fit (`dev_M02` +0.053) did not transfer to Vocadito standalone Note F1.
6. **Transformer v3 Comparison (DO NOT ADOPT v3 (keep V1_CHECKPOINT as Studio default))**: 12 fits (100.0% healthy, seeds `[339, 340]`), thresholds `0.6, 0.65, 0.7, 0.75, 0.8, 0.85` (non-saturated). On `dev_M02`, v3 `full frame_only` achieves 0.788 Onset F1 (vs v1 +0.035 [+0.023, +0.047]). On Vocadito, v3 `full frame_only` reaches 0.693 Onset F1 (vs v1 -0.026 [-0.045, -0.004]) and 0.537 Note F1 (vs v1 -0.064 [-0.098, -0.034]).
7. **Transformer v4 & v1 2-Seed Replication**: v4 ran 4 fits (100.0% healthy, seeds `[339, 340]`), thresholds `0.8, 0.85, 0.9`. On Vocadito `full frame_only` clean Note F1, v4 achieves 0.544 vs 2-seed v1 0.590 (diff -0.046 [-0.075, -0.019]). On `full bpdr` clean Onset F1, v4 reaches 0.662 vs 2-seed v1 0.703 (diff -0.041 [-0.064, -0.014]).
8. **Studio Vocadito Benchmark (39 Windows, 14 Singers)**: `pyin_4096` Note F1=0.301, `pesto_notes_from_track`=0.486, `studio_v1`=0.631; best Studio model `studio_v1` Note F1=0.631 [0.585, 0.676].
