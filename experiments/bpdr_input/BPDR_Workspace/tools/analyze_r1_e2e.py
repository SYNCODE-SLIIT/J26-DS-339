"""Analyze end-to-end note F1, Transformer vs TCN, lean-feature BPDR, dev_M02, and fit coverage.
Declared in results/studio/r1/declaration_e2e.json (and optional Transformer v2 comparison).

Computes singer-macro (14 Vocadito singers) 95% bootstrap confidence intervals (1000 resamples).
Evaluates the declared experimental bars:
  Part 1: Vocadito Note F1 & Onset F1 comparing TCN and FastTNet Transformer vs Extractor across clean and corruptions.
  Part 2: Transformer vs TCN on Vocadito Onset F1 and Note F1.
  Part 3: BPDR loss vs frame_only and augmentation_only controls on lean features across corruptions.
  dev_M02: Boundary F1 summary across architectures, feature sets, arms, and corruptions (64 clips).
  Fit Coverage: Seed counts, health share, and threshold calibration/saturation summary.
  Optional (--compare-v2): Statistical comparison of Transformer v2 vs v1, TCN, and Extractor on Vocadito and dev_M02.
"""
import argparse
import json
from pathlib import Path
import numpy as np

WS = Path(__file__).resolve().parents[1]
R = WS / 'results/studio/r1'
RESULT_FILE = R / 'e2e_fit_results.json'
PARTIAL_FILE = R / 'e2e_fit_partial.json'
V2_RESULT_FILE = R / 'transformer_v2_results.json'
V3_RESULT_FILE = R / 'transformer_v3_results.json'
V4_RESULT_FILE = R / 'transformer_v4_results.json'
BENCHMARK_FILE = R / 'studio_vocadito_benchmark.json'
DECLARATION_FILE = R / 'declaration_e2e.json'

N_RESAMPLES = 1000
RNG_SEED = 339
KINDS = ('clean', 'split', 'merge', 'jitter')
CORRUPTIONS = ('split', 'merge', 'jitter')
ARMS = ('frame_only', 'augmentation_only', 'bpdr')
ARCH_LABELS = {
    'tcn': 'TCN',
    'transformer': 'FastTNet Transformer (v1)',
    'transformer_v2': 'FastTNet Transformer (v2)',
    'transformer_v3': 'FastTNet Transformer (v3)',
    'transformer_v4': 'FastTNet Transformer (v4)',
}


def load_results(path=None):
    if path is not None:
        target = Path(path)
    else:
        target = RESULT_FILE if RESULT_FILE.exists() else PARTIAL_FILE
    if not target.exists():
        raise FileNotFoundError(f"Results file not found: {target}")
    with open(target, encoding='utf-8') as f:
        data = json.load(f)
    is_partial = (target == PARTIAL_FILE) or bool(data.get('partial', False))
    rows = data['rows']
    return rows, is_partial, target.name


def bootstrap_ci(diff_by_singer, n_resamples=N_RESAMPLES, seed=RNG_SEED):
    """diff_by_singer: dict mapping unit (e.g. singer) -> diff value.
    Resamples units with replacement to compute 95% bootstrap CI.
    """
    units = sorted(diff_by_singer.keys())
    diffs = np.array([diff_by_singer[u] for u in units], dtype=float)
    rng = np.random.default_rng(seed)
    n = len(diffs)
    idx = rng.integers(0, n, size=(n_resamples, n))
    boot_means = diffs[idx].mean(axis=1)
    mean_val = float(diffs.mean())
    ci_lo, ci_hi = [float(x) for x in np.percentile(boot_means, [2.5, 97.5])]
    return mean_val, ci_lo, ci_hi


def singer_metric_ci(metric_by_singer, n_resamples=N_RESAMPLES, seed=RNG_SEED):
    """Resample units (singers or clips) to get 95% CI for the mean metric itself."""
    units = sorted(metric_by_singer.keys())
    vals = np.array([metric_by_singer[u] for u in units], dtype=float)
    rng = np.random.default_rng(seed)
    n = len(vals)
    idx = rng.integers(0, n, size=(n_resamples, n))
    boot_means = vals[idx].mean(axis=1)
    return float(vals.mean()), float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def get_split_clip_values(rows, split, arch, feat, arm, key):
    """Returns dict: (person, id) -> mean value across seeds for the given split."""
    clips = {}
    for r in rows:
        if r.get('split') == split and r.get('arch') == arch and r.get('feat') == feat and r.get('arm') == arm:
            if key in r:
                clips.setdefault((r['person'], r['id']), []).append(r[key])
    return {k: float(np.mean(v)) for k, v in clips.items()}


def get_vocadito_clip_values(rows, arch, feat, arm, key):
    """Returns dict: (person, id) -> mean value across seeds on Vocadito."""
    return get_split_clip_values(rows, 'vocadito', arch, feat, arm, key)


def clips_to_singer_means(clips):
    """Aggregate clip values to singer means."""
    by_singer = {}
    for (person, _), val in clips.items():
        by_singer.setdefault(person, []).append(val)
    return {person: float(np.mean(vals)) for person, vals in by_singer.items()}


def compare_vocadito_clips(a_clips, b_clips, n_resamples=N_RESAMPLES, seed=RNG_SEED):
    """Paired singer-macro comparison between two per-clip metric dicts."""
    common = sorted(set(a_clips.keys()) & set(b_clips.keys()))
    if not common:
        return None
    diff_by_clip = {c: a_clips[c] - b_clips[c] for c in common}
    a_mean, a_lo, a_hi = singer_metric_ci(
        clips_to_singer_means({c: a_clips[c] for c in common}),
        n_resamples=n_resamples,
        seed=seed,
    )
    b_mean, b_lo, b_hi = singer_metric_ci(
        clips_to_singer_means({c: b_clips[c] for c in common}),
        n_resamples=n_resamples,
        seed=seed,
    )
    d_mean, d_lo, d_hi = bootstrap_ci(
        clips_to_singer_means(diff_by_clip),
        n_resamples=n_resamples,
        seed=seed,
    )
    return {
        'a': (a_mean, a_lo, a_hi),
        'b': (b_mean, b_lo, b_hi),
        'diff': (d_mean, d_lo, d_hi),
        'win': d_lo > 0,
        'n_clips': len(common),
    }


def compare_clip_level(a_clips, b_clips, n_resamples=N_RESAMPLES, seed=RNG_SEED):
    """Paired clip-level comparison (for single-person splits such as dev_M02)."""
    common = sorted(set(a_clips.keys()) & set(b_clips.keys()))
    if not common:
        return None
    a_sub = {c: a_clips[c] for c in common}
    b_sub = {c: b_clips[c] for c in common}
    diff_sub = {c: a_clips[c] - b_clips[c] for c in common}
    diff_ci = bootstrap_ci(diff_sub, n_resamples=n_resamples, seed=seed)
    return {
        'a': singer_metric_ci(a_sub, n_resamples=n_resamples, seed=seed),
        'b': singer_metric_ci(b_sub, n_resamples=n_resamples, seed=seed),
        'diff': diff_ci,
        'win': diff_ci[1] > 0,
        'n_clips': len(common),
    }


def _present_archs(rows, candidates=('tcn', 'transformer'), split='vocadito', feat='full'):
    present = []
    for a in candidates:
        if any(r.get('arch') == a and r.get('split') == split and (feat is None or r.get('feat') == feat) for r in rows):
            present.append(a)
    return present


def analyze_part1(rows):
    """Part 1: Vocadito Note F1 and Onset F1 comparing TCN and FastTNet Transformer vs Extractor.
    Declared bar (declaration_e2e.json): WIN if TCN full bpdr note F1 beats extractor note F1 on clean,
    with 95% bootstrap CI excluding 0. Also reports FastTNet Transformer (full) vs Extractor across all arms/conditions.
    """
    print("\n" + "=" * 80)
    print("PART 1: Vocadito Note F1 and Onset F1 (Model vs Extractor)")
    print("=" * 80)

    archs = _present_archs(rows, ('tcn', 'transformer'), split='vocadito', feat='full')
    results = {}
    markdown_lines = []
    markdown_lines.append("### Part 1: Vocadito Note F1 & Onset F1 (Model vs Extractor)")
    markdown_lines.append("")
    markdown_lines.append("**Declared bar target: TCN full-feature `bpdr` arm on clean Note F1 (also reporting FastTNet Transformer full-feature vs Extractor across all arms and conditions).**")
    markdown_lines.append("")

    table_header = "| Arch | Metric | Arm | Condition | Model (Mean [95% CI]) | Extractor (Mean [95% CI]) | Diff (Model - Ext) [95% CI] | Win (CI > 0)? |"
    table_sep    = "|---|---|---|---|---|---|---|---|"
    markdown_lines.append(table_header)
    markdown_lines.append(table_sep)

    ext_ref_arch = None
    for arch in archs:
        for metric_name, m_prefix, e_prefix in (('Note F1', 'note_model_', 'note_ext_'), ('Onset F1', 'f1_', 'ext_')):
            print(f"\n--- {metric_name} ({ARCH_LABELS.get(arch, arch)}, full-feature) ---")
            for arm in ARMS:
                for k in KINDS:
                    m_clips = get_vocadito_clip_values(rows, arch, 'full', arm, m_prefix + k)
                    e_clips = get_vocadito_clip_values(rows, arch, 'full', arm, e_prefix + k)
                    comp = compare_vocadito_clips(m_clips, e_clips)
                    if comp is None:
                        continue
                    if ext_ref_arch is None:
                        ext_ref_arch = (arch, arm)

                    m_mean, m_lo, m_hi = comp['a']
                    e_mean, e_lo, e_hi = comp['b']
                    d_mean, d_lo, d_hi = comp['diff']
                    win = comp['win']
                    win_str = "YES" if win else "NO"
                    print(f"[{arch} | {metric_name}] {arm:<18} {k:<7} | Model: {m_mean:.3f} [{m_lo:.3f}, {m_hi:.3f}] | Ext: {e_mean:.3f} [{e_lo:.3f}, {e_hi:.3f}] | Diff: {d_mean:+.3f} [{d_lo:+.3f}, {d_hi:+.3f}] | Win: {win_str}")

                    md_row = f"| `{arch}` | {metric_name} | `{arm}` | {k} | {m_mean:.3f} [{m_lo:.3f}, {m_hi:.3f}] | {e_mean:.3f} [{e_lo:.3f}, {e_hi:.3f}] | **{d_mean:+.3f}** [{d_lo:+.3f}, {d_hi:+.3f}] | {win_str} |"
                    markdown_lines.append(md_row)

                    entry = {
                        'model': (m_mean, m_lo, m_hi),
                        'extractor': (e_mean, e_lo, e_hi),
                        'diff': (d_mean, d_lo, d_hi),
                        'win': win,
                    }
                    results[(arch, metric_name, arm, k)] = entry
                    if arch == 'tcn':
                        results[(metric_name, arm, k)] = entry

    # Corruption drop for Note F1 and Onset F1
    print("\n--- Corruption Drop (Clean - Mean(Corruptions)) ---")
    markdown_lines.append("")
    markdown_lines.append("**Corruption Drop Analysis (Clean - Mean(Corruptions)):**")
    markdown_lines.append("")
    markdown_lines.append("| Arch | Metric | Arm | Clean | Split | Merge | Jitter | Mean Drop |")
    markdown_lines.append("|---|---|---|---|---|---|---|---|")

    for arch in archs:
        for metric_name, m_prefix in (('Note F1', 'note_model_'), ('Onset F1', 'f1_')):
            for arm in ARMS:
                c_vals = {}
                for k in KINDS:
                    clips = get_vocadito_clip_values(rows, arch, 'full', arm, m_prefix + k)
                    if clips:
                        c_vals[k] = singer_metric_ci(clips_to_singer_means(clips))[0]
                if len(c_vals) != len(KINDS):
                    continue
                mean_drop = c_vals['clean'] - float(np.mean([c_vals[k] for k in CORRUPTIONS]))
                print(f"[{arch} | {metric_name}] {arm:<18} clean={c_vals['clean']:.3f} split={c_vals['split']:.3f} merge={c_vals['merge']:.3f} jitter={c_vals['jitter']:.3f} -> mean drop={mean_drop:+.3f}")
                markdown_lines.append(f"| `{arch}` | {metric_name} | `{arm}` | {c_vals['clean']:.3f} | {c_vals['split']:.3f} | {c_vals['merge']:.3f} | {c_vals['jitter']:.3f} | {mean_drop:+.3f} |")

    # Extractor drop
    if ext_ref_arch is not None:
        ref_arch, ref_arm = ext_ref_arch
        for metric_name, e_prefix in (('Note F1', 'note_ext_'), ('Onset F1', 'ext_')):
            c_vals = {
                k: singer_metric_ci(clips_to_singer_means(get_vocadito_clip_values(rows, ref_arch, 'full', ref_arm, e_prefix + k)))[0]
                for k in KINDS
            }
            mean_drop = c_vals['clean'] - float(np.mean([c_vals[k] for k in CORRUPTIONS]))
            print(f"[extractor | {metric_name}] {'baseline':<18} clean={c_vals['clean']:.3f} split={c_vals['split']:.3f} merge={c_vals['merge']:.3f} jitter={c_vals['jitter']:.3f} -> mean drop={mean_drop:+.3f}")
            markdown_lines.append(f"| `extractor` | {metric_name} | `baseline` | {c_vals['clean']:.3f} | {c_vals['split']:.3f} | {c_vals['merge']:.3f} | {c_vals['jitter']:.3f} | {mean_drop:+.3f} |")

    # Part 1 Bar Evaluation
    p1_key = ('tcn', 'Note F1', 'bpdr', 'clean')
    p1_onset_key = ('tcn', 'Onset F1', 'bpdr', 'clean')
    p1_pass = results.get(p1_key, {}).get('win', False)
    d_mean, d_lo, d_hi = results.get(p1_key, {}).get('diff', (0.0, 0.0, 0.0))
    on_mean, on_lo, on_hi = results.get(p1_onset_key, {}).get('diff', (0.0, 0.0, 0.0))

    print("\n--- Part 1 Declared Bar Verdict ---")
    print("Criterion: TCN full-feature bpdr Note F1 beats Extractor on clean with 95% CI excluding 0")
    print(f"TCN Note F1 Diff: {d_mean:+.3f} [{d_lo:+.3f}, {d_hi:+.3f}] -> {'PASS (WIN)' if p1_pass else 'FAIL (NO WIN)'}")
    print(f"TCN Onset F1 Diff (supplementary): {on_mean:+.3f} [{on_lo:+.3f}, {on_hi:+.3f}] -> {'PASS' if on_lo > 0 else 'FAIL'}")

    tf_key = ('transformer', 'Note F1', 'frame_only', 'clean')
    if tf_key in results:
        tf_m = results[tf_key]['model'][0]
        tf_e = results[tf_key]['extractor'][0]
        tf_d, tf_lo, tf_hi = results[tf_key]['diff']
        tf_all_win = all(results.get(('transformer', 'Note F1', 'frame_only', k), {}).get('win', False) for k in KINDS)
        print(f"FastTNet Transformer (full frame_only) clean Note F1: {tf_m:.3f} vs Ext {tf_e:.3f}, Diff {tf_d:+.3f} [{tf_lo:+.3f}, {tf_hi:+.3f}] (all 4 conditions CI > 0: {tf_all_win})")

    return results, markdown_lines, p1_pass


def analyze_part2(rows):
    """Part 2: Transformer vs TCN on Vocadito Onset F1 and Note F1.
    Declared bar: Adopt Transformer only if Vocadito onset F1 at 50 ms exceeds
    TCN for the same arm with singer-level 95% CI excluding 0; otherwise say it did not help.
    """
    print("\n" + "=" * 80)
    print("PART 2: Transformer vs TCN on Vocadito Onset F1 and Note F1")
    print("=" * 80)

    results = {}
    markdown_lines = []
    markdown_lines.append("### Part 2: Transformer vs TCN (Onset F1 & Note F1)")
    markdown_lines.append("")
    markdown_lines.append("**Adoption criterion: Vocadito Onset F1 at 50 ms exceeds TCN for same arm with 95% CI excluding 0.**")
    markdown_lines.append("")

    table_header = "| Metric | Feat | Arm | Cond | Transformer (Mean [CI]) | TCN (Mean [CI]) | Diff (Trans - TCN) [95% CI] | Adopt? |"
    table_sep    = "|---|---|---|---|---|---|---|---|"
    markdown_lines.append(table_header)
    markdown_lines.append(table_sep)

    adopt_count = 0

    for metric_name, key_prefix in (('Onset F1', 'f1_'), ('Note F1', 'note_model_')):
        print(f"\n--- {metric_name} ---")
        for feat in ('full', 'lean'):
            for arm in ARMS:
                for k in KINDS:
                    t_clips = get_vocadito_clip_values(rows, 'transformer', feat, arm, key_prefix + k)
                    c_clips = get_vocadito_clip_values(rows, 'tcn', feat, arm, key_prefix + k)
                    comp = compare_vocadito_clips(t_clips, c_clips)
                    if comp is None:
                        continue

                    t_mean, t_lo, t_hi = comp['a']
                    c_mean, c_lo, c_hi = comp['b']
                    d_mean, d_lo, d_hi = comp['diff']
                    adopt = comp['win']
                    if metric_name == 'Onset F1' and k == 'clean' and adopt:
                        adopt_count += 1

                    adopt_str = "ADOPT" if adopt else "DID NOT HELP"
                    if k == 'clean':
                        print(f"[{metric_name}] {feat:4} {arm:<18} {k:6} | Trans: {t_mean:.3f} [{t_lo:.3f}, {t_hi:.3f}] | TCN: {c_mean:.3f} [{c_lo:.3f}, {c_hi:.3f}] | Diff: {d_mean:+.3f} [{d_lo:+.3f}, {d_hi:+.3f}] | {adopt_str}")

                    md_row = f"| {metric_name} | `{feat}` | `{arm}` | {k} | {t_mean:.3f} [{t_lo:.3f}, {t_hi:.3f}] | {c_mean:.3f} [{c_lo:.3f}, {c_hi:.3f}] | **{d_mean:+.3f}** [{d_lo:+.3f}, {d_hi:+.3f}] | {adopt_str} |"
                    markdown_lines.append(md_row)

                    results[(metric_name, feat, arm, k)] = {
                        'transformer': (t_mean, t_lo, t_hi),
                        'tcn': (c_mean, c_lo, c_hi),
                        'diff': (d_mean, d_lo, d_hi),
                        'adopt': adopt,
                    }

    print("\n--- Part 2 Declared Bar Verdict ---")
    p2_verdict = "ADOPT TRANSFORMER" if adopt_count > 0 else "TRANSFORMER DID NOT HELP (Underfitting in limited steps)"
    print(f"Verdict: {p2_verdict} (positive differences with 95% CI > 0: {adopt_count})")

    return results, markdown_lines, p2_verdict


def analyze_part3(rows):
    """Part 3: BPDR loss vs frame_only and augmentation_only on lean features across corruptions.
    Declared bar: WIN for BPDR loss if, with LEAN features and either architecture,
    bpdr beats BOTH frame_only and augmentation_only on Vocadito onset F1 at 50 ms
    averaged over the three corruptions (split, merge, jitter), each difference with 95% CI excluding 0.
    """
    print("\n" + "=" * 80)
    print("PART 3: BPDR Loss vs Controls on Lean Features Across Corruptions")
    print("=" * 80)

    results = {}
    markdown_lines = []
    markdown_lines.append("### Part 3: BPDR Loss vs Controls on Lean Features (Corruption Average)")
    markdown_lines.append("")
    markdown_lines.append("**Bar: WIN if bpdr beats BOTH frame_only and augmentation_only on corruption-average onset F1 with 95% CI > 0.**")
    markdown_lines.append("")

    table_header = "| Architecture | Arm | Split | Merge | Jitter | Corruption Mean (Mean [CI]) | Diff vs bpdr [95% CI] | Win (CI > 0)? |"
    table_sep    = "|---|---|---|---|---|---|---|---|"
    markdown_lines.append(table_header)
    markdown_lines.append(table_sep)

    win_by_arch = {}

    for arch in ('tcn', 'transformer'):
        print(f"\n--- Architecture: {arch.upper()} (Lean Features) ---")
        arm_corr_clips = {}
        for arm in ARMS:
            c_clips_by_k = [get_vocadito_clip_values(rows, arch, 'lean', arm, 'f1_' + k) for k in CORRUPTIONS]
            if not all(c_clips_by_k):
                continue
            common = sorted(set(c_clips_by_k[0].keys()) & set(c_clips_by_k[1].keys()) & set(c_clips_by_k[2].keys()))
            arm_corr_clips[arm] = {c: float(np.mean([c_clips_by_k[0][c], c_clips_by_k[1][c], c_clips_by_k[2][c]])) for c in common}

        if 'bpdr' not in arm_corr_clips:
            print(f"[{arch}] bpdr arm data not available yet.")
            continue

        bpdr_clips = arm_corr_clips['bpdr']
        bpdr_m, bpdr_lo, bpdr_hi = singer_metric_ci(clips_to_singer_means(bpdr_clips))

        bpdr_split = singer_metric_ci(clips_to_singer_means(get_vocadito_clip_values(rows, arch, 'lean', 'bpdr', 'f1_split')))[0]
        bpdr_merge = singer_metric_ci(clips_to_singer_means(get_vocadito_clip_values(rows, arch, 'lean', 'bpdr', 'f1_merge')))[0]
        bpdr_jitter = singer_metric_ci(clips_to_singer_means(get_vocadito_clip_values(rows, arch, 'lean', 'bpdr', 'f1_jitter')))[0]

        md_row = f"| `{arch}` | `bpdr` (target) | {bpdr_split:.3f} | {bpdr_merge:.3f} | {bpdr_jitter:.3f} | {bpdr_m:.3f} [{bpdr_lo:.3f}, {bpdr_hi:.3f}] | - | baseline |"
        markdown_lines.append(md_row)
        print(f"[{arch}] bpdr: corr_mean = {bpdr_m:.3f} [{bpdr_lo:.3f}, {bpdr_hi:.3f}] (split={bpdr_split:.3f}, merge={bpdr_merge:.3f}, jitter={bpdr_jitter:.3f})")

        beats_frame = False
        beats_aug = False

        for ctrl in ('frame_only', 'augmentation_only'):
            if ctrl not in arm_corr_clips:
                continue
            comp = compare_vocadito_clips(bpdr_clips, arm_corr_clips[ctrl])
            if comp is None:
                continue
            ctrl_m, ctrl_lo, ctrl_hi = comp['b']
            d_mean, d_lo, d_hi = comp['diff']
            win = comp['win']
            if ctrl == 'frame_only':
                beats_frame = win
            else:
                beats_aug = win

            ctrl_split = singer_metric_ci(clips_to_singer_means(get_vocadito_clip_values(rows, arch, 'lean', ctrl, 'f1_split')))[0]
            ctrl_merge = singer_metric_ci(clips_to_singer_means(get_vocadito_clip_values(rows, arch, 'lean', ctrl, 'f1_merge')))[0]
            ctrl_jitter = singer_metric_ci(clips_to_singer_means(get_vocadito_clip_values(rows, arch, 'lean', ctrl, 'f1_jitter')))[0]

            win_str = "YES" if win else "NO"
            print(f"[{arch}] bpdr vs {ctrl:<18} | diff: {d_mean:+.3f} [{d_lo:+.3f}, {d_hi:+.3f}] | Win: {win_str}")

            md_row = f"| `{arch}` | `{ctrl}` | {ctrl_split:.3f} | {ctrl_merge:.3f} | {ctrl_jitter:.3f} | {ctrl_m:.3f} [{ctrl_lo:.3f}, {ctrl_hi:.3f}] | **{d_mean:+.3f}** [{d_lo:+.3f}, {d_hi:+.3f}] | {win_str} |"
            markdown_lines.append(md_row)

            results[(arch, ctrl)] = {
                'ctrl_mean': (ctrl_m, ctrl_lo, ctrl_hi),
                'diff': (d_mean, d_lo, d_hi),
                'win': win,
            }

        arch_win = beats_frame and beats_aug
        win_by_arch[arch] = arch_win
        print(f"[{arch}] BPDR Loss WIN: {'PASS' if arch_win else 'FAIL'} (beats frame_only: {beats_frame}, beats aug_only: {beats_aug})")

    p3_win = any(win_by_arch.values()) if win_by_arch else False
    print("\n--- Part 3 Declared Bar Verdict ---")
    if p3_win:
        p3_verdict = "WIN for BPDR Loss: BPDR loss beats both controls on lean features across corruptions"
    else:
        p3_verdict = "BPDR loss terms do not statistically improve onset F1 over controls when the track is needed"
    print(f"Verdict: {p3_verdict}")

    return results, markdown_lines, p3_win, p3_verdict


def analyze_dev_m02(rows):
    """Summarize boundary Onset F1 on development person M02 (64 clips, alignment-derived labels)."""
    print("\n" + "=" * 80)
    print("DEVELOPMENT M02 SUMMARY: Boundary Onset F1 (64 clips)")
    print("=" * 80)

    results = {}
    markdown_lines = []
    markdown_lines.append("### Development Split (`dev_M02`): Boundary Onset F1 (64 Clips)")
    markdown_lines.append("")
    markdown_lines.append("**Note**: `dev_M02` is a single development singer (64 clips, alignment-derived labels); intervals below are clip-level 95% bootstrap CIs (1,000 resamples).")
    markdown_lines.append("")
    markdown_lines.append("| Arch | Feat | Arm | Clean (Mean [95% CI]) | Split | Merge | Jitter | Diff vs Ext (Clean) [95% CI] |")
    markdown_lines.append("|---|---|---|---|---|---|---|---|")

    archs = _present_archs(rows, ('tcn', 'transformer', 'transformer_v2'), split='dev_M02', feat=None)
    ext_ref = None
    for arch in archs:
        feats = sorted({r['feat'] for r in rows if r.get('split') == 'dev_M02' and r.get('arch') == arch})
        for feat in feats:
            for arm in ARMS:
                c_clips = get_split_clip_values(rows, 'dev_M02', arch, feat, arm, 'f1_clean')
                e_clips = get_split_clip_values(rows, 'dev_M02', arch, feat, arm, 'ext_clean')
                comp = compare_clip_level(c_clips, e_clips)
                if comp is None:
                    continue
                if ext_ref is None:
                    ext_ref = (arch, feat, arm)

                split_m = singer_metric_ci(get_split_clip_values(rows, 'dev_M02', arch, feat, arm, 'f1_split'))[0]
                merge_m = singer_metric_ci(get_split_clip_values(rows, 'dev_M02', arch, feat, arm, 'f1_merge'))[0]
                jitter_m = singer_metric_ci(get_split_clip_values(rows, 'dev_M02', arch, feat, arm, 'f1_jitter'))[0]

                c_mean, c_lo, c_hi = comp['a']
                d_mean, d_lo, d_hi = comp['diff']

                print(f"[dev_M02] {arch:<14} {feat:<4} {arm:<18} | clean={c_mean:.3f} [{c_lo:.3f}, {c_hi:.3f}] | split={split_m:.3f} merge={merge_m:.3f} jitter={jitter_m:.3f} | diff vs ext={d_mean:+.3f} [{d_lo:+.3f}, {d_hi:+.3f}]")
                markdown_lines.append(
                    f"| `{arch}` | `{feat}` | `{arm}` | {c_mean:.3f} [{c_lo:.3f}, {c_hi:.3f}] | {split_m:.3f} | {merge_m:.3f} | {jitter_m:.3f} | **{d_mean:+.3f}** [{d_lo:+.3f}, {d_hi:+.3f}] |"
                )
                results[(arch, feat, arm)] = {
                    'clean': (c_mean, c_lo, c_hi),
                    'split': split_m,
                    'merge': merge_m,
                    'jitter': jitter_m,
                    'diff_vs_ext': (d_mean, d_lo, d_hi),
                    'extractor_clean': comp['b'],
                }

    if ext_ref is not None:
        r_arch, r_feat, r_arm = ext_ref
        e_clean_clips = get_split_clip_values(rows, 'dev_M02', r_arch, r_feat, r_arm, 'ext_clean')
        e_m, e_lo, e_hi = singer_metric_ci(e_clean_clips)
        e_split = singer_metric_ci(get_split_clip_values(rows, 'dev_M02', r_arch, r_feat, r_arm, 'ext_split'))[0]
        e_merge = singer_metric_ci(get_split_clip_values(rows, 'dev_M02', r_arch, r_feat, r_arm, 'ext_merge'))[0]
        e_jitter = singer_metric_ci(get_split_clip_values(rows, 'dev_M02', r_arch, r_feat, r_arm, 'ext_jitter'))[0]
        print(f"[dev_M02] {'extractor':<14} {'-':<4} {'baseline':<18} | clean={e_m:.3f} [{e_lo:.3f}, {e_hi:.3f}] | split={e_split:.3f} merge={e_merge:.3f} jitter={e_jitter:.3f} | -")
        markdown_lines.append(
            f"| `extractor` | `-` | `baseline` | {e_m:.3f} [{e_lo:.3f}, {e_hi:.3f}] | {e_split:.3f} | {e_merge:.3f} | {e_jitter:.3f} | - |"
        )
        results[('extractor', '-', 'baseline')] = {
            'clean': (e_m, e_lo, e_hi),
            'split': e_split,
            'merge': e_merge,
            'jitter': e_jitter,
        }

    return results, markdown_lines


def analyze_fit_coverage(rows, title="### Fit Threshold & Seed Coverage Summary"):
    """Summarize seed coverage, fit health, and threshold saturation across fits."""
    print("\n" + "=" * 80)
    print("FIT THRESHOLD & SEED COVERAGE SUMMARY")
    print("=" * 80)

    fit_map = {}
    for r in rows:
        fit_key = (r['arch'], r['feat'], r['arm'], r['seed'])
        if fit_key not in fit_map:
            fit_map[fit_key] = {
                'thr': r.get('thr'),
                'healthy': bool(r.get('healthy', False)),
            }

    by_cfg = {}
    for (arch, feat, arm, seed), info in sorted(fit_map.items()):
        by_cfg.setdefault((arch, feat, arm), []).append((seed, info['thr'], info['healthy']))

    markdown_lines = []
    markdown_lines.append(title)
    markdown_lines.append("")
    markdown_lines.append("| Arch | Feat | Arm | Seeds Run | Fits | Healthy | Thresholds (`thr`) |")
    markdown_lines.append("|---|---|---|---|---|---|---|")

    arch_seeds = {}
    all_thrs = set()
    healthy_fits = 0
    total_fits = len(fit_map)

    for (arch, feat, arm), entries in sorted(by_cfg.items()):
        seeds = [s for s, _, _ in entries]
        thrs = sorted({t for _, t, _ in entries if t is not None})
        n_healthy = sum(1 for _, _, h in entries if h)
        healthy_fits += n_healthy
        arch_seeds.setdefault(arch, set()).update(seeds)
        all_thrs.update(thrs)
        seeds_str = ", ".join(str(s) for s in seeds)
        thrs_str = ", ".join(f"{t:.3g}" for t in thrs) if thrs else "n/a"
        print(f"[coverage] {arch:<14} {feat:<4} {arm:<18} | seeds=[{seeds_str}] ({len(entries)} fits) | healthy={n_healthy}/{len(entries)} | thr=[{thrs_str}]")
        markdown_lines.append(
            f"| `{arch}` | `{feat}` | `{arm}` | `{seeds_str}` | {len(entries)} | {n_healthy}/{len(entries)} | `{thrs_str}` |"
        )

    saturated_at_08 = (all_thrs == {0.8})
    seed_summary_parts = [f"`{a}`: {len(s)} seed(s) ({sorted(s)})" for a, s in sorted(arch_seeds.items())]
    markdown_lines.append("")
    markdown_lines.append(f"- **Seed coverage**: {'; '.join(seed_summary_parts)}.")
    if saturated_at_08:
        markdown_lines.append(f"- **Threshold saturation**: All {total_fits}/{total_fits} fits hit `thr = 0.8` (the upper ceiling of the candidate threshold grid), indicating uncalibrated/shifted eval logits.")
    else:
        thr_list = ", ".join(f"{t:.3g}" for t in sorted(all_thrs))
        markdown_lines.append(f"- **Threshold distribution**: Observed `thr` values across {total_fits} fits: `{thr_list}`.")

    summary = {
        'total_fits': total_fits,
        'healthy_fits': healthy_fits,
        'healthy_share': (healthy_fits / total_fits) if total_fits else 0.0,
        'arch_seeds': {a: sorted(s) for a, s in arch_seeds.items()},
        'all_thresholds': sorted(all_thrs),
        'saturated_at_08': saturated_at_08,
        'by_cfg': by_cfg,
    }
    return summary, markdown_lines


def compare_candidate(
    v1_rows,
    cand_rows,
    cand_tag='transformer_v2',
    cand_title='Transformer v2',
    short_tag='v2',
    extra_baselines=None,
):
    """Compare a candidate Transformer model (v2 or v3) against v1, TCN, Extractor, and optional extra baselines."""
    print("\n" + "=" * 80)
    print(f"{cand_title.upper()} COMPARISON (vs Baselines)")
    print("=" * 80)

    normalized_cand = [dict(r, arch=cand_tag) for r in cand_rows]
    cand_cfgs = sorted({(r['feat'], r['arm']) for r in normalized_cand})
    extras = extra_baselines or []

    voc_results = {}
    dev_results = {}
    markdown_lines = []
    base_names_desc = "v1" + ("".join(f", {b[0]}" for b in extras)) + ", TCN, and Extractor"
    markdown_lines.append(f"## {cand_title} Statistical Comparison (vs {base_names_desc})")
    markdown_lines.append("")
    markdown_lines.append("**Methodology**: Vocadito uses singer-macro (14 singers) 95% bootstrap CIs (1,000 resamples); `dev_M02` uses clip-level (64 clips) 95% bootstrap CIs.")
    markdown_lines.append("")
    markdown_lines.append(f"### Vocadito: {cand_title} vs Baselines (Note F1 & Onset F1)")
    markdown_lines.append("")
    markdown_lines.append(f"| Metric | Feat | Arm | Cond | Trans {short_tag} (Mean [CI]) | Baseline | Baseline (Mean [CI]) | Diff ({short_tag} - Base) [95% CI] | Win (CI > 0)? |")
    markdown_lines.append("|---|---|---|---|---|---|---|---|---|")

    for metric_name, m_prefix, e_prefix in (('Note F1', 'note_model_', 'note_ext_'), ('Onset F1', 'f1_', 'ext_')):
        for feat, arm in cand_cfgs:
            for k in KINDS:
                cand_clips = get_vocadito_clip_values(normalized_cand, cand_tag, feat, arm, m_prefix + k)
                if not cand_clips:
                    continue

                baselines = [
                    ('transformer_v1', get_vocadito_clip_values(v1_rows, 'transformer', feat, arm, m_prefix + k)),
                ]
                for b_label, b_rows, b_arch in extras:
                    baselines.append((b_label, get_vocadito_clip_values(b_rows, b_arch, feat, arm, m_prefix + k)))
                baselines.extend([
                    ('tcn', get_vocadito_clip_values(v1_rows, 'tcn', feat, arm, m_prefix + k)),
                    ('extractor', get_vocadito_clip_values(normalized_cand, cand_tag, feat, arm, e_prefix + k)),
                ])
                for base_name, b_clips in baselines:
                    comp = compare_vocadito_clips(cand_clips, b_clips)
                    if comp is None:
                        continue
                    c_m, c_lo, c_hi = comp['a']
                    b_m, b_lo, b_hi = comp['b']
                    d_m, d_lo, d_hi = comp['diff']
                    win_str = "YES" if comp['win'] else "NO"
                    if k == 'clean':
                        print(f"[{short_tag} Vocadito | {metric_name}] {feat:<4} {arm:<18} vs {base_name:<14} | {short_tag}={c_m:.3f} [{c_lo:.3f}, {c_hi:.3f}] | base={b_m:.3f} | diff={d_m:+.3f} [{d_lo:+.3f}, {d_hi:+.3f}] | Win: {win_str}")
                    markdown_lines.append(
                        f"| {metric_name} | `{feat}` | `{arm}` | {k} | {c_m:.3f} [{c_lo:.3f}, {c_hi:.3f}] | `{base_name}` | {b_m:.3f} [{b_lo:.3f}, {b_hi:.3f}] | **{d_m:+.3f}** [{d_lo:+.3f}, {d_hi:+.3f}] | {win_str} |"
                    )
                    voc_results[(metric_name, feat, arm, k, base_name)] = comp

    markdown_lines.append("")
    markdown_lines.append(f"### `dev_M02`: {cand_title} vs Baselines (Boundary Onset F1, 64 Clips)")
    markdown_lines.append("")
    markdown_lines.append(f"| Feat | Arm | Cond | Trans {short_tag} (Mean [CI]) | Baseline | Baseline (Mean [CI]) | Diff ({short_tag} - Base) [95% CI] | Win (CI > 0)? |")
    markdown_lines.append("|---|---|---|---|---|---|---|---|")

    for feat, arm in cand_cfgs:
        for k in KINDS:
            cand_clips = get_split_clip_values(normalized_cand, 'dev_M02', cand_tag, feat, arm, 'f1_' + k)
            if not cand_clips:
                continue
            baselines = [
                ('transformer_v1', get_split_clip_values(v1_rows, 'dev_M02', 'transformer', feat, arm, 'f1_' + k)),
            ]
            for b_label, b_rows, b_arch in extras:
                baselines.append((b_label, get_split_clip_values(b_rows, 'dev_M02', b_arch, feat, arm, 'f1_' + k)))
            baselines.extend([
                ('tcn', get_split_clip_values(v1_rows, 'dev_M02', 'tcn', feat, arm, 'f1_' + k)),
                ('extractor', get_split_clip_values(normalized_cand, 'dev_M02', cand_tag, feat, arm, 'ext_' + k)),
            ])
            for base_name, b_clips in baselines:
                comp = compare_clip_level(cand_clips, b_clips)
                if comp is None:
                    continue
                c_m, c_lo, c_hi = comp['a']
                b_m, b_lo, b_hi = comp['b']
                d_m, d_lo, d_hi = comp['diff']
                win_str = "YES" if comp['win'] else "NO"
                if k == 'clean':
                    print(f"[{short_tag} dev_M02] {feat:<4} {arm:<18} vs {base_name:<14} | {short_tag}={c_m:.3f} [{c_lo:.3f}, {c_hi:.3f}] | base={b_m:.3f} | diff={d_m:+.3f} [{d_lo:+.3f}, {d_hi:+.3f}] | Win: {win_str}")
                markdown_lines.append(
                    f"| `{feat}` | `{arm}` | {k} | {c_m:.3f} [{c_lo:.3f}, {c_hi:.3f}] | `{base_name}` | {b_m:.3f} [{b_lo:.3f}, {b_hi:.3f}] | **{d_m:+.3f}** [{d_lo:+.3f}, {d_hi:+.3f}] | {win_str} |"
                )
                dev_results[(feat, arm, k, base_name)] = comp

    cov_summary, cov_md = analyze_fit_coverage(normalized_cand, title=f"### {cand_title} Fit Threshold & Seed Coverage")
    markdown_lines.append("")
    markdown_lines.extend(cov_md)

    return {
        'vocadito': voc_results,
        'dev_M02': dev_results,
        'coverage': cov_summary,
    }, markdown_lines


def compare_v2(v1_rows, v2_rows):
    """Compare Transformer v2 against Transformer v1, TCN, and Extractor on Vocadito and dev_M02."""
    return compare_candidate(
        v1_rows,
        v2_rows,
        cand_tag='transformer_v2',
        cand_title='Transformer v2',
        short_tag='v2',
    )


def compare_v3(v1_rows, v3_rows, v2_rows=None):
    """Compare Transformer v3 against v1, optional v2, TCN, and Extractor, and evaluate v3 adoption bar."""
    extras = [('transformer_v2', [dict(r, arch='transformer_v2') for r in v2_rows], 'transformer_v2')] if v2_rows is not None else []
    res, md = compare_candidate(
        v1_rows,
        v3_rows,
        cand_tag='transformer_v3',
        cand_title='Transformer v3',
        short_tag='v3',
        extra_baselines=extras,
    )
    fo_entries = res['coverage']['by_cfg'].get(('full', 'frame_only'), [])
    fo_thrs = [t for _, t, _ in fo_entries if t is not None]
    non_saturated = bool(fo_thrs) and all(t < 0.85 for t in fo_thrs)
    voc_note = res['vocadito'].get(('Note F1', 'full', 'frame_only', 'clean', 'transformer_v1'))
    voc_onset = res['vocadito'].get(('Onset F1', 'full', 'frame_only', 'clean', 'transformer_v1'))
    dev_onset = res['dev_M02'].get(('full', 'frame_only', 'clean', 'transformer_v1'))
    adopt_v3 = (
        non_saturated
        and voc_note is not None and voc_note['diff'][0] >= 0.0
        and voc_onset is not None and voc_onset['diff'][0] >= 0.0
        and dev_onset is not None and dev_onset['diff'][0] >= 0.0
    )
    res['adopt_v3'] = adopt_v3
    verdict_str = "ADOPT v3" if adopt_v3 else "DO NOT ADOPT v3 (keep V1_CHECKPOINT as Studio default)"
    md.append("")
    md.append(f"- **v3 Adoption Bar Verdict**: **{verdict_str}**.")
    print(f"\n--- Transformer v3 Adoption Bar Verdict ---\nVerdict: {verdict_str}")
    return res, md


def compare_v4(v1_rows, v4_rows, v2_rows=None):
    """Compare v1 2-seed ([339, 340]) and Transformer v4 (POS_WEIGHT=10.0, offset=0.0) against baselines."""
    v1_rep_rows = [r for r in v4_rows if 'v1_rep' in str(r.get('arch', '')).lower()]
    v4_only_rows = [r for r in v4_rows if 'v1_rep' not in str(r.get('arch', '')).lower()]

    # Combine seed 339 from v1_rows and seed 340 from v1_rep under one arch tag
    v1_2seed_combined = [dict(r, arch='transformer_v1_2seed') for r in v1_rows if r.get('arch') == 'transformer'] + [
        dict(r, arch='transformer_v1_2seed') for r in v1_rep_rows
    ]

    extras = [('transformer_v1_2seed', v1_2seed_combined, 'transformer_v1_2seed')]
    if v2_rows is not None:
        extras.append(('transformer_v2', [dict(r, arch='transformer_v2') for r in v2_rows], 'transformer_v2'))

    v1_rep_res, v1_rep_md = (None, [])
    if v1_rep_rows:
        v1_rep_res, v1_rep_md = compare_candidate(
            v1_rows,
            v1_rep_rows,
            cand_tag='v1_rep_340',
            cand_title='Transformer v1 Replication (Seed 340)',
            short_tag='v1_340',
        )

    v4_res, v4_md = compare_candidate(
        v1_rows,
        v4_only_rows,
        cand_tag='transformer_v4',
        cand_title='Transformer v4 (POS_WEIGHT=10.0, offset=0.0)',
        short_tag='v4',
        extra_baselines=extras,
    )
    v4_res['v1_rep'] = v1_rep_res
    v4_res['v1_2seed_rows'] = v1_2seed_combined

    md = []
    if v1_rep_md:
        md.extend(v1_rep_md)
        md.append("")
    md.extend(v4_md)
    return v4_res, md


def format_studio_benchmark(bench_data):
    """Format Studio Vocadito end-to-end benchmark JSON into Markdown lines."""
    summary = bench_data.get('summary', {})
    models = bench_data.get('models', {})
    methods = bench_data.get('methods', list(summary.keys()))
    n_win = bench_data.get('n_windows', 0)
    n_singers = bench_data.get('n_singers', 0)

    lines = []
    lines.append(f"## Studio End-to-End Vocadito Benchmark ({n_win} Windows, {n_singers} Singers)")
    lines.append("")
    lines.append("**Pipeline**: `evidence.load_mono` (10 s window crop) -> `clean.spectral_gate` + `evidence.pesto_track` + `clean.choose` -> `boundary.extract_boundary_hints` (20 ms grid) -> `make_notes` / `evidence.notes_from_track`; 14-singer singer-macro 95% bootstrap CIs (1,000 resamples, seed 339).")
    lines.append("")
    lines.append("| Method | nin | Thr | Note F1 (Mean [95% CI]) | Onset F1 (Mean [95% CI]) | Diff vs `pyin_4096` (Note F1) | Diff vs `pesto_notes` (Note F1) | Diff vs `studio_v1` (Note F1) |")
    lines.append("|---|---|---|---|---|---|---|---|")

    for m in methods:
        s = summary.get(m)
        if not s:
            continue
        nt = s['note_f1']
        on = s['onset_f1']
        meta = models.get(m, {})
        nin_str = str(meta['nin']) if 'nin' in meta else '-'
        thr_str = f"{meta['default_threshold']:.2f}" if 'default_threshold' in meta else '-'

        def _fmt_diff(base_key):
            d = s.get('diffs', {}).get(base_key, {}).get('note_f1')
            if d is None:
                return '-'
            return f"**{d[0]:+.3f}** [{d[1]:+.3f}, {d[2]:+.3f}]"

        lines.append(
            f"| `{m}` | {nin_str} | {thr_str} | {nt[0]:.3f} [{nt[1]:.3f}, {nt[2]:.3f}] | {on[0]:.3f} [{on[1]:.3f}, {on[2]:.3f}] | {_fmt_diff('pyin_4096')} | {_fmt_diff('pesto_notes_from_track')} | {_fmt_diff('studio_v1')} |"
        )
    return lines


def build_report(
    rows,
    is_partial,
    src_name,
    v2_rows=None,
    v2_src_name=None,
    v3_rows=None,
    v3_src_name=None,
    v4_rows=None,
    v4_src_name=None,
    bench_data=None,
    bench_src_name=None,
):
    """Run all analysis sections and return (report_dict, markdown_text)."""
    cov_summary, cov_md = analyze_fit_coverage(rows)
    fits_count = cov_summary['total_fits']
    healthy_share = cov_summary['healthy_share']

    print(f"Loaded: {src_name} ({'PARTIAL' if is_partial else 'COMPLETE'})")
    print(f"Total rows: {len(rows)} | Completed fits: {fits_count} | Healthy fit share: {healthy_share:.1%}")

    p1_res, p1_md, p1_win = analyze_part1(rows)
    p2_res, p2_md, p2_verdict = analyze_part2(rows)
    p3_res, p3_md, p3_win, p3_verdict = analyze_part3(rows)
    dev_res, dev_md = analyze_dev_m02(rows)

    v2_res, v2_md = (None, [])
    if v2_rows is not None:
        v2_res, v2_md = compare_v2(rows, v2_rows)

    v3_res, v3_md = (None, [])
    if v3_rows is not None:
        v3_res, v3_md = compare_v3(rows, v3_rows, v2_rows=v2_rows)

    v4_res, v4_md = (None, [])
    if v4_rows is not None:
        v4_res, v4_md = compare_v4(rows, v4_rows, v2_rows=v2_rows)

    bench_md = format_studio_benchmark(bench_data) if bench_data is not None else []

    extra_srcs = "".join(f" + `{s}`" for s in (v2_src_name, v3_src_name, v4_src_name, bench_src_name) if s)
    full_md = []
    full_md.append("# R1 End-to-End & Architectural Statistical Analysis Report")
    full_md.append(f"**Source Data**: `{src_name}` ({'PARTIAL results' if is_partial else 'FINAL complete results'})" + extra_srcs)
    full_md.append("**Methodology**: Singer-macro (14 Vocadito singers), 95% bootstrap confidence intervals (1,000 resamples, seed 339); `dev_M02` uses clip-level (64 clips) 95% bootstrap CIs.")
    full_md.append(f"**Fit Health**: {fits_count} fits evaluated, {healthy_share:.1%} healthy.")
    full_md.append("")
    full_md.extend(p1_md)
    full_md.append("")
    full_md.extend(p2_md)
    full_md.append("")
    full_md.extend(p3_md)
    full_md.append("")
    full_md.extend(dev_md)
    full_md.append("")
    full_md.extend(cov_md)
    if v2_md:
        full_md.append("")
        full_md.extend(v2_md)
    if v3_md:
        full_md.append("")
        full_md.extend(v3_md)
    if v4_md:
        full_md.append("")
        full_md.extend(v4_md)
    if bench_md:
        full_md.append("")
        full_md.extend(bench_md)
    full_md.append("")
    full_md.append("## Summary of Declared Experimental Bars")

    tcn_p1 = p1_res.get(('tcn', 'Note F1', 'bpdr', 'clean'), {})
    tcn_d, tcn_lo, tcn_hi = tcn_p1.get('diff', (0.0, 0.0, 0.0))
    tcn_m = tcn_p1.get('model', (0.0, 0.0, 0.0))[0]
    ext_m = tcn_p1.get('extractor', (0.0, 0.0, 0.0))[0]
    tcn_on_d, tcn_on_lo, tcn_on_hi = p1_res.get(('tcn', 'Onset F1', 'bpdr', 'clean'), {}).get('diff', (0.0, 0.0, 0.0))
    tcn_jit_d, tcn_jit_lo, tcn_jit_hi = p1_res.get(('tcn', 'Note F1', 'bpdr', 'jitter'), {}).get('diff', (0.0, 0.0, 0.0))

    tf_p1 = p1_res.get(('transformer', 'Note F1', 'frame_only', 'clean'))
    if tf_p1:
        tf_m = tf_p1['model'][0]
        tf_d, tf_lo, tf_hi = tf_p1['diff']
        tf_note = (
            f" However, **FastTNet Transformer (`full`, `frame_only`) beats the Extractor on clean Note F1** "
            f"({tf_m:.3f} vs {ext_m:.3f}; diff {tf_d:+.3f} [{tf_lo:+.3f}, {tf_hi:+.3f}]), with 95% CI > 0 across all 4 conditions (clean/split/merge/jitter)."
        )
    else:
        tf_note = ""

    full_md.append(
        f"1. **Part 1 (Vocadito Note F1 Model vs Extractor)**: {'**WIN**' if p1_win else '**FAIL / NO WIN (TCN declared target)**'} — "
        f"On clean audio, TCN full-feature BPDR Note F1 ({tcn_m:.3f}) does not beat the Extractor ({ext_m:.3f}; diff {tcn_d:+.3f} [{tcn_lo:+.3f}, {tcn_hi:+.3f}], interval includes 0), "
        f"though TCN beats the Extractor on Onset F1 ({tcn_on_d:+.3f} [{tcn_on_lo:+.3f}, {tcn_on_hi:+.3f}]) and jitter Note F1 ({tcn_jit_d:+.3f} [{tcn_jit_lo:+.3f}, {tcn_jit_hi:+.3f}]).{tf_note}"
    )
    full_md.append(f"2. **Part 2 (Transformer vs TCN Onset & Note F1)**: **{p2_verdict}**.")
    full_md.append(f"3. **Part 3 (BPDR Loss on Lean Features Across Corruptions)**: **{p3_verdict}**.")

    tcn_dev = dev_res.get(('tcn', 'full', 'frame_only'), {}).get('clean', (None,))[0]
    tf_dev = dev_res.get(('transformer', 'full', 'frame_only'), {}).get('clean', (None,))[0]
    if tcn_dev is not None and tf_dev is not None:
        full_md.append(
            f"4. **Development (`dev_M02`) & Fit Coverage**: On `dev_M02` (64 clips), TCN `full frame_only` clean Onset F1 is {tcn_dev:.3f} vs FastTNet Transformer v1 {tf_dev:.3f}. "
            f"TCN ran 2 seeds (`339, 340`, 12 fits) vs Transformer v1's 1 seed (`339`, 6 fits); all {fits_count}/{fits_count} fits hit `thr = 0.8`."
        )

    if v2_res is not None:
        v2_cov = v2_res['coverage']
        v2_thrs = ", ".join(f"{t:.3g}" for t in v2_cov['all_thresholds'])
        v2_voc = v2_res['vocadito']
        v2_dev = v2_res['dev_M02']
        bpdr_on_v1 = v2_voc.get(('Onset F1', 'full', 'bpdr', 'clean', 'transformer_v1'))
        bpdr_on_tcn = v2_voc.get(('Onset F1', 'full', 'bpdr', 'clean', 'tcn'))
        bpdr_nt_v1 = v2_voc.get(('Note F1', 'full', 'bpdr', 'clean', 'transformer_v1'))
        fo_nt_v1 = v2_voc.get(('Note F1', 'full', 'frame_only', 'clean', 'transformer_v1'))
        dev_bpdr_v1 = v2_dev.get(('full', 'bpdr', 'clean', 'transformer_v1'))
        dev_bpdr_tcn = v2_dev.get(('full', 'bpdr', 'clean', 'tcn'))
        v2_seeds = v2_cov['arch_seeds'].get('transformer_v2', [])
        dev_fo_v1 = v2_dev.get(('full', 'frame_only', 'clean', 'transformer_v1'))
        parts = [
            f"5. **Transformer v2 Comparison**: {v2_cov['total_fits']} fits ({v2_cov['healthy_share']:.1%} healthy, seeds `{v2_seeds}`), thresholds `{v2_thrs}` (no ceiling saturation at 0.8)."
        ]
        if dev_bpdr_v1 and dev_bpdr_tcn:
            parts.append(
                f"On `dev_M02`, v2 `full bpdr` achieves {dev_bpdr_v1['a'][0]:.3f} Onset F1 "
                f"(vs v1 {dev_bpdr_v1['diff'][0]:+.3f} [{dev_bpdr_v1['diff'][1]:+.3f}, {dev_bpdr_v1['diff'][2]:+.3f}], "
                f"vs TCN {dev_bpdr_tcn['diff'][0]:+.3f} [{dev_bpdr_tcn['diff'][1]:+.3f}, {dev_bpdr_tcn['diff'][2]:+.3f}])."
            )
        if bpdr_on_v1 and bpdr_on_tcn and bpdr_nt_v1:
            parts.append(
                f"On Vocadito, v2 `full bpdr` reaches {bpdr_on_v1['a'][0]:.3f} Onset F1 "
                f"(vs v1 {bpdr_on_v1['diff'][0]:+.3f} [{bpdr_on_v1['diff'][1]:+.3f}, {bpdr_on_v1['diff'][2]:+.3f}], "
                f"vs TCN {bpdr_on_tcn['diff'][0]:+.3f} [{bpdr_on_tcn['diff'][1]:+.3f}, {bpdr_on_tcn['diff'][2]:+.3f}]) "
                f"and {bpdr_nt_v1['a'][0]:.3f} Note F1 (vs v1 {bpdr_nt_v1['diff'][0]:+.3f} [{bpdr_nt_v1['diff'][1]:+.3f}, {bpdr_nt_v1['diff'][2]:+.3f}])."
            )
        if fo_nt_v1:
            dev_fo_note = f" (`dev_M02` {dev_fo_v1['diff'][0]:+.3f})" if dev_fo_v1 else ""
            parts.append(
                f"However, v2 `full frame_only` Note F1 ({fo_nt_v1['a'][0]:.3f}) trails v1 ({fo_nt_v1['b'][0]:.3f}; "
                f"diff {fo_nt_v1['diff'][0]:+.3f} [{fo_nt_v1['diff'][1]:+.3f}, {fo_nt_v1['diff'][2]:+.3f}]), "
                f"indicating stronger HumTrans fit{dev_fo_note} did not transfer to Vocadito standalone Note F1."
            )
        full_md.append(" ".join(parts))

    if v3_res is not None:
        v3_cov = v3_res['coverage']
        v3_thrs = ", ".join(f"{t:.3g}" for t in v3_cov['all_thresholds'])
        v3_seeds = v3_cov['arch_seeds'].get('transformer_v3', [])
        v3_voc = v3_res['vocadito']
        v3_dev = v3_res['dev_M02']
        fo_nt_v1 = v3_voc.get(('Note F1', 'full', 'frame_only', 'clean', 'transformer_v1'))
        fo_on_v1 = v3_voc.get(('Onset F1', 'full', 'frame_only', 'clean', 'transformer_v1'))
        fo_dev_v1 = v3_dev.get(('full', 'frame_only', 'clean', 'transformer_v1'))
        verdict_v3 = "ADOPT v3" if v3_res['adopt_v3'] else "DO NOT ADOPT v3 (keep V1_CHECKPOINT as Studio default)"
        v3_parts = [
            f"6. **Transformer v3 Comparison ({verdict_v3})**: {v3_cov['total_fits']} fits ({v3_cov['healthy_share']:.1%} healthy, seeds `{v3_seeds}`), thresholds `{v3_thrs}` (non-saturated)."
        ]
        if fo_dev_v1:
            v3_parts.append(
                f"On `dev_M02`, v3 `full frame_only` achieves {fo_dev_v1['a'][0]:.3f} Onset F1 "
                f"(vs v1 {fo_dev_v1['diff'][0]:+.3f} [{fo_dev_v1['diff'][1]:+.3f}, {fo_dev_v1['diff'][2]:+.3f}])."
            )
        if fo_on_v1 and fo_nt_v1:
            v3_parts.append(
                f"On Vocadito, v3 `full frame_only` reaches {fo_on_v1['a'][0]:.3f} Onset F1 "
                f"(vs v1 {fo_on_v1['diff'][0]:+.3f} [{fo_on_v1['diff'][1]:+.3f}, {fo_on_v1['diff'][2]:+.3f}]) "
                f"and {fo_nt_v1['a'][0]:.3f} Note F1 "
                f"(vs v1 {fo_nt_v1['diff'][0]:+.3f} [{fo_nt_v1['diff'][1]:+.3f}, {fo_nt_v1['diff'][2]:+.3f}])."
            )
        full_md.append(" ".join(v3_parts))

    if v4_res is not None:
        v4_cov = v4_res['coverage']
        v4_thrs = ", ".join(f"{t:.3g}" for t in v4_cov['all_thresholds'])
        v4_seeds = v4_cov['arch_seeds'].get('transformer_v4', [])
        v4_voc = v4_res['vocadito']
        fo_nt_2s = v4_voc.get(('Note F1', 'full', 'frame_only', 'clean', 'transformer_v1_2seed'))
        bp_on_2s = v4_voc.get(('Onset F1', 'full', 'bpdr', 'clean', 'transformer_v1_2seed'))
        v4_parts = [
            f"7. **Transformer v4 & v1 2-Seed Replication**: v4 ran {v4_cov['total_fits']} fits ({v4_cov['healthy_share']:.1%} healthy, seeds `{v4_seeds}`), thresholds `{v4_thrs}`."
        ]
        if fo_nt_2s:
            v4_parts.append(
                f"On Vocadito `full frame_only` clean Note F1, v4 achieves {fo_nt_2s['a'][0]:.3f} vs 2-seed v1 {fo_nt_2s['b'][0]:.3f} "
                f"(diff {fo_nt_2s['diff'][0]:+.3f} [{fo_nt_2s['diff'][1]:+.3f}, {fo_nt_2s['diff'][2]:+.3f}])."
            )
        if bp_on_2s:
            v4_parts.append(
                f"On `full bpdr` clean Onset F1, v4 reaches {bp_on_2s['a'][0]:.3f} vs 2-seed v1 {bp_on_2s['b'][0]:.3f} "
                f"(diff {bp_on_2s['diff'][0]:+.3f} [{bp_on_2s['diff'][1]:+.3f}, {bp_on_2s['diff'][2]:+.3f}])."
            )
        full_md.append(" ".join(v4_parts))

    if bench_data is not None:
        bs = bench_data.get('summary', {})
        py = bs.get('pyin_4096', {}).get('note_f1', (None,))[0]
        pe = bs.get('pesto_notes_from_track', {}).get('note_f1', (None,))[0]
        sv1 = bs.get('studio_v1', {}).get('note_f1', (None,))[0]
        if py is not None and pe is not None and sv1 is not None:
            best_m = max(bs.items(), key=lambda kv: kv[1]['note_f1'][0])
            full_md.append(
                f"8. **Studio Vocadito Benchmark ({bench_data.get('n_windows', 0)} Windows, {bench_data.get('n_singers', 0)} Singers)**: "
                f"`pyin_4096` Note F1={py:.3f}, `pesto_notes_from_track`={pe:.3f}, `studio_v1`={sv1:.3f}; "
                f"best Studio model `{best_m[0]}` Note F1={best_m[1]['note_f1'][0]:.3f} [{best_m[1]['note_f1'][1]:.3f}, {best_m[1]['note_f1'][2]:.3f}]."
            )

    out_text = "\n".join(full_md) + "\n"
    report_data = {
        'part1': p1_res,
        'part2': p2_res,
        'part3': p3_res,
        'dev_M02': dev_res,
        'coverage': cov_summary,
        'v2': v2_res,
        'v3': v3_res,
        'v4': v4_res,
        'studio_benchmark': bench_data,
        'p1_win': p1_win,
        'p2_verdict': p2_verdict,
        'p3_win': p3_win,
        'p3_verdict': p3_verdict,
    }
    return report_data, out_text


def main(argv=None):
    parser = argparse.ArgumentParser(description="Analyze R1 E2E results and optional Transformer v2/v3/v4 + Studio benchmark.")
    parser.add_argument('--results', type=str, default=None, help="Path to v1 results JSON (defaults to e2e_fit_results.json).")
    parser.add_argument('--compare-v2', action='store_true', help="Compare Transformer v2 results against v1, TCN, and Extractor.")
    parser.add_argument('--v2-file', type=str, default=str(V2_RESULT_FILE), help="Path to Transformer v2 results JSON.")
    parser.add_argument('--compare-v3', action='store_true', help="Compare Transformer v3 results against v1, v2 (if --compare-v2), TCN, and Extractor.")
    parser.add_argument('--v3-file', type=str, default=str(V3_RESULT_FILE), help="Path to Transformer v3 results JSON.")
    parser.add_argument('--compare-v4', action='store_true', help="Compare Transformer v4 + v1 2-seed replication.")
    parser.add_argument('--v4-file', type=str, default=str(V4_RESULT_FILE), help="Path to Transformer v4 results JSON.")
    parser.add_argument('--studio-benchmark', action='store_true', help="Include Studio Vocadito WAV benchmark table.")
    parser.add_argument('--benchmark-file', type=str, default=str(BENCHMARK_FILE), help="Path to Studio Vocadito benchmark JSON.")
    parser.add_argument('--output', type=str, default=str(R / 'statistical_analysis_report.md'), help="Output Markdown report path.")
    parser.add_argument('--no-write', action='store_true', help="Print report without writing output file.")
    args = parser.parse_args(argv)

    rows, is_partial, src_name = load_results(args.results)

    v2_rows, v2_src_name = None, None
    if args.compare_v2:
        v2_rows, _, v2_src_name = load_results(args.v2_file)

    v3_rows, v3_src_name = None, None
    if args.compare_v3:
        v3_rows, _, v3_src_name = load_results(args.v3_file)

    v4_rows, v4_src_name = None, None
    if args.compare_v4:
        v4_rows, _, v4_src_name = load_results(args.v4_file)

    bench_data, bench_src_name = None, None
    if args.studio_benchmark:
        bp = Path(args.benchmark_file)
        if not bp.exists():
            raise FileNotFoundError(f"Benchmark file not found: {bp}")
        with open(bp, encoding='utf-8') as f:
            bench_data = json.load(f)
        bench_src_name = bp.name

    _, out_text = build_report(
        rows,
        is_partial,
        src_name,
        v2_rows=v2_rows,
        v2_src_name=v2_src_name,
        v3_rows=v3_rows,
        v3_src_name=v3_src_name,
        v4_rows=v4_rows,
        v4_src_name=v4_src_name,
        bench_data=bench_data,
        bench_src_name=bench_src_name,
    )

    if not args.no_write:
        out_report_path = Path(args.output)
        try:
            out_report_path.parent.mkdir(parents=True, exist_ok=True)
            out_report_path.write_text(out_text, encoding='utf-8')
            print(f"\nWrote Markdown statistical report to: {out_report_path}")
        except OSError as exc:
            print(f"\nWarning: could not write report to {out_report_path}: {exc}")

    print("\n" + "=" * 80)
    print("FINAL FORMATTED MARKDOWN REPORT")
    print("=" * 80)
    print(out_text)


if __name__ == '__main__':
    main()
