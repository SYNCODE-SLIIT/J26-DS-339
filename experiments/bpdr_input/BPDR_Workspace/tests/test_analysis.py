import json
import tempfile
import unittest
from pathlib import Path
import numpy as np

from tools.analyze_r1_e2e import (
    BENCHMARK_FILE,
    V2_RESULT_FILE,
    V3_RESULT_FILE,
    V4_RESULT_FILE,
    R,
    KINDS,
    bootstrap_ci,
    singer_metric_ci,
    clips_to_singer_means,
    get_vocadito_clip_values,
    analyze_part1,
    analyze_part2,
    analyze_part3,
    analyze_dev_m02,
    analyze_fit_coverage,
    compare_v2,
    build_report,
    load_results,
    main,
)


class AnalysisTests(unittest.TestCase):
    def test_bootstrap_ci_positive_shift(self):
        diffs = {f"singer_{i}": 0.05 + 0.01 * (i % 3) for i in range(14)}
        mean_val, lo, hi = bootstrap_ci(diffs, n_resamples=200, seed=42)
        self.assertAlmostEqual(mean_val, 0.06, places=2)
        self.assertGreater(lo, 0.0)
        self.assertGreater(hi, lo)

    def test_bootstrap_ci_zero_centered(self):
        diffs = {f"singer_{i}": -0.05 if i % 2 == 0 else 0.05 for i in range(14)}
        mean_val, lo, hi = bootstrap_ci(diffs, n_resamples=200, seed=42)
        self.assertAlmostEqual(mean_val, 0.0, places=2)
        self.assertLess(lo, 0.0)
        self.assertGreater(hi, 0.0)

    def test_singer_metric_ci(self):
        vals = {f"singer_{i}": 0.70 + 0.02 * i for i in range(14)}
        m, lo, hi = singer_metric_ci(vals, n_resamples=200, seed=42)
        self.assertAlmostEqual(m, float(np.mean(list(vals.values()))), places=4)
        self.assertLessEqual(lo, m)
        self.assertGreaterEqual(hi, m)

    def test_clips_to_singer_means(self):
        clips = {
            ("singer_1", "c1"): 0.8,
            ("singer_1", "c2"): 0.6,
            ("singer_2", "c1"): 0.5,
        }
        res = clips_to_singer_means(clips)
        self.assertAlmostEqual(res["singer_1"], 0.7)
        self.assertAlmostEqual(res["singer_2"], 0.5)

    def test_get_vocadito_clip_values(self):
        rows = [
            {"split": "vocadito", "arch": "tcn", "feat": "full", "arm": "bpdr", "person": "P1", "id": "1", "f1_clean": 0.75},
            {"split": "vocadito", "arch": "tcn", "feat": "full", "arm": "bpdr", "person": "P1", "id": "1", "f1_clean": 0.85},
            {"split": "humtrans", "arch": "tcn", "feat": "full", "arm": "bpdr", "person": "P2", "id": "2", "f1_clean": 0.50},
        ]
        clips = get_vocadito_clip_values(rows, "tcn", "full", "bpdr", "f1_clean")
        self.assertEqual(len(clips), 1)
        self.assertAlmostEqual(clips[("P1", "1")], 0.80)

    def test_integration_full_analysis(self):
        rows, is_partial, _ = load_results()
        self.assertGreater(len(rows), 0)
        self.assertFalse(is_partial)

        p1_res, p1_md, p1_win = analyze_part1(rows)
        self.assertIn(('Note F1', 'bpdr', 'clean'), p1_res)
        self.assertIn(('tcn', 'Note F1', 'bpdr', 'clean'), p1_res)
        self.assertFalse(p1_win)  # TCN declared bar: NO WIN on clean Note F1

        # FastTNet Transformer full frame_only beats Extractor across all 4 conditions
        for k in KINDS:
            entry = p1_res[('transformer', 'Note F1', 'frame_only', k)]
            self.assertTrue(entry['win'], msg=f"Expected transformer frame_only Note F1 win on {k}")
        tf_clean = p1_res[('transformer', 'Note F1', 'frame_only', 'clean')]
        self.assertAlmostEqual(tf_clean['model'][0], 0.601, places=3)
        self.assertAlmostEqual(tf_clean['extractor'][0], 0.562, places=3)
        self.assertAlmostEqual(tf_clean['diff'][0], 0.039, places=3)
        self.assertTrue(any('`transformer`' in line for line in p1_md))

        _, _, p2_verdict = analyze_part2(rows)
        self.assertEqual(p2_verdict, "ADOPT TRANSFORMER")

        _, _, p3_win, _ = analyze_part3(rows)
        self.assertFalse(p3_win)

        dev_res, dev_md = analyze_dev_m02(rows)
        self.assertIn(('tcn', 'full', 'frame_only'), dev_res)
        self.assertIn(('transformer', 'full', 'frame_only'), dev_res)
        self.assertIn(('extractor', '-', 'baseline'), dev_res)
        self.assertAlmostEqual(dev_res[('tcn', 'full', 'frame_only')]['clean'][0], 0.780, places=3)
        self.assertAlmostEqual(dev_res[('transformer', 'full', 'frame_only')]['clean'][0], 0.754, places=3)
        self.assertTrue(any('dev_M02' in line for line in dev_md))

        cov, cov_md = analyze_fit_coverage(rows)
        self.assertEqual(cov['total_fits'], 18)
        self.assertAlmostEqual(cov['healthy_share'], 1.0)
        self.assertEqual(cov['arch_seeds'], {'tcn': [339, 340], 'transformer': [339]})
        self.assertTrue(cov['saturated_at_08'])
        self.assertTrue(any('thr = 0.8' in line for line in cov_md))

    def test_compare_v2_mode_and_cli(self):
        v1_rows, is_partial, src_name = load_results()
        # Construct synthetic v2 rows from transformer full frame_only with +0.05 improvement and calibrated thr
        base_tf = [
            r for r in v1_rows
            if r['arch'] == 'transformer' and r['feat'] == 'full' and r['arm'] == 'frame_only'
        ]
        v2_rows = []
        for seed, thr in ((339, 0.50), (340, 0.55)):
            for r in base_tf:
                nr = dict(r)
                nr['arch'] = 'transformer_v2'
                nr['seed'] = seed
                nr['thr'] = thr
                nr['healthy'] = True
                for k in KINDS:
                    nr['f1_' + k] = min(1.0, nr['f1_' + k] + 0.05)
                    if 'note_model_' + k in nr:
                        nr['note_model_' + k] = min(1.0, nr['note_model_' + k] + 0.05)
                v2_rows.append(nr)

        v2_res, v2_md = compare_v2(v1_rows, v2_rows)
        self.assertTrue(v2_res['vocadito'][('Note F1', 'full', 'frame_only', 'clean', 'transformer_v1')]['win'])
        self.assertTrue(v2_res['vocadito'][('Onset F1', 'full', 'frame_only', 'clean', 'tcn')]['win'])
        self.assertTrue(v2_res['vocadito'][('Note F1', 'full', 'frame_only', 'clean', 'extractor')]['win'])
        self.assertTrue(v2_res['dev_M02'][('full', 'frame_only', 'clean', 'transformer_v1')]['win'])
        self.assertFalse(v2_res['coverage']['saturated_at_08'])
        self.assertEqual(v2_res['coverage']['arch_seeds'], {'transformer_v2': [339, 340]})
        self.assertTrue(any('Transformer v2 Statistical Comparison' in line for line in v2_md))

        _, md_text = build_report(v1_rows, is_partial, src_name, v2_rows=v2_rows, v2_src_name='synthetic_v2.json')
        self.assertIn('Transformer v2 Statistical Comparison', md_text)

        with tempfile.TemporaryDirectory() as tmpdir:
            v2_path = Path(tmpdir) / 'v2.json'
            v2_path.write_text(json.dumps({'rows': v2_rows}), encoding='utf-8')
            main(['--compare-v2', '--v2-file', str(v2_path), '--no-write'])

    def test_real_v2_v3_v4_benchmark_and_report_parity(self):
        self.assertTrue(V2_RESULT_FILE.exists())
        self.assertTrue(V3_RESULT_FILE.exists())
        self.assertTrue(V4_RESULT_FILE.exists())
        self.assertTrue(BENCHMARK_FILE.exists())
        v1_rows, is_partial, src_name = load_results()
        v2_rows, v2_partial, v2_name = load_results(V2_RESULT_FILE)
        v3_rows, v3_partial, v3_name = load_results(V3_RESULT_FILE)
        v4_rows, v4_partial, v4_name = load_results(V4_RESULT_FILE)
        bench_data = json.loads(BENCHMARK_FILE.read_text(encoding='utf-8'))
        self.assertFalse(v2_partial)
        self.assertFalse(v3_partial)
        self.assertFalse(v4_partial)
        self.assertEqual(len(v2_rows), 1164)
        self.assertEqual(len(v3_rows), 1164)
        self.assertEqual(len(v4_rows), 582)

        report_data, md_text = build_report(
            v1_rows,
            is_partial,
            src_name,
            v2_rows=v2_rows,
            v2_src_name=v2_name,
            v3_rows=v3_rows,
            v3_src_name=v3_name,
            v4_rows=v4_rows,
            v4_src_name=v4_name,
            bench_data=bench_data,
            bench_src_name=BENCHMARK_FILE.name,
        )
        v2_res = report_data['v2']
        self.assertIsNotNone(v2_res)
        self.assertEqual(v2_res['coverage']['total_fits'], 12)
        self.assertAlmostEqual(v2_res['coverage']['healthy_share'], 1.0)
        self.assertFalse(v2_res['coverage']['saturated_at_08'])
        self.assertEqual(v2_res['coverage']['arch_seeds'], {'transformer_v2': [339, 340]})

        # dev_M02 v2 full bpdr beats both v1 and tcn
        self.assertTrue(v2_res['dev_M02'][('full', 'bpdr', 'clean', 'transformer_v1')]['win'])
        self.assertTrue(v2_res['dev_M02'][('full', 'bpdr', 'clean', 'tcn')]['win'])
        # Vocadito v2 full bpdr beats tcn and extractor on Onset F1
        self.assertTrue(v2_res['vocadito'][('Onset F1', 'full', 'bpdr', 'clean', 'tcn')]['win'])
        self.assertTrue(v2_res['vocadito'][('Onset F1', 'full', 'bpdr', 'clean', 'extractor')]['win'])

        v3_res = report_data['v3']
        self.assertIsNotNone(v3_res)
        self.assertEqual(v3_res['coverage']['total_fits'], 12)
        self.assertAlmostEqual(v3_res['coverage']['healthy_share'], 1.0)
        self.assertFalse(v3_res['adopt_v3'])
        self.assertEqual(v3_res['coverage']['arch_seeds'], {'transformer_v3': [339, 340]})
        self.assertTrue(v3_res['dev_M02'][('full', 'frame_only', 'clean', 'transformer_v1')]['win'])
        self.assertLess(v3_res['vocadito'][('Note F1', 'full', 'frame_only', 'clean', 'transformer_v1')]['diff'][0], 0.0)
        self.assertLess(v3_res['vocadito'][('Onset F1', 'full', 'frame_only', 'clean', 'transformer_v1')]['diff'][0], 0.0)

        v4_res = report_data['v4']
        self.assertIsNotNone(v4_res)
        self.assertEqual(v4_res['coverage']['total_fits'], 4)
        self.assertAlmostEqual(v4_res['coverage']['healthy_share'], 1.0)
        self.assertEqual(v4_res['coverage']['arch_seeds'], {'transformer_v4': [339, 340]})
        self.assertEqual(v4_res['v1_rep']['coverage']['total_fits'], 2)
        self.assertEqual(v4_res['v1_rep']['coverage']['arch_seeds'], {'v1_rep_340': [340]})
        # Item 1a: v4 improves on dev_M02 over 2-seed v1, but loses to 2-seed v1 on Vocadito clean
        self.assertTrue(v4_res['dev_M02'][('full', 'bpdr', 'clean', 'transformer_v1_2seed')]['win'])
        self.assertLess(v4_res['vocadito'][('Note F1', 'full', 'frame_only', 'clean', 'transformer_v1_2seed')]['diff'][0], 0.0)
        self.assertLess(v4_res['vocadito'][('Onset F1', 'full', 'bpdr', 'clean', 'transformer_v1_2seed')]['diff'][0], 0.0)

        # Item 1c: Real Studio end-to-end Vocadito benchmark (39 windows, 14 singers)
        self.assertEqual(bench_data['n_windows'], 39)
        self.assertEqual(bench_data['n_singers'], 14)
        s_v1 = bench_data['summary']['studio_v1']
        self.assertGreater(s_v1['diffs']['pyin_4096']['note_f1'][1], 0.0)
        self.assertGreater(s_v1['diffs']['pesto_notes_from_track']['note_f1'][1], 0.0)
        self.assertLess(bench_data['summary']['studio_v1_pre_parity_100hz_thr05']['diffs']['studio_v1']['note_f1'][2], 0.0)
        self.assertLess(bench_data['summary']['studio_v4']['diffs']['studio_v1']['note_f1'][2], 0.0)
        self.assertLess(bench_data['summary']['studio_v4_bpdr']['diffs']['studio_v1']['note_f1'][2], 0.0)

        report_path = R / 'statistical_analysis_report.md'
        self.assertEqual(report_path.read_text(encoding='utf-8').replace('\r\n', '\n'), md_text.replace('\r\n', '\n'))


if __name__ == '__main__':
    unittest.main()
