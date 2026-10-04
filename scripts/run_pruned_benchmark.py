"""Benchmark Pruned Models vs Baselines.

Protocol:
1. Subject-Disjoint:
   - Train: subjects 01, 04, 06, 07, 08, 09, 13, 14
   - Val:   subjects 02, 03, 11
   - Test:  subjects 05, 10, 12
2. Validation Threshold Selection:
   - Evaluates each checkpoint on the validation split.
   - Finds tau* = argmax_tau F1(tau) on val (no artificial floor rules).
   - Freezes tau* permanently.
3. Sequential Testing:
   - Evaluates sequentially on the held-out test split at the frozen tau*.
   - Zero test re-tuning or peeking.
   - Computes mAP50, Macro Recall, Min Safety Recall, Precision, F1, and per-class recalls.
   - Also evaluates at standard default tau=0.25 for direct reference.
   - Computes paired deltas vs same-seed baseline and seed-aggregated Mean +/- SD.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / 'results' / 'phase0' / 'cache_manifest.csv'
GT_PATH = REPO_ROOT / 'results' / 'phase0' / 'ground_truth.parquet'
CANONICAL_AP_PATH = REPO_ROOT / 'results' / 'phase0' / 'all_evaluated_checkpoints_canonical.csv'
RESULTS_DIR = REPO_ROOT / 'results'

MODELS = ['yolo11n', 'yolo26n']
RATIOS = ['0%', '10%', '20%', '30%', '40%', '50%']
SEEDS = [0, 1, 2]
CLASS_NAMES = ['yawning', 'hand_over_mouth', 'drinking', 'phone_use']

# Parameter counts (millions) and sizes (MB)
MODEL_STATS = {
    'yolo11n': {
        '0%':  {'params_m': 2.59, 'size_mb': 5.23, 'param_red_pct': 0.0},
        '10%': {'params_m': 2.24, 'size_mb': 4.59, 'param_red_pct': -13.5},
        '20%': {'params_m': 1.94, 'size_mb': 4.02, 'param_red_pct': -24.9},
        '30%': {'params_m': 1.67, 'size_mb': 3.50, 'param_red_pct': -35.6},
        '40%': {'params_m': 1.43, 'size_mb': 3.04, 'param_red_pct': -44.8},
        '50%': {'params_m': 1.23, 'size_mb': 2.65, 'param_red_pct': -52.5},
    },
    'yolo26n': {
        '0%':  {'params_m': 2.51, 'size_mb': 5.15, 'param_red_pct': 0.0},
        '10%': {'params_m': 2.18, 'size_mb': 4.56, 'param_red_pct': -12.8},
        '20%': {'params_m': 1.91, 'size_mb': 4.04, 'param_red_pct': -23.8},
        '30%': {'params_m': 1.66, 'size_mb': 3.55, 'param_red_pct': -33.9},
        '40%': {'params_m': 1.43, 'size_mb': 3.13, 'param_red_pct': -42.8},
        '50%': {'params_m': 1.25, 'size_mb': 2.77, 'param_red_pct': -50.2},
    }
}


def run_benchmark():
    manifest = pd.read_csv(MANIFEST_PATH)
    gt = pd.read_parquet(GT_PATH)
    ap_df = pd.read_csv(CANONICAL_AP_PATH)

    val_gt = gt[gt['split'] == 'val']
    test_gt = gt[gt['split'] == 'test']
    val_gt_count = len(val_gt)
    test_gt_count = len(test_gt)

    per_seed_records = []

    print("=" * 80)
    print("EXECUTING PRUNED MODELS BENCHMARK (VALIDATION-FROZEN TAU)")
    print("=" * 80)

    for model in MODELS:
        for ratio in RATIOS:
            for seed in SEEDS:
                # 1. Look up manifest records
                val_rows = manifest[
                    (manifest['model'] == model) &
                    (manifest['prune_ratio'] == ratio) &
                    (manifest['seed'] == seed) &
                    (manifest['split'] == 'val') &
                    (manifest['ckpt_type'] == 'best')
                ]
                test_rows = manifest[
                    (manifest['model'] == model) &
                    (manifest['prune_ratio'] == ratio) &
                    (manifest['seed'] == seed) &
                    (manifest['split'] == 'test') &
                    (manifest['ckpt_type'] == 'best')
                ]
                if len(val_rows) == 0 or len(test_rows) == 0:
                    raise RuntimeError(f"Missing cache entry for {model} {ratio} seed {seed}")

                df_val = pd.read_parquet(REPO_ROOT / val_rows.iloc[0]['parquet_path'])
                df_test = pd.read_parquet(REPO_ROOT / test_rows.iloc[0]['parquet_path'])

                # 2. Validation Phase: Select tau* by maximizing F1 on val split
                best_val_f1, best_tau = 0.0, 0.25
                best_val_prec, best_val_rec = 0.0, 0.0
                confs = np.linspace(0.01, 0.95, 95)
                for c in confs:
                    tp = ((df_val['conf'] >= c) & (df_val['matched'] == True)).sum()
                    fp = ((df_val['conf'] >= c) & (df_val['matched'] == False)).sum()
                    rec = tp / val_gt_count if val_gt_count > 0 else 0.0
                    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                    if f1 > best_val_f1:
                        best_val_f1 = float(f1)
                        best_tau = float(c)
                        best_val_prec = float(prec)
                        best_val_rec = float(rec)

                # 3. Sequential Testing Phase: Evaluate strictly at FROZEN tau*
                tp_test = (df_test['conf'] >= best_tau) & (df_test['matched'] == True)
                fp_test = (df_test['conf'] >= best_tau) & (df_test['matched'] == False)
                n_tp = int(tp_test.sum())
                n_fp = int(fp_test.sum())
                test_prec = n_tp / (n_tp + n_fp) if (n_tp + n_fp) > 0 else 0.0

                test_rec_per_cls = {}
                for cid, cname in enumerate(CLASS_NAMES):
                    gt_c = int((test_gt['class_id'] == cid).sum())
                    tp_c = int((tp_test & (df_test['class_id'] == cid)).sum())
                    test_rec_per_cls[cname] = tp_c / gt_c if gt_c > 0 else 0.0

                test_macro_rec = float(np.mean(list(test_rec_per_cls.values())))
                test_min_rec = float(min(test_rec_per_cls.values()))
                test_f1 = (2 * test_prec * test_macro_rec / (test_prec + test_macro_rec)
                           if (test_prec + test_macro_rec) > 0 else 0.0)

                # Direct comparison at fixed canonical tau = 0.25
                tp_25 = (df_test['conf'] >= 0.25) & (df_test['matched'] == True)
                fp_25 = (df_test['conf'] >= 0.25) & (df_test['matched'] == False)
                prec_25 = float(tp_25.sum() / (tp_25.sum() + fp_25.sum())) if (tp_25.sum() + fp_25.sum()) > 0 else 0.0
                rec_per_cls_25 = {}
                for cid, cname in enumerate(CLASS_NAMES):
                    gt_c = int((test_gt['class_id'] == cid).sum())
                    tp_c = int((tp_25 & (df_test['class_id'] == cid)).sum())
                    rec_per_cls_25[cname] = tp_c / gt_c if gt_c > 0 else 0.0
                macro_rec_25 = float(np.mean(list(rec_per_cls_25.values())))
                min_rec_25 = float(min(rec_per_cls_25.values()))
                f1_25 = (2 * prec_25 * macro_rec_25 / (prec_25 + macro_rec_25)
                         if (prec_25 + macro_rec_25) > 0 else 0.0)

                # Canonical mAP metrics
                ap_row = ap_df[
                    (ap_df['model'] == model) &
                    (ap_df['prune_ratio'] == ratio) &
                    (ap_df['seed'] == seed)
                ].iloc[0]

                stats = MODEL_STATS[model][ratio]

                per_seed_records.append({
                    'model': model,
                    'prune_ratio': ratio,
                    'seed': seed,
                    'tau_star': best_tau,
                    'val_f1': best_val_f1,
                    'val_prec': best_val_prec,
                    'val_rec': best_val_rec,
                    'test_mAP50': float(ap_row['mAP50']),
                    'test_min_ap50': float(ap_row['min_ap50']),
                    'test_prec_tau_star': test_prec,
                    'test_macro_rec_tau_star': test_macro_rec,
                    'test_min_rec_tau_star': test_min_rec,
                    'test_f1_tau_star': test_f1,
                    'test_rec_yawn_tau_star': test_rec_per_cls['yawning'],
                    'test_rec_hom_tau_star': test_rec_per_cls['hand_over_mouth'],
                    'test_rec_drink_tau_star': test_rec_per_cls['drinking'],
                    'test_rec_phone_tau_star': test_rec_per_cls['phone_use'],
                    'test_min_rec_tau_25': min_rec_25,
                    'test_macro_rec_tau_25': macro_rec_25,
                    'test_f1_tau_25': f1_25,
                    'params_m': stats['params_m'],
                    'size_mb': stats['size_mb'],
                    'param_red_pct': stats['param_red_pct'],
                })

    df = pd.DataFrame(per_seed_records)

    # 4. Compute Paired Deltas (Pruned - Same-Seed Baseline) in Percentage Points
    for model in MODELS:
        base_df = df[(df['model'] == model) & (df['prune_ratio'] == '0%')].set_index('seed')
        for metric in [
            'test_mAP50', 'test_min_ap50',
            'test_macro_rec_tau_star', 'test_min_rec_tau_star',
            'test_f1_tau_star', 'test_min_rec_tau_25'
        ]:
            delta_col = f'delta_{metric}_pp'
            df.loc[df['model'] == model, delta_col] = df[df['model'] == model].apply(
                lambda r: (r[metric] - base_df.loc[r['seed'], metric]) * 100.0, axis=1
            )

    # 5. Aggregate Across Seeds (Mean +/- SD)
    aggregated = []
    for model in MODELS:
        m_sub = df[df['model'] == model]
        for ratio in RATIOS:
            r_sub = m_sub[m_sub['prune_ratio'] == ratio]
            agg = {
                'model': model,
                'prune_ratio': ratio,
                'tau_star_mean': float(r_sub['tau_star'].mean()),
                'tau_star_sd': float(r_sub['tau_star'].std()),
                'val_f1_mean': float(r_sub['val_f1'].mean()) * 100,
                'val_f1_sd': float(r_sub['val_f1'].std()) * 100,
                'test_mAP50_mean': float(r_sub['test_mAP50'].mean()) * 100,
                'test_mAP50_sd': float(r_sub['test_mAP50'].std()) * 100,
                'delta_mAP50_pp': float(r_sub['delta_test_mAP50_pp'].mean()),
                'test_min_ap50_mean': float(r_sub['test_min_ap50'].mean()) * 100,
                'test_min_ap50_sd': float(r_sub['test_min_ap50'].std()) * 100,
                'test_macro_rec_mean': float(r_sub['test_macro_rec_tau_star'].mean()) * 100,
                'test_macro_rec_sd': float(r_sub['test_macro_rec_tau_star'].std()) * 100,
                'delta_macro_rec_pp': float(r_sub['delta_test_macro_rec_tau_star_pp'].mean()),
                'test_min_rec_mean': float(r_sub['test_min_rec_tau_star'].mean()) * 100,
                'test_min_rec_sd': float(r_sub['test_min_rec_tau_star'].std()) * 100,
                'delta_min_rec_pp': float(r_sub['delta_test_min_rec_tau_star_pp'].mean()),
                'test_f1_mean': float(r_sub['test_f1_tau_star'].mean()) * 100,
                'test_f1_sd': float(r_sub['test_f1_tau_star'].std()) * 100,
                'delta_f1_pp': float(r_sub['delta_test_f1_tau_star_pp'].mean()),
                'test_min_rec_25_mean': float(r_sub['test_min_rec_tau_25'].mean()) * 100,
                'test_min_rec_25_sd': float(r_sub['test_min_rec_tau_25'].std()) * 100,
                'delta_min_rec_25_pp': float(r_sub['delta_test_min_rec_tau_25_pp'].mean()),
                'rec_yawn_mean': float(r_sub['test_rec_yawn_tau_star'].mean()) * 100,
                'rec_hom_mean': float(r_sub['test_rec_hom_tau_star'].mean()) * 100,
                'rec_drink_mean': float(r_sub['test_rec_drink_tau_star'].mean()) * 100,
                'rec_phone_mean': float(r_sub['test_rec_phone_tau_star'].mean()) * 100,
                'params_m': r_sub.iloc[0]['params_m'],
                'size_mb': r_sub.iloc[0]['size_mb'],
                'param_red_pct': r_sub.iloc[0]['param_red_pct'],
            }
            aggregated.append(agg)

    df_agg = pd.DataFrame(aggregated)

    # 6. Save JSON records
    output_json = {
        'metadata': {
            'protocol': 'Validation-Frozen Confidence Threshold (Max F1 on Val)',
            'splits': {
                'train': ['01', '04', '06', '07', '08', '09', '13', '14'],
                'val': ['02', '03', '11'],
                'test': ['05', '10', '12'],
                'subject_disjoint': True,
            },
            'num_models_evaluated': len(per_seed_records),
        },
        'per_seed_results': per_seed_records,
        'aggregated_results': aggregated,
    }
    with open(RESULTS_DIR / 'pruned_benchmark_results.json', 'w') as f:
        json.dump(output_json, f, indent=2)

    # 7. Generate Comprehensive Markdown Summary Table
    md_lines = [
        "# Benchmark Results: Pruned Models vs Baselines",
        "",
        "> **Protocol:** Subject-disjoint splits (Val: subjects 02, 03, 11; Test: subjects 05, 10, 12).  ",
        "> **Calibration:** Confidence threshold $\\tau^*$ selected by maximizing $F_1$ on the validation set, then permanently **frozen** for sequential testing.",
        "",
        "## Summary Table (Mean ± SD across 3 Seeds)",
        "",
        "| Architecture | Prune Ratio | Params (M) | Red. (%) | Val $\\tau^*$ | Val $F_1$ (%) | Test $\\text{mAP}_{50}$ (%) | $\\Delta\\text{mAP}_{50}$ | Test Macro Rec (%) | Test Min Rec (%) | $\\Delta\\text{Min Rec}$ | Test $F_1$ (%) | Test Min Rec ($\\tau{=}0.25$) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for row in aggregated:
        model_name = "YOLO11n" if row['model'] == 'yolo11n' else "YOLO26n"
        ratio_str = "Baseline (0%)" if row['prune_ratio'] == '0%' else f"Pruned {row['prune_ratio']}"
        params_str = f"{row['params_m']:.2f}"
        red_str = f"{row['param_red_pct']:+.1f}%" if row['param_red_pct'] != 0 else "—"
        tau_str = f"{row['tau_star_mean']:.2f} ± {row['tau_star_sd']:.2f}"
        val_f1_str = f"{row['val_f1_mean']:.1f} ± {row['val_f1_sd']:.1f}"
        map_str = f"{row['test_mAP50_mean']:.2f} ± {row['test_mAP50_sd']:.2f}"
        d_map_str = f"{row['delta_mAP50_pp']:+.2f} pp" if row['prune_ratio'] != '0%' else "—"
        macro_rec_str = f"{row['test_macro_rec_mean']:.1f} ± {row['test_macro_rec_sd']:.1f}"
        min_rec_str = f"{row['test_min_rec_mean']:.1f} ± {row['test_min_rec_sd']:.1f}"
        d_min_str = f"{row['delta_min_rec_pp']:+.2f} pp" if row['prune_ratio'] != '0%' else "—"
        f1_str = f"{row['test_f1_mean']:.1f} ± {row['test_f1_sd']:.1f}"
        rec_25_str = f"{row['test_min_rec_25_mean']:.1f} ({row['delta_min_rec_25_pp']:+.2f} pp)" if row['prune_ratio'] != '0%' else f"{row['test_min_rec_25_mean']:.1f} ± {row['test_min_rec_25_sd']:.1f}"

        md_lines.append(
            f"| {model_name} | {ratio_str} | {params_str} | {red_str} | {tau_str} | {val_f1_str} | {map_str} | {d_map_str} | {macro_rec_str} | {min_rec_str} | {d_min_str} | {f1_str} | {rec_25_str} |"
        )

    md_lines.extend([
        "",
        "## Per-Class Safety Recall at Frozen $\\tau^*$ (Mean %)",
        "",
        "| Architecture | Prune Ratio | Yawning Recall (%) | Hand-Over-Mouth Recall (%) | Drinking Recall (%) | Phone Use Recall (%) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ])

    for row in aggregated:
        model_name = "YOLO11n" if row['model'] == 'yolo11n' else "YOLO26n"
        ratio_str = "Baseline (0%)" if row['prune_ratio'] == '0%' else f"Pruned {row['prune_ratio']}"
        md_lines.append(
            f"| {model_name} | {ratio_str} | {row['rec_yawn_mean']:.1f}% | {row['rec_hom_mean']:.1f}% | {row['rec_drink_mean']:.1f}% | {row['rec_phone_mean']:.1f}% |"
        )

    summary_md_path = RESULTS_DIR / 'pruned_benchmark_summary.md'
    with open(summary_md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines) + '\n')

    print(f"\nSaved structured JSON results: {RESULTS_DIR / 'pruned_benchmark_results.json'}")
    print(f"Saved Markdown summary table:  {summary_md_path}")
    print("\nBenchmark completed successfully!")


if __name__ == '__main__':
    run_benchmark()
