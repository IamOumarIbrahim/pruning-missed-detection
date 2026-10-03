import json
import numpy as np
import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / 'results' / 'phase0' / 'cache'
CLASS_NAMES = ['yawning', 'hand_over_mouth', 'drinking', 'phone_use']

DF_GT = pd.read_parquet(REPO_ROOT / 'results' / 'phase0' / 'ground_truth.parquet')

def compute_tau_80(df_matched_tps, n_gt):
    if n_gt == 0:
        return np.nan
    confs = np.sort(df_matched_tps['conf'].values)[::-1]
    # If even all TPs don't reach 80% recall:
    if len(confs) / n_gt < 0.80:
        return np.nan
    # The (int(ceil(0.80 * n_gt)) - 1)-th confidence is the largest tau where recall >= 0.80
    idx = int(np.ceil(0.80 * n_gt)) - 1
    return float(confs[idx])

def run_r6_analysis():
    print("=" * 80)
    print("MANDATE R6: CALIBRATION DRIFT (MEDIAN, 10TH PERCENTILE, AND TAU_80)")
    print("=" * 80)

    manifest_path = REPO_ROOT / 'results' / 'phase0' / 'cache_manifest.csv'
    if not manifest_path.exists():
        print("Manifest does not exist yet. Cache still building...")
        return

    df_manifest = pd.read_csv(manifest_path)
    print(f"Manifest currently has {len(df_manifest)} entries.")

    models = ['yolo11n', 'yolo26n']
    ratios = ['0%', '10%', '20%', '30%', '40%', '50%']

    results = []

    # Map GT counts by split and pooled
    gt_counts = {
        'val': [len(DF_GT[(DF_GT['split'] == 'val') & (DF_GT['class_id'] == c)]) for c in range(4)],
        'test': [len(DF_GT[(DF_GT['split'] == 'test') & (DF_GT['class_id'] == c)]) for c in range(4)],
        'pooled': [len(DF_GT[DF_GT['split'].isin(['val', 'test']) & (DF_GT['class_id'] == c)]) for c in range(4)]
    }

    # Group available checkpoints
    ckpts_to_process = df_manifest[df_manifest['ckpt_type'] == 'best'].copy()

    for (m, r_str, s), grp in ckpts_to_process.groupby(['model', 'prune_ratio', 'seed']):
        # Check if both val and test exist
        splits_present = grp['split'].tolist()
        if 'val' not in splits_present or 'test' not in splits_present:
            continue

        p_val_path = REPO_ROOT / grp[grp['split'] == 'val']['parquet_path'].iloc[0]
        p_test_path = REPO_ROOT / grp[grp['split'] == 'test']['parquet_path'].iloc[0]

        if not p_val_path.exists() or not p_test_path.exists():
            continue

        df_val = pd.read_parquet(p_val_path)
        df_test = pd.read_parquet(p_test_path)
        df_pooled = pd.concat([df_val, df_test], ignore_index=True)

        split_dfs = {
            'val': df_val,
            'test': df_test,
            'pooled': df_pooled
        }

        for split_name, df_s in split_dfs.items():
            for c_id, c_name in enumerate(CLASS_NAMES):
                n_gt = gt_counts[split_name][c_id]
                tps = df_s[(df_s['class_id'] == c_id) & (df_s['matched'] == True)]
                tp_confs = tps['conf'].values

                if len(tp_confs) > 0:
                    med_conf = float(np.median(tp_confs))
                    p10_conf = float(np.percentile(tp_confs, 10))
                    mean_conf = float(np.mean(tp_confs))
                else:
                    med_conf = np.nan
                    p10_conf = np.nan
                    mean_conf = np.nan

                tau_80 = compute_tau_80(tps, n_gt)
                raw_recall = len(tps) / n_gt if n_gt > 0 else 0.0

                results.append({
                    'model': m,
                    'prune_ratio': r_str,
                    'seed': s,
                    'split': split_name,
                    'class_name': c_name,
                    'class_id': c_id,
                    'n_gt': n_gt,
                    'n_tp_raw': len(tp_confs),
                    'raw_recall_001': raw_recall,
                    'median_tp_conf': med_conf,
                    'p10_tp_conf': p10_conf,
                    'mean_tp_conf': mean_conf,
                    'tau_80': tau_80
                })

    if not results:
        print("No completed val+test pairs ready yet.")
        return

    df_out = pd.DataFrame(results)
    out_csv = REPO_ROOT / 'results' / 'phase0' / 'r6_calibration_drift.csv'
    df_out.to_csv(out_csv, index=False)
    print(f"Calibration drift metrics saved to: {out_csv}")

    # Summary analysis: Baseline seed spread vs Pruning effect
    print("\n--- BASELINE SEED SPREAD ANALYSIS (0% Pruning) ---")
    df_base = df_out[(df_out['prune_ratio'] == '0%') & (df_out['split'] == 'pooled')]
    for m in models:
        m_base = df_base[df_base['model'] == m]
        n_seeds = m_base['seed'].nunique()
        print(f"\nModel: {m} (K={n_seeds} baseline seeds, pooled val+test):")
        for cname in CLASS_NAMES:
            c_base = m_base[m_base['class_name'] == cname]
            med_mean = c_base['median_tp_conf'].mean()
            med_sd = c_base['median_tp_conf'].std()
            p10_mean = c_base['p10_tp_conf'].mean()
            p10_sd = c_base['p10_tp_conf'].std()
            t80_mean = c_base['tau_80'].mean()
            t80_sd = c_base['tau_80'].std()
            print(f"  {cname:15s}: Median={med_mean:.4f}±{med_sd:.4f}, P10={p10_mean:.4f}±{p10_sd:.4f}, tau_80={t80_mean:.4f}±{t80_sd:.4f}")

if __name__ == '__main__':
    run_r6_analysis()
