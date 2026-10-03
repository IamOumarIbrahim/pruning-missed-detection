import json
import numpy as np
import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CLASS_NAMES = ['yawning', 'hand_over_mouth', 'drinking', 'phone_use']
DF_GT = pd.read_parquet(REPO_ROOT / 'results' / 'phase0' / 'ground_truth.parquet')

def eval_recalls_at_tau(df_dets, df_gt_subset, tau):
    gt_counts = [len(df_gt_subset[df_gt_subset['class_id'] == c]) for c in range(4)]
    recs = []
    for c in range(4):
        if gt_counts[c] == 0:
            recs.append(1.0)
            continue
        tp = len(df_dets[(df_dets['class_id'] == c) & (df_dets['matched'] == True) & (df_dets['conf'] >= tau)])
        recs.append(tp / gt_counts[c])
    return recs

def find_tau_star(df_dets, df_gt_subset, r_floor):
    taus = np.linspace(0.01, 0.90, 900)
    for tau in reversed(taus):
        recs = eval_recalls_at_tau(df_dets, df_gt_subset, tau)
        if min(recs) >= r_floor:
            return float(tau), True
    return None, False

def run_r9_analysis():
    print("=" * 80)
    print("MANDATE R9: CHECKPOINT SENSITIVITY (BEST.PT VS LAST.PT)")
    print("=" * 80)

    gt_val = DF_GT[DF_GT['split'] == 'val']
    gt_test = DF_GT[DF_GT['split'] == 'test']

    models = ['yolo11n', 'yolo26n']
    seeds = [0, 1, 2]

    records = []

    for m in models:
        for s in seeds:
            # Files for best
            f_best_v = REPO_ROOT / 'results' / 'phase0' / 'cache' / f"{m}_baseline_s{s}_best_val.parquet"
            f_best_t = REPO_ROOT / 'results' / 'phase0' / 'cache' / f"{m}_baseline_s{s}_best_test.parquet"
            
            # Files for last
            f_last_v = REPO_ROOT / 'results' / 'phase0' / 'cache' / f"{m}_baseline_s{s}_last_val.parquet"
            f_last_t = REPO_ROOT / 'results' / 'phase0' / 'cache' / f"{m}_baseline_s{s}_last_test.parquet"

            if not (f_best_v.exists() and f_best_t.exists() and f_last_v.exists() and f_last_t.exists()):
                print(f"Skipping {m} s{s} (not all best/last cached yet)")
                continue

            df_bv = pd.read_parquet(f_best_v)
            df_bt = pd.read_parquet(f_best_t)
            df_lv = pd.read_parquet(f_last_v)
            df_lt = pd.read_parquet(f_last_t)

            # Evaluate best
            recs_25_bv = eval_recalls_at_tau(df_bv, gt_val, 0.25)
            r_base_b = min(recs_25_bv)
            r_floor_b = r_base_b - 0.05
            tau_star_b, feas_b = find_tau_star(df_bv, gt_val, r_floor_b)
            if feas_b and tau_star_b is not None:
                recs_test_b = eval_recalls_at_tau(df_bt, gt_test, tau_star_b)
                min_test_b = min(recs_test_b)
                margin_b = min_test_b - r_floor_b
            else:
                min_test_b, margin_b = np.nan, np.nan

            # Evaluate last
            recs_25_lv = eval_recalls_at_tau(df_lv, gt_val, 0.25)
            r_base_l = min(recs_25_lv)
            r_floor_l = r_base_l - 0.05
            tau_star_l, feas_l = find_tau_star(df_lv, gt_val, r_floor_l)
            if feas_l and tau_star_l is not None:
                recs_test_l = eval_recalls_at_tau(df_lt, gt_test, tau_star_l)
                min_test_l = min(recs_test_l)
                margin_l = min_test_l - r_floor_l
            else:
                min_test_l, margin_l = np.nan, np.nan

            records.append({
                'model': m, 'seed': s,
                'best_R_base': r_base_b, 'best_R_floor': r_floor_b, 'best_tau_star': tau_star_b,
                'best_test_min_recall': min_test_b, 'best_margin_pp': margin_b * 100,
                'last_R_base': r_base_l, 'last_R_floor': r_floor_l, 'last_tau_star': tau_star_l,
                'last_test_min_recall': min_test_l, 'last_margin_pp': margin_l * 100,
                'diff_tau_star (last - best)': tau_star_l - tau_star_b if (tau_star_l and tau_star_b) else np.nan,
                'diff_margin_pp (last - best)': (margin_l - margin_b) * 100 if (margin_l is not None and margin_b is not None) else np.nan
            })

    if not records:
        print("No completed best/last pairs ready yet.")
        return

    df_sens = pd.DataFrame(records)
    out_sens_csv = REPO_ROOT / 'results' / 'phase0' / 'r9_checkpoint_sensitivity_all.csv'
    df_sens.to_csv(out_sens_csv, index=False)
    print(f"Sensitivity comparison saved to: {out_sens_csv}")
    print(df_sens.to_string())

if __name__ == '__main__':
    run_r9_analysis()
