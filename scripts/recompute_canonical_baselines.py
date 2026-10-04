"""Recompute Canonical Baseline Metrics.

Strict canonical metric:
- Exact step-function recall from cached detections: TP(conf >= tau) / GT_count
- tau* defined explicitly as: largest tau among unique validation detection confidences
  such that min_c Recall_c^val(tau) >= R_floor (where R_floor = min_c Recall_c^val(0.25) - 0.05).
- Zero interpolation, zero r_curve, zero res.box.r.
"""

import numpy as np
import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DF_GT = pd.read_parquet(REPO_ROOT / 'results' / 'phase0' / 'ground_truth.parquet')

GT_VAL = DF_GT[DF_GT['split'] == 'val']
GT_TEST = DF_GT[DF_GT['split'] == 'test']

VAL_COUNTS = [len(GT_VAL[GT_VAL['class_id'] == c]) for c in range(4)]
TEST_COUNTS = [len(GT_TEST[GT_TEST['class_id'] == c]) for c in range(4)]
CLASS_NAMES = ['yawning', 'hand_over_mouth', 'drinking', 'phone_use']

def get_class_recalls(df_dets, tau, gt_counts):
    recs = []
    tps = []
    for c in range(4):
        tp = len(df_dets[(df_dets['class_id'] == c) & (df_dets['matched'] == True) & (df_dets['conf'] >= tau)])
        recs.append(tp / gt_counts[c] if gt_counts[c] > 0 else 1.0)
        tps.append(tp)
    return recs, tps

def compute_baseline_canonical(model, seed, ckpt_type):
    f_val = REPO_ROOT / 'results' / 'phase0' / 'cache' / f"{model}_baseline_s{seed}_{ckpt_type}_val.parquet"
    f_test = REPO_ROOT / 'results' / 'phase0' / 'cache' / f"{model}_baseline_s{seed}_{ckpt_type}_test.parquet"

    if not (f_val.exists() and f_test.exists()):
        return None

    df_v = pd.read_parquet(f_val)
    df_t = pd.read_parquet(f_test)

    # 1. R_base at tau=0.25 on val
    v_recs_25, v_tps_25 = get_class_recalls(df_v, 0.25, VAL_COUNTS)
    min_v_rec_25 = min(v_recs_25)
    worst_v_cls_25 = CLASS_NAMES[np.argmin(v_recs_25)]
    r_floor = min_v_rec_25 - 0.05

    # 2. Candidate thresholds: unique detection confidences on validation
    unique_confs = np.sort(df_v['conf'].unique())[::-1]  # descending

    tau_star = None
    feasible = False
    val_worst_rec_star = None
    val_worst_cls_star = None

    for tau in unique_confs:
        recs, tps = get_class_recalls(df_v, tau, VAL_COUNTS)
        min_r = min(recs)
        if min_r >= r_floor:
            tau_star = float(tau)
            feasible = True
            val_worst_rec_star = min_r
            val_worst_cls_star = CLASS_NAMES[np.argmin(recs)]
            break

    # 3. Test evaluation at tau_star
    if feasible and tau_star is not None:
        t_recs_star, t_tps_star = get_class_recalls(df_t, tau_star, TEST_COUNTS)
        test_worst_rec_star = min(t_recs_star)
        test_worst_cls_star = CLASS_NAMES[np.argmin(t_recs_star)]
        margin_pp = (test_worst_rec_star - r_floor) * 100.0
        passed = bool(margin_pp >= 0.0)
        t_worst_tp = t_tps_star[np.argmin(t_recs_star)]
        t_worst_gt = TEST_COUNTS[np.argmin(t_recs_star)]
    else:
        test_worst_rec_star = np.nan
        test_worst_cls_star = 'N/A'
        margin_pp = np.nan
        passed = False
        t_worst_tp, t_worst_gt = 0, 0

    # 4. Test evaluation at canonical tau=0.25
    t_recs_25, t_tps_25 = get_class_recalls(df_t, 0.25, TEST_COUNTS)
    test_worst_rec_25 = min(t_recs_25)
    test_worst_cls_25 = CLASS_NAMES[np.argmin(t_recs_25)]
    margin_25_pp = (test_worst_rec_25 - r_floor) * 100.0
    passed_25 = bool(margin_25_pp >= 0.0)
    t_25_tp = t_tps_25[np.argmin(t_recs_25)]
    t_25_gt = TEST_COUNTS[np.argmin(t_recs_25)]

    return {
        'model': model,
        'seed': seed,
        'ckpt': ckpt_type,
        'val_R_base': min_v_rec_25,
        'val_binding_cls_25': worst_v_cls_25,
        'val_R_floor': r_floor,
        'tau_star': tau_star,
        'val_worst_rec_star': val_worst_rec_star,
        'val_worst_cls_star': val_worst_cls_star,
        'test_worst_rec_star': test_worst_rec_star,
        'test_worst_cls_star': test_worst_cls_star,
        'test_worst_tp_gt_star': f"{t_worst_tp}/{t_worst_gt}",
        'margin_pp_star': margin_pp,
        'passed_star': passed,
        'test_worst_rec_25': test_worst_rec_25,
        'test_worst_cls_25': test_worst_cls_25,
        'test_worst_tp_gt_25': f"{t_25_tp}/{t_25_gt}",
        'margin_pp_25': margin_25_pp,
        'passed_25': passed_25
    }

def main():
    checkpoints = [
        # YOLO11n seeds 0-4 (best and last)
        ('yolo11n', 0, 'best'), ('yolo11n', 0, 'last'),
        ('yolo11n', 1, 'best'), ('yolo11n', 1, 'last'),
        ('yolo11n', 2, 'best'), ('yolo11n', 2, 'last'),
        ('yolo11n', 3, 'best'), ('yolo11n', 3, 'last'),
        ('yolo11n', 4, 'best'), ('yolo11n', 4, 'last'),
        # YOLO26n seeds 0-2 (and seed 3) (best and last)
        ('yolo26n', 0, 'best'), ('yolo26n', 0, 'last'),
        ('yolo26n', 1, 'best'), ('yolo26n', 1, 'last'),
        ('yolo26n', 2, 'best'), ('yolo26n', 2, 'last'),
        ('yolo26n', 3, 'best'), ('yolo26n', 3, 'last'),
    ]

    records = []
    for m, s, c in checkpoints:
        res = compute_baseline_canonical(m, s, c)
        if res is not None:
            records.append(res)

    df = pd.DataFrame(records)
    out_csv = REPO_ROOT / 'results' / 'phase0' / 'r1_canonical_baselines_recomputed.csv'
    df.to_csv(out_csv, index=False)
    print(f"Saved canonical baseline recomputation to: {out_csv}")
    print("\n" + "="*120)
    print("CANONICAL BASELINE RECOMPUTATION TABLE")
    print("="*120)
    cols_display = [
        'model', 'seed', 'ckpt', 'val_R_base', 'val_R_floor', 'tau_star',
        'val_worst_rec_star', 'test_worst_rec_star', 'test_worst_cls_star',
        'test_worst_tp_gt_star', 'margin_pp_star', 'passed_star',
        'test_worst_rec_25', 'test_worst_tp_gt_25', 'margin_pp_25', 'passed_25'
    ]
    print(df[cols_display].to_string(index=False))

if __name__ == '__main__':
    main()
