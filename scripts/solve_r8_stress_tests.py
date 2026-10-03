import json
import numpy as np
import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CLASS_NAMES = ['yawning', 'hand_over_mouth', 'drinking', 'phone_use']
DF_GT = pd.read_parquet(REPO_ROOT / 'results' / 'phase0' / 'ground_truth.parquet')
DF_EVENTS = pd.read_parquet(REPO_ROOT / 'results' / 'phase0' / 'events.parquet')

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

def eval_fp_and_prec(df_dets, tau, n_images):
    sub = df_dets[df_dets['conf'] >= tau]
    tp = len(sub[sub['matched'] == True])
    fp = len(sub[sub['matched'] == False])
    prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    fp_per_img = fp / n_images if n_images > 0 else 0.0
    return prec, fp_per_img

def find_tau_star(df_dets, df_gt_subset, r_floor):
    taus = np.linspace(0.01, 0.90, 900)
    for tau in reversed(taus):
        recs = eval_recalls_at_tau(df_dets, df_gt_subset, tau)
        if min(recs) >= r_floor:
            return float(tau), True
    return None, False

def run_r8_analysis():
    print("=" * 80)
    print("MANDATE R8: COMPREHENSIVE STRESS-TESTS OF THE TAU* CALIBRATION RULE")
    print("=" * 80)

    manifest_path = REPO_ROOT / 'results' / 'phase0' / 'cache_manifest.csv'
    if not manifest_path.exists():
        print("Manifest does not exist yet.")
        return

    df_manifest = pd.read_csv(manifest_path)
    baselines = [
        ('yolo11n', 0), ('yolo11n', 1), ('yolo11n', 2),
        ('yolo26n', 0), ('yolo26n', 1), ('yolo26n', 2)
    ]

    r8_records = []

    # Check which baseline caches are ready
    ready_baselines = []
    for m, s in baselines:
        v_f = REPO_ROOT / 'results' / 'phase0' / 'cache' / f"{m}_baseline_s{s}_best_val.parquet"
        t_f = REPO_ROOT / 'results' / 'phase0' / 'cache' / f"{m}_baseline_s{s}_best_test.parquet"
        if v_f.exists() and t_f.exists():
            ready_baselines.append((m, s, pd.read_parquet(v_f), pd.read_parquet(t_f)))

    print(f"Baselines ready for R8: {len(ready_baselines)} / 6")
    if not ready_baselines:
        return

    gt_val = DF_GT[DF_GT['split'] == 'val']
    gt_test = DF_GT[DF_GT['split'] == 'test']
    n_img_val = gt_val['image_id'].nunique()
    n_img_test = gt_test['image_id'].nunique()

    # --- 1. Sub-task (b): Cross-over Calibration (Calibrate Test -> Evaluate Val) ---
    print("\n--- Sub-task (b): CROSS-OVER (Calibrate on Test -> Evaluate on Val) ---")
    crossover_rows = []
    for m, s, df_v, df_t in ready_baselines:
        # Calibrate on test
        rec_25_test = eval_recalls_at_tau(df_t, gt_test, 0.25)
        r_base_t = min(rec_25_test)
        r_floor_t = r_base_t - 0.05
        tau_star_t, feasible_t = find_tau_star(df_t, gt_test, r_floor_t)

        # Evaluate on val
        if feasible_t and tau_star_t is not None:
            rec_val_star = eval_recalls_at_tau(df_v, gt_val, tau_star_t)
            min_rec_val = min(rec_val_star)
            margin_val = min_rec_val - r_floor_t
            prec_val, fp_val = eval_fp_and_prec(df_v, tau_star_t, n_img_val)
            passed = margin_val >= 0.0
        else:
            min_rec_val, margin_val, prec_val, fp_val, passed = np.nan, np.nan, np.nan, np.nan, False

        crossover_rows.append({
            'model': m, 'seed': s,
            'test_R_base': r_base_t, 'test_R_floor': r_floor_t,
            'calib_tau_star_from_test': tau_star_t,
            'val_min_recall': min_rec_val, 'val_margin_pp': margin_val * 100,
            'val_precision': prec_val, 'val_fp_per_img': fp_val,
            'crossover_pass': passed
        })
    df_crossover = pd.DataFrame(crossover_rows)
    print(df_crossover.to_string(index=False))
    df_crossover.to_csv(REPO_ROOT / 'results' / 'phase0' / 'r8_crossover_test_to_val.csv', index=False)

    # --- 2. Sub-task (c): Leave-One-Subject-Out (LOSO) across 6 subjects ---
    print("\n--- Sub-task (c): LEAVE-ONE-SUBJECT-OUT (LOSO) CV ---")
    all_6_subs = ['subject_02', 'subject_03', 'subject_11', 'subject_05', 'subject_10', 'subject_12']
    loso_rows = []

    for m, s, df_v, df_t in ready_baselines:
        df_6 = pd.concat([df_v, df_t], ignore_index=True)
        gt_6 = pd.concat([gt_val, gt_test], ignore_index=True)

        for held_out_sub in all_6_subs:
            # Training fold: other 5 subjects
            train_subs = [x for x in all_6_subs if x != held_out_sub]
            gt_train = gt_6[gt_6['subject'].isin(train_subs)]
            df_train = df_6[df_6['subject'].isin(train_subs)]

            # Test fold: held out subject
            gt_test_sub = gt_6[gt_6['subject'] == held_out_sub]
            df_test_sub = df_6[df_6['subject'] == held_out_sub]
            n_img_sub = gt_test_sub['image_id'].nunique()

            # Calibrate on 5 subjects
            recs_25 = eval_recalls_at_tau(df_train, gt_train, 0.25)
            r_base_loso = min(recs_25)
            r_floor_loso = r_base_loso - 0.05
            tau_star_loso, feasible = find_tau_star(df_train, gt_train, r_floor_loso)

            # Evaluate on held-out subject
            if feasible and tau_star_loso is not None:
                recs_heldout = eval_recalls_at_tau(df_test_sub, gt_test_sub, tau_star_loso)
                min_rec_heldout = min(recs_heldout)
                margin_heldout = min_rec_heldout - r_floor_loso
                prec_h, fp_h = eval_fp_and_prec(df_test_sub, tau_star_loso, n_img_sub)
                passed_h = margin_heldout >= 0.0
            else:
                min_rec_heldout, margin_heldout, prec_h, fp_h, passed_h = np.nan, np.nan, np.nan, np.nan, False

            loso_rows.append({
                'model': m, 'seed': s, 'held_out_subject': held_out_sub,
                'train_R_base': r_base_loso, 'train_R_floor': r_floor_loso,
                'calib_tau_star': tau_star_loso,
                'heldout_min_recall': min_rec_heldout,
                'heldout_margin_pp': margin_heldout * 100,
                'heldout_precision': prec_h, 'heldout_fp_per_img': fp_h,
                'passed': passed_h
            })

    df_loso = pd.DataFrame(loso_rows)
    out_loso_csv = REPO_ROOT / 'results' / 'phase0' / 'r8_loso_cross_validation.csv'
    df_loso.to_csv(out_loso_csv, index=False)
    print(f"LOSO results saved to: {out_loso_csv}")
    print(f"LOSO compliance pass rate: {df_loso['passed'].mean()*100:.1f}% ({df_loso['passed'].sum()}/{len(df_loso)})")

    # --- 3. Sub-task (d): Margin parameter variation delta in {5, 10, 15, 20} pp ---
    print("\n--- Sub-task (d): SAFETY MARGIN VARIATION (delta in {5, 10, 15, 20} pp) ---")
    delta_rows = []
    deltas = [0.05, 0.10, 0.15, 0.20]

    for m, s, df_v, df_t in ready_baselines:
        recs_25_val = eval_recalls_at_tau(df_v, gt_val, 0.25)
        r_base = min(recs_25_val)

        for d in deltas:
            r_floor = r_base - d
            tau_star, feasible = find_tau_star(df_v, gt_val, r_floor)

            if feasible and tau_star is not None:
                recs_test = eval_recalls_at_tau(df_t, gt_test, tau_star)
                min_rec_test = min(recs_test)
                margin_test = min_rec_test - r_floor
                prec_t, fp_t = eval_fp_and_prec(df_t, tau_star, n_img_test)
                passed = margin_test >= 0.0
            else:
                min_rec_test, margin_test, prec_t, fp_t, passed = np.nan, np.nan, np.nan, np.nan, False

            delta_rows.append({
                'model': m, 'seed': s, 'delta_pp': int(d*100),
                'R_base': r_base, 'R_floor': r_floor, 'tau_star': tau_star,
                'test_min_recall': min_rec_test, 'test_margin_pp': margin_test * 100,
                'test_precision': prec_t, 'test_fp_per_img': fp_t,
                'passed': passed
            })

    df_delta = pd.DataFrame(delta_rows)
    out_delta_csv = REPO_ROOT / 'results' / 'phase0' / 'r8_delta_margin_tradeoffs.csv'
    df_delta.to_csv(out_delta_csv, index=False)
    print(f"Margin delta tradeoffs saved to: {out_delta_csv}")
    print(df_delta.groupby('delta_pp')[['test_margin_pp', 'test_precision', 'test_fp_per_img', 'passed']].mean().to_string())

if __name__ == '__main__':
    run_r8_analysis()
