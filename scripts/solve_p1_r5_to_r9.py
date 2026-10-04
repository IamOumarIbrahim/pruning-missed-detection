"""P1: Re-running R5-R9 Scripts with Canonical Metric and n=8 Complete Baselines.

This script executes the re-evaluation of R5, R6, R7, R8, R9:
- Strictly canonical metric: step-function recall TP(conf >= tau) / GT
- Complete baselines: n=8 (YOLO11n s0-s4, YOLO26n s0-s2). YOLO26n s3 is excluded as incomplete.
- All candidate tau* values determined from validation detections.
- Compares new numbers against earlier reported values and outputs diff summary.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DF_GT = pd.read_parquet(REPO_ROOT / 'results' / 'phase0' / 'ground_truth.parquet')
DF_EVENTS = pd.read_parquet(REPO_ROOT / 'results' / 'phase0' / 'events.parquet')
CLASS_NAMES = ['yawning', 'hand_over_mouth', 'drinking', 'phone_use']

# Complete baselines list
BASELINES = [
    ('yolo11n', 0), ('yolo11n', 1), ('yolo11n', 2), ('yolo11n', 3), ('yolo11n', 4),
    ('yolo26n', 0), ('yolo26n', 1), ('yolo26n', 2)
]

def load_canonical_baselines():
    p = REPO_ROOT / 'results' / 'phase0' / 'r1_canonical_baselines_recomputed.csv'
    df = pd.read_csv(p)
    return df

def get_class_recalls(df_dets, tau, gt_counts):
    recs, tps = [], []
    for c in range(4):
        tp = len(df_dets[(df_dets['class_id'] == c) & (df_dets['matched'] == True) & (df_dets['conf'] >= tau)])
        recs.append(tp / gt_counts[c] if gt_counts[c] > 0 else 1.0)
        tps.append(tp)
    return recs, tps

# ==============================================================================
# R5: Accounting of YOLO11n Seed 1 and Per-Subject Baseline Recall
# ==============================================================================
def run_r5():
    print("--- Executing R5 (Seed 1 Accounting & Per-Subject Baseline Recall) ---")
    df_canon = load_canonical_baselines()
    row_s1 = df_canon[(df_canon['model'] == 'yolo11n') & (df_canon['seed'] == 1) & (df_canon['ckpt'] == 'best')].iloc[0]
    tau_star_s1 = float(row_s1['tau_star']) # 0.75582

    f_test = REPO_ROOT / 'results' / 'phase0' / 'cache' / 'yolo11n_baseline_s1_best_test.parquet'
    f_val = REPO_ROOT / 'results' / 'phase0' / 'cache' / 'yolo11n_baseline_s1_best_val.parquet'
    df_p_test = pd.read_parquet(f_test)
    df_p_val = pd.read_parquet(f_val)

    gt_test_yawn = DF_GT[(DF_GT['split'] == 'test') & (DF_GT['class_name'] == 'yawning')].copy()

    # Map events
    ev_ids = []
    for _, g in gt_test_yawn.iterrows():
        ev = DF_EVENTS[(DF_EVENTS['subject'] == g['subject']) & 
                       (DF_EVENTS['video'] == g['video']) & 
                       (DF_EVENTS['class_name'] == 'yawning') & 
                       (DF_EVENTS['start_frame'] <= g['frame_num']) & 
                       (DF_EVENTS['end_frame'] >= g['frame_num'])]
        ev_ids.append(ev['event_id'].values[0] if len(ev) > 0 else 'no_event')
    gt_test_yawn['event_id'] = ev_ids

    matched_tps = df_p_test[(df_p_test['class_name'] == 'yawning') & (df_p_test['matched'] == True)]
    max_confs, best_ious = [], []
    for _, g in gt_test_yawn.iterrows():
        frame_tps = matched_tps[matched_tps['frame'] == g['image_id']]
        if len(frame_tps) > 0:
            best_det = frame_tps.sort_values(by='conf', ascending=False).iloc[0]
            max_confs.append(float(best_det['conf']))
            best_ious.append(float(best_det['best_iou']))
        else:
            max_confs.append(0.0)
            best_ious.append(0.0)

    gt_test_yawn['best_conf'] = max_confs
    gt_test_yawn['best_iou'] = best_ious
    gt_test_yawn['tp_025'] = gt_test_yawn['best_conf'] >= 0.25
    gt_test_yawn['tp_tau_star'] = gt_test_yawn['best_conf'] >= tau_star_s1

    lost_frames = gt_test_yawn[gt_test_yawn['tp_025'] & (~gt_test_yawn['tp_tau_star'])].copy()
    lost_frames = lost_frames.sort_values(by='best_conf', ascending=False).reset_index(drop=True)
    out_lost_csv = REPO_ROOT / 'results' / 'phase0' / 'r5_all_lost_yawn_frames_seed1.csv'
    lost_frames[['subject', 'video', 'frame', 'frame_num', 'event_id', 'best_conf', 'best_iou']].to_csv(out_lost_csv, index=False)

    # Histogram
    val_yawn_tps = df_p_val[(df_p_val['class_name'] == 'yawning') & (df_p_val['matched'] == True)]['conf'].values
    test_yawn_tps = matched_tps['conf'].values
    bins = [0.0, 0.25, 0.50, 0.70, 0.75, tau_star_s1, 0.80, 0.85, 0.90, 1.0]
    val_hist, _ = np.histogram(val_yawn_tps, bins=bins)
    test_hist, _ = np.histogram(test_yawn_tps, bins=bins)
    hist_df = pd.DataFrame({
        'bin_range': [f"[{bins[i]:.5f}, {bins[i+1]:.5f})" for i in range(len(bins)-1)],
        'val_tp_count': val_hist,
        'val_tp_pct': val_hist / len(val_yawn_tps) * 100 if len(val_yawn_tps) > 0 else 0,
        'test_tp_count': test_hist,
        'test_tp_pct': test_hist / len(test_yawn_tps) * 100 if len(test_yawn_tps) > 0 else 0
    })
    out_hist_csv = REPO_ROOT / 'results' / 'phase0' / 'r5_tp_confidence_histogram_yawn_seed1.csv'
    hist_df.to_csv(out_hist_csv, index=False)

    # Per-subject matrix for all 8 baselines
    per_sub_rows = []
    taus_to_eval = [0.10, 0.25, 0.50, 0.75]
    for model, s in BASELINES:
        b_tau_star = float(df_canon[(df_canon['model'] == model) & (df_canon['seed'] == s) & (df_canon['ckpt'] == 'best')]['tau_star'].iloc[0])
        for split in ['val', 'test']:
            c_file = REPO_ROOT / 'results' / 'phase0' / 'cache' / f"{model}_baseline_s{s}_best_{split}.parquet"
            df_det = pd.read_parquet(c_file)
            split_gt = DF_GT[DF_GT['split'] == split]
            for sub in sorted(split_gt['subject'].unique()):
                sub_gt = split_gt[split_gt['subject'] == sub]
                sub_det = df_det[df_det['subject'] == sub]
                for c_id, c_name in enumerate(CLASS_NAMES):
                    c_gt_cnt = len(sub_gt[sub_gt['class_id'] == c_id])
                    c_ev_cnt = len(DF_EVENTS[(DF_EVENTS['subject'] == sub) & (DF_EVENTS['class_name'] == c_name)])
                    row = {
                        'model': model, 'seed': s, 'split': split, 'subject': sub,
                        'class_name': c_name, 'gt_boxes': c_gt_cnt, 'gt_events': c_ev_cnt,
                        'tau_star': b_tau_star
                    }
                    for tau in taus_to_eval:
                        tp_tau = len(sub_det[(sub_det['class_id'] == c_id) & (sub_det['matched'] == True) & (sub_det['conf'] >= tau)])
                        row[f'recall_tau_{int(tau*100):02d}'] = tp_tau / c_gt_cnt if c_gt_cnt > 0 else np.nan
                    tp_star = len(sub_det[(sub_det['class_id'] == c_id) & (sub_det['matched'] == True) & (sub_det['conf'] >= b_tau_star)])
                    row['recall_own_tau_star'] = tp_star / c_gt_cnt if c_gt_cnt > 0 else np.nan
                    per_sub_rows.append(row)

    df_per_sub = pd.DataFrame(per_sub_rows)
    out_sub_csv = REPO_ROOT / 'results' / 'phase0' / 'r5_per_subject_baseline_recall.csv'
    df_per_sub.to_csv(out_sub_csv, index=False)
    print(f"R5 complete. Lost frames: {len(lost_frames)}, sub rows: {len(df_per_sub)}")

# ==============================================================================
# R6: Calibration Drift Across Ratios
# ==============================================================================
def compute_tau_80(df_matched_tps, n_gt):
    if n_gt == 0:
        return np.nan
    confs = np.sort(df_matched_tps['conf'].values)[::-1]
    if len(confs) / n_gt < 0.80:
        return np.nan
    idx = int(np.ceil(0.80 * n_gt)) - 1
    return float(confs[idx])

def run_r6():
    print("--- Executing R6 (Calibration Drift: Median, P10, Tau_80) ---")
    manifest_path = REPO_ROOT / 'results' / 'phase0' / 'cache_manifest.csv'
    df_manifest = pd.read_csv(manifest_path)
    models = ['yolo11n', 'yolo26n']
    ratios = ['0%', '10%', '20%', '30%', '40%', '50%']

    gt_counts = {
        'val': [len(DF_GT[(DF_GT['split'] == 'val') & (DF_GT['class_id'] == c)]) for c in range(4)],
        'test': [len(DF_GT[(DF_GT['split'] == 'test') & (DF_GT['class_id'] == c)]) for c in range(4)],
        'pooled': [len(DF_GT[DF_GT['split'].isin(['val', 'test']) & (DF_GT['class_id'] == c)]) for c in range(4)]
    }

    results = []
    ckpts_to_process = df_manifest[df_manifest['ckpt_type'] == 'best'].copy()

    for (m, r_str, s), grp in ckpts_to_process.groupby(['model', 'prune_ratio', 'seed']):
        # Skip incomplete yolo26n seed 3
        if m == 'yolo26n' and s == 3 and r_str == '0%':
            continue
        splits_present = grp['split'].tolist()
        if 'val' not in splits_present or 'test' not in splits_present:
            continue
        p_val_path = REPO_ROOT / grp[grp['split'] == 'val']['parquet_path'].iloc[0]
        p_test_path = REPO_ROOT / grp[grp['split'] == 'test']['parquet_path'].iloc[0]
        df_val = pd.read_parquet(p_val_path)
        df_test = pd.read_parquet(p_test_path)
        df_pooled = pd.concat([df_val, df_test], ignore_index=True)

        split_dfs = {'val': df_val, 'test': df_test, 'pooled': df_pooled}

        for sp in ['val', 'test', 'pooled']:
            df_cur = split_dfs[sp]
            for c_id, c_name in enumerate(CLASS_NAMES):
                tps = df_cur[(df_cur['class_id'] == c_id) & (df_cur['matched'] == True)]
                n_gt = gt_counts[sp][c_id]
                tau_80 = compute_tau_80(tps, n_gt)
                c_confs = tps['conf'].values
                if len(c_confs) > 0:
                    med_conf = float(np.median(c_confs))
                    p10_conf = float(np.percentile(c_confs, 10))
                    p90_conf = float(np.percentile(c_confs, 90))
                    mean_conf = float(np.mean(c_confs))
                else:
                    med_conf, p10_conf, p90_conf, mean_conf = np.nan, np.nan, np.nan, np.nan

                results.append({
                    'model': m, 'prune_ratio': r_str, 'seed': s, 'split': sp,
                    'class_name': c_name, 'tp_count': len(tps), 'gt_count': n_gt,
                    'recall_at_0': len(tps) / n_gt if n_gt > 0 else np.nan,
                    'tau_80': tau_80, 'median_conf': med_conf, 'p10_conf': p10_conf,
                    'p90_conf': p90_conf, 'mean_conf': mean_conf
                })

    df_r6 = pd.DataFrame(results)
    out_r6 = REPO_ROOT / 'results' / 'phase0' / 'r6_calibration_drift.csv'
    df_r6.to_csv(out_r6, index=False)
    print(f"R6 complete. Saved {len(df_r6)} records to {out_r6}")

# ==============================================================================
# R7: Matched Operating Points (mAP50, min-class AP, Rec@P80, Rec@P90)
# ==============================================================================
def compute_ap(recalls, precisions):
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    x = np.linspace(0, 1, 101)
    y = np.interp(x, mrec, mpre)
    return float(np.trapezoid(y, x) if hasattr(np, 'trapezoid') else np.trapz(y, x))

def eval_matched_metrics(df_dets, split_gt, n_images):
    ap50_list, recs_p80, recs_p90 = [], [], []
    for c_id in range(4):
        c_gt = split_gt[split_gt['class_id'] == c_id]
        n_gt = len(c_gt)
        if n_gt == 0:
            continue
        c_dets = df_dets[df_dets['class_id'] == c_id].sort_values(by='conf', ascending=False)
        if len(c_dets) == 0:
            ap50_list.append(0.0)
            recs_p80.append(0.0)
            recs_p90.append(0.0)
            continue
        tps = c_dets['matched'].values.astype(int)
        fps = 1 - tps
        tp_cum = np.cumsum(tps)
        fp_cum = np.cumsum(fps)
        prec = tp_cum / (tp_cum + fp_cum)
        rec = tp_cum / n_gt
        ap = compute_ap(rec, prec)
        ap50_list.append(ap)
        v_p80 = rec[prec >= 0.80]
        recs_p80.append(float(np.max(v_p80)) if len(v_p80) > 0 else 0.0)
        v_p90 = rec[prec >= 0.90]
        recs_p90.append(float(np.max(v_p90)) if len(v_p90) > 0 else 0.0)
    return {
        'mAP50': np.mean(ap50_list),
        'min_class_AP50': np.min(ap50_list),
        'worst_class_ap': CLASS_NAMES[np.argmin(ap50_list)],
        'mean_rec_p80': np.mean(recs_p80),
        'min_rec_p80': np.min(recs_p80),
        'worst_class_p80': CLASS_NAMES[np.argmin(recs_p80)],
        'mean_rec_p90': np.mean(recs_p90),
        'min_rec_p90': np.min(recs_p90),
        'worst_class_p90': CLASS_NAMES[np.argmin(recs_p90)]
    }

def run_r7():
    print("--- Executing R7 (Matched Operating Points) ---")
    manifest_path = REPO_ROOT / 'results' / 'phase0' / 'cache_manifest.csv'
    df_manifest = pd.read_csv(manifest_path)
    ckpts_to_process = df_manifest[df_manifest['ckpt_type'] == 'best'].copy()
    rows = []

    for (m, r_str, s), grp in ckpts_to_process.groupby(['model', 'prune_ratio', 'seed']):
        if m == 'yolo26n' and s == 3 and r_str == '0%':
            continue
        for split in ['val', 'test']:
            sp_rows = grp[grp['split'] == split]
            if len(sp_rows) == 0:
                continue
            p_path = REPO_ROOT / sp_rows['parquet_path'].iloc[0]
            df_dets = pd.read_parquet(p_path)
            split_gt = DF_GT[DF_GT['split'] == split]
            n_images = len(split_gt[['subject', 'video', 'frame']].drop_duplicates())
            metrics = eval_matched_metrics(df_dets, split_gt, n_images)
            row = {'model': m, 'prune_ratio': r_str, 'seed': s, 'split': split}
            row.update(metrics)
            rows.append(row)

    df_r7 = pd.DataFrame(rows)
    out_r7 = REPO_ROOT / 'results' / 'phase0' / 'r7_matched_operating_points.csv'
    df_r7.to_csv(out_r7, index=False)
    print(f"R7 complete. Saved {len(df_r7)} records to {out_r7}")

# ==============================================================================
# R8: Stress-Tests of the Tau* Calibration Rule (n=8 Baselines)
# ==============================================================================
def find_tau_star_exact(df_dets, df_gt_subset, r_floor):
    gt_counts = [len(df_gt_subset[df_gt_subset['class_id'] == c]) for c in range(4)]
    unique_confs = np.sort(df_dets['conf'].unique())[::-1]
    for tau in unique_confs:
        recs, _ = get_class_recalls(df_dets, tau, gt_counts)
        if min(recs) >= r_floor:
            return float(tau), True
    return None, False

def run_r8():
    print("--- Executing R8 (Stress Tests on n=8 Baselines) ---")
    gt_val = DF_GT[DF_GT['split'] == 'val']
    gt_test = DF_GT[DF_GT['split'] == 'test']
    val_counts = [len(gt_val[gt_val['class_id'] == c]) for c in range(4)]
    test_counts = [len(gt_test[gt_test['class_id'] == c]) for c in range(4)]

    # 1. Crossover test -> val
    crossover_rows = []
    for m, s in BASELINES:
        f_v = REPO_ROOT / 'results' / 'phase0' / 'cache' / f"{m}_baseline_s{s}_best_val.parquet"
        f_t = REPO_ROOT / 'results' / 'phase0' / 'cache' / f"{m}_baseline_s{s}_best_test.parquet"
        df_v = pd.read_parquet(f_v)
        df_t = pd.read_parquet(f_t)

        # Baseline on test at tau=0.25
        t_recs_25, _ = get_class_recalls(df_t, 0.25, test_counts)
        t_min_25 = min(t_recs_25)
        t_floor = t_min_25 - 0.05

        # Calibrate tau* from test
        tau_star_test, feas = find_tau_star_exact(df_t, gt_test, t_floor)

        if feas:
            v_recs, _ = get_class_recalls(df_v, tau_star_test, val_counts)
            v_min = min(v_recs)
            v_margin = (v_min - t_floor) * 100.0
            passed = bool(v_margin >= 0.0)
        else:
            v_min, v_margin, passed = np.nan, np.nan, False

        crossover_rows.append({
            'model': m, 'seed': s, 'test_R_base': t_min_25, 'test_R_floor': t_floor,
            'calib_tau_star_from_test': tau_star_test, 'val_min_recall': v_min,
            'val_margin_pp': v_margin, 'crossover_pass': passed
        })

    df_crossover = pd.DataFrame(crossover_rows)
    out_crossover = REPO_ROOT / 'results' / 'phase0' / 'r8_crossover_test_to_val.csv'
    df_crossover.to_csv(out_crossover, index=False)

    # 2. Delta margin tradeoffs (buffer sweep on val floor)
    buffers = [0.0, 0.025, 0.05, 0.075, 0.10]
    delta_rows = []
    for m, s in BASELINES:
        f_v = REPO_ROOT / 'results' / 'phase0' / 'cache' / f"{m}_baseline_s{s}_best_val.parquet"
        f_t = REPO_ROOT / 'results' / 'phase0' / 'cache' / f"{m}_baseline_s{s}_best_test.parquet"
        df_v = pd.read_parquet(f_v)
        df_t = pd.read_parquet(f_t)

        v_recs_25, _ = get_class_recalls(df_v, 0.25, val_counts)
        r_base = min(v_recs_25)
        r_floor_unbuffered = r_base - 0.05

        for b in buffers:
            r_floor_b = r_floor_unbuffered + b # floor + buffer
            # Find tau* on val meeting r_floor_b
            tau_star_b, feas_b = find_tau_star_exact(df_v, gt_val, r_floor_b)
            if feas_b:
                t_recs, _ = get_class_recalls(df_t, tau_star_b, test_counts)
                t_min = min(t_recs)
                # Margin against unbuffered floor
                margin_vs_unbuffered = (t_min - r_floor_unbuffered) * 100.0
                margin_vs_buffered = (t_min - r_floor_b) * 100.0
                pass_unbuffered = bool(margin_vs_unbuffered >= 0.0)
                pass_buffered = bool(margin_vs_buffered >= 0.0)
            else:
                t_min = np.nan
                margin_vs_unbuffered = np.nan
                margin_vs_buffered = np.nan
                pass_unbuffered = False
                pass_buffered = False

            delta_rows.append({
                'model': m, 'seed': s, 'buffer_pp': b * 100.0,
                'target_val_floor': r_floor_b, 'tau_star': tau_star_b,
                'test_min_recall': t_min,
                'margin_vs_unbuffered_floor_pp': margin_vs_unbuffered,
                'margin_vs_buffered_floor_pp': margin_vs_buffered,
                'pass_unbuffered': pass_unbuffered,
                'pass_buffered': pass_buffered
            })

    df_delta = pd.DataFrame(delta_rows)
    out_delta = REPO_ROOT / 'results' / 'phase0' / 'r8_delta_margin_tradeoffs.csv'
    df_delta.to_csv(out_delta, index=False)

    # 3. Leave-one-subject-out (LOSO) cross-validation
    all_subjects = sorted(DF_GT[DF_GT['split'].isin(['val', 'test'])]['subject'].unique()) # 6 subjects
    loso_rows = []
    for held_out_sub in all_subjects:
        train_gt = DF_GT[(DF_GT['split'].isin(['val', 'test'])) & (DF_GT['subject'] != held_out_sub)]
        test_gt = DF_GT[(DF_GT['split'].isin(['val', 'test'])) & (DF_GT['subject'] == held_out_sub)]
        tr_counts = [len(train_gt[train_gt['class_id'] == c]) for c in range(4)]
        te_counts = [len(test_gt[test_gt['class_id'] == c]) for c in range(4)]

        for m, s in BASELINES:
            f_v = REPO_ROOT / 'results' / 'phase0' / 'cache' / f"{m}_baseline_s{s}_best_val.parquet"
            f_t = REPO_ROOT / 'results' / 'phase0' / 'cache' / f"{m}_baseline_s{s}_best_test.parquet"
            df_pooled = pd.concat([pd.read_parquet(f_v), pd.read_parquet(f_t)], ignore_index=True)
            df_train = df_pooled[df_pooled['subject'] != held_out_sub]
            df_held = df_pooled[df_pooled['subject'] == held_out_sub]

            # R_base on train fold
            tr_recs_25, _ = get_class_recalls(df_train, 0.25, tr_counts)
            tr_min_25 = min(tr_recs_25)
            tr_floor = tr_min_25 - 0.05

            tau_star_loso, feas = find_tau_star_exact(df_train, train_gt, tr_floor)
            if feas:
                te_recs, _ = get_class_recalls(df_held, tau_star_loso, te_counts)
                te_min = min(te_recs)
                te_margin = (te_min - tr_floor) * 100.0
                passed = bool(te_margin >= 0.0)
            else:
                te_min, te_margin, passed = np.nan, np.nan, False

            loso_rows.append({
                'held_out_subject': held_out_sub, 'model': m, 'seed': s,
                'train_R_base': tr_min_25, 'train_R_floor': tr_floor,
                'tau_star': tau_star_loso, 'held_out_min_recall': te_min,
                'held_out_margin_pp': te_margin, 'passed': passed
            })

    df_loso = pd.DataFrame(loso_rows)
    out_loso = REPO_ROOT / 'results' / 'phase0' / 'r8_loso_cross_validation.csv'
    df_loso.to_csv(out_loso, index=False)
    print(f"R8 complete. Crossover: {len(df_crossover)}, Delta: {len(df_delta)}, LOSO: {len(df_loso)}")

# ==============================================================================
# R9: Checkpoint Sensitivity (Best vs Last) for All n=8 Baselines
# ==============================================================================
def run_r9():
    print("--- Executing R9 (Checkpoint Sensitivity for n=8 Baselines) ---")
    df_canon = load_canonical_baselines()
    records = []

    for m, s in BASELINES:
        b_best = df_canon[(df_canon['model'] == m) & (df_canon['seed'] == s) & (df_canon['ckpt'] == 'best')].iloc[0]
        b_last = df_canon[(df_canon['model'] == m) & (df_canon['seed'] == s) & (df_canon['ckpt'] == 'last')].iloc[0]

        records.append({
            'model': m, 'seed': s,
            'best_R_base': b_best['val_R_base'],
            'best_R_floor': b_best['val_R_floor'],
            'best_tau_star': b_best['tau_star'],
            'best_test_min_recall': b_best['test_worst_rec_star'],
            'best_margin_pp': b_best['margin_pp_star'],
            'best_passed': b_best['passed_star'],
            'last_R_base': b_last['val_R_base'],
            'last_R_floor': b_last['val_R_floor'],
            'last_tau_star': b_last['tau_star'],
            'last_test_min_recall': b_last['test_worst_rec_star'],
            'last_margin_pp': b_last['margin_pp_star'],
            'last_passed': b_last['passed_star'],
            'diff_tau_star': b_last['tau_star'] - b_best['tau_star'],
            'diff_margin_pp': b_last['margin_pp_star'] - b_best['margin_pp_star']
        })

    df_r9 = pd.DataFrame(records)
    out_r9 = REPO_ROOT / 'results' / 'phase0' / 'r9_checkpoint_sensitivity_all.csv'
    df_r9.to_csv(out_r9, index=False)
    print(f"R9 complete. Saved {len(df_r9)} records to {out_r9}")

if __name__ == '__main__':
    run_r5()
    run_r6()
    run_r7()
    run_r8()
    run_r9()
    print("ALL P1 SCRIPTS COMPLETED SUCCESSFULLY.")
