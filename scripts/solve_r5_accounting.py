import json
import numpy as np
import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CLASS_NAMES = ['yawning', 'hand_over_mouth', 'drinking', 'phone_use']

def run_r5_analysis():
    print("=" * 80)
    print("MANDATE R5: PER-SUBJECT RECALL & DEEP ACCOUNTING OF SEED 1 YAWNING COLLAPSE")
    print("=" * 80)

    df_gt = pd.read_parquet(REPO_ROOT / 'results' / 'phase0' / 'ground_truth.parquet')
    df_events = pd.read_parquet(REPO_ROOT / 'results' / 'phase0' / 'events.parquet')

    with open(REPO_ROOT / 'results' / 'calibrated_eval_data.json') as f:
        calib_data = json.load(f)

    # 1. Forensic accounting of YOLO11n Seed 1 test yawning
    # Check if cache file exists for yolo11n_baseline_s1_best_test
    cache_s1_test = REPO_ROOT / 'results' / 'phase0' / 'cache' / 'yolo11n_baseline_s1_best_test.parquet'
    cache_s1_val = REPO_ROOT / 'results' / 'phase0' / 'cache' / 'yolo11n_baseline_s1_best_val.parquet'

    if not cache_s1_test.exists():
        print(f"Waiting for {cache_s1_test.name} to be cached by task-1570...")
        return False

    df_p_test = pd.read_parquet(cache_s1_test)
    df_p_val = pd.read_parquet(cache_s1_val)

    gt_test_yawn = df_gt[(df_gt['split'] == 'test') & (df_gt['class_name'] == 'yawning')].copy()
    
    # Map event_id
    ev_ids = []
    for _, g in gt_test_yawn.iterrows():
        ev = df_events[(df_events['subject'] == g['subject']) & 
                       (df_events['video'] == g['video']) & 
                       (df_events['class_name'] == 'yawning') & 
                       (df_events['start_frame'] <= g['frame_num']) & 
                       (df_events['end_frame'] >= g['frame_num'])]
        ev_ids.append(ev['event_id'].values[0] if len(ev) > 0 else 'no_event')
    gt_test_yawn['event_id'] = ev_ids

    # For each GT box, find best matching TP detection conf
    tau_star_s1 = calib_data['yolo11n']['0%']['1']['tau_star']  # 0.7555
    print(f"YOLO11n Seed 1 calibrated tau* = {tau_star_s1:.4f}")

    matched_tps = df_p_test[(df_p_test['class_name'] == 'yawning') & (df_p_test['matched'] == True)]

    # Match each GT box to max conf TP on that frame
    max_confs = []
    best_ious = []
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

    print(f"TP at tau=0.25: {gt_test_yawn['tp_025'].sum()} / 33 ({gt_test_yawn['tp_025'].mean():.4f})")
    print(f"TP at tau*={tau_star_s1:.4f}: {gt_test_yawn['tp_tau_star'].sum()} / 33 ({gt_test_yawn['tp_tau_star'].mean():.4f})")

    lost_frames = gt_test_yawn[gt_test_yawn['tp_025'] & (~gt_test_yawn['tp_tau_star'])].copy()
    lost_frames = lost_frames.sort_values(by='best_conf', ascending=False).reset_index(drop=True)
    print(f"Total lost frames: {len(lost_frames)}")

    out_lost_csv = REPO_ROOT / 'results' / 'phase0' / 'r5_all_lost_yawn_frames_seed1.csv'
    lost_frames[['subject', 'video', 'frame', 'frame_num', 'event_id', 'best_conf', 'best_iou']].to_csv(out_lost_csv, index=False)
    print(f"Full table of all 21 lost yawn frames saved to: {out_lost_csv}")
    print("\nTop 10 lost yawn frames (ranked by confidence):")
    print(lost_frames[['subject', 'video', 'frame', 'frame_num', 'event_id', 'best_conf', 'best_iou']].head(10).to_string(index=False))

    # Event level collapse breakdown
    print("\n--- EVENT-LEVEL LOSS BREAKDOWN ---")
    ev_summary = []
    for ev_id, grp in lost_frames.groupby('event_id'):
        tot_ev_boxes = len(gt_test_yawn[gt_test_yawn['event_id'] == ev_id])
        ev_summary.append({
            'event_id': ev_id,
            'subject': grp['subject'].iloc[0],
            'video': grp['video'].iloc[0],
            'total_event_frames': tot_ev_boxes,
            'lost_frames': len(grp),
            'max_conf_in_event': grp['best_conf'].max(),
            'min_conf_in_event': grp['best_conf'].min(),
            'confs': sorted(grp['best_conf'].tolist(), reverse=True)
        })
    df_ev_sum = pd.DataFrame(ev_summary).sort_values(by='max_conf_in_event', ascending=False)
    print(df_ev_sum[['event_id', 'subject', 'video', 'total_event_frames', 'lost_frames', 'max_conf_in_event']].to_string(index=False))

    # TP confidence histogram comparison between val and test
    val_yawn_tps = df_p_val[(df_p_val['class_name'] == 'yawning') & (df_p_val['matched'] == True)]['conf'].values
    test_yawn_tps = matched_tps['conf'].values

    print("\n--- TP-CONFIDENCE DISTRIBUTION FOR YAWNING (SEED 1) ---")
    print(f"Val Yawning TPs: count={len(val_yawn_tps)}, median={np.median(val_yawn_tps):.4f}, mean={np.mean(val_yawn_tps):.4f}, max={np.max(val_yawn_tps):.4f}, 90th={np.percentile(val_yawn_tps, 90):.4f}")
    print(f"Test Yawning TPs: count={len(test_yawn_tps)}, median={np.median(test_yawn_tps):.4f}, mean={np.mean(test_yawn_tps):.4f}, max={np.max(test_yawn_tps):.4f}, 90th={np.percentile(test_yawn_tps, 90):.4f}")

    bins = [0.0, 0.25, 0.50, 0.70, 0.75, 0.7555, 0.80, 0.85, 0.90, 1.0]
    val_hist, _ = np.histogram(val_yawn_tps, bins=bins)
    test_hist, _ = np.histogram(test_yawn_tps, bins=bins)
    
    hist_df = pd.DataFrame({
        'bin_range': [f"[{bins[i]:.4f}, {bins[i+1]:.4f})" for i in range(len(bins)-1)],
        'val_tp_count': val_hist,
        'val_tp_pct': val_hist / len(val_yawn_tps) * 100,
        'test_tp_count': test_hist,
        'test_tp_pct': test_hist / len(test_yawn_tps) * 100
    })
    out_hist_csv = REPO_ROOT / 'results' / 'phase0' / 'r5_tp_confidence_histogram_yawn_seed1.csv'
    hist_df.to_csv(out_hist_csv, index=False)
    print("\nTP Confidence Histogram Table:")
    print(hist_df.to_string(index=False))

    # Per-subject x per-class recall for all 6 baselines
    print("\n--- PER-SUBJECT x PER-CLASS RECALL FOR BASELINES ---")
    baselines = [
        ('yolo11n', 0), ('yolo11n', 1), ('yolo11n', 2),
        ('yolo26n', 0), ('yolo26n', 1), ('yolo26n', 2)
    ]
    per_sub_rows = []
    taus_to_eval = [0.10, 0.25, 0.50, 0.75]

    for model, s in baselines:
        b_tau_star = calib_data[model]['0%'][str(s)]['tau_star']
        for split in ['val', 'test']:
            c_file = REPO_ROOT / 'results' / 'phase0' / 'cache' / f"{model}_baseline_s{s}_best_{split}.parquet"
            if not c_file.exists():
                print(f"Skipping {c_file.name} (not cached yet)")
                continue
            df_det = pd.read_parquet(c_file)
            split_gt = df_gt[df_gt['split'] == split]
            
            for sub in sorted(split_gt['subject'].unique()):
                sub_gt = split_gt[split_gt['subject'] == sub]
                sub_det = df_det[df_det['subject'] == sub]
                
                for c_id, c_name in enumerate(CLASS_NAMES):
                    c_gt_cnt = len(sub_gt[sub_gt['class_id'] == c_id])
                    c_ev_cnt = len(df_events[(df_events['subject'] == sub) & (df_events['class_name'] == c_name)])
                    
                    row = {
                        'model': model,
                        'seed': s,
                        'split': split,
                        'subject': sub,
                        'class_name': c_name,
                        'gt_boxes': c_gt_cnt,
                        'gt_events': c_ev_cnt,
                        'tau_star': b_tau_star
                    }
                    
                    for tau in taus_to_eval:
                        tp_tau = len(sub_det[(sub_det['class_id'] == c_id) & (sub_det['matched'] == True) & (sub_det['conf'] >= tau)])
                        row[f'recall_tau_{int(tau*100):02d}'] = tp_tau / c_gt_cnt if c_gt_cnt > 0 else np.nan
                        
                    tp_star = len(sub_det[(sub_det['class_id'] == c_id) & (sub_det['matched'] == True) & (sub_det['conf'] >= b_tau_star)])
                    row['recall_own_tau_star'] = tp_star / c_gt_cnt if c_gt_cnt > 0 else np.nan
                    
                    per_sub_rows.append(row)

    if per_sub_rows:
        df_per_sub = pd.DataFrame(per_sub_rows)
        out_sub_csv = REPO_ROOT / 'results' / 'phase0' / 'r5_per_subject_baseline_recall.csv'
        df_per_sub.to_csv(out_sub_csv, index=False)
        print(f"Per-subject baseline recall table saved to: {out_sub_csv}")

    return True

if __name__ == '__main__':
    run_r5_analysis()
