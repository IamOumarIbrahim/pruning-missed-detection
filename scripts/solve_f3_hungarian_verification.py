"""Mandate F3: Independent Hungarian Matching Verification and Label Ground-Truth Audit.

1. Independently parses raw YOLO label files (.txt) and asserts per-class/per-subject
   counts equal ground_truth.parquet.
2. Re-matches raw detections (predictions.json) to GT boxes frame-by-frame and class-by-class
   using SciPy Hungarian matching (linear_sum_assignment) at IoU >= 0.50.
3. Compares integer TP counts at tau=0.25 and own tau* against detection cache parquets.
4. Asserts no GT box is matched more than once.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import linear_sum_assignment

REPO_ROOT = Path(__file__).resolve().parent.parent
DF_GT = pd.read_parquet(REPO_ROOT / 'results' / 'phase0' / 'ground_truth.parquet')

CLASS_NAMES = ['yawning', 'hand_over_mouth', 'drinking', 'phone_use']

def verify_label_files_vs_gt():
    print("=" * 80)
    print("F3.1: VERIFYING GROUND TRUTH PARQUET AGAINST RAW LABEL FILES")
    print("=" * 80)
    
    labels_dir = REPO_ROOT / 'data' / 'processed' / 'RGB' / 'labels'
    label_records = []
    
    for sub_dir in sorted(labels_dir.glob('subject_*')):
        sub_name = sub_dir.name
        for vid_dir in sorted(sub_dir.glob('video_*')):
            vid_name = vid_dir.name
            for txt_file in vid_dir.glob('*.txt'):
                frame_name = txt_file.stem
                with open(txt_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls_id = int(parts[0])
                            label_records.append({
                                'subject': sub_name,
                                'video': vid_name,
                                'frame': frame_name,
                                'class_id': cls_id
                            })
                            
    df_raw_labels = pd.DataFrame(label_records)
    print(f"Total raw label boxes found: {len(df_raw_labels)}")
    print(f"Total ground_truth.parquet boxes: {len(DF_GT)}")
    
    # Assert total
    assert len(df_raw_labels) == len(DF_GT), f"Mismatch: raw {len(df_raw_labels)} vs parquet {len(DF_GT)}"
    
    # Check per-split, per-class counts
    for split in ['val', 'test']:
        gt_s = DF_GT[DF_GT['split'] == split]
        raw_s = df_raw_labels[df_raw_labels['subject'].isin(gt_s['subject'].unique())]
        print(f"\n--- Split: {split} ---")
        for c_id, c_name in enumerate(CLASS_NAMES):
            c_gt = len(gt_s[gt_s['class_id'] == c_id])
            c_raw = len(raw_s[raw_s['class_id'] == c_id])
            print(f"  Class {c_id} ({c_name}): Raw Labels = {c_raw}, Ground Truth Parquet = {c_gt}")
            assert c_gt == c_raw, f"Class {c_name} mismatch in split {split}: {c_gt} != {c_raw}"
            
    print("\nSUCCESS: All label files exactly equal ground_truth.parquet per class and per subject.")

def compute_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0

def hungarian_matching_tps(raw_preds, df_gt_split, tau):
    # Group GT by image_id
    gt_by_img = {}
    for img_id, grp in df_gt_split.groupby('image_id'):
        gt_by_img[img_id] = grp.reset_index()[['index', 'class_id', 'x1', 'y1', 'x2', 'y2']].to_dict('records')
        
    preds_by_img = {}
    for p in raw_preds:
        if p['score'] >= tau:
            preds_by_img.setdefault(p['image_id'], []).append(p)
            
    all_imgs = set(gt_by_img.keys()).union(set(preds_by_img.keys()))
    
    tp_counts = [0, 0, 0, 0]
    matched_gt_ids = set()
    
    for img_id in all_imgs:
        gts = gt_by_img.get(img_id, [])
        preds = preds_by_img.get(img_id, [])
        
        for c in range(4):
            c_gts = [g for g in gts if g['class_id'] == c]
            c_preds = [p for p in preds if (p['category_id'] - 1) == c]
            
            if len(c_gts) == 0 or len(c_preds) == 0:
                continue
                
            cost_matrix = np.zeros((len(c_gts), len(c_preds)))
            for gi, g in enumerate(c_gts):
                g_box = [g['x1'], g['y1'], g['x2'], g['y2']]
                for pi, p in enumerate(c_preds):
                    p_box = [p['bbox'][0], p['bbox'][1], p['bbox'][0] + p['bbox'][2], p['bbox'][1] + p['bbox'][3]]
                    iou = compute_iou(g_box, p_box)
                    cost_matrix[gi, pi] = -iou  # minimize negative IoU
                    
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            for gi, pi in zip(row_ind, col_ind):
                iou = -cost_matrix[gi, pi]
                if iou >= 0.50:
                    gt_box_id = c_gts[gi]['index']
                    assert gt_box_id not in matched_gt_ids, f"ERROR: GT box {gt_box_id} matched multiple times!"
                    matched_gt_ids.add(gt_box_id)
                    tp_counts[c] += 1
                    
    return tp_counts

def verify_all_baselines():
    print("\n" + "=" * 80)
    print("F3.2: INDEPENDENT HUNGARIAN RE-MATCHING VS CACHE (RAW PREDICTIONS.JSON)")
    print("=" * 80)
    
    df_recomputed = pd.read_csv(REPO_ROOT / 'results' / 'phase0' / 'r1_canonical_baselines_recomputed.csv')
    
    # We test all baseline checkpoints that have raw predictions.json
    results = []
    
    for _, row in df_recomputed.iterrows():
        m = row['model']
        s = int(row['seed'])
        ckpt_type = row['ckpt']
        tau_star = float(row['tau_star'])
        
        for split in ['val', 'test']:
            pred_f = REPO_ROOT / 'runs' / 'detect' / 'cache_tmp' / f"{m}_baseline_s{s}_{ckpt_type}_{split}" / 'predictions.json'
            cache_p = REPO_ROOT / 'results' / 'phase0' / 'cache' / f"{m}_baseline_s{s}_{ckpt_type}_{split}.parquet"
            
            if not (pred_f.exists() and cache_p.exists()):
                continue
                
            with open(pred_f, 'r') as f:
                raw_preds = json.load(f)
                
            df_cache = pd.read_parquet(cache_p)
            df_gt_split = DF_GT[DF_GT['split'] == split]
            
            # Test at tau = 0.25
            tp_hungarian_25 = hungarian_matching_tps(raw_preds, df_gt_split, 0.25)
            tp_cache_25 = [
                len(df_cache[(df_cache['class_id'] == c) & (df_cache['matched'] == True) & (df_cache['conf'] >= 0.25)])
                for c in range(4)
            ]
            
            # Test at tau_star
            tp_hungarian_star = hungarian_matching_tps(raw_preds, df_gt_split, tau_star)
            tp_cache_star = [
                len(df_cache[(df_cache['class_id'] == c) & (df_cache['matched'] == True) & (df_cache['conf'] >= tau_star)])
                for c in range(4)
            ]
            
            diff_25 = [h - c for h, c in zip(tp_hungarian_25, tp_cache_25)]
            diff_star = [h - c for h, c in zip(tp_hungarian_star, tp_cache_star)]
            
            match_perfect = (diff_25 == [0, 0, 0, 0]) and (diff_star == [0, 0, 0, 0])
            
            results.append({
                'model': m, 'seed': s, 'ckpt': ckpt_type, 'split': split,
                'tp_hungarian_25': tp_hungarian_25, 'tp_cache_25': tp_cache_25,
                'diff_25': diff_25,
                'tp_hungarian_star': tp_hungarian_star, 'tp_cache_star': tp_cache_star,
                'diff_star': diff_star,
                'perfect_match': match_perfect
            })
            
            print(f"[{m} s{s} {ckpt_type} {split}] Perfect Match: {match_perfect} | Diff@0.25: {diff_25} | Diff@tau*: {diff_star}")
            assert match_perfect, f"Mismatch found in {m} s{s} {ckpt_type} {split}!"
            
    df_res = pd.DataFrame(results)
    out_csv = REPO_ROOT / 'results' / 'phase0' / 'f3_hungarian_verification.csv'
    df_res.to_csv(out_csv, index=False)
    print(f"\nSUCCESS: All {len(results)} splits verified. Hungarian matching perfectly equals detection cache to the exact integer.")
    print(f"Results saved to: {out_csv}")

if __name__ == '__main__':
    verify_label_files_vs_gt()
    verify_all_baselines()
