import json
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent
CLASS_NAMES = ['yawning', 'hand_over_mouth', 'drinking', 'phone_use']

# Load GT counts and parquet
df_gt = pd.read_parquet(REPO_ROOT / 'results' / 'phase0' / 'ground_truth.parquet')

with open(REPO_ROOT / 'results' / 'calibrated_eval_data.json') as f:
    calib_data = json.load(f)

def run_r2_audit():
    print("=" * 80)
    print("MANDATE R2: RESOLUTION OF 0.1 METRIC DISCREPANCY")
    print("=" * 80)

    # We will test yolo11n seed 0 and yolo26n seed 0 on val and test
    models_to_test = [
        ('yolo11n', 0, 'models/yolo11n/baseline/seed_0/weights/best.pt', calib_data['yolo11n']['0%']['0']['tau_star']),
        ('yolo26n', 0, 'models/yolo26n/baseline/seed_0/weights/best.pt', calib_data['yolo26n']['0%']['0']['tau_star'])
    ]

    splits = ['val', 'test']
    
    detailed_rows = []

    for model_name, seed, ckpt_path, tau_star in models_to_test:
        m = YOLO(ckpt_path)
        for split in splits:
            split_gt = df_gt[df_gt['split'] == split]
            gt_counts = [len(split_gt[split_gt['class_id'] == c]) for c in range(4)]
            gt_by_img = {}
            for img_id, grp in split_gt.groupby('image_id'):
                gt_by_img[img_id] = grp[['class_id', 'x1', 'y1', 'x2', 'y2']].to_dict('records')

            # 1. Run conf=0.001 pass to get (b) r_curve and raw predictions for (c)
            out_raw_dir = REPO_ROOT / 'runs' / 'detect' / 'r2_audit' / f"{model_name}_{split}_raw"
            res_001 = m.val(
                data='configs/dmd_rgb.yaml',
                split=split,
                batch=64,
                device=0,
                workers=0,
                conf=0.001,
                iou=0.7,
                max_det=300,
                save_json=True,
                plots=False,
                verbose=False,
                project=str(out_raw_dir.parent),
                name=out_raw_dir.name,
                exist_ok=True
            )
            
            with open(out_raw_dir / 'predictions.json') as f:
                preds_raw = json.load(f)
                
            # Extract r_curve from res_001
            r_curve_001 = None
            for cr in res_001.box.curves_results:
                if cr[3] == 'Recall' and cr[2] == 'Confidence':
                    r_curve_001 = (cr[0], cr[1])
                    break
            
            # Offline match predictions.json to GT (one-to-one greedy matching by IoU >= 0.50)
            preds_by_img = {}
            for p in preds_raw:
                preds_by_img.setdefault(p['image_id'], []).append(p)
                
            matched_records = []
            for img_id, p_list in preds_by_img.items():
                p_list_sorted = sorted(p_list, key=lambda x: x['score'], reverse=True)
                img_gts = gt_by_img.get(img_id, [])
                matched_gt = set()
                for p in p_list_sorted:
                    p_cat = p['category_id'] - 1
                    p_box = [p['bbox'][0], p['bbox'][1], p['bbox'][0] + p['bbox'][2], p['bbox'][1] + p['bbox'][3]]
                    best_iou = 0.0
                    best_gt_idx = -1
                    for g_idx, g in enumerate(img_gts):
                        if g_idx in matched_gt or g['class_id'] != p_cat:
                            continue
                        xA, yA = max(p_box[0], g['x1']), max(p_box[1], g['y1'])
                        xB, yB = min(p_box[2], g['x2']), min(p_box[3], g['y2'])
                        inter = max(0, xB - xA) * max(0, yB - yA)
                        area1 = (p_box[2] - p_box[0]) * (p_box[3] - p_box[1])
                        area2 = (g['x2'] - g['x1']) * (g['y2'] - g['y1'])
                        union = area1 + area2 - inter
                        iou = inter / union if union > 0 else 0.0
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = g_idx
                    is_tp = (best_iou >= 0.50 and best_gt_idx >= 0)
                    if is_tp:
                        matched_gt.add(best_gt_idx)
                    matched_records.append({
                        'class_id': p_cat,
                        'conf': p['score'],
                        'is_tp': is_tp
                    })
            df_matched = pd.DataFrame(matched_records)
            
            target_taus = [0.10, 0.25, 0.50, tau_star]
            
            for tau in target_taus:
                # (a) Run model.val(conf=tau)
                t_dir = REPO_ROOT / 'runs' / 'detect' / 'r2_audit' / f"{model_name}_{split}_conf_{int(tau*1000)}"
                res_tau = m.val(
                    data='configs/dmd_rgb.yaml',
                    split=split,
                    batch=64,
                    device=0,
                    workers=0,
                    conf=tau,
                    iou=0.7,
                    max_det=300,
                    save_json=False,
                    plots=False,
                    verbose=False,
                    project=str(t_dir.parent),
                    name=t_dir.name,
                    exist_ok=True
                )
                
                # Attribute read in (a): res_tau.box.r
                reported_attr_r = list(res_tau.box.r)
                
                # (b) Read r_curve from res_001 at tau
                idx_b = int(np.argmin(np.abs(r_curve_001[0] - tau)))
                r_curve_val = [float(r_curve_001[1][c, idx_b]) for c in range(4)]
                
                # (c) Offline recompute exact
                for c in range(4):
                    n_gt = gt_counts[c]
                    sub_c = df_matched[(df_matched['class_id'] == c) & (df_matched['is_tp'] == True) & (df_matched['conf'] >= tau)]
                    n_tp_c = len(sub_c)
                    rec_c = n_tp_c / n_gt if n_gt > 0 else 0.0
                    
                    # Integer TP for (a)
                    # Note: in ap_per_class, tp = (r * nt).round()
                    tp_a = int(round(reported_attr_r[c] * n_gt))
                    tp_b = int(round(r_curve_val[c] * n_gt))
                    tp_c = int(n_tp_c)
                    
                    detailed_rows.append({
                        'model': model_name,
                        'split': split,
                        'tau_target': tau,
                        'class_name': CLASS_NAMES[c],
                        'GT': n_gt,
                        'a_reported_attr_r': reported_attr_r[c],
                        'a_TP': tp_a,
                        'b_rcurve_recall': r_curve_val[c],
                        'b_TP': tp_b,
                        'c_exact_recall': rec_c,
                        'c_TP': tp_c,
                        'diff_c_minus_b': rec_c - r_curve_val[c],
                        'diff_c_minus_a': rec_c - reported_attr_r[c]
                    })
                    
    df_res = pd.DataFrame(detailed_rows)
    df_res.to_csv(REPO_ROOT / 'results' / 'phase0' / 'r2_3way_comparison.csv', index=False)
    print("3-way comparison table generated and saved to results/phase0/r2_3way_comparison.csv")
    
    # Now check whether R_base, floors, tau*, margins change for all 6 baselines
    print("\n" + "="*80)
    print("RECOMPUTING BASELINE CALIBRATIONS WITH EXACT (c) RECALL")
    print("="*80)
    
if __name__ == '__main__':
    run_r2_audit()
