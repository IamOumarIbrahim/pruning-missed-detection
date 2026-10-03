import json
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent
CLASS_NAMES = ['yawning', 'hand_over_mouth', 'drinking', 'phone_use']
GT_TEST_COUNTS = [33, 28, 56, 497]

def run_metric_validation():
    print("="*75)
    print("PHASE 0.1: METRIC VALIDATION & DISCREPANCY AUDIT")
    print("="*75)
    
    with open(REPO_ROOT / 'results' / 'calibrated_eval_data.json') as f:
        calib_data = json.load(f)
        
    df_gt = pd.read_parquet(REPO_ROOT / 'results' / 'phase0' / 'ground_truth.parquet')
    df_gt_test = df_gt[df_gt['split'] == 'test']
    
    test_models = [
        ('yolo11n', 'models/yolo11n/baseline/seed_0/weights/best.pt', calib_data['yolo11n']['0%']['0']['tau_star']),
        ('yolo26n', 'models/yolo26n/baseline/seed_0/weights/best.pt', calib_data['yolo26n']['0%']['0']['tau_star'])
    ]
    
    results_summary = []
    
    for model_name, ckpt_p, tau_star in test_models:
        print(f"\n=======================================================")
        print(f"AUDITING MODEL: {model_name} (tau* = {tau_star:.4f})")
        print(f"=======================================================")
        m = YOLO(ckpt_p)
        
        # 1. Run baseline at conf=0.001 to get raw predictions.json for offline recompute
        out_dir = REPO_ROOT / 'runs' / 'detect' / 'phase0_audit' / f"{model_name}_raw"
        res_raw = m.val(
            data='configs/dmd_rgb.yaml',
            split='test',
            batch=64,
            conf=0.001,
            iou=0.7,
            max_det=300,
            save_json=True,
            plots=False,
            project=str(out_dir.parent),
            name=out_dir.name,
            exist_ok=True
        )
        with open(out_dir / 'predictions.json') as f:
            preds_raw = json.load(f)
            
        # Match preds_raw offline
        gt_by_img = {}
        for img_id, grp in df_gt_test.groupby('image_id'):
            gt_by_img[img_id] = grp[['class_id', 'x1', 'y1', 'x2', 'y2']].to_dict('records')
            
        preds_by_img = {}
        for p in preds_raw:
            preds_by_img.setdefault(p['image_id'], []).append(p)
            
        matched_preds = []
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
                    # IoU
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
                matched_preds.append({
                    'class_id': p_cat,
                    'conf': p['score'],
                    'matched': is_tp
                })
        df_matched = pd.DataFrame(matched_preds)
        
        # Test thresholds: 0.25, 0.50, tau*
        target_taus = [0.25, 0.50, tau_star]
        
        for tau in target_taus:
            print(f"\n--- Threshold tau = {tau:.4f} ---")
            # A: Real inference in Ultralytics with conf = tau
            t_dir = REPO_ROOT / 'runs' / 'detect' / 'phase0_audit' / f"{model_name}_conf_{int(tau*1000)}"
            res_tau = m.val(
                data='configs/dmd_rgb.yaml',
                split='test',
                batch=64,
                conf=tau,
                iou=0.7,
                max_det=300,
                save_json=False,
                plots=False,
                project=str(t_dir.parent),
                name=t_dir.name,
                exist_ok=True
            )
            # Read Ultralytics per-class recall from res_tau
            # Ultralytics res_tau.box.r is an array of recalls per class at conf=tau
            ultra_recalls = list(res_tau.box.r) if hasattr(res_tau.box, 'r') else []
            
            # B: Offline recompute
            offline_recalls = []
            for c_id in range(4):
                n_gt = GT_TEST_COUNTS[c_id]
                n_tp = len(df_matched[(df_matched['class_id'] == c_id) & (df_matched['matched'] == True) & (df_matched['conf'] >= tau)])
                offline_recalls.append(n_tp / n_gt if n_gt > 0 else 0.0)
                
            # C: Stored in calibrated_eval_data.json
            if abs(tau - 0.25) < 1e-4:
                stored_rec = calib_data[model_name]['0%']['0']['test_at_tau_25']
            elif abs(tau - 0.50) < 1e-4:
                stored_rec = calib_data[model_name]['0%']['0']['test_at_tau_50']
            else:
                stored_rec = calib_data[model_name]['0%']['0']['test_at_tau_star']
            stored_min_rec = stored_rec['min_recall']
            
            print(f"{'Class':15s} | {'Real Ultralytics':18s} | {'Offline Recompute':18s} | {'Diff (Real - Off)':18s}")
            print("-" * 75)
            for c_id, cname in enumerate(CLASS_NAMES):
                u_val = ultra_recalls[c_id] if len(ultra_recalls) > c_id else -1.0
                o_val = offline_recalls[c_id]
                diff = u_val - o_val
                print(f"{cname:15s} | {u_val:18.4f} | {o_val:18.4f} | {diff:+18.4f}")
                results_summary.append({
                    'model': model_name,
                    'tau': tau,
                    'class': cname,
                    'ultra_val': u_val,
                    'offline_val': o_val,
                    'diff': diff,
                    'stored_min_recall': stored_min_rec
                })
                
    pd.DataFrame(results_summary).to_csv(REPO_ROOT / 'results' / 'phase0' / 'metric_validation_summary.csv', index=False)
    print("\nMetric validation complete. Summary saved to results/phase0/metric_validation_summary.csv")

if __name__ == '__main__':
    run_metric_validation()
