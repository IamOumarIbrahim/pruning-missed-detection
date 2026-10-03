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

def match_predictions_to_gt(preds, split_gt):
    gt_by_img = {}
    for img_id, grp in split_gt.groupby('image_id'):
        gt_by_img[img_id] = grp[['class_id', 'x1', 'y1', 'x2', 'y2']].to_dict('records')

    preds_by_img = {}
    for p in preds:
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
                iou = compute_iou(p_box, [g['x1'], g['y1'], g['x2'], g['y2']])
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
    return pd.DataFrame(matched_records)

def run_r2_audit():
    print("=" * 80)
    print("MANDATE R2: RESOLUTION OF 0.1 METRIC DISCREPANCY")
    print("=" * 80)

    models_to_test = [
        ('yolo11n', 0, 'models/yolo11n/baseline/seed_0/weights/best.pt', calib_data['yolo11n']['0%']['0']['tau_star']),
        ('yolo26n', 0, 'models/yolo26n/baseline/seed_0/weights/best.pt', calib_data['yolo26n']['0%']['0']['tau_star'])
    ]

    splits = ['val', 'test']
    detailed_rows = []

    for model_name, seed, ckpt_path, tau_star in models_to_test:
        m = YOLO(ckpt_path)
        for split in splits:
            print(f"\nRunning {model_name} seed {seed} on {split} split...")
            split_gt = df_gt[df_gt['split'] == split]
            gt_counts = [len(split_gt[split_gt['class_id'] == c]) for c in range(4)]

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
                
            r_curve_001 = None
            for cr in res_001.box.curves_results:
                if cr[3] == 'Recall' and cr[2] == 'Confidence':
                    r_curve_001 = (cr[0], cr[1])
                    break
            
            df_matched = match_predictions_to_gt(preds_raw, split_gt)
            
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
                
                # Attribute read in (a): res_tau.box.r (Recall at max F1)
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
    out_csv = REPO_ROOT / 'results' / 'phase0' / 'r2_3way_comparison.csv'
    df_res.to_csv(out_csv, index=False)
    print(f"\n3-way comparison table generated and saved to {out_csv}")
    
    # Baseline calibration recomputation with exact recall
    print("\n" + "="*80)
    print("RECOMPUTING BASELINE CALIBRATIONS WITH EXACT (c) RECALL")
    print("="*80)
    
    baseline_checkpoints = [
        ('yolo11n', 0, 'models/yolo11n/baseline/seed_0/weights/best.pt'),
        ('yolo11n', 1, 'models/yolo11n/baseline/seed_1/weights/best.pt'),
        ('yolo11n', 2, 'models/yolo11n/baseline/seed_2/weights/best.pt'),
        ('yolo26n', 0, 'models/yolo26n/baseline/seed_0/weights/best.pt'),
        ('yolo26n', 1, 'models/yolo26n/baseline/seed_1/weights/best.pt'),
        ('yolo26n', 2, 'models/yolo26n/baseline/seed_2/weights/best.pt'),
    ]
    
    calib_recomp = []
    
    taus_grid = np.linspace(0.01, 0.90, 900)
    
    for model_name, seed, ckpt_p in baseline_checkpoints:
        m = YOLO(ckpt_p)
        print(f"Evaluating {model_name} seed {seed}...")
        
        # Val predictions
        out_v = REPO_ROOT / 'runs' / 'detect' / 'r2_calib' / f"{model_name}_s{seed}_val"
        m.val(data='configs/dmd_rgb.yaml', split='val', batch=64, device=0, workers=0, conf=0.001, iou=0.7, max_det=300, save_json=True, plots=False, verbose=False, project=str(out_v.parent), name=out_v.name, exist_ok=True)
        with open(out_v / 'predictions.json') as f:
            preds_val = json.load(f)
        df_matched_val = match_predictions_to_gt(preds_val, df_gt[df_gt['split'] == 'val'])
        val_gt_counts = [len(df_gt[(df_gt['split'] == 'val') & (df_gt['class_id'] == c)]) for c in range(4)]
        
        # Test predictions
        out_t = REPO_ROOT / 'runs' / 'detect' / 'r2_calib' / f"{model_name}_s{seed}_test"
        m.val(data='configs/dmd_rgb.yaml', split='test', batch=64, device=0, workers=0, conf=0.001, iou=0.7, max_det=300, save_json=True, plots=False, verbose=False, project=str(out_t.parent), name=out_t.name, exist_ok=True)
        with open(out_t / 'predictions.json') as f:
            preds_test = json.load(f)
        df_matched_test = match_predictions_to_gt(preds_test, df_gt[df_gt['split'] == 'test'])
        test_gt_counts = [len(df_gt[(df_gt['split'] == 'test') & (df_gt['class_id'] == c)]) for c in range(4)]
        
        # 1. Exact R_base on val at tau=0.25
        val_rec_25 = []
        for c in range(4):
            tp = len(df_matched_val[(df_matched_val['class_id'] == c) & (df_matched_val['is_tp'] == True) & (df_matched_val['conf'] >= 0.25)])
            val_rec_25.append(tp / val_gt_counts[c])
        r_base_exact = min(val_rec_25)
        worst_c_base = CLASS_NAMES[val_rec_25.index(r_base_exact)]
        r_floor_exact = r_base_exact - 0.05
        
        # 2. Find exact tau* on val
        best_tau_exact = None
        for tau in reversed(taus_grid):
            recs_t = []
            for c in range(4):
                tp = len(df_matched_val[(df_matched_val['class_id'] == c) & (df_matched_val['is_tp'] == True) & (df_matched_val['conf'] >= tau)])
                recs_t.append(tp / val_gt_counts[c])
            if min(recs_t) >= r_floor_exact:
                best_tau_exact = float(tau)
                break
                
        # 3. Test exact recall at exact tau*
        test_rec_tau_star = []
        for c in range(4):
            tp = len(df_matched_test[(df_matched_test['class_id'] == c) & (df_matched_test['is_tp'] == True) & (df_matched_test['conf'] >= best_tau_exact)])
            test_rec_tau_star.append(tp / test_gt_counts[c])
        test_min_rec_exact = min(test_rec_tau_star)
        worst_c_test = CLASS_NAMES[test_rec_tau_star.index(test_min_rec_exact)]
        margin_exact = test_min_rec_exact - r_floor_exact
        passed_exact = margin_exact >= 0.0
        
        # Compare with JSON values
        json_rec = calib_data[model_name]['0%'][str(seed)]
        json_r_base = json_rec['val_at_tau_25']['min_recall']
        json_r_floor = json_rec['r_floor']
        json_tau_star = json_rec['tau_star']
        json_test_min = json_rec['test_at_tau_star']['min_recall']
        json_margin = json_rec['test_margin']
        json_pass = json_rec['passed_safety']
        
        calib_recomp.append({
            'model': model_name,
            'seed': seed,
            'json_r_base': json_r_base,
            'exact_r_base': r_base_exact,
            'diff_r_base': r_base_exact - json_r_base,
            'json_r_floor': json_r_floor,
            'exact_r_floor': r_floor_exact,
            'json_tau_star': json_tau_star,
            'exact_tau_star': best_tau_exact,
            'diff_tau_star': best_tau_exact - json_tau_star if (best_tau_exact is not None and json_tau_star is not None) else None,
            'json_test_min_rec': json_test_min,
            'exact_test_min_rec': test_min_rec_exact,
            'json_margin': json_margin,
            'exact_margin': margin_exact,
            'json_pass': json_pass,
            'exact_pass': passed_exact
        })
        
    df_calib = pd.DataFrame(calib_recomp)
    out_calib_csv = REPO_ROOT / 'results' / 'phase0' / 'baseline_recalibration_audit.csv'
    df_calib.to_csv(out_calib_csv, index=False)
    print(f"\nBaseline recalibration saved to {out_calib_csv}")
    print(df_calib.to_string())

if __name__ == '__main__':
    run_r2_audit()
