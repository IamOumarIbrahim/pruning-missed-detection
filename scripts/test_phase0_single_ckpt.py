import os
import sys
import json
import time
from pathlib import Path
import pandas as pd
import numpy as np
import shutil

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DATA_RGB = REPO_ROOT / 'data' / 'processed' / 'RGB'

CLASS_NAMES = ['yawning', 'hand_over_mouth', 'drinking', 'phone_use']

# 1. Build Ground Truth table
def build_ground_truth_table():
    records = []
    splits = {
        'train': DATA_RGB / 'yolo' / 'train.txt',
        'val': DATA_RGB / 'yolo' / 'val.txt',
        'test': DATA_RGB / 'yolo' / 'test.txt'
    }
    for split_name, txt_path in splits.items():
        with open(txt_path) as f:
            lines = [l.strip() for l in f if l.strip()]
        for l in lines:
            parts = Path(l).parts
            sub = [p for p in parts if p.startswith('subject_')][0]
            vid = [p for p in parts if p.startswith('video_')][0]
            stem = Path(l).stem
            frame_num = int(stem.split('frame_')[-1])
            
            lbl_file = DATA_RGB / 'labels' / sub / vid / f"{stem}.txt"
            if lbl_file.exists():
                with open(lbl_file) as lf:
                    for line in lf:
                        lp = line.strip().split()
                        if len(lp) >= 5:
                            cls_id = int(lp[0])
                            cx, cy, w, h = float(lp[1]), float(lp[2]), float(lp[3]), float(lp[4])
                            x1 = (cx - w/2) * 640
                            y1 = (cy - h/2) * 640
                            x2 = (cx + w/2) * 640
                            y2 = (cy + h/2) * 640
                            records.append({
                                'split': split_name,
                                'subject': sub,
                                'video': vid,
                                'frame': stem,
                                'frame_num': frame_num,
                                'image_id': stem,
                                'class_id': cls_id,
                                'class_name': CLASS_NAMES[cls_id],
                                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                            })
    df_gt = pd.DataFrame(records)
    out_path = REPO_ROOT / 'results' / 'phase0' / 'ground_truth.parquet'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_gt.to_parquet(out_path, index=False)
    print(f"Ground Truth table built: {len(df_gt)} boxes across {df_gt['image_id'].nunique()} frames.")
    return df_gt


def compute_iou(box1, box2):
    # box: [x1, y1, x2, y2]
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def test_single_checkpoint():
    df_gt = build_ground_truth_table()
    
    from ultralytics import YOLO
    import prune.pruner
    
    ckpt_path = "models/yolo11n/baseline/seed_0/weights/best.pt"
    m = YOLO(ckpt_path)
    
    out_dir = REPO_ROOT / 'runs' / 'detect' / 'phase0_tmp' / 'yolo11n_base_s0'
    res = m.val(
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
    
    pred_json = out_dir / 'predictions.json'
    print(f"Pred JSON exists: {pred_json.exists()}")
    with open(pred_json) as f:
        preds = json.load(f)
    print(f"Total raw predictions: {len(preds)}")
    
    # Match against test GT
    gt_test = df_gt[df_gt['split'] == 'test']
    gt_by_img = {}
    for img_id, grp in gt_test.groupby('image_id'):
        gt_by_img[img_id] = grp[['class_id', 'x1', 'y1', 'x2', 'y2']].to_dict('records')
        
    preds_by_img = {}
    for p in preds:
        img_id = p['image_id']
        preds_by_img.setdefault(img_id, []).append(p)
        
    matched_preds = []
    for img_id, p_list in preds_by_img.items():
        p_list_sorted = sorted(p_list, key=lambda x: x['score'], reverse=True)
        img_gts = gt_by_img.get(img_id, [])
        matched_gt_indices = set()
        
        parts = img_id.split('_')
        sub = f"{parts[0]}_{parts[1]}"
        vid = f"{parts[2]}_{parts[3]}"
        frame = f"{parts[4]}_{parts[5]}"
        frame_num = int(parts[5])
        
        for p in p_list_sorted:
            p_cat = p['category_id'] - 1  # COCO export uses 1-based category IDs
            p_box = [p['bbox'][0], p['bbox'][1], p['bbox'][0] + p['bbox'][2], p['bbox'][1] + p['bbox'][3]]
            
            best_iou = 0.0
            best_gt_idx = -1
            for g_idx, g in enumerate(img_gts):
                if g_idx in matched_gt_indices:
                    continue
                if g['class_id'] != p_cat:
                    continue
                g_box = [g['x1'], g['y1'], g['x2'], g['y2']]
                iou = compute_iou(p_box, g_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = g_idx
                    
            is_tp = False
            if best_iou >= 0.50 and best_gt_idx >= 0:
                is_tp = True
                matched_gt_indices.add(best_gt_idx)
                
            matched_preds.append({
                'checkpoint_id': 'yolo11n_baseline_seed_0',
                'split': 'test',
                'subject': sub,
                'video': vid,
                'frame': img_id,
                'frame_num': frame_num,
                'class_id': p_cat,
                'class_name': CLASS_NAMES[p_cat],
                'conf': p['score'],
                'x1': p_box[0], 'y1': p_box[1], 'x2': p_box[2], 'y2': p_box[3],
                'matched': is_tp,
                'best_iou': best_iou
            })
            
    df_matched = pd.DataFrame(matched_preds)
    out_parquet = REPO_ROOT / 'results' / 'phase0' / 'test_matched_sample.parquet'
    df_matched.to_parquet(out_parquet, index=False)
    print(f"Matched {len(df_matched)} detections. TPs: {df_matched['matched'].sum()}")
    
    # Calculate recall at tau=0.25 from these cached matched detections
    for c_id, c_name in enumerate(CLASS_NAMES):
        c_gt = len(gt_test[gt_test['class_id'] == c_id])
        c_tp_25 = len(df_matched[(df_matched['class_id'] == c_id) & (df_matched['matched'] == True) & (df_matched['conf'] >= 0.25)])
        rec_25 = c_tp_25 / c_gt if c_gt > 0 else 0.0
        print(f"  {c_name:15s} (GT={c_gt:3d}): TPs={c_tp_25:3d}, Recall @ tau=0.25 = {rec_25:.4f}")

if __name__ == '__main__':
    test_single_checkpoint()
