import json
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_RGB = REPO_ROOT / 'data' / 'processed' / 'RGB'

CLASS_NAMES = ['yawning', 'hand_over_mouth', 'drinking', 'phone_use']

def analyze_per_subject_and_seed1():
    print("="*75)
    print("PHASE 0.3: PER-SUBJECT AUDIT & YOLO11N SEED 1 YAWNING COLLAPSE")
    print("="*75)
    
    df_gt = pd.read_parquet(REPO_ROOT / 'results' / 'phase0' / 'ground_truth.parquet')
    df_events = pd.read_parquet(REPO_ROOT / 'results' / 'phase0' / 'events.parquet')
    df_gt_test = df_gt[df_gt['split'] == 'test']
    
    # Check YOLO11n baseline seed 1
    ckpt_s1 = "models/yolo11n/baseline/seed_1/weights/best.pt"
    m_s1 = YOLO(ckpt_s1)
    
    out_dir = REPO_ROOT / 'runs' / 'detect' / 'phase0_audit' / 'yolo11n_base_s1'
    res = m_s1.val(
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
        preds = json.load(f)
        
    print(f"Loaded {len(preds)} predictions for YOLO11n seed 1")
    
    # Match against test GT
    gt_by_img = {}
    for img_id, grp in df_gt_test.groupby('image_id'):
        gt_by_img[img_id] = grp[['class_id', 'x1', 'y1', 'x2', 'y2', 'class_name']].to_dict('records')
        
    preds_by_img = {}
    for p in preds:
        preds_by_img.setdefault(p['image_id'], []).append(p)
        
    # We will compute max conf per GT box and per GT event
    box_matches = []
    
    for _, gt_row in df_gt_test.iterrows():
        img_id = gt_row['image_id']
        c_cat = gt_row['class_id']
        g_box = [gt_row['x1'], gt_row['y1'], gt_row['x2'], gt_row['y2']]
        
        img_preds = preds_by_img.get(img_id, [])
        # Find best matching prediction for this GT box
        best_conf = 0.0
        best_iou = 0.0
        for p in img_preds:
            p_cat = p['category_id'] - 1
            if p_cat != c_cat:
                continue
            p_box = [p['bbox'][0], p['bbox'][1], p['bbox'][0] + p['bbox'][2], p['bbox'][1] + p['bbox'][3]]
            
            # IoU
            xA, yA = max(p_box[0], g_box[0]), max(p_box[1], g_box[1])
            xB, yB = min(p_box[2], g_box[2]), min(p_box[3], g_box[3])
            inter = max(0, xB - xA) * max(0, yB - yA)
            area1 = (p_box[2] - p_box[0]) * (p_box[3] - p_box[1])
            area2 = (g_box[2] - g_box[0]) * (g_box[3] - g_box[1])
            union = area1 + area2 - inter
            iou = inter / union if union > 0 else 0.0
            
            if iou >= 0.50 and p['score'] > best_conf:
                best_conf = p['score']
                best_iou = iou
                
        box_matches.append({
            'subject': gt_row['subject'],
            'video': gt_row['video'],
            'image_id': img_id,
            'frame_num': gt_row['frame_num'],
            'class_name': gt_row['class_name'],
            'class_id': c_cat,
            'best_conf': best_conf,
            'best_iou': best_iou
        })
        
    df_box_matches = pd.DataFrame(box_matches)
    
    # Analyze YAWNING specifically on test split
    yawns = df_box_matches[df_box_matches['class_name'] == 'yawning']
    print(f"\n--- YAWNING RECALL ON TEST SET (YOLO11n Seed 1) ---")
    print(f"Total Yawn GT frames: {len(yawns)}")
    
    tau_star = 0.7555
    print(f"Total detected @ tau=0.25: {len(yawns[yawns['best_conf'] >= 0.25])} / {len(yawns)} ({len(yawns[yawns['best_conf'] >= 0.25])/len(yawns):.4f})")
    print(f"Total detected @ tau={tau_star:.4f}: {len(yawns[yawns['best_conf'] >= tau_star])} / {len(yawns)} ({len(yawns[yawns['best_conf'] >= tau_star])/len(yawns):.4f})")
    
    print("\nPer-Subject Yawn Breakdown:")
    for sub in sorted(yawns['subject'].unique()):
        sub_yawns = yawns[yawns['subject'] == sub]
        det_25 = len(sub_yawns[sub_yawns['best_conf'] >= 0.25])
        det_star = len(sub_yawns[sub_yawns['best_conf'] >= tau_star])
        print(f"  {sub:12s}: Total {len(sub_yawns):2d} fr | @0.25: {det_25:2d}/{len(sub_yawns):2d} ({det_25/len(sub_yawns):.3f}) | @tau* ({tau_star}): {det_star:2d}/{len(sub_yawns):2d} ({det_star/len(sub_yawns):.3f})")
        
    # Match to events
    test_yawn_events = df_events[(df_events['split'] == 'test') & (df_events['class_name'] == 'yawning')]
    print(f"\n--- YAWN EVENTS IN TEST SPLIT (Total {len(test_yawn_events)} events) ---")
    event_summary = []
    for _, ev in test_yawn_events.iterrows():
        ev_boxes = yawns[(yawns['subject'] == ev['subject']) & (yawns['video'] == ev['video']) & 
                         (yawns['frame_num'] >= ev['start_frame']) & (yawns['frame_num'] <= ev['end_frame'])]
        max_ev_conf = ev_boxes['best_conf'].max() if len(ev_boxes) > 0 else 0.0
        frames_at_25 = len(ev_boxes[ev_boxes['best_conf'] >= 0.25])
        frames_at_star = len(ev_boxes[ev_boxes['best_conf'] >= tau_star])
        
        event_summary.append({
            'event_id': ev['event_id'],
            'subject': ev['subject'],
            'video': ev['video'],
            'frames': f"[{ev['start_frame']}..{ev['end_frame']}] ({ev['num_frames']} fr)",
            'max_conf': max_ev_conf,
            'det_25': frames_at_25 > 0,
            'det_star': frames_at_star > 0,
            'num_frames': ev['num_frames'],
            'frames_at_star': frames_at_star
        })
        print(f"  {ev['event_id']} ({ev['subject']} {ev['video']}): {ev['num_frames']:2d} fr [{ev['start_frame']:4d}..{ev['end_frame']:4d}] | Max Conf = {max_ev_conf:.4f} | Det@0.25: {frames_at_25>0} | Det@tau*: {frames_at_star>0} ({frames_at_star}/{ev['num_frames']} fr)")
        
    pd.DataFrame(event_summary).to_csv(REPO_ROOT / 'results' / 'phase0' / 'yawn_events_seed1_test.csv', index=False)
    
if __name__ == '__main__':
    analyze_per_subject_and_seed1()
