import json
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent
CLASS_NAMES = ['yawning', 'hand_over_mouth', 'drinking', 'phone_use']

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

def run_r5_investigation():
    print("=" * 80)
    print("MANDATE R5: DEEP FORENSIC ACCOUNTING OF YOLO11n SEED 1 YAWNING COLLAPSE")
    print("=" * 80)

    # 1. Load GT and events
    df_gt = pd.read_parquet(REPO_ROOT / 'results' / 'phase0' / 'ground_truth.parquet')
    df_events = pd.read_parquet(REPO_ROOT / 'results' / 'phase0' / 'events.parquet')
    
    gt_test_yawn = df_gt[(df_gt['split'] == 'test') & (df_gt['class_name'] == 'yawning')].copy()
    print(f"Total test yawning GT boxes: {len(gt_test_yawn)}")
    print("Per-subject breakdown of test yawning GT boxes:")
    print(gt_test_yawn['subject'].value_counts())

    # Map each GT box to its event_id
    box_events = []
    for _, g in gt_test_yawn.iterrows():
        # find matching event
        ev = df_events[(df_events['subject'] == g['subject']) & 
                       (df_events['video'] == g['video']) & 
                       (df_events['class_name'] == 'yawning') & 
                       (df_events['start_frame'] <= g['frame_num']) & 
                       (df_events['end_frame'] >= g['frame_num'])]
        ev_id = ev['event_id'].values[0] if len(ev) > 0 else 'no_event'
        box_events.append(ev_id)
    gt_test_yawn['event_id'] = box_events

    # 2. Run inference on test split with seed 1
    ckpt_path = REPO_ROOT / 'models' / 'yolo11n' / 'baseline' / 'seed_1' / 'weights' / 'best.pt'
    m = YOLO(str(ckpt_path))
    out_dir = REPO_ROOT / 'runs' / 'detect' / 'r5_diag' / 'yolo11n_s1_test'
    res = m.val(
        data='configs/dmd_rgb.yaml',
        split='test',
        batch=64,
        device=0,
        workers=0,
        conf=0.001,
        iou=0.7,
        max_det=300,
        save_json=True,
        plots=False,
        verbose=False,
        project=str(out_dir.parent),
        name=out_dir.name,
        exist_ok=True
    )

    with open(out_dir / 'predictions.json') as f:
        preds = json.load(f)

    # 3. For each GT box, find the highest-confidence matching detection (IoU >= 0.50, class=0)
    preds_by_img = {}
    for p in preds:
        if p['category_id'] == 1:  # yawning (1-indexed in COCO json)
            preds_by_img.setdefault(p['image_id'], []).append(p)

    best_confs = []
    best_ious = []
    tp_25_list = []
    tp_tau_star_list = []
    tau_star = 0.7555  # calibrated tau* for yolo11n seed 1

    for _, g in gt_test_yawn.iterrows():
        img_id = g['image_id']
        p_candidates = preds_by_img.get(img_id, [])
        g_box = [g['x1'], g['y1'], g['x2'], g['y2']]
        best_c = 0.0
        best_i = 0.0
        for p in p_candidates:
            p_box = [p['bbox'][0], p['bbox'][1], p['bbox'][0] + p['bbox'][2], p['bbox'][1] + p['bbox'][3]]
            iou = compute_iou(p_box, g_box)
            if iou >= 0.50:
                if p['score'] > best_c:
                    best_c = p['score']
                    best_i = iou
        best_confs.append(best_c)
        best_ious.append(best_i)
        tp_25_list.append(best_c >= 0.25)
        tp_tau_star_list.append(best_c >= tau_star)

    gt_test_yawn['best_conf'] = best_confs
    gt_test_yawn['best_iou'] = best_ious
    gt_test_yawn['tp_at_025'] = tp_25_list
    gt_test_yawn['tp_at_tau_star'] = tp_tau_star_list

    # Identify the lost frames: TP at tau=0.25, but FN at tau*=0.7555
    lost_frames = gt_test_yawn[gt_test_yawn['tp_at_025'] & (~gt_test_yawn['tp_at_tau_star'])].copy()
    lost_frames = lost_frames.sort_values(by='best_conf', ascending=False)
    
    print(f"\nTotal TPs at tau=0.25: {sum(tp_25_list)} / 33 ({sum(tp_25_list)/33:.4f})")
    print(f"Total TPs at tau*=0.7555: {sum(tp_tau_star_list)} / 33 ({sum(tp_tau_star_list)/33:.4f})")
    print(f"Total lost frames: {len(lost_frames)}")
    
    print("\n--- ALL LOST FRAMES AT TAU* = 0.7555 ---")
    cols_to_show = ['subject', 'video', 'frame', 'frame_num', 'event_id', 'best_conf', 'best_iou']
    print(lost_frames[cols_to_show].to_string(index=False))

    lost_csv = REPO_ROOT / 'results' / 'phase0' / 'r5_all_lost_yawn_frames_seed1.csv'
    lost_frames[cols_to_show].to_csv(lost_csv, index=False)
    print(f"\nAll 21 lost yawn frames saved to {lost_csv}")

    # Also analyze event-level collapse
    print("\n--- EVENT-LEVEL LOSS ACCOUNTING ---")
    for ev_id, grp in lost_frames.groupby('event_id'):
        print(f"Event {ev_id} ({grp['subject'].iloc[0]} {grp['video'].iloc[0]}): {len(grp)} frames lost. Confs: {sorted(grp['best_conf'].tolist(), reverse=True)}")

if __name__ == '__main__':
    run_r5_investigation()
