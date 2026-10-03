import os
import sys
import json
import hashlib
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import prune.pruner
CACHE_DIR = REPO_ROOT / 'results' / 'phase0' / 'cache'
MANIFEST_PATH = REPO_ROOT / 'results' / 'phase0' / 'cache_manifest.csv'

CLASS_NAMES = ['yawning', 'hand_over_mouth', 'drinking', 'phone_use']

# Load GT
DF_GT = pd.read_parquet(REPO_ROOT / 'results' / 'phase0' / 'ground_truth.parquet')

def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()

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

def match_preds_to_gt(preds, split_name, ckpt_meta):
    split_gt = DF_GT[DF_GT['split'] == split_name]
    gt_by_img = {}
    for img_id, grp in split_gt.groupby('image_id'):
        gt_by_img[img_id] = grp[['class_id', 'x1', 'y1', 'x2', 'y2']].to_dict('records')

    preds_by_img = {}
    for p in preds:
        preds_by_img.setdefault(p['image_id'], []).append(p)

    matched_records = []
    # Process each image that appeared in predictions or GT
    all_imgs = set(preds_by_img.keys()).union(set(gt_by_img.keys()))

    for img_id in all_imgs:
        p_list = preds_by_img.get(img_id, [])
        p_list_sorted = sorted(p_list, key=lambda x: x['score'], reverse=True)
        img_gts = gt_by_img.get(img_id, [])
        matched_gt_indices = set()

        parts = img_id.split('_')
        sub = f"{parts[0]}_{parts[1]}"
        vid = f"{parts[2]}_{parts[3]}"
        frame_num = int(parts[5])

        for p in p_list_sorted:
            p_cat = p['category_id'] - 1
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

            matched_records.append({
                'checkpoint_id': ckpt_meta['checkpoint_id'],
                'model': ckpt_meta['model'],
                'prune_ratio': ckpt_meta['prune_ratio'],
                'seed': ckpt_meta['seed'],
                'ckpt_type': ckpt_meta['ckpt_type'],
                'split': split_name,
                'tag': ckpt_meta['tag'],
                'subject': sub,
                'video': vid,
                'frame': img_id,
                'frame_num': frame_num,
                'class_id': p_cat,
                'class_name': CLASS_NAMES[p_cat],
                'conf': p['score'],
                'x1': p_box[0], 'y1': p_box[1], 'x2': p_box[2], 'y2': p_box[3],
                'matched': is_tp,
                'best_iou': best_iou,
                'gt_match_idx': best_gt_idx
            })

    return pd.DataFrame(matched_records)

def get_all_target_checkpoints():
    checkpoints = []
    models = ['yolo11n', 'yolo26n']
    ratios = [('baseline', '0%'), ('10pct', '10%'), ('20pct', '20%'), ('30pct', '30%'), ('40pct', '40%'), ('50pct', '50%')]
    seeds = [0, 1, 2]

    # 36 sweep checkpoints
    for m in models:
        for r_dir, r_str in ratios:
            for s in seeds:
                candidates = [
                    Path(f'models/{m}/{r_dir}/seed_{s}/weights/best.pt'),
                    Path(f'runs/detect/models/{m}/pruning_fp32/{r_dir}/seed_{s}/weights/best.pt'),
                    Path(f'runs/detect/models/{m}/{r_dir}/seed_{s}/weights/best.pt'),
                    Path(f'models/{m}/baseline/seed_{s}/weights/best.pt') if r_dir == 'baseline' else None
                ]
                for c in candidates:
                    if c and c.exists():
                        checkpoints.append({
                            'checkpoint_id': f"{m}_{r_dir}_s{s}_best",
                            'model': m,
                            'prune_ratio': r_str,
                            'seed': s,
                            'ckpt_type': 'best',
                            'tag': 'eval',
                            'path': c.resolve()
                        })
                        break

    # Baseline seeds 3 and 4
    extras = [
        ('yolo11n', '0%', 3, Path('models/yolo11n/baseline/seed_3/weights/best.pt')),
        ('yolo11n', '0%', 4, Path('models/yolo11n/baseline/seed_4/weights/best.pt')),
        ('yolo26n', '0%', 3, Path('models/yolo26n/baseline/seed_3/weights/best.pt')),
    ]
    for m, r_str, s, p in extras:
        if p.exists():
            checkpoints.append({
                'checkpoint_id': f"{m}_baseline_s{s}_best",
                'model': m,
                'prune_ratio': r_str,
                'seed': s,
                'ckpt_type': 'best',
                'tag': 'eval',
                'path': p.resolve()
            })

    # Last checkpoints for R9 sensitivity
    for m in models:
        for s in [0, 1, 2]:
            p = Path(f'models/{m}/baseline/seed_{s}/weights/last.pt')
            if p.exists():
                checkpoints.append({
                    'checkpoint_id': f"{m}_baseline_s{s}_last",
                    'model': m,
                    'prune_ratio': '0%',
                    'seed': s,
                    'ckpt_type': 'last',
                    'tag': 'eval',
                    'path': p.resolve()
                })

    return checkpoints

def build_cache():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    checkpoints = get_all_target_checkpoints()
    print(f"Total target checkpoints identified: {len(checkpoints)}")

    manifest_entries = []
    if MANIFEST_PATH.exists():
        df_old_manifest = pd.read_csv(MANIFEST_PATH)
        manifest_entries = df_old_manifest.to_dict('records')
    cached_ids = set([f"{e['checkpoint_id']}_{e['split']}" for e in manifest_entries])

    for i, ckpt in enumerate(checkpoints):
        ckpt_id = ckpt['checkpoint_id']
        model_path = ckpt['path']
        m = YOLO(str(model_path))

        # Splits to evaluate
        splits_to_run = ['val', 'test']
        # If baseline, also evaluate on train split with tag 'train-fit'
        if ckpt['prune_ratio'] == '0%' and ckpt['ckpt_type'] == 'best':
            splits_to_run.append('train')

        for split in splits_to_run:
            cache_key = f"{ckpt_id}_{split}"
            out_parquet = CACHE_DIR / f"{cache_key}.parquet"

            if cache_key in cached_ids and out_parquet.exists():
                print(f"[{i+1}/{len(checkpoints)}] Already cached: {cache_key}")
                continue

            print(f"[{i+1}/{len(checkpoints)}] Caching: {cache_key} ({split} split)...")
            run_tag = 'train-fit' if split == 'train' else 'eval'
            ckpt_meta = dict(ckpt)
            ckpt_meta['tag'] = run_tag

            out_run = REPO_ROOT / 'runs' / 'detect' / 'cache_tmp' / cache_key
            res = m.val(
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
                project=str(out_run.parent),
                name=out_run.name,
                exist_ok=True
            )

            with open(out_run / 'predictions.json') as f:
                preds = json.load(f)

            df_matched = match_preds_to_gt(preds, split, ckpt_meta)
            df_matched.to_parquet(out_parquet, index=False)
            sha256_hash = compute_sha256(out_parquet)

            entry = {
                'checkpoint_id': ckpt_id,
                'model': ckpt['model'],
                'prune_ratio': ckpt['prune_ratio'],
                'seed': ckpt['seed'],
                'ckpt_type': ckpt['ckpt_type'],
                'split': split,
                'tag': run_tag,
                'parquet_path': str(out_parquet.relative_to(REPO_ROOT)),
                'sha256': sha256_hash,
                'num_detections': len(df_matched),
                'num_tp': int(df_matched['matched'].sum()),
                'created_at': datetime.now().isoformat()
            }
            manifest_entries.append(entry)
            cached_ids.add(cache_key)

            # Persist manifest immediately
            pd.DataFrame(manifest_entries).to_csv(MANIFEST_PATH, index=False)
            print(f"  -> Saved {out_parquet.name} ({len(df_matched)} dets, {entry['num_tp']} TPs, SHA-256: {sha256_hash[:12]}...)")

    print("\nFull detection cache build complete!")
    print(f"Total cached datasets: {len(manifest_entries)}")
    print(f"Manifest written to: {MANIFEST_PATH}")

if __name__ == '__main__':
    build_cache()
