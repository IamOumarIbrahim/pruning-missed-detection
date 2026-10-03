"""Route A Stage 1: Sequential Pruning Screening (60% to 90%, Seed 0).

Strictly adheres to PROTOCOL.md:
- Architectures: YOLO11n, YOLO26n
- Ratios: 60%, 70%, 80%, 90%
- Seed: 0 (sequential execution)
- Stopping rule: Loss divergence, NaN, or mAP50 drop > 30 pp relative to baseline.
- Flagging criteria:
    1. Concealment: Delta mAP50 >= -3.0 pp AND Delta Worst-Class Recall <= -7.5 pp
    2. Knee: Worst-Class Recall(r) - Worst-Class Recall(r - 10%) <= -10.0 pp
- Full dose-response reporting commitment regardless of hypothesis outcome.
"""

import os
import sys
import json
import time
import shutil
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from ultralytics import YOLO
import prune.pruner
from prune.pruner import prune_model, PrunedDetectionTrainer, get_model_info

DATA_YAML = str(REPO_ROOT / 'configs' / 'dmd_rgb.yaml')
STATUS_JSON = REPO_ROOT / 'results' / 'route_a_status.json'
RESULTS_CSV = REPO_ROOT / 'results' / 'route_a_stage1_results.csv'
LOG_FILE = REPO_ROOT / 'results' / 'route_a_stage1.log'

STAGE1_RATIOS = [0.60, 0.70, 0.80, 0.90]
MODELS = ['yolo11n', 'yolo26n']
CLASS_NAMES = ['yawning', 'hand_over_mouth', 'drinking', 'phone_use']

DF_GT = pd.read_parquet(REPO_ROOT / 'results' / 'phase0' / 'ground_truth.parquet')

def log(msg):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    formatted = f"[{ts}] {msg}"
    print(formatted, flush=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(formatted + '\n')

def update_status(payload):
    STATUS_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload['last_updated'] = datetime.now().isoformat()
    with open(STATUS_JSON, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)

def compute_ap(recalls, precisions):
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    x = np.linspace(0, 1, 101)
    y = np.interp(x, mrec, mpre)
    return float(np.trapezoid(y, x) if hasattr(np, 'trapezoid') else np.trapz(y, x))

def eval_metrics_from_preds(preds, split_name):
    split_gt = DF_GT[DF_GT['split'] == split_name]
    gt_by_img = {}
    for img_id, grp in split_gt.groupby('image_id'):
        gt_by_img[img_id] = grp[['class_id', 'x1', 'y1', 'x2', 'y2']].to_dict('records')

    preds_by_img = {}
    for p in preds:
        preds_by_img.setdefault(p['image_id'], []).append(p)

    matched_records = []
    all_imgs = set(preds_by_img.keys()).union(set(gt_by_img.keys()))

    for img_id in all_imgs:
        p_list = preds_by_img.get(img_id, [])
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
                xA = max(p_box[0], g['x1'])
                yA = max(p_box[1], g['y1'])
                xB = min(p_box[2], g['x2'])
                yB = min(p_box[3], g['y2'])
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
                'class_id': p_cat, 'conf': p['score'], 'matched': is_tp
            })

    df_m = pd.DataFrame(matched_records)
    ap50_list = []
    rec_p80_list = []
    
    for c_id in range(4):
        n_gt = len(split_gt[split_gt['class_id'] == c_id])
        if n_gt == 0:
            continue
        c_dets = df_m[df_m['class_id'] == c_id].sort_values(by='conf', ascending=False)
        if len(c_dets) == 0:
            ap50_list.append(0.0)
            rec_p80_list.append(0.0)
            continue
        tps = c_dets['matched'].values.astype(int)
        fps = (1 - tps)
        tp_cum = np.cumsum(tps)
        fp_cum = np.cumsum(fps)
        prec = tp_cum / (tp_cum + fp_cum)
        rec = tp_cum / n_gt
        ap50_list.append(compute_ap(rec, prec))
        valid_p80 = rec[prec >= 0.80]
        rec_p80_list.append(float(np.max(valid_p80)) if len(valid_p80) > 0 else 0.0)

    return {
        'mAP50': float(np.mean(ap50_list)),
        'min_class_AP50': float(np.min(ap50_list)),
        'worst_class_rec_p80': float(np.min(rec_p80_list)),
        'per_class_AP50': ap50_list,
        'per_class_rec_p80': rec_p80_list
    }

def run_stage1_screening():
    log("=" * 80)
    log("ROUTE A STAGE 1: SEQUENTIAL SCREENING (60% to 90%, Seed 0)")
    log("=" * 80)

    # Reference metrics for Seed 0 from Phase 0 R7 audit table
    df_r7 = pd.read_csv(REPO_ROOT / 'results' / 'phase0' / 'r7_matched_operating_points.csv')

    stage1_records = []
    if RESULTS_CSV.exists():
        stage1_records = pd.read_csv(RESULTS_CSV).to_dict('records')

    completed_keys = set([f"{r['model']}_{int(r['prune_ratio']*100)}pct" for r in stage1_records])

    for model_name in MODELS:
        log(f"\n=======================================================")
        log(f"STAGE 1 SCREENING FOR ARCHITECTURE: {model_name}")
        log(f"=======================================================")

        r7_base = df_r7[(df_r7['model'] == model_name) & (df_r7['prune_ratio'] == '0%') & (df_r7['seed'] == 0)].iloc[0]
        base_map50 = float(r7_base['test_mAP50'])
        base_min_ap50 = float(r7_base['test_min_AP50'])
        base_worst_rec_p80 = float(r7_base['test_worst_rec_p80'])
        log(f"Baseline Seed 0 reference: mAP50 = {base_map50:.4f}, min-class AP50 = {base_min_ap50:.4f}, worst Rec@P80 = {base_worst_rec_p80:.4f}")

        # Baseline weights check
        base_weights = REPO_ROOT / 'models' / model_name / 'baseline' / 'seed_0' / 'weights' / 'best.pt'
        if not base_weights.exists():
            log(f"ERROR: Baseline weights not found at {base_weights}")
            continue

        # Starting reference for knee slope check (starts at 50% ratio if available, otherwise baseline)
        r7_50 = df_r7[(df_r7['model'] == model_name) & (df_r7['prune_ratio'] == '50%') & (df_r7['seed'] == 0)]
        if len(r7_50) > 0:
            prev_ratio_worst_recall = float(r7_50.iloc[0]['test_worst_rec_p80'])
            log(f"Starting knee reference from 50% pruning: worst Rec@P80 = {prev_ratio_worst_recall:.4f}")
        else:
            prev_ratio_worst_recall = base_worst_rec_p80
            log(f"Starting knee reference from 0% baseline: worst Rec@P80 = {prev_ratio_worst_recall:.4f}")

        for ratio in STAGE1_RATIOS:
            ratio_pct = int(ratio * 100)
            key = f"{model_name}_{ratio_pct}pct"

            if key in completed_keys:
                log(f"Run {key} already completed in results CSV. Skipping.")
                # Update previous ratio recall for continuity
                existing_match = [r for r in stage1_records if r['model'] == model_name and int(r['prune_ratio']*100) == ratio_pct]
                if existing_match:
                    prev_ratio_worst_recall = existing_match[0]['worst_class_rec_p80']
                continue

            target_dir = REPO_ROOT / 'models' / model_name / 'pruning_fp32' / f'{ratio_pct}pct' / 'seed_0'
            target_pt = target_dir / 'weights' / 'best.pt'

            log(f"\n--- Processing {model_name} at {ratio_pct}% Pruning (Seed 0) ---")
            update_status({
                'status': 'running',
                'current_model': model_name,
                'current_ratio': ratio,
                'stage': 'Stage 1 Screen'
            })

            # 1. Prune and Fine-tune if best.pt does not exist
            if not (target_pt.exists() and target_pt.stat().st_size > 1_000_000):
                log(f"Starting pruning and 100-epoch fine-tuning for {key}...")
                train_start = time.time()
                try:
                    # Prune baseline model checkpoint
                    pruned_dir = REPO_ROOT / 'models' / model_name / 'pruned' / f'{ratio_pct}pct'
                    pruned_path = pruned_dir / 'seed_0_pruned.pt'
                    pruned_path.parent.mkdir(parents=True, exist_ok=True)
                    if not (pruned_path.exists() and pruned_path.stat().st_size > 100_000):
                        log(f"Pruning {model_name} at {ratio:.0%} from {base_weights}...")
                        prune_model(str(base_weights), ratio, str(pruned_path), imgsz=640)
                        info = get_model_info(str(pruned_path))
                        log(f"Pruned checkpoint created: params={info['num_params']:,}, flops={info['flops_g']}G, size={info['size_mb']:.2f}MB")

                    # Fine-tune using Ultralytics with PrunedDetectionTrainer
                    log(f"Fine-tuning {model_name} {ratio_pct}pct (seed 0) for 100 epochs...")
                    ft_model = YOLO(str(pruned_path))
                    train_results = ft_model.train(
                        trainer=PrunedDetectionTrainer,
                        data=DATA_YAML,
                        epochs=100,
                        batch=16,
                        imgsz=640,
                        patience=0,
                        amp=True,
                        seed=0,
                        project=str(REPO_ROOT / 'models' / model_name / 'pruning_fp32' / f'{ratio_pct}pct'),
                        name='seed_0',
                        exist_ok=True,
                        device=0,
                        workers=8,
                        cache=False,
                        verbose=False
                    )

                    # Locate trained best.pt weights
                    candidates = [
                        target_pt,
                        target_dir / 'weights' / 'last.pt',
                    ]
                    if hasattr(ft_model, 'trainer') and ft_model.trainer and hasattr(ft_model.trainer, 'save_dir'):
                        s_dir = Path(ft_model.trainer.save_dir) / 'weights'
                        candidates.extend([s_dir / 'best.pt', s_dir / 'last.pt'])
                    if hasattr(train_results, 'save_dir'):
                        r_dir = Path(train_results.save_dir) / 'weights'
                        candidates.extend([r_dir / 'best.pt', r_dir / 'last.pt'])
                    runs_dir = REPO_ROOT / 'runs' / 'detect' / 'models' / model_name / 'pruning_fp32' / f'{ratio_pct}pct' / 'seed_0' / 'weights'
                    candidates.extend([runs_dir / 'best.pt', runs_dir / 'last.pt'])

                    saved = False
                    for c in candidates:
                        if c and Path(c).exists() and Path(c).stat().st_size > 1_000_000:
                            target_pt.parent.mkdir(parents=True, exist_ok=True)
                            if Path(c).resolve() != target_pt.resolve():
                                shutil.copy2(c, target_pt)
                            saved = True
                            break
                    if not saved:
                        raise RuntimeError(f"Failed to find valid weights for {key}")

                    train_duration = time.time() - train_start
                    log(f"Training completed successfully in {train_duration/3600:.2f} hours. Saved to {target_pt}")
                except Exception as e:
                    log(f"CRITICAL: Training failed for {key}: {str(e)}")
                    log(f"Early Stopping Rule Triggered for {model_name} at {ratio_pct}%. Halting further pruning for this architecture.")
                    break

            # 2. Evaluate on test split
            log(f"Evaluating {key} on test split...")
            m = YOLO(str(target_pt))
            out_eval = REPO_ROOT / 'runs' / 'detect' / 'route_a_stage1' / key
            res = m.val(
                data=DATA_YAML, split='test', batch=64, device=0, workers=0,
                conf=0.001, iou=0.7, max_det=300, save_json=True, plots=False, verbose=False,
                project=str(out_eval.parent), name=out_eval.name, exist_ok=True
            )
            pred_file = out_eval / 'predictions.json'
            if not pred_file.exists() and hasattr(res, 'save_dir'):
                pred_file = Path(res.save_dir) / 'predictions.json'

            with open(pred_file) as f:
                preds = json.load(f)

            metrics = eval_metrics_from_preds(preds, 'test')
            mAP50 = metrics['mAP50']
            min_AP50 = metrics['min_class_AP50']
            worst_rec_p80 = metrics['worst_class_rec_p80']

            delta_mAP50 = (mAP50 - base_map50) * 100
            delta_min_AP50 = (min_AP50 - base_min_ap50) * 100
            delta_worst_rec = (worst_rec_p80 - base_worst_rec_p80) * 100

            # 3. Check Stopping Rule: Delta mAP50 < -30 pp (complete collapse)
            if delta_mAP50 < -30.0:
                log(f"Model collapse detected: mAP50 dropped by {delta_mAP50:.2f} pp (> 30 pp).")
                log(f"Early Stopping Rule Triggered for {model_name}. Higher ratios will not be executed.")
                stopping_triggered = True
            else:
                stopping_triggered = False

            # 4. Check Flagging Criteria
            # Concealment: mAP50 stable (delta >= -3.0 pp) while worst recall dropped <= -7.5 pp
            # Knee: slope drop <= -10.0 pp relative to preceding ratio
            flag_concealment = bool((delta_mAP50 >= -3.0) and (delta_worst_rec <= -7.5))
            step_drop = (worst_rec_p80 - prev_ratio_worst_recall) * 100
            flag_knee = bool(step_drop <= -10.0)
            prev_ratio_worst_recall = worst_rec_p80

            flagged_for_stage2 = bool(flag_concealment or flag_knee)

            record = {
                'model': model_name,
                'prune_ratio': ratio,
                'seed': 0,
                'mAP50': mAP50,
                'min_class_AP50': min_AP50,
                'worst_class_rec_p80': worst_rec_p80,
                'delta_mAP50_pp': delta_mAP50,
                'delta_min_AP50_pp': delta_min_AP50,
                'delta_worst_rec_pp': delta_worst_rec,
                'step_drop_worst_rec_pp': step_drop,
                'flag_concealment': flag_concealment,
                'flag_knee': flag_knee,
                'flagged_for_stage2': flagged_for_stage2,
                'stopping_triggered': stopping_triggered
            }
            stage1_records.append(record)
            pd.DataFrame(stage1_records).to_csv(RESULTS_CSV, index=False)
            log(f"Result for {key}: mAP50={mAP50:.4f} (Δ={delta_mAP50:+.2f} pp), min-AP50={min_AP50:.4f} (Δ={delta_min_AP50:+.2f} pp), worst Rec@P80={worst_rec_p80:.4f} (Δ={delta_worst_rec:+.2f} pp, step={step_drop:+.2f} pp), Flagged={flagged_for_stage2}")

            if stopping_triggered:
                break

    log("\nRoute A Stage 1 Screening Completed!")
    log(f"Results saved to: {RESULTS_CSV}")
    update_status({
        'status': 'idle',
        'stage': 'Stage 1 Completed'
    })

if __name__ == '__main__':
    run_stage1_screening()
