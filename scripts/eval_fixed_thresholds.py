"""Fixed-threshold evaluation at tau=0.25 and tau=0.50.

Evaluates checkpoints on the test set and records per-class and macro recall,
precision, and min safety recall at fixed confidence thresholds (tau=0.25, 0.50).
Exposes the raw recall degradation curve without dynamic threshold compensation.
"""

import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import prune.pruner  # Ensure C3k2_v2 and pruner classes are in namespace
import argparse
import glob
import json
import os
import time
import numpy as np
from ultralytics import YOLO

CLASSES = ['yawning', 'hand_over_mouth', 'drinking', 'phone_use']
DATA_YAML = 'configs/dmd_rgb.yaml'


def extract_fixed_metrics(val_results, thresholds=(0.25, 0.50)):
    r_curve = None
    p_curve = None
    for cr in val_results.box.curves_results:
        x, y, xl, yl = cr
        if yl == 'Recall' and xl == 'Confidence':
            r_curve = (x, y)
        elif yl == 'Precision' and xl == 'Confidence':
            p_curve = (x, y)

    if r_curve is None or p_curve is None:
        raise ValueError("curves_results does not contain Recall or Precision curves.")

    metrics_out = {}
    for tau in thresholds:
        tau_key = f"tau_{int(tau*100):02d}"
        idx = int(np.argmin(np.abs(r_curve[0] - tau)))
        actual_tau = float(r_curve[0][idx])

        rec_per_cls = {CLASSES[i]: round(float(r_curve[1][i, idx]), 4) for i in range(len(CLASSES))}
        prec_per_cls = {CLASSES[i]: round(float(p_curve[1][i, idx]), 4) for i in range(len(CLASSES))}

        macro_r = round(float(np.mean(list(rec_per_cls.values()))), 4)
        min_r = round(float(np.min(list(rec_per_cls.values()))), 4)
        macro_p = round(float(np.mean(list(prec_per_cls.values()))), 4)

        metrics_out[tau_key] = {
            'target_tau': tau,
            'actual_tau': actual_tau,
            'macro_recall': macro_r,
            'min_recall': min_r,
            'macro_precision': macro_p,
            'per_class_recall': rec_per_cls,
            'per_class_precision': prec_per_cls,
        }

    return metrics_out


def evaluate_checkpoint(model_path, data_yaml=DATA_YAML, batch=32, device=0):
    model_path = Path(model_path)
    is_onnx = model_path.suffix == '.onnx'

    if is_onnx:
        m = YOLO(str(model_path), task='detect')
        res = m.val(
            data=data_yaml,
            split='test',
            batch=1,
            workers=0,
            device='cpu',
            verbose=False,
            project='runs/detect/fixed_eval',
            name='tmp',
            exist_ok=True,
            plots=False,
        )
    else:
        m = YOLO(str(model_path))
        res = m.val(
            data=data_yaml,
            split='test',
            batch=batch,
            workers=0,
            device=device,
            verbose=False,
            project='runs/detect/fixed_eval',
            name='tmp',
            exist_ok=True,
            plots=False,
        )

    return extract_fixed_metrics(res, thresholds=(0.25, 0.50))


def main():
    parser = argparse.ArgumentParser(description='Evaluate fixed thresholds tau=0.25 and tau=0.50')
    parser.add_argument('--filter', type=str, default='all', choices=['all', 'pt', 'onnx'],
                        help='Filter model types to evaluate (pt, onnx, or all)')
    parser.add_argument('--overwrite', action='store_true', help='Recompute even if fixed_eval exists')
    parser.add_argument('--batch', type=int, default=32, help='Batch size for PyTorch models')
    parser.add_argument('--device', default=0, help='CUDA device or cpu')
    args = parser.parse_args()

    files = sorted(glob.glob('results/**/metrics.json', recursive=True))
    print(f"Total metrics files: {len(files)}")

    eval_queue = []
    for f in files:
        with open(f) as fp:
            d = json.load(fp)

        if not args.overwrite and 'fixed_eval' in d and 'tau_25' in d['fixed_eval'] and 'tau_50' in d['fixed_eval']:
            continue

        model_path = d.get('engine_path') or d.get('weights_path')
        if not model_path or not Path(model_path).exists():
            candidate_paths = [
                Path(f).parent / 'weights' / 'best.pt',
                Path(str(model_path).replace('.pt', '.onnx')) if model_path else None,
                Path(str(model_path).replace('.engine', '.onnx')) if model_path else None,
            ]
            for cand in candidate_paths:
                if cand and cand.exists():
                    model_path = str(cand)
                    break

        if not model_path or not Path(model_path).exists():
            print(f"Skipping {f}: model path not found ({model_path})")
            continue

        is_onnx = Path(model_path).suffix == '.onnx'
        if args.filter == 'pt' and is_onnx:
            continue
        if args.filter == 'onnx' and not is_onnx:
            continue

        eval_queue.append((f, model_path, is_onnx))

    print(f"Queued for evaluation: {len(eval_queue)} models (filter={args.filter})")

    t_start = time.time()
    for idx, (metrics_file, model_path, is_onnx) in enumerate(eval_queue, 1):
        t0 = time.time()
        with open(metrics_file) as fp:
            d = json.load(fp)

        m_name = d.get('model', '')
        p_label = d.get('pruning_label', '')
        quant = d.get('quantization', '')
        seed = d.get('seed', '')
        tag = f"{m_name} {p_label} {quant} seed {seed} ({'ONNX' if is_onnx else 'PT'})"

        print(f"[{idx}/{len(eval_queue)}] Evaluating {tag}...")
        try:
            fixed_metrics = evaluate_checkpoint(
                model_path,
                batch=args.batch,
                device=args.device
            )

            d['fixed_eval'] = fixed_metrics
            d['recall_tau_025'] = fixed_metrics['tau_25']['macro_recall']
            d['min_recall_tau_025'] = fixed_metrics['tau_25']['min_recall']
            d['precision_tau_025'] = fixed_metrics['tau_25']['macro_precision']
            d['recall_tau_050'] = fixed_metrics['tau_50']['macro_recall']
            d['min_recall_tau_050'] = fixed_metrics['tau_50']['min_recall']
            d['precision_tau_050'] = fixed_metrics['tau_50']['macro_precision']

            with open(metrics_file, 'w') as fp:
                json.dump(d, fp, indent=2)

            dt = time.time() - t0
            r25 = fixed_metrics['tau_25']['macro_recall']
            min_r25 = fixed_metrics['tau_25']['min_recall']
            r50 = fixed_metrics['tau_50']['macro_recall']
            min_r50 = fixed_metrics['tau_50']['min_recall']
            print(f"    Done in {dt:.1f}s | tau=0.25: Rec={r25:.4f} (min={min_r25:.4f}) | tau=0.50: Rec={r50:.4f} (min={min_r50:.4f})")

        except Exception as e:
            print(f"    ERROR evaluating {model_path}: {e}")

    total_time = time.time() - t_start
    print(f"\nFinished evaluation in {total_time/60:.1f} minutes.")


if __name__ == '__main__':
    main()
