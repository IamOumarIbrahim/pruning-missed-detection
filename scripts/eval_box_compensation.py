"""Evaluate bounding box density, false positives, and multi-threshold behavior across all 36 models.

Tests Hypothesis 1: Does the model compensate for reduced recall by inflating
bounding box predictions on test images (3,213 images: 614 positive, 2,599 negative)?
"""

import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import json
import time
import numpy as np
from ultralytics import YOLO

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

MODELS = ['yolo11n', 'yolo26n']
RATIO_DIRS = [
    ('0%', 'baseline'),
    ('10%', 'pruning_fp32/10pct'),
    ('20%', 'pruning_fp32/20pct'),
    ('30%', 'pruning_fp32/30pct'),
    ('40%', 'pruning_fp32/40pct'),
    ('50%', 'pruning_fp32/50pct'),
]
SEEDS = [0, 1, 2]
DATA_YAML = 'configs/dmd_rgb.yaml'
CLASSES = ['yawning', 'hand_over_mouth', 'drinking', 'phone_use']
GT_COUNTS = [33, 28, 56, 497]  # Total 614 positive annotations on test set
TOTAL_IMAGES = 3213
THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]


def find_weights(model_name, ratio_dir, seed):
    candidates = [
        REPO_ROOT / 'models' / model_name / ratio_dir / f'seed_{seed}' / 'weights' / 'best.pt',
        REPO_ROOT / 'runs' / 'detect' / 'models' / model_name / ratio_dir / f'seed_{seed}' / 'weights' / 'best.pt',
        REPO_ROOT / 'models' / model_name / 'baseline' / f'seed_{seed}' / 'weights' / 'best.pt',
    ]
    for c in candidates:
        if c.exists():
            return str(c.resolve())
    return None


def run_analysis():
    print(f"Running bounding box and threshold compensation analysis across {len(MODELS)*len(RATIO_DIRS)*len(SEEDS)} checkpoints...")
    all_results = {}

    for model_name in MODELS:
        all_results[model_name] = {}
        for r_lbl, r_dir in RATIO_DIRS:
            all_results[model_name][r_lbl] = {
                'seeds': [],
                'aggregated': {}
            }
            seed_data = []

            for seed in SEEDS:
                w_path = find_weights(model_name, r_dir, seed)
                if not w_path:
                    continue

                m = YOLO(w_path)
                val_res = m.val(
                    data=DATA_YAML,
                    split='test',
                    batch=32,
                    device=0,
                    verbose=False,
                    plots=False,
                    project='runs/detect/box_diag',
                    name='tmp',
                    exist_ok=True
                )

                r_curve, p_curve = None, None
                for cr in val_res.box.curves_results:
                    if cr[3] == 'Recall' and cr[2] == 'Confidence':
                        r_curve = (cr[0], cr[1])
                    elif cr[3] == 'Precision' and cr[2] == 'Confidence':
                        p_curve = (cr[0], cr[1])

                seed_thresh_metrics = {}
                for tau in THRESHOLDS:
                    idx = int(np.argmin(np.abs(r_curve[0] - tau)))
                    rec_per_cls = [float(r_curve[1][c, idx]) for c in range(4)]
                    prec_per_cls = [float(p_curve[1][c, idx]) for c in range(4)]

                    tp_per_cls = [rec_per_cls[c] * GT_COUNTS[c] for c in range(4)]
                    pred_per_cls = [tp_per_cls[c] / prec_per_cls[c] if prec_per_cls[c] > 0 else 0.0 for c in range(4)]

                    total_tp = sum(tp_per_cls)
                    total_pred = sum(pred_per_cls)
                    total_fp = total_pred - total_tp
                    min_rec = min(rec_per_cls)
                    macro_prec = float(np.mean(prec_per_cls))
                    boxes_per_img = total_pred / TOTAL_IMAGES

                    f1_h = (2 * macro_prec * min_rec / (macro_prec + min_rec)) if (macro_prec + min_rec) > 0 else 0.0

                    seed_thresh_metrics[f"tau_{int(tau*100):02d}"] = {
                        'tau': tau,
                        'min_recall': min_rec,
                        'macro_precision': macro_prec,
                        'f1_harmonic': f1_h,
                        'total_pred': total_pred,
                        'total_tp': total_tp,
                        'total_fp': total_fp,
                        'boxes_per_img': boxes_per_img,
                    }

                seed_record = {
                    'seed': seed,
                    'thresholds': seed_thresh_metrics,
                }
                seed_data.append(seed_record)
                all_results[model_name][r_lbl]['seeds'].append(seed_record)

            # Aggregate across seeds
            agg_thresh = {}
            for tau in THRESHOLDS:
                t_key = f"tau_{int(tau*100):02d}"
                min_recs = [s['thresholds'][t_key]['min_recall'] for s in seed_data]
                precs = [s['thresholds'][t_key]['macro_precision'] for s in seed_data]
                f1s = [s['thresholds'][t_key]['f1_harmonic'] for s in seed_data]
                preds = [s['thresholds'][t_key]['total_pred'] for s in seed_data]
                tps = [s['thresholds'][t_key]['total_tp'] for s in seed_data]
                fps = [s['thresholds'][t_key]['total_fp'] for s in seed_data]
                bpis = [s['thresholds'][t_key]['boxes_per_img'] for s in seed_data]

                def s_stat(arr):
                    m_val = float(np.mean(arr))
                    std_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
                    return {'mean': m_val, 'std': std_val}

                agg_thresh[t_key] = {
                    'tau': tau,
                    'min_recall': s_stat(min_recs),
                    'macro_precision': s_stat(precs),
                    'f1_harmonic': s_stat(f1s),
                    'total_pred': s_stat(preds),
                    'total_tp': s_stat(tps),
                    'total_fp': s_stat(fps),
                    'boxes_per_img': s_stat(bpis),
                }

            all_results[model_name][r_lbl]['aggregated'] = agg_thresh
            print(f"[{model_name} {r_lbl}] Complete: at tau=0.10 FP={agg_thresh['tau_10']['total_fp']['mean']:.1f}, MinRec={agg_thresh['tau_10']['min_recall']['mean']:.4f} | at tau=0.25 FP={agg_thresh['tau_25']['total_fp']['mean']:.1f}, MinRec={agg_thresh['tau_25']['min_recall']['mean']:.4f}")

    out_file = REPO_ROOT / 'results' / 'box_compensation_data.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)
    print(f"Analysis saved to {out_file}")
    return all_results


if __name__ == '__main__':
    run_analysis()
