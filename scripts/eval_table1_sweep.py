"""Evaluation script for Table 1: FP32 Pruning Sweep across tau*, tau=0.10, and tau=0.25.

Evaluates all 36 checkpoints (2 models x 6 pruning ratios x 3 seeds) on the test split.
Extracts per-class recall, worst-case (minimum class) recall, precision, and F1 score
at tau*, tau=0.10, and tau=0.25.
Saves raw and aggregated metrics to results/table1_data.json.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import prune.pruner  # Ensure custom modules and classes are available
import json
import os
import time
import numpy as np
from ultralytics import YOLO

CLASSES = ['yawning', 'hand_over_mouth', 'drinking', 'phone_use']
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


def extract_metrics_at_tau(r_curve, p_curve, tau):
    """Extract per-class and worst-case recall and precision at threshold tau."""
    idx = int(np.argmin(np.abs(r_curve[0] - tau)))
    actual_tau = float(r_curve[0][idx])

    rec_per_cls = {CLASSES[i]: float(r_curve[1][i, idx]) for i in range(len(CLASSES))}
    prec_per_cls = {CLASSES[i]: float(p_curve[1][i, idx]) for i in range(len(CLASSES))}

    min_rec = float(min(rec_per_cls.values()))
    worst_class = [k for k, v in rec_per_cls.items() if v == min_rec][0]
    macro_rec = float(np.mean(list(rec_per_cls.values())))
    macro_prec = float(np.mean(list(prec_per_cls.values())))

    f1_harmonic = (2 * macro_prec * min_rec / (macro_prec + min_rec)) if (macro_prec + min_rec) > 0 else 0.0
    f1_arithmetic = (macro_prec + min_rec) / 2.0

    return {
        'target_tau': tau,
        'actual_tau': actual_tau,
        'min_recall': min_rec,
        'worst_class': worst_class,
        'macro_recall': macro_rec,
        'macro_precision': macro_prec,
        'f1_harmonic': f1_harmonic,
        'f1_arithmetic': f1_arithmetic,
        'per_class_recall': rec_per_cls,
        'per_class_precision': prec_per_cls,
    }


def find_model_weights(metrics_dict, model_name, ratio_dir, seed):
    """Locate weights path with robust fallbacks."""
    weights_path = metrics_dict.get('weights_path')
    if weights_path and Path(weights_path).exists():
        return str(Path(weights_path).resolve())

    candidates = [
        REPO_ROOT / 'models' / model_name / ratio_dir / f'seed_{seed}' / 'weights' / 'best.pt',
        REPO_ROOT / 'runs' / 'detect' / 'models' / model_name / ratio_dir / f'seed_{seed}' / 'weights' / 'best.pt',
        REPO_ROOT / 'models' / model_name / 'baseline' / f'seed_{seed}' / 'weights' / 'best.pt',
    ]
    for c in candidates:
        if c.exists():
            return str(c.resolve())
    return None


def run_evaluation():
    results_out = {
        'metadata': {
            'evaluated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'models': MODELS,
            'classes': CLASSES,
            'target_thresholds': ['tau_star', 0.10, 0.25],
        },
        'raw_runs': [],
        'aggregated': {},
    }

    total_runs = len(MODELS) * len(RATIO_DIRS) * len(SEEDS)
    current_run = 0

    print(f"Starting Table 1 evaluation sweep across {total_runs} checkpoints...")
    t_start = time.time()

    for model_name in MODELS:
        results_out['aggregated'][model_name] = {}

        for ratio_label, ratio_dir in RATIO_DIRS:
            seed_metrics = []

            for seed in SEEDS:
                current_run += 1
                metrics_file = REPO_ROOT / 'results' / model_name / ratio_dir / f'seed_{seed}' / 'metrics.json'
                if not metrics_file.exists():
                    print(f"[{current_run}/{total_runs}] WARNING: Missing {metrics_file}")
                    continue

                with open(metrics_file) as fp:
                    m_data = json.load(fp)

                tau_star = m_data.get('tau_star', 0.10)
                weights_path = find_model_weights(m_data, model_name, ratio_dir, seed)

                if not weights_path:
                    print(f"[{current_run}/{total_runs}] ERROR: Weights not found for {model_name} {ratio_label} seed {seed}")
                    continue

                print(f"[{current_run}/{total_runs}] Evaluating {model_name} {ratio_label} seed {seed} (tau*={tau_star:.4f})...")
                t0 = time.time()

                yolo_model = YOLO(weights_path)
                val_res = yolo_model.val(
                    data=DATA_YAML,
                    split='test',
                    batch=32,
                    workers=0,
                    device=0,
                    verbose=False,
                    plots=False,
                    exist_ok=True,
                    project='runs/detect/eval_table1',
                    name='tmp',
                )

                r_curve, p_curve = None, None
                for cr in val_res.box.curves_results:
                    x, y, xl, yl = cr
                    if yl == 'Recall' and xl == 'Confidence':
                        r_curve = (x, y)
                    elif yl == 'Precision' and xl == 'Confidence':
                        p_curve = (x, y)

                if r_curve is None or p_curve is None:
                    raise RuntimeError("Failed to extract curves from validation results.")

                m_tau_star = extract_metrics_at_tau(r_curve, p_curve, tau_star)
                m_tau_10 = extract_metrics_at_tau(r_curve, p_curve, 0.10)
                m_tau_25 = extract_metrics_at_tau(r_curve, p_curve, 0.25)

                # Also retrieve existing precision_at_tau from metrics.json for complete reference
                orig_prec_star = m_data.get('precision_at_tau')
                orig_min_rec_star = m_data.get('min_safety_recall')

                run_record = {
                    'model': model_name,
                    'pruning_ratio': ratio_label,
                    'seed': seed,
                    'weights_path': weights_path,
                    'tau_star': tau_star,
                    'tau_star_eval': m_tau_star,
                    'tau_10_eval': m_tau_10,
                    'tau_25_eval': m_tau_25,
                    'orig_precision_at_tau': orig_prec_star,
                    'orig_min_safety_recall': orig_min_rec_star,
                }
                seed_metrics.append(run_record)
                results_out['raw_runs'].append(run_record)

                dt = time.time() - t0
                print(f"    Done in {dt:.1f}s | tau*: MinRec={m_tau_star['min_recall']:.4f}, Prec={m_tau_star['macro_precision']:.4f}, F1={m_tau_star['f1_harmonic']:.4f} | "
                      f"tau=0.10: MinRec={m_tau_10['min_recall']:.4f} | tau=0.25: MinRec={m_tau_25['min_recall']:.4f}")

            # Aggregate across seeds
            if seed_metrics:
                def stats(key_fn):
                    vals = [key_fn(s) for s in seed_metrics]
                    mean_val = float(np.mean(vals))
                    std_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                    return {'mean': mean_val, 'std': std_val, 'values': vals}

                agg_entry = {
                    'pruning_label': ratio_label,
                    'n_seeds': len(seed_metrics),
                    'tau_star': stats(lambda s: s['tau_star']),
                    'tau_star_min_recall': stats(lambda s: s['tau_star_eval']['min_recall']),
                    'tau_star_precision': stats(lambda s: s['tau_star_eval']['macro_precision']),
                    'tau_star_f1_harmonic': stats(lambda s: s['tau_star_eval']['f1_harmonic']),
                    'tau_star_f1_arithmetic': stats(lambda s: s['tau_star_eval']['f1_arithmetic']),
                    'tau_10_min_recall': stats(lambda s: s['tau_10_eval']['min_recall']),
                    'tau_10_precision': stats(lambda s: s['tau_10_eval']['macro_precision']),
                    'tau_10_f1_harmonic': stats(lambda s: s['tau_10_eval']['f1_harmonic']),
                    'tau_10_f1_arithmetic': stats(lambda s: s['tau_10_eval']['f1_arithmetic']),
                    'tau_25_min_recall': stats(lambda s: s['tau_25_eval']['min_recall']),
                    'tau_25_precision': stats(lambda s: s['tau_25_eval']['macro_precision']),
                    'tau_25_f1_harmonic': stats(lambda s: s['tau_25_eval']['f1_harmonic']),
                    'tau_25_f1_arithmetic': stats(lambda s: s['tau_25_eval']['f1_arithmetic']),
                }
                results_out['aggregated'][model_name][ratio_label] = agg_entry

    total_time = time.time() - t_start
    print(f"\nAll evaluations complete in {total_time / 60:.2f} minutes.")

    out_file = REPO_ROOT / 'results' / 'table1_data.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(results_out, f, indent=2)
    print(f"Results successfully saved to {out_file}")

    return results_out


if __name__ == '__main__':
    run_evaluation()
