"""Out-of-sample validation-calibrated evaluation sweep.

Strictly follows pre-registered protocol:
1. R_base is computed on the validation split ('val') at canonical default tau = 0.25.
2. R_floor = R_base_val - 0.05.
3. tau* is selected strictly on the validation split:
   tau* = max {tau in [0.01, 0.90] : min_c Recall_c^val(tau) >= R_floor}.
4. tau* is evaluated out-of-sample on the held-out test split ('test').
5. Extracts per-class AP50, min-class AP50, achieved test margin, and floor compliance rate.
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import prune.pruner
from ultralytics import YOLO

DATA_YAML = "configs/dmd_rgb.yaml"
OUTPUT_JSON = REPO_ROOT / "results" / "calibrated_eval_data.json"
MODELS = ['yolo11n', 'yolo26n']
RATIO_DIRS = [
    ('0%', 'baseline'),
    ('10%', 'pruning_fp32/10pct'),
    ('20%', 'pruning_fp32/20pct'),
    ('30%', 'pruning_fp32/30pct'),
    ('40%', 'pruning_fp32/40pct'),
    ('50%', 'pruning_fp32/50pct'),
    ('60%', 'pruning_fp32/60pct'),
    ('70%', 'pruning_fp32/70pct'),
    ('80%', 'pruning_fp32/80pct'),
    ('90%', 'pruning_fp32/90pct'),
]
SEEDS = [0, 1, 2, 3, 4]
CLASS_NAMES = ['yawning', 'hand_over_mouth', 'drinking', 'phone_use']
# Test ground truth annotation counts:
# yawn: 33, hand: 28, drink: 56, phone: 497
GT_TEST_COUNTS = [33, 28, 56, 497]
# Val ground truth annotation counts:
# yawn: 32, hand: 30, drink: 54, phone: 523
GT_VAL_COUNTS = [32, 30, 54, 523]


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


def run_val_and_curves(model, split):
    """Run model.val on specified split and extract curves and per-class AP."""
    res = model.val(
        data=DATA_YAML,
        split=split,
        batch=64,
        device=0,
        workers=0,
        verbose=False,
        plots=False,
        project='runs/detect/calib_diag',
        name='tmp',
        exist_ok=True
    )
    r_curve, p_curve = None, None
    for cr in res.box.curves_results:
        if cr[3] == 'Recall' and cr[2] == 'Confidence':
            r_curve = (cr[0], cr[1])
        elif cr[3] == 'Precision' and cr[2] == 'Confidence':
            p_curve = (cr[0], cr[1])

    # Per-class AP50 and AP50:95
    # res.box.maps is array of shape (4,) for per-class mAP50:95
    # res.box.ap50 is per-class AP at IoU 0.50
    ap50_per_cls = {}
    for i, cname in enumerate(CLASS_NAMES):
        if hasattr(res.box, 'ap50') and len(res.box.ap50) > i:
            ap50_per_cls[cname] = float(res.box.ap50[i])
        elif hasattr(res.box, 'all_ap') and res.box.all_ap is not None:
            # all_ap shape is (num_classes, 10)
            ap50_per_cls[cname] = float(res.box.all_ap[i, 0])
        else:
            ap50_per_cls[cname] = float(res.box.maps[i])

    metrics_dict = {
        'map50': float(res.box.map50),
        'map50_95': float(res.box.map),
        'ap50_per_class': ap50_per_cls,
        'min_ap50': float(min(ap50_per_cls.values())),
        'worst_ap50_class': min(ap50_per_cls, key=ap50_per_cls.get),
        'r_curve': r_curve,
        'p_curve': p_curve,
    }
    return metrics_dict


def get_metrics_at_tau(r_curve, p_curve, tau, gt_counts):
    idx = int(np.argmin(np.abs(r_curve[0] - tau)))
    actual_tau = float(r_curve[0][idx])

    recs = [float(r_curve[1][c, idx]) for c in range(4)]
    precs = [float(p_curve[1][c, idx]) for c in range(4)]

    min_rec = float(min(recs))
    worst_class = CLASS_NAMES[recs.index(min_rec)]
    macro_p = float(np.mean(precs))

    tps = [recs[c] * gt_counts[c] for c in range(4)]
    preds = [tps[c] / precs[c] if precs[c] > 0 else 0.0 for c in range(4)]
    total_tp = sum(tps)
    total_pred = sum(preds)
    micro_p = float(total_tp / total_pred) if total_pred > 0 else 0.0
    total_fp = float(total_pred - total_tp)

    f1_macro = float(2 * macro_p * min_rec / (macro_p + min_rec)) if (macro_p + min_rec) > 0 else 0.0

    return {
        'actual_tau': actual_tau,
        'min_recall': min_rec,
        'worst_class': worst_class,
        'per_class_recall': {CLASS_NAMES[c]: recs[c] for c in range(4)},
        'macro_precision': macro_p,
        'micro_precision': micro_p,
        'per_class_precision': {CLASS_NAMES[c]: precs[c] for c in range(4)},
        'f1_macro': f1_macro,
        'total_fp': total_fp,
    }


def find_tau_star_on_val(r_curve, p_curve, r_floor, gt_counts):
    """Boundary-clamping rule: tau* = max {tau : min_c Recall_c^val(tau) >= r_floor}."""
    taus = np.linspace(0.01, 0.90, 900)
    best_tau = None

    # Check from highest threshold downwards
    for tau in reversed(taus):
        m = get_metrics_at_tau(r_curve, p_curve, tau, gt_counts)
        if m['min_recall'] >= r_floor:
            best_tau = tau
            return float(best_tau), True, m

    # Infeasible: record feasible=False explicitly and do not assign a pseudo tau*
    best_m = get_metrics_at_tau(r_curve, p_curve, 0.01, gt_counts)
    return None, False, best_m


def main():
    print("=" * 70)
    print("Starting Validation-Calibrated Protocol Evaluation across all 36 models")
    print("=" * 70)

    all_data = {}
    if OUTPUT_JSON.exists():
        with open(OUTPUT_JSON, 'r', encoding='utf-8') as f:
            all_data = json.load(f)

    # 1. Establish R_base per model family on validation split at tau = 0.25
    r_base_info = {}
    for model_name in MODELS:
        print(f"\n--- Calibrating Baseline R_base for {model_name} on Validation Split ---")
        val_recalls_at_25 = []
        for seed in SEEDS:
            w_path = find_weights(model_name, 'baseline', seed)
            if not w_path:
                continue
            m = YOLO(w_path)
            val_out = run_val_and_curves(m, split='val')
            m25 = get_metrics_at_tau(val_out['r_curve'], val_out['p_curve'], 0.25, GT_VAL_COUNTS)
            val_recalls_at_25.append(m25['min_recall'])
            print(f"  Seed {seed} Val Min Recall @ tau=0.25: {m25['min_recall']:.4f} (Worst: {m25['worst_class']})")

        r_base_mean = float(np.mean(val_recalls_at_25))
        r_floor = r_base_mean - 0.05
        r_base_info[model_name] = {
            'r_base_val': r_base_mean,
            'r_floor': r_floor,
            'per_seed_r_base': val_recalls_at_25,
            'per_seed_r_floor': [r - 0.05 for r in val_recalls_at_25]
        }
        print(f"--> {model_name}: R_base^val = {r_base_mean:.4f} | R_floor = {r_floor:.4f}")

    all_data['calibration_info'] = r_base_info

    # 2. Evaluate all 36 checkpoints
    for model_name in MODELS:
        if model_name not in all_data:
            all_data[model_name] = {}

        r_floor_mean = r_base_info[model_name]['r_floor']
        per_seed_floors = r_base_info[model_name]['per_seed_r_floor']

        for r_lbl, r_dir in RATIO_DIRS:
            if r_lbl not in all_data[model_name]:
                all_data[model_name][r_lbl] = {}

            for seed in SEEDS:
                seed_key = str(seed)
                if seed_key in all_data[model_name][r_lbl]:
                    print(f"Skipping {model_name} {r_lbl} seed {seed} (already evaluated)")
                    continue

                w_path = find_weights(model_name, r_dir, seed)
                if not w_path:
                    print(f"Missing weights for {model_name} {r_lbl} seed {seed}")
                    continue

                print(f"\nProcessing {model_name} {r_lbl} seed {seed}...")
                t0 = time.time()
                m = YOLO(w_path)

                # A. Val evaluation for tau* selection
                val_out = run_val_and_curves(m, split='val')
                seed_floor = per_seed_floors[seed]
                tau_star, feasible, val_m_star = find_tau_star_on_val(
                    val_out['r_curve'], val_out['p_curve'], seed_floor, GT_VAL_COUNTS
                )
                print(f"  [VAL] tau* = {tau_star:.4f} (feasible: {feasible}) | Val Min Recall = {val_m_star['min_recall']:.4f} vs floor {seed_floor:.4f}")

                # B. Test evaluation (Out-of-sample)
                test_out = run_val_and_curves(m, split='test')
                test_m_star = get_metrics_at_tau(test_out['r_curve'], test_out['p_curve'], tau_star, GT_TEST_COUNTS)
                test_m_25 = get_metrics_at_tau(test_out['r_curve'], test_out['p_curve'], 0.25, GT_TEST_COUNTS)
                test_m_50 = get_metrics_at_tau(test_out['r_curve'], test_out['p_curve'], 0.50, GT_TEST_COUNTS)

                achieved_margin = test_m_star['min_recall'] - seed_floor
                passed_test_floor = bool(test_m_star['min_recall'] >= seed_floor)

                dt = time.time() - t0
                print(f"  [TEST out-of-sample @ tau*={tau_star:.4f}]: Min Recall = {test_m_star['min_recall']:.4f} (Margin: {achieved_margin:+.4f}, Pass: {passed_test_floor}) | Prec = {test_m_star['macro_precision']:.4f} | Min AP50 = {test_out['min_ap50']:.4f}")

                # Pack metrics
                record = {
                    'model': model_name,
                    'pruning_ratio': r_lbl,
                    'seed': seed,
                    'val_r_floor': seed_floor,
                    'tau_star': tau_star,
                    'val_feasible': feasible,
                    'val_min_recall_at_tau_star': val_m_star['min_recall'],
                    'test_mAP50': test_out['map50'],
                    'test_mAP50_95': test_out['map50_95'],
                    'test_min_ap50': test_out['min_ap50'],
                    'test_worst_ap50_class': test_out['worst_ap50_class'],
                    'test_ap50_per_class': test_out['ap50_per_class'],
                    'test_at_tau_star': {
                        'min_recall': test_m_star['min_recall'],
                        'worst_class': test_m_star['worst_class'],
                        'macro_precision': test_m_star['macro_precision'],
                        'micro_precision': test_m_star['micro_precision'],
                        'f1_macro': test_m_star['f1_macro'],
                        'achieved_margin': achieved_margin,
                        'passed_floor': passed_test_floor,
                        'total_fp': test_m_star['total_fp'],
                        'per_class_recall': test_m_star['per_class_recall'],
                        'per_class_precision': test_m_star['per_class_precision'],
                    },
                    'test_at_tau_25': {
                        'min_recall': test_m_25['min_recall'],
                        'worst_class': test_m_25['worst_class'],
                        'macro_precision': test_m_25['macro_precision'],
                        'f1_macro': test_m_25['f1_macro'],
                        'total_fp': test_m_25['total_fp'],
                    },
                    'test_at_tau_50': {
                        'min_recall': test_m_50['min_recall'],
                        'worst_class': test_m_50['worst_class'],
                        'macro_precision': test_m_50['macro_precision'],
                        'f1_macro': test_m_50['f1_macro'],
                        'total_fp': test_m_50['total_fp'],
                    },
                    'eval_time_sec': dt,
                }

                all_data[model_name][r_lbl][seed_key] = record

                with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
                    json.dump(all_data, f, indent=2)

    print(f"\nAll 36 models evaluated! Results saved to {OUTPUT_JSON}")


if __name__ == '__main__':
    main()
