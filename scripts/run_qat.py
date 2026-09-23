"""Step 4b: QAT for failing PTQ-INT8 ratios.

QAT selection rule: per model, run QAT (single seed) on the failing
PTQ-INT8 ratio with the smallest min-class-recall gap to R_floor.
Skipped for models with zero failing ratios.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from eval.metrics import evaluate_model
from eval.threshold import optimize_threshold
from eval.benchmark import benchmark_fps
from quant.qat import run_qat as _run_qat

# ── Protocol constants ──────────────────────────────────────────────
MODELS = ['yolo11n', 'yolo26n']
SEEDS = [0, 1, 2]
PRUNING_RATIO_BASE = 0.10
PRUNING_MULTIPLIERS = [0, 1, 2, 3, 4, 5]
IMGSZ = 640
DATA_YAML = 'configs/dmd_rgb.yaml'


def find_qat_candidate(model_name, rbase_info, base_ratio):
    """Return the failing INT8 ratio closest to R_floor, or None."""
    r_floor = rbase_info['r_floor']
    best_candidate = None
    best_gap = float('inf')

    for mult in PRUNING_MULTIPLIERS:
        ratio = base_ratio * mult
        ratio_label = f'{int(ratio * 100)}pct' if mult > 0 else '0pct'

        recalls = []
        for seed in SEEDS:
            mpath = (Path('results') / model_name / 'joint_int8'
                     / ratio_label / f'seed_{seed}' / 'metrics.json')
            if not mpath.exists():
                continue
            with open(mpath) as f:
                m = json.load(f)
            recalls.append(m['min_safety_recall'])

        if not recalls:
            continue

        mean_recall = float(np.mean(recalls))
        if mean_recall < r_floor:                       # fails
            gap = r_floor - mean_recall
            if gap < best_gap:
                best_gap = gap
                best_candidate = {
                    'multiplier': mult,
                    'ratio': ratio,
                    'ratio_label': ratio_label,
                    'mean_recall': mean_recall,
                    'gap': gap,
                }
    return best_candidate


def main():
    parser = argparse.ArgumentParser(description='Step 4b: QAT')
    parser.add_argument('--models', nargs='+', default=MODELS)
    parser.add_argument('--base-ratio', type=float,
                        default=PRUNING_RATIO_BASE)
    parser.add_argument('--data', default=DATA_YAML)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    for model_name in args.models:
        print(f'\n{"="*60}')
        print(f'QAT check: {model_name}')
        print(f'{"="*60}')

        rbase_path = (Path('results') / model_name
                      / 'baseline' / 'r_base.json')
        with open(rbase_path) as f:
            rbase_info = json.load(f)

        candidate = find_qat_candidate(model_name, rbase_info,
                                       args.base_ratio)
        if candidate is None:
            print('No failing PTQ-INT8 ratios. QAT skipped.')
            continue

        mult = candidate['multiplier']
        ratio_label = candidate['ratio_label']
        print(f'QAT candidate: {ratio_label} '
              f'(gap={candidate["gap"]:.4f})')

        # Source weights for QAT: seed 0 pruned+finetuned FP32
        if mult == 0:
            src = (Path('models') / model_name / 'baseline'
                   / 'seed_0' / 'weights' / 'best.pt')
        else:
            src = (Path('models') / model_name / 'pruning_fp32'
                   / ratio_label / 'seed_0' / 'weights' / 'best.pt')

        output_dir = Path('models') / model_name / 'qat' / ratio_label
        qat_result = _run_qat(
            str(src), args.data, str(output_dir),
            seed=0, device=args.device,
        )

        ev = evaluate_model(qat_result['engine_path'], args.data,
                            split='test')
        r_floor_0 = rbase_info['r_floor_per_seed']['0']
        opt = optimize_threshold(ev['matches'], ev['gt_counts'], r_floor_0)
        fps_info = benchmark_fps(qat_result['engine_path'], imgsz=IMGSZ,
                                 device=args.device)
        engine_size = (Path(qat_result['engine_path']).stat().st_size
                       / (1024 * 1024))

        metrics = {
            'model': model_name,
            'pruning_ratio': candidate['ratio'],
            'pruning_label': f'{mult}R' if mult > 0 else '0%',
            'quantization': 'int8_qat',
            'seed': 0,
            'engine_path': qat_result['engine_path'],
            'tau_star': opt['tau_star'],
            'feasible': opt['feasible'],
            'map50': ev['map50'],
            'map50_95': ev['map50_95'],
            'recall_at_tau': opt['metrics']['macro_recall'],
            'precision_at_tau': opt['metrics']['precision'],
            'min_safety_recall': opt['metrics']['min_safety_recall'],
            'per_class_recall': opt['metrics']['per_class_recall'],
            'size_mb': engine_size,
            'fps': fps_info.get('fps'),
            'r_floor_k': r_floor_0,
        }

        out_dir = Path('results') / model_name / 'qat' / ratio_label
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f'QAT done: tau*={opt["tau_star"]:.4f}  '
              f'min_recall={opt["metrics"]["min_safety_recall"]:.4f}')


if __name__ == '__main__':
    main()
