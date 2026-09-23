"""Step 1: Train baseline models and establish R_base / R_floor.

Trains each model with K seeds at FP32 0% pruning, evaluates on the
test set, computes R_base per seed, and derives R_floor.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO
from eval.metrics import evaluate_model
from eval.threshold import compute_r_base, optimize_threshold
from eval.benchmark import benchmark_fps
from prune.pruner import get_model_info

# ── Protocol constants ──────────────────────────────────────────────
MODELS = ['yolo11n', 'yolo26n']
SEEDS = [0, 1, 2]
EPOCHS = 100
BATCH = 16
IMGSZ = 640
DELTA = 0.05
DATA_YAML = 'configs/dmd_rgb.yaml'


def train_single(model_name, seed, data_yaml, project_dir, device=0):
    """Train one model with one seed. Returns path to best weights."""
    model = YOLO(f'{model_name}.pt')
    model.train(
        data=data_yaml,
        epochs=EPOCHS,
        batch=BATCH,
        imgsz=IMGSZ,
        patience=0,
        amp=True,
        seed=seed,
        project=str(project_dir),
        name=f'seed_{seed}',
        exist_ok=True,
        device=device,
    )
    best_pt = Path(project_dir) / f'seed_{seed}' / 'weights' / 'best.pt'
    if best_pt.exists():
        return str(best_pt)
    last_pt = Path(project_dir) / f'seed_{seed}' / 'weights' / 'last.pt'
    if last_pt.exists():
        return str(last_pt)
    return str(best_pt)


def main():
    parser = argparse.ArgumentParser(description='Step 1: Train baselines')
    parser.add_argument('--models', nargs='+', default=MODELS)
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    parser.add_argument('--data', default=DATA_YAML)
    parser.add_argument('--delta', type=float, default=DELTA)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    for model_name in args.models:
        print(f'\n{"="*60}')
        print(f'Model: {model_name}')
        print(f'{"="*60}')

        model_dir = Path('models') / model_name / 'baseline'
        results_dir = Path('results') / model_name / 'baseline'
        results_dir.mkdir(parents=True, exist_ok=True)

        # Phase 1: Train all seeds
        weight_paths = {}
        for seed in args.seeds:
            print(f'\n--- Training seed {seed} ---')
            weight_paths[seed] = train_single(
                model_name, seed, args.data, model_dir, args.device,
            )

        # Phase 2: Evaluate all seeds, compute R_base
        matches_per_seed = []
        gt_counts_per_seed = []
        eval_results = {}

        for seed in args.seeds:
            print(f'\n--- Evaluating seed {seed} ---')
            ev = evaluate_model(weight_paths[seed], args.data, split='test')
            eval_results[seed] = ev
            matches_per_seed.append(ev['matches'])
            gt_counts_per_seed.append(ev['gt_counts'])

        r_base, r_base_per_seed = compute_r_base(
            matches_per_seed, gt_counts_per_seed,
        )
        r_floor = r_base - args.delta
        r_floor_per_seed = [rb - args.delta for rb in r_base_per_seed]

        print(f'\nR_base = {r_base:.4f}')
        print(f'R_floor = {r_floor:.4f} (delta = {args.delta})')

        rbase_info = {
            'r_base': r_base,
            'r_base_per_seed': {
                str(s): v for s, v in zip(args.seeds, r_base_per_seed)
            },
            'delta': args.delta,
            'r_floor': r_floor,
            'r_floor_per_seed': {
                str(s): v for s, v in zip(args.seeds, r_floor_per_seed)
            },
        }
        with open(results_dir / 'r_base.json', 'w') as f:
            json.dump(rbase_info, f, indent=2)

        # FPS measured once per (r=0, q=FP32)
        fps_info = benchmark_fps(weight_paths[args.seeds[0]], imgsz=IMGSZ,
                                 device=args.device)

        # Phase 3: Optimise threshold per seed and save
        for seed_idx, seed in enumerate(args.seeds):
            ev = eval_results[seed]
            opt = optimize_threshold(
                ev['matches'], ev['gt_counts'],
                r_floor_per_seed[seed_idx],
            )
            info = get_model_info(weight_paths[seed])

            metrics = {
                'model': model_name,
                'pruning_ratio': 0.0,
                'pruning_label': '0%',
                'quantization': 'fp32',
                'seed': seed,
                'weights_path': weight_paths[seed],
                'tau_star': opt['tau_star'],
                'feasible': opt['feasible'],
                'map50': ev['map50'],
                'map50_95': ev['map50_95'],
                'recall_at_tau': opt['metrics']['macro_recall'],
                'precision_at_tau': opt['metrics']['precision'],
                'min_safety_recall': opt['metrics']['min_safety_recall'],
                'per_class_recall': opt['metrics']['per_class_recall'],
                'size_mb': info['size_mb'],
                'flops_g': info['flops_g'],
                'fps': fps_info['fps'],
                'r_base_k': r_base_per_seed[seed_idx],
                'r_floor_k': r_floor_per_seed[seed_idx],
            }

            seed_dir = results_dir / f'seed_{seed}'
            seed_dir.mkdir(parents=True, exist_ok=True)
            with open(seed_dir / 'metrics.json', 'w') as f:
                json.dump(metrics, f, indent=2)

            print(f'Seed {seed}: tau*={opt["tau_star"]:.4f}  '
                  f'min_recall={opt["metrics"]["min_safety_recall"]:.4f}  '
                  f'precision={opt["metrics"]["precision"]:.4f}')

        print(f'\nBaseline complete for {model_name}.')


if __name__ == '__main__':
    main()
