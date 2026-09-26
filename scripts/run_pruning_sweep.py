"""Step 2: Pruning sweep at FP32.

For each model x ratio x seed: prune baseline, fine-tune 100 epochs,
evaluate, optimise threshold with R_floor from step 1.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO
from prune.pruner import prune_model, get_model_info, PrunedDetectionTrainer
from eval.metrics import evaluate_model
from eval.threshold import optimize_threshold
from eval.benchmark import benchmark_fps

# ── Protocol constants ──────────────────────────────────────────────
MODELS = ['yolo11n', 'yolo26n']
SEEDS = [0, 1, 2]
PRUNING_RATIO_BASE = 0.10          # R = 10 %
PRUNING_MULTIPLIERS = [1, 2, 3, 4, 5]
EPOCHS = 100
BATCH = 16
IMGSZ = 640
DATA_YAML = 'configs/dmd_rgb.yaml'


def main():
    parser = argparse.ArgumentParser(description='Step 2: Pruning sweep')
    parser.add_argument('--models', nargs='+', default=MODELS)
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    parser.add_argument('--base-ratio', type=float,
                        default=PRUNING_RATIO_BASE)
    parser.add_argument('--data', default=DATA_YAML)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    for model_name in args.models:
        print(f'\n{"="*60}')
        print(f'Pruning sweep: {model_name}')
        print(f'{"="*60}')

        rbase_path = Path('results') / model_name / 'baseline' / 'r_base.json'
        with open(rbase_path) as f:
            rbase_info = json.load(f)

        for mult in PRUNING_MULTIPLIERS:
            ratio = args.base_ratio * mult
            ratio_label = f'{int(ratio * 100)}pct'
            print(f'\n--- Ratio: {ratio:.0%} ({mult}R) ---')

            fps_measured = False
            fps_info = {}

            for seed in args.seeds:
                print(f'  Seed {seed}:')

                baseline_weights = (
                    Path('models') / model_name / 'baseline'
                    / f'seed_{seed}' / 'weights' / 'best.pt'
                )

                # Prune
                prune_dir = (Path('models') / model_name / 'pruned'
                             / ratio_label)
                pruned_path = prune_dir / f'seed_{seed}_pruned.pt'
                prune_model(str(baseline_weights), ratio, str(pruned_path))

                # Fine-tune
                ft_dir = (Path('models') / model_name / 'pruning_fp32'
                          / ratio_label)
                ft_model = YOLO(str(pruned_path))
                ft_model.train(
                    trainer=PrunedDetectionTrainer,
                    data=args.data,
                    epochs=EPOCHS,
                    batch=BATCH,
                    imgsz=IMGSZ,
                    patience=0,
                    amp=True,
                    seed=seed,
                    project=str(ft_dir),
                    name=f'seed_{seed}',
                    exist_ok=True,
                    device=args.device,
                )
                target_weights_dir = ft_dir / f'seed_{seed}' / 'weights'
                best_weights = target_weights_dir / 'best.pt'
                candidates = [best_weights, target_weights_dir / 'last.pt']
                if hasattr(ft_model, 'trainer') and ft_model.trainer and hasattr(ft_model.trainer, 'save_dir'):
                    t_dir = Path(ft_model.trainer.save_dir) / 'weights'
                    candidates.extend([t_dir / 'best.pt', t_dir / 'last.pt'])
                ext_dir = Path('C:/Dev/repos/Public repos/DMS-Eval/runs/detect/models') / model_name / 'pruning_fp32' / ratio_label / f'seed_{seed}' / 'weights'
                candidates.extend([ext_dir / 'best.pt', ext_dir / 'last.pt'])

                for cand in candidates:
                    if cand.exists():
                        if not best_weights.exists():
                            target_weights_dir.mkdir(parents=True, exist_ok=True)
                            import shutil
                            shutil.copy2(cand, best_weights)
                        best_weights = cand
                        break

                # Evaluate
                ev = evaluate_model(str(best_weights), args.data,
                                    split='test')
                r_floor_k = rbase_info['r_floor_per_seed'][str(seed)]
                opt = optimize_threshold(ev['matches'], ev['gt_counts'],
                                         r_floor_k)
                info = get_model_info(str(best_weights))

                if not fps_measured:
                    fps_info = benchmark_fps(str(best_weights), imgsz=IMGSZ,
                                            device=args.device)
                    fps_measured = True

                metrics = {
                    'model': model_name,
                    'pruning_ratio': ratio,
                    'pruning_label': f'{mult}R',
                    'quantization': 'fp32',
                    'seed': seed,
                    'weights_path': str(best_weights),
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
                    'fps': fps_info.get('fps'),
                    'r_floor_k': r_floor_k,
                }

                out_dir = (Path('results') / model_name / 'pruning_fp32'
                           / ratio_label / f'seed_{seed}')
                out_dir.mkdir(parents=True, exist_ok=True)
                with open(out_dir / 'metrics.json', 'w') as f:
                    json.dump(metrics, f, indent=2)

                print(f'    tau*={opt["tau_star"]:.4f}  '
                      f'min_recall='
                      f'{opt["metrics"]["min_safety_recall"]:.4f}')


if __name__ == '__main__':
    main()
