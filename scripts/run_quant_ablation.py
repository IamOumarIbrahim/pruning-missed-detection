"""Step 3: Quantization-only ablation at 0 % pruning.

Exports baseline weights to FP16 / INT8 / INT4 TensorRT engines,
evaluates each, and optimises thresholds.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.metrics import evaluate_model
from eval.threshold import optimize_threshold
from eval.benchmark import benchmark_fps
from quant.ptq import export_ptq

# ── Protocol constants ──────────────────────────────────────────────
MODELS = ['yolo11n', 'yolo26n']
SEEDS = [0, 1, 2]
QUANT_LEVELS = ['fp16', 'int8', 'int4']      # fp32 = baseline row
IMGSZ = 640
DATA_YAML = 'configs/dmd_rgb.yaml'


def main():
    parser = argparse.ArgumentParser(description='Step 3: Quant ablation')
    parser.add_argument('--models', nargs='+', default=MODELS)
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    parser.add_argument('--data', default=DATA_YAML)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    for model_name in args.models:
        print(f'\n{"="*60}')
        print(f'Quant ablation: {model_name}')
        print(f'{"="*60}')

        rbase_path = (Path('results') / model_name
                      / 'baseline' / 'r_base.json')
        with open(rbase_path) as f:
            rbase_info = json.load(f)

        for qlevel in QUANT_LEVELS:
            print(f'\n--- Quant: {qlevel} ---')

            fps_measured = False
            fps_info = {}

            for seed in args.seeds:
                print(f'  Seed {seed}:')

                baseline_weights = (
                    Path('models') / model_name / 'baseline'
                    / f'seed_{seed}' / 'weights' / 'best.pt'
                )

                engine_dir = (Path('models') / model_name / 'quant_only'
                              / qlevel / f'seed_{seed}')
                engine_path = export_ptq(
                    str(baseline_weights), qlevel, str(engine_dir),
                    data_yaml=args.data, imgsz=IMGSZ, device=args.device,
                )

                ev = evaluate_model(engine_path, args.data, split='test')
                r_floor_k = rbase_info['r_floor_per_seed'][str(seed)]
                opt = optimize_threshold(ev['matches'], ev['gt_counts'],
                                         r_floor_k)

                if not fps_measured:
                    fps_info = benchmark_fps(engine_path, imgsz=IMGSZ,
                                            device=args.device)
                    fps_measured = True

                engine_size = (Path(engine_path).stat().st_size
                               / (1024 * 1024))

                metrics = {
                    'model': model_name,
                    'pruning_ratio': 0.0,
                    'pruning_label': '0%',
                    'quantization': qlevel,
                    'seed': seed,
                    'engine_path': engine_path,
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
                    'r_floor_k': r_floor_k,
                }

                out_dir = (Path('results') / model_name / 'quant_only'
                           / qlevel / f'seed_{seed}')
                out_dir.mkdir(parents=True, exist_ok=True)
                with open(out_dir / 'metrics.json', 'w') as f:
                    json.dump(metrics, f, indent=2)

                print(f'    tau*={opt["tau_star"]:.4f}  '
                      f'min_recall='
                      f'{opt["metrics"]["min_safety_recall"]:.4f}')


if __name__ == '__main__':
    main()
