"""Step 5: Final selection per model family.

Among configs that Pass, selects the one maximising FPS.
Ties within 2 % of max FPS are broken by highest mean precision.
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

MODELS = ['yolo11n', 'yolo26n']


def load_all_results(model_name):
    """Recursively collect every metrics.json under results/{model}."""
    root = Path('results') / model_name
    out = []
    for p in root.rglob('metrics.json'):
        if 'r_base' not in p.name:
            with open(p) as f:
                out.append(json.load(f))
    return out


def aggregate(all_metrics, r_floor):
    """Group by (pruning_ratio, quantization) and apply pass criterion."""
    groups = defaultdict(list)
    for m in all_metrics:
        key = (m['pruning_ratio'], m['quantization'])
        groups[key].append(m)

    configs = []
    for (pr, q), seeds in groups.items():
        recalls = [m['min_safety_recall'] for m in seeds]
        precs = [m['precision_at_tau'] for m in seeds]
        m50 = [m['map50'] for m in seeds]
        m50_95 = [m['map50_95'] for m in seeds]
        mac_rec = [m['recall_at_tau'] for m in seeds]

        mean_r = float(np.mean(recalls))
        std_r = float(np.std(recalls))
        mean_p = float(np.mean(precs))

        passed = mean_r >= r_floor
        boundary = passed and (mean_r - std_r < r_floor)

        fps_vals = [m.get('fps') for m in seeds if m.get('fps')]
        fps = fps_vals[0] if fps_vals else None

        size_vals = [m.get('size_mb') for m in seeds if m.get('size_mb')]
        flops_vals = [m.get('flops_g') for m in seeds if m.get('flops_g')]

        configs.append({
            'pruning_ratio': pr,
            'quantization': q,
            'mean_min_recall': mean_r,
            'std_min_recall': std_r,
            'mean_precision': mean_p,
            'std_precision': float(np.std(precs)),
            'mean_map50': float(np.mean(m50)),
            'std_map50': float(np.std(m50)),
            'mean_map50_95': float(np.mean(m50_95)),
            'std_map50_95': float(np.std(m50_95)),
            'mean_recall_at_tau': float(np.mean(mac_rec)),
            'std_recall_at_tau': float(np.std(mac_rec)),
            'fps': fps,
            'size_mb': float(np.mean(size_vals)) if size_vals else None,
            'flops_g': float(np.mean(flops_vals)) if flops_vals else None,
            'passed': passed,
            'boundary': boundary,
            'n_seeds': len(seeds),
        })
    return configs


def select_best(configs):
    """Max FPS among passing configs; ties within 2 % broken by precision."""
    passing = [c for c in configs if c['passed'] and c.get('fps')]
    if not passing:
        return None
    max_fps = max(c['fps'] for c in passing)
    tied = [c for c in passing if c['fps'] >= max_fps * 0.98]
    tied.sort(key=lambda c: c['mean_precision'], reverse=True)
    return tied[0]


def main():
    parser = argparse.ArgumentParser(description='Step 5: Selection')
    parser.add_argument('--models', nargs='+', default=MODELS)
    args = parser.parse_args()

    for model_name in args.models:
        print(f'\n{"="*60}')
        print(f'Selection: {model_name}')
        print(f'{"="*60}')

        rbase_path = (Path('results') / model_name
                      / 'baseline' / 'r_base.json')
        with open(rbase_path) as f:
            rbase_info = json.load(f)
        r_floor = rbase_info['r_floor']

        all_m = load_all_results(model_name)
        cfgs = aggregate(all_m, r_floor)

        out_dir = Path('results') / model_name
        with open(out_dir / 'aggregated.json', 'w') as f:
            json.dump(cfgs, f, indent=2, default=str)

        best = select_best(cfgs)
        if best:
            with open(out_dir / 'selected.json', 'w') as f:
                json.dump(best, f, indent=2)
            print(f'Selected: ratio={best["pruning_ratio"]:.0%}  '
                  f'quant={best["quantization"]}  '
                  f'FPS={best["fps"]:.1f}')
        else:
            print('No passing config found.')


if __name__ == '__main__':
    main()
