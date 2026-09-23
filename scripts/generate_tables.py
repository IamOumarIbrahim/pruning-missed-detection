"""Generate result tables (Tables 1-5) from saved metrics."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODELS = ['yolo11n', 'yolo26n']


def main():
    for model_name in MODELS:
        rbase_path = (Path('results') / model_name
                      / 'baseline' / 'r_base.json')
        if not rbase_path.exists():
            print(f'No results for {model_name}')
            continue

        with open(rbase_path) as f:
            rbase_info = json.load(f)
        r_floor = rbase_info['r_floor']

        print(f'\n=== {model_name}  '
              f'(R_base={rbase_info["r_base"]:.4f}  '
              f'R_floor={r_floor:.4f}) ===\n')

        agg_path = Path('results') / model_name / 'aggregated.json'
        if not agg_path.exists():
            print('  Run step 5 (run_selection.py) first.')
            continue
        with open(agg_path) as f:
            configs = json.load(f)

        for c in sorted(configs, key=lambda x: (x['quantization'],
                                                 x['pruning_ratio'])):
            flag = ' (boundary)' if c.get('boundary') else ''
            status = 'Pass' if c['passed'] else 'FAIL'
            fps_str = f'{c["fps"]:.1f}' if c.get('fps') else 'N/A'
            print(f'  ratio={c["pruning_ratio"]:.0%}  '
                  f'quant={c["quantization"]:<8s}  '
                  f'recall={c["mean_min_recall"]:.3f}'
                  f'+/-{c["std_min_recall"]:.3f}  '
                  f'prec={c["mean_precision"]:.3f}  '
                  f'mAP50={c["mean_map50"]:.3f}  '
                  f'FPS={fps_str}  '
                  f'{status}{flag}')

        sel_path = Path('results') / model_name / 'selected.json'
        if sel_path.exists():
            with open(sel_path) as f:
                sel = json.load(f)
            fps_str = f'{sel["fps"]:.1f}' if sel.get('fps') else 'N/A'
            print(f'\n  ** Selected: ratio={sel["pruning_ratio"]:.0%}  '
                  f'quant={sel["quantization"]}  FPS={fps_str}')


if __name__ == '__main__':
    main()
