"""Update all metrics.json files with exact FLOPs (GFLOPs) and parameter counts."""

import json, glob
from pathlib import Path

# Architecture parameters and GFLOPs measured at 640x640
ARCH_STATS = {
    'yolo11n': {
        0:  {'params': 2590620, 'flops_g': 6.5028},
        10: {'params': 2241826, 'flops_g': 5.6299},
        20: {'params': 1944901, 'flops_g': 4.9205},
        30: {'params': 1668655, 'flops_g': 4.2984},
        40: {'params': 1430838, 'flops_g': 3.7368},
        50: {'params': 1231676, 'flops_g': 3.3006},
    },
    'yolo26n': {
        0:  {'params': 2505360, 'flops_g': 5.8986},
        10: {'params': 2183875, 'flops_g': 5.0299},
        20: {'params': 1909995, 'flops_g': 4.3253},
        30: {'params': 1655988, 'flops_g': 3.7166},
        40: {'params': 1434254, 'flops_g': 3.1570},
        50: {'params': 1247456, 'flops_g': 2.7260},
    },
}

def main():
    files = glob.glob('results/**/metrics.json', recursive=True)
    updated = 0

    for f in sorted(files):
        p = Path(f)
        with open(p) as fp:
            d = json.load(fp)

        model = d.get('model', 'yolo11n')
        ratio_pct = int(round(d.get('pruning_ratio', 0.0) * 100))

        if model in ARCH_STATS and ratio_pct in ARCH_STATS[model]:
            stats = ARCH_STATS[model][ratio_pct]
            d['params'] = stats['params']
            d['parameters'] = stats['params']
            d['flops_g'] = stats['flops_g']

            with open(p, 'w') as fp:
                json.dump(d, fp, indent=2)
            updated += 1

    print(f'Successfully updated FLOPs and parameter counts in {updated}/{len(files)} metrics.json files.')

if __name__ == '__main__':
    main()
