import json
from pathlib import Path
import numpy as np
import torch

DATA_YAML = "configs/dmd_rgb.yaml"
SEEDS = [0, 1, 2]
RATIO_DIRS = [
    ('0%', 'baseline'),
    ('10%', 'pruning_fp32/10pct'),
    ('20%', 'pruning_fp32/20pct'),
]
CLASS_NAMES = ['yawning', 'hand_over_mouth', 'drinking', 'phone_use']
REPO_ROOT = Path(".").resolve()

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

def main():
    from ultralytics import YOLO
    for model in ['yolo26n']:
        print(f"================ {model} ================")
        for r_lbl, r_dir in RATIO_DIRS:
            print(f"\n=== Ratio {r_lbl} ===")
            for seed in SEEDS:
                w = find_weights(model, r_dir, seed)
                m = YOLO(w)
                res = m.val(
                    data=DATA_YAML,
                    split='test',
                    batch=32,
                    device=0,
                    workers=0,
                    verbose=False,
                    plots=False,
                    project='runs/detect/diag',
                    name='tmp',
                    exist_ok=True
                )
                r_curve = None
                p_curve = None
                for cr in res.box.curves_results:
                    if cr[3] == 'Recall' and cr[2] == 'Confidence':
                        r_curve = (cr[0], cr[1])
                    elif cr[3] == 'Precision' and cr[2] == 'Confidence':
                        p_curve = (cr[0], cr[1])
                
                idx = int(np.argmin(np.abs(r_curve[0] - 0.5)))
                rec_per_cls = {CLASS_NAMES[c]: round(float(r_curve[1][c, idx]), 4) for c in range(4)}
                prec_per_cls = {CLASS_NAMES[c]: round(float(p_curve[1][c, idx]), 4) for c in range(4)}
                min_c = min(rec_per_cls, key=rec_per_cls.get)
                print(f"  Seed {seed}: Min Recall = {rec_per_cls[min_c]:.4f} (Worst Class: {min_c})")
                print(f"    Recalls: {rec_per_cls}")
                print(f"    Precisions: {prec_per_cls}")

if __name__ == '__main__':
    main()
