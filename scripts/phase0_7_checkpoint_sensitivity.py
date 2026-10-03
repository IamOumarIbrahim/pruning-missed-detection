import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import json
import numpy as np
import pandas as pd
from ultralytics import YOLO
import prune.pruner
from scripts.eval_val_calibrated_sweep import run_val_and_curves, find_tau_star_on_val, get_metrics_at_tau, GT_VAL_COUNTS, GT_TEST_COUNTS

print("="*75)
print("PHASE 0.7: CHECKPOINT SENSITIVITY (BEST.PT vs LAST.PT)")
print("="*75)

models_to_check = [
    ('yolo11n_seed_0', 'models/yolo11n/baseline/seed_0/weights/best.pt', 'models/yolo11n/baseline/seed_0/weights/last.pt'),
    ('yolo26n_seed_0', 'models/yolo26n/baseline/seed_0/weights/best.pt', 'models/yolo26n/baseline/seed_0/weights/last.pt'),
]

records = []

for tag, best_p, last_p in models_to_check:
    print(f"\n--- Checking {tag} ---")
    for ckpt_type, path_str in [('best', best_p), ('last', last_p)]:
        m = YOLO(path_str)
        
        # Val
        val_out = run_val_and_curves(m, split='val')
        val_m25 = get_metrics_at_tau(val_out['r_curve'], val_out['p_curve'], 0.25, GT_VAL_COUNTS)
        r_base = val_m25['min_recall']
        r_floor = r_base - 0.05
        tau_star, feasible, val_m_star = find_tau_star_on_val(val_out['r_curve'], val_out['p_curve'], r_floor, GT_VAL_COUNTS)
        
        # Test
        test_out = run_val_and_curves(m, split='test')
        test_m_star = get_metrics_at_tau(test_out['r_curve'], test_out['p_curve'], tau_star, GT_TEST_COUNTS)
        test_m25 = get_metrics_at_tau(test_out['r_curve'], test_out['p_curve'], 0.25, GT_TEST_COUNTS)
        margin = test_m_star['min_recall'] - r_floor
        
        records.append({
            'model': tag,
            'ckpt': ckpt_type,
            'val_r_base': r_base,
            'val_r_floor': r_floor,
            'tau_star': tau_star,
            'val_min_recall_star': val_m_star['min_recall'],
            'test_min_recall_star': test_m_star['min_recall'],
            'test_worst_class_star': test_m_star['worst_class'],
            'test_margin_star_pp': margin * 100,
            'test_min_recall_25': test_m25['min_recall'],
            'test_margin_25_pp': (test_m25['min_recall'] - r_floor) * 100,
            'test_mAP50': test_out['map50'],
            'test_mAP50_95': test_out['map50_95']
        })
        print(f"  [{ckpt_type:4s}]: Val R_base={r_base:.4f} | tau*={tau_star:.4f} | Test Min Recall={test_m_star['min_recall']:.4f} (Margin: {margin*100:+.2f} pp)")

df_sens = pd.DataFrame(records)
df_sens.to_csv(REPO_ROOT / 'results' / 'phase0' / 'checkpoint_sensitivity.csv', index=False)
print("\nCheckpoint sensitivity complete. Saved to results/phase0/checkpoint_sensitivity.csv")
