import json
import numpy as np

with open('results/calibrated_eval_data.json') as f:
    d = json.load(f)

print("="*70)
print("MAP50-95 VS WORST-CLASS AP50 ACROSS PRUNING RATIOS (TEST SET)")
print("="*70)

for m in ['yolo11n', 'yolo26n']:
    print(f"\n--- {m.upper()} ---")
    print(f"{'Ratio':8s} | {'mAP50':15s} | {'mAP50-95':15s} | {'Min AP50':15s} | {'Worst Class':15s}")
    print("-" * 75)
    for r in ['0%', '10%', '20%', '30%', '40%', '50%']:
        map50_vals = [d[m][r][s]['test_mAP50'] for s in ['0', '1', '2']]
        map95_vals = [d[m][r][s]['test_mAP50_95'] for s in ['0', '1', '2']]
        min_ap_vals = [d[m][r][s]['test_min_ap50'] for s in ['0', '1', '2']]
        worst_classes = [d[m][r][s]['test_worst_ap50_class'] for s in ['0', '1', '2']]
        
        m50_str = f"{np.mean(map50_vals):.3f} ± {np.std(map50_vals, ddof=1):.3f}"
        m95_str = f"{np.mean(map95_vals):.3f} ± {np.std(map95_vals, ddof=1):.3f}"
        map_str = f"{np.mean(min_ap_vals):.3f} ± {np.std(min_ap_vals, ddof=1):.3f}"
        w_class = ", ".join(set(worst_classes))
        
        print(f"{r:8s} | {m50_str:15s} | {m95_str:15s} | {map_str:15s} | {w_class:15s}")
