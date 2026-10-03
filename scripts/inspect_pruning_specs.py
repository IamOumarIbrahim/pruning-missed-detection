import json
from pathlib import Path
import sys
sys.path.insert(0, '.')
import prune.pruner
from prune.pruner import get_model_info

print("="*70)
print("MEASURED PARAMS AND FLOPS PER RATIO")
print("="*70)

for m in ['yolo11n', 'yolo26n']:
    print(f"\n--- {m.upper()} ---")
    print(f"{'Ratio':8s} | {'Num Params':12s} | {'Param Cut':10s} | {'FLOPs (G)':10s} | {'FLOPs Cut':10s} | {'Size (MB)':10s}")
    print("-" * 75)
    base_p = f"models/{m}/baseline/seed_0/weights/best.pt"
    base_info = get_model_info(base_p)
    base_params = base_info['num_params']
    base_flops = base_info['flops_g']
    
    for r_lbl in ['0pct', '10pct', '20pct', '30pct', '40pct', '50pct']:
        if r_lbl == '0pct':
            p = base_p
        else:
            p = f"models/{m}/pruning_fp32/{r_lbl}/seed_0/weights/best.pt"
            if not Path(p).exists():
                p = f"models/{m}/pruned/{r_lbl}/seed_0_pruned.pt"
        
        if Path(p).exists():
            info = get_model_info(p)
            param_cut = (1.0 - info['num_params'] / base_params) * 100
            flops_cut = (1.0 - info['flops_g'] / base_flops) * 100 if info['flops_g'] and base_flops else 0.0
            print(f"{r_lbl:8s} | {info['num_params']:12,d} | {param_cut:9.1f}% | {info['flops_g']:9.2f}G | {flops_cut:9.1f}% | {info['size_mb']:9.2f}MB")
        else:
            print(f"{r_lbl:8s} | Not available")
