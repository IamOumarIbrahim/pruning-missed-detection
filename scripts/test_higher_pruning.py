import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import prune.pruner
from prune.pruner import prune_model, get_model_info

base_w = "models/yolo11n/baseline/seed_0/weights/best.pt"
for r in [0.50, 0.60, 0.70, 0.80, 0.90]:
    out_p = f"models/yolo11n/pruned/test_{int(r*100)}pct.pt"
    try:
        prune_model(base_w, r, out_p)
        info = get_model_info(out_p)
        print(f"Pruned {int(r*100)}%: params={info['num_params']:,}, flops={info['flops_g']}G, size={info['size_mb']:.2f}MB")
    except Exception as e:
        print(f"Pruned {int(r*100)}% failed: {e}")
