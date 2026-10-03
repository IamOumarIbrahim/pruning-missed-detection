import json
from pathlib import Path

print("="*70)
print("RECORDED PARAMS AND FLOPS IN RESULTS DIRECTORY")
print("="*70)

for m in ['yolo11n', 'yolo26n']:
    print(f"\n--- {m.upper()} ---")
    print(f"{'Ratio':8s} | {'Num Params':12s} | {'FLOPs (G)':10s} | {'Size (MB)':10s} | {'FPS':8s}")
    print("-" * 65)
    for r in ['baseline', '10pct', '20pct', '30pct', '40pct', '50pct']:
        if r == 'baseline':
            p = Path(f"results/{m}/baseline/seed_0/metrics.json")
            lbl = "0%"
        else:
            p = Path(f"results/{m}/pruning_fp32/{r}/seed_0/metrics.json")
            lbl = r.replace('pct', '%')
            
        if p.exists():
            with open(p) as f:
                d = json.load(f)
            params = d.get('params', d.get('parameters', 'N/A'))
            flops = d.get('flops_g', 'N/A')
            size = d.get('size_mb', 'N/A')
            fps = d.get('fps', 'N/A')
            p_str = f"{params:,d}" if isinstance(params, int) else str(params)
            f_str = f"{flops:.2f}G" if isinstance(flops, (int, float)) else str(flops)
            s_str = f"{size:.2f}MB" if isinstance(size, (int, float)) else str(size)
            fps_str = f"{fps:.1f}" if isinstance(fps, (int, float)) else str(fps)
            print(f"{lbl:8s} | {p_str:12s} | {f_str:10s} | {s_str:10s} | {fps_str:8s}")
        else:
            print(f"{lbl:8s} | File not found: {p}")
