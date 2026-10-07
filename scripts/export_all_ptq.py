"""Export remaining Step 2 pruned models (60%, 70%, 80%, 90%) to calibrated INT8 ONNX."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from quant.ptq import export_ptq

MODELS = ['yolo11n', 'yolo26n']
RATIOS = ['60pct', '70pct', '80pct', '90pct']
SEEDS = [0, 1, 2]
CALIB_YAML = REPO_ROOT / 'configs' / 'calibration.yaml'

def main():
    total = len(MODELS) * len(RATIOS) * len(SEEDS)
    current = 0
    print(f"Starting PTQ export for {total} checkpoints (60% to 90%)...")
    
    for model in MODELS:
        for ratio in RATIOS:
            for seed in SEEDS:
                current += 1
                src_pt = REPO_ROOT / 'models' / model / 'pruning_fp32' / ratio / f'seed_{seed}' / 'weights' / 'best.pt'
                out_dir = REPO_ROOT / 'models' / model / 'joint_int8' / ratio / f'seed_{seed}'
                out_onnx = out_dir / 'model_int8.onnx'
                
                if out_onnx.exists() and out_onnx.stat().st_size > 500_000:
                    print(f"[{current}/{total}] {model.upper()} {ratio} seed {seed}: already exists ({out_onnx.stat().st_size / (1024*1024):.2f} MB), skipping.")
                    continue
                
                if not src_pt.exists():
                    print(f"[{current}/{total}] ERROR: Source checkpoint missing: {src_pt}")
                    continue
                
                print(f"[{current}/{total}] Exporting {model.upper()} {ratio} seed {seed} from {src_pt}...")
                out_dir.mkdir(parents=True, exist_ok=True)
                res = export_ptq(
                    model_path=str(src_pt),
                    quant_level='int8',
                    output_dir=str(out_dir),
                    data_yaml=str(CALIB_YAML),
                    imgsz=640,
                    device=0
                )
                print(f"  -> Successfully exported to {res} ({Path(res).stat().st_size / (1024*1024):.2f} MB)")

if __name__ == '__main__':
    main()
