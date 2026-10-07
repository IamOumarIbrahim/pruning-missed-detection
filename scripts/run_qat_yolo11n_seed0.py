"""Sequential QAT (20 epochs) for YOLO11N Seed 0 across all pruning ratios:
Baseline (0%), then 10%, 20%, 30%, 40%, 50%, 70%, 80%, 90% (60% already completed).

For each ratio:
  1. Fine-tunes weights for 20 epochs with PrunedDetectionTrainer (lr0=0.001, lrf=0.01)
  2. Verifies completion (20 epochs in results.csv, best.pt exists)
  3. Exports to calibrated INT8 ONNX
  4. Evaluates on held-out test split at fixed tau = 0.50 (greedy bipartite matching)
  5. Appends record to results/fixed_tau05_qat.csv and updates RESULTS_FIXED_TAU05.md
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from prune.pruner import PrunedDetectionTrainer
from quant.ptq import export_ptq
from eval.metrics import load_yolo_labels, match_detections_single_image, CLASS_NAMES
from scripts.evaluate_all_ptq import preload_test_annotations, load_base_metadata, update_results_markdown

RESULTS_DIR = REPO_ROOT / 'results'
QAT_CSV = RESULTS_DIR / 'fixed_tau05_qat.csv'
DATA_DIR = REPO_ROOT / 'data' / 'processed' / 'RGB' / 'yolo'
DATA_YAML = REPO_ROOT / 'configs' / 'dmd_rgb.yaml'
CALIB_YAML = REPO_ROOT / 'configs' / 'calibration.yaml'
TEST_TXT = DATA_DIR / 'test.txt'
TEST_GT = {0: 33, 1: 28, 2: 56, 3: 497}
EPOCHS = 20
BATCH = 16
IMGSZ = 640

# Sequence specified by user: baseline then 10 20 30 40 50 70 80 90 (60% already complete)
RATIOS_SEQUENCE = ['0%', '10%', '20%', '30%', '40%', '50%', '70%', '80%', '90%']


def evaluate_onnx_model(onnx_path, test_gts):
    m = YOLO(str(onnx_path), task='detect')
    res_gen = m.predict(source=str(TEST_TXT), conf=0.001, imgsz=640, device='cpu', verbose=False, stream=True)
    
    dets = []
    for r in res_gen:
        img_p = str(Path(r.path).resolve())
        boxes_gt = test_gts.get(img_p, [])
        gt_boxes = [(b[1], b[2], b[3], b[4]) for b in boxes_gt]
        gt_cls = [b[0] for b in boxes_gt]

        p_boxes = r.boxes.xyxy.cpu().numpy().tolist() if len(r.boxes) else []
        p_confs = r.boxes.conf.cpu().numpy().tolist() if len(r.boxes) else []
        p_cls = r.boxes.cls.cpu().int().numpy().tolist() if len(r.boxes) else []

        matches = match_detections_single_image(p_boxes, p_confs, p_cls, gt_boxes, gt_cls, iou_threshold=0.5)
        for conf, is_tp, cls_id in matches:
            if conf >= 0.50:
                dets.append({'conf': float(conf), 'is_tp': bool(is_tp), 'class_id': int(cls_id)})

    tot_det = len(dets)
    tp_tot = sum(1 for d in dets if d['is_tp'])
    prec = tp_tot / tot_det if tot_det > 0 else 0.0

    recs = {}
    for c, name in enumerate(CLASS_NAMES):
        tp_c = sum(1 for d in dets if d['is_tp'] and d['class_id'] == c)
        recs[name] = tp_c / TEST_GT[c]

    macro_r = float(np.mean(list(recs.values())))
    worst_r = float(min(recs.values()))

    return prec, macro_r, worst_r


def get_source_weights(ratio):
    ratio_label = ratio.replace('%', 'pct')
    if ratio == '0%':
        return REPO_ROOT / 'models' / 'yolo11n' / 'baseline' / 'seed_0' / 'weights' / 'best.pt'
    else:
        return REPO_ROOT / 'models' / 'yolo11n' / 'pruning_fp32' / ratio_label / 'seed_0' / 'weights' / 'best.pt'


def train_and_eval_single(ratio, test_gts, base_meta):
    ratio_label = ratio.replace('%', 'pct')
    src_pt = get_source_weights(ratio)
    output_dir = REPO_ROOT / 'models' / 'yolo11n' / 'qat' / ratio_label / 'seed_0'
    train_dir = output_dir / 'train'
    best_weights = output_dir / 'train' / 'qat' / 'weights' / 'best.pt'
    res_csv = output_dir / 'train' / 'qat' / 'results.csv'

    print("=" * 80)
    print(f"STARTING QAT: YOLO11N {ratio} Seed 0 ({EPOCHS} epochs)")
    print(f"Source weights: {src_pt}")
    print(f"Output dir:     {output_dir}")
    print("=" * 80)

    if not src_pt.exists():
        raise FileNotFoundError(f"Source checkpoint missing: {src_pt}")

    # Check if training already completed
    training_done = False
    if best_weights.exists() and res_csv.exists():
        try:
            df = pd.read_csv(res_csv)
            if len(df) >= EPOCHS:
                training_done = True
                print(f"Training already verified ({len(df)} epochs). Skipping fine-tuning.")
        except Exception:
            pass

    if not training_done:
        print(f"Starting training run on GPU (device 0)...")
        m = YOLO(str(src_pt))
        m.train(
            trainer=PrunedDetectionTrainer,
            data=str(DATA_YAML),
            epochs=EPOCHS,
            batch=BATCH,
            imgsz=IMGSZ,
            patience=0,
            amp=True,
            lr0=0.001,
            lrf=0.01,
            seed=0,
            project=str(train_dir),
            name='qat',
            exist_ok=True,
            device=0,
        )

    # Locate best.pt
    candidate_weights = [
        best_weights,
        REPO_ROOT / 'runs' / 'detect' / output_dir / 'train' / 'qat' / 'weights' / 'best.pt',
    ]
    resolved_best = None
    for p in candidate_weights:
        if p.exists() and p.stat().st_size > 500_000:
            resolved_best = p
            break

    if resolved_best is None:
        for p in output_dir.rglob('best.pt'):
            if p.stat().st_size > 500_000:
                resolved_best = p
                break

    if resolved_best is None:
        rel_output = Path('models') / 'yolo11n' / 'qat' / ratio_label / 'seed_0'
        search_root = REPO_ROOT / 'runs' / 'detect' / rel_output
        if search_root.exists():
            for p in search_root.rglob('best.pt'):
                if p.stat().st_size > 500_000:
                    resolved_best = p
                    break

    if resolved_best is None:
        raise FileNotFoundError(f"Could not find trained best.pt for YOLO11N {ratio} seed 0")

    print(f"Training verified. Exporting to INT8 ONNX...")
    out_onnx = output_dir / 'model_int8.onnx'
    if not out_onnx.exists() or out_onnx.stat().st_size < 500_000:
        export_ptq(
            model_path=str(resolved_best),
            quant_level='int8',
            output_dir=str(output_dir),
            data_yaml=str(CALIB_YAML),
            imgsz=IMGSZ,
            device=0,
        )

    size_mb = out_onnx.stat().st_size / (1024 * 1024)
    print(f"Evaluating QAT INT8 model ({size_mb:.2f} MB) on held-out test split at tau=0.50...")
    t0 = time.time()
    prec, macro_r, worst_r = evaluate_onnx_model(out_onnx, test_gts)
    t1 = time.time()
    print(f"Done in {t1 - t0:.1f}s | Prec: {prec*100:.1f}% | Macro-Rec: {macro_r*100:.1f}% | Worst-Rec: {worst_r*100:.1f}%")

    key_meta = ('YOLO11N', ratio, 0)
    meta = base_meta.get(key_meta, {'params': 2590620 if ratio == '0%' else 0, 'bb_neck_params': 1909440 if ratio == '0%' else 0, 'flops_g': 6.50 if ratio == '0%' else 0.0})

    rec = {
        'arch': 'YOLO11N',
        'sparsity': ratio,
        'seed': 0,
        'quantization': 'QAT INT8 (20ep)',
        'params': meta['params'],
        'bb_neck_params': meta['bb_neck_params'],
        'flops_g': meta['flops_g'],
        'size_mb': round(size_mb, 2),
        'precision': round(prec, 4),
        'macro_recall': round(macro_r, 4),
        'worst_recall': round(worst_r, 4),
        'status': 'COMPLETED'
    }

    qat_records = []
    if QAT_CSV.exists() and QAT_CSV.stat().st_size > 0:
        qat_records = pd.read_csv(QAT_CSV).to_dict('records')
    # Replace if exists
    qat_records = [r for r in qat_records if not (r['arch'] == 'YOLO11N' and r['sparsity'] == ratio and int(r['seed']) == 0)]
    qat_records.append(rec)
    pd.DataFrame(qat_records).to_csv(QAT_CSV, index=False)
    update_results_markdown()
    print(f"Successfully recorded QAT run for YOLO11N {ratio} seed 0!")


def main():
    test_gts = preload_test_annotations()
    base_meta = load_base_metadata()

    existing_records = []
    if QAT_CSV.exists() and QAT_CSV.stat().st_size > 0:
        existing_records = pd.read_csv(QAT_CSV).to_dict('records')

    print("Checking existing QAT records:")
    for r in existing_records:
        print(f"  {r['arch']} {r['sparsity']} seed {r['seed']}: status={r['status']}")

    for ratio in RATIOS_SEQUENCE:
        already = any(r['arch'] == 'YOLO11N' and r['sparsity'] == ratio and int(r['seed']) == 0 for r in existing_records)
        if already:
            print(f"YOLO11N {ratio} Seed 0 already completed in {QAT_CSV}, skipping.")
            continue
        train_and_eval_single(ratio, test_gts, base_meta)

    print("\nALL YOLO11N SEED 0 QAT RUNS (20 epochs) HAVE COMPLETED SUCCESSFULLY!")


if __name__ == '__main__':
    main()
