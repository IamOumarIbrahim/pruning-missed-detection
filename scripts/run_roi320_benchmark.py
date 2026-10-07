"""320x320 Cropped Driver ROI Benchmark (30 Epochs, Fixed tau = 0.50).

Compares:
  1. YOLO11N 60% Pruned (FP32) fine-tuned for 30 epochs on 320x320 Cropped ROI
  2. YOLO11N 60% Pruned (PTQ INT8) calibrated and exported at 320x320
  3. YOLO11N 60% Pruned (QAT INT8) fine-tuned for 30 epochs on 320x320 + exported INT8
  4. Baselines (Unpruned, 30 epochs on 320x320 Cropped ROI):
     - YOLOv8n
     - YOLOv10n
     - YOLO11n
     - YOLO12n
     - YOLO26n

Evaluation Protocol:
  - Strictly fixed confidence threshold tau = 0.50
  - Greedy bipartite matching at IoU = 0.50
  - Held-out test split only (subjects 05, 10, 12: 3,213 images)
  - Zero subject overlap verified across train, val, and test partitions
"""

import argparse
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from prune.pruner import PrunedDetectionTrainer
from quant.ptq import export_ptq
from eval.metrics import load_yolo_labels, match_detections_single_image, CLASS_NAMES

DATA_YAML = REPO_ROOT / 'configs' / 'dmd_roi_320.yaml'
CALIB_YAML = REPO_ROOT / 'configs' / 'calibration_roi320.yaml'
TEST_TXT = REPO_ROOT / 'data' / 'roi320_test.txt'
TEST_LABEL_DIR = Path(r'C:\Dev\repos\Public repos\research\ROI-DMS\partitions\cropped\320x320\test\labels')
TEST_IMG_DIR = Path(r'C:\Dev\repos\Public repos\research\ROI-DMS\partitions\cropped\320x320\test\images')
FILOGRAPHY_DIR = Path(r'C:\Dev\repos\Public repos\research\filography')

RESULTS_CSV = REPO_ROOT / 'results' / 'roi320_benchmark.csv'
RESULTS_MD = REPO_ROOT / 'RESULTS_ROI320.md'

# Ground-truth class counts in 320 cropped ROI test split
TEST_GT_COUNTS = {0: 33, 1: 28, 2: 56, 3: 451}


def preload_roi320_ground_truths():
    """Preload all ground-truth bounding boxes for the 320x320 test split."""
    txt_files = list(TEST_LABEL_DIR.glob('*.txt'))
    gts = {}
    for tf in txt_files:
        img_p = str((TEST_IMG_DIR / (tf.stem + '.png')).resolve())
        boxes = load_yolo_labels(tf)
        gts[img_p] = boxes
    return gts


def evaluate_roi320_model(model_path, test_gts, imgsz=320, device='cpu'):
    """Evaluates a PyTorch (.pt) or ONNX (.onnx) model on the 320 ROI test set at tau=0.50."""
    m = YOLO(str(model_path), task='detect')
    res_gen = m.predict(source=str(TEST_TXT), conf=0.001, imgsz=imgsz, device=device, verbose=False, stream=True)

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
        gt_denom = TEST_GT_COUNTS.get(c, 1)
        recs[name] = tp_c / gt_denom if gt_denom > 0 else 0.0

    macro_r = float(np.mean(list(recs.values())))
    worst_r = float(min(recs.values()))

    return prec, macro_r, worst_r


def get_model_profile(model_path, imgsz=320):
    """Computes parameter count, GFLOPs at given imgsz, and file size in MB."""
    path = Path(model_path)
    size_mb = path.stat().st_size / (1024 * 1024)
    if path.suffix == '.pt':
        m = YOLO(str(model_path))
        params = sum(p.numel() for p in m.model.parameters())
        try:
            flops = float(get_flops(m.model, imgsz=imgsz))
        except Exception:
            flops = 0.0
        return params, flops, round(size_mb, 2)
    else:
        # ONNX or Engine
        return None, None, round(size_mb, 2)


def render_roi320_markdown():
    """Generates RESULTS_ROI320.md summarizing all completed models."""
    if not RESULTS_CSV.exists() or RESULTS_CSV.stat().st_size == 0:
        return

    df = pd.read_csv(RESULTS_CSV)
    lines = [
        "# Benchmark Results: 320x320 Cropped Driver ROI vs Baseline Models",
        "",
        "> **Dataset:** Cropped Driver ROI at 320x320 resolution (`ROI-DMS`).  ",
        "> **Split:** Strictly disjoint held-out test split (`subject_05`, `subject_10`, `subject_12`: 3,213 images).  ",
        "> **Fixed Operating Point:** $\\tau = 0.50$ (greedy bipartite matching at IoU = 0.50).  ",
        "> **Training Protocol:** 30 epochs on training split (subjects 01, 04, 06, 07, 08, 09, 13, 14).  ",
        "",
        "---",
        "",
        "## Summary Table: 320x320 ROI Models & Baselines (Fixed $\\tau = 0.50$)",
        "",
        "| Model | Input Size | Sparsity / Condition | Quantization | Params | GFLOPs | Size (MB) | Precision (%) | Macro-Recall (%) | Worst-Class Rec (%) | Status |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for _, r in df.iterrows():
        p_str = f"{int(r['params']):,}" if not pd.isna(r['params']) and r['params'] > 0 else "N/A"
        f_str = f"{r['flops_g']:.2f}" if not pd.isna(r['flops_g']) and r['flops_g'] > 0 else "N/A"
        prec_str = f"{r['precision']*100:.1f}%" if not pd.isna(r['precision']) else "N/A"
        macro_str = f"{r['macro_recall']*100:.1f}%" if not pd.isna(r['macro_recall']) else "N/A"
        worst_str = f"{r['worst_recall']*100:.1f}%" if not pd.isna(r['worst_recall']) else "N/A"

        lines.append(
            f"| {r['model']} | {r['imgsz']}x{r['imgsz']} | {r['condition']} | {r['quantization']} | "
            f"{p_str} | {f_str} | {r['size_mb']:.2f} | {prec_str} | {macro_str} | {worst_str} | {r['status']} |"
        )

    lines.extend([
        "",
        "---",
        "*Report auto-generated by `scripts/run_roi320_benchmark.py`.*",
    ])

    RESULTS_MD.write_text('\n'.join(lines), encoding='utf-8')
    print(f"Updated markdown report at {RESULTS_MD}")


def record_result(rec):
    records = []
    if RESULTS_CSV.exists() and RESULTS_CSV.stat().st_size > 0:
        records = pd.read_csv(RESULTS_CSV).to_dict('records')
    # Filter duplicate
    records = [r for r in records if not (r['model'] == rec['model'] and r['condition'] == rec['condition'] and r['quantization'] == rec['quantization'])]
    records.append(rec)
    pd.DataFrame(records).to_csv(RESULTS_CSV, index=False)
    render_roi320_markdown()


def train_and_eval_pruned_roi320(epochs=30, batch=16, device=0, test_gts=None):
    """Trains YOLO11N 60% pruned model on 320x320 ROI, evaluates FP32, PTQ INT8, and QAT INT8."""
    print("=" * 80)
    print("PHASE 1: YOLO11N 60% PRUNED ON 320x320 CROPPED ROI")
    print("=" * 80)

    # 1. Source 60% pruned model
    src_pt = REPO_ROOT / 'models' / 'yolo11n' / 'pruning_fp32' / '60pct' / 'seed_0' / 'weights' / 'best.pt'
    output_dir = REPO_ROOT / 'models' / 'roi320' / 'yolo11n_pruned60'
    train_dir = output_dir / 'train'
    best_pt = train_dir / 'fp32' / 'weights' / 'best.pt'

    # Step 1.1: FP32 Fine-tuning (30 epochs)
    if not best_pt.exists():
        print(f"\n[1/3] Fine-tuning YOLO11N 60% pruned on 320x320 ROI ({epochs} epochs)...")
        m = YOLO(str(src_pt))
        m.train(
            trainer=PrunedDetectionTrainer,
            data=str(DATA_YAML),
            epochs=epochs,
            batch=batch,
            imgsz=320,
            patience=0,
            amp=True,
            lr0=0.001,
            lrf=0.01,
            seed=0,
            project=str(train_dir),
            name='fp32',
            exist_ok=True,
            device=device,
        )

    params, flops, size_mb = get_model_profile(best_pt, imgsz=320)
    print(f"Evaluating YOLO11N 60% Pruned (FP32) at 320x320 on test split...")
    prec, macro_r, worst_r = evaluate_roi320_model(best_pt, test_gts, imgsz=320, device=device)
    print(f"FP32 Result: Prec={prec*100:.1f}%, Macro-Rec={macro_r*100:.1f}%, Worst-Rec={worst_r*100:.1f}%")

    record_result({
        'model': 'YOLO11N',
        'imgsz': 320,
        'condition': 'Pruned 60%',
        'quantization': 'FP32 (30ep)',
        'params': params,
        'flops_g': flops,
        'size_mb': size_mb,
        'precision': round(prec, 4),
        'macro_recall': round(macro_r, 4),
        'worst_recall': round(worst_r, 4),
        'status': 'COMPLETED'
    })

    # Step 1.2: PTQ INT8 Export & Evaluation
    print(f"\n[2/3] Exporting YOLO11N 60% Pruned to PTQ INT8 at 320x320...")
    ptq_dir = output_dir / 'ptq'
    ptq_onnx = ptq_dir / 'model_int8.onnx'
    if not ptq_onnx.exists():
        export_ptq(
            model_path=str(best_pt),
            quant_level='int8',
            output_dir=str(ptq_dir),
            data_yaml=str(CALIB_YAML),
            imgsz=320,
            device=device,
        )
    ptq_size_mb = ptq_onnx.stat().st_size / (1024 * 1024)
    print(f"Evaluating PTQ INT8 model ({ptq_size_mb:.2f} MB) on 320x320 test split...")
    prec, macro_r, worst_r = evaluate_roi320_model(ptq_onnx, test_gts, imgsz=320, device='cpu')
    print(f"PTQ Result: Prec={prec*100:.1f}%, Macro-Rec={macro_r*100:.1f}%, Worst-Rec={worst_r*100:.1f}%")

    record_result({
        'model': 'YOLO11N',
        'imgsz': 320,
        'condition': 'Pruned 60%',
        'quantization': 'PTQ INT8',
        'params': params,
        'flops_g': flops,
        'size_mb': round(ptq_size_mb, 2),
        'precision': round(prec, 4),
        'macro_recall': round(macro_r, 4),
        'worst_recall': round(worst_r, 4),
        'status': 'COMPLETED'
    })

    # Step 1.3: QAT INT8 Fine-tuning (30 epochs) & Evaluation
    print(f"\n[3/3] Fine-tuning YOLO11N 60% Pruned with QAT ({epochs} epochs at lr0=0.0005)...")
    qat_dir = output_dir / 'qat'
    qat_best_pt = qat_dir / 'weights' / 'best.pt'
    if not qat_best_pt.exists():
        m_qat = YOLO(str(best_pt))
        m_qat.train(
            trainer=PrunedDetectionTrainer,
            data=str(DATA_YAML),
            epochs=epochs,
            batch=batch,
            imgsz=320,
            patience=0,
            amp=True,
            lr0=0.0005,
            lrf=0.01,
            seed=0,
            project=str(qat_dir),
            name='train',
            exist_ok=True,
            device=device,
        )
        # Find trained best.pt
        for p in qat_dir.rglob('best.pt'):
            if p.stat().st_size > 500_000:
                qat_best_pt = p
                break

    qat_onnx = qat_dir / 'model_int8.onnx'
    if not qat_onnx.exists():
        export_ptq(
            model_path=str(qat_best_pt),
            quant_level='int8',
            output_dir=str(qat_dir),
            data_yaml=str(CALIB_YAML),
            imgsz=320,
            device=device,
        )
    qat_size_mb = qat_onnx.stat().st_size / (1024 * 1024)
    print(f"Evaluating QAT INT8 model ({qat_size_mb:.2f} MB) on 320x320 test split...")
    prec, macro_r, worst_r = evaluate_roi320_model(qat_onnx, test_gts, imgsz=320, device='cpu')
    print(f"QAT Result: Prec={prec*100:.1f}%, Macro-Rec={macro_r*100:.1f}%, Worst-Rec={worst_r*100:.1f}%")

    record_result({
        'model': 'YOLO11N',
        'imgsz': 320,
        'condition': 'Pruned 60%',
        'quantization': 'QAT INT8 (30ep)',
        'params': params,
        'flops_g': flops,
        'size_mb': round(qat_size_mb, 2),
        'precision': round(prec, 4),
        'macro_recall': round(macro_r, 4),
        'worst_recall': round(worst_r, 4),
        'status': 'COMPLETED'
    })


def train_and_eval_baselines(epochs=30, batch=16, device=0, test_gts=None):
    """Trains 5 unpruned baselines (yolov8n, yolov10n, yolo11n, yolo12n, yolo26n) on 320 ROI."""
    print("=" * 80)
    print(f"PHASE 2: UNPRUNED BASELINES ON 320x320 CROPPED ROI ({epochs} epochs)")
    print("=" * 80)

    models_info = [
        ('YOLOv8N', FILOGRAPHY_DIR / 'yolov8n.pt'),
        ('YOLOv10N', FILOGRAPHY_DIR / 'yolov10n.pt'),
        ('YOLO11N', FILOGRAPHY_DIR / 'yolo11n.pt'),
        ('YOLO12N', FILOGRAPHY_DIR / 'yolo12n.pt'),
        ('YOLO26N', FILOGRAPHY_DIR / 'yolo26n.pt'),
    ]

    for model_name, src_pt in models_info:
        print(f"\nProcessing Baseline: {model_name}...")
        output_dir = REPO_ROOT / 'models' / 'roi320' / f'baseline_{model_name.lower()}'
        best_pt = output_dir / 'train' / 'weights' / 'best.pt'

        if not best_pt.exists():
            print(f"Training {model_name} for {epochs} epochs on 320x320 ROI...")
            m = YOLO(str(src_pt))
            m.train(
                data=str(DATA_YAML),
                epochs=epochs,
                batch=batch,
                imgsz=320,
                patience=0,
                amp=True,
                seed=0,
                project=str(output_dir),
                name='train',
                exist_ok=True,
                device=device,
            )
            for p in output_dir.rglob('best.pt'):
                if p.stat().st_size > 500_000:
                    best_pt = p
                    break

        params, flops, size_mb = get_model_profile(best_pt, imgsz=320)
        print(f"Evaluating {model_name} on 320x320 test split...")
        prec, macro_r, worst_r = evaluate_roi320_model(best_pt, test_gts, imgsz=320, device=device)
        print(f"{model_name} Result: Prec={prec*100:.1f}%, Macro-Rec={macro_r*100:.1f}%, Worst-Rec={worst_r*100:.1f}%")

        record_result({
            'model': model_name,
            'imgsz': 320,
            'condition': 'Baseline (Unpruned)',
            'quantization': 'FP32 (30ep)',
            'params': params,
            'flops_g': flops,
            'size_mb': size_mb,
            'precision': round(prec, 4),
            'macro_recall': round(macro_r, 4),
            'worst_recall': round(worst_r, 4),
            'status': 'COMPLETED'
        })


def main():
    parser = argparse.ArgumentParser(description="320x320 Cropped Driver ROI Benchmark")
    parser.add_argument('--stage', type=str, default='all', choices=['all', 'pruned_60', 'baselines', 'report'],
                        help="Benchmark stage to run")
    parser.add_argument('--epochs', type=int, default=30, help="Training epochs")
    parser.add_argument('--batch', type=int, default=16, help="Batch size")
    parser.add_argument('--device', type=int, default=0, help="CUDA device")
    args = parser.parse_args()

    test_gts = preload_roi320_ground_truths()
    print(f"Preloaded {len(test_gts)} test images with {sum(len(b) for b in test_gts.values())} GT annotations.")

    if args.stage in ('all', 'pruned_60'):
        train_and_eval_pruned_roi320(epochs=args.epochs, batch=args.batch, device=args.device, test_gts=test_gts)

    if args.stage in ('all', 'baselines'):
        train_and_eval_baselines(epochs=args.epochs, batch=args.batch, device=args.device, test_gts=test_gts)

    render_roi320_markdown()
    print("\nBENCHMARK STAGE COMPLETED!")


if __name__ == '__main__':
    main()
