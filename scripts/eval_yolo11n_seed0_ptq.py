"""Evaluate PTQ INT8 for YOLO11N Seed 0 across all pruning ratios on held-out test split at tau=0.50."""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from eval.metrics import load_yolo_labels, match_detections_single_image, CLASS_NAMES
from scripts.evaluate_all_ptq import preload_test_annotations, load_base_metadata, update_results_markdown, PTQ_CSV

DATA_DIR = REPO_ROOT / 'data' / 'processed' / 'RGB' / 'yolo'
TEST_TXT = DATA_DIR / 'test.txt'
TEST_GT = {0: 33, 1: 28, 2: 56, 3: 497}
RATIOS = ['0%', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%']

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

def main():
    print("=" * 70)
    print("Evaluating PTQ INT8 for YOLO11N Seed 0 across all ratios (tau = 0.50)")
    print("=" * 70)

    test_gts = preload_test_annotations()
    base_meta = load_base_metadata()

    existing_records = []
    if PTQ_CSV.exists() and PTQ_CSV.stat().st_size > 0:
        existing_records = pd.read_csv(PTQ_CSV).to_dict('records')

    # Prioritize 60% first, then 10-50, 70-90
    eval_order = ['60%', '10%', '20%', '30%', '40%', '50%', '70%', '80%', '90%', '0%']

    for ratio in eval_order:
        ratio_label = ratio.replace('%', 'pct')
        already = any(r['arch'] == 'YOLO11N' and r['sparsity'] == ratio and int(r['seed']) == 0 for r in existing_records)
        if already:
            print(f"YOLO11N {ratio} Seed 0: already evaluated, skipping.")
            continue

        onnx_path = REPO_ROOT / 'models' / 'yolo11n' / 'joint_int8' / ratio_label / 'seed_0' / 'model_int8.onnx'
        if not onnx_path.exists():
            print(f"ONNX not found: {onnx_path}")
            continue

        size_mb = onnx_path.stat().st_size / (1024 * 1024)
        print(f"Evaluating YOLO11N {ratio} Seed 0 ({size_mb:.2f} MB)...")
        t0 = time.time()
        prec, macro_r, worst_r = evaluate_onnx_model(onnx_path, test_gts)
        t1 = time.time()
        print(f"  -> Done in {t1 - t0:.1f}s | Prec: {prec*100:.1f}% | Macro-Rec: {macro_r*100:.1f}% | Worst-Rec: {worst_r*100:.1f}%")

        key_meta = ('YOLO11N', ratio, 0)
        meta = base_meta.get(key_meta, {'params': 0, 'bb_neck_params': 0, 'flops_g': 0.0})

        rec = {
            'arch': 'YOLO11N',
            'sparsity': ratio,
            'seed': 0,
            'quantization': 'PTQ INT8',
            'params': meta['params'],
            'bb_neck_params': meta['bb_neck_params'],
            'flops_g': meta['flops_g'],
            'size_mb': round(size_mb, 2),
            'precision': round(prec, 4),
            'macro_recall': round(macro_r, 4),
            'worst_recall': round(worst_r, 4),
            'status': 'COMPLETED'
        }
        existing_records.append(rec)
        pd.DataFrame(existing_records).to_csv(PTQ_CSV, index=False)
        update_results_markdown()

    print("YOLO11N Seed 0 PTQ evaluation completed!")

if __name__ == '__main__':
    main()
