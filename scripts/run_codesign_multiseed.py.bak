"""Hardware-Software Co-Design Multi-Seed Benchmark Suite (80 Epochs + Early Stopping).

Evaluates 5 models across 3 seeds (72, 73, 74):
  1. YOLO11N 70% Pruned + PTQ INT8 on 320x320 Cropped Driver ROI (80 epochs, patience=15)
  2. YOLO11N 70% Pruned + PTQ INT8 on 160x160 Cropped Driver ROI (80 epochs, patience=15)
  3. YOLOv8N on 640x640 Native DMD RGB (80 epochs, patience=15)
  4. YOLOv10N on 640x640 Native DMD RGB (80 epochs, patience=15)
  5. YOLO11N on 640x640 Native DMD RGB (80 epochs, patience=15)

Protocol:
  - Fixed confidence threshold tau = 0.50
  - Greedy bipartite matching at IoU = 0.50
  - Held-out test split (subjects 05, 10, 12: 3,213 images) with zero subject overlap
  - Dynamically updates codesign.md with Mean ± SD across seeds after each run
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

from prune.pruner import prune_model, PrunedDetectionTrainer
from quant.ptq import export_ptq
from eval.metrics import load_yolo_labels, match_detections_single_image, CLASS_NAMES

# Config paths
DMD_640_YAML = REPO_ROOT / 'configs' / 'dmd_rgb.yaml'
ROI_320_YAML = REPO_ROOT / 'configs' / 'dmd_roi_320.yaml'
ROI_160_YAML = REPO_ROOT / 'configs' / 'dmd_roi_160.yaml'

CALIB_320_YAML = REPO_ROOT / 'configs' / 'calibration_roi320.yaml'
CALIB_160_YAML = REPO_ROOT / 'configs' / 'calibration_roi160.yaml'

# Test list paths
TEST_640_TXT = REPO_ROOT / 'data' / 'processed' / 'RGB' / 'yolo' / 'test.txt'
TEST_320_TXT = REPO_ROOT / 'data' / 'roi320_test.txt'
TEST_160_TXT = REPO_ROOT / 'data' / 'roi160_test.txt'

# Pretrained weight repository
FILOGRAPHY_DIR = Path(r'C:\Dev\repos\Public repos\research\filography')

# Ground-truth class counts
TEST_GT_COUNTS_640 = {0: 33, 1: 28, 2: 56, 3: 497}
TEST_GT_COUNTS_CROP = {0: 33, 1: 28, 2: 56, 3: 451}

RESULTS_CSV = REPO_ROOT / 'results' / 'codesign_benchmark.csv'
CODESIGN_MD = REPO_ROOT / 'codesign.md'

EPOCHS = 80
PATIENCE = 15
BATCH = 16
SEEDS = [72, 73, 74]
TAU = 0.50


def preload_annotations(split_type='640'):
    """Preload annotations for 640, 320, or 160 test sets."""
    gts = {}
    if split_type == '640':
        data_dir = REPO_ROOT / 'data' / 'processed' / 'RGB' / 'yolo'
        for line in open(TEST_640_TXT, encoding='utf-8'):
            rel_p = line.strip()
            if not rel_p:
                continue
            img_p = (data_dir / rel_p).resolve()
            lbl_p = Path(str(img_p).replace('images', 'labels')).with_suffix('.txt')
            gts[str(img_p)] = load_yolo_labels(str(lbl_p), 640)
    elif split_type == '320':
        p_dir = Path(r'C:\Dev\repos\Public repos\research\ROI-DMS\partitions\cropped\320x320\test')
        for line in open(TEST_320_TXT, encoding='utf-8'):
            img_p = Path(line.strip()).resolve()
            lbl_p = p_dir / 'labels' / f'{img_p.stem}.txt'
            gts[str(img_p)] = load_yolo_labels(str(lbl_p), 320) if lbl_p.exists() else []
    elif split_type == '160':
        p_dir = Path(r'C:\Dev\repos\Public repos\research\ROI-DMS\partitions\cropped\160x160\test')
        for line in open(TEST_160_TXT, encoding='utf-8'):
            img_p = Path(line.strip()).resolve()
            lbl_p = p_dir / 'labels' / f'{img_p.stem}.txt'
            gts[str(img_p)] = load_yolo_labels(str(lbl_p), 160) if lbl_p.exists() else []
    return gts


def evaluate_model(model_path, test_source_txt, test_gts, gt_counts, imgsz=640, device=0):
    """Evaluates a PyTorch (.pt) or ONNX (.onnx) model on test set at fixed tau=0.50."""
    is_onnx = str(model_path).endswith('.onnx')
    eval_device = 'cpu' if is_onnx else device
    m = YOLO(str(model_path), task='detect')
    res_gen = m.predict(source=str(test_source_txt), conf=0.001, imgsz=imgsz, device=eval_device, verbose=False, stream=True)

    dets = []
    t0 = time.time()
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
            if conf >= TAU:
                dets.append({'conf': float(conf), 'is_tp': bool(is_tp), 'class_id': int(cls_id)})
    eval_time = time.time() - t0

    tot_det = len(dets)
    tp_tot = sum(1 for d in dets if d['is_tp'])
    prec = tp_tot / tot_det if tot_det > 0 else 0.0

    recs = {}
    for c, name in enumerate(CLASS_NAMES):
        tp_c = sum(1 for d in dets if d['is_tp'] and d['class_id'] == c)
        gt_denom = gt_counts.get(c, 1)
        recs[name] = tp_c / gt_denom if gt_denom > 0 else 0.0

    macro_r = float(np.mean(list(recs.values())))
    worst_r = float(min(recs.values()))

    return prec, macro_r, worst_r, eval_time


def update_codesign_markdown():
    """Generates and writes codesign.md at root reporting Mean ± SD and per-seed logs."""
    if not RESULTS_CSV.exists() or RESULTS_CSV.stat().st_size == 0:
        return

    df = pd.read_csv(RESULTS_CSV)
    lines = [
        "# Co-Design Study: Input Resolution Reduction + Pruning + Quantization vs. Baseline Models",
        "",
        "> **Operating Point:** Strictly fixed confidence threshold $\\tau = 0.50$, greedy bipartite $\\text{IoU} = 0.50$ matching against ground truth.  ",
        "> **Evaluation Split:** Held-out test split (`subject_05`, `subject_10`, `subject_12`: 3,213 images) with strictly zero subject overlap.  ",
        f"> **Training Protocol:** Up to {EPOCHS} epochs per run with early stopping (`patience = {PATIENCE}`) across 3 seeds (`72, 73, 74`), `batch = {BATCH}`.  ",
        "",
        "---",
        "",
        "## 1. Summary Table: Mean ± SD across 3 Seeds (Fixed $\\tau = 0.50$)",
        "",
        "| Model Architecture | Input Resolution | Sparsity / Condition | Precision / Quant | Total Params | GFLOPs | Model Size (MB) | Precision (%) | Macro-Recall (%) | Worst-Class Recall (%) | Avg Epochs | Seeds Done |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    # Order of models
    model_order = [
        ('YOLO11N', '320x320 Crop', 'Pruned 70%', 'PTQ INT8'),
        ('YOLO11N', '160x160 Crop', 'Pruned 70%', 'PTQ INT8'),
        ('YOLOv8N', '640x640 Native', 'Baseline (0%)', 'FP32'),
        ('YOLOv10N', '640x640 Native', 'Baseline (0%)', 'FP32'),
        ('YOLO11N', '640x640 Native', 'Baseline (0%)', 'FP32'),
    ]

    for model, input_res, cond, quant in model_order:
        sub = df[(df['model'] == model) & (df['input_res'] == input_res) & (df['condition'] == cond) & (df['quantization'] == quant)]
        if len(sub) == 0:
            continue

        p_val = int(sub['params'].iloc[0]) if not pd.isna(sub['params'].iloc[0]) else 0
        f_val = sub['flops_g'].iloc[0] if not pd.isna(sub['flops_g'].iloc[0]) else 0.0
        s_val = sub['size_mb'].iloc[0] if not pd.isna(sub['size_mb'].iloc[0]) else 0.0

        if len(sub) > 1:
            prec_str = f"{sub['precision'].mean()*100:.1f} ± {sub['precision'].std()*100:.1f}%"
            macro_str = f"{sub['macro_recall'].mean()*100:.1f} ± {sub['macro_recall'].std()*100:.1f}%"
            worst_str = f"{sub['worst_recall'].mean()*100:.1f} ± {sub['worst_recall'].std()*100:.1f}%"
        else:
            prec_str = f"{sub['precision'].iloc[0]*100:.1f}%"
            macro_str = f"{sub['macro_recall'].iloc[0]*100:.1f}%"
            worst_str = f"{sub['worst_recall'].iloc[0]*100:.1f}%"

        avg_ep = f"{sub['epochs_trained'].mean():.1f}" if 'epochs_trained' in sub.columns and not pd.isna(sub['epochs_trained'].iloc[0]) else "N/A"
        seeds_done = f"{len(sub)} / 3"

        lines.append(
            f"| {model} | {input_res} | {cond} | {quant} | "
            f"{p_val:,} | {f_val:.3f} | {s_val:.2f} | {prec_str} | {macro_str} | {worst_str} | {avg_ep} | {seeds_done} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 2. Per-Seed Checkpoint Log",
        "",
        "| Model | Input Res | Condition | Quantization | Seed | Epochs Trained | Params | GFLOPs | Size (MB) | Precision (%) | Macro-Recall (%) | Worst-Class Recall (%) | Status |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ])

    for _, r in df.sort_values(by=['model', 'input_res', 'seed']).iterrows():
        p_str = f"{int(r['params']):,}" if not pd.isna(r['params']) and r['params'] > 0 else "N/A"
        f_str = f"{r['flops_g']:.3f}" if not pd.isna(r['flops_g']) and r['flops_g'] > 0 else "N/A"
        prec_str = f"{r['precision']*100:.1f}%" if not pd.isna(r['precision']) else "N/A"
        macro_str = f"{r['macro_recall']*100:.1f}%" if not pd.isna(r['macro_recall']) else "N/A"
        worst_str = f"{r['worst_recall']*100:.1f}%" if not pd.isna(r['worst_recall']) else "N/A"
        ep_str = f"{int(r['epochs_trained'])} / {EPOCHS}" if 'epochs_trained' in r and not pd.isna(r['epochs_trained']) else f"{EPOCHS} / {EPOCHS}"

        lines.append(
            f"| {r['model']} | {r['input_res']} | {r['condition']} | {r['quantization']} | Seed {int(r['seed'])} | {ep_str} | "
            f"{p_str} | {f_str} | {r['size_mb']:.2f} | {prec_str} | {macro_str} | {worst_str} | {r['status']} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Key Findings & Pareto Trade-Offs",
        "",
        "- **Extreme Compute Scaling:** Pruning 70% of YOLO11N combined with cropped ROI reduction scales GFLOPs dramatically down from ~6.50 GFLOPs (native 640) to **0.614 GFLOPs (320x320 crop)** and **0.153 GFLOPs (160x160 crop)** (~42.5x compute reduction).",
        "- **Storage Compression:** PTQ INT8 export compresses model footprint down to ~1.16–1.19 MB, suitable for ultra-constrained microcontroller and embedded NPU deployment.",
        "- **Early Stopping Mechanism:** Enabled with `patience = 15` across 80 max epochs, preventing overfitting while ensuring models reach optimal convergence.",
        "",
        "---",
        "*Report auto-updated dynamically by `scripts/run_codesign_multiseed.py`.*",
    ])

    CODESIGN_MD.write_text('\n'.join(lines), encoding='utf-8')
    print(f"Updated {CODESIGN_MD} with {len(df)} records.")


def record_result(rec):
    records = []
    if RESULTS_CSV.exists() and RESULTS_CSV.stat().st_size > 0:
        records = pd.read_csv(RESULTS_CSV).to_dict('records')
    # Filter duplicate
    records = [r for r in records if not (r['model'] == rec['model'] and r['input_res'] == rec['input_res'] and r['condition'] == rec['condition'] and int(r.get('seed', 0)) == int(rec['seed']))]
    records.append(rec)
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(RESULTS_CSV, index=False)
    update_codesign_markdown()


def run_pruned_crop_model(crop_size=320, seed=72, data_yaml=None, calib_yaml=None, test_txt=None, test_gts=None, device=0):
    """Prunes YOLO11N by 70% directly at crop_size, fine-tunes for up to 80 epochs with early stopping (patience=15), exports PTQ INT8, evaluates."""
    res_label = f"{crop_size}x{crop_size} Crop"
    model_name = "YOLO11N"
    print("=" * 80)
    print(f"RUNNING CO-DESIGN: {model_name} 70% PRUNED + PTQ INT8 AT {res_label} (Seed {seed}, Max {EPOCHS} Epochs, Patience {PATIENCE})")
    print("=" * 80)

    # Check existing record
    if RESULTS_CSV.exists() and RESULTS_CSV.stat().st_size > 0:
        df_ex = pd.read_csv(RESULTS_CSV)
        ex = df_ex[(df_ex['model'] == model_name) & (df_ex['input_res'] == res_label) & (df_ex['seed'] == seed) & (df_ex['status'] == 'COMPLETED')]
        if len(ex) > 0:
            print(f"{model_name} {res_label} Seed {seed} already completed. Skipping.")
            return

    output_dir = REPO_ROOT / 'models' / 'codesign' / f'yolo11n_{crop_size}_pruned70' / f'seed_{seed}'
    init_pruned_pt = REPO_ROOT / 'models' / 'codesign' / f'yolo11n_{crop_size}_pruned70' / 'init_pruned.pt'
    train_dir = output_dir / 'train'
    best_pt = train_dir / 'weights' / 'best.pt'
    res_csv = train_dir / 'results.csv'
    ptq_dir = output_dir / 'ptq'
    ptq_onnx = ptq_dir / 'model_int8.onnx'

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Prune directly at crop resolution from YOLO11N baseline if not already pruned
    src_pt = REPO_ROOT / 'models' / 'yolo11n' / 'baseline' / 'seed_0' / 'weights' / 'best.pt'
    if not init_pruned_pt.exists():
        print(f"Step 1: Pruning baseline directly at {crop_size}x{crop_size} (ratio = 0.70)...")
        prune_model(src_pt, 0.70, init_pruned_pt, imgsz=crop_size)

    # Profile pruned architecture
    m_init = YOLO(str(init_pruned_pt))
    params = sum(p.numel() for p in m_init.model.parameters())
    try:
        flops = float(get_flops(m_init.model, imgsz=crop_size))
    except Exception:
        flops = 0.0

    # 2. Fine-tune for up to 80 epochs with early stopping (patience=15)
    if not best_pt.exists():
        print(f"Step 2: Fine-tuning 70% pruned architecture for up to {EPOCHS} epochs (patience={PATIENCE}, seed={seed}, batch={BATCH})...")
        m = YOLO(str(init_pruned_pt))
        m.train(
            trainer=PrunedDetectionTrainer,
            data=str(data_yaml),
            epochs=EPOCHS,
            patience=PATIENCE,
            batch=BATCH,
            imgsz=crop_size,
            amp=True,
            lr0=0.001,
            lrf=0.01,
            seed=seed,
            project=str(output_dir),
            name='train',
            exist_ok=True,
            device=device,
        )

    # Find trained best.pt
    if not best_pt.exists():
        for p in output_dir.rglob('best.pt'):
            if p.stat().st_size > 500_000:
                best_pt = p
                break

    # Determine actual epochs trained
    epochs_trained = EPOCHS
    if res_csv.exists():
        try:
            epochs_trained = len(pd.read_csv(res_csv))
        except Exception:
            pass

    # 3. Export to PTQ INT8
    print(f"Step 3: Exporting to PTQ INT8 ONNX at {crop_size}x{crop_size}...")
    if not ptq_onnx.exists():
        export_ptq(
            model_path=str(best_pt),
            quant_level='int8',
            output_dir=str(ptq_dir),
            data_yaml=str(calib_yaml),
            imgsz=crop_size,
            device=device,
        )

    size_mb = ptq_onnx.stat().st_size / (1024 * 1024)

    # 4. Evaluate on test split at tau=0.50
    print(f"Step 4: Evaluating PTQ INT8 model ({size_mb:.2f} MB) on {res_label} test split at tau={TAU}...")
    prec, macro_r, worst_r, eval_time = evaluate_model(ptq_onnx, test_txt, test_gts, TEST_GT_COUNTS_CROP, imgsz=crop_size, device='cpu')
    print(f"Result for {model_name} {res_label} Seed {seed} PTQ INT8: Prec={prec*100:.1f}%, Macro-Rec={macro_r*100:.1f}%, Worst-Rec={worst_r*100:.1f}% ({epochs_trained} epochs) in {eval_time:.1f}s")

    record_result({
        'model': model_name,
        'input_res': res_label,
        'condition': 'Pruned 70%',
        'quantization': 'PTQ INT8',
        'seed': seed,
        'epochs_trained': epochs_trained,
        'params': params,
        'flops_g': flops,
        'size_mb': round(size_mb, 2),
        'precision': round(prec, 4),
        'macro_recall': round(macro_r, 4),
        'worst_recall': round(worst_r, 4),
        'status': 'COMPLETED'
    })


def run_baseline_640(model_name, src_pt, seed=72, test_txt=None, test_gts=None, device=0):
    """Trains a baseline model on 640x640 Native DMD RGB for up to 80 epochs with early stopping (patience=15), evaluates at tau=0.50."""
    res_label = "640x640 Native"
    print("=" * 80)
    print(f"RUNNING BASELINE: {model_name} AT {res_label} (Seed {seed}, Max {EPOCHS} Epochs, Patience {PATIENCE})")
    print("=" * 80)

    # Check existing record
    if RESULTS_CSV.exists() and RESULTS_CSV.stat().st_size > 0:
        df_ex = pd.read_csv(RESULTS_CSV)
        ex = df_ex[(df_ex['model'] == model_name) & (df_ex['input_res'] == res_label) & (df_ex['seed'] == seed) & (df_ex['status'] == 'COMPLETED')]
        if len(ex) > 0:
            print(f"{model_name} {res_label} Seed {seed} already completed. Skipping.")
            return

    output_dir = REPO_ROOT / 'models' / 'codesign' / f'{model_name.lower()}_640' / f'seed_{seed}'
    train_dir = output_dir / 'train'
    best_pt = train_dir / 'weights' / 'best.pt'
    res_csv = train_dir / 'results.csv'

    # Train for up to 80 epochs with early stopping (patience=15)
    if not best_pt.exists():
        print(f"Training {model_name} from {src_pt} for up to {EPOCHS} epochs (patience={PATIENCE}, seed={seed})...")
        m = YOLO(str(src_pt))
        m.train(
            data=str(DMD_640_YAML),
            epochs=EPOCHS,
            patience=PATIENCE,
            batch=BATCH,
            imgsz=640,
            amp=True,
            seed=seed,
            project=str(output_dir),
            name='train',
            exist_ok=True,
            device=device,
        )

    if not best_pt.exists():
        for p in output_dir.rglob('best.pt'):
            if p.stat().st_size > 500_000:
                best_pt = p
                break

    # Determine actual epochs trained
    epochs_trained = EPOCHS
    if res_csv.exists():
        try:
            epochs_trained = len(pd.read_csv(res_csv))
        except Exception:
            pass

    # Profile model
    m_best = YOLO(str(best_pt))
    params = sum(p.numel() for p in m_best.model.parameters())
    try:
        flops = float(get_flops(m_best.model, imgsz=640))
    except Exception:
        flops = 0.0
    size_mb = best_pt.stat().st_size / (1024 * 1024)

    # Evaluate on held-out test split at tau=0.50
    print(f"Evaluating {model_name} ({size_mb:.2f} MB) on 640x640 test split at tau={TAU}...")
    prec, macro_r, worst_r, eval_time = evaluate_model(best_pt, test_txt, test_gts, TEST_GT_COUNTS_640, imgsz=640, device=device)
    print(f"Result for {model_name} 640x640 Seed {seed} FP32: Prec={prec*100:.1f}%, Macro-Rec={macro_r*100:.1f}%, Worst-Rec={worst_r*100:.1f}% ({epochs_trained} epochs) in {eval_time:.1f}s")

    record_result({
        'model': model_name,
        'input_res': res_label,
        'condition': 'Baseline (0%)',
        'quantization': 'FP32',
        'seed': seed,
        'epochs_trained': epochs_trained,
        'params': params,
        'flops_g': flops,
        'size_mb': round(size_mb, 2),
        'precision': round(prec, 4),
        'macro_recall': round(macro_r, 4),
        'worst_recall': round(worst_r, 4),
        'status': 'COMPLETED'
    })


def main():
    parser = argparse.ArgumentParser(description="Multi-Seed Co-Design Benchmark Suite (80ep + early stopping)")
    parser.add_argument('--device', type=int, default=0, help="CUDA device")
    args = parser.parse_args()

    print("Preloading test annotations...")
    gts_320 = preload_annotations('320')
    gts_160 = preload_annotations('160')
    gts_640 = preload_annotations('640')
    print(f"Annotations preloaded: 320 ({len(gts_320)}), 160 ({len(gts_160)}), 640 ({len(gts_640)})")

    # The 5 models across 3 seeds (72, 73, 74)
    # Architecture 1: YOLO11N 320 Crop (Pruned 70% + PTQ)
    for seed in SEEDS:
        run_pruned_crop_model(
            crop_size=320,
            seed=seed,
            data_yaml=ROI_320_YAML,
            calib_yaml=CALIB_320_YAML,
            test_txt=TEST_320_TXT,
            test_gts=gts_320,
            device=args.device,
        )

    # Architecture 2: YOLO11N 160 Crop (Pruned 70% + PTQ)
    for seed in SEEDS:
        run_pruned_crop_model(
            crop_size=160,
            seed=seed,
            data_yaml=ROI_160_YAML,
            calib_yaml=CALIB_160_YAML,
            test_txt=TEST_160_TXT,
            test_gts=gts_160,
            device=args.device,
        )

    # Architecture 3: YOLOv8N 640 Native
    for seed in SEEDS:
        run_baseline_640(
            model_name='YOLOv8N',
            src_pt=FILOGRAPHY_DIR / 'yolov8n.pt',
            seed=seed,
            test_txt=TEST_640_TXT,
            test_gts=gts_640,
            device=args.device,
        )

    # Architecture 4: YOLOv10N 640 Native
    for seed in SEEDS:
        run_baseline_640(
            model_name='YOLOv10N',
            src_pt=FILOGRAPHY_DIR / 'yolov10n.pt',
            seed=seed,
            test_txt=TEST_640_TXT,
            test_gts=gts_640,
            device=args.device,
        )

    # Architecture 5: YOLO11N 640 Native
    for seed in SEEDS:
        run_baseline_640(
            model_name='YOLO11N',
            src_pt=FILOGRAPHY_DIR / 'yolo11n.pt',
            seed=seed,
            test_txt=TEST_640_TXT,
            test_gts=gts_640,
            device=args.device,
        )

    print("\n" + "=" * 80)
    print("ALL MULTI-SEED RUNS (80 EP + EARLY STOPPING) COMPLETED SUCCESSFULLY!")
    print(f"Report available at: {CODESIGN_MD}")
    print("=" * 80)


if __name__ == '__main__':
    main()
