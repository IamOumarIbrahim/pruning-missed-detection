"""Quantization-Aware Training (QAT) for 60% pruned YOLO11N & YOLO26N (20 epochs).

Trains 6 runs sequentially on GPU:
  - YOLO11N 60% (Seeds 0, 1, 2)
  - YOLO26N 60% (Seeds 0, 1, 2)
Exports to calibrated INT8 ONNX, evaluates on held-out test split at fixed tau = 0.50,
and updates RESULTS_FIXED_TAU05.md.
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

DATA_DIR = REPO_ROOT / 'data' / 'processed' / 'RGB' / 'yolo'
DATA_YAML = REPO_ROOT / 'configs' / 'dmd_rgb.yaml'
CALIB_YAML = REPO_ROOT / 'configs' / 'calibration.yaml'
TEST_TXT = DATA_DIR / 'test.txt'
RESULTS_DIR = REPO_ROOT / 'results'
RESULTS_MD = REPO_ROOT / 'RESULTS_FIXED_TAU05.md'
QAT_CSV = RESULTS_DIR / 'fixed_tau05_qat.csv'
PTQ_CSV = RESULTS_DIR / 'fixed_tau05_ptq.csv'

MODELS = ['yolo11n', 'yolo26n']
SEEDS = [0, 1, 2]
EPOCHS = 20
IMGSZ = 640
BATCH = 16
TEST_GT = {0: 33, 1: 28, 2: 56, 3: 497}


def preload_test_annotations():
    gts = {}
    for line in open(TEST_TXT):
        rel_p = line.strip()
        img_p = (DATA_DIR / rel_p).resolve()
        lbl_p = Path(str(img_p).replace('images', 'labels')).with_suffix('.txt')
        gts[str(img_p)] = load_yolo_labels(str(lbl_p), 640)
    return gts


def load_base_metadata():
    csv_further = RESULTS_DIR / 'fixed_tau05_checkpoints_further.csv'
    records = []
    if csv_further.exists():
        records.extend(pd.read_csv(csv_further).to_dict('records'))
    
    meta = {}
    for r in records:
        key = (r['arch'].upper(), r['sparsity'], int(r['seed']))
        meta[key] = {
            'params': int(r['params']),
            'bb_neck_params': int(r.get('bb_neck_params', 0)) if not pd.isna(r.get('bb_neck_params', 0)) else 0,
            'flops_g': float(r['flops_g']) if not pd.isna(r.get('flops_g', 0)) else 0.0,
        }
    return meta


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


def update_results_markdown():
    """Reread base FP32 results, PTQ results, and QAT results, then format RESULTS_FIXED_TAU05.md."""
    # 1. Load FP32 records
    csv_0to50 = RESULTS_DIR / 'fixed_tau05_checkpoints_0to50.csv'
    csv_further = RESULTS_DIR / 'fixed_tau05_checkpoints_further.csv'
    fp32_records = []
    if csv_0to50.exists():
        fp32_records.extend(pd.read_csv(csv_0to50).to_dict('records'))
    if csv_further.exists():
        fp32_records.extend(pd.read_csv(csv_further).to_dict('records'))
    df_fp32 = pd.DataFrame(fp32_records)

    # 2. Load PTQ records
    ptq_records = []
    if PTQ_CSV.exists() and PTQ_CSV.stat().st_size > 0:
        ptq_records = pd.read_csv(PTQ_CSV).to_dict('records')
    df_ptq = pd.DataFrame(ptq_records)

    # 3. Load QAT records
    qat_records = []
    if QAT_CSV.exists() and QAT_CSV.stat().st_size > 0:
        qat_records = pd.read_csv(QAT_CSV).to_dict('records')
    df_qat = pd.DataFrame(qat_records)

    sparsity_order = ['0%', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%']

    lines = [
        "# Benchmark Results: Fixed Operating Point at $\\tau = 0.50$ (Ablated by Quantization & Pruning)",
        "",
        "> **Protocol:** One fixed confidence threshold $\\tau = 0.50$ across every model, every seed, every condition.  ",
        "> **Split:** Held-out test split (`subject_05`, `subject_10`, `subject_12`) only. Greedy bipartite IoU=0.50 matching.  ",
        "> **Report Structure:** Split by **No quantization (FP32/FP16)**, **PTQ (INT8)**, and **QAT (INT8)**.  ",
        "",
        "---",
        "",
        "## Section 1: No Quantization (FP32 Baseline & Pruned Models)",
        "",
        "### 1.1 Summary Table: Mean ± SD across 3 Seeds",
        "",
        "| Architecture | Sparsity | Total Params | BB+Neck Params | GFLOPs | Trained Size (MB) | Precision (%) | Macro-Recall (%) | Worst-Class Rec (%) | Achieved Param Red (%) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for arch in ['YOLO11N', 'YOLO26N']:
        sub_arch = df_fp32[df_fp32['arch'] == arch]
        base_sub = sub_arch[sub_arch['sparsity'] == '0%']
        base_params = base_sub['params'].iloc[0] if len(base_sub) > 0 else 1

        for sp in sparsity_order:
            sub = sub_arch[sub_arch['sparsity'] == sp]
            if len(sub) == 0:
                continue

            p_mean = int(sub['params'].mean())
            bb_mean = int(sub['bb_neck_params'].mean()) if 'bb_neck_params' in sub.columns else 0
            f_mean = sub['flops_g'].mean()
            s_mean = sub['size_mb'].mean()

            prec_str = f"{sub['precision'].mean()*100:.1f} ± {sub['precision'].std()*100:.1f}%" if len(sub) > 1 else f"{sub['precision'].iloc[0]*100:.1f}%"
            macro_str = f"{sub['macro_recall'].mean()*100:.1f} ± {sub['macro_recall'].std()*100:.1f}%" if len(sub) > 1 else f"{sub['macro_recall'].iloc[0]*100:.1f}%"
            worst_str = f"{sub['worst_recall'].mean()*100:.1f} ± {sub['worst_recall'].std()*100:.1f}%" if len(sub) > 1 else f"{sub['worst_recall'].iloc[0]*100:.1f}%"

            achieved_red = (1 - p_mean / base_params) * 100.0 if base_params > 0 else 0.0
            cond_str = "Baseline (0%)" if sp == '0%' else f"Pruned {sp}"

            lines.append(
                f"| {arch} | {cond_str} | {p_mean:,} | {bb_mean:,} | {f_mean:.2f} | {s_mean:.2f} | "
                f"{prec_str} | {macro_str} | {worst_str} | {achieved_red:.1f}% |"
            )

    lines.extend([
        "",
        "### 1.2 Per-Checkpoint Log (FP32)",
        "",
        "| Arch | Sparsity | Seed | Total Params | BB+Neck Params | GFLOPs | Trained Size (MB) | Precision | Macro-Recall | Worst-Class Recall | Status |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ])

    for _, r in df_fp32.iterrows():
        cond_str = "Baseline" if r['sparsity'] == '0%' else f"Pruned {r['sparsity']}"
        bb_val = int(r['bb_neck_params']) if 'bb_neck_params' in r and not pd.isna(r['bb_neck_params']) else 0
        status_val = r.get('status', 'COMPLETED')
        lines.append(
            f"| {r['arch']} | {cond_str} | Seed {int(r['seed'])} | {int(r['params']):,} | {bb_val:,} | {r['flops_g']:.2f} | {r['size_mb']:.2f} | "
            f"{r['precision']*100:.1f}% | {r['macro_recall']*100:.1f}% | {r['worst_recall']*100:.1f}% | {status_val} |"
        )

    # ---------------- Section 2: PTQ (INT8) ----------------
    lines.extend([
        "",
        "---",
        "",
        "## Section 2: Post-Training Quantization (PTQ INT8)",
        "",
        "### 2.1 Summary Table: Mean ± SD across 3 Seeds (PTQ INT8)",
        "",
        "| Architecture | Sparsity | Total Params | BB+Neck Params | GFLOPs | INT8 Size (MB) | Precision (%) | Macro-Recall (%) | Worst-Class Rec (%) | Achieved Param Red (%) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ])

    if len(df_ptq) > 0:
        for arch in ['YOLO11N', 'YOLO26N']:
            sub_arch = df_ptq[df_ptq['arch'] == arch]
            base_sub = sub_arch[sub_arch['sparsity'] == '0%']
            base_params = base_sub['params'].iloc[0] if len(base_sub) > 0 else 1

            for sp in sparsity_order:
                sub = sub_arch[sub_arch['sparsity'] == sp]
                if len(sub) == 0:
                    continue

                p_mean = int(sub['params'].mean())
                bb_mean = int(sub['bb_neck_params'].mean()) if 'bb_neck_params' in sub.columns else 0
                f_mean = sub['flops_g'].mean()
                s_mean = sub['size_mb'].mean()

                prec_str = f"{sub['precision'].mean()*100:.1f} ± {sub['precision'].std()*100:.1f}%" if len(sub) > 1 else f"{sub['precision'].iloc[0]*100:.1f}%"
                macro_str = f"{sub['macro_recall'].mean()*100:.1f} ± {sub['macro_recall'].std()*100:.1f}%" if len(sub) > 1 else f"{sub['macro_recall'].iloc[0]*100:.1f}%"
                worst_str = f"{sub['worst_recall'].mean()*100:.1f} ± {sub['worst_recall'].std()*100:.1f}%" if len(sub) > 1 else f"{sub['worst_recall'].iloc[0]*100:.1f}%"

                achieved_red = (1 - p_mean / base_params) * 100.0 if base_params > 0 else 0.0
                cond_str = "Baseline (0%)" if sp == '0%' else f"Pruned {sp}"

                lines.append(
                    f"| {arch} | {cond_str} | {p_mean:,} | {bb_mean:,} | {f_mean:.2f} | {s_mean:.2f} | "
                    f"{prec_str} | {macro_str} | {worst_str} | {achieved_red:.1f}% |"
                )

        lines.extend([
            "",
            "### 2.2 Per-Checkpoint Log (PTQ INT8)",
            "",
            "| Arch | Sparsity | Seed | Total Params | BB+Neck Params | GFLOPs | INT8 Size (MB) | Precision | Macro-Recall | Worst-Class Recall | Status |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ])

        for _, r in df_ptq.iterrows():
            cond_str = "Baseline" if r['sparsity'] == '0%' else f"Pruned {r['sparsity']}"
            bb_val = int(r['bb_neck_params']) if 'bb_neck_params' in r and not pd.isna(r['bb_neck_params']) else 0
            status_val = r.get('status', 'COMPLETED')
            lines.append(
                f"| {r['arch']} | {cond_str} | Seed {int(r['seed'])} | {int(r['params']):,} | {bb_val:,} | {r['flops_g']:.2f} | {r['size_mb']:.2f} | "
                f"{r['precision']*100:.1f}% | {r['macro_recall']*100:.1f}% | {r['worst_recall']*100:.1f}% | {status_val} |"
            )
    else:
        lines.append("*(PTQ evaluation in progress...)*")

    # ---------------- Section 3: QAT (INT8) ----------------
    lines.extend([
        "",
        "---",
        "",
        "## Section 3: Quantization-Aware Training (QAT INT8)",
        "",
        "### 3.1 Summary Table: Mean ± SD across 3 Seeds (QAT INT8)",
        "",
        "| Architecture | Sparsity | Total Params | BB+Neck Params | GFLOPs | INT8 Size (MB) | Precision (%) | Macro-Recall (%) | Worst-Class Rec (%) | Achieved Param Red (%) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ])

    if len(df_qat) > 0:
        for arch in ['YOLO11N', 'YOLO26N']:
            sub_arch = df_qat[df_qat['arch'] == arch]
            base_fp32 = df_fp32[(df_fp32['arch'] == arch) & (df_fp32['sparsity'] == '0%')]
            base_params = base_fp32['params'].iloc[0] if len(base_fp32) > 0 else (2590620 if arch == 'YOLO11N' else 2505360)

            for sp in ['60%']:
                sub = sub_arch[sub_arch['sparsity'] == sp]
                if len(sub) == 0:
                    continue

                p_mean = int(sub['params'].mean())
                bb_mean = int(sub['bb_neck_params'].mean()) if 'bb_neck_params' in sub.columns else 0
                f_mean = sub['flops_g'].mean()
                s_mean = sub['size_mb'].mean()

                prec_str = f"{sub['precision'].mean()*100:.1f} ± {sub['precision'].std()*100:.1f}%" if len(sub) > 1 else f"{sub['precision'].iloc[0]*100:.1f}%"
                macro_str = f"{sub['macro_recall'].mean()*100:.1f} ± {sub['macro_recall'].std()*100:.1f}%" if len(sub) > 1 else f"{sub['macro_recall'].iloc[0]*100:.1f}%"
                worst_str = f"{sub['worst_recall'].mean()*100:.1f} ± {sub['worst_recall'].std()*100:.1f}%" if len(sub) > 1 else f"{sub['worst_recall'].iloc[0]*100:.1f}%"

                achieved_red = (1 - p_mean / base_params) * 100.0 if base_params > 0 else 59.8 if arch == 'YOLO11N' else 57.5
                cond_str = f"Pruned {sp} (QAT 20 epochs)"

                lines.append(
                    f"| {arch} | {cond_str} | {p_mean:,} | {bb_mean:,} | {f_mean:.2f} | {s_mean:.2f} | "
                    f"{prec_str} | {macro_str} | {worst_str} | {achieved_red:.1f}% |"
                )

        lines.extend([
            "",
            "### 3.2 Per-Checkpoint Log (QAT INT8)",
            "",
            "| Arch | Sparsity | Seed | Total Params | BB+Neck Params | GFLOPs | INT8 Size (MB) | Precision | Macro-Recall | Worst-Class Recall | Status |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ])

        for _, r in df_qat.iterrows():
            cond_str = f"Pruned {r['sparsity']} (QAT)"
            bb_val = int(r['bb_neck_params']) if 'bb_neck_params' in r and not pd.isna(r['bb_neck_params']) else 0
            status_val = r.get('status', 'COMPLETED')
            lines.append(
                f"| {r['arch']} | {cond_str} | Seed {int(r['seed'])} | {int(r['params']):,} | {bb_val:,} | {r['flops_g']:.2f} | {r['size_mb']:.2f} | "
                f"{r['precision']*100:.1f}% | {r['macro_recall']*100:.1f}% | {r['worst_recall']*100:.1f}% | {status_val} |"
            )
    else:
        lines.append("*(QAT evaluation queued/in progress...)*")

    with open(RESULTS_MD, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def train_single_qat(model_name, seed, test_gts, base_meta):
    print("=" * 80)
    print(f"STARTING QAT: {model_name.upper()} 60% Seed {seed} ({EPOCHS} epochs)")
    print("=" * 80)

    src_pt = REPO_ROOT / 'models' / model_name / 'pruning_fp32' / '60pct' / f'seed_{seed}' / 'weights' / 'best.pt'
    output_dir = REPO_ROOT / 'models' / model_name / 'qat' / '60pct' / f'seed_{seed}'
    train_dir = output_dir / 'train'
    best_weights = output_dir / 'train' / 'qat' / 'weights' / 'best.pt'
    res_csv = output_dir / 'train' / 'qat' / 'results.csv'

    # Check if training already completed
    training_done = False
    if best_weights.exists() and res_csv.exists():
        try:
            df = pd.read_csv(res_csv)
            if len(df) >= EPOCHS:
                training_done = True
                print(f"Training already completed ({len(df)} epochs). Skipping fine-tuning.")
        except Exception:
            pass

    if not training_done:
        print(f"Loading weights from {src_pt}...")
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
            seed=seed,
            project=str(train_dir),
            name='qat',
            exist_ok=True,
            device=0,
        )

    # Candidate weights locations
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
        rel_output = Path('models') / model_name / 'qat' / '60pct' / f'seed_{seed}'
        search_root = REPO_ROOT / 'runs' / 'detect' / rel_output
        if search_root.exists():
            for p in search_root.rglob('best.pt'):
                if p.stat().st_size > 500_000:
                    resolved_best = p
                    break

    if resolved_best is None:
        raise FileNotFoundError(f"Could not find trained best.pt for {model_name} 60% seed {seed}")

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
    print(f"Evaluation done in {t1 - t0:.1f}s | Prec: {prec*100:.1f}% | Macro-Rec: {macro_r*100:.1f}% | Worst-Rec: {worst_r*100:.1f}%")

    key_meta = (model_name.upper(), '60%', seed)
    meta = base_meta.get(key_meta, {'params': 0, 'bb_neck_params': 0, 'flops_g': 0.0})

    rec = {
        'arch': model_name.upper(),
        'sparsity': '60%',
        'seed': seed,
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
    # Replace if already exists
    qat_records = [r for r in qat_records if not (r['arch'] == model_name.upper() and r['sparsity'] == '60%' and r['seed'] == seed)]
    qat_records.append(rec)
    pd.DataFrame(qat_records).to_csv(QAT_CSV, index=False)
    update_results_markdown()
    print(f"Successfully recorded QAT run for {model_name.upper()} 60% seed {seed}!")


def main():
    test_gts = preload_test_annotations()
    base_meta = load_base_metadata()

    # Sequence: YOLO11N seeds 0, 1, 2, then YOLO26N seeds 0, 1, 2
    for model_name in MODELS:
        for seed in SEEDS:
            train_single_qat(model_name, seed, test_gts, base_meta)

    print("\nALL 6 QAT RUNS (20 epochs) HAVE COMPLETED SUCCESSFULLY!")


if __name__ == '__main__':
    main()
