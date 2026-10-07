"""Evaluate all 60 PTQ INT8 models on held-out test split at fixed tau = 0.50.

Caches per-checkpoint evaluation results to results/fixed_tau05_ptq.csv
and updates RESULTS_FIXED_TAU05.md with Section 2 (PTQ INT8).
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from eval.metrics import load_yolo_labels, match_detections_single_image, CLASS_NAMES

DATA_DIR = REPO_ROOT / 'data' / 'processed' / 'RGB' / 'yolo'
TEST_TXT = DATA_DIR / 'test.txt'
RESULTS_DIR = REPO_ROOT / 'results'
RESULTS_MD = REPO_ROOT / 'RESULTS_FIXED_TAU05.md'
PTQ_CSV = RESULTS_DIR / 'fixed_tau05_ptq.csv'

MODELS = ['yolo11n', 'yolo26n']
RATIOS = ['0%', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%']
SEEDS = [0, 1, 2]
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
    csv_0to50 = RESULTS_DIR / 'fixed_tau05_checkpoints_0to50.csv'
    csv_further = RESULTS_DIR / 'fixed_tau05_checkpoints_further.csv'
    records = []
    if csv_0to50.exists():
        records.extend(pd.read_csv(csv_0to50).to_dict('records'))
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
    qat_csv = RESULTS_DIR / 'fixed_tau05_qat.csv'
    qat_records = []
    if qat_csv.exists() and qat_csv.stat().st_size > 0:
        qat_records = pd.read_csv(qat_csv).to_dict('records')
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
            "### 3.2 Per-Checkpoint Log (QAT INT8)",
            "",
            "| Arch | Sparsity | Seed | Total Params | BB+Neck Params | GFLOPs | INT8 Size (MB) | Precision | Macro-Recall | Worst-Class Recall | Status |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ])

        for _, r in df_qat.iterrows():
            cond_str = "Baseline" if r['sparsity'] == '0%' else f"Pruned {r['sparsity']}"
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


def main():
    print("=" * 80)
    print("PTQ INT8 EVALUATION PIPELINE (Fixed tau = 0.50)")
    print("=" * 80)

    test_gts = preload_test_annotations()
    base_meta = load_base_metadata()

    existing_records = []
    if PTQ_CSV.exists() and PTQ_CSV.stat().st_size > 0:
        existing_records = pd.read_csv(PTQ_CSV).to_dict('records')

    update_results_markdown()

    total_models = len(MODELS) * len(RATIOS) * len(SEEDS)
    current = 0

    for model in MODELS:
        for ratio in RATIOS:
            for seed in SEEDS:
                current += 1
                ratio_label = ratio.replace('%', 'pct')
                key_meta = (model.upper(), ratio, seed)
                meta = base_meta.get(key_meta, {'params': 0, 'bb_neck_params': 0, 'flops_g': 0.0})

                already = any(r['arch'] == model.upper() and r['sparsity'] == ratio and r['seed'] == seed for r in existing_records)
                if already:
                    print(f"[{current}/{total_models}] {model.upper()} {ratio} seed {seed}: already evaluated, skipping.")
                    continue

                onnx_path = REPO_ROOT / 'models' / model / 'joint_int8' / ratio_label / f'seed_{seed}' / 'model_int8.onnx'
                if not onnx_path.exists():
                    print(f"[{current}/{total_models}] {model.upper()} {ratio} seed {seed}: ONNX file not found ({onnx_path}), waiting/skipping.")
                    continue

                size_mb = onnx_path.stat().st_size / (1024 * 1024)
                print(f"[{current}/{total_models}] Evaluating {model.upper()} {ratio} seed {seed} (Size: {size_mb:.2f} MB)...")
                t0 = time.time()
                prec, macro_r, worst_r = evaluate_onnx_model(onnx_path, test_gts)
                t1 = time.time()
                print(f"  -> Done in {t1 - t0:.1f}s | Prec: {prec*100:.1f}% | Macro-Rec: {macro_r*100:.1f}% | Worst-Rec: {worst_r*100:.1f}%")

                rec = {
                    'arch': model.upper(),
                    'sparsity': ratio,
                    'seed': seed,
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

    print("PTQ INT8 evaluation complete!")


if __name__ == '__main__':
    main()
