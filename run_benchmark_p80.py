"""Benchmark: Pruned Models vs Baselines at Matched Precision P = 0.80.

Schema:
- Summary Table: Mean +/- SD across 3 seeds
- Per-Checkpoint Table: 36 individual models
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from ultralytics import YOLO

import prune.pruner
from eval.metrics import load_yolo_labels, match_detections_single_image

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / 'data' / 'processed' / 'RGB' / 'yolo'
RESULTS_DIR = REPO_ROOT / 'results'
RESULTS_CSV = RESULTS_DIR / 'live_benchmark_p80.csv'
RESULTS_MD = REPO_ROOT / 'RESULTS_SUNDAY.md'

MODELS = ['yolo11n', 'yolo26n']
RATIOS = ['0%', '10%', '20%', '30%', '40%', '50%']
SEEDS = [0, 1, 2]
CLASS_NAMES = ['yawning', 'hand_over_mouth', 'drinking', 'phone_use']


def get_weights_path(model, ratio, seed):
    if ratio == '0%':
        return REPO_ROOT / 'models' / model / 'baseline' / f'seed_{seed}' / 'weights' / 'best.pt'
    else:
        label = ratio.replace('%', 'pct')
        return REPO_ROOT / 'models' / model / 'pruning_fp32' / label / f'seed_{seed}' / 'weights' / 'best.pt'


def preload_test_annotations():
    txt_path = DATA_DIR / 'test.txt'
    gts = {}
    gt_counts = {c: 0 for c in range(len(CLASS_NAMES))}

    for line in open(txt_path):
        rel_p = line.strip()
        img_p = (DATA_DIR / rel_p).resolve()
        lbl_p = Path(str(img_p).replace('images', 'labels')).with_suffix('.txt')
        boxes = load_yolo_labels(str(lbl_p), 640)
        img_str = str(img_p)
        gts[img_str] = boxes
        for b in boxes:
            gt_counts[b[0]] += 1

    return gts, gt_counts


def write_markdown_table(records, total_models):
    now_str = time.strftime('%Y-%m-%d %H:%M:%S')
    completed = len(records)
    status_str = f"In Progress ({completed}/{total_models} completed)" if completed < total_models else f"COMPLETED ({completed}/{total_models} models)"

    df = pd.DataFrame(records)

    lines = [
        "# Benchmark Results: Pruned Models vs Baselines (Matched Precision P = 0.80)",
        "",
        f"> **Status:** {status_str} — Last Updated: `{now_str}`  ",
        "> **Protocol:** Test split evaluated at lowest confidence threshold $\\tau_{@P80}$ where cumulative running precision satisfies $P \\ge 0.80$.  ",
        "> **Splits:** Held-out subject-disjoint test partition (`subject_05`, `subject_10`, `subject_12`).  ",
        "",
        "## Summary: Mean ± SD across 3 Seeds",
        "",
        "| Architecture | Condition | $\\tau_{@P80}$ | Precision (%) | Macro-Rec @ P80 (%) | Worst-Class Rec @ P80 (%) | Mean $\\Delta$ vs Base |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    # Generate summary rows for conditions with at least 1 seed completed (Mean +/- SD if 3 seeds)
    if len(df) > 0:
        for m in MODELS:
            m_sub = df[df['model'] == m]
            for r in RATIOS:
                r_sub = m_sub[m_sub['ratio'] == r]
                n_seeds = len(r_sub)
                if n_seeds > 0:
                    cond = "Baseline (0%)" if r == '0%' else f"Pruned {r}"
                    if n_seeds > 1:
                        tau_m = f"{r_sub['tau_80'].mean():.4f} ± {r_sub['tau_80'].std():.4f}"
                        prec_m = f"{r_sub['precision'].mean()*100:.1f} ± {r_sub['precision'].std()*100:.1f}%"
                        macro_m = f"{r_sub['macro_rec'].mean()*100:.1f} ± {r_sub['macro_rec'].std()*100:.1f}%"
                        worst_m = f"{r_sub['worst_rec'].mean()*100:.1f} ± {r_sub['worst_rec'].std()*100:.1f}%"
                        d_m = f"{r_sub['delta_pp'].mean():+.2f} ± {r_sub['delta_pp'].std():.2f} pp" if r != '0%' else "—"
                    else:
                        tau_m = f"{r_sub['tau_80'].iloc[0]:.4f}"
                        prec_m = f"{r_sub['precision'].iloc[0]*100:.1f}%"
                        macro_m = f"{r_sub['macro_rec'].iloc[0]*100:.1f}%"
                        worst_m = f"{r_sub['worst_rec'].iloc[0]*100:.1f}%"
                        d_m = f"{r_sub['delta_pp'].iloc[0]:+.2f} pp" if r != '0%' else "—"

                    tag = f" ({n_seeds}/3 seeds)" if n_seeds < 3 else ""
                    lines.append(
                        f"| {m.upper()} | {cond}{tag} | {tau_m} | {prec_m} | {macro_m} | {worst_m} | {d_m} |"
                    )

    lines.extend([
        "",
        "## Per-Checkpoint Log (All 36 Models)",
        "",
        "| Model | Condition | Seed | $\\tau_{@P80}$ | Precision | Macro-Rec @ P80 | Worst-Class Rec @ P80 | Binding Class (TP/GT) | $\\Delta$ vs Base |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ])

    for r in records:
        cond = "Baseline" if r['ratio'] == '0%' else f"Pruned {r['ratio']}"
        tau_str = f"{r['tau_80']:.4f}"
        prec_str = f"{r['precision']*100:.1f}%"
        macro_str = f"{r['macro_rec']*100:.1f}%"
        worst_str = f"{r['worst_rec']*100:.1f}%"
        binding_str = f"{r['binding_class']} ({r['binding_tp']}/{r['binding_gt']})"
        delta_str = f"{r['delta_pp']:+.2f} pp" if r['ratio'] != '0%' else "—"

        lines.append(
            f"| {r['model'].upper()} | {cond} | Seed {r['seed']} | {tau_str} | {prec_str} | {macro_str} | {worst_str} | {binding_str} | {delta_str} |"
        )

    with open(RESULTS_MD, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
        f.flush()
        os.fsync(f.fileno())


def main():
    print("=" * 80)
    print("STARTING BENCHMARK: RECALL AT MATCHED PRECISION P = 0.80")
    print(f"Target Output: {RESULTS_MD} and {RESULTS_CSV}")
    print("=" * 80)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    gts, gt_counts = preload_test_annotations()
    print(f"Preloaded Test GT: {gt_counts} (Total: {sum(gt_counts.values())} GT instances)")

    base_macro = {}
    records = []

    # Resume from CSV if exists
    if RESULTS_CSV.exists() and RESULTS_CSV.stat().st_size > 0:
        try:
            df_old = pd.read_csv(RESULTS_CSV)
            for _, row in df_old.iterrows():
                r_dict = row.to_dict()
                records.append(r_dict)
                if str(r_dict['ratio']) == '0%':
                    base_macro[(r_dict['model'], int(r_dict['seed']))] = float(r_dict['macro_rec'])
        except Exception:
            pass

    total_models = len(MODELS) * len(RATIOS) * len(SEEDS)
    current = len(records)
    write_markdown_table(records, total_models)

    for model_name in MODELS:
        for ratio in RATIOS:
            for seed in SEEDS:
                already = any(r['model'] == model_name and r['ratio'] == ratio and r['seed'] == seed for r in records)
                if already:
                    continue

                current += 1
                name_str = f"{model_name.upper()} {ratio} (Seed {seed})"
                weights_path = get_weights_path(model_name, ratio, seed)
                if not weights_path.exists():
                    print(f"ERROR: Missing weights {weights_path}")
                    continue

                t0 = time.time()
                yolo_model = YOLO(str(weights_path))

                # 1. Forward pass on test split
                dets = []
                res_gen = yolo_model.predict(
                    source=str(DATA_DIR / 'test.txt'),
                    conf=0.01, batch=16, device=0,
                    verbose=False, stream=True
                )

                for r in res_gen:
                    img_p = str(Path(r.path).resolve())
                    boxes_gt = gts.get(img_p, [])
                    gt_boxes = [(b[1], b[2], b[3], b[4]) for b in boxes_gt]
                    gt_cls = [b[0] for b in boxes_gt]

                    p_boxes = r.boxes.xyxy.cpu().numpy().tolist() if len(r.boxes) else []
                    p_confs = r.boxes.conf.cpu().numpy().tolist() if len(r.boxes) else []
                    p_cls = r.boxes.cls.cpu().int().numpy().tolist() if len(r.boxes) else []

                    matches = match_detections_single_image(p_boxes, p_confs, p_cls, gt_boxes, gt_cls, iou_threshold=0.5)
                    for conf, is_tp, cls_id in matches:
                        dets.append({'conf': float(conf), 'cls': int(cls_id), 'is_tp': bool(is_tp)})

                # 2. Sort descending by confidence score
                dets.sort(key=lambda x: x['conf'], reverse=True)

                # 3. Find lowest confidence threshold tau_80 where running precision >= 0.80
                cum_tp, cum_fp = 0, 0
                best_k = 0
                for k, d in enumerate(dets):
                    if d['is_tp']:
                        cum_tp += 1
                    else:
                        cum_fp += 1
                    prec = cum_tp / (cum_tp + cum_fp)
                    if prec >= 0.80:
                        best_k = k

                tau_80 = dets[best_k]['conf'] if dets else 0.25
                tp_at_80 = sum(1 for d in dets[:best_k+1] if d['is_tp'])
                prec_at_80 = tp_at_80 / (best_k + 1) if dets else 0.0

                # 4. Compute per-class recall and worst-class
                recs = {}
                tps = {}
                for c in range(len(CLASS_NAMES)):
                    tp_c = sum(1 for d in dets[:best_k+1] if d['is_tp'] and d['cls'] == c)
                    recs[c] = tp_c / gt_counts[c] if gt_counts[c] > 0 else 1.0
                    tps[c] = tp_c

                worst_c = min(recs, key=recs.get)
                macro_rec = float(np.mean(list(recs.values())))
                worst_rec = float(recs[worst_c])
                binding_name = CLASS_NAMES[worst_c]
                binding_tp = tps[worst_c]
                binding_gt = gt_counts[worst_c]

                # Store baseline macro recall for computing deltas
                if ratio == '0%':
                    base_macro[(model_name, seed)] = macro_rec
                    delta_pp = 0.0
                else:
                    base_val = base_macro.get((model_name, seed), macro_rec)
                    delta_pp = (macro_rec - base_val) * 100.0

                elapsed = time.time() - t0

                row_dict = {
                    'model': model_name,
                    'ratio': ratio,
                    'seed': seed,
                    'tau_80': tau_80,
                    'precision': prec_at_80,
                    'macro_rec': macro_rec,
                    'worst_rec': worst_rec,
                    'binding_class': binding_name,
                    'binding_tp': binding_tp,
                    'binding_gt': binding_gt,
                    'delta_pp': delta_pp,
                    'elapsed_sec': elapsed
                }
                records.append(row_dict)

                # Append to CSV
                is_first = not RESULTS_CSV.exists() or RESULTS_CSV.stat().st_size == 0
                pd.DataFrame([row_dict]).to_csv(RESULTS_CSV, mode='a', header=is_first, index=False)

                # Overwrite and flush RESULTS_SUNDAY.md
                write_markdown_table(records, total_models)

                # Print clean line to stdout
                d_str = f"{delta_pp:+.2f} pp" if ratio != '0%' else "baseline"
                print(
                    f"[{current}/{total_models}] {name_str} | "
                    f"tau_80={tau_80:.4f} | Prec={prec_at_80*100:.1f}% | "
                    f"MacroRec={macro_rec*100:.1f}% | Worst={binding_name} ({binding_tp}/{binding_gt} = {worst_rec*100:.1f}%) | "
                    f"Delta={d_str} (Time: {elapsed:.1f}s)"
                )
                sys.stdout.flush()

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETED SUCCESSFULLY.")
    print(f"Results recorded in: {RESULTS_MD}")
    print("=" * 80)


if __name__ == '__main__':
    main()
