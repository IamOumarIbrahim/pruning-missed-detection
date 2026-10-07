"""Benchmark: Pruned Models vs Baselines at Matched Precision P = 0.90 with Floor tau >= 0.35.

Rules:
- Strictly sequential on CUDA:0.
- Pure in-memory forward passes on held-out test split (subjects 05, 10, 12).
- Candidate thresholds tau in [0.35, 0.95]. Lowest tau where precision >= 0.90.
- If precision >= 0.90 cannot be achieved for tau in [0.35, 0.95], record INFEASIBLE with Rec@P90 = 0.0%.
- Fixed comparator at tau = 0.35.
- Output: RESULTS_SUNDAY_P90.md and results/benchmark_p90.csv.
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
RESULTS_CSV = RESULTS_DIR / 'benchmark_p90.csv'
RESULTS_MD = REPO_ROOT / 'RESULTS_SUNDAY_P90.md'

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
        "# Benchmark Results: Pruned Models vs Baselines (Matched Precision P = 0.90, Floor tau >= 0.35)",
        "",
        f"> **Status:** {status_str} — Last Updated: `{now_str}`  ",
        "> **Protocol:** Test split evaluated at lowest confidence threshold $\\tau_{@P90} \\ge 0.35$ where cumulative running precision satisfies $P \\ge 0.90$. If impossible in $[0.35, 0.95]$, marked INFEASIBLE.  ",
        "> **Splits:** Held-out subject-disjoint test partition (`subject_05`, `subject_10`, `subject_12`).  ",
        "",
        "## Summary: Mean ± SD across 3 Seeds",
        "",
        "| Architecture | Condition | tau_@P90 | Precision (%) | Macro-Rec @ P90 (%) | Worst-Class Rec @ P90 (%) | Mean Delta vs Base | Rec @ Fixed tau=0.35 |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    if len(df) > 0:
        for m in MODELS:
            m_sub = df[df['model'] == m]
            for r in RATIOS:
                r_sub = m_sub[m_sub['ratio'] == r]
                n_seeds = len(r_sub)
                if n_seeds > 0:
                    cond = "Baseline (0%)" if r == '0%' else f"Pruned {r}"
                    feasible_sub = r_sub[r_sub['is_feasible'] == True]
                    n_feasible = len(feasible_sub)

                    if n_feasible == 0:
                        tau_m = "INFEASIBLE"
                        prec_m = "N/A"
                        macro_m = "0.0 ± 0.0%"
                        worst_m = "0.0 ± 0.0%"
                        d_m = f"{r_sub['delta_pp'].mean():+.2f} ± {r_sub['delta_pp'].std():.2f} pp" if r != '0%' and 'delta_pp' in r_sub else "N/A"
                    elif n_feasible > 1:
                        tau_m = f"{feasible_sub['tau_90'].astype(float).mean():.4f} ± {feasible_sub['tau_90'].astype(float).std():.4f}"
                        prec_m = f"{feasible_sub['precision'].mean()*100:.1f} ± {feasible_sub['precision'].std()*100:.1f}%"
                        macro_m = f"{r_sub['macro_rec'].mean()*100:.1f} ± {r_sub['macro_rec'].std()*100:.1f}%"
                        worst_m = f"{r_sub['worst_rec'].mean()*100:.1f} ± {r_sub['worst_rec'].std()*100:.1f}%"
                        d_m = f"{r_sub['delta_pp'].mean():+.2f} ± {r_sub['delta_pp'].std():.2f} pp" if r != '0%' else "—"
                    else:
                        tau_m = f"{feasible_sub['tau_90'].astype(float).iloc[0]:.4f}"
                        prec_m = f"{feasible_sub['precision'].iloc[0]*100:.1f}%"
                        macro_m = f"{r_sub['macro_rec'].iloc[0]*100:.1f}%"
                        worst_m = f"{r_sub['worst_rec'].iloc[0]*100:.1f}%"
                        d_m = f"{r_sub['delta_pp'].iloc[0]:+.2f} pp" if r != '0%' else "—"

                    # Fixed tau=0.35 recall
                    if n_seeds > 1:
                        rec35_m = f"{r_sub['rec_tau35'].mean()*100:.1f} ± {r_sub['rec_tau35'].std()*100:.1f}%"
                    else:
                        rec35_m = f"{r_sub['rec_tau35'].iloc[0]*100:.1f}%"

                    tag = f" ({n_seeds}/3 seeds)" if n_seeds < 3 else ""
                    lines.append(
                        f"| {m.upper()} | {cond}{tag} | {tau_m} | {prec_m} | {macro_m} | {worst_m} | {d_m} | {rec35_m} |"
                    )

    lines.extend([
        "",
        "## Per-Checkpoint Log (All 36 Models)",
        "",
        "| Model | Condition | Seed | tau_@P90 | Precision | Macro-Rec @ P90 | Worst-Class Rec @ P90 | Binding Class (TP/GT) | Delta vs Base | Rec @ Fixed tau=0.35 |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ])

    for r in records:
        cond = "Baseline" if r['ratio'] == '0%' else f"Pruned {r['ratio']}"
        if r['is_feasible']:
            tau_str = f"{float(r['tau_90']):.4f}"
            prec_str = f"{r['precision']*100:.1f}%"
        else:
            tau_str = "INFEASIBLE"
            prec_str = "N/A"

        macro_str = f"{r['macro_rec']*100:.1f}%"
        worst_str = f"{r['worst_rec']*100:.1f}%"
        binding_str = f"{r['binding_class']} ({r['binding_tp']}/{r['binding_gt']})"
        delta_str = f"{r['delta_pp']:+.2f} pp" if r['ratio'] != '0%' else "—"
        rec35_str = f"{r['rec_tau35']*100:.1f}%"

        lines.append(
            f"| {r['model'].upper()} | {cond} | Seed {r['seed']} | {tau_str} | {prec_str} | {macro_str} | {worst_str} | {binding_str} | {delta_str} | {rec35_str} |"
        )

    with open(RESULTS_MD, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
        f.flush()
        os.fsync(f.fileno())


def main():
    print("=" * 80)
    print("STARTING BENCHMARK: RECALL AT MATCHED PRECISION P = 0.90 (FLOOR TAU >= 0.35)")
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

                # 1. Pure in-memory forward pass on test split
                dets = []
                res_gen = yolo_model.predict(
                    source=str(DATA_DIR / 'test.txt'),
                    conf=0.10, batch=16, device=0,
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

                # 2. Sort test predictions descending by confidence
                dets.sort(key=lambda x: x['conf'], reverse=True)

                # 3. Fixed comparator at tau = 0.35
                tp_35 = {c: 0 for c in range(len(CLASS_NAMES))}
                fp_35 = 0
                for d in dets:
                    if d['conf'] >= 0.35:
                        if d['is_tp']:
                            tp_35[d['cls']] += 1
                        else:
                            fp_35 += 1
                total_tp_35 = sum(tp_35.values())
                prec_35 = total_tp_35 / (total_tp_35 + fp_35) if (total_tp_35 + fp_35) > 0 else 0.0
                recs_35 = [tp_35[c] / gt_counts[c] if gt_counts[c] > 0 else 1.0 for c in range(len(CLASS_NAMES))]
                macro_rec_35 = float(np.mean(recs_35))

                # 4. Search for lowest threshold tau_90 in [0.35, 0.95] where P >= 0.90
                candidates = []
                if prec_35 >= 0.90:
                    candidates.append((0.3500, prec_35, len([d for d in dets if d['conf'] >= 0.35])))

                cum_tp, cum_fp = 0, 0
                for k, d in enumerate(dets):
                    if d['is_tp']:
                        cum_tp += 1
                    else:
                        cum_fp += 1

                    is_last = (k == len(dets) - 1) or (dets[k+1]['conf'] != d['conf'])
                    if is_last and (0.35 <= d['conf'] <= 0.95):
                        prec = cum_tp / (cum_tp + cum_fp)
                        if prec >= 0.90:
                            candidates.append((d['conf'], prec, k + 1))

                if len(candidates) > 0:
                    min_cand = min(candidates, key=lambda x: x[0])
                    tau_90 = float(min_cand[0])
                    prec_90 = float(min_cand[1])
                    n_dets = min_cand[2]

                    tps = {c: sum(1 for d in dets[:n_dets] if d['is_tp'] and d['cls'] == c) for c in range(len(CLASS_NAMES))}
                    recs = {c: tps[c] / gt_counts[c] if gt_counts[c] > 0 else 1.0 for c in range(len(CLASS_NAMES))}
                    macro_rec_90 = float(np.mean(list(recs.values())))
                    worst_c = min(recs, key=recs.get)
                    worst_rec_90 = float(recs[worst_c])
                    binding_class = CLASS_NAMES[worst_c]
                    binding_tp = tps[worst_c]
                    binding_gt = gt_counts[worst_c]
                    is_feasible = True
                else:
                    tau_90 = "INFEASIBLE"
                    prec_90 = 0.0
                    macro_rec_90 = 0.0
                    worst_rec_90 = 0.0
                    binding_class = "N/A"
                    binding_tp = 0
                    binding_gt = 0
                    is_feasible = False

                # Store baseline macro recall for computing deltas
                if ratio == '0%':
                    base_macro[(model_name, seed)] = macro_rec_90
                    delta_pp = 0.0
                else:
                    base_val = base_macro.get((model_name, seed), macro_rec_90)
                    delta_pp = (macro_rec_90 - base_val) * 100.0

                elapsed = time.time() - t0

                row_dict = {
                    'model': model_name,
                    'ratio': ratio,
                    'seed': seed,
                    'tau_90': tau_90,
                    'precision': prec_90,
                    'macro_rec': macro_rec_90,
                    'worst_rec': worst_rec_90,
                    'binding_class': binding_class,
                    'binding_tp': binding_tp,
                    'binding_gt': binding_gt,
                    'delta_pp': delta_pp,
                    'rec_tau35': macro_rec_35,
                    'prec_tau35': prec_35,
                    'is_feasible': is_feasible,
                    'elapsed_sec': elapsed
                }
                records.append(row_dict)

                # Append to CSV
                is_first = not RESULTS_CSV.exists() or RESULTS_CSV.stat().st_size == 0
                pd.DataFrame([row_dict]).to_csv(RESULTS_CSV, mode='a', header=is_first, index=False)

                # Overwrite and flush markdown table
                write_markdown_table(records, total_models)

                # Print clean line to stdout
                tau_repr = f"{tau_90:.4f}" if is_feasible else "INFEASIBLE"
                d_repr = f"{delta_pp:+.2f} pp" if ratio != '0%' else "baseline"
                print(
                    f"[{current}/{total_models}] {name_str} | "
                    f"tau_90={tau_repr} | Prec={prec_90*100:.1f}% | "
                    f"MacroRec={macro_rec_90*100:.1f}% | Worst={binding_class} ({binding_tp}/{binding_gt} = {worst_rec_90*100:.1f}%) | "
                    f"Delta={d_repr} | Rec@0.35={macro_rec_35*100:.1f}% (Time: {elapsed:.1f}s)"
                )
                sys.stdout.flush()

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETED SUCCESSFULLY.")
    print(f"Results recorded in: {RESULTS_MD}")
    print("=" * 80)


if __name__ == '__main__':
    main()
