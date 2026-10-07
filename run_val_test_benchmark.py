"""Comprehensive Benchmark Suite:
1. Save raw detections at conf=0.001 for val and test across all 36 checkpoints.
2. Protocol 1: Val-Tuned tau (>= 0.35, P_val >= 0.90) frozen on test, with transfer gap and floor binding flag.
3. Protocol 2: Baseline checkpoint's tau frozen across all pruning levels.
4. Protocol 3: Fixed operating point at tau = 0.35.
5. Protocol 4: Worst-class recall tracked on baseline's worst class.
6. Table 5: Oracle test-matched benchmark (P >= 0.90, tau >= 0.35) as a separate labeled table.
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
RAW_DETS_DIR = RESULTS_DIR / 'raw_detections'
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


def preload_annotations(split_name):
    txt_path = DATA_DIR / f'{split_name}.txt'
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


def extract_raw_detections_for_split(yolo_model, split_name, gts):
    txt_path = DATA_DIR / f'{split_name}.txt'
    res_gen = yolo_model.predict(
        source=str(txt_path),
        conf=0.001, batch=16, device=0,
        verbose=False, stream=True
    )
    records = []
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
            records.append({
                'conf': float(conf),
                'is_tp': bool(is_tp),
                'class_id': int(cls_id)
            })
    return records


def get_or_extract_detections(model, ratio, seed, val_gts, test_gts):
    RAW_DETS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = RAW_DETS_DIR / f"{model}_{ratio.replace('%', 'pct')}_seed{seed}.parquet"
    if out_file.exists():
        df = pd.read_parquet(out_file)
        val_dets = df[df['split'] == 'val'].to_dict('records')
        test_dets = df[df['split'] == 'test'].to_dict('records')
        return val_dets, test_dets

    weights_path = get_weights_path(model, ratio, seed)
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing weights: {weights_path}")

    yolo_model = YOLO(str(weights_path))
    val_dets = extract_raw_detections_for_split(yolo_model, 'val', val_gts)
    test_dets = extract_raw_detections_for_split(yolo_model, 'test', test_gts)

    df_val = pd.DataFrame(val_dets)
    df_val['split'] = 'val'
    df_test = pd.DataFrame(test_dets)
    df_test['split'] = 'test'
    df_all = pd.concat([df_val, df_test], ignore_index=True)
    df_all.to_parquet(out_file, index=False)

    return val_dets, test_dets


def evaluate_at_threshold(dets, gt_counts, tau):
    filtered = [d for d in dets if d['conf'] >= tau]
    total_det = len(filtered)
    tp_per_class = {c: sum(1 for d in filtered if d['is_tp'] and d['class_id'] == c) for c in range(len(CLASS_NAMES))}
    total_tp = sum(tp_per_class.values())
    total_fp = total_det - total_tp
    precision = total_tp / total_det if total_det > 0 else 0.0

    recs = {c: tp_per_class[c] / gt_counts[c] if gt_counts[c] > 0 else 1.0 for c in range(len(CLASS_NAMES))}
    macro_rec = float(np.mean(list(recs.values())))

    worst_c = min(recs, key=recs.get)
    worst_rec = recs[worst_c]

    return {
        'precision': precision,
        'macro_rec': macro_rec,
        'per_class_rec': recs,
        'tp_per_class': tp_per_class,
        'worst_class_id': worst_c,
        'worst_rec': worst_rec,
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_det': total_det
    }


def tune_tau_val(val_dets, val_gt_counts, min_precision=0.90, floor_tau=0.35, ceil_tau=0.95):
    val_dets_sorted = sorted(val_dets, key=lambda x: x['conf'], reverse=True)

    # 1. Check boundary at floor_tau
    metrics_floor = evaluate_at_threshold(val_dets_sorted, val_gt_counts, floor_tau)
    if metrics_floor['precision'] >= min_precision:
        return floor_tau, True, True

    # 2. Check all detections in [floor_tau, ceil_tau]
    cum_tp, cum_fp = 0, 0
    candidates = []
    for k, d in enumerate(val_dets_sorted):
        if d['is_tp']:
            cum_tp += 1
        else:
            cum_fp += 1
        is_last = (k == len(val_dets_sorted) - 1) or (val_dets_sorted[k+1]['conf'] != d['conf'])
        if is_last and (floor_tau <= d['conf'] <= ceil_tau):
            prec = cum_tp / (cum_tp + cum_fp)
            if prec >= min_precision:
                candidates.append((d['conf'], prec))

    if candidates:
        min_cand = min(candidates, key=lambda x: x[0])
        tau = float(min_cand[0])
        is_floor = (tau <= floor_tau + 1e-4)
        return tau, is_floor, True
    else:
        return None, False, False


def generate_full_markdown_report(records_p1, records_p2, records_p3, oracle_csv_path, total_models):
    now_str = time.strftime('%Y-%m-%d %H:%M:%S')
    completed = len(records_p1)
    status_str = f"In Progress ({completed}/{total_models} models completed)" if completed < total_models else f"COMPLETED ({completed}/{total_models} models)"

    lines = [
        "# Benchmark Results: Subject-Disjoint Validation-Tuned vs Frozen Test Deployments",
        "",
        f"> **Status:** {status_str} — Last Updated: `{now_str}`  ",
        "> **Methodology:** Raw detections extracted at `conf=0.001` on CUDA:0 and evaluated offline.  ",
        "> **Partitions:** Validation split (`subject_02, 03, 11`), Held-out test split (`subject_05, 10, 12`).  ",
        "",
        "---",
        "",
        "## Table 1: Protocol 1 — Val-Tuned $\\tau_{\\text{val}}$ Frozen on Test Split",
        "",
        "> **Rule:** Operating threshold $\\tau_{\\text{val}} \\ge 0.35$ tuned on validation split where $P_{\\text{val}} \\ge 0.90$.  ",
        "> Frozen $\\tau_{\\text{val}}$ evaluated on test split. Transfer Gap = $P_{\\text{test}} - P_{\\text{val}}$.  ",
        "> `[BINDING]` indicates the 0.35 confidence floor was active.  ",
        "",
        "### Summary: Mean ± SD across 3 Seeds (Protocol 1)",
        "",
        "| Architecture | Condition | $\\tau_{\\text{val}}$ | Val Prec (%) | Test Prec (%) | Transfer Gap (pp) | Macro-Rec @ $\\tau_{\\text{val}}$ (%) | Base Worst-Cls Rec (%) | Mean $\\Delta$ vs Base | Floor Binding |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    df1 = pd.DataFrame(records_p1)
    if len(df1) > 0:
        for m in MODELS:
            m_sub = df1[df1['model'] == m]
            for r in RATIOS:
                r_sub = m_sub[m_sub['ratio'] == r]
                n_seeds = len(r_sub)
                if n_seeds > 0:
                    cond = "Baseline (0%)" if r == '0%' else f"Pruned {r}"
                    if n_seeds > 1:
                        tau_s = f"{r_sub['tau_val'].mean():.4f} ± {r_sub['tau_val'].std():.4f}"
                        pval_s = f"{r_sub['prec_val'].mean()*100:.1f} ± {r_sub['prec_val'].std()*100:.1f}%"
                        ptest_s = f"{r_sub['prec_test'].mean()*100:.1f} ± {r_sub['prec_test'].std()*100:.1f}%"
                        gap_s = f"{r_sub['transfer_gap_pp'].mean():+.2f} ± {r_sub['transfer_gap_pp'].std():.2f} pp"
                        rec_s = f"{r_sub['macro_rec_test'].mean()*100:.1f} ± {r_sub['macro_rec_test'].std()*100:.1f}%"
                        bw_s = f"{r_sub['base_worst_rec'].mean()*100:.1f} ± {r_sub['base_worst_rec'].std()*100:.1f}%"
                        d_s = f"{r_sub['delta_macro_pp'].mean():+.2f} ± {r_sub['delta_macro_pp'].std():.2f} pp" if r != '0%' else "—"
                        floor_s = f"{r_sub['floor_binding'].sum()}/{n_seeds} ({r_sub['floor_binding'].mean()*100:.0f}%)"
                    else:
                        tau_s = f"{r_sub['tau_val'].iloc[0]:.4f}"
                        pval_s = f"{r_sub['prec_val'].iloc[0]*100:.1f}%"
                        ptest_s = f"{r_sub['prec_test'].iloc[0]*100:.1f}%"
                        gap_s = f"{r_sub['transfer_gap_pp'].iloc[0]:+.2f} pp"
                        rec_s = f"{r_sub['macro_rec_test'].iloc[0]*100:.1f}%"
                        bw_s = f"{r_sub['base_worst_rec'].iloc[0]*100:.1f}%"
                        d_s = f"{r_sub['delta_macro_pp'].iloc[0]:+.2f} pp" if r != '0%' else "—"
                        floor_s = "YES [BINDING]" if r_sub['floor_binding'].iloc[0] else "NO"

                    tag = f" ({n_seeds}/3 seeds)" if n_seeds < 3 else ""
                    lines.append(
                        f"| {m.upper()} | {cond}{tag} | {tau_s} | {pval_s} | {ptest_s} | {gap_s} | {rec_s} | {bw_s} | {d_s} | {floor_s} |"
                    )

    lines.extend([
        "",
        "### Per-Checkpoint Log: Protocol 1 (All 36 Checkpoints)",
        "",
        "| Model | Condition | Seed | $\\tau_{\\text{val}}$ | Floor Binding | Val Prec | Test Prec | Transfer Gap | Macro-Rec @ $\\tau_{\\text{val}}$ | Base Worst-Cls Rec (TP/GT) | Min Cls Rec (TP/GT) | $\\Delta$ vs Base |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ])
    for r in records_p1:
        cond = "Baseline" if r['ratio'] == '0%' else f"Pruned {r['ratio']}"
        floor_flag = "**YES [BINDING]**" if r['floor_binding'] else "NO"
        d_str = f"{r['delta_macro_pp']:+.2f} pp" if r['ratio'] != '0%' else "—"
        lines.append(
            f"| {r['model'].upper()} | {cond} | Seed {r['seed']} | {r['tau_val']:.4f} | {floor_flag} | "
            f"{r['prec_val']*100:.1f}% | {r['prec_test']*100:.1f}% | {r['transfer_gap_pp']:+.2f} pp | "
            f"{r['macro_rec_test']*100:.1f}% | {r['base_worst_name']} ({r['base_worst_tp']}/{r['base_worst_gt']} = {r['base_worst_rec']*100:.1f}%) | "
            f"{r['min_cls_name']} ({r['min_cls_tp']}/{r['min_cls_gt']} = {r['min_cls_rec']*100:.1f}%) | {d_str} |"
        )

    # Table 2: Baseline tau frozen across pruning levels
    lines.extend([
        "",
        "---",
        "",
        "## Table 2: Protocol 2 — Baseline Checkpoint's $\\tau_{\\text{base}}$ Frozen Across All Pruning Levels",
        "",
        "> **Rule:** For each architecture and seed, freeze the baseline's val-tuned threshold $\\tau_{\\text{base}}$ and evaluate pruned models on test split.  ",
        "",
        "### Summary: Mean ± SD across 3 Seeds (Protocol 2)",
        "",
        "| Architecture | Condition | Frozen $\\tau_{\\text{base}}$ | Test Prec (%) | Macro-Rec @ $\\tau_{\\text{base}}$ (%) | Base Worst-Cls Rec (%) | Mean $\\Delta$ vs Base |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ])
    df2 = pd.DataFrame(records_p2)
    if len(df2) > 0:
        for m in MODELS:
            m_sub = df2[df2['model'] == m]
            for r in RATIOS:
                r_sub = m_sub[m_sub['ratio'] == r]
                n_seeds = len(r_sub)
                if n_seeds > 0:
                    cond = "Baseline (0%)" if r == '0%' else f"Pruned {r}"
                    if n_seeds > 1:
                        tau_s = f"{r_sub['tau_base'].mean():.4f} ± {r_sub['tau_base'].std():.4f}"
                        ptest_s = f"{r_sub['prec_test'].mean()*100:.1f} ± {r_sub['prec_test'].std()*100:.1f}%"
                        rec_s = f"{r_sub['macro_rec_test'].mean()*100:.1f} ± {r_sub['macro_rec_test'].std()*100:.1f}%"
                        bw_s = f"{r_sub['base_worst_rec'].mean()*100:.1f} ± {r_sub['base_worst_rec'].std()*100:.1f}%"
                        d_s = f"{r_sub['delta_macro_pp'].mean():+.2f} ± {r_sub['delta_macro_pp'].std():.2f} pp" if r != '0%' else "—"
                    else:
                        tau_s = f"{r_sub['tau_base'].iloc[0]:.4f}"
                        ptest_s = f"{r_sub['prec_test'].iloc[0]*100:.1f}%"
                        rec_s = f"{r_sub['macro_rec_test'].iloc[0]*100:.1f}%"
                        bw_s = f"{r_sub['base_worst_rec'].iloc[0]*100:.1f}%"
                        d_s = f"{r_sub['delta_macro_pp'].iloc[0]:+.2f} pp" if r != '0%' else "—"
                    tag = f" ({n_seeds}/3 seeds)" if n_seeds < 3 else ""
                    lines.append(
                        f"| {m.upper()} | {cond}{tag} | {tau_s} | {ptest_s} | {rec_s} | {bw_s} | {d_s} |"
                    )

    # Table 3: Fixed tau = 0.35 benchmark
    lines.extend([
        "",
        "---",
        "",
        "## Table 3: Protocol 3 — Fixed Operating Point at $\\tau = 0.35$",
        "",
        "> **Rule:** Fixed unadjusted threshold $\\tau = 0.35$ on held-out test split.  ",
        "",
        "### Summary: Mean ± SD across 3 Seeds (Protocol 3)",
        "",
        "| Architecture | Condition | Test Prec (%) | Macro-Rec @ 0.35 (%) | Base Worst-Cls Rec (%) | Mean $\\Delta$ vs Base |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ])
    df3 = pd.DataFrame(records_p3)
    if len(df3) > 0:
        for m in MODELS:
            m_sub = df3[df3['model'] == m]
            for r in RATIOS:
                r_sub = m_sub[m_sub['ratio'] == r]
                n_seeds = len(r_sub)
                if n_seeds > 0:
                    cond = "Baseline (0%)" if r == '0%' else f"Pruned {r}"
                    if n_seeds > 1:
                        ptest_s = f"{r_sub['prec_test'].mean()*100:.1f} ± {r_sub['prec_test'].std()*100:.1f}%"
                        rec_s = f"{r_sub['macro_rec_test'].mean()*100:.1f} ± {r_sub['macro_rec_test'].std()*100:.1f}%"
                        bw_s = f"{r_sub['base_worst_rec'].mean()*100:.1f} ± {r_sub['base_worst_rec'].std()*100:.1f}%"
                        d_s = f"{r_sub['delta_macro_pp'].mean():+.2f} ± {r_sub['delta_macro_pp'].std():.2f} pp" if r != '0%' else "—"
                    else:
                        ptest_s = f"{r_sub['prec_test'].iloc[0]*100:.1f}%"
                        rec_s = f"{r_sub['macro_rec_test'].iloc[0]*100:.1f}%"
                        bw_s = f"{r_sub['base_worst_rec'].iloc[0]*100:.1f}%"
                        d_s = f"{r_sub['delta_macro_pp'].iloc[0]:+.2f} pp" if r != '0%' else "—"
                    tag = f" ({n_seeds}/3 seeds)" if n_seeds < 3 else ""
                    lines.append(
                        f"| {m.upper()} | {cond}{tag} | {ptest_s} | {rec_s} | {bw_s} | {d_s} |"
                    )

    # Table 4: Oracle Test-Matched Benchmark
    lines.extend([
        "",
        "---",
        "",
        "## Table 4: Labeled Reference — Oracle Test-Matched Benchmark ($P \\ge 0.90$, Floor $\\tau \\ge 0.35$)",
        "",
        "> **Rule (Oracle Reference):** Lowest confidence threshold $\\tau_{@P90} \\ge 0.35$ evaluated directly on test split where $P_{\\text{test}} \\ge 0.90$.  ",
        "",
    ])
    if Path(oracle_csv_path).exists():
        df_ora = pd.read_csv(oracle_csv_path)
        lines.extend([
            "### Summary: Mean ± SD across 3 Seeds (Oracle Reference)",
            "",
            "| Architecture | Condition | $\\tau_{@P90}$ | Test Prec (%) | Macro-Rec @ P90 (%) | Worst-Class Rec @ P90 (%) | Mean $\\Delta$ vs Base |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ])
        for m in MODELS:
            m_sub = df_ora[df_ora['model'] == m]
            for r in RATIOS:
                r_sub = m_sub[m_sub['ratio'] == r]
                n_seeds = len(r_sub)
                if n_seeds > 0:
                    cond = "Baseline (0%)" if r == '0%' else f"Pruned {r}"
                    if n_seeds > 1:
                        tau_s = f"{r_sub['tau_90'].astype(float).mean():.4f} ± {r_sub['tau_90'].astype(float).std():.4f}"
                        ptest_s = f"{r_sub['precision'].mean()*100:.1f} ± {r_sub['precision'].std()*100:.1f}%"
                        rec_s = f"{r_sub['macro_rec'].mean()*100:.1f} ± {r_sub['macro_rec'].std()*100:.1f}%"
                        w_s = f"{r_sub['worst_rec'].mean()*100:.1f} ± {r_sub['worst_rec'].std()*100:.1f}%"
                        d_s = f"{r_sub['delta_pp'].mean():+.2f} ± {r_sub['delta_pp'].std():.2f} pp" if r != '0%' else "—"
                    else:
                        tau_s = f"{float(r_sub['tau_90'].iloc[0]):.4f}"
                        ptest_s = f"{r_sub['precision'].iloc[0]*100:.1f}%"
                        rec_s = f"{r_sub['macro_rec'].iloc[0]*100:.1f}%"
                        w_s = f"{r_sub['worst_rec'].iloc[0]*100:.1f}%"
                        d_s = f"{r_sub['delta_pp'].iloc[0]:+.2f} pp" if r != '0%' else "—"
                    lines.append(
                        f"| {m.upper()} | {cond} | {tau_s} | {ptest_s} | {rec_s} | {w_s} | {d_s} |"
                    )

    with open(RESULTS_MD, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
        f.flush()
        os.fsync(f.fileno())


def main():
    print("=" * 80)
    print("STARTING BENCHMARK: VAL-TUNED TO FROZEN-TEST + RAW DETECTIONS EXPORT")
    print(f"Target Output: {RESULTS_MD}")
    print(f"Raw Detections Dir: {RAW_DETS_DIR}")
    print("=" * 80)

    val_gts, val_gt_counts = preload_annotations('val')
    test_gts, test_gt_counts = preload_annotations('test')
    print(f"Preloaded Val GT: {val_gt_counts} (Total: {sum(val_gt_counts.values())})")
    print(f"Preloaded Test GT: {test_gt_counts} (Total: {sum(test_gt_counts.values())})")

    total_models = len(MODELS) * len(RATIOS) * len(SEEDS)

    # First, collect or load raw detections for all 36 models
    # We will do this progressively and print progress
    all_dets = {}  # (model, ratio, seed) -> (val_dets, test_dets)
    idx = 0
    for model in MODELS:
        for ratio in RATIOS:
            for seed in SEEDS:
                idx += 1
                name_str = f"{model.upper()} {ratio} (Seed {seed})"
                t0 = time.time()
                val_dets, test_dets = get_or_extract_detections(model, ratio, seed, val_gts, test_gts)
                elapsed = time.time() - t0
                all_dets[(model, ratio, seed)] = (val_dets, test_dets)
                print(f"[{idx}/{total_models}] {name_str} | val={len(val_dets)} dets, test={len(test_dets)} dets | Extracted in {elapsed:.1f}s")
                sys.stdout.flush()

    print("\n" + "=" * 80)
    print("ALL 36 CHECKPOINTS EXTRACTED. COMPUTING METRICS OFFLINE...")
    print("=" * 80)

    # 1. First find baseline worst class on test at tau=0.35 and tau_val
    base_info = {}  # (model, seed) -> {'tau_val', 'base_worst_cls', 'base_macro_p1', 'base_macro_p2', 'base_macro_p3'}
    for model in MODELS:
        for seed in SEEDS:
            val_dets, test_dets = all_dets[(model, '0%', seed)]
            tau_val, floor_binding, is_feas = tune_tau_val(val_dets, val_gt_counts, min_precision=0.90, floor_tau=0.35)
            if tau_val is None:
                tau_val = 0.35
            
            # Baseline performance on test at tau_val
            ev_test_base = evaluate_at_threshold(test_dets, test_gt_counts, tau_val)
            ev_test_35 = evaluate_at_threshold(test_dets, test_gt_counts, 0.35)
            
            # Identify baseline's worst class on test (at tau_val)
            worst_c = ev_test_base['worst_class_id']
            base_info[(model, seed)] = {
                'tau_val': tau_val,
                'worst_class_id': worst_c,
                'worst_class_name': CLASS_NAMES[worst_c],
                'macro_p1': ev_test_base['macro_rec'],
                'macro_p2': ev_test_base['macro_rec'],
                'macro_p3': ev_test_35['macro_rec']
            }

    # Now compute records for Protocol 1, Protocol 2, Protocol 3
    records_p1 = []
    records_p2 = []
    records_p3 = []

    for model in MODELS:
        for ratio in RATIOS:
            for seed in SEEDS:
                val_dets, test_dets = all_dets[(model, ratio, seed)]
                base = base_info[(model, seed)]
                b_worst_c = base['worst_class_id']
                b_worst_name = base['worst_class_name']

                # Protocol 1: Val-Tuned tau
                tau_val, floor_binding, is_feas = tune_tau_val(val_dets, val_gt_counts, min_precision=0.90, floor_tau=0.35)
                if tau_val is None:
                    tau_val = 0.35
                    is_feas = False
                
                ev_val = evaluate_at_threshold(val_dets, val_gt_counts, tau_val)
                ev_test1 = evaluate_at_threshold(test_dets, test_gt_counts, tau_val)

                p_val = ev_val['precision']
                p_test = ev_test1['precision']
                gap = (p_test - p_val) * 100.0
                rec_test1 = ev_test1['macro_rec']
                d_p1 = (rec_test1 - base['macro_p1']) * 100.0 if ratio != '0%' else 0.0

                bw_rec1 = ev_test1['per_class_rec'][b_worst_c]
                bw_tp1 = ev_test1['tp_per_class'][b_worst_c]
                bw_gt1 = test_gt_counts[b_worst_c]

                min_c1 = ev_test1['worst_class_id']
                min_name1 = CLASS_NAMES[min_c1]
                min_tp1 = ev_test1['tp_per_class'][min_c1]
                min_gt1 = test_gt_counts[min_c1]
                min_rec1 = ev_test1['worst_rec']

                records_p1.append({
                    'model': model,
                    'ratio': ratio,
                    'seed': seed,
                    'tau_val': tau_val,
                    'floor_binding': floor_binding,
                    'prec_val': p_val,
                    'prec_test': p_test,
                    'transfer_gap_pp': gap,
                    'macro_rec_test': rec_test1,
                    'base_worst_name': b_worst_name,
                    'base_worst_tp': bw_tp1,
                    'base_worst_gt': bw_gt1,
                    'base_worst_rec': bw_rec1,
                    'min_cls_name': min_name1,
                    'min_cls_tp': min_tp1,
                    'min_cls_gt': min_gt1,
                    'min_cls_rec': min_rec1,
                    'delta_macro_pp': d_p1,
                    'is_feasible': is_feas
                })

                # Protocol 2: Baseline tau frozen
                tau_base = base['tau_val']
                ev_test2 = evaluate_at_threshold(test_dets, test_gt_counts, tau_base)
                bw_rec2 = ev_test2['per_class_rec'][b_worst_c]
                d_p2 = (ev_test2['macro_rec'] - base['macro_p2']) * 100.0 if ratio != '0%' else 0.0
                records_p2.append({
                    'model': model,
                    'ratio': ratio,
                    'seed': seed,
                    'tau_base': tau_base,
                    'prec_test': ev_test2['precision'],
                    'macro_rec_test': ev_test2['macro_rec'],
                    'base_worst_rec': bw_rec2,
                    'delta_macro_pp': d_p2
                })

                # Protocol 3: Fixed tau = 0.35
                ev_test3 = evaluate_at_threshold(test_dets, test_gt_counts, 0.35)
                bw_rec3 = ev_test3['per_class_rec'][b_worst_c]
                d_p3 = (ev_test3['macro_rec'] - base['macro_p3']) * 100.0 if ratio != '0%' else 0.0
                records_p3.append({
                    'model': model,
                    'ratio': ratio,
                    'seed': seed,
                    'prec_test': ev_test3['precision'],
                    'macro_rec_test': ev_test3['macro_rec'],
                    'base_worst_rec': bw_rec3,
                    'delta_macro_pp': d_p3
                })

    # Save CSVs
    pd.DataFrame(records_p1).to_csv(RESULTS_DIR / 'benchmark_val_tuned.csv', index=False)
    pd.DataFrame(records_p2).to_csv(RESULTS_DIR / 'benchmark_frozen_base_tau.csv', index=False)
    pd.DataFrame(records_p3).to_csv(RESULTS_DIR / 'benchmark_fixed_tau35.csv', index=False)

    oracle_csv = RESULTS_DIR / 'oracle_test_benchmark_p90.csv'
    generate_full_markdown_report(records_p1, records_p2, records_p3, oracle_csv, total_models)

    print("\n" + "=" * 80)
    print(f"BENCHMARK COMPLETED. Comprehensive report generated at: {RESULTS_MD}")
    print("=" * 80)


if __name__ == '__main__':
    main()
