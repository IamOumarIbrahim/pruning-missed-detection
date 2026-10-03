"""Generate comprehensive, honest Table 1 incorporating False Positive counts and multi-threshold evaluation."""

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def generate_comprehensive_table1():
    with open('results/box_compensation_data.json', encoding='utf-8') as f:
        box_data = json.load(f)
    with open('results/table1_data.json', encoding='utf-8') as f:
        t1_data = json.load(f)

    lines = [
        "### Table 1: Comprehensive FP32 Pruning Sweep Across Operating Thresholds",
        "",
        "All values reported as mean ± sample std over K=3 independent seeds (ddof=1) on the test split (3,213 frames: 614 positive, 2,599 negative).",
        "- **Min Recall:** Worst-case class recall $\\min_{c \\in \\mathcal{C}} \\text{Recall}_c$ across safety classes (`phone_use`, `drinking`, `yawning`, `hand_over_mouth`).",
        "- **Precision:** Macro-averaged precision across classes at the corresponding threshold.",
        "- **Min Recall F1:** Harmonic mean $2 \\cdot \\frac{P \\cdot R_{\\min}}{P + R_{\\min}}$.",
        "- **FP (False Positives):** Total false alarms on the test set (measures bounding box proliferation).",
        "",
        "#### Part A: Standard Operational Points (τ*, τ=0.10, τ=0.25) & False Alarm Accounting",
        "",
        "| Model | Pruning | Mean τ* | Min Rec @ τ* | Prec @ τ* | F1 @ τ* | Min Rec @ 0.10 | Prec @ 0.10 | F1 @ 0.10 | FP @ 0.10 | Min Rec @ 0.25 | Prec @ 0.25 | F1 @ 0.25 | FP @ 0.25 |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    display = {'yolo11n': 'YOLO11n', 'yolo26n': 'YOLO26n'}

    for m in ['yolo11n', 'yolo26n']:
        m_name = display[m]
        for r in ['0%', '10%', '20%', '30%', '40%', '50%']:
            b_agg = box_data[m][r]['aggregated']
            t_agg = t1_data['aggregated'][m][r]

            t_star = f"{t_agg['tau_star']['mean']:.3f}"
            r_star = f"{t_agg['tau_star_min_recall']['mean']:.3f}±{t_agg['tau_star_min_recall']['std']:.3f}"
            p_star = f"{t_agg['tau_star_precision']['mean']:.3f}±{t_agg['tau_star_precision']['std']:.3f}"
            f1_star = f"{t_agg['tau_star_f1_harmonic']['mean']:.3f}"

            r_10 = f"{b_agg['tau_10']['min_recall']['mean']:.3f}±{b_agg['tau_10']['min_recall']['std']:.3f}"
            p_10 = f"{b_agg['tau_10']['macro_precision']['mean']:.3f}"
            f1_10 = f"{b_agg['tau_10']['f1_harmonic']['mean']:.3f}"
            fp_10 = f"{b_agg['tau_10']['total_fp']['mean']:.1f}"

            r_25 = f"{b_agg['tau_25']['min_recall']['mean']:.3f}±{b_agg['tau_25']['min_recall']['std']:.3f}"
            p_25 = f"{b_agg['tau_25']['macro_precision']['mean']:.3f}"
            f1_25 = f"{b_agg['tau_25']['f1_harmonic']['mean']:.3f}"
            fp_25 = f"{b_agg['tau_25']['total_fp']['mean']:.1f}"

            lines.append(f"| {m_name} | {r} | {t_star} | {r_star} | {p_star} | {f1_star} | {r_10} | {p_10} | {f1_10} | {fp_10} | {r_25} | {p_25} | {f1_25} | {fp_25} |")

    lines.extend([
        "",
        "#### Part B: High-Certainty Operating Points (τ=0.40, τ=0.50) — Isolating Intrinsic Capacity Loss",
        "*(At higher thresholds, noisy bounding boxes are filtered out, exposing the unmasked structural degradation caused by pruning)*",
        "",
        "| Model | Pruning | Min Rec @ 0.40 | Prec @ 0.40 | F1 @ 0.40 | FP @ 0.40 | Min Rec @ 0.50 | Prec @ 0.50 | F1 @ 0.50 | FP @ 0.50 |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])

    for m in ['yolo11n', 'yolo26n']:
        m_name = display[m]
        for r in ['0%', '10%', '20%', '30%', '40%', '50%']:
            b_agg = box_data[m][r]['aggregated']
            r_40 = f"{b_agg['tau_40']['min_recall']['mean']:.3f}±{b_agg['tau_40']['min_recall']['std']:.3f}"
            p_40 = f"{b_agg['tau_40']['macro_precision']['mean']:.3f}"
            f1_40 = f"{b_agg['tau_40']['f1_harmonic']['mean']:.3f}"
            fp_40 = f"{b_agg['tau_40']['total_fp']['mean']:.1f}"

            r_50 = f"{b_agg['tau_50']['min_recall']['mean']:.3f}±{b_agg['tau_50']['min_recall']['std']:.3f}"
            p_50 = f"{b_agg['tau_50']['macro_precision']['mean']:.3f}"
            f1_50 = f"{b_agg['tau_50']['f1_harmonic']['mean']:.3f}"
            fp_50 = f"{b_agg['tau_50']['total_fp']['mean']:.1f}"

            lines.append(f"| {m_name} | {r} | {r_40} | {p_40} | {f1_40} | {fp_40} | {r_50} | {p_50} | {f1_50} | {fp_50} |")

    return "\n".join(lines)


if __name__ == '__main__':
    print(generate_comprehensive_table1())
