"""Format Table 1 in markdown from results/table1_data.json."""

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def generate_table1_markdown():
    json_path = Path('results') / 'table1_data.json'
    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)

    agg = data['aggregated']

    def fmt(m, s):
        if s == 0.0 or s is None:
            return f"{m:.3f}"
        return f"{m:.3f} ± {s:.3f}"

    lines = [
        "### Table 1: FP32 Pruning Sweep Across Operating Thresholds (τ*, τ=0.10, τ=0.25)",
        "",
        "Reported as mean ± sample standard deviation over K=3 seeds (ddof=1).",
        "- **Worst-Case Safety Recall (Min Recall)**: $\\min_{c \\in \\mathcal{C}} \\text{Recall}_c$ across safety classes (`phone_use`, `drinking`, `yawning`, `hand_over_mouth`).",
        "- **Precision**: Macro-averaged precision across classes at the corresponding threshold.",
        "- **Min Recall F1**: Harmonic mean $2 \\cdot \\frac{\\text{Precision} \\cdot \\text{Min Recall}}{\\text{Precision} + \\text{Min Recall}}$.",
        "",
        "| Model | Pruning | Mean τ* | Min Recall @ τ* | Precision @ τ* | Min Recall F1 @ τ* | Min Recall @ τ=0.10 | Precision @ τ=0.10 | Min Recall F1 @ τ=0.10 | Min Recall @ τ=0.25 | Precision @ τ=0.25 | Min Recall F1 @ τ=0.25 |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    display_names = {'yolo11n': 'YOLO11n', 'yolo26n': 'YOLO26n'}

    for model_key, ratio_dict in agg.items():
        m_name = display_names.get(model_key, model_key)
        for r_label, r_data in ratio_dict.items():
            t_str = f"{r_data['tau_star']['mean']:.3f}"
            r_star = fmt(r_data['tau_star_min_recall']['mean'], r_data['tau_star_min_recall']['std'])
            p_star = fmt(r_data['tau_star_precision']['mean'], r_data['tau_star_precision']['std'])
            f1_star = fmt(r_data['tau_star_f1_harmonic']['mean'], r_data['tau_star_f1_harmonic']['std'])

            r_10 = fmt(r_data['tau_10_min_recall']['mean'], r_data['tau_10_min_recall']['std'])
            p_10 = fmt(r_data['tau_10_precision']['mean'], r_data['tau_10_precision']['std'])
            f1_10 = fmt(r_data['tau_10_f1_harmonic']['mean'], r_data['tau_10_f1_harmonic']['std'])

            r_25 = fmt(r_data['tau_25_min_recall']['mean'], r_data['tau_25_min_recall']['std'])
            p_25 = fmt(r_data['tau_25_precision']['mean'], r_data['tau_25_precision']['std'])
            f1_25 = fmt(r_data['tau_25_f1_harmonic']['mean'], r_data['tau_25_f1_harmonic']['std'])

            lines.append(
                f"| {m_name} | {r_label} | {t_str} | {r_star} | {p_star} | {f1_star} | {r_10} | {p_10} | {f1_10} | {r_25} | {p_25} | {f1_25} |"
            )

    return "\n".join(lines)


if __name__ == '__main__':
    md = generate_table1_markdown()
    print(md)
