import json
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

with open('results/box_compensation_data.json', encoding='utf-8') as f:
    data = json.load(f)

for m in ['yolo11n', 'yolo26n']:
    print(f"\n================ {m.upper()} ================")
    print("Ratio | tau=0.10 MinRec | tau=0.10 FP | tau=0.25 MinRec | tau=0.25 FP | tau=0.40 MinRec | tau=0.40 FP | tau=0.50 MinRec | tau=0.50 FP")
    print("---|---|---|---|---|---|---|---|---")
    for r in ['0%', '10%', '20%', '30%', '40%', '50%']:
        agg = data[m][r]['aggregated']
        r10 = f"{agg['tau_10']['min_recall']['mean']:.3f}±{agg['tau_10']['min_recall']['std']:.3f}"
        fp10 = f"{agg['tau_10']['total_fp']['mean']:.1f}"
        r25 = f"{agg['tau_25']['min_recall']['mean']:.3f}±{agg['tau_25']['min_recall']['std']:.3f}"
        fp25 = f"{agg['tau_25']['total_fp']['mean']:.1f}"
        r40 = f"{agg['tau_40']['min_recall']['mean']:.3f}±{agg['tau_40']['min_recall']['std']:.3f}"
        fp40 = f"{agg['tau_40']['total_fp']['mean']:.1f}"
        r50 = f"{agg['tau_50']['min_recall']['mean']:.3f}±{agg['tau_50']['min_recall']['std']:.3f}"
        fp50 = f"{agg['tau_50']['total_fp']['mean']:.1f}"
        print(f"{r:4s}  | {r10:15s} | {fp10:11s} | {r25:15s} | {fp25:11s} | {r40:15s} | {fp40:11s} | {r50:15s} | {fp50:11s}")

    print("\nRatio | P@0.10 | F1@0.10 | P@0.25 | F1@0.25 | P@0.40 | F1@0.40 | P@0.50 | F1@0.50")
    print("---|---|---|---|---|---|---|---|---")
    for r in ['0%', '10%', '20%', '30%', '40%', '50%']:
        agg = data[m][r]['aggregated']
        p10 = f"{agg['tau_10']['macro_precision']['mean']:.3f}"
        f1_10 = f"{agg['tau_10']['f1_harmonic']['mean']:.3f}"
        p25 = f"{agg['tau_25']['macro_precision']['mean']:.3f}"
        f1_25 = f"{agg['tau_25']['f1_harmonic']['mean']:.3f}"
        p40 = f"{agg['tau_40']['macro_precision']['mean']:.3f}"
        f1_40 = f"{agg['tau_40']['f1_harmonic']['mean']:.3f}"
        p50 = f"{agg['tau_50']['macro_precision']['mean']:.3f}"
        f1_50 = f"{agg['tau_50']['f1_harmonic']['mean']:.3f}"
        print(f"{r:4s}  | {p10:6s} | {f1_10:7s} | {p25:6s} | {f1_25:7s} | {p40:6s} | {f1_40:7s} | {p50:6s} | {f1_50:7s}")
