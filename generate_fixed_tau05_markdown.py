import pandas as pd
import numpy as np
from pathlib import Path

df = pd.read_csv('results/fixed_tau05_checkpoints_0to50.csv')
output_md = Path('RESULTS_FIXED_TAU05.md')

lines = [
    "# Benchmark Results: Fixed Operating Point at $\\tau = 0.50$",
    "",
    "> **Protocol:** One fixed confidence threshold $\\tau = 0.50$ across every model, every seed, every sparsity level.  ",
    "> **No threshold tuning:** $\\tau$ is not tuned on validation or test. No other thresholds computed.  ",
    "> **Split:** Held-out test split (`subject_05`, `subject_10`, `subject_12`) only. Standard IoU=0.5 matching.  ",
    "",
    "---",
    "",
    "## 1. Summary Table: Mean ± SD across 3 Seeds (0% to 50% Sparsity)",
    "",
    "| Architecture | Sparsity | Parameters | GFLOPs | Model Size (MB) | Precision (%) | Yawning Rec (%) | Hand-Mouth Rec (%) | Drinking Rec (%) | Phone-Use Rec (%) | Macro-Recall (%) | Worst-Class Rec (%) |",
    "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
]

for arch in ['YOLO11N', 'YOLO26N']:
    sub_arch = df[df['arch'] == arch]
    for sparsity in ['0%', '10%', '20%', '30%', '40%', '50%']:
        sub = sub_arch[sub_arch['sparsity'] == sparsity]
        
        p_mean = sub['params'].iloc[0]
        f_mean = sub['flops_g'].iloc[0]
        s_mean = sub['size_mb'].iloc[0]
        
        prec_str = f"{sub['precision'].mean()*100:.1f} ± {sub['precision'].std()*100:.1f}%"
        yawn_str = f"{sub['yawning_rec'].mean()*100:.1f} ± {sub['yawning_rec'].std()*100:.1f}%"
        hand_str = f"{sub['hand_mouth_rec'].mean()*100:.1f} ± {sub['hand_mouth_rec'].std()*100:.1f}%"
        drink_str = f"{sub['drinking_rec'].mean()*100:.1f} ± {sub['drinking_rec'].std()*100:.1f}%"
        phone_str = f"{sub['phone_use_rec'].mean()*100:.1f} ± {sub['phone_use_rec'].std()*100:.1f}%"
        macro_str = f"{sub['macro_recall'].mean()*100:.1f} ± {sub['macro_recall'].std()*100:.1f}%"
        worst_str = f"{sub['worst_recall'].mean()*100:.1f} ± {sub['worst_recall'].std()*100:.1f}%"
        
        cond_str = "Baseline (0%)" if sparsity == '0%' else f"Pruned {sparsity}"
        lines.append(
            f"| {arch} | {cond_str} | {p_mean:,} | {f_mean:.2f} | {s_mean:.2f} | "
            f"{prec_str} | {yawn_str} | {hand_str} | {drink_str} | {phone_str} | {macro_str} | {worst_str} |"
        )

lines.extend([
    "",
    "---",
    "",
    "## 2. Per-Checkpoint Log (All 36 Checkpoints — 0% to 50%)",
    "",
    "| Arch | Sparsity | Seed | Params | GFLOPs | Size (MB) | Precision | Yawning | Hand-Mouth | Drinking | Phone-Use | Macro-Recall | Worst-Class Recall |",
    "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
])

for _, r in df.iterrows():
    cond_str = "Baseline" if r['sparsity'] == '0%' else f"Pruned {r['sparsity']}"
    lines.append(
        f"| {r['arch']} | {cond_str} | Seed {int(r['seed'])} | {int(r['params']):,} | {r['flops_g']:.2f} | {r['size_mb']:.2f} | "
        f"{r['precision']*100:.1f}% | {r['yawning_rec']*100:.1f}% | {r['hand_mouth_rec']*100:.1f}% | "
        f"{r['drinking_rec']*100:.1f}% | {r['phone_use_rec']*100:.1f}% | {r['macro_recall']*100:.1f}% | {r['worst_recall']*100:.1f}% |"
    )

lines.extend([
    "",
    "---",
    "",
    "## 3. Visualizations",
    "",
    "- **Plot 1: Recall vs Sparsity (Fixed $\\tau = 0.50$):** `results/plot1_recall_vs_sparsity.png`",
    "- **Plot 2: Recall vs Complexity (GFLOPs & Parameter Count):** `results/plot2_recall_vs_complexity.png`",
    "",
    "---",
    "",
    "## 4. Plain-Language Empirical Summary",
    "",
    "### Where Recall Starts to Drop (and Corresponding GFLOPs / Params):",
    "1. **YOLO11N:**",
    "   - **Baseline (0%):** 2.59M parameters, 6.50 GFLOPs $\\rightarrow$ Macro-Recall: **86.4%**, Worst-Class Recall: **80.8%** (`yawning` / `phone_use`).",
    "   - **20% Sparsity:** 1.94M parameters (-25%), 4.92 GFLOPs (-24%) $\\rightarrow$ Macro-Recall actually *peaks* at **88.0%** (+1.6 pp), Worst-Class: **80.1%**.",
    "   - **30% Sparsity:** 1.67M parameters (-36%), 4.30 GFLOPs (-34%) $\\rightarrow$ Macro-Recall: **87.2%**, Worst-Class: **79.0%**.",
    "   - **Onset of Drop (40% Sparsity):** At 1.43M parameters (-45%) and 3.74 GFLOPs (-42%), Macro-Recall experiences its first mild drop to **84.4%** (-2.0 pp) and Worst-Class drops to **76.8%** (driven by Seed 0 yawning at 69.7%).",
    "   - **50% Sparsity:** 1.23M parameters (-52%), 3.30 GFLOPs (-49%) $\\rightarrow$ Macro-Recall recovers to **85.6%** (-0.8 pp vs baseline), Worst-Class: **78.7%**.",
    "",
    "2. **YOLO26N:**",
    "   - **Baseline (0%):** 2.51M parameters, 5.90 GFLOPs $\\rightarrow$ Macro-Recall: **85.8%**, Worst-Class Recall: **73.7%** (`yawning`).",
    "   - **10% Sparsity:** 2.18M parameters (-13%), 5.03 GFLOPs (-15%) $\\rightarrow$ Immediate drop in yawning recall to **64.6%** (Seed 1 dropped to 57.6%), reducing Macro-Recall to **81.4%** (-4.4 pp).",
    "   - **20% to 50% Sparsity:** Recovers and plateaus at **83.8% – 84.6%** Macro-Recall and **71.7% – 73.7%** Worst-Class Recall down to 1.25M parameters (-50%) and 2.73 GFLOPs (-54%).",
    "",
    "---",
    "",
    "## 5. Protocol for Step 2 (60%, 70%, 80%, 90% Further Pruning)",
    "",
    "> **Status:** Step 1 (0% to 50%) is complete. Step 2 requires executing 24 fine-tuning runs (each 100 epochs on 9,087 images, ~2.5 hours per run, totaling ~60 GPU hours).",
    "",
    "- **Pruning Algorithm:** Global L1-norm structured channel pruning via `torch_pruning.pruner.GroupNormPruner` with `GroupMagnitudeImportance(p=1)`.",
    "- **Excluded Layers:** Output detection heads (`Detect`), spatial attention (`C2PSA`, `Attention`), and layers with $< 8$ output channels.",
    "- **Fine-Tuning Recipe:** `PrunedDetectionTrainer` on `configs/dmd_rgb.yaml`, 100 epochs, batch size 16, imgsz 640, learning rate `lr0=0.01`, `lrf=0.01`, optimizer `auto`, AMP enabled, patience 0.",
])

with open(output_md, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')

print(f"Written to {output_md}")
