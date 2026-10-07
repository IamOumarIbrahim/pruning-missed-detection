import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def generate_plots():
    base_csv = Path('results/fixed_tau05_checkpoints_0to50.csv')
    further_csv = Path('results/fixed_tau05_checkpoints_further.csv')
    dfs = [pd.read_csv(base_csv)]
    if further_csv.exists() and further_csv.stat().st_size > 0:
        dfs.append(pd.read_csv(further_csv))
    df = pd.concat(dfs, ignore_index=True)
    df['sparsity_int'] = df['sparsity'].str.replace('%', '').astype(int)

    metrics = [
        ('macro_recall', 'Macro-Recall', '#1f77b4', '-'),
        ('worst_recall', 'Worst-Class Recall', '#d62728', '--'),
        ('yawning_rec', 'Yawning', '#2ca02c', ':'),
        ('hand_mouth_rec', 'Hand-over-Mouth', '#9467bd', ':'),
        ('drinking_rec', 'Drinking', '#ff7f0e', ':'),
        ('phone_use_rec', 'Phone-Use', '#8c564b', ':'),
    ]

    # Helper function to plot 1x2 panel for any x-metric
    def make_plot(x_col, x_label, scale, title_metric, filename):
        fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
        for ax_idx, arch in enumerate(['YOLO11N', 'YOLO26N']):
            ax = axes[ax_idx]
            sub = df[df['arch'] == arch].sort_values(x_col)
            sparsities = sorted(sub['sparsity_int'].unique())

            for col, label, color, ls in metrics:
                mean_x = []
                mean_y = []
                for s in sparsities:
                    s_rows = sub[sub['sparsity_int'] == s]
                    x_val = s_rows[x_col].mean() / scale
                    vals = s_rows[col] * 100.0
                    mean_x.append(x_val)
                    mean_y.append(vals.mean())
                    # Individual seed points
                    ax.scatter(s_rows[x_col] / scale, vals, color=color, alpha=0.35, s=20, edgecolors='none')
                ax.plot(mean_x, mean_y, label=label, color=color, linestyle=ls, linewidth=2.0)

            # Add nominal sparsity tags along Macro-Recall line
            for s in sparsities:
                s_rows = sub[sub['sparsity_int'] == s]
                x_val = s_rows[x_col].mean() / scale
                m_val = s_rows['macro_recall'].mean() * 100.0
                ax.annotate(f"{s}%", (x_val, m_val), textcoords="offset points", xytext=(0, 7),
                            ha='center', fontsize=8, color='#333333', fontweight='bold', alpha=0.85)

            ax.set_title(f'{arch} — Recall vs {title_metric} (Fixed tau = 0.5)', fontsize=13, fontweight='bold')
            ax.set_xlabel(x_label, fontsize=11)
            if ax_idx == 0:
                ax.set_ylabel('Recall (%)', fontsize=11)
            ax.set_ylim(50, 102)
            ax.grid(True, linestyle='--', alpha=0.6)
            if ax_idx == 0:
                ax.legend(loc='lower left', frameon=True, fontsize=9)

        plt.tight_layout()
        out_p = Path('results') / filename
        plt.savefig(out_p, dpi=300)
        plt.close()
        print(f"Plot saved to {out_p}")

    # Plot 1: Achieved GFLOPs
    make_plot('flops_g', 'Achieved GFLOPs', 1.0, 'Achieved GFLOPs', 'plot1_recall_vs_gflops.png')
    # Backward compatibility
    if (Path('results') / 'plot1_recall_vs_gflops.png').exists():
        import shutil
        shutil.copy2('results/plot1_recall_vs_gflops.png', 'results/plot1_recall_vs_sparsity.png')

    # Plot 2: Total Parameters
    make_plot('params', 'Total Parameters (Millions)', 1e6, 'Total Parameters', 'plot2_recall_vs_total_params.png')
    if (Path('results') / 'plot2_recall_vs_total_params.png').exists():
        import shutil
        shutil.copy2('results/plot2_recall_vs_total_params.png', 'results/plot2_recall_vs_complexity.png')

    # Plot 3: Backbone + Neck Parameters
    if 'bb_neck_params' in df.columns:
        make_plot('bb_neck_params', 'Backbone + Neck Parameters (Millions)', 1e6, 'Backbone + Neck Parameters', 'plot3_recall_vs_bb_neck_params.png')

if __name__ == '__main__':
    generate_plots()
