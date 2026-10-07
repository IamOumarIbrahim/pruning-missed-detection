# Benchmark Results: Subject-Disjoint Validation-Tuned vs Frozen Test Deployments

> **Status:** COMPLETED (36/36 models) — Last Updated: `2026-10-04 19:27:18`  
> **Methodology:** Raw detections extracted at `conf=0.001` on CUDA:0 and evaluated offline.  
> **Partitions:** Validation split (`subject_02, 03, 11`), Held-out test split (`subject_05, 10, 12`).  

---

## Table 1: Protocol 1 — Val-Tuned $\tau_{\text{val}}$ Frozen on Test Split

> **Rule:** Operating threshold $\tau_{\text{val}} \ge 0.35$ tuned on validation split where $P_{\text{val}} \ge 0.90$.  
> Frozen $\tau_{\text{val}}$ evaluated on test split. Transfer Gap = $P_{\text{test}} - P_{\text{val}}$.  
> `[BINDING]` indicates the 0.35 confidence floor was active.  

### Summary: Mean ± SD across 3 Seeds (Protocol 1)

| Architecture | Condition | $\tau_{\text{val}}$ | Val Prec (%) | Test Prec (%) | Transfer Gap (pp) | Macro-Rec @ $\tau_{\text{val}}$ (%) | Base Worst-Cls Rec (%) | Mean $\Delta$ vs Base | Floor Binding |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| YOLO11N | Baseline (0%) | 0.3500 ± 0.0000 | 95.5 ± 1.3% | 91.3 ± 2.0% | -4.18 ± 2.21 pp | 89.7 ± 1.0% | 85.2 ± 0.8% | — | 3/3 (100%) |
| YOLO11N | Pruned 10% | 0.3500 ± 0.0000 | 96.3 ± 1.5% | 91.3 ± 1.4% | -4.96 ± 0.33 pp | 88.6 ± 2.2% | 82.9 ± 1.9% | -1.12 ± 1.99 pp | 3/3 (100%) |
| YOLO11N | Pruned 20% | 0.3500 ± 0.0000 | 96.0 ± 0.3% | 91.5 ± 1.8% | -4.43 ± 1.93 pp | 90.2 ± 1.2% | 84.1 ± 1.9% | +0.53 ± 2.16 pp | 3/3 (100%) |
| YOLO11N | Pruned 30% | 0.3500 ± 0.0000 | 96.0 ± 0.8% | 92.1 ± 1.2% | -3.91 ± 1.39 pp | 90.9 ± 1.4% | 83.0 ± 2.3% | +1.19 ± 0.49 pp | 3/3 (100%) |
| YOLO11N | Pruned 40% | 0.3500 ± 0.0000 | 94.5 ± 1.6% | 91.8 ± 1.6% | -2.66 ± 1.34 pp | 90.4 ± 1.0% | 84.0 ± 0.3% | +0.75 ± 1.15 pp | 3/3 (100%) |
| YOLO11N | Pruned 50% | 0.3500 ± 0.0000 | 96.6 ± 0.9% | 90.2 ± 2.1% | -6.48 ± 2.80 pp | 89.6 ± 1.6% | 84.7 ± 1.0% | -0.07 ± 1.43 pp | 3/3 (100%) |
| YOLO26N | Baseline (0%) | 0.3500 ± 0.0000 | 96.1 ± 0.9% | 93.0 ± 0.8% | -3.13 ± 1.11 pp | 89.4 ± 1.0% | 82.0 ± 3.3% | — | 3/3 (100%) |
| YOLO26N | Pruned 10% | 0.3500 ± 0.0000 | 96.8 ± 1.0% | 93.6 ± 1.5% | -3.24 ± 0.94 pp | 85.2 ± 2.1% | 73.8 ± 9.3% | -4.21 ± 2.23 pp | 3/3 (100%) |
| YOLO26N | Pruned 20% | 0.3500 ± 0.0000 | 97.2 ± 0.5% | 93.4 ± 0.2% | -3.83 ± 0.25 pp | 87.1 ± 2.9% | 80.8 ± 7.6% | -2.34 ± 2.25 pp | 3/3 (100%) |
| YOLO26N | Pruned 30% | 0.3500 ± 0.0000 | 96.8 ± 0.8% | 92.9 ± 2.6% | -3.89 ± 1.86 pp | 86.6 ± 3.2% | 82.7 ± 4.7% | -2.84 ± 3.79 pp | 3/3 (100%) |
| YOLO26N | Pruned 40% | 0.3500 ± 0.0000 | 95.9 ± 0.9% | 91.9 ± 1.1% | -3.98 ± 2.01 pp | 87.3 ± 2.2% | 80.7 ± 6.4% | -2.15 ± 2.94 pp | 3/3 (100%) |
| YOLO26N | Pruned 50% | 0.3500 ± 0.0000 | 96.5 ± 0.6% | 92.1 ± 0.9% | -4.39 ± 1.53 pp | 87.5 ± 0.9% | 82.5 ± 1.2% | -1.92 ± 0.63 pp | 3/3 (100%) |

### Per-Checkpoint Log: Protocol 1 (All 36 Checkpoints)

| Model | Condition | Seed | $\tau_{\text{val}}$ | Floor Binding | Val Prec | Test Prec | Transfer Gap | Macro-Rec @ $\tau_{\text{val}}$ | Base Worst-Cls Rec (TP/GT) | Min Cls Rec (TP/GT) | $\Delta$ vs Base |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| YOLO11N | Baseline | Seed 0 | 0.3500 | **YES [BINDING]** | 94.2% | 89.8% | -4.33 pp | 89.6% | phone_use (424/497 = 85.3%) | phone_use (424/497 = 85.3%) | — |
| YOLO11N | Baseline | Seed 1 | 0.3500 | **YES [BINDING]** | 95.5% | 93.6% | -1.89 pp | 88.7% | phone_use (419/497 = 84.3%) | phone_use (419/497 = 84.3%) | — |
| YOLO11N | Baseline | Seed 2 | 0.3500 | **YES [BINDING]** | 96.8% | 90.5% | -6.30 pp | 90.8% | phone_use (427/497 = 85.9%) | phone_use (427/497 = 85.9%) | — |
| YOLO11N | Pruned 10% | Seed 0 | 0.3500 | **YES [BINDING]** | 94.7% | 89.8% | -4.90 pp | 86.2% | phone_use (407/497 = 81.9%) | phone_use (407/497 = 81.9%) | -3.40 pp |
| YOLO11N | Pruned 10% | Seed 1 | 0.3500 | **YES [BINDING]** | 97.8% | 92.4% | -5.32 pp | 89.0% | phone_use (406/497 = 81.7%) | phone_use (406/497 = 81.7%) | +0.24 pp |
| YOLO11N | Pruned 10% | Seed 2 | 0.3500 | **YES [BINDING]** | 96.4% | 91.7% | -4.68 pp | 90.6% | phone_use (423/497 = 85.1%) | phone_use (423/497 = 85.1%) | -0.20 pp |
| YOLO11N | Pruned 20% | Seed 0 | 0.3500 | **YES [BINDING]** | 95.8% | 90.5% | -5.27 pp | 89.7% | phone_use (409/497 = 82.3%) | phone_use (409/497 = 82.3%) | +0.14 pp |
| YOLO11N | Pruned 20% | Seed 1 | 0.3500 | **YES [BINDING]** | 95.8% | 93.6% | -2.22 pp | 91.6% | phone_use (428/497 = 86.1%) | phone_use (428/497 = 86.1%) | +2.86 pp |
| YOLO11N | Pruned 20% | Seed 2 | 0.3500 | **YES [BINDING]** | 96.3% | 90.5% | -5.81 pp | 89.4% | phone_use (417/497 = 83.9%) | phone_use (417/497 = 83.9%) | -1.40 pp |
| YOLO11N | Pruned 30% | Seed 0 | 0.3500 | **YES [BINDING]** | 96.4% | 90.9% | -5.51 pp | 90.4% | phone_use (403/497 = 81.1%) | phone_use (403/497 = 81.1%) | +0.77 pp |
| YOLO11N | Pruned 30% | Seed 1 | 0.3500 | **YES [BINDING]** | 95.2% | 92.0% | -3.12 pp | 89.8% | phone_use (425/497 = 85.5%) | phone_use (425/497 = 85.5%) | +1.06 pp |
| YOLO11N | Pruned 30% | Seed 2 | 0.3500 | **YES [BINDING]** | 96.5% | 93.4% | -3.10 pp | 92.5% | phone_use (410/497 = 82.5%) | phone_use (410/497 = 82.5%) | +1.73 pp |
| YOLO11N | Pruned 40% | Seed 0 | 0.3500 | **YES [BINDING]** | 93.9% | 90.1% | -3.82 pp | 89.3% | phone_use (419/497 = 84.3%) | yawning (27/33 = 81.8%) | -0.29 pp |
| YOLO11N | Pruned 40% | Seed 1 | 0.3500 | **YES [BINDING]** | 93.3% | 92.1% | -1.19 pp | 90.7% | phone_use (416/497 = 83.7%) | phone_use (416/497 = 83.7%) | +1.99 pp |
| YOLO11N | Pruned 40% | Seed 2 | 0.3500 | **YES [BINDING]** | 96.2% | 93.3% | -2.96 pp | 91.3% | phone_use (417/497 = 83.9%) | phone_use (417/497 = 83.9%) | +0.57 pp |
| YOLO11N | Pruned 50% | Seed 0 | 0.3500 | **YES [BINDING]** | 96.8% | 88.0% | -8.85 pp | 91.2% | phone_use (426/497 = 85.7%) | phone_use (426/497 = 85.7%) | +1.58 pp |
| YOLO11N | Pruned 50% | Seed 1 | 0.3500 | **YES [BINDING]** | 97.4% | 90.2% | -7.21 pp | 87.9% | phone_use (421/497 = 84.7%) | phone_use (421/497 = 84.7%) | -0.79 pp |
| YOLO11N | Pruned 50% | Seed 2 | 0.3500 | **YES [BINDING]** | 95.6% | 92.2% | -3.39 pp | 89.8% | phone_use (416/497 = 83.7%) | phone_use (416/497 = 83.7%) | -1.00 pp |
| YOLO26N | Baseline | Seed 0 | 0.3500 | **YES [BINDING]** | 96.5% | 92.2% | -4.33 pp | 90.2% | phone_use (424/497 = 85.3%) | phone_use (424/497 = 85.3%) | — |
| YOLO26N | Baseline | Seed 1 | 0.3500 | **YES [BINDING]** | 95.1% | 93.0% | -2.15 pp | 89.8% | yawning (27/33 = 81.8%) | yawning (27/33 = 81.8%) | — |
| YOLO26N | Baseline | Seed 2 | 0.3500 | **YES [BINDING]** | 96.7% | 93.8% | -2.90 pp | 88.2% | yawning (26/33 = 78.8%) | yawning (26/33 = 78.8%) | — |
| YOLO26N | Pruned 10% | Seed 0 | 0.3500 | **YES [BINDING]** | 97.6% | 93.8% | -3.82 pp | 87.2% | phone_use (407/497 = 81.9%) | phone_use (407/497 = 81.9%) | -2.95 pp |
| YOLO26N | Pruned 10% | Seed 1 | 0.3500 | **YES [BINDING]** | 97.1% | 95.0% | -2.16 pp | 83.1% | yawning (21/33 = 63.6%) | yawning (21/33 = 63.6%) | -6.78 pp |
| YOLO26N | Pruned 10% | Seed 2 | 0.3500 | **YES [BINDING]** | 95.7% | 92.0% | -3.74 pp | 85.4% | yawning (25/33 = 75.8%) | yawning (25/33 = 75.8%) | -2.90 pp |
| YOLO26N | Pruned 20% | Seed 0 | 0.3500 | **YES [BINDING]** | 97.6% | 93.6% | -4.02 pp | 87.0% | phone_use (436/497 = 87.7%) | yawning (24/33 = 72.7%) | -3.18 pp |
| YOLO26N | Pruned 20% | Seed 1 | 0.3500 | **YES [BINDING]** | 97.3% | 93.3% | -3.92 pp | 90.0% | yawning (27/33 = 81.8%) | yawning (27/33 = 81.8%) | +0.20 pp |
| YOLO26N | Pruned 20% | Seed 2 | 0.3500 | **YES [BINDING]** | 96.7% | 93.1% | -3.55 pp | 84.2% | yawning (24/33 = 72.7%) | yawning (24/33 = 72.7%) | -4.05 pp |
| YOLO26N | Pruned 30% | Seed 0 | 0.3500 | **YES [BINDING]** | 97.4% | 94.2% | -3.25 pp | 83.0% | phone_use (404/497 = 81.3%) | yawning (22/33 = 66.7%) | -7.20 pp |
| YOLO26N | Pruned 30% | Seed 1 | 0.3500 | **YES [BINDING]** | 97.0% | 94.6% | -2.43 pp | 88.8% | yawning (29/33 = 87.9%) | phone_use (415/497 = 83.5%) | -1.02 pp |
| YOLO26N | Pruned 30% | Seed 2 | 0.3500 | **YES [BINDING]** | 95.8% | 89.8% | -5.99 pp | 87.9% | yawning (26/33 = 78.8%) | yawning (26/33 = 78.8%) | -0.31 pp |
| YOLO26N | Pruned 40% | Seed 0 | 0.3500 | **YES [BINDING]** | 96.8% | 90.8% | -6.00 pp | 84.7% | phone_use (390/497 = 78.5%) | phone_use (390/497 = 78.5%) | -5.46 pp |
| YOLO26N | Pruned 40% | Seed 1 | 0.3500 | **YES [BINDING]** | 94.9% | 92.9% | -1.99 pp | 88.7% | yawning (25/33 = 75.8%) | yawning (25/33 = 75.8%) | -1.17 pp |
| YOLO26N | Pruned 40% | Seed 2 | 0.3500 | **YES [BINDING]** | 96.1% | 92.1% | -3.95 pp | 88.4% | yawning (29/33 = 87.9%) | phone_use (407/497 = 81.9%) | +0.18 pp |
| YOLO26N | Pruned 50% | Seed 0 | 0.3500 | **YES [BINDING]** | 96.7% | 92.2% | -4.45 pp | 87.5% | phone_use (417/497 = 83.9%) | yawning (26/33 = 78.8%) | -2.62 pp |
| YOLO26N | Pruned 50% | Seed 1 | 0.3500 | **YES [BINDING]** | 95.7% | 92.9% | -2.83 pp | 88.4% | yawning (27/33 = 81.8%) | yawning (27/33 = 81.8%) | -1.44 pp |
| YOLO26N | Pruned 50% | Seed 2 | 0.3500 | **YES [BINDING]** | 97.0% | 91.1% | -5.88 pp | 86.6% | yawning (27/33 = 81.8%) | yawning (27/33 = 81.8%) | -1.68 pp |

---

## Table 2: Protocol 2 — Baseline Checkpoint's $\tau_{\text{base}}$ Frozen Across All Pruning Levels

> **Rule:** For each architecture and seed, freeze the baseline's val-tuned threshold $\tau_{\text{base}}$ and evaluate pruned models on test split.  

### Summary: Mean ± SD across 3 Seeds (Protocol 2)

| Architecture | Condition | Frozen $\tau_{\text{base}}$ | Test Prec (%) | Macro-Rec @ $\tau_{\text{base}}$ (%) | Base Worst-Cls Rec (%) | Mean $\Delta$ vs Base |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| YOLO11N | Baseline (0%) | 0.3500 ± 0.0000 | 91.3 ± 2.0% | 89.7 ± 1.0% | 85.2 ± 0.8% | — |
| YOLO11N | Pruned 10% | 0.3500 ± 0.0000 | 91.3 ± 1.4% | 88.6 ± 2.2% | 82.9 ± 1.9% | -1.12 ± 1.99 pp |
| YOLO11N | Pruned 20% | 0.3500 ± 0.0000 | 91.5 ± 1.8% | 90.2 ± 1.2% | 84.1 ± 1.9% | +0.53 ± 2.16 pp |
| YOLO11N | Pruned 30% | 0.3500 ± 0.0000 | 92.1 ± 1.2% | 90.9 ± 1.4% | 83.0 ± 2.3% | +1.19 ± 0.49 pp |
| YOLO11N | Pruned 40% | 0.3500 ± 0.0000 | 91.8 ± 1.6% | 90.4 ± 1.0% | 84.0 ± 0.3% | +0.75 ± 1.15 pp |
| YOLO11N | Pruned 50% | 0.3500 ± 0.0000 | 90.2 ± 2.1% | 89.6 ± 1.6% | 84.7 ± 1.0% | -0.07 ± 1.43 pp |
| YOLO26N | Baseline (0%) | 0.3500 ± 0.0000 | 93.0 ± 0.8% | 89.4 ± 1.0% | 82.0 ± 3.3% | — |
| YOLO26N | Pruned 10% | 0.3500 ± 0.0000 | 93.6 ± 1.5% | 85.2 ± 2.1% | 73.8 ± 9.3% | -4.21 ± 2.23 pp |
| YOLO26N | Pruned 20% | 0.3500 ± 0.0000 | 93.4 ± 0.2% | 87.1 ± 2.9% | 80.8 ± 7.6% | -2.34 ± 2.25 pp |
| YOLO26N | Pruned 30% | 0.3500 ± 0.0000 | 92.9 ± 2.6% | 86.6 ± 3.2% | 82.7 ± 4.7% | -2.84 ± 3.79 pp |
| YOLO26N | Pruned 40% | 0.3500 ± 0.0000 | 91.9 ± 1.1% | 87.3 ± 2.2% | 80.7 ± 6.4% | -2.15 ± 2.94 pp |
| YOLO26N | Pruned 50% | 0.3500 ± 0.0000 | 92.1 ± 0.9% | 87.5 ± 0.9% | 82.5 ± 1.2% | -1.92 ± 0.63 pp |

---

## Table 3: Protocol 3 — Fixed Operating Point at $\tau = 0.35$

> **Rule:** Fixed unadjusted threshold $\tau = 0.35$ on held-out test split.  

### Summary: Mean ± SD across 3 Seeds (Protocol 3)

| Architecture | Condition | Test Prec (%) | Macro-Rec @ 0.35 (%) | Base Worst-Cls Rec (%) | Mean $\Delta$ vs Base |
| :--- | :--- | :--- | :--- | :--- | :--- |
| YOLO11N | Baseline (0%) | 91.3 ± 2.0% | 89.7 ± 1.0% | 85.2 ± 0.8% | — |
| YOLO11N | Pruned 10% | 91.3 ± 1.4% | 88.6 ± 2.2% | 82.9 ± 1.9% | -1.12 ± 1.99 pp |
| YOLO11N | Pruned 20% | 91.5 ± 1.8% | 90.2 ± 1.2% | 84.1 ± 1.9% | +0.53 ± 2.16 pp |
| YOLO11N | Pruned 30% | 92.1 ± 1.2% | 90.9 ± 1.4% | 83.0 ± 2.3% | +1.19 ± 0.49 pp |
| YOLO11N | Pruned 40% | 91.8 ± 1.6% | 90.4 ± 1.0% | 84.0 ± 0.3% | +0.75 ± 1.15 pp |
| YOLO11N | Pruned 50% | 90.2 ± 2.1% | 89.6 ± 1.6% | 84.7 ± 1.0% | -0.07 ± 1.43 pp |
| YOLO26N | Baseline (0%) | 93.0 ± 0.8% | 89.4 ± 1.0% | 82.0 ± 3.3% | — |
| YOLO26N | Pruned 10% | 93.6 ± 1.5% | 85.2 ± 2.1% | 73.8 ± 9.3% | -4.21 ± 2.23 pp |
| YOLO26N | Pruned 20% | 93.4 ± 0.2% | 87.1 ± 2.9% | 80.8 ± 7.6% | -2.34 ± 2.25 pp |
| YOLO26N | Pruned 30% | 92.9 ± 2.6% | 86.6 ± 3.2% | 82.7 ± 4.7% | -2.84 ± 3.79 pp |
| YOLO26N | Pruned 40% | 91.9 ± 1.1% | 87.3 ± 2.2% | 80.7 ± 6.4% | -2.15 ± 2.94 pp |
| YOLO26N | Pruned 50% | 92.1 ± 0.9% | 87.5 ± 0.9% | 82.5 ± 1.2% | -1.92 ± 0.63 pp |

---

## Table 4: Labeled Reference — Oracle Test-Matched Benchmark ($P \ge 0.90$, Floor $\tau \ge 0.35$)

> **Rule (Oracle Reference):** Lowest confidence threshold $\tau_{@P90} \ge 0.35$ evaluated directly on test split where $P_{\text{test}} \ge 0.90$.  

### Summary: Mean ± SD across 3 Seeds (Oracle Reference)

| Architecture | Condition | $\tau_{@P90}$ | Test Prec (%) | Macro-Rec @ P90 (%) | Worst-Class Rec @ P90 (%) | Mean $\Delta$ vs Base |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| YOLO11N | Baseline (0%) | 0.3555 ± 0.0095 | 91.4 ± 1.9% | 89.7 ± 1.0% | 85.0 ± 0.8% | — |
| YOLO11N | Pruned 10% | 0.3621 ± 0.0209 | 91.4 ± 1.2% | 88.6 ± 2.2% | 82.8 ± 2.0% | -1.10 ± 1.96 pp |
| YOLO11N | Pruned 20% | 0.3500 ± 0.0000 | 91.5 ± 1.8% | 90.2 ± 1.2% | 84.1 ± 1.9% | +0.57 ± 2.15 pp |
| YOLO11N | Pruned 30% | 0.3500 ± 0.0000 | 92.1 ± 1.2% | 90.9 ± 1.4% | 83.0 ± 2.3% | +1.22 ± 0.45 pp |
| YOLO11N | Pruned 40% | 0.3500 ± 0.0000 | 91.8 ± 1.6% | 90.4 ± 1.0% | 83.1 ± 1.2% | +0.79 ± 1.11 pp |
| YOLO11N | Pruned 50% | 0.3804 ± 0.0527 | 90.8 ± 1.2% | 89.2 ± 1.1% | 84.3 ± 0.5% | -0.44 ± 0.80 pp |
| YOLO26N | Baseline (0%) | 0.3500 ± 0.0000 | 93.0 ± 0.8% | 89.4 ± 1.0% | 82.0 ± 3.3% | — |
| YOLO26N | Pruned 10% | 0.3500 ± 0.0000 | 93.6 ± 1.5% | 85.2 ± 2.1% | 73.8 ± 9.3% | -4.21 ± 2.23 pp |
| YOLO26N | Pruned 20% | 0.3500 ± 0.0000 | 93.4 ± 0.2% | 87.1 ± 2.9% | 75.8 ± 5.2% | -2.34 ± 2.25 pp |
| YOLO26N | Pruned 30% | 0.3507 ± 0.0012 | 92.9 ± 2.5% | 86.6 ± 3.2% | 76.3 ± 8.7% | -2.84 ± 3.79 pp |
| YOLO26N | Pruned 40% | 0.3500 ± 0.0000 | 91.9 ± 1.1% | 87.3 ± 2.2% | 78.7 ± 3.1% | -2.15 ± 2.94 pp |
| YOLO26N | Pruned 50% | 0.3500 ± 0.0000 | 92.1 ± 0.9% | 87.5 ± 0.9% | 80.8 ± 1.7% | -1.92 ± 0.63 pp |
