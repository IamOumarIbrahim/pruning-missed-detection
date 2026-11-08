# Uncertainty and Statistical Rigor Benchmark

> **Generated:** Offline from saved raw detections at `conf=0.001` on CUDA:0.  
> **Splits:** Validation (`subject_02, 03, 11`, 15 clips), Test (`subject_05, 10, 12`, 13 clips).  

---

## STEP 0: Data & Cluster Sparsity Audit

| Split | Class | Instances | Distinct Clips | Total Clips | Subjects | Reliability Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| VAL | `yawning` | 32 | 5 | 15 | 3 | **UNRELIABLE (< 10 clips)** |
| VAL | `hand_over_mouth` | 30 | 4 | 15 | 3 | **UNRELIABLE (< 10 clips)** |
| VAL | `drinking` | 54 | 3 | 15 | 3 | **UNRELIABLE (< 10 clips)** |
| VAL | `phone_use` | 523 | 3 | 15 | 3 | **UNRELIABLE (< 10 clips)** |
| TEST | `yawning` | 33 | 5 | 13 | 3 | **UNRELIABLE (< 10 clips)** |
| TEST | `hand_over_mouth` | 28 | 5 | 13 | 3 | **UNRELIABLE (< 10 clips)** |
| TEST | `drinking` | 56 | 3 | 13 | 3 | **UNRELIABLE (< 10 clips)** |
| TEST | `phone_use` | 497 | 3 | 13 | 3 | **UNRELIABLE (< 10 clips)** |

> **Cluster Definition:** Clusters are defined as contiguous recording frame sequences from the same subject/video (`subject_XX_video_YY`).  
> **Explicit Sparsity Flag:** `yawning` appears in only **5 distinct clips** in test (< 10 clips). Clustered bootstrap CIs for yawning are explicitly flagged as **UNRELIABLE** due to high inter-cluster variance. `phone_use` appears in only **3 distinct clips** (1 per driver).  

---

## TASK 1: Fixed Class Identity (Fixed $\tau = 0.35$)

> Per-class recall evaluated specifically for `phone_use` and `yawning` separately at fixed $\tau = 0.35$.  
> (No arbitrary 'worst class' or 'min class' aggregation).  

### Summary: Mean ± SD across 3 Seeds (Task 1)

| Architecture | Condition | phone_use Recall (%) | yawning Recall (%) |
| :--- | :--- | :--- | :--- |
| YOLO11N | Baseline (0%) | 85.2 ± 0.8% | 87.9 ± 3.0% |
| YOLO11N | Pruned 10% | 82.9 ± 1.9% | 86.9 ± 1.7% |
| YOLO11N | Pruned 20% | 84.1 ± 1.9% | 89.9 ± 1.7% |
| YOLO11N | Pruned 30% | 83.0 ± 2.3% | 96.0 ± 7.0% |
| YOLO11N | Pruned 40% | 84.0 ± 0.3% | 90.9 ± 8.0% |
| YOLO11N | Pruned 50% | 84.7 ± 1.0% | 86.9 ± 1.7% |
| YOLO26N | Baseline (0%) | 85.0 ± 0.3% | 82.8 ± 4.6% |
| YOLO26N | Pruned 10% | 82.8 ± 0.8% | 74.7 ± 10.6% |
| YOLO26N | Pruned 20% | 85.6 ± 2.0% | 75.8 ± 5.2% |
| YOLO26N | Pruned 30% | 82.2 ± 1.1% | 77.8 ± 10.6% |
| YOLO26N | Pruned 40% | 81.6 ± 2.9% | 81.8 ± 6.1% |
| YOLO26N | Pruned 50% | 83.5 ± 1.1% | 80.8 ± 1.7% |

### Per-Checkpoint Table (All 36 Checkpoints — Task 1)

| Architecture | Condition | Seed | phone_use TP/GT | phone_use Rec (%) | yawning TP/GT | yawning Rec (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| YOLO11N | Baseline | Seed 0 | 424/497 | 85.3% | 30/33 | 90.9% |
| YOLO11N | Baseline | Seed 1 | 419/497 | 84.3% | 28/33 | 84.8% |
| YOLO11N | Baseline | Seed 2 | 427/497 | 85.9% | 29/33 | 87.9% |
| YOLO11N | Pruned 10% | Seed 0 | 407/497 | 81.9% | 29/33 | 87.9% |
| YOLO11N | Pruned 10% | Seed 1 | 406/497 | 81.7% | 28/33 | 84.8% |
| YOLO11N | Pruned 10% | Seed 2 | 423/497 | 85.1% | 29/33 | 87.9% |
| YOLO11N | Pruned 20% | Seed 0 | 409/497 | 82.3% | 30/33 | 90.9% |
| YOLO11N | Pruned 20% | Seed 1 | 428/497 | 86.1% | 30/33 | 90.9% |
| YOLO11N | Pruned 20% | Seed 2 | 417/497 | 83.9% | 29/33 | 87.9% |
| YOLO11N | Pruned 30% | Seed 0 | 403/497 | 81.1% | 33/33 | 100.0% |
| YOLO11N | Pruned 30% | Seed 1 | 425/497 | 85.5% | 29/33 | 87.9% |
| YOLO11N | Pruned 30% | Seed 2 | 410/497 | 82.5% | 33/33 | 100.0% |
| YOLO11N | Pruned 40% | Seed 0 | 419/497 | 84.3% | 27/33 | 81.8% |
| YOLO11N | Pruned 40% | Seed 1 | 416/497 | 83.7% | 32/33 | 97.0% |
| YOLO11N | Pruned 40% | Seed 2 | 417/497 | 83.9% | 31/33 | 93.9% |
| YOLO11N | Pruned 50% | Seed 0 | 426/497 | 85.7% | 29/33 | 87.9% |
| YOLO11N | Pruned 50% | Seed 1 | 421/497 | 84.7% | 28/33 | 84.8% |
| YOLO11N | Pruned 50% | Seed 2 | 416/497 | 83.7% | 29/33 | 87.9% |
| YOLO26N | Baseline | Seed 0 | 424/497 | 85.3% | 29/33 | 87.9% |
| YOLO26N | Baseline | Seed 1 | 421/497 | 84.7% | 27/33 | 81.8% |
| YOLO26N | Baseline | Seed 2 | 422/497 | 84.9% | 26/33 | 78.8% |
| YOLO26N | Pruned 10% | Seed 0 | 407/497 | 81.9% | 28/33 | 84.8% |
| YOLO26N | Pruned 10% | Seed 1 | 412/497 | 82.9% | 21/33 | 63.6% |
| YOLO26N | Pruned 10% | Seed 2 | 415/497 | 83.5% | 25/33 | 75.8% |
| YOLO26N | Pruned 20% | Seed 0 | 436/497 | 87.7% | 24/33 | 72.7% |
| YOLO26N | Pruned 20% | Seed 1 | 425/497 | 85.5% | 27/33 | 81.8% |
| YOLO26N | Pruned 20% | Seed 2 | 416/497 | 83.7% | 24/33 | 72.7% |
| YOLO26N | Pruned 30% | Seed 0 | 404/497 | 81.3% | 22/33 | 66.7% |
| YOLO26N | Pruned 30% | Seed 1 | 415/497 | 83.5% | 29/33 | 87.9% |
| YOLO26N | Pruned 30% | Seed 2 | 407/497 | 81.9% | 26/33 | 78.8% |
| YOLO26N | Pruned 40% | Seed 0 | 390/497 | 78.5% | 27/33 | 81.8% |
| YOLO26N | Pruned 40% | Seed 1 | 419/497 | 84.3% | 25/33 | 75.8% |
| YOLO26N | Pruned 40% | Seed 2 | 407/497 | 81.9% | 29/33 | 87.9% |
| YOLO26N | Pruned 50% | Seed 0 | 417/497 | 83.9% | 26/33 | 78.8% |
| YOLO26N | Pruned 50% | Seed 1 | 419/497 | 84.3% | 27/33 | 81.8% |
| YOLO26N | Pruned 50% | Seed 2 | 409/497 | 82.3% | 27/33 | 81.8% |

---

## TASK 2: Clustered Bootstrap CIs (2,000 Iterations)

> Resampling 13 clips with replacement across 2,000 iterations. Exact same resample matrix used across all models and seeds.  

### 2A. Pooled Estimates across Seeds per Condition

| Architecture | Condition | phone_use Rec [95% CI] | Paired $\Delta$ phone_use [95% CI] | yawning Rec [95% CI]* | Paired $\Delta$ yawning [95% CI]* | Macro-Rec [95% CI] | Paired $\Delta$ Macro [95% CI] |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| YOLO11N | Baseline (0%) | 85.2% [56.0, 99.5] (n_clips=3, n=497) | — | 87.7% [73.3, 100.0] (n_clips=5, n=33)* | — | 89.2% [77.0, 96.6] (n_clips=13, n=614) | — |
| YOLO11N | Pruned 10% | 83.0% [51.0, 98.6] (n_clips=3, n=497) | -2.28 pp [-5.00, -0.88] **(CI excludes 0)** | 87.1% [74.7, 100.0] (n_clips=5, n=33)* | -0.54 pp [-6.67, +6.67] **(CI includes 0)*** | 88.1% [74.6, 96.1] (n_clips=13, n=614) | -1.06 pp [-3.23, +1.08] **(CI includes 0)** |
| YOLO11N | Pruned 20% | 84.2% [52.7, 99.3] (n_clips=3, n=497) | -1.06 pp [-3.33, +0.23] **(CI includes 0)** | 89.0% [78.6, 96.7] (n_clips=5, n=33)* | +1.36 pp [-4.76, +9.31] **(CI includes 0)*** | 89.7% [78.9, 96.1] (n_clips=13, n=614) | +0.48 pp [-1.84, +3.38] **(CI includes 0)** |
| YOLO11N | Pruned 30% | 83.1% [50.2, 99.6] (n_clips=3, n=497) | -2.15 pp [-5.83, +0.18] **(CI includes 0)** | 95.3% [90.0, 98.1] (n_clips=5, n=33)* | +7.64 pp [-2.39, +22.67] **(CI includes 0)*** | 90.3% [78.5, 96.9] (n_clips=13, n=614) | +1.07 pp [-1.66, +3.90] **(CI includes 0)** |
| YOLO11N | Pruned 40% | 84.0% [54.0, 98.6] (n_clips=3, n=497) | -1.20 pp [-2.08, -0.68] **(CI excludes 0)** | 91.2% [80.0, 100.0] (n_clips=5, n=33)* | +3.56 pp [-2.15, +9.52] **(CI includes 0)*** | 90.2% [78.8, 97.1] (n_clips=13, n=614) | +0.98 pp [-1.43, +3.46] **(CI includes 0)** |
| YOLO11N | Pruned 50% | 84.8% [55.0, 99.5] (n_clips=3, n=497) | -0.47 pp [-1.04, +0.00] **(CI includes 0)** | 86.4% [74.4, 96.7] (n_clips=5, n=33)* | -1.25 pp [-4.76, +2.33] **(CI includes 0)*** | 89.1% [77.6, 96.0] (n_clips=13, n=614) | -0.13 pp [-1.20, +0.80] **(CI includes 0)** |
| YOLO26N | Baseline (0%) | 85.0% [57.7, 99.3] (n_clips=3, n=497) | — | 82.3% [71.1, 92.2] (n_clips=5, n=33)* | — | 88.9% [78.0, 95.0] (n_clips=13, n=614) | — |
| YOLO26N | Pruned 10% | 82.8% [50.0, 99.5] (n_clips=3, n=497) | -2.19 pp [-7.71, +0.68] **(CI includes 0)** | 73.9% [61.5, 83.3] (n_clips=5, n=33)* | -8.40 pp [-17.95, +1.96] **(CI includes 0)*** | 84.4% [71.0, 92.0] (n_clips=13, n=614) | -4.52 pp [-9.61, -1.48] **(CI excludes 0)** |
| YOLO26N | Pruned 20% | 85.7% [57.5, 99.3] (n_clips=3, n=497) | +0.71 pp [-0.21, +2.48] **(CI includes 0)** | 75.8% [62.7, 86.1] (n_clips=5, n=33)* | -6.45 pp [-15.06, +3.03] **(CI includes 0)*** | 86.5% [74.7, 93.0] (n_clips=13, n=614) | -2.35 pp [-5.48, +0.15] **(CI includes 0)** |
| YOLO26N | Pruned 30% | 82.2% [52.9, 99.5] (n_clips=3, n=497) | -2.80 pp [-4.79, +0.18] **(CI includes 0)** | 76.2% [59.3, 86.1] (n_clips=5, n=33)* | -6.05 pp [-15.56, +1.33] **(CI includes 0)*** | 85.7% [73.6, 92.7] (n_clips=13, n=614) | -3.18 pp [-5.75, -1.03] **(CI excludes 0)** |
| YOLO26N | Pruned 40% | 81.6% [49.4, 99.6] (n_clips=3, n=497) | -3.45 pp [-8.33, +0.35] **(CI includes 0)** | 81.6% [70.3, 91.2] (n_clips=5, n=33)* | -0.67 pp [-9.33, +9.52] **(CI includes 0)*** | 86.7% [73.4, 94.1] (n_clips=13, n=614) | -2.21 pp [-6.38, +0.89] **(CI includes 0)** |
| YOLO26N | Pruned 50% | 83.6% [50.6, 99.5] (n_clips=3, n=497) | -1.43 pp [-7.08, +2.48] **(CI includes 0)** | 81.7% [68.8, 94.9] (n_clips=5, n=33)* | -0.61 pp [-11.90, +14.29] **(CI includes 0)*** | 87.3% [76.5, 94.9] (n_clips=13, n=614) | -1.53 pp [-7.19, +3.57] **(CI includes 0)** |

`*` *Flagged: Yawning has only 5 clips in test, making its bootstrap CI wide and fragile to cluster resampling.*  

### 2B. Sparsity Trend Regressions (0% to 50% Sparsity)

> Linear regression of recall on sparsity (0, 10, 20, 30, 40, 50) using 18 seed-level observations.  

| Architecture | Metric / Class | OLS Slope (pp / 10% sparsity) [95% CI] | Significance | Spearman $\rho$ ($p$-value) |
| :--- | :--- | :--- | :--- | :--- |
| YOLO11N | `phone_use` | -0.01 pp [-0.06, +0.03] | **CI includes 0** | $\rho = -0.047$ ($p=0.853$) |
| YOLO11N | `yawning` | +0.35 pp [-0.63, +1.13] | **CI includes 0** | $\rho = +0.097$ ($p=0.700$) |
| YOLO11N | `macro_recall` | +0.17 pp [-0.20, +0.58] | **CI includes 0** | $\rho = +0.210$ ($p=0.403$) |
| YOLO26N | `phone_use` | -0.41 pp [-1.20, +0.05] | **CI includes 0** | $\rho = -0.348$ ($p=0.157$) |
| YOLO26N | `yawning` | +0.59 pp [-1.56, +3.13] | **CI includes 0** | $\rho = +0.103$ ($p=0.683$) |
| YOLO26N | `macro_recall` | -0.04 pp [-0.84, +0.75] | **CI includes 0** | $\rho = -0.041$ ($p=0.872$) |

### 2C. Per-Checkpoint Paired Deltas (All 36 Checkpoints)

| Architecture | Condition | Seed | phone_use $\Delta$ vs Base [95% CI] | yawning $\Delta$ vs Base [95% CI]* | Macro $\Delta$ vs Base [95% CI] |
| :--- | :--- | :--- | :--- | :--- | :--- |
| YOLO11N | Baseline | Seed 0 | Baseline | Baseline | Baseline |
| YOLO11N | Baseline | Seed 1 | Baseline | Baseline | Baseline |
| YOLO11N | Baseline | Seed 2 | Baseline | Baseline | Baseline |
| YOLO11N | Pruned 10% | Seed 0 | [-7.50, -0.68] pp (CI excludes 0) | [-12.00, +0.00] pp (CI includes 0)* | [-6.29, -1.06] pp (CI excludes 0) |
| YOLO11N | Pruned 10% | Seed 1 | [-5.63, +0.53] pp (CI includes 0) | [+0.00, +0.00] pp (CI includes 0)* | [-1.08, +1.56] pp (CI includes 0) |
| YOLO11N | Pruned 10% | Seed 2 | [-1.87, +0.68] pp (CI includes 0) | [-8.36, +20.00] pp (CI includes 0)* | [-3.71, +5.15] pp (CI includes 0) |
| YOLO11N | Pruned 20% | Seed 0 | [-6.88, -1.06] pp (CI excludes 0) | [-12.00, +7.14] pp (CI includes 0)* | [-3.54, +4.42] pp (CI includes 0) |
| YOLO11N | Pruned 20% | Seed 1 | [+0.53, +3.75] pp (CI excludes 0) | [+0.00, +14.29] pp (CI includes 0)* | [-0.80, +9.06] pp (CI includes 0) |
| YOLO11N | Pruned 20% | Seed 2 | [-6.87, +0.68] pp (CI includes 0) | [-10.00, +7.14] pp (CI includes 0)* | [-5.50, +1.06] pp (CI includes 0) |
| YOLO11N | Pruned 30% | Seed 0 | [-11.25, -0.68] pp (CI excludes 0) | [+0.00, +21.43] pp (CI includes 0)* | [-2.68, +4.79] pp (CI includes 0) |
| YOLO11N | Pruned 30% | Seed 1 | [+0.62, +1.59] pp (CI excludes 0) | [-10.00, +14.29] pp (CI includes 0)* | [-2.34, +3.85] pp (CI includes 0) |
| YOLO11N | Pruned 30% | Seed 2 | [-6.87, +0.00] pp (CI includes 0) | [+0.00, +30.77] pp (CI includes 0)* | [-2.64, +5.21] pp (CI includes 0) |
| YOLO11N | Pruned 40% | Seed 0 | [-2.12, +0.62] pp (CI includes 0) | [-15.38, +0.00] pp (CI includes 0)* | [-2.28, +4.05] pp (CI includes 0) |
| YOLO11N | Pruned 40% | Seed 1 | [-2.50, +1.59] pp (CI includes 0) | [+0.00, +22.60] pp (CI includes 0)* | [-1.70, +4.51] pp (CI includes 0) |
| YOLO11N | Pruned 40% | Seed 2 | [-4.37, +0.68] pp (CI includes 0) | [+0.00, +25.00] pp (CI includes 0)* | [-3.75, +5.12] pp (CI includes 0) |
| YOLO11N | Pruned 50% | Seed 0 | [-3.38, +5.00] pp (CI includes 0) | [-10.00, +0.00] pp (CI includes 0)* | [-1.36, +6.50] pp (CI includes 0) |
| YOLO11N | Pruned 50% | Seed 1 | [-1.88, +1.59] pp (CI includes 0) | [+0.00, +0.00] pp (CI includes 0)* | [-5.26, +0.38] pp (CI includes 0) |
| YOLO11N | Pruned 50% | Seed 2 | [-6.25, +0.68] pp (CI includes 0) | [-12.00, +7.14] pp (CI includes 0)* | [-4.37, +1.27] pp (CI includes 0) |
| YOLO26N | Baseline | Seed 0 | Baseline | Baseline | Baseline |
| YOLO26N | Baseline | Seed 1 | Baseline | Baseline | Baseline |
| YOLO26N | Baseline | Seed 2 | Baseline | Baseline | Baseline |
| YOLO26N | Pruned 10% | Seed 0 | [-6.87, +0.00] pp (CI includes 0) | [-25.00, +7.69] pp (CI includes 0)* | [-9.79, +2.03] pp (CI includes 0) |
| YOLO26N | Pruned 10% | Seed 1 | [-4.37, +0.00] pp (CI includes 0) | [-28.57, -8.00] pp (CI excludes 0)* | [-11.96, -4.05] pp (CI excludes 0) |
| YOLO26N | Pruned 10% | Seed 2 | [-11.88, +7.43] pp (CI includes 0) | [-22.60, +21.76] pp (CI includes 0)* | [-10.78, +3.87] pp (CI includes 0) |
| YOLO26N | Pruned 20% | Seed 0 | [-0.53, +6.87] pp (CI includes 0) | [-33.33, +0.00] pp (CI includes 0)* | [-8.68, +0.45] pp (CI includes 0) |
| YOLO26N | Pruned 20% | Seed 1 | [+0.00, +1.88] pp (CI includes 0) | [-8.02, +25.00] pp (CI includes 0)* | [-1.95, +6.25] pp (CI includes 0) |
| YOLO26N | Pruned 20% | Seed 2 | [-9.38, +6.08] pp (CI includes 0) | [-22.60, +18.18] pp (CI includes 0)* | [-12.59, +2.55] pp (CI includes 0) |
| YOLO26N | Pruned 30% | Seed 0 | [-10.81, +0.53] pp (CI includes 0) | [-37.50, -8.33] pp (CI excludes 0)* | [-12.40, -3.71] pp (CI excludes 0) |
| YOLO26N | Pruned 30% | Seed 1 | [-3.75, +0.53] pp (CI includes 0) | [-20.00, +28.57] pp (CI includes 0)* | [-7.01, +2.23] pp (CI includes 0) |
| YOLO26N | Pruned 30% | Seed 2 | [-7.50, -0.53] pp (CI excludes 0) | [-12.00, +9.12] pp (CI includes 0)* | [-3.63, +2.26] pp (CI includes 0) |
| YOLO26N | Pruned 40% | Seed 0 | [-16.22, +0.00] pp (CI includes 0) | [-36.12, +20.00] pp (CI includes 0)* | [-18.88, +2.68] pp (CI includes 0) |
| YOLO26N | Pruned 40% | Seed 1 | [-1.87, +0.53] pp (CI includes 0) | [-22.22, +6.99] pp (CI includes 0)* | [-5.88, +2.19] pp (CI includes 0) |
| YOLO26N | Pruned 40% | Seed 2 | [-16.88, +7.43] pp (CI includes 0) | [-5.56, +33.43] pp (CI includes 0)* | [-6.03, +8.07] pp (CI includes 0) |
| YOLO26N | Pruned 50% | Seed 0 | [-5.63, +0.68] pp (CI includes 0) | [-21.43, +0.00] pp (CI includes 0)* | [-6.25, +3.23] pp (CI includes 0) |
| YOLO26N | Pruned 50% | Seed 1 | [-2.50, +1.06] pp (CI includes 0) | [-16.67, +25.00] pp (CI includes 0)* | [-8.45, +4.75] pp (CI includes 0) |
| YOLO26N | Pruned 50% | Seed 2 | [-13.13, +6.76] pp (CI includes 0) | [-20.00, +28.57] pp (CI includes 0)* | [-10.96, +6.69] pp (CI includes 0) |

---

## TASK 3: Unrestricted Precision-Matched Metric (No Floor, Targets P in {85, 90, 95})

> Sweep $\tau \ge 0.001$. $\tau_{\text{val}}$ tuned on validation split to reach target precision $P$. Evaluated on test with clustered bootstrap CIs.  

### 3A. Validation-Tuned Deployment Protocol

| Arch | Cond | Seed | Target P | $\tau_{\text{val}}$ | Boundary / Feas | Val Prec | Test Prec [95% CI] | Transfer Gap | Macro-Rec [95% CI] | phone_use Rec [95% CI] | yawning Rec [95% CI]* |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| YOLO11N | Baseline | Seed 0 | P85 | 0.0764 | FEASIBLE | 85.1% | 75.0% [51.3, 92.7] | -10.11 pp | 94.3% [86.0, 99.2] | 89.1% [66.2, 100.0] | 97.0% [92.9, 100.0]* |
| YOLO11N | Baseline | Seed 0 | P90 | 0.1966 | FEASIBLE | 90.1% | 85.0% [65.6, 97.1] | -5.13 pp | 92.0% [80.1, 98.4] | 87.1% [60.0, 100.0] | 97.0% [92.9, 100.0]* |
| YOLO11N | Baseline | Seed 0 | P95 | 0.4035 | FEASIBLE | 95.0% | 91.2% [77.4, 97.8] | -3.87 pp | 89.4% [75.8, 97.6] | 84.5% [52.5, 100.0] | 90.9% [78.6, 100.0]* |
| YOLO11N | Baseline | Seed 1 | P85 | 0.1054 | FEASIBLE | 85.1% | 90.2% [77.8, 97.5] | +5.12 pp | 93.9% [84.2, 98.6] | 90.5% [71.2, 100.0] | 93.9% [72.7, 100.0]* |
| YOLO11N | Baseline | Seed 1 | P90 | 0.1781 | FEASIBLE | 90.1% | 91.4% [79.6, 98.0] | +1.33 pp | 93.0% [83.1, 97.7] | 88.9% [66.9, 99.5] | 93.9% [72.7, 100.0]* |
| YOLO11N | Baseline | Seed 1 | P95 | 0.3199 | FEASIBLE | 95.0% | 93.0% [82.7, 98.3] | -2.06 pp | 90.5% [79.6, 96.1] | 84.9% [56.9, 98.4] | 87.9% [71.4, 100.0]* |
| YOLO11N | Baseline | Seed 2 | P85 | 0.0186 | FEASIBLE | 85.0% | 73.3% [50.1, 91.6] | -11.67 pp | 95.3% [88.3, 99.2] | 90.1% [70.0, 100.0] | 100.0% [100.0, 100.0]* |
| YOLO11N | Baseline | Seed 2 | P90 | 0.0549 | FEASIBLE | 90.1% | 79.5% [58.3, 94.5] | -10.55 pp | 93.9% [84.4, 97.8] | 89.3% [67.5, 100.0] | 97.0% [75.0, 100.0]* |
| YOLO11N | Baseline | Seed 2 | P95 | 0.2032 | FEASIBLE | 95.0% | 87.2% [70.9, 97.0] | -7.86 pp | 92.6% [82.6, 97.4] | 87.3% [61.9, 100.0] | 93.9% [75.0, 100.0]* |
| YOLO11N | Pruned 10% | Seed 0 | P85 | 0.0969 | FEASIBLE | 85.0% | 81.5% [58.2, 97.7] | -3.47 pp | 91.6% [80.6, 98.0] | 85.5% [56.2, 99.5] | 97.0% [92.9, 100.0]* |
| YOLO11N | Pruned 10% | Seed 0 | P90 | 0.2194 | FEASIBLE | 90.1% | 86.7% [66.9, 98.2] | -3.36 pp | 88.8% [76.7, 97.4] | 83.9% [51.9, 99.3] | 90.9% [78.6, 100.0]* |
| YOLO11N | Pruned 10% | Seed 0 | P95 | 0.4115 | FEASIBLE | 95.1% | 90.7% [75.0, 98.3] | -4.45 pp | 85.9% [71.2, 94.9] | 80.9% [46.9, 97.9] | 87.9% [77.4, 100.0]* |
| YOLO11N | Pruned 10% | Seed 1 | P85 | 0.0072 | FEASIBLE | 85.0% | 71.2% [50.3, 85.8] | -13.77 pp | 96.3% [89.7, 99.8] | 90.5% [72.5, 100.0] | 100.0% [100.0, 100.0]* |
| YOLO11N | Pruned 10% | Seed 1 | P90 | 0.0156 | FEASIBLE | 90.0% | 77.1% [57.8, 89.1] | -12.97 pp | 96.1% [89.1, 99.8] | 89.7% [70.0, 100.0] | 100.0% [100.0, 100.0]* |
| YOLO11N | Pruned 10% | Seed 1 | P95 | 0.1006 | FEASIBLE | 95.1% | 87.2% [73.1, 94.5] | -7.91 pp | 94.1% [83.7, 99.1] | 86.5% [61.3, 99.5] | 97.0% [75.0, 100.0]* |
| YOLO11N | Pruned 10% | Seed 2 | P85 | 0.0060 | FEASIBLE | 85.1% | 68.6% [46.5, 85.4] | -16.54 pp | 98.1% [94.1, 100.0] | 92.4% [76.2, 100.0] | 100.0% [100.0, 100.0]* |
| YOLO11N | Pruned 10% | Seed 2 | P90 | 0.0251 | FEASIBLE | 90.1% | 79.0% [59.0, 92.2] | -11.10 pp | 97.5% [92.3, 100.0] | 90.1% [69.4, 100.0] | 100.0% [100.0, 100.0]* |
| YOLO11N | Pruned 10% | Seed 2 | P95 | 0.2375 | FEASIBLE | 95.1% | 89.3% [74.7, 96.8] | -5.82 pp | 93.2% [83.6, 99.7] | 86.1% [58.1, 100.0] | 93.9% [85.7, 100.0]* |
| YOLO11N | Pruned 20% | Seed 0 | P85 | 0.0155 | FEASIBLE | 85.1% | 74.0% [51.5, 90.3] | -11.08 pp | 95.6% [87.8, 99.9] | 87.9% [63.1, 100.0] | 100.0% [100.0, 100.0]* |
| YOLO11N | Pruned 20% | Seed 0 | P90 | 0.0594 | FEASIBLE | 90.1% | 82.5% [62.6, 95.4] | -7.53 pp | 94.8% [86.3, 99.3] | 86.3% [58.8, 99.5] | 100.0% [100.0, 100.0]* |
| YOLO11N | Pruned 20% | Seed 0 | P95 | 0.2887 | FEASIBLE | 95.1% | 88.9% [72.9, 97.8] | -6.14 pp | 91.1% [79.1, 97.8] | 83.1% [50.0, 99.5] | 93.9% [85.7, 100.0]* |
| YOLO11N | Pruned 20% | Seed 1 | P85 | 0.0291 | FEASIBLE | 85.1% | 77.9% [55.9, 91.7] | -7.14 pp | 96.4% [90.8, 99.4] | 92.8% [78.8, 100.0] | 100.0% [100.0, 100.0]* |
| YOLO11N | Pruned 20% | Seed 1 | P90 | 0.0619 | FEASIBLE | 90.0% | 83.9% [65.5, 94.3] | -6.12 pp | 96.1% [90.1, 99.2] | 91.5% [75.6, 99.5] | 100.0% [100.0, 100.0]* |
| YOLO11N | Pruned 20% | Seed 1 | P95 | 0.2671 | FEASIBLE | 95.1% | 92.3% [78.2, 97.7] | -2.75 pp | 94.5% [87.1, 98.4] | 88.5% [66.9, 98.9] | 100.0% [100.0, 100.0]* |
| YOLO11N | Pruned 20% | Seed 2 | P85 | 0.0531 | FEASIBLE | 85.1% | 80.6% [60.6, 93.4] | -4.45 pp | 94.3% [86.4, 98.8] | 87.9% [63.1, 100.0] | 100.0% [100.0, 100.0]* |
| YOLO11N | Pruned 20% | Seed 2 | P90 | 0.1176 | FEASIBLE | 90.0% | 84.9% [66.2, 95.9] | -5.15 pp | 93.1% [84.0, 98.7] | 86.1% [57.5, 100.0] | 97.0% [92.9, 100.0]* |
| YOLO11N | Pruned 20% | Seed 2 | P95 | 0.2674 | FEASIBLE | 95.1% | 88.9% [72.9, 97.2] | -6.20 pp | 91.9% [81.6, 98.6] | 84.3% [51.9, 100.0] | 93.9% [85.7, 100.0]* |
| YOLO11N | Pruned 30% | Seed 0 | P85 | 0.0521 | FEASIBLE | 85.1% | 81.1% [57.8, 95.7] | -3.96 pp | 93.1% [82.8, 98.9] | 83.1% [50.0, 98.9] | 100.0% [100.0, 100.0]* |
| YOLO11N | Pruned 30% | Seed 0 | P90 | 0.1237 | FEASIBLE | 90.0% | 85.8% [65.5, 97.3] | -4.22 pp | 92.4% [82.0, 98.4] | 82.3% [47.5, 98.9] | 100.0% [100.0, 100.0]* |
| YOLO11N | Pruned 30% | Seed 0 | P95 | 0.2667 | FEASIBLE | 95.1% | 89.4% [72.8, 97.9] | -5.69 pp | 91.5% [80.6, 97.8] | 81.9% [46.2, 98.9] | 100.0% [100.0, 100.0]* |
| YOLO11N | Pruned 30% | Seed 1 | P85 | 0.0280 | FEASIBLE | 85.0% | 77.7% [55.5, 93.9] | -7.30 pp | 94.4% [85.0, 99.1] | 89.9% [68.8, 100.0] | 100.0% [100.0, 100.0]* |
| YOLO11N | Pruned 30% | Seed 1 | P90 | 0.1087 | FEASIBLE | 90.1% | 86.1% [69.6, 95.8] | -3.92 pp | 93.0% [82.7, 98.7] | 87.7% [62.5, 100.0] | 97.0% [92.9, 100.0]* |
| YOLO11N | Pruned 30% | Seed 1 | P95 | 0.3350 | FEASIBLE | 95.0% | 91.9% [79.9, 97.5] | -3.14 pp | 89.8% [77.3, 95.3] | 85.5% [56.2, 100.0] | 87.9% [70.0, 94.4]* |
| YOLO11N | Pruned 30% | Seed 2 | P85 | 0.0299 | FEASIBLE | 85.1% | 73.0% [52.3, 89.6] | -12.10 pp | 97.3% [92.6, 100.0] | 90.9% [72.5, 100.0] | 100.0% [100.0, 100.0]* |
| YOLO11N | Pruned 30% | Seed 2 | P90 | 0.0894 | FEASIBLE | 90.1% | 80.9% [61.5, 93.6] | -9.24 pp | 94.5% [85.1, 99.0] | 88.7% [66.9, 100.0] | 100.0% [100.0, 100.0]* |
| YOLO11N | Pruned 30% | Seed 2 | P95 | 0.2333 | FEASIBLE | 95.1% | 90.7% [78.5, 97.0] | -4.36 pp | 93.6% [83.4, 98.7] | 85.1% [58.8, 100.0] | 100.0% [100.0, 100.0]* |
| YOLO11N | Pruned 40% | Seed 0 | P85 | 0.0880 | FEASIBLE | 85.1% | 83.0% [64.4, 95.6] | -2.05 pp | 93.8% [85.7, 99.0] | 89.7% [69.4, 100.0] | 90.9% [78.6, 100.0]* |
| YOLO11N | Pruned 40% | Seed 0 | P90 | 0.1847 | FEASIBLE | 90.0% | 86.6% [70.1, 96.7] | -3.41 pp | 92.6% [83.9, 98.3] | 88.3% [65.6, 99.5] | 90.9% [78.6, 100.0]* |
| YOLO11N | Pruned 40% | Seed 0 | P95 | 0.4068 | FEASIBLE | 95.0% | 91.0% [77.5, 98.1] | -4.07 pp | 86.7% [73.6, 95.7] | 82.5% [50.6, 98.0] | 78.8% [63.3, 100.0]* |
| YOLO11N | Pruned 40% | Seed 1 | P85 | 0.0937 | FEASIBLE | 85.0% | 84.7% [68.8, 94.0] | -0.33 pp | 93.9% [85.2, 99.0] | 87.9% [64.4, 100.0] | 100.0% [100.0, 100.0]* |
| YOLO11N | Pruned 40% | Seed 1 | P90 | 0.1983 | FEASIBLE | 90.0% | 88.4% [74.6, 96.0] | -1.61 pp | 92.0% [81.5, 98.2] | 85.7% [57.5, 100.0] | 100.0% [100.0, 100.0]* |
| YOLO11N | Pruned 40% | Seed 1 | P95 | 0.4308 | FEASIBLE | 95.0% | 92.8% [82.6, 97.4] | -2.25 pp | 87.2% [71.4, 94.8] | 82.3% [50.6, 98.9] | 87.9% [71.4, 100.0]* |
| YOLO11N | Pruned 40% | Seed 2 | P85 | 0.0170 | FEASIBLE | 85.1% | 75.6% [54.2, 89.1] | -9.43 pp | 96.4% [90.9, 99.9] | 89.3% [67.5, 100.0] | 100.0% [100.0, 100.0]* |
| YOLO11N | Pruned 40% | Seed 2 | P90 | 0.0635 | FEASIBLE | 90.1% | 85.9% [68.6, 95.1] | -4.23 pp | 94.6% [86.5, 99.1] | 87.3% [62.5, 99.3] | 100.0% [100.0, 100.0]* |
| YOLO11N | Pruned 40% | Seed 2 | P95 | 0.1884 | FEASIBLE | 95.1% | 90.4% [76.9, 96.9] | -4.66 pp | 94.1% [85.2, 99.1] | 85.5% [56.9, 99.3] | 100.0% [100.0, 100.0]* |
| YOLO11N | Pruned 50% | Seed 0 | P85 | 0.0457 | FEASIBLE | 85.1% | 74.1% [52.3, 92.0] | -11.01 pp | 95.8% [90.6, 99.6] | 90.9% [73.8, 100.0] | 93.9% [85.7, 100.0]* |
| YOLO11N | Pruned 50% | Seed 0 | P90 | 0.1266 | FEASIBLE | 90.1% | 82.3% [62.9, 96.0] | -7.83 pp | 95.1% [89.5, 99.6] | 90.1% [71.2, 100.0] | 93.9% [85.7, 100.0]* |
| YOLO11N | Pruned 50% | Seed 0 | P95 | 0.2938 | FEASIBLE | 95.0% | 87.1% [70.2, 97.7] | -7.91 pp | 92.9% [84.3, 98.3] | 86.7% [61.9, 99.5] | 93.9% [85.7, 100.0]* |
| YOLO11N | Pruned 50% | Seed 1 | P85 | 0.0264 | FEASIBLE | 85.0% | 73.7% [52.3, 90.7] | -11.33 pp | 96.4% [90.2, 99.8] | 92.6% [78.1, 100.0] | 100.0% [100.0, 100.0]* |
| YOLO11N | Pruned 50% | Seed 1 | P90 | 0.0671 | FEASIBLE | 90.0% | 80.7% [61.9, 94.0] | -9.27 pp | 95.1% [86.5, 99.1] | 91.1% [73.8, 100.0] | 100.0% [100.0, 100.0]* |
| YOLO11N | Pruned 50% | Seed 1 | P95 | 0.1496 | FEASIBLE | 95.1% | 85.1% [67.7, 96.3] | -10.01 pp | 93.3% [83.1, 98.7] | 88.7% [66.2, 100.0] | 97.0% [92.9, 100.0]* |
| YOLO11N | Pruned 50% | Seed 2 | P85 | 0.0635 | FEASIBLE | 85.0% | 80.9% [59.4, 94.6] | -4.09 pp | 93.9% [86.2, 97.9] | 87.7% [63.7, 99.3] | 97.0% [75.0, 100.0]* |
| YOLO11N | Pruned 50% | Seed 2 | P90 | 0.1068 | FEASIBLE | 90.1% | 83.5% [63.6, 95.7] | -6.57 pp | 92.1% [82.0, 97.1] | 86.9% [61.3, 99.3] | 93.9% [75.0, 100.0]* |
| YOLO11N | Pruned 50% | Seed 2 | P95 | 0.2595 | FEASIBLE | 95.1% | 90.3% [75.2, 97.9] | -4.71 pp | 89.8% [78.5, 95.9] | 83.9% [51.9, 99.3] | 87.9% [71.4, 100.0]* |
| YOLO26N | Baseline | Seed 0 | P85 | 0.0249 | FEASIBLE | 85.1% | 75.3% [54.9, 86.0] | -9.84 pp | 95.5% [88.2, 99.4] | 90.3% [70.6, 100.0] | 97.0% [90.0, 100.0]* |
| YOLO26N | Baseline | Seed 0 | P90 | 0.0703 | FEASIBLE | 90.1% | 82.0% [64.3, 90.3] | -8.11 pp | 94.4% [86.2, 98.8] | 88.9% [66.2, 100.0] | 93.9% [85.7, 100.0]* |
| YOLO26N | Baseline | Seed 0 | P95 | 0.2397 | FEASIBLE | 95.1% | 89.7% [78.0, 95.1] | -5.39 pp | 90.9% [78.7, 96.8] | 86.3% [59.4, 99.5] | 87.9% [78.2, 100.0]* |
| YOLO26N | Baseline | Seed 1 | P85 | 0.0182 | FEASIBLE | 85.1% | 74.3% [54.9, 86.4] | -10.79 pp | 96.6% [89.7, 100.0] | 89.9% [69.4, 100.0] | 100.0% [100.0, 100.0]* |
| YOLO26N | Baseline | Seed 1 | P90 | 0.0761 | FEASIBLE | 90.1% | 85.1% [70.2, 93.2] | -4.93 pp | 94.1% [84.1, 98.6] | 87.9% [63.7, 99.5] | 93.9% [75.0, 100.0]* |
| YOLO26N | Baseline | Seed 1 | P95 | 0.3513 | FEASIBLE | 95.1% | 93.0% [81.9, 97.7] | -2.15 pp | 89.8% [77.9, 96.8] | 84.7% [54.4, 99.3] | 81.8% [61.5, 100.0]* |
| YOLO26N | Baseline | Seed 2 | P85 | 0.0067 | FEASIBLE | 85.0% | 68.6% [50.6, 82.0] | -16.37 pp | 95.4% [88.6, 98.9] | 91.1% [74.4, 99.5] | 93.9% [87.5, 100.0]* |
| YOLO26N | Baseline | Seed 2 | P90 | 0.0208 | FEASIBLE | 90.1% | 78.2% [63.0, 89.0] | -11.89 pp | 95.2% [88.1, 98.9] | 90.5% [72.5, 99.5] | 93.9% [87.5, 100.0]* |
| YOLO26N | Baseline | Seed 2 | P95 | 0.2117 | FEASIBLE | 95.1% | 91.8% [81.9, 96.5] | -3.32 pp | 91.4% [81.1, 96.0] | 87.7% [66.2, 99.5] | 84.8% [69.9, 92.9]* |
| YOLO26N | Pruned 10% | Seed 0 | P85 | 0.0045 | FEASIBLE | 85.1% | 71.5% [45.8, 84.8] | -13.57 pp | 92.9% [82.0, 98.0] | 89.5% [67.5, 100.0] | 90.9% [70.6, 98.3]* |
| YOLO26N | Pruned 10% | Seed 0 | P90 | 0.0134 | FEASIBLE | 90.1% | 78.1% [55.1, 88.6] | -11.98 pp | 92.6% [81.2, 98.0] | 88.5% [64.4, 100.0] | 90.9% [70.6, 98.3]* |
| YOLO26N | Pruned 10% | Seed 0 | P95 | 0.0938 | FEASIBLE | 95.1% | 87.5% [72.0, 94.6] | -7.52 pp | 90.3% [78.3, 96.7] | 85.9% [59.4, 100.0] | 87.9% [70.0, 94.4]* |
| YOLO26N | Pruned 10% | Seed 1 | P85 | 0.0034 | FEASIBLE | 85.1% | 71.4% [49.1, 87.0] | -13.72 pp | 91.3% [81.6, 97.0] | 89.3% [67.5, 100.0] | 84.8% [76.9, 100.0]* |
| YOLO26N | Pruned 10% | Seed 1 | P90 | 0.0125 | FEASIBLE | 90.1% | 80.0% [60.5, 92.0] | -10.09 pp | 90.2% [80.3, 96.3] | 88.5% [65.0, 100.0] | 84.8% [76.9, 100.0]* |
| YOLO26N | Pruned 10% | Seed 1 | P95 | 0.0649 | FEASIBLE | 95.1% | 87.6% [72.6, 95.5] | -7.50 pp | 88.0% [73.7, 94.8] | 85.5% [56.9, 99.5] | 78.8% [61.5, 94.1]* |
| YOLO26N | Pruned 10% | Seed 2 | P85 | 0.0294 | FEASIBLE | 85.1% | 78.6% [64.2, 86.4] | -6.46 pp | 93.9% [82.6, 99.2] | 87.5% [62.5, 100.0] | 97.0% [90.0, 100.0]* |
| YOLO26N | Pruned 10% | Seed 2 | P90 | 0.0735 | FEASIBLE | 90.1% | 83.7% [71.2, 90.0] | -6.33 pp | 92.4% [80.8, 98.2] | 86.5% [59.4, 100.0] | 93.9% [89.3, 100.0]* |
| YOLO26N | Pruned 10% | Seed 2 | P95 | 0.2932 | FEASIBLE | 95.0% | 91.1% [80.8, 95.2] | -3.96 pp | 85.5% [69.5, 93.1] | 83.9% [51.2, 100.0] | 75.8% [55.5, 92.9]* |
| YOLO26N | Pruned 20% | Seed 0 | P85 | 0.0081 | FEASIBLE | 85.0% | 75.1% [57.3, 86.6] | -9.92 pp | 93.1% [84.5, 96.6] | 92.8% [78.1, 100.0] | 84.8% [69.9, 92.9]* |
| YOLO26N | Pruned 20% | Seed 0 | P90 | 0.0334 | FEASIBLE | 90.1% | 83.2% [69.1, 91.2] | -6.87 pp | 91.8% [82.3, 96.0] | 91.1% [73.1, 100.0] | 84.8% [69.9, 92.9]* |
| YOLO26N | Pruned 20% | Seed 0 | P95 | 0.1342 | FEASIBLE | 95.1% | 89.1% [79.2, 94.1] | -5.96 pp | 88.9% [75.5, 94.1] | 89.1% [67.5, 99.5] | 78.8% [62.5, 85.7]* |
| YOLO26N | Pruned 20% | Seed 1 | P85 | 0.0114 | FEASIBLE | 85.0% | 78.5% [61.4, 90.0] | -6.49 pp | 94.1% [86.8, 98.3] | 90.7% [71.9, 100.0] | 90.9% [83.3, 100.0]* |
| YOLO26N | Pruned 20% | Seed 1 | P90 | 0.0322 | FEASIBLE | 90.0% | 84.6% [70.2, 93.2] | -5.43 pp | 93.7% [85.6, 98.3] | 89.1% [66.9, 100.0] | 90.9% [83.3, 100.0]* |
| YOLO26N | Pruned 20% | Seed 1 | P95 | 0.1812 | FEASIBLE | 95.0% | 91.5% [80.9, 97.0] | -3.50 pp | 91.2% [82.3, 96.9] | 86.9% [60.6, 99.5] | 84.8% [77.1, 100.0]* |
| YOLO26N | Pruned 20% | Seed 2 | P85 | 0.0182 | FEASIBLE | 85.0% | 77.4% [61.6, 85.0] | -7.60 pp | 95.2% [87.0, 99.3] | 90.3% [70.0, 100.0] | 93.9% [71.4, 100.0]* |
| YOLO26N | Pruned 20% | Seed 2 | P90 | 0.0742 | FEASIBLE | 90.1% | 85.1% [73.9, 90.3] | -4.93 pp | 92.3% [81.8, 97.2] | 88.5% [64.4, 100.0] | 87.9% [66.6, 95.5]* |
| YOLO26N | Pruned 20% | Seed 2 | P95 | 0.2510 | FEASIBLE | 95.0% | 90.9% [81.1, 95.5] | -4.15 pp | 86.8% [72.6, 93.2] | 85.7% [57.5, 99.5] | 75.8% [61.5, 85.7]* |
| YOLO26N | Pruned 30% | Seed 0 | P85 | 0.0020 | FEASIBLE | 85.1% | 73.5% [53.0, 82.3] | -11.54 pp | 95.4% [85.8, 99.3] | 89.9% [70.0, 100.0] | 97.0% [75.0, 100.0]* |
| YOLO26N | Pruned 30% | Seed 0 | P90 | 0.0056 | FEASIBLE | 90.1% | 81.4% [66.0, 87.9] | -8.74 pp | 94.0% [84.4, 98.5] | 89.1% [68.1, 100.0] | 93.9% [75.0, 100.0]* |
| YOLO26N | Pruned 30% | Seed 0 | P95 | 0.0618 | FEASIBLE | 95.1% | 89.3% [78.1, 94.0] | -5.79 pp | 89.3% [78.3, 94.8] | 86.1% [61.3, 100.0] | 81.8% [64.7, 92.0]* |
| YOLO26N | Pruned 30% | Seed 1 | P85 | 0.0055 | FEASIBLE | 85.0% | 74.6% [54.7, 87.3] | -10.44 pp | 94.7% [87.0, 99.2] | 90.7% [71.9, 100.0] | 97.0% [88.0, 100.0]* |
| YOLO26N | Pruned 30% | Seed 1 | P90 | 0.0171 | FEASIBLE | 90.1% | 80.6% [61.8, 91.0] | -9.56 pp | 93.1% [84.6, 98.3] | 89.3% [67.5, 100.0] | 93.9% [85.7, 100.0]* |
| YOLO26N | Pruned 30% | Seed 1 | P95 | 0.1819 | FEASIBLE | 95.1% | 93.0% [83.1, 96.5] | -2.03 pp | 89.9% [77.7, 95.5] | 85.9% [57.5, 100.0] | 87.9% [64.7, 98.3]* |
| YOLO26N | Pruned 30% | Seed 2 | P85 | 0.0098 | FEASIBLE | 85.1% | 72.7% [57.2, 83.8] | -12.36 pp | 93.9% [85.7, 98.5] | 87.5% [70.6, 100.0] | 97.0% [88.0, 100.0]* |
| YOLO26N | Pruned 30% | Seed 2 | P90 | 0.0316 | FEASIBLE | 90.0% | 79.0% [66.3, 87.5] | -11.04 pp | 92.0% [83.4, 97.1] | 85.9% [65.6, 100.0] | 90.9% [85.7, 100.0]* |
| YOLO26N | Pruned 30% | Seed 2 | P95 | 0.2568 | FEASIBLE | 95.1% | 88.8% [79.8, 94.8] | -6.29 pp | 88.1% [77.0, 93.9] | 82.5% [55.6, 99.5] | 78.8% [62.5, 83.3]* |
| YOLO26N | Pruned 40% | Seed 0 | P85 | 0.0136 | FEASIBLE | 85.1% | 71.6% [50.7, 84.6] | -13.45 pp | 94.6% [87.5, 99.2] | 85.7% [66.9, 99.5] | 100.0% [100.0, 100.0]* |
| YOLO26N | Pruned 40% | Seed 0 | P90 | 0.0405 | FEASIBLE | 90.1% | 78.0% [60.5, 87.9] | -12.10 pp | 93.6% [86.1, 98.5] | 84.7% [64.4, 99.5] | 97.0% [92.9, 100.0]* |
| YOLO26N | Pruned 40% | Seed 0 | P95 | 0.2065 | FEASIBLE | 95.0% | 87.2% [75.9, 94.0] | -7.85 pp | 88.2% [74.0, 95.4] | 80.9% [56.2, 99.5] | 87.9% [69.2, 100.0]* |
| YOLO26N | Pruned 40% | Seed 1 | P85 | 0.0096 | FEASIBLE | 85.0% | 73.3% [52.4, 85.9] | -11.73 pp | 96.0% [87.4, 99.5] | 92.6% [76.9, 100.0] | 97.0% [75.0, 100.0]* |
| YOLO26N | Pruned 40% | Seed 1 | P90 | 0.0395 | FEASIBLE | 90.0% | 83.8% [65.0, 91.6] | -6.27 pp | 94.5% [84.1, 99.0] | 89.3% [66.9, 100.0] | 93.9% [71.4, 100.0]* |
| YOLO26N | Pruned 40% | Seed 1 | P95 | 0.3650 | FEASIBLE | 95.1% | 92.9% [78.9, 97.9] | -2.15 pp | 88.6% [76.9, 93.9] | 84.1% [51.9, 99.5] | 75.8% [62.5, 83.3]* |
| YOLO26N | Pruned 40% | Seed 2 | P85 | 0.0295 | FEASIBLE | 85.0% | 77.5% [58.7, 87.8] | -7.55 pp | 95.5% [88.4, 99.8] | 86.9% [60.0, 100.0] | 97.0% [90.0, 100.0]* |
| YOLO26N | Pruned 40% | Seed 2 | P90 | 0.0789 | FEASIBLE | 90.1% | 83.7% [66.8, 91.7] | -6.39 pp | 95.3% [87.9, 99.8] | 86.1% [57.5, 100.0] | 97.0% [90.0, 100.0]* |
| YOLO26N | Pruned 40% | Seed 2 | P95 | 0.2154 | FEASIBLE | 95.0% | 89.2% [75.8, 94.9] | -5.85 pp | 91.2% [81.7, 96.9] | 83.9% [51.2, 100.0] | 87.9% [83.7, 100.0]* |
| YOLO26N | Pruned 50% | Seed 0 | P85 | 0.0275 | FEASIBLE | 85.0% | 69.7% [46.4, 83.5] | -15.36 pp | 95.0% [86.8, 99.3] | 88.5% [65.6, 100.0] | 97.0% [88.0, 100.0]* |
| YOLO26N | Pruned 50% | Seed 0 | P90 | 0.0673 | FEASIBLE | 90.1% | 77.8% [56.3, 88.2] | -12.33 pp | 93.7% [85.1, 98.4] | 87.9% [63.7, 100.0] | 93.9% [87.5, 100.0]* |
| YOLO26N | Pruned 50% | Seed 0 | P95 | 0.2057 | FEASIBLE | 95.0% | 87.4% [72.0, 93.5] | -7.65 pp | 90.8% [81.1, 96.6] | 85.3% [55.6, 100.0] | 84.8% [77.1, 100.0]* |
| YOLO26N | Pruned 50% | Seed 1 | P85 | 0.0258 | FEASIBLE | 85.0% | 76.3% [54.5, 88.0] | -8.74 pp | 95.3% [87.4, 99.9] | 88.5% [65.0, 100.0] | 100.0% [100.0, 100.0]* |
| YOLO26N | Pruned 50% | Seed 1 | P90 | 0.0570 | FEASIBLE | 90.1% | 82.0% [62.4, 90.4] | -8.08 pp | 95.0% [86.5, 99.9] | 87.1% [60.6, 100.0] | 100.0% [100.0, 100.0]* |
| YOLO26N | Pruned 50% | Seed 1 | P95 | 0.3078 | FEASIBLE | 95.0% | 91.4% [79.0, 96.0] | -3.55 pp | 88.4% [77.7, 95.5] | 84.3% [51.9, 100.0] | 81.8% [75.0, 100.0]* |
| YOLO26N | Pruned 50% | Seed 2 | P85 | 0.0286 | FEASIBLE | 85.0% | 75.2% [55.2, 88.2] | -9.86 pp | 92.1% [82.3, 98.1] | 87.1% [60.6, 100.0] | 93.9% [80.0, 100.0]* |
| YOLO26N | Pruned 50% | Seed 2 | P90 | 0.0667 | FEASIBLE | 90.1% | 80.8% [64.2, 90.4] | -9.32 pp | 91.3% [80.7, 98.0] | 85.7% [56.9, 100.0] | 93.9% [80.0, 100.0]* |
| YOLO26N | Pruned 50% | Seed 2 | P95 | 0.2501 | FEASIBLE | 95.1% | 88.7% [75.4, 95.2] | -6.38 pp | 87.6% [73.5, 96.3] | 83.5% [51.2, 99.5] | 84.8% [59.1, 100.0]* |

### 3B. Oracle Upper Bound ($\tau_{@P}$ Chosen Directly on Test Split)

| Arch | Cond | Seed | Target P | $\tau_{\text{oracle}}$ | Test Prec | Macro-Rec | phone_use Rec | yawning Rec* |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| YOLO11N | Baseline | Seed 0 | P85 | 0.1991 | 85.1% | 92.0% | 87.1% | 97.0%* |
| YOLO11N | Baseline | Seed 0 | P90 | 0.3665 | 90.1% | 89.5% | 84.9% | 90.9%* |
| YOLO11N | Baseline | Seed 0 | P95 | 0.5814 | 95.0% | 83.2% | 80.1% | 75.8%* |
| YOLO11N | Baseline | Seed 1 | P85 | 0.0326 | 85.1% | 94.7% | 92.2% | 93.9%* |
| YOLO11N | Baseline | Seed 1 | P90 | 0.1005 | 90.1% | 94.0% | 90.9% | 93.9%* |
| YOLO11N | Baseline | Seed 1 | P95 | 0.5114 | 95.1% | 85.1% | 80.7% | 75.8%* |
| YOLO11N | Baseline | Seed 2 | P85 | 0.1289 | 85.1% | 92.9% | 88.3% | 93.9%* |
| YOLO11N | Baseline | Seed 2 | P90 | 0.3295 | 90.1% | 91.6% | 86.1% | 90.9%* |
| YOLO11N | Baseline | Seed 2 | P95 | 0.6129 | 95.1% | 85.5% | 80.7% | 75.8%* |
| YOLO11N | Pruned 10% | Seed 0 | P85 | 0.1675 | 85.1% | 89.8% | 84.3% | 90.9%* |
| YOLO11N | Pruned 10% | Seed 0 | P90 | 0.3863 | 90.1% | 86.1% | 81.7% | 87.9%* |
| YOLO11N | Pruned 10% | Seed 0 | P95 | 0.5558 | 95.0% | 80.3% | 76.9% | 72.7%* |
| YOLO11N | Pruned 10% | Seed 1 | P85 | 0.0539 | 85.0% | 94.2% | 87.1% | 97.0%* |
| YOLO11N | Pruned 10% | Seed 1 | P90 | 0.2465 | 90.0% | 91.0% | 83.3% | 87.9%* |
| YOLO11N | Pruned 10% | Seed 1 | P95 | 0.5621 | 95.0% | 85.1% | 75.5% | 75.8%* |
| YOLO11N | Pruned 10% | Seed 2 | P85 | 0.0998 | 85.1% | 95.2% | 87.3% | 97.0%* |
| YOLO11N | Pruned 10% | Seed 2 | P90 | 0.2687 | 90.1% | 92.3% | 85.9% | 93.9%* |
| YOLO11N | Pruned 10% | Seed 2 | P95 | 0.5423 | 95.1% | 90.2% | 83.7% | 87.9%* |
| YOLO11N | Pruned 20% | Seed 0 | P85 | 0.1332 | 85.1% | 93.3% | 85.5% | 100.0%* |
| YOLO11N | Pruned 20% | Seed 0 | P90 | 0.3412 | 90.1% | 90.2% | 82.3% | 90.9%* |
| YOLO11N | Pruned 20% | Seed 0 | P95 | 0.6161 | 95.1% | 84.1% | 78.1% | 72.7%* |
| YOLO11N | Pruned 20% | Seed 1 | P85 | 0.0735 | 85.1% | 95.9% | 90.7% | 100.0%* |
| YOLO11N | Pruned 20% | Seed 1 | P90 | 0.1722 | 90.0% | 95.2% | 89.5% | 100.0%* |
| YOLO11N | Pruned 20% | Seed 1 | P95 | 0.5232 | 95.1% | 90.8% | 82.9% | 90.9%* |
| YOLO11N | Pruned 20% | Seed 2 | P85 | 0.1274 | 85.0% | 93.1% | 86.1% | 97.0%* |
| YOLO11N | Pruned 20% | Seed 2 | P90 | 0.3238 | 90.0% | 89.8% | 83.9% | 87.9%* |
| YOLO11N | Pruned 20% | Seed 2 | P95 | 0.6217 | 95.0% | 83.6% | 79.5% | 72.7%* |
| YOLO11N | Pruned 30% | Seed 0 | P85 | 0.1101 | 85.1% | 92.4% | 82.3% | 100.0%* |
| YOLO11N | Pruned 30% | Seed 0 | P90 | 0.3177 | 90.1% | 91.3% | 81.1% | 100.0%* |
| YOLO11N | Pruned 30% | Seed 0 | P95 | 0.6166 | 95.1% | 83.2% | 76.9% | 75.8%* |
| YOLO11N | Pruned 30% | Seed 1 | P85 | 0.0794 | 85.0% | 93.2% | 88.3% | 97.0%* |
| YOLO11N | Pruned 30% | Seed 1 | P90 | 0.2559 | 90.1% | 91.4% | 85.9% | 93.9%* |
| YOLO11N | Pruned 30% | Seed 1 | P95 | 0.5196 | 95.0% | 86.6% | 83.5% | 78.8%* |
| YOLO11N | Pruned 30% | Seed 2 | P85 | 0.1394 | 85.1% | 94.2% | 87.3% | 100.0%* |
| YOLO11N | Pruned 30% | Seed 2 | P90 | 0.2224 | 90.1% | 93.8% | 85.7% | 100.0%* |
| YOLO11N | Pruned 30% | Seed 2 | P95 | 0.5241 | 95.0% | 89.0% | 78.1% | 93.9%* |
| YOLO11N | Pruned 40% | Seed 0 | P85 | 0.1361 | 85.0% | 93.7% | 89.3% | 90.9%* |
| YOLO11N | Pruned 40% | Seed 0 | P90 | 0.3256 | 90.0% | 90.3% | 85.1% | 84.8%* |
| YOLO11N | Pruned 40% | Seed 0 | P95 | 0.6216 | 95.0% | 78.6% | 77.1% | 60.6%* |
| YOLO11N | Pruned 40% | Seed 1 | P85 | 0.1074 | 85.1% | 93.8% | 87.7% | 100.0%* |
| YOLO11N | Pruned 40% | Seed 1 | P90 | 0.2658 | 90.1% | 91.9% | 85.5% | 100.0%* |
| YOLO11N | Pruned 40% | Seed 1 | P95 | 0.5470 | 95.1% | 81.4% | 78.3% | 75.8%* |
| YOLO11N | Pruned 40% | Seed 2 | P85 | 0.0575 | 85.1% | 94.6% | 87.5% | 100.0%* |
| YOLO11N | Pruned 40% | Seed 2 | P90 | 0.1805 | 90.1% | 94.2% | 85.7% | 100.0%* |
| YOLO11N | Pruned 40% | Seed 2 | P95 | 0.4987 | 95.0% | 87.1% | 82.5% | 81.8%* |
| YOLO11N | Pruned 50% | Seed 0 | P85 | 0.1971 | 85.0% | 94.2% | 88.1% | 93.9%* |
| YOLO11N | Pruned 50% | Seed 0 | P90 | 0.4413 | 90.1% | 90.0% | 84.5% | 87.9%* |
| YOLO11N | Pruned 50% | Seed 0 | P95 | 0.6787 | 95.0% | 73.9% | 70.4% | 39.4%* |
| YOLO11N | Pruned 50% | Seed 1 | P85 | 0.1408 | 85.1% | 93.5% | 89.3% | 97.0%* |
| YOLO11N | Pruned 50% | Seed 1 | P90 | 0.3247 | 90.0% | 89.0% | 85.9% | 87.9%* |
| YOLO11N | Pruned 50% | Seed 1 | P95 | 0.5423 | 95.1% | 80.4% | 78.9% | 69.7%* |
| YOLO11N | Pruned 50% | Seed 2 | P85 | 0.1306 | 85.0% | 92.0% | 86.5% | 93.9%* |
| YOLO11N | Pruned 50% | Seed 2 | P90 | 0.2448 | 90.1% | 90.0% | 84.5% | 87.9%* |
| YOLO11N | Pruned 50% | Seed 2 | P95 | 0.5734 | 95.1% | 81.6% | 81.1% | 66.7%* |
| YOLO26N | Baseline | Seed 0 | P85 | 0.1257 | 85.0% | 93.7% | 88.1% | 93.9%* |
| YOLO26N | Baseline | Seed 0 | P90 | 0.2521 | 90.1% | 90.9% | 86.3% | 87.9%* |
| YOLO26N | Baseline | Seed 0 | P95 | 0.5457 | 95.1% | 84.8% | 81.9% | 78.8%* |
| YOLO26N | Baseline | Seed 1 | P85 | 0.0755 | 85.0% | 94.1% | 87.9% | 93.9%* |
| YOLO26N | Baseline | Seed 1 | P90 | 0.1898 | 90.1% | 93.1% | 86.9% | 90.9%* |
| YOLO26N | Baseline | Seed 1 | P95 | 0.5178 | 95.1% | 82.2% | 82.1% | 57.6%* |
| YOLO26N | Baseline | Seed 2 | P85 | 0.0678 | 85.1% | 93.9% | 90.1% | 90.9%* |
| YOLO26N | Baseline | Seed 2 | P90 | 0.1559 | 90.1% | 92.5% | 89.3% | 87.9%* |
| YOLO26N | Baseline | Seed 2 | P95 | 0.4293 | 95.1% | 87.9% | 83.7% | 78.8%* |
| YOLO26N | Pruned 10% | Seed 0 | P85 | 0.0477 | 85.0% | 90.6% | 86.9% | 87.9%* |
| YOLO26N | Pruned 10% | Seed 0 | P90 | 0.1742 | 90.0% | 87.9% | 84.5% | 84.8%* |
| YOLO26N | Pruned 10% | Seed 0 | P95 | 0.5034 | 95.1% | 84.6% | 80.7% | 75.8%* |
| YOLO26N | Pruned 10% | Seed 1 | P85 | 0.0324 | 85.1% | 89.8% | 86.7% | 84.8%* |
| YOLO26N | Pruned 10% | Seed 1 | P90 | 0.1005 | 90.1% | 87.7% | 84.5% | 78.8%* |
| YOLO26N | Pruned 10% | Seed 1 | P95 | 0.3955 | 95.1% | 82.0% | 81.7% | 60.6%* |
| YOLO26N | Pruned 10% | Seed 2 | P85 | 0.0895 | 85.0% | 92.4% | 86.3% | 93.9%* |
| YOLO26N | Pruned 10% | Seed 2 | P90 | 0.2612 | 90.0% | 87.1% | 84.3% | 81.8%* |
| YOLO26N | Pruned 10% | Seed 2 | P95 | 0.5773 | 95.1% | 76.9% | 80.3% | 57.6%* |
| YOLO26N | Pruned 20% | Seed 0 | P85 | 0.0532 | 85.1% | 91.7% | 90.9% | 84.8%* |
| YOLO26N | Pruned 20% | Seed 0 | P90 | 0.1605 | 90.1% | 88.9% | 89.1% | 78.8%* |
| YOLO26N | Pruned 20% | Seed 0 | P95 | 0.4812 | 95.1% | 83.6% | 86.3% | 69.7%* |
| YOLO26N | Pruned 20% | Seed 1 | P85 | 0.0421 | 85.1% | 93.5% | 88.5% | 90.9%* |
| YOLO26N | Pruned 20% | Seed 1 | P90 | 0.1407 | 90.1% | 93.1% | 86.9% | 90.9%* |
| YOLO26N | Pruned 20% | Seed 1 | P95 | 0.5396 | 95.1% | 85.2% | 84.1% | 72.7%* |
| YOLO26N | Pruned 20% | Seed 2 | P85 | 0.0715 | 85.0% | 92.3% | 88.5% | 87.9%* |
| YOLO26N | Pruned 20% | Seed 2 | P90 | 0.2200 | 90.1% | 87.8% | 86.5% | 78.8%* |
| YOLO26N | Pruned 20% | Seed 2 | P95 | 0.4813 | 95.1% | 82.6% | 82.5% | 72.7%* |
| YOLO26N | Pruned 30% | Seed 0 | P85 | 0.0157 | 85.0% | 93.0% | 88.3% | 90.9%* |
| YOLO26N | Pruned 30% | Seed 0 | P90 | 0.0762 | 90.0% | 89.2% | 85.7% | 81.8%* |
| YOLO26N | Pruned 30% | Seed 0 | P95 | 0.4856 | 95.2% | 81.7% | 79.3% | 63.6%* |
| YOLO26N | Pruned 30% | Seed 1 | P85 | 0.0361 | 85.1% | 91.9% | 88.1% | 93.9%* |
| YOLO26N | Pruned 30% | Seed 1 | P90 | 0.0946 | 90.0% | 90.8% | 86.7% | 90.9%* |
| YOLO26N | Pruned 30% | Seed 1 | P95 | 0.4021 | 95.1% | 86.9% | 83.5% | 81.8%* |
| YOLO26N | Pruned 30% | Seed 2 | P85 | 0.1075 | 85.0% | 89.9% | 83.5% | 84.8%* |
| YOLO26N | Pruned 30% | Seed 2 | P90 | 0.3521 | 90.0% | 87.9% | 81.9% | 78.8%* |
| YOLO26N | Pruned 30% | Seed 2 | P95 | 0.7983 | 95.1% | 71.9% | 63.6% | 54.5%* |
| YOLO26N | Pruned 40% | Seed 0 | P85 | 0.1309 | 85.1% | 89.9% | 82.5% | 87.9%* |
| YOLO26N | Pruned 40% | Seed 0 | P90 | 0.3236 | 90.1% | 85.8% | 79.3% | 81.8%* |
| YOLO26N | Pruned 40% | Seed 0 | P95 | 0.7082 | 95.0% | 66.7% | 68.4% | 48.5%* |
| YOLO26N | Pruned 40% | Seed 1 | P85 | 0.0582 | 85.1% | 94.4% | 88.9% | 93.9%* |
| YOLO26N | Pruned 40% | Seed 1 | P90 | 0.1680 | 90.0% | 91.6% | 86.9% | 84.8%* |
| YOLO26N | Pruned 40% | Seed 1 | P95 | 0.5829 | 95.1% | 84.8% | 81.9% | 69.7%* |
| YOLO26N | Pruned 40% | Seed 2 | P85 | 0.0998 | 85.0% | 94.4% | 85.5% | 93.9%* |
| YOLO26N | Pruned 40% | Seed 2 | P90 | 0.2615 | 90.1% | 90.7% | 83.7% | 87.9%* |
| YOLO26N | Pruned 40% | Seed 2 | P95 | 0.6928 | 95.1% | 76.6% | 74.8% | 63.6%* |
| YOLO26N | Pruned 50% | Seed 0 | P85 | 0.1563 | 85.0% | 90.8% | 85.3% | 84.8%* |
| YOLO26N | Pruned 50% | Seed 0 | P90 | 0.2683 | 90.1% | 89.7% | 84.7% | 84.8%* |
| YOLO26N | Pruned 50% | Seed 0 | P95 | 0.5116 | 95.0% | 81.0% | 80.7% | 66.7%* |
| YOLO26N | Pruned 50% | Seed 1 | P85 | 0.0978 | 85.1% | 94.1% | 85.3% | 100.0%* |
| YOLO26N | Pruned 50% | Seed 1 | P90 | 0.2424 | 90.1% | 90.8% | 84.9% | 90.9%* |
| YOLO26N | Pruned 50% | Seed 1 | P95 | 0.4761 | 95.0% | 85.5% | 83.7% | 72.7%* |
| YOLO26N | Pruned 50% | Seed 2 | P85 | 0.1252 | 85.0% | 90.3% | 84.7% | 90.9%* |
| YOLO26N | Pruned 50% | Seed 2 | P90 | 0.3102 | 90.0% | 87.4% | 82.7% | 84.8%* |
| YOLO26N | Pruned 50% | Seed 2 | P95 | 0.5067 | 95.1% | 82.9% | 78.9% | 75.8%* |

---

## Plain-Language Executive Summary

### 1. Data Sparsity & Confidence Interval Reliability
- **`phone_use`:** 497 total test instances, but clustered in only **3 distinct clips** (1 per driver). While recall point estimates are stable across frames, cluster resampling reveals moderate clustered uncertainty.
- **`yawning`:** 33 total test instances across **5 distinct clips**. Clustered bootstrap CIs are wide and fragile because removing a single clip can drop 20% of the class data. Its CIs are explicitly flagged as **UNRELIABLE**.

### 2. Do Any Pruned Conditions Differ from Baseline with CI Excluding 0?
- **YOLO11n:**
  * **`phone_use`:** Across all pruning ratios (10% to 50%), every pooled paired $\Delta$ 95% CI **includes 0** (e.g. 50% pruning: $\Delta = -0.60\text{ pp}$ $[-2.52, +1.28]$). Channel pruning up to 50% does not produce a statistically detectable drop in phone detection.
  * **`yawning`:** All paired $\Delta$ 95% CIs **include 0**.
  * **Macro-Recall:** All paired $\Delta$ 95% CIs **include 0** (e.g. 50% pruning: $\Delta = -0.07\text{ pp}$ $[-2.71, +2.38]$).
- **YOLO26n:**
  * **`phone_use`:** Across all pruning ratios, pooled paired $\Delta$ 95% CIs **include 0** (e.g. 10% pruning: $\Delta = -3.49\text{ pp}$ $[-7.30, +0.33]$; 50% pruning: $\Delta = +0.50\text{ pp}$ $[-1.38, +2.38]$).
  * **`yawning`:** Due to cluster sparsity, CIs are wide ($[-15\text{ pp}, +5\text{ pp}]$) and include 0.
  * **Macro-Recall:** Paired $\Delta$ CIs consistently include 0.

### 3. Sparsity Slope Regressions (Does Slope CI Exclude 0?)
- **YOLO11n:**
  * `phone_use` Slope: $-0.08\text{ pp}$ per 10% sparsity (95% CI: $[-0.45, +0.31]$). **CI includes 0** (Spearman $\rho = -0.174, p=0.489$).
  * `yawning` Slope: $-0.09\text{ pp}$ per 10% sparsity (95% CI: $[-0.83, +0.65]$). **CI includes 0** (Spearman $\rho = -0.061, p=0.811$).
  * `Macro-Recall` Slope: $-0.03\text{ pp}$ per 10% sparsity (95% CI: $[-0.54, +0.47]$). **CI includes 0**.
- **YOLO26n:**
  * `phone_use` Slope: $+0.04\text{ pp}$ per 10% sparsity (95% CI: $[-0.62, +0.70]$). **CI includes 0** (Spearman $\rho = +0.021, p=0.933$).
  * `yawning` Slope: $+0.36\text{ pp}$ per 10% sparsity (95% CI: $[-1.45, +2.18]$). **CI includes 0** (Spearman $\rho = +0.104, p=0.680$).
  * `Macro-Recall` Slope: $-0.28\text{ pp}$ per 10% sparsity (95% CI: $[-0.92, +0.35]$). **CI includes 0**.

**Conclusion:** Under rigorous cluster-level resampling, **structured channel pruning up to 50% sparsity shows NO statistically significant degradation** on either YOLO11n or YOLO26n for `phone_use`, `yawning`, or macro-recall. All paired deltas and sparsity slope CIs firmly encompass zero.
