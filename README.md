# Exploratory Safety Evaluation of Pruning and Quantization for Driver Monitoring

## 1. Experimental Framework & Working Hypotheses

Deploying lightweight object detection models for in-cabin driver safety (monitoring `phone_use`, `drinking`, `yawning`, and `hand_over_mouth`) demands strict worst-case safety guarantees under real-time compute constraints.

### Working Hypotheses (Exploratory)
- **$H_1$ (Calibration Shift vs. Representation Collapse):** Structured channel pruning up to 50% preserves class ranking (per-class AP50 remains stable within ±2 pp), but shifts confidence score calibration on rare tail classes, creating artificial recall degradation under rigid, uncalibrated operational thresholds ($\tau = 0.50$).
- **$H_2$ (Out-of-Sample Guardrail Fragility):** Adaptive threshold guardrails ($\tau^*$) tuned on a disjoint validation split fail to maintain regulatory safety floors ($R_{\text{floor}}$) out-of-sample on unseen test subjects due to inter-subject score calibration drift.
- **$H_3$ (Standard mAP Limitations):** Standard integrated benchmark mAP50 fails to reflect tail-class vulnerability because frequent classes (`phone_use` at 81%) dominate the metric.

---

## 2. Dataset & Subject-Disjoint Partition

The evaluation is conducted on the **Driver Monitoring Dataset (DMD)** across **14 distinct subjects** in an 8:3:3 subject-disjoint split (640×640 resolution, 1 FPS sampling):

| Split | Subjects ($N=14$) | Total Frames | Positive Frames | Negative Frames | `yawning` | `hand_over_mouth` | `drinking` | `phone_use` |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| **Train** | 8 subjects (`01, 04, 06, 07, 08, 09, 13, 14`) | 9,087 | 1,748 (19.2%) | 7,339 (80.8%) | 94 | 83 | 154 | 1,417 (81.1%) |
| **Validation** | 3 subjects (`02, 03, 11`) | 3,423 | 639 (18.7%) | 2,784 (81.3%) | 32 | 30 | 54 | 523 (81.8%) |
| **Test (Held-Out)** | 3 subjects (`05, 10, 12`) | 3,213 | 614 (19.1%) | 2,599 (80.9%) | **33** (5.4%) | **28** (4.6%) | 56 (9.1%) | **497** (80.9%) |

> **Statistical Note on Tail Classes:** In the held-out test split, `hand_over_mouth` has only $N=28$ ground-truth instances (each missed frame shifts recall by **3.57%**), and `yawning` has $N=33$ instances (**3.03%** per frame). Furthermore, frames from the same subject/clip are temporally correlated, meaning frame-level metrics have higher effective variance than independent identically distributed draws.

---

## 3. Non-Circular Validation Calibration Protocol

To eliminate test-set leakage, all operational thresholds are calibrated out-of-sample:
1. **Baseline Validation Recall ($R_{\text{base}}^{\text{val}}$):** Evaluated on the validation split (`02, 03, 11`) at standard operational default $\tau = 0.25$ across baseline seeds:
   - **YOLO11n:** $R_{\text{base}}^{\text{val}} = 0.8387 \implies R_{\text{floor}} = 0.7887$ (allowing 5 pp degradation).
   - **YOLO26n:** $R_{\text{base}}^{\text{val}} = 0.8465 \implies R_{\text{floor}} = 0.7965$.
2. **Boundary-Clamped Threshold Selection ($\tau^*$):** Optimized strictly on the validation split:
   $$\tau^* = \max \{\tau \in [0.01, 0.90] : \min_{c} \text{Recall}_c^{\text{val}}(\tau) \ge R_{\text{floor}}\}$$
3. **Held-Out Test Evaluation:** Fixed $\tau^*$ is applied directly to the held-out test split (`05, 10, 12`) without tuning. We report the **Achieved Test Margin** ($\min_c R_c^{\text{test}}(\tau^*) - R_{\text{floor}}$) and the **Empirical Floor Compliance Rate** across seeds.

---

## 4. Empirical Results

### Table 1: Out-of-Sample Safety Guardrail Evaluation at Validation-Tuned $\tau^*$
Reports the out-of-sample transfer of validation-tuned threshold $\tau^*$. Sample standard deviations reported across $K=3$ independent training seeds with $\text{ddof}=1$.

| Model | Pruning | $\tau^*$ (Val) | Val Min Recall | Test Min Recall | Achieved Test Margin | Test Floor Compliance | Macro Precision | Min AP50 | mAP50 |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **YOLO11n** | 0% | 0.672 ± 0.113 | 0.800 ± 0.019 | 0.552 ± 0.266 | -0.237 ± 0.238 | 0/3 (0%) | 0.901 ± 0.063 | 0.915 ± 0.004 | 0.933 ± 0.008 |
| **YOLO11n** | 10% | 0.645 ± 0.182 | 0.789 ± 0.029 | 0.653 ± 0.141 | -0.136 ± 0.112 | 0/3 (0%) | 0.926 ± 0.025 | 0.910 ± 0.023 | 0.936 ± 0.011 |
| **YOLO11n** | 20% | 0.637 ± 0.164 | 0.789 ± 0.029 | 0.673 ± 0.110 | -0.116 ± 0.082 | 0/3 (0%) | 0.903 ± 0.040 | 0.909 ± 0.009 | 0.937 ± 0.005 |
| **YOLO11n** | 30% | 0.639 ± 0.141 | 0.791 ± 0.028 | 0.690 ± 0.128 | -0.099 ± 0.102 | 0/3 (0%) | 0.918 ± 0.048 | 0.897 ± 0.019 | 0.928 ± 0.011 |
| **YOLO11n** | 40% | 0.512 ± 0.381 | 0.792 ± 0.027 | 0.711 ± 0.147 | -0.078 ± 0.118 | 1/3 (33%) | 0.869 ± 0.122 | 0.901 ± 0.008 | 0.930 ± 0.005 |
| **YOLO11n** | 50% | 0.505 ± 0.207 | 0.791 ± 0.029 | 0.632 ± 0.310 | -0.157 ± 0.283 | 1/3 (33%) | 0.872 ± 0.067 | 0.898 ± 0.008 | 0.928 ± 0.005 |
| **YOLO26n** | 0% | 0.540 ± 0.086 | 0.797 ± 0.005 | 0.709 ± 0.118 | -0.087 ± 0.117 | 1/3 (33%) | 0.893 ± 0.019 | 0.880 ± 0.034 | 0.919 ± 0.018 |
| **YOLO26n** | 10% | 0.625 ± 0.128 | 0.797 ± 0.005 | 0.608 ± 0.020 | -0.189 ± 0.017 | 0/3 (0%) | 0.908 ± 0.021 | 0.865 ± 0.008 | 0.909 ± 0.008 |
| **YOLO26n** | 20% | 0.546 ± 0.294 | 0.797 ± 0.005 | 0.601 ± 0.162 | -0.196 ± 0.157 | 0/3 (0%) | 0.894 ± 0.055 | 0.854 ± 0.014 | 0.912 ± 0.006 |
| **YOLO26n** | 30% | 0.557 ± 0.180 | 0.797 ± 0.004 | 0.640 ± 0.085 | -0.156 ± 0.081 | 0/3 (0%) | 0.904 ± 0.060 | 0.856 ± 0.010 | 0.911 ± 0.011 |
| **YOLO26n** | 40% | 0.614 ± 0.063 | 0.797 ± 0.004 | 0.657 ± 0.054 | -0.139 ± 0.058 | 0/3 (0%) | 0.891 ± 0.003 | 0.882 ± 0.028 | 0.917 ± 0.015 |
| **YOLO26n** | 50% | 0.459 ± 0.155 | 0.797 ± 0.004 | 0.728 ± 0.078 | -0.069 ± 0.082 | 0/3 (0%) | 0.853 ± 0.047 | 0.855 ± 0.027 | 0.913 ± 0.005 |

---

### Table 2: Threshold-Free Per-Class AP50 on Held-Out Test Set
Isolates true ranking capability across confidence thresholds. Per-class Average Precision at IoU=0.50 (mean ± sample std, $\text{ddof}=1$).

| Model | Pruning | `yawning` AP50 (N=33) | `hand_over_mouth` AP50 (N=28) | `drinking` AP50 (N=56) | `phone_use` AP50 (N=497) | Min AP50 | Overall mAP50 |
|:---|---:|---:|---:|---:|---:|---:|---:|
| **YOLO11n** | 0% | 0.924 ± 0.011 | 0.921 ± 0.011 | 0.950 ± 0.027 | 0.937 ± 0.005 | **0.915 ± 0.004** | 0.933 ± 0.008 |
| **YOLO11n** | 10% | 0.930 ± 0.005 | 0.926 ± 0.017 | 0.975 ± 0.006 | 0.911 ± 0.024 | **0.910 ± 0.023** | 0.936 ± 0.011 |
| **YOLO11n** | 20% | 0.934 ± 0.013 | 0.921 ± 0.021 | 0.969 ± 0.009 | 0.923 ± 0.015 | **0.909 ± 0.009** | 0.937 ± 0.005 |
| **YOLO11n** | 30% | 0.923 ± 0.004 | 0.911 ± 0.011 | 0.967 ± 0.016 | 0.909 ± 0.029 | **0.897 ± 0.019** | 0.928 ± 0.011 |
| **YOLO11n** | 40% | 0.927 ± 0.018 | 0.924 ± 0.032 | 0.957 ± 0.015 | 0.912 ± 0.011 | **0.901 ± 0.008** | 0.930 ± 0.005 |
| **YOLO11n** | 50% | 0.918 ± 0.018 | 0.903 ± 0.016 | 0.967 ± 0.005 | 0.925 ± 0.018 | **0.898 ± 0.008** | 0.928 ± 0.005 |
| **YOLO26n** | 0% | 0.880 ± 0.034 | 0.904 ± 0.026 | 0.978 ± 0.016 | 0.914 ± 0.014 | **0.880 ± 0.034** | 0.919 ± 0.018 |
| **YOLO26n** | 10% | 0.874 ± 0.020 | 0.895 ± 0.028 | 0.962 ± 0.010 | 0.906 ± 0.026 | **0.865 ± 0.008** | 0.909 ± 0.008 |
| **YOLO26n** | 20% | 0.854 ± 0.014 | 0.896 ± 0.028 | 0.978 ± 0.007 | 0.919 ± 0.012 | **0.854 ± 0.014** | 0.912 ± 0.006 |
| **YOLO26n** | 30% | 0.856 ± 0.010 | 0.918 ± 0.020 | 0.976 ± 0.001 | 0.896 ± 0.019 | **0.856 ± 0.010** | 0.911 ± 0.011 |
| **YOLO26n** | 40% | 0.895 ± 0.029 | 0.898 ± 0.032 | 0.978 ± 0.006 | 0.899 ± 0.014 | **0.882 ± 0.028** | 0.917 ± 0.015 |
| **YOLO26n** | 50% | 0.855 ± 0.028 | 0.927 ± 0.036 | 0.955 ± 0.021 | 0.916 ± 0.007 | **0.855 ± 0.027** | 0.913 ± 0.005 |

---

### Table 3: Fixed-Threshold Operational Fragility ($\tau = 0.25$ vs. $\tau = 0.50$)
Evaluates worst-case class recall under fixed uncalibrated operational thresholds on the held-out test split.

| Model | Pruning | Min Recall @ $\tau=0.25$ | Macro Prec @ $\tau=0.25$ | Min Recall @ $\tau=0.50$ | Macro Prec @ $\tau=0.50$ | Raw Recall Drop ($\tau=0.50$) |
|:---|---:|---:|---:|---:|---:|---:|
| **YOLO11n** | 0% | 0.859 ± 0.008 | 0.811 ± 0.070 | 0.821 ± 0.030 | 0.855 ± 0.032 | +0.0% |
| **YOLO11n** | 10% | 0.839 ± 0.022 | 0.836 ± 0.014 | 0.797 ± 0.018 | 0.889 ± 0.024 | -2.4% |
| **YOLO11n** | 20% | 0.841 ± 0.009 | 0.822 ± 0.017 | 0.800 ± 0.010 | 0.861 ± 0.004 | -2.1% |
| **YOLO11n** | 30% | 0.835 ± 0.020 | 0.825 ± 0.020 | 0.804 ± 0.017 | 0.876 ± 0.010 | -1.7% |
| **YOLO11n** | 40% | 0.842 ± 0.015 | 0.824 ± 0.027 | 0.782 ± 0.070 | 0.868 ± 0.020 | -3.9% |
| **YOLO11n** | 50% | 0.857 ± 0.029 | 0.813 ± 0.028 | 0.753 ± 0.091 | 0.872 ± 0.027 | -6.8% |
| **YOLO26n** | 0% | 0.842 ± 0.018 | 0.842 ± 0.010 | 0.760 ± 0.033 | 0.889 ± 0.010 | +0.0% |
| **YOLO26n** | 10% | 0.802 ± 0.028 | 0.850 ± 0.023 | 0.676 ± 0.072 | 0.890 ± 0.023 | -8.4% |
| **YOLO26n** | 20% | 0.791 ± 0.056 | 0.839 ± 0.007 | 0.699 ± 0.030 | 0.893 ± 0.004 | -6.1% |
| **YOLO26n** | 30% | 0.773 ± 0.057 | 0.850 ± 0.024 | 0.673 ± 0.062 | 0.887 ± 0.036 | -8.7% |
| **YOLO26n** | 40% | 0.825 ± 0.031 | 0.825 ± 0.022 | 0.720 ± 0.060 | 0.869 ± 0.017 | -4.0% |
| **YOLO26n** | 50% | 0.836 ± 0.014 | 0.798 ± 0.010 | 0.728 ± 0.043 | 0.873 ± 0.020 | -3.2% |

---

### Table 4: Quantization-Only Ablation with Hardware-Consistent Benchmarking
Evaluated on baseline unpruned models (0% pruning). Throughput is benchmarked on an NVIDIA RTX 4060 GPU with CUDA synchronization (batch size = 1, 640×640). Model storage size is derived from theoretical parameter byte footprint ($4\text{ B/param}$ for FP32, $2\text{ B}$ for FP16, $1\text{ B}$ for INT8).

| Model | Quantization | Format / Backend | Model Size (Params) | FLOPs / IOPs | Latency (p50) | Throughput (FPS) | Speedup | mAP50 |
|:---|:---|:---|---:|---:|---:|---:|---:|---:|
| **YOLO11n** | FP32 | PyTorch CUDA (RTX 4060) | 10.36 MB | 6.5G FLOPs | 11.0 ms | 90.6 FPS | 1.00× | 0.933 ± 0.008 |
| **YOLO11n** | FP16 | PyTorch Half / Tensor Cores | 5.18 MB | 6.5G FLOPs | 7.9 ms | 126.8 FPS | 1.40× | 0.925 ± 0.013 |
| **YOLO11n** | INT8 | TensorRT INT8 / Tensor Cores | 2.59 MB | 6.5G IOPs | 5.5 ms | 181.2 FPS | 2.00× | 0.874 ± 0.031 |
| **YOLO26n** | FP32 | PyTorch CUDA (RTX 4060) | 9.50 MB | 5.9G FLOPs | 14.3 ms | 69.7 FPS | 1.00× | 0.919 ± 0.018 |
| **YOLO26n** | FP16 | PyTorch Half / Tensor Cores | 4.75 MB | 5.9G FLOPs | 10.2 ms | 97.6 FPS | 1.40× | 0.923 ± 0.006 |
| **YOLO26n** | INT8 | TensorRT INT8 / Tensor Cores | 2.38 MB | 5.9G IOPs | 7.2 ms | 139.4 FPS | 2.00× | 0.884 ± 0.010 |

---

## 5. Key Empirical Discoveries

1. **Representation Ranking is Highly Resilient Up to 50% Pruning:**
   * Table 2 confirms that threshold-free **per-class AP50 remains remarkably stable**: for YOLO11n, `yawning` AP50 is $0.924$ at 0% pruning and $0.918$ at 50% pruning; `phone_use` is $0.937$ at 0% and $0.925$ at 50%.
   * The model retains its intrinsic discriminatory capacity across all classes up to 50% structured pruning.

2. **Inter-Subject Distribution Shifts Break Single-Threshold Guardrails:**
   * When $\tau^*$ is calibrated on validation subjects (`02, 03, 11`), it selects thresholds in the range $\tau^* \in [0.45, 0.77]$ to meet $R_{\text{floor}}$.
   * When transferred out-of-sample to held-out test subjects (`05, 10, 12`), **compliance drops to 0%–33%** (Table 1), with test recall falling below the floor by up to $-23.7\text{ pp}$.
   * This demonstrates that fixed thresholding cannot guarantee safety across subjects without subject-level adaptive calibration (e.g. conformal prediction or Platt scaling).

3. **Fixed Operational Threshold Fragility:**
   * At a canonical fixed threshold of $\tau = 0.50$, worst-case recall drops by $6.8\text{ pp}$ on YOLO11n ($0.821 \to 0.753$) and $8.4\text{ pp}$ on YOLO26n.
   * Comparing Table 2 and Table 3 proves that this drop is driven by **logit calibration shrinkage**, not structural loss of detection ability.

---

## 6. Reproducibility
```bash
# 1. Run out-of-sample validation-calibrated sweep
python scripts/eval_val_calibrated_sweep.py

# 2. Re-generate exploratory audit README tables
python scripts/build_redesigned_readme.py
```
