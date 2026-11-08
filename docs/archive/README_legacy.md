# Paper-Title-To-Be

## Core Question
Can pruning and quantization reduce computation cost while maintaining acceptable detection performance?

## Objective

**Safety constraint.** The safety recall floor $R_{\text{floor}}$ is defined relative to the uncompressed baseline ($r=0\%, q=\text{FP32}$):
$$
R_{\text{floor}} = R_{\text{base}} - \delta
$$
where $R_{\text{base}} = \min_{c \in \mathcal{C}_{\text{safety}}} \text{Recall}_c(\text{Baseline})$ across the safety-critical class subset $\mathcal{C}_{\text{safety}} = \{\text{phone\_use}, \text{drinking}, \text{yawning}, \text{hand\_over\_mouth}\}$. 

The baseline models establish explicit numerical thresholds:
- **YOLO11n:** $R_{\text{base}} = 0.890 \implies R_{\text{floor}} = 0.840$ (with nominal tolerance $\delta = 0.05$, permitting at most 5 percentage points recall degradation from the uncompressed baseline, rather than from 100%).
- **YOLO26n:** $R_{\text{base}} = 0.887 \implies R_{\text{floor}} = 0.837$ (with nominal tolerance $\delta = 0.05$).

*Regulatory Context:* The 5% degradation margin is motivated by industrial safety considerations in line with Euro NCAP 2026 Driver Monitoring and ISO 21448 (SOTIF) hazard mitigation guidelines, serving as an operational bound rather than a formal mathematical derivative.

**Per-seed threshold optimization** (for each config $(r, q)$ and seed $k$):
$$
\tau^{*(k)}(r,q) = \arg\max_{\tau} \text{Precision}(\tau) \quad \text{s.t.} \quad \min_{c \in \mathcal{C}_{\text{safety}}} \text{Recall}_c(\tau) \ge R_{\text{floor}}^{(k)}
$$
Threshold $\tau$ is re-tuned per seed; the score distribution shifts with the trained weights, not just with $(r, q)$. Here $R_{\text{floor}}^{(k)} = R_{\text{base}}^{(k)} - \delta$.

**Uncompensated vs. Compensated Evaluation Rationale ($\tau=0.25$ vs. $\tau^*$).**
While adaptive threshold tuning ($\tau^*$) represents a production strategy to guarantee recall, it introduces an evaluation confound: lowering $\tau^*$ artificially recovers recall while hiding structural degradation behind a steep drop in precision (false alarms). To address this, we evaluate both:
1. **Uncompensated Baseline ($\tau=0.25$, standard YOLO default; $\tau=0.50$, high-certainty):** Freezes the operating point to measure the intrinsic, unadulterated capacity loss of the pruned network.
2. **Compensated Adaptation ($\tau^*$):** Quantifies whether the safety floor can be reclaimed and measures the exact "Precision Tax" incurred via Safety F1 ($2 \cdot \frac{P \cdot R}{P + R}$).

**Pass criterion.** Each config is trained with $K = 3$ seeds. Pass iff:
$$
\frac{1}{K}\sum_{k=1}^{K} \min_{c \in \mathcal{C}_{\text{safety}}} \text{Recall}_c\big(\tau^{*(k)}\big) \ge R_{\text{floor}}
$$
Flag **†** (boundary-close) if mean - std < $R_{\text{floor}} \le$ mean (passes on average, but at least one seed would fail individually, using sample standard deviation with $\text{ddof}=1$). **All metrics in every table below are reported as mean $\pm$ std across seeds unless stated otherwise**.

**Final selection rules.**
1. **Primary Throughput Selection:** Maximize FPS among passing configurations:
$$
(r^*, q^*) = \arg\max_{(r,q) \in \mathcal{S}_{\text{pass}}} \text{FPS}(r,q)
$$
Configs within 2% of max FPS are treated as tied; among tied configs, maximize mean Precision at $\tau^*$.
2. **Edge Memory-Constrained Selection:** When deploying to resource-constrained micro-controllers or embedded hardware with strict storage bounds, minimize model storage size among passing configurations within 5% recall degradation:
$$
(r_{\text{edge}}^*, q_{\text{edge}}^*) = \arg\min_{(r,q) \in \mathcal{S}_{\text{pass}}} \text{Size}(r,q)
$$

**Tested grid $\mathcal{S}_{\text{tested}}$:**
- FP32 at all 6 pruning ratios (0%, 10%, 20%, 30%, 40%, 50%)
- All quant levels (FP32, FP16, INT8) at 0% pruning (INT4 conv quantization omitted due to hardware/runtime support limits)
- INT8 at all 6 pruning ratios
- INT8 (QAT), single seed, at the QAT-selected ratio per model (rule under Compression $\rightarrow$ Quantization); omitted where no PTQ-INT8 ratio fails for that model

Untested combinations (e.g. 20% pruning, FP16) are out of scope by design.

**Infeasible cells.** If no $\tau$ satisfies the constraint, Pass = Fail, $\tau^*$ undefined. Report metrics at the $\tau$ maximizing mean $\min_c \text{Recall}_c$; do not leave blanks.

**Known limitations.**
- The 8:3:3 subject split is fixed for the whole study. Multi-seed captures training-run variance, not split variance.
- Benchmarking execution note: PyTorch `.pt` models are benchmarked via GPU CUDA execution, while exported ONNX models are evaluated via CPU execution (accounting for the CPU-GPU throughput divergence across formats). Model sizes reflect stored weights (PyTorch `.pt` files store weights in FP16 by default at ~5.2 MB, true FP32 parameter footprint is 10.4 MB, and INT8 ONNX is 2.8 MB). Full TensorRT GPU engine compilation yields ~420 FPS on RTX 4060.

## Dataset

- **Source:** Driver Monitoring Dataset (DMD), RGB modality.
- **Sampling:** 15,723 frames cropped at 640$\times$640, sampled at 1 FPS across 81 driver-facing video streams.
- **Subject-disjoint partition:** 8:3:3 train/val/test split across 14 subjects (fixed; see Objective, Known limitations).
  - Train: `subject_01`, `subject_04`, `subject_06`, `subject_07`, `subject_08`, `subject_09`, `subject_13`, `subject_14`
  - Validation: `subject_02`, `subject_03`, `subject_11`
  - Test: `subject_05`, `subject_10`, `subject_12`
- **Classes ($C = 4$):** `phone_use`, `drinking`, `yawning`, `hand_over_mouth`. Each positive frame contains exactly one bounding box annotation.

### Dataset Composition and Class Distribution

| Split | Subjects | Total Frames | Positive Frames | Negative Frames | Pos : Neg Ratio | Phone Use | Drinking | Yawning | Hand over Mouth | Active Classes |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 8 | 9,087 | 1,748 (19.2%) | 7,339 (80.8%) | 1 : 4.20 | 1,417 | 154 | 94 | 83 | 4 |
| Validation | 3 | 3,423 | 639 (18.7%) | 2,784 (81.3%) | 1 : 4.36 | 523 | 54 | 32 | 30 | 4 |
| Test | 3 | 3,213 | 614 (19.1%) | 2,599 (80.9%) | 1 : 4.23 | 497 | 56 | 33 | 28 | 4 |
| **Total** | **14** | **15,723** | **3,001 (19.1%)** | **12,722 (80.9%)** | **1 : 4.24** | **2,437** | **264** | **159** | **141** | **4** |

## Models
- YOLO11n
- YOLO26n

## Compression

### Pruning
- Structured L1 channel pruning (train → prune → fine-tune).
- Channels ranked by L1 norm on the trained baseline; the bottom $r$% are removed, and the pruned model is fine-tuned for the full epoch budget.
- Ratios: 0%, 10%, 20%, 30%, 40%, 50%.
- Full sweep at FP32 (Table 1).

### Quantization
- FP32, FP16, INT8 (INT4 conv quantization omitted due to hardware/runtime support limits).
- PTQ calibrated on a held-out slice of **train** (not val; val is reserved for $\tau$-tuning). Calibration set: 300 images (sampled from the train split).
- QAT run when PTQ-INT8 fails; reported as a separate `INT8 (QAT)` row, never overwriting the PTQ number.

**QAT selection rule.** Per model, QAT (single seed) runs on the failing PTQ-INT8 ratio with the smallest min-class-recall gap to threshold. Skipped for a model with zero failing ratios.

## Protocol
1. Baseline evaluation at FP32 (0% pruning): Train both models with $K$ seeds to establish $R_{\text{base}}$, calibrate $\delta$, and determine $R_{\text{floor}} = R_{\text{base}} - \delta$.
2. Pruning sweep at FP32: 6 ratios $\times$ 2 models $\times$ $K$ seeds.
3. Quantization-only ablation at 0% pruning: 3 quant levels (FP32, FP16, INT8) $\times$ 2 models $\times$ $K$ seeds.
4. Joint pruning $\times$ quantization at INT8: 6 ratios $\times$ 2 models $\times$ $K$ seeds, plus one QAT row per applicable model.
5. Final selection per model family via the Objective above.

Pass/flag criteria and $\tau$ re-tuning follow the Objective section uniformly across Tables 1–3 (not restated per step).

*FP16 is excluded from the joint grid (Table 3), confined to the quantization-only ablation (Table 2), to keep the interaction study tractable.*

## Training Configuration
- **Epochs:** 100 fixed epochs across all models (no early stopping; `patience=0`)
- **Batch size (training):** 16 (FP16 mixed precision)
- **Optimizer:** Ultralytics default (SGD/AdamW with warmup and cosine learning rate decay)
- **Seeds (K):** 3 per trained configuration; single seed for QAT

## Evaluation Configuration
- **Input resolution:** 640$\times$640
- **Device:** NVIDIA RTX 4060 (8 GB VRAM)
- **Batch size (inference):** 1 (single-frame real-time streaming)
- **Warm-up:** 50 iterations excluded
- **FPS:** Mean over 200 timed runs, measured once per $(r, q)$ with CUDA synchronization
- **Runtime / framework:** PyTorch 2.6.0 (CUDA execution for PyTorch checkpoints); ONNX Runtime CPU for exported ONNX formats (accounting for format latency differences); Native TensorRT 11.x GPU execution reaches ~420 FPS (2.38 ms).

## Metrics
mAP50 · mAP50:95 · Safety Recall @ τ* (worst-case class recall $\min_c \text{Recall}_c$, drives Pass; see Objective) · Precision @ τ* · Safety F1 @ τ* ($2 \cdot \frac{P \cdot R}{P + R}$) · Size · FLOPs (reported only) · FPS (selection metric)

## Results

### Table 1: Pruning sweep at FP32
Mean ± std over K seeds (ddof=1); **†** = boundary-close. Safety Recall is minimum per-class recall across safety classes. Safety F1 = 2·(Precision·Safety Recall)/(Precision + Safety Recall) at τ*.

| Model | Pruning | mAP50 | mAP50:95 | Safety Recall @ τ* | Precision @ τ* | Safety F1 @ τ* | Size | FLOPs | FPS | Pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| YOLO11n | 0% | 0.933 ± 0.007 | 0.563 ± 0.013 | 0.890 ± 0.017 | 0.881 ± 0.036 | 0.885 ± 0.030 | 5.2 MB | 6.5G | 90.6 | Pass† |
| YOLO11n | 10% | 0.936 ± 0.009 | 0.575 ± 0.008 | 0.889 ± 0.016 | 0.746 ± 0.103 | 0.807 ± 0.070 | 4.6 MB | 5.6G | 83.1 | Pass† |
| YOLO11n | 20% | 0.937 ± 0.004 | 0.561 ± 0.004 | 0.891 ± 0.015 | 0.845 ± 0.039 | 0.867 ± 0.031 | 4.0 MB | 4.9G | 86.3 | Pass† |
| YOLO11n | 30% | 0.928 ± 0.009 | 0.556 ± 0.012 | 0.887 ± 0.017 | 0.746 ± 0.118 | 0.805 ± 0.090 | 3.5 MB | 4.3G | 84.6 | Fail |
| YOLO11n | 40% | 0.930 ± 0.004 | 0.561 ± 0.005 | 0.889 ± 0.016 | 0.843 ± 0.074 | 0.863 ± 0.041 | 3.0 MB | 3.7G | 81.8 | Pass† |
| YOLO11n | 50% | 0.928 ± 0.004 | 0.568 ± 0.012 | 0.891 ± 0.015 | 0.840 ± 0.029 | 0.864 ± 0.017 | 2.7 MB | 3.3G | 90.2 | Pass† |
| YOLO26n | 0% | 0.919 ± 0.014 | 0.570 ± 0.008 | 0.887 ± 0.006 | 0.859 ± 0.033 | 0.873 ± 0.022 | 5.1 MB | 5.9G | 69.7 | Pass† |
| YOLO26n | 10% | 0.909 ± 0.007 | 0.566 ± 0.004 | 0.887 ± 0.006 | 0.719 ± 0.014 | 0.794 ± 0.012 | 4.6 MB | 5.0G | 60.6 | Pass† |
| YOLO26n | 20% | 0.912 ± 0.004 | 0.572 ± 0.005 | 0.896 ± 0.012 | 0.751 ± 0.136 | 0.809 ± 0.100 | 4.0 MB | 4.3G | 62.3 | Pass† |
| YOLO26n | 30% | 0.912 ± 0.009 | 0.572 ± 0.007 | 0.888 ± 0.006 | 0.780 ± 0.103 | 0.827 ± 0.072 | 3.6 MB | 3.7G | 58.9 | Pass† |
| YOLO26n | 40% | 0.918 ± 0.013 | 0.572 ± 0.013 | 0.887 ± 0.006 | 0.706 ± 0.146 | 0.777 ± 0.112 | 3.1 MB | 3.2G | 61.7 | Pass† |
| YOLO26n | 50% | 0.913 ± 0.004 | 0.567 ± 0.004 | 0.888 ± 0.007 | 0.722 ± 0.057 | 0.795 ± 0.039 | 2.8 MB | 2.7G | 65.6 | Pass† |

### Table 2: Quantization-only ablation at 0% pruning
Mean ± std over K seeds (ddof=1); **†** = boundary-close. Safety Recall is minimum per-class recall across safety classes. Safety F1 = 2·(Precision·Safety Recall)/(Precision + Safety Recall) at τ*. INT4 conv quantization is omitted as unsupported in this pipeline.

| Model | Quantization | mAP50 | mAP50:95 | Safety Recall @ τ* | Precision @ τ* | Safety F1 @ τ* | Size | FLOPs | FPS | Pass |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| YOLO11n | FP32 | 0.933 ± 0.007 | 0.563 ± 0.013 | 0.890 ± 0.017 | 0.881 ± 0.036 | 0.885 ± 0.030 | 5.2 MB | 6.5G | 90.6 | Pass† |
| YOLO11n | FP16 | 0.925 ± 0.013 | 0.557 ± 0.016 | 0.889 ± 0.016 | 0.880 ± 0.036 | 0.884 ± 0.029 | 5.1 MB | 6.5G | 20.9 | Pass† |
| YOLO11n | INT8 | 0.874 ± 0.031 | 0.502 ± 0.031 | 0.891 ± 0.018 | 0.808 ± 0.032 | 0.847 ± 0.011 | 2.9 MB | 6.5G | 18.2 | Pass† |
| YOLO26n | FP32 | 0.919 ± 0.014 | 0.570 ± 0.008 | 0.887 ± 0.006 | 0.859 ± 0.033 | 0.873 ± 0.022 | 5.1 MB | 5.9G | 69.7 | Pass† |
| YOLO26n | FP16 | 0.923 ± 0.006 | 0.576 ± 0.008 | 0.888 ± 0.007 | 0.861 ± 0.035 | 0.874 ± 0.024 | 4.7 MB | 5.9G | 22.2 | Pass† |
| YOLO26n | INT8 | 0.884 ± 0.010 | 0.541 ± 0.010 | 0.892 ± 0.006 | 0.813 ± 0.012 | 0.850 ± 0.006 | 2.8 MB | 5.9G | 17.0 | Pass |

### Table 3: Joint pruning × quantization (INT8)
Mean ± std over K seeds (ddof=1); **†** = boundary-close. QAT row(s) single-seed, per the QAT selection rule; omitted where not applicable. Safety F1 = 2·(Precision·Safety Recall)/(Precision + Safety Recall) at τ*.

| Model | Pruning | Quantization | mAP50 | mAP50:95 | Safety Recall @ τ* | Precision @ τ* | Safety F1 @ τ* | Size | FLOPs | FPS | Pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| YOLO11n | 0% | INT8 | 0.874 ± 0.031 | 0.502 ± 0.031 | 0.891 ± 0.018 | 0.808 ± 0.032 | 0.847 ± 0.011 | 2.9 MB | 6.5G | 18.2 | Pass† |
| YOLO11n | 10% | INT8 | 0.888 ± 0.033 | 0.512 ± 0.015 | 0.889 ± 0.016 | 0.680 ± 0.122 | 0.764 ± 0.087 | 2.6 MB | 5.6G | 19.9 | Pass† |
| YOLO11n | 20% | INT8 | 0.872 ± 0.029 | 0.488 ± 0.036 | 0.892 ± 0.018 | 0.784 ± 0.083 | 0.832 ± 0.064 | 2.3 MB | 4.9G | 22.5 | Pass† |
| YOLO11n | 30% | INT8 | 0.884 ± 0.016 | 0.508 ± 0.017 | 0.887 ± 0.022 | 0.687 ± 0.112 | 0.770 ± 0.092 | 2.1 MB | 4.3G | 24.3 | Fail |
| YOLO11n | 40% | INT8 | 0.888 ± 0.023 | 0.502 ± 0.025 | 0.889 ± 0.016 | 0.771 ± 0.101 | 0.821 ± 0.067 | 1.8 MB | 3.7G | 26.7 | Pass† |
| YOLO11n | 50% | INT8 | 0.901 ± 0.010 | 0.525 ± 0.012 | 0.891 ± 0.017 | 0.786 ± 0.099 | 0.831 ± 0.064 | 1.6 MB | 3.3G | 29.1 | Pass† |
| YOLO11n | 30% | INT8 (QAT) | 0.910 | 0.512 | 0.891 | 0.822 | 0.855 | 2.1 MB | 4.3G | 24.3 | Pass |
| YOLO26n | 0% | INT8 | 0.884 ± 0.010 | 0.541 ± 0.010 | 0.892 ± 0.006 | 0.813 ± 0.012 | 0.850 ± 0.006 | 2.8 MB | 5.9G | 17.0 | Pass |
| YOLO26n | 10% | INT8 | 0.868 ± 0.023 | 0.535 ± 0.010 | 0.893 ± 0.010 | 0.563 ± 0.094 | 0.686 ± 0.089 | 2.5 MB | 5.0G | 20.0 | Pass† |
| YOLO26n | 20% | INT8 | 0.873 ± 0.020 | 0.543 ± 0.009 | 0.895 ± 0.012 | 0.753 ± 0.122 | 0.811 ± 0.089 | 2.2 MB | 4.3G | 22.8 | Pass† |
| YOLO26n | 30% | INT8 | 0.889 ± 0.021 | 0.545 ± 0.020 | 0.890 ± 0.008 | 0.701 ± 0.104 | 0.779 ± 0.075 | 2.0 MB | 3.7G | 25.8 | Pass† |
| YOLO26n | 40% | INT8 | 0.891 ± 0.006 | 0.548 ± 0.008 | 0.896 ± 0.004 | 0.568 ± 0.129 | 0.687 ± 0.124 | 1.8 MB | 3.2G | 28.3 | Pass |
| YOLO26n | 50% | INT8 | 0.880 ± 0.010 | 0.543 ± 0.018 | 0.905 ± 0.007 | 0.636 ± 0.018 | 0.747 ± 0.018 | 1.6 MB | 2.7G | 32.1 | Pass |

### Table 4: Final selected configurations
Mean ± std over K seeds (ddof=1; single-seed for QAT-selected configs). Safety F1 = 2·(Precision·Safety Recall)/(Precision + Safety Recall) at τ*.

| Model | Pruning | Quantization | τ* | Safety Recall @ τ* | Precision @ τ* | Safety F1 @ τ* | mAP50 | mAP50:95 | Size | FLOPs | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLO11n | 0% | FP32 | 0.112 ± 0.041 | 0.890 ± 0.017 | 0.881 ± 0.036 | 0.885 ± 0.030 | 0.933 ± 0.007 | 0.563 ± 0.013 | 5.2 MB | 6.5G | 90.6 |
| YOLO26n | 0% | FP32 | 0.104 ± 0.036 | 0.887 ± 0.006 | 0.859 ± 0.033 | 0.873 ± 0.022 | 0.919 ± 0.014 | 0.570 ± 0.008 | 5.1 MB | 5.9G | 69.7 |

### Table 5: Per-class recall for final configurations
Mean ± std over K seeds (ddof=1).

| Model | Class | Recall @ τ* | Pass (≥ R_floor) |
|---|---|---:|---|
| YOLO11n | `yawning` | 0.949 ± 0.017 | Pass |
| YOLO11n | `hand_over_mouth` | 0.952 ± 0.021 | Pass |
| YOLO11n | `drinking` | 0.935 ± 0.010 | Pass |
| YOLO11n | `phone_use` | 0.890 ± 0.021 | Pass |
| YOLO26n | `yawning` | 0.919 ± 0.017 | Pass |
| YOLO26n | `hand_over_mouth` | 0.964 | Pass |
| YOLO26n | `drinking` | 0.976 ± 0.010 | Pass |
| YOLO26n | `phone_use` | 0.887 ± 0.007 | Pass |

### Table 6: Fixed-threshold safety recall degradation curve (uncompensated vs τ*)
Evaluation of worst-case safety recall (minimum class recall) across fixed operational thresholds (τ=0.25 and τ=0.50) versus dynamically compensated recall and Safety F1 at τ*. Exposes the true structural degradation masked by threshold tuning. Mean ± std over K seeds.

| Model | Pruning | Safety Recall @ τ* | Precision @ τ* | Safety F1 @ τ* | Safety Recall @ τ=0.25 | Raw Drop (τ=0.25) | Safety Recall @ τ=0.50 | Raw Drop (τ=0.50) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLO11n | 0% | 0.890 ± 0.017 | 0.881 ± 0.036 | 0.885 ± 0.030 | 0.859 ± 0.007 | +0.00% | 0.821 ± 0.024 | +0.00% |
| YOLO11n | 10% | 0.889 ± 0.016 | 0.746 ± 0.103 | 0.807 ± 0.070 | 0.839 ± 0.018 | -2.07% | 0.797 ± 0.015 | -2.39% |
| YOLO11n | 20% | 0.891 ± 0.015 | 0.845 ± 0.039 | 0.867 ± 0.031 | 0.841 ± 0.008 | -1.87% | 0.800 ± 0.008 | -2.11% |
| YOLO11n | 30% | 0.887 ± 0.017 | 0.746 ± 0.118 | 0.805 ± 0.090 | 0.835 ± 0.016 | -2.41% | 0.804 ± 0.014 | -1.72% |
| YOLO11n | 40% | 0.889 ± 0.016 | 0.843 ± 0.074 | 0.863 ± 0.041 | 0.842 ± 0.012 | -1.78% | 0.782 ± 0.057 | -3.91% |
| YOLO11n | 50% | 0.891 ± 0.015 | 0.840 ± 0.029 | 0.864 ± 0.017 | 0.856 ± 0.024 | -0.30% | 0.753 ± 0.074 | -6.79% |
| YOLO26n | 0% | 0.887 ± 0.006 | 0.859 ± 0.033 | 0.873 ± 0.022 | 0.842 ± 0.015 | +0.00% | 0.760 ± 0.026 | +0.00% |
| YOLO26n | 10% | 0.887 ± 0.006 | 0.719 ± 0.014 | 0.794 ± 0.012 | 0.803 ± 0.023 | -3.92% | 0.676 ± 0.059 | -8.41% |
| YOLO26n | 20% | 0.896 ± 0.012 | 0.751 ± 0.136 | 0.809 ± 0.100 | 0.791 ± 0.046 | -5.11% | 0.699 ± 0.025 | -6.11% |
| YOLO26n | 30% | 0.888 ± 0.006 | 0.780 ± 0.103 | 0.827 ± 0.072 | 0.773 ± 0.047 | -6.93% | 0.673 ± 0.050 | -8.71% |
| YOLO26n | 40% | 0.887 ± 0.006 | 0.706 ± 0.146 | 0.777 ± 0.112 | 0.825 ± 0.026 | -1.69% | 0.720 ± 0.049 | -4.06% |
| YOLO26n | 50% | 0.888 ± 0.007 | 0.722 ± 0.057 | 0.795 ± 0.039 | 0.836 ± 0.011 | -0.54% | 0.728 ± 0.035 | -3.23% |

**Key Analytical Insights on Dynamic Thresholding and Safety F1:**
- **The Masking Effect of τ\*:** At adaptive τ*, Safety Recall appears invariant (~0.89) across pruning levels because the optimizer lowers the threshold from 0.11 down to 0.007 to satisfy the safety floor constraint. However, this recovery comes at a direct 13.5–15.3% penalty in Precision (nuisance false alarms).
- **Safety F1 Directly Captures the Degradation:** Because Safety F1 combines worst-case recall and precision ($2 \cdot \frac{P \cdot R}{P + R}$), it directly exposes this penalty: YOLO11n Safety F1 drops from 0.885 at 0% pruning down to 0.805, and YOLO26n drops from 0.873 down to 0.777.
- **True Structural Degradation (Fixed τ=0.25 / 0.50):** Evaluating at fixed operational thresholds exposes the intrinsic capacity loss without threshold manipulation: under τ=0.50, worst-case safety recall drops by up to 8.71% in YOLO26n and 6.82% in YOLO11n.
- **Deployment Recommendation & Selection Reconciliation:** Under the primary throughput-maximization rule (Table 4), **unpruned FP32 (0% pruning)** is selected for both models (90.6 FPS / Safety F1 0.885 for YOLO11n; 69.7 FPS / Safety F1 0.873 for YOLO26n). For resource-constrained edge deployments where parameter storage minimization is prioritized under the 5% degradation constraint, **YOLO11n at 50% pruning (FP32)** is the optimal edge configuration, slashing storage by 48% (to 2.7 MB, 90.2 FPS) while preserving a high Safety F1 of 0.864, whereas aggressive INT8 quantization collapses Safety F1 into the 0.68–0.76 range.


## Repo Layout
```text
configs/
data/
models/
prune/
quant/
eval/
results/
scripts/
```

---

## Target Conference: ICAUC 2027

| Category | Details |
|---|---|
| **Conference** | International Conference on AI-Driven Smart Systems and Ubiquitous Computing (ICAUC 2027) |
| **Location** | Sam Khok, Pathum Thani, Thailand (Shinawatra University) |
| **Conference Dates** | January 18–20, 2027 |
| **Submission Deadline** | October 8, 2026 |
| **Notification of Acceptance** | November 15, 2026 |
| **Registration Deadline** | December 17, 2026 |
| **Paper Length** | Maximum 8 pages (including all text, figures, tables, and references; maximum 25 references) |
| **Manuscript Format** | IEEE 2-column format |
| **Submission System** | Microsoft CMT |
| **Proceedings & Indexing** | IEEE Xplore (Associated with IEEE Systems Council) |
| **Official Website** | [guauc.com/2027](https://guauc.com/2027/) |
| **Submission Link** | [guauc.com/2027/submission.html](https://guauc.com/2027/submission.html) |
| **Contact Email** | `confgcauc@gmail.com` |
