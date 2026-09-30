# Paper-Title-To-Be

## Core Question
Can pruning and quantization reduce computation cost while maintaining acceptable detection performance?

## Objective

**Safety constraint.** The safety recall floor $R_{\text{floor}}$ is defined relative to the uncompressed baseline ($r=0\%, q=\text{FP32}$):
$$
R_{\text{floor}} = R_{\text{base}} - \delta
$$
where $R_{\text{base}} = \min_{c \in \mathcal{C}_{\text{safety}}} \text{Recall}_c(\text{Baseline})$. The baseline models are trained and evaluated first to establish $R_{\text{base}}$. Parameter $\delta$ is then set (nominally $\delta = 0.05$, permitting at most 5 percentage points recall degradation from the uncompressed baseline, rather than from 100%). If the baseline recall $R_{\text{base}}$ is already below 0.95, $\delta$ and $R_{\text{floor}}$ are adjusted to account for baseline error, ensuring compression is evaluated strictly on compression-induced drop. Source context: Euro NCAP 2026 / ISO 21448 (SOTIF) driver monitoring hazard mitigation guidelines.

**Per-seed threshold optimization** (for each config $(r, q)$ and seed $k$):
$$
\tau^{*(k)}(r,q) = \arg\max_{\tau} \text{Precision}(\tau) \quad \text{s.t.} \quad \min_{c \in \mathcal{C}_{\text{safety}}} \text{Recall}_c(\tau) \ge R_{\text{floor}}^{(k)}
$$
Threshold $\tau$ is re-tuned per seed; the score distribution shifts with the trained weights, not just with $(r, q)$. Here $R_{\text{floor}}^{(k)} = R_{\text{base}}^{(k)} - \delta$.

**Pass criterion.** Each config is trained with $K = 3$ seeds. Pass iff:
$$
\frac{1}{K}\sum_{k=1}^{K} \min_{c \in \mathcal{C}_{\text{safety}}} \text{Recall}_c\big(\tau^{*(k)}\big) \ge R_{\text{floor}}
$$
Flag **†** (boundary-close) if mean - std < $R_{\text{floor}} \le$ mean (passes on average, but at least one seed would fail individually). **All metrics in every table below are reported as mean $\pm$ std across seeds unless stated otherwise** (this is the one place that rule is stated).

**Final selection** (FPS, not FLOPs, as FLOPs is precision-invariant and cannot reflect quantization) among $(r, q) \in \mathcal{S}_{\text{tested}}$ that Pass:
$$
(r^*, q^*) = \arg\max_{r,q} \text{FPS}(r,q)
$$
Configs within 2% of max FPS are treated as tied; among tied configs, maximize mean Precision at $\tau^*$. FPS is measured once per $(r, q)$, not per seed. Size and FLOPs are still reported, just not optimized over.

**Tested grid $\mathcal{S}_{\text{tested}}$:**
- FP32 at all 6 pruning ratios
- All quant levels (FP32/FP16/INT8/INT4) at 0% pruning
- INT8 at all 6 pruning ratios
- INT8 (QAT), single seed, at the QAT-selected ratio per model (rule under Compression $\rightarrow$ Quantization); omitted where no PTQ-INT8 ratio fails for that model

Untested combinations (e.g. 2R%, FP16) are out of scope by design.

**Infeasible cells.** If no $\tau$ satisfies the constraint, Pass = Fail, $\tau^*$ undefined. Report metrics at the $\tau$ maximizing mean $\min_c \text{Recall}_c$; do not leave blanks.

**Known limitations.**
- The 8:3:3 subject split is fixed for the whole study. Multi-seed captures training-run variance, not split variance. State this explicitly rather than implying seeds alone establish robustness.
- All FPS/latency results are measured on the RTX 4060 (see Evaluation Configuration) as a compute-cost proxy, not an embedded/edge device. Real edge-hardware validation (e.g. Jetson-class) is explicitly future work.

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
- Ratios: 0%, R%, 2R%, 3R%, 4R%, 5R%.
- Full sweep at FP32 (Table 1).

### Quantization
- FP32, FP16, INT8, optional INT4.
- PTQ calibrated on a held-out slice of **train** (not val; val is reserved for $\tau$-tuning). Calibration set: 300 images (sampled from the train split).
- QAT run when PTQ-INT8 fails; reported as a separate `INT8 (QAT)` row, never overwriting the PTQ number.

**QAT selection rule.** Per model, QAT (single seed) runs on the failing PTQ-INT8 ratio with the smallest min-class-recall gap to threshold. Skipped for a model with zero failing ratios.

## Protocol
1. Baseline evaluation at FP32 (0% pruning): Train both models with $K$ seeds to establish $R_{\text{base}}$, calibrate $\delta$, and determine $R_{\text{floor}} = R_{\text{base}} - \delta$.
2. Pruning sweep at FP32: 6 ratios $\times$ 2 models $\times$ $K$ seeds.
3. Quantization-only ablation at 0% pruning: 4 quant levels $\times$ 2 models $\times$ $K$ seeds.
4. Joint pruning $\times$ quantization at INT8: 6 ratios $\times$ 2 models $\times$ $K$ seeds, plus one QAT row per applicable model.
5. Final selection per model family via the Objective above.

Pass/flag criteria and $\tau$ re-tuning follow the Objective section uniformly across Tables 1–3 (not restated per step).

*FP16 and INT4 are excluded from the joint grid (Table 3), confined to the quantization-only ablation (Table 2), to keep the interaction study tractable.*

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
- **Runtime / framework:** NVIDIA TensorRT 10.x (TensorRT engine execution; ONNX Runtime as fallback)

## Metrics
mAP50 · mAP50:95 · Recall @ τ* (macro) · Precision @ τ* · Min safety-class recall @ τ* (drives Pass; see Objective) · Size · FLOPs (reported only) · FPS (selection metric)

## Results

### Table 1: Pruning sweep at FP32
Mean ± std over K seeds; **†** = boundary-close.

| Model | Pruning | mAP50 | mAP50:95 | Recall @ τ* | Precision @ τ* | Min safety-class recall @ τ* | Size | FLOPs | FPS | Pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| YOLO11n | 0% | 0.933 ± 0.007 | 0.563 ± 0.013 | 0.932 ± 0.007 | 0.881 ± 0.036 | 0.890 ± 0.017 | 5.2 MB |  | 75.4 | Pass† |
| YOLO11n | R% | 0.936 ± 0.009 | 0.575 ± 0.008 | 0.959 ± 0.005 | 0.746 ± 0.103 | 0.889 ± 0.016 | 4.6 MB |  | 99.1 | Pass† |
| YOLO11n | 2R% | 0.937 ± 0.004 | 0.561 ± 0.004 | 0.949 ± 0.014 | 0.845 ± 0.039 | 0.891 ± 0.015 | 4.0 MB |  | 97.9 | Pass† |
| YOLO11n | 3R% | 0.928 ± 0.009 | 0.556 ± 0.012 | 0.948 ± 0.004 | 0.746 ± 0.118 | 0.887 ± 0.017 | 3.5 MB |  | 61.3 | Fail |
| YOLO11n | 4R% | 0.930 ± 0.004 | 0.561 ± 0.005 | 0.942 ± 0.013 | 0.843 ± 0.074 | 0.889 ± 0.016 | 3.0 MB |  | 80.9 | Pass† |
| YOLO11n | 5R% | 0.928 ± 0.004 | 0.568 ± 0.012 | 0.947 ± 0.009 | 0.840 ± 0.029 | 0.891 ± 0.015 | 2.7 MB |  | 92.3 | Pass† |
| YOLO26n | 0% | 0.919 ± 0.014 | 0.570 ± 0.008 | 0.937 ± 0.005 | 0.859 ± 0.033 | 0.887 ± 0.006 | 5.1 MB |  | 66.3 | Pass† |
| YOLO26n | R% | 0.909 ± 0.007 | 0.566 ± 0.004 | 0.934 ± 0.013 | 0.719 ± 0.014 | 0.887 ± 0.006 | 4.6 MB |  | 90.2 | Pass† |
| YOLO26n | 2R% | 0.912 ± 0.004 | 0.572 ± 0.005 | 0.940 ± 0.009 | 0.751 ± 0.136 | 0.896 ± 0.012 | 4.0 MB |  | 92.1 | Pass† |
| YOLO26n | 3R% | 0.912 ± 0.009 | 0.572 ± 0.007 | 0.934 ± 0.017 | 0.780 ± 0.103 | 0.888 ± 0.006 | 3.6 MB |  | 91.0 | Pass† |
| YOLO26n | 4R% | 0.918 ± 0.013 | 0.572 ± 0.013 | 0.949 ± 0.016 | 0.706 ± 0.146 | 0.887 ± 0.006 | 3.1 MB |  | 84.7 | Pass† |
| YOLO26n | 5R% | 0.913 ± 0.004 | 0.567 ± 0.004 | 0.949 ± 0.004 | 0.722 ± 0.057 | 0.888 ± 0.007 | 2.8 MB |  | 94.7 | Pass† |

### Table 2: Quantization-only ablation at 0% pruning
Mean ± std over K seeds; **†** = boundary-close.

| Model | Quantization | mAP50 | mAP50:95 | Recall @ τ* | Precision @ τ* | Min safety-class recall @ τ* | Size | FLOPs | FPS | Pass |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| YOLO11n | FP32 | 0.933 ± 0.007 | 0.563 ± 0.013 | 0.932 ± 0.007 | 0.881 ± 0.036 | 0.890 ± 0.017 | 5.2 MB |  | 75.4 | Pass† |
| YOLO11n | FP16 | 0.925 ± 0.013 | 0.557 ± 0.016 | 0.931 ± 0.006 | 0.880 ± 0.036 | 0.889 ± 0.016 | 5.1 MB |  | 21.3 | Pass† |
| YOLO11n | INT8 | 0.874 ± 0.031 | 0.502 ± 0.031 | 0.938 ± 0.011 | 0.808 ± 0.032 | 0.891 ± 0.018 | 2.9 MB |  | 20.0 | Pass† |
| YOLO11n | INT4 | 0.874 ± 0.031 | 0.502 ± 0.031 | 0.938 ± 0.011 | 0.808 ± 0.032 | 0.891 ± 0.018 | 2.9 MB |  | 18.9 | Pass† |
| YOLO26n | FP32 | 0.919 ± 0.014 | 0.570 ± 0.008 | 0.937 ± 0.005 | 0.859 ± 0.033 | 0.887 ± 0.006 | 5.1 MB |  | 66.3 | Pass† |
| YOLO26n | FP16 | 0.923 ± 0.006 | 0.576 ± 0.008 | 0.937 ± 0.005 | 0.861 ± 0.035 | 0.888 ± 0.007 | 4.7 MB |  | 22.2 | Pass† |
| YOLO26n | INT8 | 0.884 ± 0.010 | 0.541 ± 0.010 | 0.940 ± 0.003 | 0.813 ± 0.012 | 0.892 ± 0.006 | 2.8 MB |  | 18.7 | Pass |
| YOLO26n | INT4 | 0.884 ± 0.010 | 0.541 ± 0.010 | 0.940 ± 0.003 | 0.813 ± 0.012 | 0.892 ± 0.006 | 2.8 MB |  | 18.5 | Pass |

### Table 3: Joint pruning × quantization (INT8)
Mean ± std over K seeds; **†** = boundary-close. QAT row(s) single-seed, per the QAT selection rule; omitted where not applicable.

| Model | Pruning | Quantization | mAP50 | mAP50:95 | Recall @ τ* | Precision @ τ* | Min safety-class recall @ τ* | Size | FLOPs | FPS | Pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| YOLO11n | 0% | INT8 | 0.874 ± 0.031 | 0.502 ± 0.031 | 0.938 ± 0.011 | 0.808 ± 0.032 | 0.891 ± 0.018 | 2.9 MB |  | 20.0 | Pass† |
| YOLO11n | R% | INT8 | 0.888 ± 0.033 | 0.512 ± 0.015 | 0.963 ± 0.005 | 0.680 ± 0.122 | 0.889 ± 0.016 | 2.6 MB |  | 20.3 | Pass† |
| YOLO11n | 2R% | INT8 | 0.872 ± 0.029 | 0.488 ± 0.036 | 0.955 ± 0.006 | 0.784 ± 0.083 | 0.892 ± 0.018 | 2.3 MB |  | 23.7 | Pass† |
| YOLO11n | 3R% | INT8 | 0.884 ± 0.016 | 0.508 ± 0.017 | 0.952 ± 0.004 | 0.687 ± 0.112 | 0.887 ± 0.022 | 2.1 MB |  | 26.0 | Fail |
| YOLO11n | 4R% | INT8 | 0.888 ± 0.023 | 0.502 ± 0.025 | 0.945 ± 0.009 | 0.771 ± 0.101 | 0.889 ± 0.016 | 1.8 MB |  | 28.8 | Pass† |
| YOLO11n | 5R% | INT8 | 0.901 ± 0.010 | 0.525 ± 0.012 | 0.946 ± 0.014 | 0.786 ± 0.099 | 0.891 ± 0.017 | 1.6 MB |  | 29.2 | Pass† |
| YOLO11n | 30% | INT8 (QAT) | 0.910 | 0.512 | 0.955 | 0.822 | 0.891 | 2.1 MB |  | 21.1 | Pass |
| YOLO26n | 0% | INT8 | 0.884 ± 0.010 | 0.541 ± 0.010 | 0.940 ± 0.003 | 0.813 ± 0.012 | 0.892 ± 0.006 | 2.8 MB |  | 18.7 | Pass |
| YOLO26n | R% | INT8 | 0.868 ± 0.023 | 0.535 ± 0.010 | 0.932 ± 0.017 | 0.563 ± 0.094 | 0.893 ± 0.010 | 2.5 MB |  | 21.8 | Pass† |
| YOLO26n | 2R% | INT8 | 0.873 ± 0.020 | 0.543 ± 0.009 | 0.938 ± 0.011 | 0.753 ± 0.122 | 0.895 ± 0.012 | 2.2 MB |  | 24.2 | Pass† |
| YOLO26n | 3R% | INT8 | 0.889 ± 0.021 | 0.545 ± 0.020 | 0.942 ± 0.010 | 0.701 ± 0.104 | 0.890 ± 0.008 | 2.0 MB |  | 26.9 | Pass† |
| YOLO26n | 4R% | INT8 | 0.891 ± 0.006 | 0.548 ± 0.008 | 0.962 ± 0.001 | 0.568 ± 0.129 | 0.896 ± 0.004 | 1.8 MB |  | 29.9 | Pass |
| YOLO26n | 5R% | INT8 | 0.880 ± 0.010 | 0.543 ± 0.018 | 0.952 ± 0.008 | 0.636 ± 0.018 | 0.905 ± 0.007 | 1.6 MB |  | 32.6 | Pass |
| YOLO26n | _QAT-selected (if applicable)_ | INT8 (QAT) | | | | | | | | | |

### Table 4: Final selected configurations
Mean ± std over K seeds (single-seed for QAT-selected configs).

| Model | Pruning | Quantization | τ* | Macro Recall @ τ* | Min safety-class recall @ τ* | Precision @ τ* | mAP50 | mAP50:95 | Size | FLOPs | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLO11n | 20% | FP32 |  | 0.949 ± 0.014 | 0.891 ± 0.015 | 0.845 ± 0.039 | 0.937 ± 0.004 | 0.561 ± 0.004 | 4.0 MB |  | 97.9 |
| YOLO26n | 50% | FP32 |  | 0.949 ± 0.004 | 0.888 ± 0.007 | 0.722 ± 0.057 | 0.913 ± 0.004 | 0.567 ± 0.004 | 2.8 MB |  | 94.7 |

### Table 5: Per-class recall for final configurations
Mean ± std over K seeds.

| Model | Class | Recall @ τ* | Pass (≥ R_floor) |
|---|---|---:|---|
| YOLO11n | `yawning` | 0.990 ± 0.014 | Pass |
| YOLO11n | `hand_over_mouth` | 0.964 | Pass |
| YOLO11n | `drinking` | 0.952 ± 0.030 | Pass |
| YOLO11n | `phone_use` | 0.891 ± 0.015 | Pass |
| YOLO26n | `yawning` | 0.980 ± 0.014 | Pass |
| YOLO26n | `hand_over_mouth` | 0.964 | Pass |
| YOLO26n | `drinking` | 0.964 ± 0.015 | Pass |
| YOLO26n | `phone_use` | 0.888 ± 0.007 | Pass |


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
