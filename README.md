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
| YOLO11n | 0% | | | | | | | | | |
| YOLO11n | R% | | | | | | | | | |
| YOLO11n | 2R% | | | | | | | | | |
| YOLO11n | 3R% | | | | | | | | | |
| YOLO11n | 4R% | | | | | | | | | |
| YOLO11n | 5R% | | | | | | | | | |
| YOLO26n | 0% | | | | | | | | | |
| YOLO26n | R% | | | | | | | | | |
| YOLO26n | 2R% | | | | | | | | | |
| YOLO26n | 3R% | | | | | | | | | |
| YOLO26n | 4R% | | | | | | | | | |
| YOLO26n | 5R% | | | | | | | | | |

### Table 2: Quantization-only ablation at 0% pruning
Mean ± std over K seeds; **†** = boundary-close.

| Model | Quantization | mAP50 | mAP50:95 | Recall @ τ* | Precision @ τ* | Min safety-class recall @ τ* | Size | FLOPs | FPS | Pass |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| YOLO11n | FP32 | | | | | | | | | |
| YOLO11n | FP16 | | | | | | | | | |
| YOLO11n | INT8 | | | | | | | | | |
| YOLO11n | INT4 | | | | | | | | | |
| YOLO26n | FP32 | | | | | | | | | |
| YOLO26n | FP16 | | | | | | | | | |
| YOLO26n | INT8 | | | | | | | | | |
| YOLO26n | INT4 | | | | | | | | | |

### Table 3: Joint pruning × quantization (INT8)
Mean ± std over K seeds; **†** = boundary-close. QAT row(s) single-seed, per the QAT selection rule; omitted where not applicable.

| Model | Pruning | Quantization | mAP50 | mAP50:95 | Recall @ τ* | Precision @ τ* | Min safety-class recall @ τ* | Size | FLOPs | FPS | Pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| YOLO11n | 0% | INT8 | | | | | | | | | |
| YOLO11n | R% | INT8 | | | | | | | | | |
| YOLO11n | 2R% | INT8 | | | | | | | | | |
| YOLO11n | 3R% | INT8 | | | | | | | | | |
| YOLO11n | 4R% | INT8 | | | | | | | | | |
| YOLO11n | 5R% | INT8 | | | | | | | | | |
| YOLO11n | _QAT-selected (if applicable)_ | INT8 (QAT) | | | | | | | | | |
| YOLO26n | 0% | INT8 | | | | | | | | | |
| YOLO26n | R% | INT8 | | | | | | | | | |
| YOLO26n | 2R% | INT8 | | | | | | | | | |
| YOLO26n | 3R% | INT8 | | | | | | | | | |
| YOLO26n | 4R% | INT8 | | | | | | | | | |
| YOLO26n | 5R% | INT8 | | | | | | | | | |
| YOLO26n | _QAT-selected (if applicable)_ | INT8 (QAT) | | | | | | | | | |

### Table 4: Final selected configurations
Mean ± std over K seeds (single-seed for QAT-selected configs).

| Model | Pruning | Quantization | τ* | Macro Recall @ τ* | Min safety-class recall @ τ* | Precision @ τ* | mAP50 | mAP50:95 | Size | FLOPs | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLO11n | | | | | | | | | | | |
| YOLO26n | | | | | | | | | | | |

### Table 5: Per-class recall for final configurations
Mean ± std over K seeds.

| Model | Class | Recall @ τ* | Pass (≥ R_floor) |
|---|---|---:|---|
| YOLO11n | | | |
| YOLO26n | | | |

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