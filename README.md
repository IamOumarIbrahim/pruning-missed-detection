# Pruning-Induced Missed Detections in Edge-Deployed DMS
`pruning-missed-detection` — methodology spec & experiment templates

*A living methodology spec, not a paper draft — sections below are either fully drafted (Methodology, Experiment templates, Limitations) or still outline-only (Introduction, Related Work, Discussion, Conclusion), and the Outline marks which is which. Updated to fold in reviewer feedback: dataset task definition, the confidence-threshold procedure, the F-β formula, a hardware/benchmarking protocol, and a new Limitations section. Bracketed items — `[X]`, `[TBD]`, etc. — are open decisions, not answers; resolve every item on the [Open Decisions Checklist](#open-decisions-checklist) before Table 1's first cell gets filled in.*

## Contents

- [Core Question](#core-question)
- [Outline](#outline)
- [3. Methodology](#3-methodology)
  - [Dataset](#dataset)
  - [Definition of SAFE threshold](#definition-of-safe-threshold)
  - [Weighted Metric (F-β)](#weighted-metric-f-β)
  - [Hardware & Benchmarking Protocol](#hardware--benchmarking-protocol)
  - [Confidence Threshold / Operating-Point Selection](#confidence-threshold--operating-point-selection)
- [4. Experiments](#4-experiments)
  - [Table 1: Pruning Sweep](#table-1-pruning-sweep)
  - [Table 2: Quantization Comparison](#table-2-quantization-comparison)
  - [Table 3: Knowledge Distillation](#table-3-knowledge-distillation)
  - [Table 4: Proposed Method](#table-4-proposed-method)
- [5. Discussion — Key Findings & Narrative](#5-discussion--key-findings--narrative)
- [6. Limitations & Threats to Validity](#6-limitations--threats-to-validity)
- [Open Decisions Checklist](#open-decisions-checklist)
- [References](#references)

---

## Core Question

> **The Core Question:** "How much can a real-time detector be pruned before missed detections on safety-critical classes cross an acceptable risk threshold, and once knowledge distillation and INT8 quantization are already applied, does pruning still buy meaningful latency or storage headroom for edge-deployed DMS without crossing that threshold?"
>
> *Refined from the original framing, which was ambiguous between "does pruning beat quantization/KD as an alternative" and "does pruning still help once quantization/KD are already applied." Table 4 tests the latter, so the question now matches it.*

---

## Outline

1. **Introduction** *(outline only — not yet drafted).* DMS on edge devices has strict latency and safety constraints.

2. **Related Work** *(outline only — not yet drafted).* Pruning, quantization, and knowledge distillation are typically evaluated separately — most compression papers report aggregate mAP on general benchmarks with no cost model for a missed detection. No existing work jointly evaluates all three under a single false-negative-weighted metric for a safety-critical edge task; that gap is what this project targets. Once real citations are added, structure around three sub-threads: (a) structured vs. unstructured pruning for real-time detectors, (b) detector-specific quantization — DFL is a known PTQ pain point across the YOLOv8–YOLO11 lineage, and YOLO26 removes DFL entirely, (c) KD applied to detection heads specifically, logit-based vs. feature-based.

3. **Methodology** *(drafted — [jump to section](#3-methodology)).* Safety-weighted F-β metric, FN-penalty threshold, dataset, hardware & benchmarking protocol, confidence-threshold selection procedure.

4. **Experiments** *(drafted — [jump to section](#4-experiments); templates only, no data filled in yet)*
   - 4.1 Benchmarking: Tables 1–3. Each technique in isolation.
   - 4.2 Proposed Method: Table 4. How we combine them to build a deployable DMS model.

5. **Discussion** *(skeleton only — [jump to section](#5-discussion--key-findings--narrative)).* Trade-offs, the QAT vs. PTQ anomaly on "n" models, compounding errors across the pipeline, and architecture-dependent compression behavior. YOLO26 isn't just a resized YOLO11 — it removes Distribution Focal Loss (DFL) and NMS in favor of a native one-to-one end-to-end head, and adds STAL (Small-Target-Aware Label Assignment) for small objects. Three implications worth discussing explicitly: (1) DFL's softmax-over-bins regression has historically been tricky to quantize cleanly in the YOLOv8–YOLO11 family, so YOLO26 lacking it could explain a different PTQ/QAT gap than YOLO11n shows — this reframes the "anomaly" as a testable hypothesis rather than an unexplained finding; (2) removing NMS (a variable-cost, hard-to-batch post-processing step) likely changes how latency scales under pruning versus a model that still runs it, so findings shouldn't be assumed to transfer between the two families by default; (3) STAL specifically targets small-object accuracy, and phone-in-hand is small in-frame, so pruning-induced small-object degradation deserves its own line rather than folding into an aggregate Recall number.

6. **Limitations & Threats to Validity** *(drafted — [jump to section](#6-limitations--threats-to-validity)).*

7. **Conclusion** *(outline only — not yet drafted).* The combined pipeline provides the best safety-to-latency ratio for edge DMS.

---

## 3. Methodology

### Dataset

**DMD (Driver Monitoring Dataset)** — Vicomtech / Ortega et al., 2020, funded by the EU Horizon 2020 VI-DAS project.

DMD is primarily a temporal action, fatigue, and gaze/head-pose dataset — 93 classes across temporal, geometric, and context annotations, recorded from synchronized body/face/hand camera views, in OpenLABEL/VCD format — not a ready-made multi-class object-detection benchmark in the COCO sense. Only a small number of objects carry spatial `object_in_scene` annotation suitable for a YOLO-style bounding-box target: **phone, hairbrush/comb, bottle**. Everything else (drowsiness, distraction state, gaze region) is a temporal or state label rather than a box, and would need a deliberate, documented conversion (e.g. a fixed face/hand region-of-interest standing in for a box) before a detector could train on it at all.

As of DMD's 2026 revision, simulator recordings and the Depth/IR streams have been removed from public availability — only RGB material from the real-car scenario, drawn from a restricted subset of the original driver pool, is currently distributable. Any plan assuming IR footage (common in production DMS for night driving) or simulator data no longer holds on this data; this is scoped out explicitly in Limitations & Threats to Validity rather than discovered mid-experiment.

**Decisions to make explicit before Table 1 is touched:**

- **Which classes:** [TBD] — leading candidate is **phone use** as the primary safety-critical class: it has native bounding-box annotation, it's a small in-frame object (which stresses the pruning-robustness question directly), and it's scored as its own category in Euro NCAP's 2026 protocol (see SAFE threshold, below). Still open: whether hairbrush/bottle are in scope as secondary distraction classes, or excluded as not safety-relevant.
- **Split strategy:** [TBD] — must be subject-independent across train/val/test. DMD has a limited driver pool; if the same driver appears in both train and test, recall numbers get inflated by identity leakage rather than genuine generalization.
- **Class imbalance:** [TBD] — document the sampling or loss-weighting strategy used, since it interacts directly with the confidence-threshold procedure below.

### Definition of SAFE threshold

"Anchor to ISO 26262 / Euro NCAP" undersold the standards landscape. Neither hands over a recall percentage directly — the standards that actually deal with perception performance limits and AI-specific risk are ISO 21448 (SOTIF) and ISO/PAS 8800, and both explicitly leave the number-picking to the practitioner:

| Standard | What it actually governs | Hands you a recall number? |
|:---|:---|:---|
| ISO 26262 | Functional safety of electrical/electronic systems; hazards from component malfunction or failure | No |
| ISO 21448 (SOTIF) | Safety of the intended function; hazards from performance limitations even with no fault (e.g. a correctly-working camera that still misses something) | No — but this is the standard actually concerned with perception accuracy gaps |
| ISO/PAS 8800 (Dec. 2024) | AI/ML-specific extension covering training-data gaps, unrepresented edge cases, distribution shift | No — assumes the vehicle-level risk budget has already been broken into a verifiable per-component number via 26262/21448 |
| Euro NCAP 2026 protocol | Rating criteria: direct monitoring required, drowsiness classified at Karolinska Sleepiness Scale > 7, detection required at highway speeds (50 km/h+), phone-use pattern differentiation scored separately | No literal recall %, but concrete behavioral targets worth anchoring to |
| EU GSR (DDAW / ADDW regulations) | Binding regulation, deadlines since July 2024, requiring drowsiness and distraction warning systems | No |

Treat X, Y, and Z below as research-chosen values justified by literature review — not values copied out of a standard.

A pruned model is deemed safe if it passes all three:

1. **Recall ≥ [X]%** on the safety-critical class set, measured at the operating point defined in [Confidence Threshold / Operating-Point Selection](#confidence-threshold--operating-point-selection) below (not raw or overall recall across every class).
2. **Inference latency ≤ [Y] ms** at batch = 1, on the named runtime (see [Hardware & Benchmarking Protocol](#hardware--benchmarking-protocol)).
3. **Storage ≤ [Z] MB** for the exported model artifact.

> *Illustrative starting point only — not a decision:* X = 90%, Y = 33 ms (30 FPS, a common camera frame-rate baseline for in-cabin sensing), Z = 10 MB (roughly double a nano-scale checkpoint's typical FP32 size, leaving headroom before quantization shrinks it further). Replace after the literature review.

### Weighted Metric (F-β)

"Weighted F1" was ambiguous. In most ML tooling (e.g. scikit-learn's `average='weighted'`), "weighted F1" means per-class F1 averaged by class frequency — unrelated to penalizing false negatives. What this project actually needs is an **F-β score**, which explicitly trades precision for recall:

**F_β = (1 + β²) × (Precision × Recall) / (β² × Precision + Recall)**

- FN penalized 5× more than FP → β = √5 ≈ 2.24
- FN penalized 10× more than FP → β = √10 ≈ 3.16

FN penalty weight: **[5x / 10x — TBD]**, so β = **[TBD]**. The **F-β** column replaces "Weighted F1" throughout the tables below; state the chosen β once here and use it consistently everywhere.

**Safety Margin** (Table 4): Safety Margin (pp) = Achieved Recall (%) − Required Recall Threshold [X]%. Positive means passing; negative means the pipeline has crossed the safety cliff.

**Recovery %** (Table 3): Recovery % = (Student metric − Student Baseline metric) / (Teacher metric − Student Baseline metric) × 100 — computed separately for mAP and for Recall, since Recall is the number the SAFE threshold actually checks.

### Hardware & Benchmarking Protocol

- **The RTX 4060 (8 GB) is a proxy, not the target.** "Edge-deployed DMS" implies embedded automotive silicon, which a desktop RTX 4060 won't match in latency or power profile. Every latency claim below is a **relative** comparison between compression techniques on one fixed, accessible platform — not an absolute claim about production deployability.
- **Report latency at batch = 1.** A DMS processes one camera frame at a time, so large-batch throughput isn't the number that matters here. FPS = 1000 / latency_ms(batch=1).
- **Name the inference runtime.** Raw PyTorch eager-mode latency understates INT8/FP16 gains substantially. Export and benchmark through **[TensorRT / ONNX Runtime — TBD]**, and record which one in every table caption, not just in prose.
- **KD stage, 8 GB budget.** Run the teacher in `eval()` / `no_grad()` mode — it then only needs activation memory for a forward pass, not gradients or optimizer state, so even a YOLO11x/YOLO26x teacher should fit alongside a nano student. Use AMP (FP16) for the student's training pass. Because the standard YOLO recipe leans heavily on mosaic/mixup augmentation, the teacher's forward pass should happen inside the training loop, on the same augmented batch the student sees (**online, paired distillation**), rather than from pre-computed, cached teacher logits — offline caching only makes sense if augmentation is turned down for the distillation stage, which is itself a decision worth stating explicitly. Online vs. offline: **[TBD]**.

### Confidence Threshold / Operating-Point Selection

Every Recall, Precision, and FN Rate cell in all four tables depends on a detection-confidence threshold — without a fixed, written procedure, no two rows are actually comparable. Applied identically to every model, in every table:

1. Fix IoU = 0.5 for matching predictions to ground truth.
2. For each model, sweep the confidence threshold on the **validation** set and pick the lowest threshold that still achieves Recall ≥ [X]% on the safety-critical class set (X from the [SAFE threshold](#definition-of-safe-threshold) definition above).
3. Freeze that per-model threshold, then report Precision, FN Rate, and F-β at that operating point on the **held-out test** set.

The threshold becomes a per-model *output* of this procedure rather than a shared constant — the comparison that matters is "can this model still clear the recall bar," not "which model scores best at some arbitrary fixed threshold."

---

## 4. Experiments

*(Ablation Study & Proposed Method — Section 4 of the [Outline](#outline). 4.1 isolates each compression technique; 4.2 combines the winners into the deployable pipeline.)*

### 4.1 Benchmarking (Tables 1–3)

Each technique below is evaluated on its own, against an unmodified baseline, before anything is combined.

#### Table 1: Pruning Sweep

> **Note:** All pruned models must be fine-tuned before evaluation. Unstructured pruning gives no real latency benefit without sparse-kernel support — treat it as a secondary/illustrative comparison at most. **Structured (channel) pruning** is the primary axis of this study, since that's what an actual edge-inference benefit depends on. Pruning criterion: **[TBD — e.g. L1-norm channel pruning, BN-scale/Slimming, or a dependency-graph-aware method such as Torch-Pruning]**.

| Model | Pruning Ratio | Pruning Type | Criterion | Batch Size | Fine-tune Epochs | Params (M) | Storage (MB) | Latency (ms) | FPS | FLOPS | mAP | Recall | Precision | F-β | FN Rate % | Safety Pass? |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| YOLO11n | 0% (baseline) | - | - |  | - |  |  |  |  |  |  |  |  |  |  |  |
| YOLO11n | 20% |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| YOLO11n | 40% |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| YOLO11n | 60% |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| YOLO11n | 80% |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| YOLO26n | 0% (baseline) | - | - |  | - |  |  |  |  |  |  |  |  |  |  |  |
| YOLO26n | 20% |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| YOLO26n | 40% |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| YOLO26n | 60% |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| YOLO26n | 80% |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

*This coarse sweep (20/40/60/80%) is a starting point — add finer-grained rows (e.g. 45%, 50%, 55%) once it shows where Recall first drops below [X]%; pinpointing that cliff precisely is the actual research question.*

#### Table 2: Quantization Comparison

> **Note:** Parameters do not change with quantization — bit-width is tracked instead. **Expected finding:** QAT-INT8 may be slower than PTQ-INT8 on 'n' models due to tensor reformatting layers. Test this separately for YOLO11n (has DFL) and YOLO26n (DFL removed), since DFL's regression head is a plausible source of exactly that kind of reformatting overhead — if the anomaly shrinks or disappears on YOLO26n, that's worth a full paragraph in Discussion, not a footnote. Runtime: **[TBD — must match Hardware & Benchmarking Protocol]**. Calibration set size (PTQ): **[TBD]**.

| Model | Precision | Weight Bit-Width | Runtime | Calibration Set Size | Storage (MB) | Latency (ms, batch=1) | FPS | mAP | Recall | Precision | F-β | FN Rate % | Safety Pass? |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| YOLO11n | FP32 | 32 | - | - |  |  |  |  |  |  |  |  |  |
| YOLO11n | FP16 | 16 |  | - |  |  |  |  |  |  |  |  |  |
| YOLO11n | INT8 PTQ | 8 |  |  |  |  |  |  |  |  |  |  |  |
| YOLO11n | INT8 QAT | 8 |  | - |  |  |  |  |  |  |  |  |  |
| YOLO26n | FP32 | 32 | - | - |  |  |  |  |  |  |  |  |  |
| YOLO26n | FP16 | 16 |  | - |  |  |  |  |  |  |  |  |  |
| YOLO26n | INT8 PTQ | 8 |  |  |  |  |  |  |  |  |  |  |  |
| YOLO26n | INT8 QAT | 8 |  | - |  |  |  |  |  |  |  |  |  |

#### Table 3: Knowledge Distillation

> **Note:** The "Student Baseline" row establishes the performance floor; Recovery % shows how much of the teacher's advantage KD recovers. This table tests KD as a standalone technique on the **unpruned** nano model (full-size teacher → full-size student, no pruning involved) — consistent with "each technique in isolation" from Section 4.1. Table 4's "+ KD Recovery" step applies KD to the **pruned** checkpoint from Table 1 instead — a different use of the same three letters, not a preview of this table.

| Student Model | Teacher Model | Teacher Storage (MB) | Teacher mAP | Student Baseline mAP | Student mAP | Student Baseline Recall | Student Recall | Recovery % (mAP) | Recovery % (Recall) | F-β | Latency (ms) | Storage (MB) | FN Rate % | Safety Pass? |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| YOLO11n | - (no KD) | - | - |  | - |  | - | - | - |  |  |  |  |  |
| YOLO11n | YOLO11m |  |  |  |  |  |  |  |  |  |  |  |  |  |
| YOLO11n | YOLO11x |  |  |  |  |  |  |  |  |  |  |  |  |  |
| YOLO26n | - (no KD) | - | - |  | - |  | - | - | - |  |  |  |  |  |
| YOLO26n | YOLO26m |  |  |  |  |  |  |  |  |  |  |  |  |  |
| YOLO26n | YOLO26x |  |  |  |  |  |  |  |  |  |  |  |  |  |

### 4.2 Proposed Method (Table 4)

Table 4 takes the best-performing configuration from each of Tables 1–3 and chains them into the actual deployment candidate.

#### Table 4: Proposed Method

> [!CAUTION]
> **Order of Operations:** Prune → Fine-tune → Distill → Quantize (QAT).
> The "Safety Margin" tracks how close the model is to the safety cliff — if it goes negative, the pipeline fails. **Safety Pass? is left blank for every row, including Final, until the experiment actually says yes or no** — a template shouldn't pre-decide its own result.

| Model | Step | Method | Precision | Params (M) | Latency (ms) | Storage (MB) | Recall | Safety Margin (pp) | FN Rate % | mAP | Safety Pass? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| YOLO11n | Baseline | None | FP32 |  |  |  |  | - |  |  |  |
| YOLO11n | Step 1 | Pruning (best ratio from Table 1) | FP32 |  |  |  |  |  |  |  |  |
| YOLO11n | Step 2 | + KD Recovery (pruned checkpoint as student) | FP32 |  |  |  |  |  |  |  |  |
| YOLO11n | Step 3 | + INT8 QAT | INT8 |  |  |  |  |  |  |  |  |
| **YOLO11n** | **Final** | **Pruned + KD + QAT** | **INT8** |  |  |  |  |  |  |  |  |
| YOLO26n | Baseline | None | FP32 |  |  |  |  | - |  |  |  |
| YOLO26n | Step 1 | Pruning (best ratio from Table 1) | FP32 |  |  |  |  |  |  |  |  |
| YOLO26n | Step 2 | + KD Recovery (pruned checkpoint as student) | FP32 |  |  |  |  |  |  |  |  |
| YOLO26n | Step 3 | + INT8 QAT | INT8 |  |  |  |  |  |  |  |  |
| **YOLO26n** | **Final** | **Pruned + KD + QAT** | **INT8** |  |  |  |  |  |  |  |  |

---

## 5. Discussion — Key Findings & Narrative

*This section is the skeleton for the full Discussion (Outline item 5). Update as data is collected; the bracketed examples below are placeholders for the actual findings, not predictions.*

- **Pruning Limits:** [e.g., YOLO11n failed safety at 30% pruning due to compounding error.]
- **KD Value:** [e.g., KD recovered X% of recall, allowing 40% pruning.]
- **Quantization Anomaly:** [e.g., QAT required for safety, but slower than PTQ on the edge device; check whether this holds for both YOLO11n and YOLO26n, or only the DFL-based one.]
- **Hero Model:** [e.g., YOLO26n demonstrated superior robustness to compression; check whether this tracks with its small-object (STAL) design specifically on the phone class.]

---

## 6. Limitations & Threats to Validity

- **Hardware generalizability.** All latency figures come from a desktop RTX 4060, not the eventual embedded automotive target. Conclusions about *relative* ranking between compression techniques are far more defensible than conclusions about *absolute* latency being production-ready.
- **Single-GPU compute budget.** State explicitly whether results are single-seed — a reasonable constraint at this project scale, but it should be stated rather than implied. Consider spending whatever multi-seed budget exists specifically on the final Table 4 configurations, since those carry the project's actual safety claim.
- **Dataset coverage after the 2026 revision.** No IR, no simulator data, a restricted subject pool. Any claim about robustness to night driving or low light is currently untestable on this data and should be explicitly scoped out of the conclusion rather than implied by omission.
- **Subject independence.** State the split explicitly; if it isn't subject-independent, recall numbers are optimistic.
- **Operating-point sensitivity.** Results depend on the confidence-threshold procedure above. A brief sensitivity check — how much Recall/FN Rate moves for a ±0.05 threshold shift — would meaningfully strengthen the safety claim.
- **Compression order not itself ablated.** The pipeline tests presence/absence of each stage in one fixed order (Prune → Fine-tune → Distill → Quantize). Whether that order is optimal — versus, say, distilling during the pruning fine-tune rather than after it — is out of scope given the compute budget, but deserves one acknowledging sentence rather than being presented as self-evidently correct.

---

## Open Decisions Checklist
*(Resolve before Table 1's first cell gets filled in.)*

- [ ] Safety-critical class list finalized (recommend phone use as the anchor class)
- [ ] Train/val/test split confirmed subject-independent
- [ ] X (recall threshold), Y (latency budget), Z (storage budget) set from literature review — not left as placeholders
- [ ] Confidence-threshold selection procedure implemented and applied identically across all four tables
- [ ] β value for F-β chosen and stated once
- [ ] Inference runtime (TensorRT / ONNX Runtime) chosen and used consistently for every latency number
- [ ] Pruning criterion (e.g. L1-norm, BN-scale, Torch-Pruning) named
- [ ] Online vs. offline KD decision made and stated, given the 8 GB VRAM budget and mosaic/mixup augmentation

---

## References

- Ultralytics, YOLO26 model documentation: https://docs.ultralytics.com/models/yolo26
- Vicomtech, DMD Driver Monitoring Dataset (incl. 2026 data-availability update): https://github.com/Vicomtech/DMD-Driver-Monitoring-Dataset
- Ortega et al., "DMD: A Large-Scale Multi-Modal Driver Monitoring Dataset for Attention and Alertness Analysis" (2020): https://arxiv.org/pdf/2008.12085
- Smart Eye, on Euro NCAP's 2026 driver monitoring changes: https://smarteye.se/blog/euro-ncap-2026-whats-changing/ and https://smarteye.se/blog/driver-monitoring-euro-ncap-2026/
- Anyverse, on EU GSR and Euro NCAP in-cabin monitoring mandates: https://anyverse.ai/in-cabin-monitoring-navigating-europes-safe-driving-new-standards-3/
- Jama Software / UL Solutions, on ISO 21448 (SOTIF) and ISO/PAS 8800: https://www.jamasoftware.com/requirements-management-guide/automotive-engineering/sotif/ and https://www.ul.com/sis/blog/safety-related-systems-road-vehicles-artificial-intelligence-are-addressed-isopas-88002024
