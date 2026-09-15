# Pruning-Induced Missed Detections in Edge-Deployed DMS
`pruning-missed-detection` — methodology spec & experiment templates

*A living methodology spec, not a paper draft — sections below are either fully drafted (Related Work, Methodology, Experiment templates, Limitations) or still outline-only (Introduction, Discussion, Conclusion), and the Outline marks which is which. Bracketed items — `[X]`, `[TBD]`, etc. — are open decisions, not answers; resolve every item on the [Open Decisions Checklist](#open-decisions-checklist) before Table 1's first cell gets filled in.*

> [!TIP]
> **Revision pass (Sept 2026).** This pass cross-checked every dataset/standard/architecture claim in the previous version against current sources, drafted the Related Work section from an actual literature search, and ran a first-principles methodology audit for gaps the original spec didn't cover (frame-vs-event-level recall, statistical power on a rare class, KD-method choice, camera-view selection, calibration-set leakage, and a few others). Every section below now carries one of three tags:
> - 🟢 **`[!TIP]`** — checked, internally consistent, safe to freeze. Stop relitigating these.
> - 🟡 **`[!NOTE]`** — genuinely has more than one reasonable path. Alternatives are listed (usually the top 3); pick one and move on.
> - 🔴 **`[!CAUTION]`** — blocks correct execution of the experiments below it until a human makes a call. Read these before writing any code.
>
> **Target venue (working call):** journal for the full study (IEEE Access / IEEE Trans. on Intelligent Vehicles / MDPI Sensors or Electronics), with an optional fast conference paper (IEEE IV Symposium / ITSC) carved out of Tables 1–2 alone, given how new YOLO26 is. Full reasoning is in the chat message that shipped this revision, not repeated here.

## Contents

- [Core Question](#core-question)
- [Outline](#outline)
- [2. Related Work](#2-related-work)
- [3. Methodology](#3-methodology)
  - [Dataset](#dataset)
  - [Definition of SAFE threshold](#definition-of-safe-threshold)
  - [Detection Granularity: Frame-Level vs. Event-Level Recall](#detection-granularity-frame-level-vs-event-level-recall)
  - [Weighted Metric (F-β)](#weighted-metric-f-β)
  - [Hardware & Benchmarking Protocol](#hardware--benchmarking-protocol)
  - [Confidence Threshold / Operating-Point Selection](#confidence-threshold--operating-point-selection)
  - [Statistical Power & Reporting Uncertainty](#statistical-power--reporting-uncertainty)
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

> [!TIP]
> Well-scoped already — this is the one piece of prose in the whole spec that doesn't need another pass. Every table below should trace back to it.

> **The Core Question:** "How much can a real-time detector be pruned before missed detections on safety-critical classes cross an acceptable risk threshold, and once knowledge distillation and INT8 quantization are already applied, does pruning still buy meaningful latency or storage headroom for edge-deployed DMS without crossing that threshold?"
>
> *Refined from the original framing, which was ambiguous between "does pruning beat quantization/KD as an alternative" and "does pruning still help once quantization/KD are already applied." Table 4 tests the latter, so the question now matches it.*

---

## Outline

1. **Introduction** *(outline only — not yet drafted).* DMS on edge devices has strict latency and safety constraints.

   > [!NOTE]
   > **Proposed contributions (draft — will firm up once Tables 1–2 produce real numbers, but worth writing down now so the paper has a spine to write toward):**
   > 1. A safety-weighted, operating-point-frozen evaluation protocol for detector compression — per-model confidence-threshold selection against a recall floor, an FN-weighted F-β, an explicit hardware/benchmarking protocol, and a reported Safety Margin — assembled as one reusable procedure. The safety-metrics literature and the compression literature currently only offer these pieces separately (see Related Work).
   > 2. The first systematic pruning + quantization + KD ablation — as far as this literature pass found — for a real-time detector evaluated under a false-negative-cost lens, on an actual in-cabin DMS task rather than a generic classification benchmark or an outward-facing (VRU/pedestrian) detector.
   > 3. A timely natural experiment comparing compression robustness across a DFL+NMS architecture (YOLO11) and a DFL-free, NMS-free, small-object-aware architecture (YOLO26, released January 2026) — testing, rather than assuming, whether YOLO26's deployment-oriented redesign holds up under aggressive pruning and quantization on a small, safety-critical object class.
   > 4. Empirical evidence on whether compression-stage *order* matters for detection the way it's been shown to for classification — positioned directly against the closest published ordering study (see Related Work and the Table 4 CAUTION).
   >
   > Also worth one explicit sentence in the Introduction: **DDAW (fatigue, steering-pattern-based) and ADDW/Euro NCAP (distraction, camera+gaze-based) are different regulatory hooks.** This project's phone-detection task is motivated by the latter, not the former — say so up front so a reviewer doesn't conflate the two GSR systems.

2. **Related Work** *(drafted — [jump to section](#2-related-work)).* Structured vs. unstructured pruning; detector-specific quantization and the DFL question; KD for detection heads (logit vs. feature-based); the compounded gap this project fills.

3. **Methodology** *(drafted — [jump to section](#3-methodology)).* Safety-weighted F-β metric, FN-penalty threshold, dataset, detection granularity, hardware & benchmarking protocol, confidence-threshold selection procedure, statistical power.

4. **Experiments** *(drafted — [jump to section](#4-experiments); templates only, no data filled in yet)*
   - 4.1 Benchmarking: Tables 1–3. Each technique in isolation.
   - 4.2 Proposed Method: Table 4. How we combine them to build a deployable DMS model.

5. **Discussion** *(skeleton only — [jump to section](#5-discussion--key-findings--narrative)).* Trade-offs, the QAT vs. PTQ anomaly on "n" models, compounding errors across the pipeline, and architecture-dependent compression behavior. YOLO26 isn't just a resized YOLO11 — it removes Distribution Focal Loss (DFL) and NMS in favor of a native one-to-one end-to-end head, and adds STAL (Small-Target-Aware Label Assignment) for small objects. Three implications worth discussing explicitly: (1) DFL's softmax-over-bins regression has historically been tricky to quantize cleanly in the YOLOv8–YOLO11 family, so YOLO26 lacking it could explain a different PTQ/QAT gap than YOLO11n shows — this reframes the "anomaly" as a testable hypothesis rather than an unexplained finding; (2) removing NMS (a variable-cost, hard-to-batch post-processing step) likely changes how latency scales under pruning versus a model that still runs it, so findings shouldn't be assumed to transfer between the two families by default; (3) STAL specifically targets small-object accuracy, and phone-in-hand is small in-frame, so pruning-induced small-object degradation deserves its own line rather than folding into an aggregate Recall number.

6. **Limitations & Threats to Validity** *(drafted — [jump to section](#6-limitations--threats-to-validity)).*

7. **Conclusion** *(outline only — not yet drafted).* The combined pipeline provides the best safety-to-latency ratio for edge DMS.

---

## 2. Related Work

> [!NOTE]
> Real, verified citations below — but framing and emphasis will shift once you've read the primary sources yourself. Treat "Closest Related Work, and the Gap" as a strong working hypothesis, not a settled fact.

### (a) Structured vs. unstructured pruning for real-time detectors

Structured (channel) pruning is the axis this project cares about — unstructured pruning needs sparse-kernel hardware support to turn into real latency gains, which is why Table 1 already treats it as secondary. Classic filter-importance criteria remain the simplest, most reproducible baselines: **L1-norm magnitude pruning** (Li et al., "Pruning Filters for Efficient ConvNets," 2017) and **BN-scale / Network Slimming** (Liu et al., ICCV 2017). The more modern default is dependency-graph-aware structured pruning — **DepGraph / Torch-Pruning** (Fang et al., CVPR 2023) — which automatically handles the cross-layer coupling in YOLO's concat-heavy neck blocks (C2f, C3k2) that per-layer criteria don't reason about on their own. See the Pruning Criterion NOTE under [Table 1](#table-1-pruning-sweep).

### (b) Detector-specific quantization

DFL's softmax-over-bins regression head has a specific, citable quantization cost. Recent architectural analyses of YOLO26 explicitly connect probability-distribution box heads to precision loss and reformatting/dequantization overhead under INT8 export on mobile and edge NPUs, and frame YOLO26's move to direct box regression as a response to exactly that problem — a "deployment-aware design" choice, not a cosmetic one. That gives Table 2's QAT-vs-PTQ anomaly hypothesis (see the Table 2 note) a real citation to test against, not just an informal expectation.

### (c) Knowledge distillation for detection heads

Three genuinely different families exist here, and the choice is **not architecture-neutral** for this project specifically — see the KD-method CAUTION under [Table 3](#table-3-knowledge-distillation) for the full argument:

- **Response/logit-based** (Hinton, Vinyals & Dean, "Distilling the Knowledge in a Neural Network," 2015 — adapted to detection outputs). Architecture-agnostic.
- **Localization Distillation, "LD"** (Zheng et al., CVPR 2022). Distills localization knowledge through the box's *probability-distribution* representation — i.e., it leans on the same distributional (DFL-style) machinery that YOLO26 removed. Notably, LD's own finding is that logit/distribution mimicking **outperforms** feature imitation specifically for localization knowledge, which complicates any easy assumption that "feature-based KD is just better."
- **Feature-based** (FGD — Yang et al., CVPR 2022; CWD — Shu et al., ICCV 2021). Architecture-agnostic like response-based KD; distills intermediate feature maps rather than final outputs.

### Closest related work, and the gap

> [!TIP]
> This synthesis is well-grounded across multiple independent searches. The three-way gap it describes is the actual argument for why this paper needs to exist, and it's safe to build the Introduction around it.

The single closest methodological analog found is a 2026 study that runs pruning, INT8 QAT, and KD as an ordered three-stage pipeline and ablates the stage order — but entirely on **CIFAR-10/100 image classification** (ResNet-18, WRN-28-10, VGG-16-BN), with no detection task, no safety framing, and no false-negative cost anywhere in the metric. Its central finding — KD works best applied **after** quantization, recovering accuracy specifically within the constrained sparse+INT8 regime, not before it — is the literature's one concrete challenge to this project's current Table 4 ordering (see that CAUTION).

On the safety-metrics side, a small but active line of work already argues that plain precision/recall under-serves safety-critical detection: a criticality-weighted recall metric that scores missed detections by how dangerous the missed object would have been; a cost-sensitive framework with explicit user-set budgets on missed vs. false detections; and a 2026 decision-support framework for asymmetric, only-approximately-known error costs in industrial detection. **None of these three test their metric under model compression** — they assume a fixed model and ask "what's the right operating threshold," not "does the safety margin survive being pruned, quantized, and distilled."

On the automotive side, edge-deployed KD for safety-relevant detection already exists — a 2026 paper distills a YOLOv8 student for vulnerable-road-user (pedestrian/cyclist) detection under INT8 quantization — but it's **outward-facing** perception, not in-cabin driver monitoring, and it studies KD + quantization only, not the three-way pipeline. The much larger body of driver-distraction-detection-with-YOLO literature (mostly proposing attention modules or lightweight architectural blocks bolted onto YOLOv8/v12) repeatedly *names* pruning and quantization as future work rather than delivering them — one representative paper explicitly lists exploring "model compression techniques, such as quantization and pruning" as a stated next step. That's as clean a citable confirmation of the gap as this kind of project usually gets.

**The argument, stated plainly:** nobody has combined (1) a joint pruning + quantization + KD compression pipeline, (2) evaluated under a false-negative-cost-weighted safety metric with a frozen operating point, (3) on a real in-cabin DMS detection task, (4) across an architecture transition (DFL+NMS → DFL-free, NMS-free) that happened specifically for deployment reasons. Each of (1)–(4) exists somewhere in isolation in the literature above; their intersection does not yet exist. That intersection is this paper.

---

## 3. Methodology

### Dataset

**DMD (Driver Monitoring Dataset)** — Vicomtech / Ortega et al., 2020, funded by the EU Horizon 2020 VI-DAS project.

DMD is primarily a temporal action, fatigue, and gaze/head-pose dataset — 93 classes across temporal, geometric, and context annotations, recorded from synchronized body/face/hand camera views, in OpenLABEL/VCD format — not a ready-made multi-class object-detection benchmark in the COCO sense. Only a small number of objects carry spatial `object_in_scene` annotation suitable for a YOLO-style bounding-box target: **phone, hairbrush/comb, bottle**. Everything else (drowsiness, distraction state, gaze region) is a temporal or state label rather than a box, and would need a deliberate, documented conversion (e.g. a fixed face/hand region-of-interest standing in for a box) before a detector could train on it at all.

As of DMD's 2026 revision, simulator recordings and the Depth/IR streams have been removed from public availability — only RGB material from the real-car scenario, drawn from a restricted subset of the original driver pool, is currently distributable. Any plan assuming IR footage (common in production DMS for night driving) or simulator data no longer holds on this data; this is scoped out explicitly in Limitations & Threats to Validity rather than discovered mid-experiment.

> [!CAUTION]
> **Confirm a current Data Use Agreement is in place before downloading anything.** The 2026 revision happened specifically *because of* license and hosting-constraint changes, not as a routine update — any pre-2026 access arrangement, download, or locally-cached copy of DMD may not reflect the current terms (which subjects are usable, which streams are permitted). This is a five-minute check now versus a real problem if it surfaces after months of experiments on data that turns out to be under revised terms.

> [!CAUTION]
> **Which of DMD's three synchronized camera views (face / hands / body) feeds the detector is not yet specified anywhere in this spec — and it isn't a free choice.**
> 1. **Hands view.** Most direct line of sight to the phone object itself; likely the best raw detection accuracy for "is a phone present," but the object can still be occluded by the hand holding it, and this view says nothing about where the driver is *looking*, which is what several Euro NCAP phone-use sub-criteria actually score.
> 2. **Face view.** Captures head-down / gaze-at-lap patterns that correlate with phone use even when the phone itself isn't in frame — closer to how Euro NCAP's gaze-based criteria actually work — but the phone object is smaller and often not visible at all, which cuts against training a bounding-box detector on it.
> 3. **Multi-view** (hands + face, fused or trained/evaluated separately). Most realistic for a production DMS, and lets the paper report both an "object presence" and a "driver-engagement" signal, but roughly doubles the annotation/pipeline work and reopens "which view failed" inside every compression result.
>
> Also unresolved: what fraction of DMD's video frames actually carry `object_in_scene` spatial annotation for these classes (spatial annotation on video is rarely dense), and therefore how the image-level train/val/test set is actually assembled from the underlying clips. Both belong on the [Open Decisions Checklist](#open-decisions-checklist), not just in this callout.

**Decisions to make explicit before Table 1 is touched:**

- **Which classes:** [TBD] — leading candidate is **phone use** as the primary safety-critical class: it has native bounding-box annotation, it's a small in-frame object (which stresses the pruning-robustness question directly), and it's scored as its own category in Euro NCAP's 2026 protocol (see SAFE threshold, below). Still open: whether hairbrush/bottle are in scope as secondary distraction classes, or excluded as not safety-relevant. *(If more than one class ends up in the safety-critical set, decide **micro-averaged** recall — pool TP/FN across classes, matches "did any camera-relevant hazard get missed" — versus **macro-averaged** — average of per-class recall, protects a rare class like hairbrush from being statistically swamped by phone's larger sample count. These give materially different numbers under DMD's likely class imbalance; state which one "Recall" means in every table header, not just in prose.)*
- **Split strategy:** [TBD] — must be subject-independent across train/val/test. DMD has a limited driver pool; if the same driver appears in both train and test, recall numbers get inflated by identity leakage rather than genuine generalization.
- **Class imbalance:** [TBD] — document the sampling or loss-weighting strategy used, since it interacts directly with the confidence-threshold procedure below.

### Definition of SAFE threshold

"Anchor to ISO 26262 / Euro NCAP" undersold the standards landscape. Neither hands over a recall percentage directly — the standards that actually deal with perception performance limits and AI-specific risk are ISO 21448 (SOTIF) and ISO/PAS 8800, and both explicitly leave the number-picking to the practitioner:

| Standard | What it actually governs | Hands you a recall number? |
|:---|:---|:---|
| ISO 26262 | Functional safety of electrical/electronic systems; hazards from component malfunction or failure | No |
| ISO 21448 (SOTIF) | Safety of the intended function; hazards from performance limitations even with no fault (e.g. a correctly-working camera that still misses something) | No — but this is the standard actually concerned with perception accuracy gaps |
| ISO/PAS 8800 (approved 13 Dec. 2024) | AI/ML-specific extension covering training-data gaps, unrepresented edge cases, distribution shift; treats training/validation/test data itself as a safety artifact that must be specified, justified, and controlled | No — assumes the vehicle-level risk budget has already been broken into a verifiable per-component number via 26262/21448 |
| Euro NCAP 2026 protocol | Rating criteria: direct monitoring required, drowsiness classified at Karolinska Sleepiness Scale > 7 at highway speeds (50 km/h+), phone-use pattern differentiation (looking at / holding / interacting with) scored as its own category | No literal recall %, but concrete behavioral targets worth anchoring to |
| EU GSR — DDAW (fatigue) since Jul 2024; ADDW (distraction) since Jul 2026 | Binding regulation. **DDAW** is steering/lane-pattern-based and largely camera-independent; **ADDW** is the camera+gaze-based system this project's phone-detection task actually motivates — don't conflate the two | No |

> [!TIP]
> The standards table and the "no standard hands you a number" framing are correct and hold up — this is a genuinely accurate account of a landscape that's easy to oversimplify. Freeze the table; only X/Y/Z below remain open.

Treat X, Y, and Z below as research-chosen values justified by literature review — not values copied out of a standard.

A pruned model is deemed safe if it passes all three:

1. **Recall ≥ [X]%** on the safety-critical class set, measured at the operating point defined in [Confidence Threshold / Operating-Point Selection](#confidence-threshold--operating-point-selection) below (not raw or overall recall across every class), at the [detection granularity](#detection-granularity-frame-level-vs-event-level-recall) chosen below.
2. **Inference latency ≤ [Y] ms** at batch = 1, on the named runtime (see [Hardware & Benchmarking Protocol](#hardware--benchmarking-protocol)).
3. **Storage ≤ [Z] MB** for the exported model artifact.

> *Illustrative starting point only — not a decision:* X = 90%, Y = 33 ms (30 FPS, a common camera frame-rate baseline for in-cabin sensing), Z = 10 MB (roughly double a nano-scale checkpoint's typical FP32 size, leaving headroom before quantization shrinks it further). Replace after the literature review.

### Detection Granularity: Frame-Level vs. Event-Level Recall

> [!CAUTION]
> **What "a missed detection" means is not yet defined anywhere in this spec, and it changes every number in every table.**
> A DMS isn't scored per video frame in practice — Euro NCAP's phone-use criteria care whether the *behavior* was caught, typically within a time budget, not whether every single frame of a multi-second phone-use event individually cleared the confidence threshold. As written, Recall / FN Rate / F-β are implicitly **frame-level**, which can both over-penalize a model that flickers in and out of detecting a sustained event, and under-penalize a model that gets lucky on isolated frames while systematically missing the sustained pattern. Pick one before Table 1:
> 1. **Frame-level (current default).** Simplest, matches standard object-detection benchmarking conventions, directly comparable to the compression literature. Weakest tie to what a real DMS alert pipeline actually does.
> 2. **Event-level.** A ground-truth phone-use interval counts as "detected" if the model fires on ≥1 frame (or ≥k consecutive frames, to reject spurious single-frame flicker) within it, within some latency budget. Matches how Euro NCAP and a real alerting pipeline actually work; needs a new metric-computation script built around temporal event boundaries (already implicit in DMD's annotations, since they're the source of the bounding-box intervals).
> 3. **Both, frame-level primary.** Report frame-level Recall/F-β as the headline (for comparability with the compression literature); add event-level detection rate as a secondary "does this survive as a product" check in Table 4 and Discussion only, not in every cell of Tables 1–3. **Best ROI-to-effort ratio** — it doesn't require rebuilding four tables' worth of metric code around event logic, but it stops the paper from over-claiming operational safety from a purely per-frame number.

### Weighted Metric (F-β)

> [!TIP]
> The formula and the β-selection logic below were independently re-derived, not just trusted — they're correct. Freeze this section.

"Weighted F1" was ambiguous. In most ML tooling (e.g. scikit-learn's `average='weighted'`), "weighted F1" means per-class F1 averaged by class frequency — unrelated to penalizing false negatives. What this project actually needs is an **F-β score**, which explicitly trades precision for recall:

**F_β = (1 + β²) × (Precision × Recall) / (β² × Precision + Recall)**

- FN penalized 5× more than FP → β = √5 ≈ 2.24
- FN penalized 10× more than FP → β = √10 ≈ 3.16

*Why the square root, spelled out (worth keeping in the paper's methods section verbatim, since a reviewer will ask):* rewriting F_β in terms of raw counts gives F_β = TP / (TP + w_FN·FN + w_FP·FP), where w_FN = β²/(1+β²) and w_FP = 1/(1+β²). The ratio of those two weights is exactly **β²**. So "FN costs 5× what FP costs" means w_FN/w_FP = 5, i.e. β² = 5, i.e. β = √5. This makes the choice of β fully auditable rather than an asserted constant.

FN penalty weight: **[5x / 10x — TBD]**, so β = **[TBD]**. The **F-β** column replaces "Weighted F1" throughout the tables below; state the chosen β once here and use it consistently everywhere. *(Edge case worth one footnote in the methods section: define F_β = 0 when TP = 0, i.e. when precision or recall is undefined, rather than leaving the cell blank — a badly-broken pruned model that predicts nothing shouldn't silently disappear from a sweep.)*

**Safety Margin** (Table 4): Safety Margin (pp) = Achieved Recall (%) − Required Recall Threshold [X]%. Positive means passing; negative means the pipeline has crossed the safety cliff.

**Recovery %** (Table 3): Recovery % = (Student metric − Student Baseline metric) / (Teacher metric − Student Baseline metric) × 100 — computed separately for mAP and for Recall, since Recall is the number the SAFE threshold actually checks.

### Hardware & Benchmarking Protocol

- **The RTX 4060 (8 GB) is a proxy, not the target.** "Edge-deployed DMS" implies embedded automotive silicon, which a desktop RTX 4060 won't match in latency or power profile. Every latency claim below is a **relative** comparison between compression techniques on one fixed, accessible platform — not an absolute claim about production deployability.
- **Report latency at batch = 1.** A DMS processes one camera frame at a time, so large-batch throughput isn't the number that matters here. FPS = 1000 / latency_ms(batch=1).
- **Name the inference runtime.** Raw PyTorch eager-mode latency understates INT8/FP16 gains substantially. Export and benchmark through the runtime chosen below, and record which one in every table caption, not just in prose.
- **KD stage, 8 GB budget.** Run the teacher in `eval()` / `no_grad()` mode — it then only needs activation memory for a forward pass, not gradients or optimizer state, so even a YOLO11x/YOLO26x teacher should fit alongside a nano student. Use AMP (FP16) for the student's training pass. Because the standard YOLO recipe leans heavily on mosaic/mixup augmentation, the teacher's forward pass should happen inside the training loop, on the same augmented batch the student sees (**online, paired distillation**), rather than from pre-computed, cached teacher logits — offline caching only makes sense if augmentation is turned down for the distillation stage, which is itself a decision worth stating explicitly. Online vs. offline: **[TBD]**.

> [!NOTE]
> **Runtime choice (currently `[TBD]`) — three real options, and YOLO26 exports to all three:**
> 1. **TensorRT.** Best absolute latency on the RTX 4060 proxy hardware and on NVIDIA Jetson-class automotive edge silicon, which makes the "relative ranking should transfer toward the real target" argument in Limitations easiest to defend. Costs: NVIDIA-only, extra export/calibration complexity, version-sensitive.
> 2. **ONNX Runtime.** More portable, simpler export path, easier for reviewers to reproduce without matching hardware. Typically slower than TensorRT on the same GPU — matters less for *relative* comparisons across compression techniques, but name it as a limitation if chosen.
> 3. **OpenVINO.** Only makes sense if the eventual target silicon is Intel-based; probably not the right default given the RTX-4060-as-GPU-proxy framing already committed to above.
>
> **Leaning:** TensorRT, for consistency with "GPU proxy standing in for GPU-class edge silicon" — but this is a one-line decision to make and state once, before Table 2's first row. Re-benchmarking everything after switching runtimes mid-project is the single most wasteful mistake available at this stage.

> [!TIP]
> **Latency-measurement rigor — add these four lines to the protocol now; they cost nothing today and prevent noisy numbers later:**
> - Discard the first ~50 inference iterations as warm-up (covers CUDA context / kernel-autotuning settle-in); measure over the next 300–500 and report the **median**, not the mean — latency distributions are right-skewed by occasional OS/driver jitter.
> - Explicitly synchronize per iteration (`torch.cuda.synchronize()`, or the TensorRT/ONNX-Runtime execution-context equivalent) — unsynchronized async dispatch silently understates latency.
> - Fix GPU clocks, or at minimum log clock state and thermal readings, if the RTX 4060 is in a laptop/SFF form factor — a sustained INT8 benchmarking sweep is exactly the workload that triggers mid-run thermal throttling, which would quietly bias later table rows relative to earlier ones.
> - Keep these four rules identical across every table — name them next to the runtime in every caption, not just once in prose.

### Confidence Threshold / Operating-Point Selection

> [!TIP]
> The procedure itself (threshold picked on val, frozen, then evaluated on test) is sound and has no leakage problem. Freeze the procedure; the one addition below is about reporting, not about redesigning it.

Every Recall, Precision, and FN Rate cell in all four tables depends on a detection-confidence threshold — without a fixed, written procedure, no two rows are actually comparable. Applied identically to every model, in every table:

1. Fix IoU = 0.5 for matching predictions to ground truth.
2. For each model, sweep the confidence threshold on the **validation** set and pick the lowest threshold that still achieves Recall ≥ [X]% on the safety-critical class set (X from the [SAFE threshold](#definition-of-safe-threshold) definition above), at the detection granularity chosen above.
3. Freeze that per-model threshold, then report Precision, FN Rate, and F-β at that operating point on the **held-out test** set.

The threshold becomes a per-model *output* of this procedure rather than a shared constant — the comparison that matters is "can this model still clear the recall bar," not "which model scores best at some arbitrary fixed threshold."

*One addition:* once multi-seed runs are in play (see [Statistical Power](#statistical-power--reporting-uncertainty) below), "per model" implicitly becomes "per model, per seed." Report the frozen threshold itself as a value with variance across seeds for any row that has multiple seeds, rather than a single number — a threshold that swings a lot between seeds is itself a finding about the model's stability, not noise to be averaged away.

### Statistical Power & Reporting Uncertainty

> [!CAUTION]
> **Two compounding problems, neither currently addressed anywhere in this spec, and both undermine the paper's central "cliff" claim if left unhandled:**
> 1. **Sample size on the safety-critical class.** If "phone" positives in the held-out test set number in the low hundreds — plausible after the 2026 subject-pool restriction — a naive recall estimate can easily carry a ±5–10pp margin, which is comparable in size to the "cliff" the pruning sweep is trying to locate. Report a **confidence interval on every Recall cell** (Wilson or Jeffreys interval, not the normal approximation — these behave better near 0%/100% and with small n), not a bare percentage.
> 2. **Frames are not independent trials.** DMD annotations come from continuous video; consecutive frames of the same phone-use event are highly correlated. A per-frame bootstrap CI will understate true uncertainty. Resample at the **clip/event level** (block bootstrap), not the frame level.
>
> **Practical consequence:** budget for multi-seed training — the existing Limitations section already flags this for Table 4; extend it to also cover the two or three pruning ratios that bracket the cliff in Table 1, since that's the number the whole paper's headline claim rests on — and report event-level bootstrap CIs, not just point estimates, before treating any single "X% at 47% pruning" result as real rather than noise.

---

## 4. Experiments

*(Ablation Study & Proposed Method — Section 4 of the [Outline](#outline). 4.1 isolates each compression technique; 4.2 combines the winners into the deployable pipeline.)*

### 4.1 Benchmarking (Tables 1–3)

Each technique below is evaluated on its own, against an unmodified baseline, before anything is combined.

#### Table 1: Pruning Sweep

> [!TIP]
> The core framing here (structured over unstructured, coarse-then-fine sweep) is sound. Only the pruning criterion and the fine-tune budget strategy remain open — see the two NOTE boxes below.

> **Note:** All pruned models must be fine-tuned before evaluation. Unstructured pruning gives no real latency benefit without sparse-kernel support — treat it as a secondary/illustrative comparison at most. **Structured (channel) pruning** is the primary axis of this study, since that's what an actual edge-inference benefit depends on.

| Model | Pruning Ratio | Pruning Type | Criterion | Batch Size | Fine-tune Epochs | Params (M) | Storage (MB) | Latency (ms) | FPS | FLOPS | mAP | Recall | Precision | F-β | FN Rate % | Safety Pass? |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| YOLO11n | 0% (baseline) | - | - |  | - |  |  |  |  |  |  |  |  |  |  |  |
| YOLO11n | 20% |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| YOLO11n | 40% |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| YOLO11n | 60% |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| YOLO11n | 80% |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| YOLO26n | 0% (baseline) | - | - |  | - |  |  |  |  |  |  |  |  |  |  |  |
| YOLO26n | 20% |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| YOLO26n | 40% |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| YOLO26n | 60% |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| YOLO26n | 80% |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

*This coarse sweep (20/40/60/80%) is a starting point — add finer-grained rows (e.g. 45%, 50%, 55%) once it shows where Recall first drops below [X]%; pinpointing that cliff precisely is the actual research question.*

> [!NOTE]
> **Pruning criterion (currently `[TBD]`) — three real options:**
> 1. **L1-norm magnitude pruning** (Li et al., 2017). Simplest, most reproducible, easiest to justify in a methods section; judges each filter independently, so it can misjudge importance inside tightly-coupled blocks.
> 2. **BN-scale / Network Slimming** (Liu et al., ICCV 2017). Uses the learned batch-norm scaling factor as an importance proxy; needs a sparsity-inducing regularizer added to training first — an extra training-recipe decision this spec doesn't currently make.
> 3. **Torch-Pruning / DepGraph** (Fang et al., CVPR 2023) with a group-level importance criterion (e.g. `GroupNormPruner`). Automatically handles the cross-layer dependencies in YOLO's concat-heavy C2f/C3k2 necks that options 1–2 don't reason about — the more modern default, and confirmed to support YOLOv7/YOLOv8 directly. **YOLO11 and YOLO26 compatibility should be smoke-tested on a single checkpoint before committing**, since architecture-specific ops in the newer heads aren't guaranteed to be covered out of the box.
>
> **Leaning:** option 3, for the dependency-safety reason above, with the explicit smoke-test caveat.

> [!NOTE]
> **Fine-tune epoch budget across pruning ratios — three options:**
> 1. **Fixed epoch budget for every ratio.** Simplest, cleanest "fair comparison" argument, but heavier pruning plausibly needs more epochs to recover — a fixed low budget could make aggressive pruning look worse than it structurally is.
> 2. **Early-stopping on validation mAP/loss.** More realistic recovery estimate per ratio; "Fine-tune Epochs" stops being identical across rows and needs its own reporting convention (report epochs actually used).
> 3. **Budget scaled with pruning ratio** (e.g., roughly proportional to removed capacity). A middle ground seen in some pruning literature; needs a stated scaling rule to stay reproducible.
>
> **Leaning:** option 2 — the table already has a "Fine-tune Epochs" column, so reporting epochs-used costs nothing structurally, and recovery is the thing actually being measured. A fixed low budget risks *manufacturing* the cliff rather than discovering it.

#### Table 2: Quantization Comparison

> [!TIP]
> Parameters do not change with quantization — bit-width is tracked instead. **Hypothesis, worth testing exactly as framed:** QAT-INT8 may be slower than PTQ-INT8 on 'n' models due to tensor reformatting layers. Test this separately for YOLO11n (has DFL) and YOLO26n (DFL removed) — a recent architectural analysis of YOLO26 explicitly argues that DFL's softmax-over-bins regression is a known source of extra reformatting/dequantization overhead under INT8 export, and that removing it was a deliberate deployment-driven design choice, not an accident. If the QAT/PTQ gap shrinks or disappears on YOLO26n, that's a clean empirical confirmation worth a full paragraph in Discussion, not a footnote — this project may be one of the first to test that design claim rather than take it on faith.
>
> **mAP convention:** state once, use everywhere. Recommend **mAP@0.5**, matching the IoU=0.5 already fixed for the confidence-threshold operating-point procedure, rather than the stricter COCO-style mAP@[.5:.95]. Consistency matters more than which one is "more standard"; add the stricter version later as a secondary column if reviewers want it, but don't let two IoU conventions coexist across tables.
>
> **Runtime:** resolve once, in [Hardware & Benchmarking Protocol](#hardware--benchmarking-protocol) above — don't re-decide per table.

> [!CAUTION]
> **PTQ calibration set — both size *and* source are currently `[TBD]`, and source matters more than the placeholder suggests.** Source must be **train**, not validation or test — val is already spoken for by the confidence-threshold procedure, and reusing it for calibration muddies which split is doing what, even though it isn't classic leakage in the accuracy-inflation sense. Size: given the safety-critical class is likely rare, a naive random calibration sample risks containing few or zero examples of it, leaving exactly the channels that matter most poorly calibrated. **Stratify the calibration sample to guarantee a minimum count of safety-critical-class instances** — don't just take the first N images off disk.

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

> [!CAUTION]
> **Table 3 doesn't yet specify *how* KD transfers knowledge — this decision is missing from the Open Decisions Checklist entirely, not just left as a blank cell — and it is not architecture-neutral, which matters directly for this project's YOLO11-vs-YOLO26 framing:**
> 1. **Response/logit-based (classic).** Distill on final classification + box-regression outputs directly (Hinton et al., 2015, adapted to detection). Simplest to implement inside Ultralytics' existing loss; architecture-agnostic — identical for YOLO11 and YOLO26.
> 2. **Localization Distillation (LD).** Zheng et al., CVPR 2022 — distills localization knowledge through the box's soft probability-distribution representation, i.e. **it is built on the same distributional (DFL-style) head that YOLO26 removed.** LD is a natural, well-cited fit for YOLO11; it is **not directly applicable to YOLO26** without nontrivial adaptation, since YOLO26 regresses boxes directly rather than as a distribution. If LD (or anything DFL-shaped) is used for the YOLO11 arm of this table but a different method is used for YOLO26, the "does KD help YOLO11 vs. YOLO26 differently" comparison is confounded by *method* choice, not just by architecture — that needs to be stated explicitly in Discussion, not discovered by a reviewer.
> 3. **Feature-based (FGD or CWD).** Yang et al., CVPR 2022 (FGD) or Shu et al., ICCV 2021 (CWD) — distills intermediate feature maps rather than final outputs. Architecture-agnostic like option 1, but historically stronger than plain logit-matching for dense detectors overall — though LD's own results show logit/distribution mimicking beats feature imitation *specifically for localization*, so "feature-based is always better" isn't a safe assumption either. Pick based on what's actually being tested.
>
> **Recommended default, given the project's actual research question:** use the same **architecture-agnostic method (1 or 3)** for both YOLO11 and YOLO26, so the cross-architecture comparison in this table isn't confounded by KD-method choice. Treat LD-on-YOLO11 as an optional *bonus* row — clearly labeled as using a YOLO11-only technique — rather than the main comparison, if there's time to run it.

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
> **Order of Operations — current default is Prune → Fine-tune → Distill → Quantize (QAT), and the closest published analog to this exact pipeline disagrees with that ordering.** A 2026 CIFAR-scale study of the same three techniques (pruning, INT8 QAT, KD) as an ordered pipeline found pruning acts best as a *pre-conditioner*, INT8 QAT does the dominant latency work, and **KD works best applied last — after quantization — to recover accuracy specifically within the constrained sparse+INT8 regime**, not before it. That's the opposite of this table's current order. Three ways to handle it:
> 1. **Keep Prune → Distill → Quantize** (current). Defensible on the logic that a full-precision student can more easily absorb a full-precision teacher's soft labels before the extra noise of quantization is introduced — but state that reasoning explicitly now, since it's a claim being made *against* a specific published finding, not a neutral default.
> 2. **Switch to Prune → Quantize → Distill**, matching the closest related work — if the same ordering wins here too, that's mild external validation of this project's own results.
> 3. **Run both orderings as a small ablation** (even just on the single best-pruning-ratio checkpoint per architecture, not the full sweep) and report which one actually wins for detection. This turns a likely reviewer objection into a genuine, cheap, extra finding, and it's the option that best matches this project's own Core Question about compounding compression effects.
>
> Whichever is picked, **do not silently keep the current order without addressing this** — a reviewer familiar with the ordering literature will ask, and "we didn't consider it" is a weaker answer than any of the three options above.
>
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
- **Quantization Anomaly:** [e.g., QAT required for safety, but slower than PTQ on the edge device; check whether this holds for both YOLO11n and YOLO26n, or only the DFL-based one — see the citable hypothesis under Table 2.]
- **Hero Model:** [e.g., YOLO26n demonstrated superior robustness to compression; check whether this tracks with its small-object (STAL) design specifically on the phone class.]
- **Compression-Order Effect:** [Does Prune→Distill→Quantize or Prune→Quantize→Distill win for detection? The closest published analog found the latter for classification — confirm or refute for detection here, referencing the Table 4 CAUTION either way.]
- **KD-Method Confound:** [If a DFL-dependent method (LD) was used for YOLO11 but not YOLO26 in Table 3, say so explicitly here, before a reviewer finds it first.]

---

## 6. Limitations & Threats to Validity

- **Hardware generalizability.** All latency figures come from a desktop RTX 4060, not the eventual embedded automotive target. Conclusions about *relative* ranking between compression techniques are far more defensible than conclusions about *absolute* latency being production-ready.
- **Single-GPU compute budget.** State explicitly whether results are single-seed — a reasonable constraint at this project scale, but it should be stated rather than implied. Consider spending whatever multi-seed budget exists specifically on the final Table 4 configurations, since those carry the project's actual safety claim.
- **Dataset coverage after the 2026 revision.** No IR, no simulator data, a restricted subject pool. Any claim about robustness to night driving or low light is currently untestable on this data and should be explicitly scoped out of the conclusion rather than implied by omission.
- **Subject independence.** State the split explicitly; if it isn't subject-independent, recall numbers are optimistic.
- **Operating-point sensitivity.** Results depend on the confidence-threshold procedure above. A brief sensitivity check — how much Recall/FN Rate moves for a ±0.05 threshold shift — would meaningfully strengthen the safety claim.
- **Compression order not itself ablated.** The pipeline tests presence/absence of each stage in one fixed order (Prune → Fine-tune → Distill → Quantize). Whether that order is optimal — versus, say, quantizing before distilling, as the closest published analog found for classification — is addressed head-on in the Table 4 CAUTION rather than left implicit here.
- **Statistical power on the safety-critical class.** See [Statistical Power & Reporting Uncertainty](#statistical-power--reporting-uncertainty) — recall estimates on a rare, temporally-correlated class carry real uncertainty a bare point estimate hides. Any "cliff" claim should ship with a confidence interval, not just a table cell.
- **Detection granularity.** All reported Recall/FN Rate/F-β figures are frame-level unless explicitly stated otherwise (see [Detection Granularity](#detection-granularity-frame-level-vs-event-level-recall)). This is a real simplification relative to how a deployed DMS is actually scored, and the Conclusion should not imply an operational safety claim the frame-level metric can't support.
- **Camera-view scope.** Results are specific to whichever DMD camera view(s) are selected (see the Dataset CAUTION) and should not be assumed to transfer to a different camera placement without re-validation.
- **KD-method / architecture confound.** If different KD methods end up used for YOLO11 and YOLO26 in Table 3 (plausible, since Localization Distillation is DFL-dependent), any claim that "YOLO26 responds better/worse to KD than YOLO11" is confounded by method choice and must be scoped accordingly.

---

## Open Decisions Checklist
*(Resolve before Table 1's first cell gets filled in. Grouped by where the full reasoning lives — 🟡 = alternatives listed, pick one; 🔴 = blocking.)*

**Dataset** ([jump](#dataset))
- [ ] Safety-critical class list finalized, and micro- vs. macro-averaging decided if more than one class is in scope
- [ ] Train/val/test split confirmed subject-independent
- [ ] 🟡 Camera view(s) selected — face / hands / multi-view
- [ ] Frame-sampling strategy from video → image dataset documented (which frames carry spatial annotation, at what rate)
- [ ] 🔴 Current DMD Data Use Agreement confirmed against the 2026 revised terms

**Metric definition** ([jump](#definition-of-safe-threshold))
- [ ] X (recall threshold), Y (latency budget), Z (storage budget) set from literature review — not left as placeholders
- [ ] 🔴 Detection granularity chosen — frame-level / event-level / both
- [ ] β value for F-β chosen and stated once
- [ ] mAP IoU convention stated once (recommend @0.5, for consistency with the matching IoU already fixed)

**Protocol** ([jump](#hardware--benchmarking-protocol))
- [ ] Confidence-threshold selection procedure implemented and applied identically across all four tables
- [ ] 🟡 Inference runtime chosen — TensorRT / ONNX Runtime / OpenVINO (leaning TensorRT)
- [ ] Latency measurement protocol adopted (warm-up, N iterations, sync, thermal control)
- [ ] 🔴 PTQ calibration set source (train, not val/test) and size fixed, stratified for the safety-critical class

**Techniques** ([jump](#table-1-pruning-sweep))
- [ ] 🟡 Pruning criterion named — L1-norm / BN-scale / Torch-Pruning+DepGraph (leaning Torch-Pruning)
- [ ] 🟡 Fine-tune epoch-budget strategy across pruning ratios decided (leaning early-stopping)
- [ ] 🔴 KD method named, and its architecture-neutrality (or lack of it) stated explicitly
- [ ] Online vs. offline KD decision made and stated
- [ ] 🔴 Compression-stage order for Table 4 confirmed against the literature's contradicting finding

**Rigor** ([jump](#statistical-power--reporting-uncertainty))
- [ ] Multi-seed plan set (at minimum: the pruning ratios bracketing the cliff in Table 1, and every row of Table 4)
- [ ] Confidence-interval method for Recall chosen (Wilson/Jeffreys; clip-level bootstrap, not frame-level)

---

## References

**Dataset & standards**
- Ultralytics, YOLO26 model documentation: https://docs.ultralytics.com/models/yolo26
- Vicomtech, DMD Driver Monitoring Dataset (incl. 2026 data-availability update): https://github.com/Vicomtech/DMD-Driver-Monitoring-Dataset
- Ortega et al., "DMD: A Large-Scale Multi-Modal Driver Monitoring Dataset for Attention and Alertness Analysis" (2020): https://arxiv.org/pdf/2008.12085
- Smart Eye, on Euro NCAP's 2026 driver monitoring changes: https://smarteye.se/blog/euro-ncap-2026-whats-changing/ and https://smarteye.se/blog/driver-monitoring-euro-ncap-2026/
- Smart Eye, on DDAW vs. ADDW and the EU GSR timeline: https://smarteye.se/?p=12321
- Anyverse, on EU GSR and Euro NCAP in-cabin monitoring mandates: https://anyverse.ai/in-cabin-monitoring-navigating-europes-safe-driving-new-standards-3/
- Jama Software / UL Solutions, on ISO 21448 (SOTIF) and ISO/PAS 8800: https://www.jamasoftware.com/requirements-management-guide/automotive-engineering/sotif/ and https://www.ul.com/sis/blog/safety-related-systems-road-vehicles-artificial-intelligence-are-addressed-isopas-88002024

**Closest related work (compression pipelines & automotive edge KD)**
- "Prune-Quantize-Distill: An Ordered Pipeline for Efficient Neural Network Compression" (2026): https://arxiv.org/pdf/2604.04988 — CIFAR classification only; found Prune→Quantize→Distill outperforms other orderings, contradicting this doc's current Table 4 default.
- Karjol & Hanna, "Edge AI for Automotive Vulnerable Road User Safety: Deployable Detection via Knowledge Distillation": https://arxiv.org/pdf/2604.26857 — automotive KD + INT8 for external (VRU) detection, not in-cabin.

**Safety- and cost-weighted detection metrics**
- "Evaluating Object (mis)Detection from a Safety and Reliability Perspective: Discussion and Measures": https://arxiv.org/pdf/2203.02205 — proposes a criticality-weighted recall.
- "Cost-Sensitive Uncertainty-Based Failure Recognition for Object Detection" (UAI 2024): https://arxiv.org/abs/2404.17427
- "Managing Cost–Stability Trade-Offs in Industrial Object Detection: A Unified Decision Support Framework" (2026): https://doi.org/10.3390/a19050409

**Knowledge distillation for detection heads**
- Hinton, Vinyals & Dean, "Distilling the Knowledge in a Neural Network" (2015): arXiv:1503.02531
- Zheng et al., "Localization Distillation for Dense Object Detection" (CVPR 2022): https://arxiv.org/abs/2102.12252
- Yang et al., "Focal and Global Knowledge Distillation for Detectors" (CVPR 2022): https://openaccess.thecvf.com/content/CVPR2022/html/Yang_Focal_and_Global_Knowledge_Distillation_for_Detectors_CVPR_2022_paper.html
- Shu et al., "Channel-Wise Knowledge Distillation for Dense Prediction" (ICCV 2021): https://openaccess.thecvf.com/content/ICCV2021/html/Shu_Channel-Wise_Knowledge_Distillation_for_Dense_Prediction_ICCV_2021_paper.html

**Pruning & YOLO26 architecture analyses**
- Li, Kadav, Durdanovic, Samet & Graf, "Pruning Filters for Efficient ConvNets" (2017): arXiv:1608.08710
- Liu, Li, Shen, Huang, Yan & Zhang, "Learning Efficient Convolutional Networks through Network Slimming" (ICCV 2017)
- Fang et al., "DepGraph: Towards Any Structural Pruning" (CVPR 2023) / Torch-Pruning: https://pypi.org/project/torch-pruning/
- YOLO26 architectural analysis connecting DFL removal to quantization-friendliness: https://arxiv.org/html/2605.24831v2
- Chakrabarty, "YOLO26: An Analysis of NMS-Free End-to-End Framework for Real-Time Object Detection": https://arxiv.org/pdf/2601.12882v2

**Adjacent driver-distraction-YOLO literature (motivates the gap)**
- Example of the "compression as stated future work" pattern common in this space: https://pmc.ncbi.nlm.nih.gov/articles/PMC10649436/