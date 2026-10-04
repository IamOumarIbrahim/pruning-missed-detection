# PROTOCOL.md: Experimental Protocol Specification
**Status:** DRAFT — NOT FROZEN (Awaiting formal user approval via 'APPROVED:').
**Repository:** `IamOumarIbrahim/pruning-missed-detection`

---

## 1. Experimental Overview & Pre-Conditions

This protocol establishes the prospective execution rules for investigating whether structured channel pruning in lightweight edge object detectors causes catastrophic tail-class safety recall degradation that is obscured by standard aggregate benchmark metrics ($\text{mAP}_{50}$).

### 1.1 Strict Ground Rules
1. **Execution Freeze:** No GPU fine-tuning, training, model downloading, or dataset restructuring shall occur until this protocol is finalized, approved by the user with an explicit chat message beginning "APPROVED:", and committed. UI clicks, tool approvals, or automated approvals are void.
2. **Spent Test Split Status:** The local test partition (`subject_05`, `subject_10`, `subject_12`) has been used for exploratory threshold sweeps during preliminary audits. Under this protocol, all evaluations on this split are designated as **exploratory**. Confirmatory claims require unseen subjects (Route B) or strict out-of-sample cross-validation.
3. **Statistical Integrity:** No single-seed claims or uncorrected $t$/Welch statistics shall be reported for small sample sizes. All evaluation uncertainty intervals must use **video-cluster bootstrap** (and subject-cluster bootstrap where applicable) conditioning on the trained checkpoints.
4. **Canonical Metric Standard:** All threshold-dependent recalls must use the exact canonical step function from cached detections ($\text{matched GT at conf} \ge \tau / \text{GT}$). Linear interpolation (`r_curve`, `np.interp`) and max-$F_1$ heuristics (`box.r`) are prohibited.

---

## 2. Route A: Extended Pruning Dose-Response (60% to 90%)

Route A tests the hypothesis that pushing structured pruning beyond 50% into extreme regimes (60%, 70%, 80%, 90%) reveals the "knee" where worst-class safety recall collapses while aggregate $\text{mAP}_{50}$ remains comparatively stable.

### 2.1 Stage 1: Sequential Screening
To minimize wasted GPU compute, Stage 1 evaluates a single seed ($s=0$) sequentially across extreme pruning ratios:
$$\text{Ratio Grid: } r \in \{60\%, 70\%, 80\%, 90\%\} \quad \text{for } M \in \{\text{YOLO11n}, \text{YOLO26n}\}$$

- **Execution Order:** Train sequentially $60\% \to 70\% \to 80\% \to 90\%$.
- **Early Stopping Rule (Empirically Derived):** If fine-tuning at ratio $r$ fails (loss divergence / NaN / infinite loss), or if all-class $\text{mAP}_{50}$ drops by more than $30.0\text{ pp}$ relative to baseline (complete model collapse, matching the evaluation MDE of $30.02\text{ pp}$), training halts for that architecture, and higher ratios for that model are not executed.
- **Reporting Commitment:** The full dose-response curve (0% to the stopping point) will be reported in full, regardless of whether the hypothesis is supported or refuted.

### 2.2 Stage 1 Screening Criteria (Data-Derived from Step 2 Audit)
Screening criteria are derived from verified baseline seed variance ($K=5$ YOLO11n seed SD $= 3.01\text{ pp}$ at fixed $\tau=0.25$; $\text{Rec}_{P80}\text{ SD} = 1.86\text{ pp}$) and nested bootstrap evaluation uncertainty ($\text{SE} = 10.72\text{ pp}$, $\text{MDE} = 30.02\text{ pp}$):
- **Flagging Criteria for Stage 2 Seed Expansion:**
  A ratio $r$ is flagged for multi-seed expansion ($s \in \{1, 2, 3, 4\}$) if either:
  1. **Concealment Pattern:** Aggregate $\Delta \text{mAP}_{50} \ge -3.0\text{ pp}$ (aggregate metric stable), while worst-class threshold-free recall drops by $\Delta \text{Rec}_{P80} \le -10.0\text{ pp}$ (exceeding $3 \times \text{baseline seed SD}$).
  2. **Safety Knee:** $\text{Rec}_{P80}(r) - \text{Rec}_{P80}(r - 10\%) \le -15.0\text{ pp}$ (slope discontinuity exceeding half of the evaluation MDE).

### 2.3 Stage 2: Confirmatory Seed Expansion
- For any ratio $r$ meeting the Stage 1 flagging criteria, expand to $K=5$ seeds ($s \in \{0, 1, 2, 3, 4\}$) using identical hyperparameter seeds.
- If no ratio meets the flagging criteria, report the full descriptive dose-response curve as an empirical null result on concealment.

### 2.4 Primary Endpoints for Route A
1. **Primary Endpoint:** Worst-Class Recall at Matched Precision = 0.80 ($\text{Rec}_{P80}$), isolating representation capacity from score calibration.
2. **Secondary Endpoint:** Worst-Class Recall at Matched Precision = 0.90 ($\text{Rec}_{P90}$).
3. **Threshold-Free Ranking:** Min-Class $\text{AP}_{50}$ and Macro $\text{mAP}_{50}$.
4. **Calibration Drift Metrics:** Median TP confidence, 10th percentile TP confidence, and $\tau_{80}$ per class.
5. **Infeasible-Floor Handling:** If a pruned model cannot achieve $\text{Recall}_c \ge R_{\text{floor}}$ at any $\tau \ge 0.01$, record explicitly as `Feasible = False`, `tau_star = None`, rather than clamping to 0.01.

---

## 3. Route B: Confirmatory Safety Guardrail Protocol

Route B rigorously evaluates operational safety guardrails ($\tau^*$ selection and fixed operating points) under strict out-of-sample validation to prevent winner's curse boundary collapse.

### 3.1 Candidate Threshold Selection Rules
1. **Canonical Fixed Threshold:** $\tau = 0.25$ (deployment baseline; empirical test pass rate $= 8/8$ on `best.pt`, $6/8$ on `last.pt`).
2. **Buffered Max-$\tau$ Rule:**
   $$\tau^*_b = \max \left\{ \tau \in \mathcal{C}^{\text{val}} : \min_{c} \text{Recall}_c^{\text{val}}(\tau) \ge R_{\text{floor}} + b \right\}$$
   Where $b = 7.5\text{ pp}$ guarantees $\ge 90\%$ test compliance out-of-sample by forcing $\tau^*$ to retreat into the high-recall plateau ($\tau^* \approx 0.05\text{--}0.15$).
3. **Clopper-Pearson Exact Event Rule:**
   $R_{\text{floor}} = 0.05^{1/N_{\text{events}}}$. (Note: In Phase 0 audit, this yielded $1/8$ pass rate due to severe winner's curse when events $N \le 7$).

### 3.2 Evaluation Options for Route B

#### Option B.1: External Unseen Subjects (Zero Retraining)
- **Data Source:** Official unrestricted DMD subjects $\{23, 28, 29, 33, 36, 37\}$ (6 subjects not present in local training/val/test splits).
- **Procedure:** Obtain RGB face recordings, generate ground-truth cue bounding boxes following standard DMD annotation protocol, and evaluate existing 36 sweep checkpoints.
- **Compute Cost:** 0 GPU training hours (inference and evaluation only, ~2 hours).
- **Statistical Power:** Provides 6 completely pristine, out-of-distribution confirmatory subjects.

#### Option B.2: Subject-Level Cross-Validation (K-Fold LOSO)
- If external subjects are not annotated, execute 5-fold subject-disjoint cross-validation across the 14 local subjects.
- **Compute Cost:** 5 folds $\times$ 6 ratios $\times$ 2 models $\times$ 2.2 h $\approx 132$ GPU hours.

### 3.3 Sample Size & Event Resolution Requirements
For a statistical claim that test recall is "within 5 pp" of validation recall at $90\%$ recall:
- Standard error of sample proportion: $\text{SE} = \sqrt{\frac{p(1-p)}{N}} = \sqrt{\frac{0.90 \times 0.10}{N}} \le 0.025$ ($95\%\text{ CI width} \le 10\text{ pp}$).
- Requires $N \ge \frac{0.09}{0.000625} = 144$ independent events per class.
- Current test set contains only:
  - `yawning`: 18 events (14 on subject_12)
  - `hand_over_mouth`: 15 events (13 on subject_12)
- Therefore, within-5-pp claims on tail classes in the current test set are fundamentally **statistically underpowered** due to event-count limits ($N < 20$).

---

## 4. Route C: Matched-Precision Edge Safety Evaluation (Zero Retraining)

Route C decouples confidence calibration from representation collapse by benchmarking all existing 36 models at matched operating precisions without running any GPU training.

### 4.1 Objectives & Endpoints
- **Primary Hypothesis:** Pruning preserves representation ranking ($AP_{50}$ within $\pm 1.5\text{ pp}$) and matched-precision recall ($\text{Rec}_{P80}$ within $\pm 3.0\text{ pp}$), demonstrating that observed fixed-$\tau$ variations reflect threshold rigidity rather than capacity loss.
- **Compute Cost:** 0 GPU training hours (completed in Phase 0 audit cache).
- **Deliverables:** Matched operating point curves ($\text{Rec}_{P80}$, $\text{Rec}_{P90}$, $\text{Rec}_{\text{FP05}}$) and per-subject forensic error breakdown.

---

## 5. Route R: Conformal & Subject-Adaptive Guardrail Protocol

Route R resolves the out-of-sample guardrail failure identified in Phase 0 by replacing rigid single thresholds with formal distribution-free risk control.

### 5.1 Formulation
- Instead of boundary-clamped point thresholds ($\tau^*$), apply split conformal prediction or learn-then-test calibration on validation subjects to guarantee a user-specified bound on the false negative rate:
  $$P(\text{Recall}_c \ge 1 - \alpha) \ge 1 - \delta$$
- Evaluates subject-level calibration adaptation using the driver's first 30 seconds of driving video.

---

## 6. Execution Governance & Decision Gate

Upon completion of Phase 0:
1. User reviews the Phase 0 consolidated deliverable and selects the prospective route (**Route A**, **Route B**, **Route C**, **Route R**, or a combination).
2. To unlock execution, the user must provide an explicit chat confirmation starting with `"APPROVED:"`.
3. Upon approval, `STATUS` in this document is updated to `FROZEN-BY-USER <git-hash>`, the lock in `scripts/run_route_a_stage1.py` disengages, and execution proceeds.
