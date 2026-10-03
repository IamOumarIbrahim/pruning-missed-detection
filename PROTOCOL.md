# PROTOCOL.md: Experimental Protocol Specification
**Status:** DRAFT — not yet committed as final (pending Phase 0 review and user approval)
**Repository:** `IamOumarIbrahim/pruning-missed-detection`
**Target Commit Hash:** [To be recorded upon formal freezing before any Phase 1 execution]

---

## 1. Experimental Overview & Pre-Conditions

This protocol establishes the prospective, preregistered execution rules for investigating whether structured channel pruning in lightweight edge object detectors causes catastrophic tail-class safety recall degradation that is obscured by standard aggregate benchmark metrics ($\text{mAP}_{50}$).

### 1.1 Strict Ground Rules
1. **Execution Freeze:** No GPU fine-tuning, training, model downloading, or dataset restructuring shall occur until this protocol is finalized, approved by the user, and committed with an immutable Git commit hash.
2. **Spent Test Split Status:** The local test partition (`subject_05`, `subject_10`, `subject_12`) has been used for exploratory threshold sweeps during preliminary audits. Under this protocol, all evaluations on this split are designated as **exploratory**. Confirmatory claims require unseen subjects (Route B) or strict out-of-sample cross-validation.
3. **Statistical Integrity:** No single-seed claims or uncorrected $t$/Welch statistics shall be reported for $n=3$ seeds. All evaluation uncertainty intervals must use **video-cluster bootstrap** (and subject-cluster bootstrap where applicable) conditioning on the trained checkpoints.

---

## 2. Route A: Extended Pruning Dose-Response (60% to 90%)

Route A tests the hypothesis that pushing structured pruning beyond 50% into extreme regimes (60%, 70%, 80%, 90%) reveals the "knee" where worst-class safety recall collapses while aggregate $\text{mAP}_{50}$ remains comparatively stable.

### 2.1 Stage 1: Sequential Screening
To minimize wasted GPU compute, Stage 1 evaluates a single seed ($s=0$) sequentially across extreme pruning ratios:
$$\text{Ratio Grid: } r \in \{60\%, 70\%, 80\%, 90\%\} \quad \text{for } M \in \{\text{YOLO11n}, \text{YOLO26n}\}$$

- **Execution Order:** Train sequentially $60\% \to 70\% \to 80\% \to 90\%$.
- **Early Stopping Rule:** If fine-tuning at ratio $r$ fails (loss divergence / NaN / infinite loss), or if all-class $\text{mAP}_{50}$ drops by more than $30\text{ pp}$ relative to baseline (complete model collapse), training halts for that architecture, and higher ratios for that model are not executed.
- **Reporting Commitment:** The full dose-response curve (0% to the stopping point) will be reported in full, regardless of whether the hypothesis is supported or refuted.

### 2.2 Stage 1 Screening Criteria (Data-Derived Thresholds)
Rather than arbitrary round numbers, screening criteria are grounded in the baseline seed standard deviation measured during Phase 0:
- Baseline seed standard deviation:
  - $\text{SD}(\text{min-class AP}_{50}) \approx 3.4\text{ pp}$ (YOLO26n) to $1.2\text{ pp}$ (YOLO11n).
  - Minimum detectable effect at $\alpha = 0.05$ with single seed screening requires $\ge 2 \times \text{SD}_{\text{baseline}}$.
- **Flagging Criteria for Stage 2 Seed Expansion:**
  A ratio $r$ is flagged for multi-seed expansion ($s \in \{1, 2, 3, 4\}$) if either:
  1. **Concealment Pattern:** $\Delta \text{mAP}_{50} \ge -3.0\text{ pp}$ (aggregate metric stable), while $\Delta \text{Worst-Class Recall} \le -7.5\text{ pp}$ (significant tail safety drop exceeding $2 \times \text{SD}$).
  2. **Safety Knee:** $\text{Worst-Class Recall}(r) - \text{Worst-Class Recall}(r - 10\%) \le -10.0\text{ pp}$ (steep slope discontinuity).

### 2.3 Stage 2: Confirmatory Seed Expansion
- For any ratio $r$ meeting the Stage 1 flagging criteria, expand to $K=5$ seeds ($s \in \{0, 1, 2, 3, 4\}$) using identical hyperparameter seeds.
- If no ratio meets the flagging criteria, report the full descriptive dose-response curve as an empirical null result on concealment.

### 2.4 Primary Endpoints for Route A
1. **Worst-Class Recall at Matched Precision = 0.80 ($\text{Rec}_{P80}$)**: Threshold-free operational metric.
2. **Min-Class $\text{AP}_{50}$**: Area under precision-recall curve for the worst-performing class.
3. **Aggregate $\text{mAP}_{50}$ and $\text{mAP}_{50\text{--}95}$**: Standard COCO metrics.
4. **Calibration Drift Metrics:** Median TP confidence, 10th percentile TP confidence, and $\tau_{80}$ per class.
5. **Infeasible-Floor Handling:** If a pruned model cannot achieve $\text{Recall}_c \ge R_{\text{floor}}$ at any $\tau \ge 0.01$, record explicitly as `Feasible = False`, `tau_star = None`, rather than clamping to 0.01.

---

## 3. Route B: Confirmatory Safety Guardrail Protocol

Route B rigorously evaluates operational safety guardrails ($\tau^*$ selection and fixed operating points) under strict out-of-sample validation to prevent winner's curse boundary collapse.

### 3.1 Candidate Threshold Selection Rules
1. **Canonical Fixed Threshold:** $\tau = 0.25$ (deployment baseline).
2. **Max-$\tau$ with Margin $\delta$ (LOSO-Optimized):**
   $$\tau^* = \max \left\{ \tau \in [0.01, 0.90] : \min_{c} \text{Recall}_c^{\text{val}}(\tau) \ge R_{\text{base}}^{\text{val}} - \delta \right\}$$
   Where $\delta \in \{10\text{ pp}, 15\text{ pp}\}$ is chosen to guarantee positive test margin with $\ge 90\%$ probability based on LOSO cross-validation.
3. **Clopper-Pearson Exact Binomial Lower Bound Rule:**
   For each class $c$ with $N_c$ validation events and $K_c$ detected events:
   Compute the exact $95\%$ lower confidence bound $p_{\text{lower}}(c, \tau)$.
   Select $\tau^*_{\text{CP}} = \max \{ \tau : \min_c p_{\text{lower}}(c, \tau) \ge R_{\text{floor}} \}$.

### 3.2 Evaluation Options for Route B

#### Option B.1: External Unseen Subjects (Zero Retraining)
- **Data Source:** Official unrestricted DMD subjects $\{23, 28, 29, 33, 36, 37\}$ (6 subjects not present in local training/val/test splits).
- **Procedure:** Obtain RGB face recordings, generate ground-truth cue bounding boxes following the standard DMS-Eval Label Studio protocol, and evaluate existing 36 sweep checkpoints.
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
- Therefore, within-5-pp claims on tail classes in the current test set are fundamentally **statistically underpowered** due to event-count limits ($N < 20$). Route B explicitly documents this resolution limit.

---

## 4. Execution Governance & Decision Gate
Upon completion of Phase 0:
1. User reviews the Phase 0 report and selects **Route A**, **Route B**, or **A + B**.
2. Upon user selection, this `PROTOCOL.md` is finalized, the git commit hash is recorded, and execution begins.
