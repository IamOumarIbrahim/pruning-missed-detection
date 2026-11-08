# PROTOCOL.md: Experimental Protocol Specification
**Status:** DRAFT — NOT FROZEN (Awaiting formal user approval via 'APPROVED:').
**Repository:** `IamOumarIbrahim/pruning-missed-detection`

---

## 1. Experimental Overview & Pre-Conditions

This protocol establishes prospective execution rules for investigating whether structured channel pruning in lightweight edge object detectors causes catastrophic tail-class safety recall degradation that is obscured by standard aggregate benchmark metrics ($\text{mAP}_{50}$).

### 1.1 Strict Ground Rules
1. **Execution Freeze:** No GPU fine-tuning, training, model downloading, or dataset restructuring shall occur until this protocol is finalized, approved by the user with an explicit chat message beginning "APPROVED:", and committed. UI clicks, tool approvals, or automated approvals are void.
2. **Spent Test Split Status:** The local test partition (`subject_05`, `subject_10`, `subject_12`) has been used for exploratory threshold sweeps during preliminary audits. Under this protocol, all evaluations on this split are permanently designated as **exploratory**. Confirmatory claims require unseen subjects or prospective cross-validation (Route RESTART).
3. **Statistical Integrity:** No single-seed claims or uncorrected $t$/Welch statistics shall be reported for small sample sizes ($n=3$). All evaluation uncertainty intervals must use **video-cluster bootstrap** (and subject-cluster bootstrap where applicable) conditioning on the trained checkpoints.
4. **Canonical Metric Standard:** All threshold-dependent recalls must use the exact canonical step function from cached detections ($\text{matched GT at conf} \ge \tau / \text{GT}$). Linear interpolation (`r_curve`, `np.interp`) and max-$F_1$ heuristics (`box.r`) are prohibited.
5. **Rewritten Core Thesis:**
   > *"Lightweight driver-monitoring object detectors exhibit severe out-of-sample safety guardrail failure on rare tail behaviors under validation-tuned confidence thresholds—not primarily from structured pruning representation collapse up to 50% sparsity, but from winner's-curse boundary selection and extreme inter-subject event sparsity that undermine single-threshold transfer."*

---

## 2. Reviewer Comment -> Status -> Evidence Mapping

| Reviewer Comment / Hypothesis | Current Status | Primary Empirical Evidence | Remaining Confounders / Caveats |
|:---|:---|:---|:---|
| **R1: "Pruning selectively destroys tail-class representation (concealment)"** | **UNDERPOWERED** | Dose-response 0%–50% shows paired $\Delta \text{min-AP}_{50}$ mean of $-0.26\text{ pp}$ (YOLO11n) and $-1.08\text{ pp}$ (YOLO26n), well within baseline seed SD ($2.19\text{ pp}$). ANOVA $\eta^2$ ($33\%, 25\%$) does not exceed null sampling noise ($26.3\%, 29.4\%$). | Regimes $\ge 60\%$ unmeasured. Sample size $n=3$ seeds has evaluation MDE $= 14.53\text{ pp}$. |
| **R2: "Validation-tuned threshold $\tau^*$ fails out-of-sample"** | **SUPPORTED** | Across 8 complete baselines, $\tau^*$ passes only 2/8 (25%) on `best.pt` and 0/8 (0%) on `last.pt`, with mean margin $-16.49\text{ pp}$. In contrast, deployment default $\tau=0.25$ passes 8/8 (100%) on `best.pt` and 6/8 (75%) on `last.pt`. | Winner's-curse boundary selection vs val-to-test confidence shift are NOT separated due to floor anchoring at $R_{\text{base}} - 5\text{ pp}$. |
| **R3: "Inter-subject confidence shift breaks calibration"** | **UNDERPOWERED** | Minimum 2-sided permutation $p$-value for $3+3$ subjects is $2/\binom{6}{3} = 0.10$. Confidence quantiles overlap baseline seed variance. | Confounded with event-level noise (val floor set by 7 `hand_over_mouth` events; test tail dominated by `subject_12`). |
| **R4: "Aggregate mAP conceals worst-class safety degradation"** | **UNDERPOWERED** | Min-class AP50 tracks mAP50 within $\pm 1.5\text{ pp}$ across 0%–50% ratios. At $\tau=0.25$, worst-class recall tracks baseline within $\pm 2.6\text{ pp}$. | Concealment may emerge at extreme ratios ($>50\%$) where capacity is exhausted. |
| **R5: "Checkpoint selection (`best.pt` vs `last.pt`) drives compliance"** | **SUPPORTED** | Switching `best.pt` to `last.pt` reduces $\tau^*$ compliance from 2/8 to 0/8, shifting mean test margin from $-16.49\text{ pp}$ to $-19.18\text{ pp}$. | Baseline sample size $n=2$ models $\times$ seeds; primarily reflects validation overfitting of `best.pt`. |

---

## 3. Prospective Route Specifications

### ROUTE CONTINUE: Sequential Deep-Pruning Screen (60% to 90%, Seed 0)
- **Goal:** Determine whether catastrophic representation collapse and metric concealment emerge at extreme sparsity ($60\%\text{--}90\%$).
- **Design:** Sequential screening on seed 0: ratios $\in \{60\%, 70\%, 80\%, 90\%\}$ for YOLO11n and YOLO26n (8 runs total).
- **Compute Budget:** ~17 GPU hours ($8 \times \sim 2.1\text{ h}$).
- **Training Recipe:** Identical to baseline seed-0 runs (identical `args.yaml` diff must be empty); fine-tuned from same-seed baseline for 100 epochs.
- **Checkpointing:** Cache `best.pt` (primary, consistent with existing 0%–50% grid) and `last.pt` (sensitivity).
- **Stopping / Collapse Criterion (Data-Derived):**
  - Training halts for an architecture branch at the first occurrence of:
    1. Loss divergence, infinite loss, or NaN.
    2. Model collapse: all-class $\text{mAP}_{50}$ drops by $> 30.0\text{ pp}$ relative to baseline.
    3. Severe capacity collapse: $\Delta \text{min-class AP}_{50} \le -14.53\text{ pp}$ (exceeding paired evaluation MDE of $14.53\text{ pp}$).
  - Higher ratios for that model branch are cancelled immediately upon collapse.
- **Flagging Rule for Multi-Seed Expansion (Capped at 20 GPU-h):**
  - Expand to $K=3$ seeds ($s \in \{1, 2\}$) ONLY at a specific ratio $r$ if:
    - **Concealment:** $\Delta \text{mAP}_{50} \ge -3.0\text{ pp}$ AND $\Delta \text{Rec}_{P80} \le -6.57\text{ pp}$ ($3 \times \text{seed SD}$).
    - **Safety Knee:** $\text{Rec}_{P80}(r) - \text{Rec}_{P80}(r - 10\%) \le -14.53\text{ pp}$ (paired evaluation MDE).
  - Maximum extra compute capped at 20 GPU hours before requesting user approval.
  - Do NOT run the 61-run $K=5$ queue. Do NOT resume YOLO26n seed 3.
- **Exact Launch Command:**
  ```bash
  python scripts/run_route_a_stage1.py
  ```
- **What Outcomes Support / Refute:**
  - *If concealment observed at $\ge 60\%$:* Supports that pruning conceals tail-class degradation, but bounds the phenomenon to extreme parameter regimes ($>50\%$).
  - *If smooth degradation or collapse across all metrics:* Refutes the concealment hypothesis; establishes that edge detectors degrade gracefully or collapse globally.

---

### ROUTE RESTART: Subject-Balanced 7-Fold Cross-Validation (Baselines Only)
- **Goal:** Prospective, unbiased measurement of out-of-sample guardrail transfer on unspent subjects with balanced tail events.
- **Design:** 14 DMD subjects partitioned into 7 folds of 2 held-out subjects each.
- **Tail-Balanced Pairing:**
  - `subject_10` (0 `hand_over_mouth`, 1 `yawning`) is paired with tail-rich `subject_12` (22 `hand_over_mouth`, 24 `yawning`).
  - Remaining pairs balanced so every fold contains $\ge 4$ tail events:
    - Fold 1: `(subject_10, subject_12)` (22 hom, 25 yawn)
    - Fold 2: `(subject_02, subject_05)` (12 hom, 16 yawn)
    - Fold 3: `(subject_03, subject_11)` (24 hom, 24 yawn)
    - Fold 4: `(subject_01, subject_04)` (balanced)
    - Fold 5: `(subject_06, subject_07)` (balanced)
    - Fold 6: `(subject_08, subject_09)` (balanced)
    - Fold 7: `(subject_13, subject_14)` (balanced)
  - Total events available across 14 subjects: 63 `yawning`, 44 `hand_over_mouth`, 57 `drinking`, 58 `phone_use`.
- **Training Recipe:**
  - Train on 12 subjects, 100 epochs, identical training arguments.
  - **`last.pt` ONLY:** No validation-based checkpoint selection (`val` used strictly for logging).
  - Execute YOLO11n (7 folds $\approx 15\text{ GPU-h}$), then YOLO26n (7 folds $\approx 15\text{ GPU-h}$). Total baselines compute: ~30 GPU hours.
  - Adding pruned folds (~31 GPU-h) requires separate user approval.
- **Evaluation Protocol:**
  - For each fold: calibrate rules on the 6 other folds' out-of-fold pooled detections; evaluate on the held-out fold.
  - Frozen rule menu:
    1. Fixed $\tau = 0.25$
    2. Max-$\tau$ (unbuffered)
    3. Buffered max-$\tau$ ($b \in \{2.5, 5.0, 7.5\}\text{ pp}$)
    4. Plateau rule (descending grid)
    5. Clopper-Pearson rule
  - Disclose in paper that rule menu was motivated by Phase 0 exploratory audit on spent test split.
  - Report estimates with 95% bootstrap intervals, not deterministic guarantees.
- **Exact Launch Command:**
  ```bash
  python scripts/train_7fold_cv.py --model yolo11n --checkpoint last
  ```
- **What Outcomes Support / Refute:**
  - *If $\tau^*$ compliance remains low across balanced folds:* Proves conclusively that winner's-curse boundary selection is an intrinsic property of the calibration rule, independent of the local test split.
  - *If buffered / fixed rules achieve $>90\%$ compliance across folds:* Establishes a validated, publishable deployment standard for safety guardrails in safety-critical edge vision.

---

## 4. Execution Governance & Decision Gate

Upon delivery of Phase A audit report:
1. User reviews the Phase A report and decides between **Route CONTINUE** and **Route RESTART**.
2. Training remains strictly locked by `ALLOW_TRAINING` and `STATUS: FROZEN-BY-USER <hash>`.
3. To unlock, user issues chat message:
   `APPROVED: CONTINUE` or `APPROVED: RESTART` followed by `APPROVED: FREEZE <commit-hash>`.
