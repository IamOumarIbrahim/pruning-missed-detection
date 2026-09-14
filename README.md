# pruning-missed-detection

> **The Core Question:** "How much can we prune an object detector before it becomes a safety risk? And does it provide any significant inference benefits for edge-deployed DMS over other compression techniques?"

## Outline

1. Introduction
DMS on edge devices has strict latency and safety constraints.

2. Related Work
We discuss pruning, quantization, and KD separately (and note that most papers only test one).

3. Methodology
We explain our safety-weighted F1 metric and our FN-penalty threshold.

4. Experiments
- 4.1 Benchmarking: Tables 1, 2, and 3. "Here is what each technique does in isolation."
- 4.2 Proposed Method: Table 4. "Here is how we combine them to build a deployable DMS model."

5. Discussion
We analyze the trade-offs. We mention the QAT vs. PTQ anomaly on "n" models. We discuss the compounding errors.

6. Conclusion
The combined pipeline provides the best safety-to-latency ratio for edge DMS.

### Dataset 
DMD 

### Definiton of SAFE threshold
A pruned-model is deemed safe if it passes the following criteria:
1. Recall >= [X]% for safety-critical classes (Anchor to ISO 26262 / Euro NCAP)
2. Inference Latency <= [Y] ms on target edge device
3. Storage <= [Z] MB

> The finalized criteria will be conducted via literature review and anchored to automotive standards.

The final weighted F1 is [TBD] (False negatives penalized [5x/10x] more than false positives)

---

## Ablation Study

### Table 1: Pruning Sweep

> **Note:** All pruned models must be fine-tuned before evaluation. Unstructured pruning may not reduce latency on standard GPUs.

| Model | Pruning Ratio | Pruning Type | Fine-tune Epochs | Params (M) | Storage (MB) | Latency (ms) | FPS | FLOPS | mAP | Recall | Precision | Weighted F1 | FN Rate % | Safety Pass? |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| YOLO11n | 0% | - | - | | | | | | | | | | | |
| YOLO11n | R% | | | | | | | | | | | | | |
| YOLO11n | 2R% | | | | | | | | | | | | | |
| YOLO11n | 3R% | | | | | | | | | | | | | |
| YOLO26n | 0% | - | - | | | | | | | | | | | |
| YOLO26n | R% | | | | | | | | | | | | | |
| YOLO26n | 2R% | | | | | | | | | | | | | |
| YOLO26n | 3R% | | | | | | | | | | | | | |

---

### Table 2: Quantization Comparison

> **Note:** Parameters do not change with quantization. We track bit-width instead. 
> **Expected Finding:** QAT-INT8 may be slower than PTQ-INT8 on 'n' models due to tensor reformatting layers.

| Model | Precision | Weight Bit-Width | Storage (MB) | Latency (ms) | FPS | mAP | Recall | Precision | Weighted F1 | FN Rate % | Safety Pass? |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| YOLO11n | FP32 | 32 | | | | | | | | | |
| YOLO11n | FP16 | 16 | | | | | | | | | |
| YOLO11n | INT8 PTQ | 8 | | | | | | | | | |
| YOLO11n | INT8 QAT | 8 | | | | | | | | | |
| YOLO26n | FP32 | 32 | | | | | | | | | |
| YOLO26n | FP16 | 16 | | | | | | | | | |
| YOLO26n | INT8 PTQ | 8 | | | | | | | | | |
| YOLO26n | INT8 QAT | 8 | | | | | | | | | |

---

### Table 3: Knowledge Distillation

> **Note:** The "Student Baseline" row establishes the performance floor. The "Recovery %" column proves KD's value in mitigating accuracy loss.

| Student Model | Teacher Model | Teacher Storage (MB) | Teacher mAP | Student Baseline mAP | Student mAP | Student Recall | Recovery % | Student Weighted F1 | Latency (ms) | Storage (MB) | FN Rate % | Safety Pass? |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| YOLO11n | - | - | - | | - | | - | | | | | |
| YOLO11n | YOLO11m | | | | | | | | | | | |
| YOLO11n | YOLO11x | | | | | | | | | | | |
| YOLO26n | - | - | - | | - | | - | | | | | |
| YOLO26n | YOLO26m | | | | | | | | | | | |
| YOLO26n | YOLO26x | | | | | | | | | | | |

---

### Table 4: Proposed Method 

> [!CAUTION]
> The Order of Operations : Prune -> Fine-tune -> Distill -> Quantize (QAT).
> The "Safety Margin" tracks how close the model is to the safety cliff. If it goes negative, the pipeline fails.

| Model | Step | Method | Precision | Params (M) | Latency (ms) | Storage (MB) | Recall | Safety Margin | FN Rate % | mAP | Safety Pass? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| YOLO11n | Baseline | None | FP32 | | | | | - | | | |
| YOLO11n | Step 1 | Pruning (Best R%) | FP32 | | | | | | | | |
| YOLO11n | Step 2 | + KD Recovery | FP32 | | | | | | | | |
| YOLO11n | Step 3 | + INT8 QAT | INT8 | | | | | | | | |
| **YOLO11n** | **Final** | **Pruned + KD + QAT**| **INT8** | | | | | | | | **Yes** |
| YOLO26n | Baseline | None | FP32 | | | | | - | | | |
| YOLO26n | Step 1 | Pruning (Best R%) | FP32 | | | | | | | | |
| YOLO26n | Step 2 | + KD Recovery | FP32 | | | | | | | | |
| YOLO26n | Step 3 | + INT8 QAT | INT8 | | | | | | | | |
| **YOLO26n** | **Final** | **Pruned + KD + QAT**| **INT8** | | | | | | | | **Yes** |

---

## Key Findings & Narrative

> *This section will form the skeleton of your Discussion section. Update as data is collected.*

- **Pruning Limits:** [e.g., YOLO11n failed safety at 30% pruning due to compounding error.]
- **KD Value:** [e.g., KD recovered X% of recall, allowing 40% pruning.]
- **Quantization Anomaly:** [e.g., QAT required for safety, but slower than PTQ on the edge device.]
- **Hero Model:** [e.g., YOLO26n demonstrated superior robustness to compression.]