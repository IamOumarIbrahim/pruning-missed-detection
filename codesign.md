# Co-Design Study: Input Resolution Reduction + Pruning + Quantization vs. Baseline Models

> **Operating Point:** Strictly fixed confidence threshold $\tau = 0.50$, greedy bipartite $\text{IoU} = 0.50$ matching against ground truth.  
> **Evaluation Split:** Held-out test split (`subject_05`, `subject_10`, `subject_12`: 3,213 images) with strictly zero subject overlap.  
> **Training Protocol:** Up to 80 epochs per run with early stopping (`patience = 15`) across 3 seeds (`72, 73, 74`), `batch = 16`.  

---

## 1. Summary Table: Mean ± SD across 3 Seeds (Fixed $\tau = 0.50$)

| Model Architecture | Input Resolution | Sparsity / Condition | Precision / Quant | Total Params | GFLOPs | Model Size (MB) | Precision (%) | Macro-Recall (%) | Worst-Class Recall (%) | Avg Epochs | Seeds Done |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| YOLO11N | 320x320 Crop | Pruned 70% | PTQ INT8 | 890,533 | 0.614 | 1.19 | 79.5 ± 0.0% | 79.9 ± 0.0% | 60.6 ± 0.0% | 70.0 | 3 / 3 |
| YOLO11N | 160x160 Crop | Pruned 70% | PTQ INT8 | 890,533 | 0.153 | 1.16 | 77.1% | 72.4% | 45.5% | 80.0 | 1 / 3 |

---

## 2. Per-Seed Checkpoint Log

| Model | Input Res | Condition | Quantization | Seed | Epochs Trained | Params | GFLOPs | Size (MB) | Precision (%) | Macro-Recall (%) | Worst-Class Recall (%) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| YOLO11N | 160x160 Crop | Pruned 70% | PTQ INT8 | Seed 72 | 80 / 80 | 890,533 | 0.153 | 1.16 | 77.1% | 72.4% | 45.5% | COMPLETED |
| YOLO11N | 320x320 Crop | Pruned 70% | PTQ INT8 | Seed 72 | 70 / 80 | 890,533 | 0.614 | 1.19 | 79.5% | 79.9% | 60.6% | COMPLETED |
| YOLO11N | 320x320 Crop | Pruned 70% | PTQ INT8 | Seed 73 | 70 / 80 | 890,533 | 0.614 | 1.19 | 79.5% | 79.9% | 60.6% | COMPLETED |
| YOLO11N | 320x320 Crop | Pruned 70% | PTQ INT8 | Seed 74 | 70 / 80 | 890,533 | 0.614 | 1.19 | 79.5% | 79.9% | 60.6% | COMPLETED |

---

## 3. Key Findings & Pareto Trade-Offs

- **Extreme Compute Scaling:** Pruning 70% of YOLO11N combined with cropped ROI reduction scales GFLOPs dramatically down from ~6.50 GFLOPs (native 640) to **0.614 GFLOPs (320x320 crop)** and **0.153 GFLOPs (160x160 crop)** (~42.5x compute reduction).
- **Storage Compression:** PTQ INT8 export compresses model footprint down to ~1.16–1.19 MB, suitable for ultra-constrained microcontroller and embedded NPU deployment.
- **Early Stopping Mechanism:** Enabled with `patience = 15` across 80 max epochs, preventing overfitting while ensuring models reach optimal convergence.

---
*Report auto-updated dynamically by `scripts/run_codesign_multiseed.py`.*