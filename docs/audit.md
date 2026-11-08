# Audit of the submitted results

What was wrong in the code behind the first submission, how it was confirmed, and what changed.
Every number in the revised paper comes from `scripts/evaluate.py` and `scripts/score.py`.

## Findings that changed results

| # | Finding | Evidence | Fix |
| ---: | :--- | :--- | :--- |
| 1 | Phase II precision, recall and F1 were read at the test-set F1 peak, not at the validation threshold. `model.val(conf=tau)` filters detections, but `box.mp` and `box.mr` then come from the best point of the test curve. | Baseline 640 seed 1 reports identical P/R (0.869/0.861) at tau = 0.402, 0.25 and 0.50 in `summary_table.csv`. | One harness computes metrics at the frozen threshold. |
| 2 | Phase II mAP used detections already cut at conf = 0.25. | mAP@0.5 was 0.889 for a model that scores above 0.90 on the full list. | mAP uses all detections down to conf = 0.001. |
| 3 | Crop recall dropped lost objects from the denominator: 451 phone boxes instead of 497. | `TEST_GT_COUNTS_CROP` in the old co-design script. | All methods score against the same 3,213 frames and 614 boxes. A box the crop removes is a miss. |
| 4 | "QAT" had no fake quantization. It was fine-tuning followed by PTQ. | No quantizer modules in the training code. | True QAT with NVIDIA ModelOpt through Ultralytics `quantize=8`. |
| 5 | PTQ recall collapsed in some seeds, up to 27% macro-recall at 80% sparsity. Two causes. MinMax ranges were stretched by rare outlier activations (range 167 against a median of 14), which compressed confidences while ranking survived. The head outputs were also quantized. | One bad seed: true-class confidence fell from 0.77 to 0.37 with correlation 0.97, mAP@0.5 stayed at 0.87. Moving-average ranges moved it from 27.2% to 77.5%, and another seed from 42.9% to 86.3%. Class-balanced calibration did not help (30.4%). | Moving-average calibration, with head outputs and decoding in float. |
| 6 | `optimizer=auto` ignored `lr0=0.001` and `momentum=0.937`. The co-design runs and the "QAT" runs trained at lr = 0.01 with MuSGD. | Ultralytics log line: `ignoring 'lr0=0.001'`, then `MuSGD(lr=0.01, momentum=0.9)`. | Stated in the paper. QAT sets `optimizer=SGD` explicitly. |
| 7 | The co-design pruned the 640 full-frame model, not the model trained on the crop. | `src_pt` in the old co-design script. | The pipeline prunes the crop-trained model. |
| 8 | Precision meant different things: micro in Phases I and III, macro in Phase II. F1 was computed from macro P and macro R in Phase II, not as the mean of class F1. | The old evaluators. Micro precision hides a class at 0.29 precision behind phone use. | Macro precision and mean class F1 in all phases. |
| 9 | Standard deviation used `ddof=0` in Phase II and `ddof=1` elsewhere. | `np.std` against `DataFrame.std`. | `ddof=1` everywhere. |
| 10 | Recall/GFLOPs at the 80 px input divided by a rounded 0.1 GFLOPs. The network runs at 96 px, which is 0.15 GFLOPs. | `check_imgsz` rounds 80 to 96. | GFLOPs are measured at the true input size. |
| 11 | Sizes compared a `.pt` file with an ONNX file. A `.pt` holds FP16 weights and metadata. | 5.23 MB `.pt` against 10.7 MB FP32 ONNX for the same network. | All sizes are ONNX files. |
| 12 | The same configuration appeared as 1.31 MB and 1.19 MB. | The fixed-shape graph stores anchor tables that grow with input size: 8,400 anchors at 640, 2,100 at 320, a difference of 126,000 bytes. | Weights and anchor tables are reported apart. |
| 13 | Ultralytics was not pinned. Versions 8.4.120 and 8.4.174 export different INT8 graphs and select YOLO26 end-to-end detections differently. | The same YOLO26n weights give 85.8% macro-recall on 8.4.120 (the published value, reproduced exactly) and 89.0% on 8.4.174. YOLO11n is unaffected. | `requirements.txt` pins every package that affects results. All numbers use 8.4.174. |
| 14 | Ultralytics stops NMS at a wall-clock limit and silently drops boxes, so results could depend on machine load. | `NMS time limit exceeded` warning during a PTQ run on a busy CPU. | The limit is lifted in `dms/evaluate.py`. |
| 15 | Collage was scored per tile fragment, so one object could count as several ground truths. Baseline and crop were scored per object. | Collage at 320 px moves from recall 0.826 to 0.932 under object-level scoring. | All methods are scored per object. A stricter 50% visibility rule is reported as a sensitivity check. |

## Reproducibility and repository

| Finding | Fix |
| :--- | :--- |
| Baseline weights sat behind junctions into a folder that had moved. | Copied into `models/`. |
| Absolute `C:\Dev\...` paths in configs and scripts. | Paths relative to the repository root. |
| Phase II code lived in a second repository. | Merged here. |
| 30 overlapping scripts with duplicated evaluators. | One package, `dms/`, and five scripts. |
| `update_github.py` committed and pushed every 30 minutes. | Removed. |
| No tests. | `tests/` covers matching, thresholds and coordinate mapping. |
| `C3k2_v2` docstring claimed global pruning. | Documented as it runs: per group. |

## Corrections to the first submission text

- YOLO11n stayed within 2.0 pp of baseline up to 70% sparsity, not 1.0 pp.
- YOLO26n stayed within 4.4 pp up to 60% sparsity, not 2.4 pp.
- QAT at 80% sparsity was marked unsafe at 81.5%, above the 81.4% floor.
- The 80 px input is resized to 96 px. It is not padded.
- The resolution sweep stops at 640 px. The 1280 px runs were never reported.
- The crop window is off-center: x spans 140 to 540, y spans 120 to 520.
- YOLO26n was described with re-parameterized convolutions. Its defining changes are NMS-free inference and the removal of DFL.
