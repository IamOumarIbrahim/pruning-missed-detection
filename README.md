# Model–Input Co-Design for Edge Driver Monitoring

Code and results for *Model Compression and Input Optimization Co-Design for Edge Driver
Monitoring Systems* (ICAUC 2027). We ask how far a YOLO driver-behavior detector can be shrunk,
through pruning, INT8 quantization, and smaller inputs, before it misses drowsiness and
distraction cues. The safety metric is macro-recall over four classes on drivers the model has
never seen.

![Test macro-recall of every INT8 configuration against compute](results/figures/tradeoff.png)

## What we found

- **Reshape the input, then quantize.** A 320-pixel overlapping collage with INT8 PTQ reaches
  **92.9% macro-recall at 1.61 GFLOPs and 3.0 MB**: 4.0× less compute and 3.4× less storage than
  the FP32 baseline (86.4%), with recall 6.5 pp *higher* (95% CI +0.9 to +13.6).
- **Pruning works at full resolution only.** YOLO11n holds the 5-point recall floor up to 70%
  channel sparsity at 640 px. At 320 px and below, the same sparsity fails, and the configuration
  picked on three validation drivers lost 12.6 pp on the test drivers.
- **A static crop loses peripheral cues.** It magnifies the face, so yawning improves, but it cuts
  9% of phone-use boxes. The overlapping collage keeps both and wins at every resolution down to 160 px.
- **INT8 needs robust calibration.** With MinMax ranges, PTQ recall swung by up to 32 pp between
  seeds while mAP barely moved: outlier activations shrank the confidences. Moving-average ranges
  fix it:

![PTQ recall across seeds before and after the calibration fix](results/figures/calibration.png)

How the numbers were checked, and what changed from the first submission:
[docs/audit.md](docs/audit.md).

## Reproduce

```bash
pip install -r requirements.txt
python scripts/build_partitions.py      # crop, collage and resized inputs (needs DMD frames, see data/README.md)
python scripts/train.py baseline        # then: prune, input, ptq, qat, codesign --method collage --resolution 320 --sparsity 0
python scripts/evaluate.py              # inference once per model, cached in results/detections
python scripts/score.py                 # metrics at tau = 0.5 for every run
python scripts/profile_models.py        # parameters, GFLOPs, ONNX sizes
python scripts/report.py                # summary tables, bootstrap intervals, selection
python scripts/paper_assets.py          # LaTeX tables and figures for the paper
python scripts/build_paper.py           # clean and highlighted PDFs
pytest tests
```

Every finished step is skipped on a re-run, so an interrupted pipeline resumes where it stopped.
QAT needs `nvidia-modelopt` (in `requirements.txt`).

## Layout

| Path | Contents |
| :--- | :--- |
| `dms/` | The library: data and input transforms, pruning, quantization, metrics, statistics |
| `scripts/` | One script per pipeline stage, listed above |
| `configs/experiment.yaml` | Every setting of the paper: splits, seeds, sparsities, crop and collage geometry |
| `results/tables/` | Per-run metrics (`runs.csv`), phase summaries, uncertainty, model costs |
| `paper/` | Manuscript source; tables and figures in `paper/generated/` are written by the scripts |
| `data/` | Annotations and split lists; DMD frames are obtained separately |
| `tests/` | Metric, matching, coordinate-mapping and pruning-equivalence tests |

Checkpoints are not in git. Evaluation needs `models/`, laid out as in `dms/runs.py`.

## Protocol in one paragraph

Subjects are split 8:3:3 (train, validation, test), so no driver appears in two splits. Every
model is scored on the same 3,213 test frames at one confidence threshold, τ = 0.5, with greedy
class-aware matching at IoU 0.5. Detections from a crop or collage are mapped back to the full
frame and matched to the visible part of each box, and a box the input removes counts as a miss.
Results are the mean ± SD over three seeds. Intervals come from a bootstrap over whole recording
clips, because the rare classes come from only five test clips each.
