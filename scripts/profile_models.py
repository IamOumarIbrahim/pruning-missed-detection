"""Measure cost for every configuration: parameters, GFLOPs and deployable file size.

Size is always an ONNX file. A .pt checkpoint stores FP16 weights plus metadata, so it is not
comparable with an INT8 ONNX graph. Anchor tables are baked into a fixed-shape graph and grow
with the input size, so they are measured apart from the weights.
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ultralytics import YOLO  # noqa: E402
from ultralytics.utils.torch_utils import get_flops  # noqa: E402

from dms import prune, quant, runs  # noqa: E402,F401
from dms.paths import MODELS, TABLES, enter_root  # noqa: E402


def fp32_graph(run: runs.Run) -> Path:
    """FP32 ONNX of a configuration, exported on demand."""
    if run.phase == 1:
        path = MODELS / run.arch / "onnx" / f"{run.sparsity}pct" / "model_fp32.onnx"
    elif run.phase == 2:
        path = MODELS / "input_onnx" / f"{run.method}_{run.resolution}" / "model_fp32.onnx"
    else:
        path = run.weights.parent.parent / "ptq" / "model_fp32.onnx"
    if not path.exists():
        quant.export_onnx(run.weights, path, run.imgsz)
    return path


def seed_size(path: Path) -> float:
    return path.stat().st_size / 2 ** 20 if path.exists() else float("nan")


def main():
    enter_root()
    rows = []
    for run in runs.phase1() + runs.phase2() + runs.phase3():
        if run.quant != "fp32" or run.seed != runs.SEEDS[0] or not run.weights.exists():
            continue
        net = YOLO(str(run.weights)).model
        graph = fp32_graph(run)
        row = {"phase": run.phase, "arch": run.arch, "sparsity": run.sparsity, "method": run.method,
               "resolution": run.resolution, "imgsz": run.imgsz,
               "params": sum(p.numel() for p in net.parameters()),
               "gflops": get_flops(net, imgsz=run.imgsz),
               "pt_mb": seed_size(run.weights), "fp32_mb": seed_size(graph)}
        if run.phase == 1:
            ptq = runs.quantized_dir(run.arch, "ptq", run.sparsity, run.seed) / "model_int8.onnx"
            qat = runs.quantized_dir(run.arch, "qat", run.sparsity, run.seed) / "model_int8.onnx"
        else:
            base = run.weights.parent.parent
            ptq, qat = base / "ptq" / "model_int8.onnx", base / "qat" / "model_int8.onnx"
        row["ptq_mb"], row["qat_mb"] = seed_size(ptq), seed_size(qat)
        split = quant.onnx_breakdown(ptq if ptq.exists() else graph)
        row["anchor_mb"], row["int8_weights_mb"] = split["anchor_tables_mb"], split["weights_mb"]
        rows.append(row)
        print(row, flush=True)
    with open(TABLES / "profiles.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
