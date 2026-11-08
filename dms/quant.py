"""INT8 post-training quantization with ONNX Runtime.

Scheme: QDQ static quantization, unsigned 8-bit asymmetric activations, signed 8-bit
symmetric per-channel weights, on 300 training-subject frames.

Two choices keep recall stable:
  * Ranges use a moving average of per-frame extremes. Plain MinMax lets one outlier activation
    set the range of a tensor. Confidences then shrink in some seeds while ranking survives:
    the same checkpoint goes from 27% to 78% macro-recall at tau = 0.5.
  * The final convolution of each head branch and the box/class decoding stay in floating
    point. Class logits span 40-150 units, so one 8-bit step would be 0.15-0.59 logit.
"""
from __future__ import annotations

import re
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import numpy_helper
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat,
                                      QuantType, quantize_static)
from ultralytics import YOLO

from . import prune  # noqa: F401  (registers C3k2_v2 so pruned checkpoints load)
from .data import calibration_images, load_input


# nvidia-modelopt 0.48 reads this attribute, which torch 2.6 does not define yet.
if not hasattr(torch.compiler, "is_exporting"):
    torch.compiler.is_exporting = lambda: False


class _Frames(CalibrationDataReader):
    def __init__(self, paths, size, input_name):
        self.paths, self.size, self.name = paths, size, input_name
        self.rewind()

    def get_next(self):
        path = next(self.iterator, None)
        return None if path is None else {self.name: load_input(path, self.size)}

    def rewind(self):
        self.iterator = iter(self.paths)


@contextmanager
def scratch_copy(checkpoint: Path):
    """Exporters write beside their input; keep that away from the checkpoint folder."""
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / "model.pt"
        shutil.copy2(checkpoint, local)
        yield local


def export_onnx(checkpoint: Path, target: Path, imgsz: int) -> Path:
    """Export a checkpoint to a fixed-shape FP32 ONNX graph."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with scratch_copy(checkpoint) as local:
        exported = YOLO(str(local)).export(format="onnx", imgsz=imgsz, opset=17, simplify=True,
                                           dynamic=False, device="cpu", verbose=False)
        shutil.move(str(exported), target)
    return target


def float_head_nodes(graph: onnx.GraphProto) -> list[str]:
    """Head nodes kept in floating point: output convolutions and all box/class decoding."""
    head = max(int(n.name.split("/")[1].split(".")[1]) for n in graph.node
               if n.name.startswith("/model."))
    prefix = f"/model.{head}/"
    keep_quantized = re.compile(r"/cv[23]\.\d/cv[23]\.\d\.[01]/")
    return [n.name for n in graph.node
            if n.name.startswith(prefix) and not keep_quantized.search(n.name)]


def quantize(fp32: Path, target: Path, method: str, resolution: int, imgsz: int,
             scope: str = "full", per_channel: bool = True) -> Path:
    """Statically quantize an FP32 ONNX graph.

    scope "full" quantizes everything before the float head; "conv" only the convolutions.
    """
    graph = onnx.load(str(fp32)).graph
    float_head = set(float_head_nodes(graph))
    if scope == "conv":
        exclude = [n.name for n in graph.node
                   if n.op_type not in {"Conv", "MatMul", "Gemm"} or n.name in float_head]
    else:
        exclude = sorted(float_head)
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    quantize_static(
        str(fp32), str(target),
        _Frames(calibration_images(method, resolution), imgsz, graph.input[0].name),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=per_channel,
        calibrate_method=CalibrationMethod.MinMax,
        nodes_to_exclude=exclude,
        extra_options={"CalibMovingAverage": True, "CalibMovingAverageConstant": 0.01},
    )
    return target


def fold_weight_quantizers(source: Path, target: Path) -> Path:
    """Store QAT weights as int8, so the file is a true INT8 artifact.

    A QAT export keeps float weights behind QuantizeLinear nodes. This replaces each such node
    by the integer weights it would produce, leaving the learned scales in DequantizeLinear.
    """
    model = onnx.load(str(source))
    graph = model.graph
    weights = {t.name: t for t in graph.initializer}
    constants = {n.output[0]: numpy_helper.to_array(n.attribute[0].t)
                 for n in graph.node if n.op_type == "Constant"}
    dequantizers = {n.input[0]: n for n in graph.node if n.op_type == "DequantizeLinear"}
    for node in [n for n in graph.node if n.op_type == "QuantizeLinear" and n.input[0] in weights]:
        tensor = numpy_helper.to_array(weights[node.input[0]])
        scale, zero = constants[node.input[1]], constants[node.input[2]]
        axis = next((a.i for a in node.attribute if a.name == "axis"), 1)
        shape = [-1 if i == axis else 1 for i in range(tensor.ndim)]
        codes = np.rint(tensor / scale.reshape(shape)) + zero.reshape(shape)
        codes = np.clip(codes, -128, 127).astype(np.int8)
        name = f"{node.input[0]}_int8"
        graph.initializer.append(numpy_helper.from_array(codes, name))
        dequantizers[node.output[0]].input[0] = name
        graph.initializer.remove(weights[node.input[0]])
        graph.node.remove(node)
    onnx.save(model, str(target))
    return Path(target)


def onnx_breakdown(path: Path) -> dict:
    """File size split into weights and the anchor tables baked into a fixed-shape graph."""
    model = onnx.load(str(path))
    tensors = [t for t in model.graph.initializer]
    tensors += [n.attribute[0].t for n in model.graph.node if n.op_type == "Constant"]
    tables = [t for t in tensors if len(t.dims) == 3 and t.dims[0] == 1 and t.dims[-1] >= 100]
    table_bytes = sum(len(t.raw_data) for t in tables)
    return {"file_mb": Path(path).stat().st_size / 2 ** 20,
            "anchor_tables_mb": table_bytes / 2 ** 20,
            "weights_mb": (sum(len(t.raw_data) for t in tensors) - table_bytes) / 2 ** 20}
