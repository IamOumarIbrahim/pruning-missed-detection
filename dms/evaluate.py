"""Unified evaluation: one harness for every checkpoint of every phase.

Inference runs once at conf=0.001 on validation and test and is cached, so metrics can be
recomputed without touching the model. The operating point is the fixed threshold TAU in all
three phases. A validation-tuned threshold is kept alongside it as a sensitivity check, and
the test split never influences any choice.
"""
from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import onnxruntime as ort
import torch
from ultralytics import YOLO
from ultralytics.utils import nms

from . import metrics, prune  # noqa: F401  (prune registers C3k2_v2 for pruned checkpoints)
from .data import image_paths, to_frame, visible_regions
from .paths import CFG, DETECTIONS


TAU = 0.5  # the one operating point of the paper

# Ultralytics stops NMS after a wall-clock limit and silently drops the remaining boxes, which
# would make results depend on machine load. Lift the limit.
_nms = nms.non_max_suppression
nms.non_max_suppression = lambda *a, **k: _nms(*a, **{**k, "max_time_img": 3600.0})

# ONNX Runtime's spinning thread pool collapses when other jobs share the CPU.
ONNX_THREADS = int(os.environ.get("DMS_ONNX_THREADS", 6))


def _compact(stream):
    """Keep only the detections of each result, so a whole split fits in memory."""
    for r in stream:
        yield SimpleNamespace(path=r.path, boxes=r.boxes.cpu())
        r.orig_img = None


def _stack(rows: dict) -> dict:
    """Merge per-image rows into arrays sorted by image."""
    if not rows["image"]:
        return {"image": np.zeros(0, int), "conf": np.zeros(0), "cls": np.zeros(0, int),
                "box": np.zeros((0, 4))}
    out = {k: np.concatenate(v) for k, v in rows.items()}
    order = np.argsort(out["image"], kind="stable")
    return {k: v[order] for k, v in out.items()}


def _run_onnx(weights: Path, paths: list[Path], imgsz: int) -> dict:
    """Direct ONNX inference: same preprocessing, decoding and NMS as the Ultralytics predictor."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = ONNX_THREADS
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    session = ort.InferenceSession(str(weights), options, providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    rows = {k: [] for k in ("image", "conf", "cls", "box")}
    for index, path in enumerate(paths):
        img = cv2.imread(str(path))
        if img.shape[:2] != (imgsz, imgsz):
            img = cv2.resize(img, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
        blob = (img[None, :, :, ::-1].transpose(0, 3, 1, 2).astype(np.float32) / 255.0).copy()
        output = torch.from_numpy(session.run(None, {name: blob})[0])
        kept = _nms(output, conf_thres=0.001, iou_thres=0.7, nc=4, max_time_img=3600.0)[0]
        if len(kept):
            rows["image"].append(np.full(len(kept), index))
            rows["conf"].append(kept[:, 4].numpy())
            rows["cls"].append(kept[:, 5].numpy().astype(int))
            rows["box"].append((kept[:, :4] / imgsz).clamp(0, 1).numpy())
    return _stack(rows)


def _run_inference(weights: Path, paths: list[Path], imgsz: int, device) -> dict:
    if str(weights).endswith(".onnx"):
        return _run_onnx(weights, paths, imgsz)
    model = YOLO(str(weights), task="detect")
    index = {p.name: i for i, p in enumerate(paths)}
    rows = {k: [] for k in ("image", "conf", "cls", "box")}
    with tempfile.TemporaryDirectory() as tmp:  # a Python list would become one huge batch
        manifest = Path(tmp) / "images.txt"
        manifest.write_text(chr(10).join(str(p) for p in paths), encoding="utf-8")
        stream = model.predict(source=str(manifest), conf=0.001, iou=0.7, imgsz=imgsz, device=device,
                               batch=32, verbose=False, stream=True, save=False)
        results = list(_compact(stream))
    if len(results) != len(paths):
        raise RuntimeError(f"{weights}: predicted {len(results)} of {len(paths)} images")
    for result in results:
        n = len(result.boxes)
        if n == 0:
            continue
        rows["image"].append(np.full(n, index[Path(result.path).name]))
        rows["conf"].append(result.boxes.conf.numpy())
        rows["cls"].append(result.boxes.cls.numpy().astype(int))
        rows["box"].append(result.boxes.xyxyn.numpy())
    return _stack(rows)


def detections(run_id: str, weights: Path, method: str, resolution: int, imgsz: int, split: str,
               device=0, refresh: bool = False) -> dict:
    """Input-space detections of one run on one split, from cache when possible."""
    cache = DETECTIONS / run_id / f"{split}.npz"
    if cache.exists() and not refresh:
        with np.load(cache) as f:
            return {k: f[k] for k in f.files}
    start = time.time()
    dets = _run_inference(weights, image_paths(method, resolution, split), imgsz, device)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **dets)
    print(f"  {run_id} {split}: {len(dets['conf'])} detections, {time.time() - start:.0f}s", flush=True)
    return dets


def scored_run(run) -> dict:
    """Matched detections of a registered run on validation and test, in frame coordinates."""
    out = {}
    for split in ("val", "test"):
        dets = to_frame(detections(run.id, run.weights, run.method, run.resolution, run.imgsz, split),
                        run.method)
        out[split] = metrics.score_split(dets, visible_regions(split, run.method))
        if split == "test":
            out["strict"] = metrics.score_split(dets, visible_regions(split, run.method, 0.5))
    return out


def select_config_tau(val_by_seed: list) -> float:
    """Sensitivity check: the threshold maximising validation macro-F1 averaged over seeds."""
    curves = [[metrics.at_threshold(v, t)["macro_f1"] for t in metrics.TAU_GRID] for v in val_by_seed]
    return float(metrics.TAU_GRID[int(np.argmax(np.mean(curves, axis=0)))])


def row(run, scored: dict, tuned_tau: float) -> dict:
    """CSV row of one run: primary metrics at TAU, and at the validation-tuned threshold."""
    ap = metrics.map_scores(scored["test"])
    out = {"id": run.id, "phase": run.phase, "arch": run.arch, "sparsity": run.sparsity,
           "quant": run.quant, "method": run.method, "resolution": run.resolution,
           "seed": run.seed, "tuned_tau": tuned_tau, "map50": ap["map50"], "map50_95": ap["map50_95"]}
    views = (("val", scored["val"], TAU), ("test", scored["test"], TAU),
             ("tuned", scored["test"], tuned_tau), ("strict", scored["strict"], TAU))
    for prefix, data, tau in views:
        m = metrics.at_threshold(data, tau)
        for key in ("macro_recall", "macro_precision", "macro_f1", "worst_recall", "micro_precision"):
            out[f"{prefix}_{key}"] = m[key]
        for i, name in enumerate(CFG["classes"]):
            out[f"{prefix}_recall_{name}"] = float(m["recall"][i])
            out[f"{prefix}_precision_{name}"] = float(m["precision"][i])
    return out
