"""Dataset access and input transforms.

Every method is scored in the same place: the 640 x 640 frame, on the same ground-truth
boxes. Detections from a crop or a collage are mapped back to frame coordinates and matched
against the part of each box that the method keeps visible. A box the method removes
entirely stays in the denominator and counts as a miss.
"""
from __future__ import annotations

import random
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from .metrics import iou_matrix
from .paths import CFG, DATA, PARTITIONS, RGB

FRAME = 640
MIN_VISIBLE = 2.0  # px, also the rule used when the partition labels were clipped


def _yolo_to_xyxy(rows: np.ndarray, size: float = FRAME) -> np.ndarray:
    cx, cy, w, h = rows.T
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1) * size


def _read_label(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists() or path.stat().st_size == 0:
        return np.zeros(0, int), np.zeros((0, 4))
    rows = np.loadtxt(path, ndmin=2)
    return rows[:, 0].astype(int), _yolo_to_xyxy(rows[:, 1:5])


@lru_cache(maxsize=None)
def split_stems(split: str) -> tuple[str, ...]:
    """Image stems of a split in a fixed order."""
    listing = (RGB / "yolo" / f"{split}.txt").read_text(encoding="utf-8").split()
    return tuple(sorted(Path(p).stem for p in listing))


def frame_path(stem: str) -> Path:
    m = re.match(r"(subject_\d+)_(video_\d+)_", stem)
    return RGB / "images" / m[1] / m[2] / f"{stem}.jpg"


def clip_of(stem: str) -> str:
    """Recording a frame came from: the unit of statistical independence."""
    m = re.match(r"(subject_\d+)_(video_\d+)_", stem)
    return f"{m[1]}/{m[2]}"


@lru_cache(maxsize=None)
def frame_ground_truth(split: str) -> tuple:
    """Per-image (classes, xyxy boxes) on the 640 x 640 frame."""
    out = []
    for stem in split_stems(split):
        m = re.match(r"(subject_\d+)_(video_\d+)_", stem)
        out.append(_read_label(RGB / "labels" / m[1] / m[2] / f"{stem}.txt"))
    return tuple(out)


def _clip(boxes: np.ndarray, window, min_fraction: float = 0.0) -> np.ndarray:
    """Visible part of each box inside a window.

    A region is zeroed when it is under `MIN_VISIBLE` px per side, or holds less than
    `min_fraction` of the box area.
    """
    x1, y1, x2, y2 = window
    out = np.stack([np.maximum(boxes[:, 0], x1), np.maximum(boxes[:, 1], y1),
                    np.minimum(boxes[:, 2], x2), np.minimum(boxes[:, 3], y2)], axis=1)
    w, h = out[:, 2] - out[:, 0], out[:, 3] - out[:, 1]
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    visible = (w >= MIN_VISIBLE) & (h >= MIN_VISIBLE) & (w * h >= min_fraction * area)
    out[~visible] = 0.0
    return out


def visible_regions(split: str, method: str, min_fraction: float = 0.0) -> list:
    """Per-image (classes, regions) where regions is (G, K, 4) for `metrics.score_split`."""
    out = []
    for classes, boxes in frame_ground_truth(split):
        if method == "baseline":
            regions = boxes[:, None, :]
        elif method == "cropped":
            regions = _clip(boxes, CFG["input"]["crop_box"], min_fraction)[:, None, :]
        else:
            tiles = [t["src"] for t in CFG["input"]["collage_tiles"].values()]
            regions = np.stack([_clip(boxes, w, min_fraction) for w in tiles], axis=1)
        out.append((classes, regions))
    return out


def image_paths(method: str, resolution: int, split: str) -> list[Path]:
    """Input images of a split for an input method, in `split_stems` order."""
    if method == "baseline" and resolution == FRAME:
        return [frame_path(s) for s in split_stems(split)]
    root = PARTITIONS / method / f"{resolution}x{resolution}" / split / "images"
    return [root / f"{s}.jpg" for s in split_stems(split)]


def dataset_yaml(method: str, resolution: int) -> Path:
    """Write (once) the Ultralytics dataset file of a partition and return its path."""
    root = PARTITIONS / method / f"{resolution}x{resolution}"
    path = root / "dataset.yaml"
    if not path.exists():
        names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CFG["classes"]))
        rel = root.relative_to(DATA.parent).as_posix()
        path.write_text(f"path: {rel}\ntrain: train/images\nval: val/images\n"
                        f"test: test/images\nnames:\n{names}\n", encoding="utf-8")
    return path


def calibration_images(method: str, resolution: int) -> list[Path]:
    """Fixed sample of training-subject frames for PTQ range calibration."""
    q = CFG["quantization"]
    stems = random.Random(q["calibration_seed"]).sample(list(split_stems("train")),
                                                        q["calibration_images"])
    paths = {s: p for s, p in zip(split_stems("train"), image_paths(method, resolution, "train"))}
    return [paths[s] for s in stems]


def load_input(path: Path, size: int) -> np.ndarray:
    """RGB float32 NCHW tensor in [0, 1], as the detector consumes it."""
    img = Image.open(path).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.Resampling.BILINEAR)
    return (np.asarray(img, dtype=np.float32) / 255.0).transpose(2, 0, 1)[None]


def _nms(boxes: np.ndarray, conf: np.ndarray, cls: np.ndarray, iou: float = 0.5) -> np.ndarray:
    """Class-wise non-maximum suppression; returns kept indices."""
    keep = []
    for c in np.unique(cls):
        idx = np.where(cls == c)[0]
        idx = idx[np.argsort(-conf[idx], kind="stable")]
        while len(idx):
            keep.append(idx[0])
            if len(idx) == 1:
                break
            overlap = iou_matrix(boxes[idx[:1]], boxes[idx[1:]])[0]
            idx = idx[1:][overlap < iou]
    return np.array(sorted(keep), dtype=int)


def to_frame(dets: dict, method: str) -> dict:
    """Map normalised input-space boxes back to the 640 x 640 frame."""
    box = dets["box"].astype(np.float64)
    if method == "baseline":
        return {**dets, "box": box * FRAME}
    if method == "cropped":
        x1, y1, x2, y2 = CFG["input"]["crop_box"]
        scale = np.array([x2 - x1, y2 - y1, x2 - x1, y2 - y1], dtype=float)
        return {**dets, "box": box * scale + np.array([x1, y1, x1, y1], dtype=float)}
    if method == "collage":
        return _collage_to_frame(dets, box * FRAME)
    raise ValueError(method)


def _collage_to_frame(dets: dict, box: np.ndarray) -> dict:
    """Undo the tiling, then merge the duplicates that the overlap between tiles creates."""
    cx, cy = (box[:, 0] + box[:, 2]) / 2, (box[:, 1] + box[:, 3]) / 2
    out = box.copy()
    for tile in CFG["input"]["collage_tiles"].values():
        sx1, sy1, sx2, sy2 = tile["src"]
        dx1, dy1, dx2, dy2 = tile["dst"]
        hit = (cx >= dx1) & (cx < dx2) & (cy >= dy1) & (cy < dy2)
        shift = np.array([sx1 - dx1, sy1 - dy1, sx1 - dx1, sy1 - dy1], dtype=float)
        out[hit] = box[hit] + shift
        out[hit, 0::2] = np.clip(out[hit, 0::2], sx1, sx2)
        out[hit, 1::2] = np.clip(out[hit, 1::2], sy1, sy2)
    keep_all = []
    bounds = np.searchsorted(dets["image"], np.arange(dets["image"].max() + 2))
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        if hi > lo:
            keep_all.append(lo + _nms(out[lo:hi], dets["conf"][lo:hi], dets["cls"][lo:hi]))
    keep = np.concatenate(keep_all)
    return {"image": dets["image"][keep], "conf": dets["conf"][keep],
            "cls": dets["cls"][keep], "box": out[keep]}
