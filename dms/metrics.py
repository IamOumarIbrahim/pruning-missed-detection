"""Detection metrics shared by all three phases.

One definition everywhere:
  * greedy, class-aware, confidence-ordered matching at IoU 0.5;
  * macro-recall, macro-precision and macro-F1 are unweighted means over the four classes;
  * mAP is computed on the full detection list (conf >= 0.001), never on a thresholded one.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

IOU_LADDER = np.linspace(0.5, 0.95, 10)
TAU_GRID = np.round(np.arange(0.05, 0.9501, 0.005), 3)


@dataclass
class Scored:
    """Detections of one model on one split, already matched against ground truth."""

    conf: np.ndarray  # (D,) sorted descending
    cls: np.ndarray  # (D,)
    tp: np.ndarray  # (D, 10) bool, one column per IoU threshold in IOU_LADDER
    first: np.ndarray  # (D, 10) bool, True for the first detection of each ground-truth box
    image: np.ndarray  # (D,) index into the split's image list
    gt_counts: np.ndarray  # (C,) ground-truth instances per class
    gt_image_cls: np.ndarray  # (G, 2) image index and class of every ground-truth box


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between (N, 4) and (M, 4) xyxy boxes."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def greedy_match(iou, confs, classes, gt_classes, owners, iou_threshold):
    """Match detections to visible regions, highest confidence first.

    Each region takes one detection. `owners` maps regions to ground-truth boxes, so an object
    shown twice (overlapping collage tiles) can be found twice without a false positive, and
    still counts once for recall. Returns the TP mask and the first-detection mask.
    """
    tp = np.zeros(len(confs), dtype=bool)
    first = np.zeros(len(confs), dtype=bool)
    if iou.size == 0:
        return tp, first
    free = np.ones(iou.shape[1], dtype=bool)
    found = set()
    for i in np.argsort(-confs, kind="stable"):
        candidates = np.where(free & (gt_classes == classes[i]))[0]
        if len(candidates) == 0:
            continue
        j = candidates[np.argmax(iou[i, candidates])]
        if iou[i, j] >= iou_threshold:
            tp[i] = True
            free[j] = False
            first[i] = owners[j] not in found
            found.add(owners[j])
    return tp, first


def score_split(dets: dict, ground_truth: list, num_classes: int = 4) -> Scored:
    """Match stored detections against ground truth.

    `dets` holds arrays image/conf/cls/box sorted by image. `ground_truth[k]` is the pair
    (classes, regions) of image k, where regions has shape (G, K, 4): the K visible regions
    of each of its G boxes, in the coordinate frame of the detections. A box with no visible
    region stays in the denominator.
    """
    n_images = len(ground_truth)
    bounds = np.searchsorted(dets["image"], np.arange(n_images + 1))
    conf, cls, tp, first, image = [], [], [], [], []
    for k in range(n_images):
        lo, hi = bounds[k], bounds[k + 1]
        if hi == lo:
            continue
        gt_c, regions = ground_truth[k]
        c, y, b = dets["conf"][lo:hi], dets["cls"][lo:hi], dets["box"][lo:hi]
        per_box = regions.shape[1] if len(gt_c) else 1
        flat = regions.reshape(-1, 4) if len(gt_c) else np.zeros((0, 4))
        region_cls = np.repeat(gt_c, per_box)
        owners = np.repeat(np.arange(len(gt_c)), per_box)
        iou = iou_matrix(b, flat)
        hits = [greedy_match(iou, c, y, region_cls, owners, t) for t in IOU_LADDER]
        tp.append(np.stack([h[0] for h in hits], axis=1))
        first.append(np.stack([h[1] for h in hits], axis=1))
        conf.append(c), cls.append(y), image.append(np.full(hi - lo, k))

    gt_counts = np.zeros(num_classes, dtype=int)
    gt_rows = []
    for k, (gt_c, _) in enumerate(ground_truth):
        for c in gt_c:
            gt_counts[int(c)] += 1
            gt_rows.append((k, int(c)))
    gt_rows = np.array(gt_rows, dtype=int).reshape(-1, 2)

    if not conf:
        empty = np.zeros((0, 10), bool)
        return Scored(np.zeros(0), np.zeros(0, int), empty, empty, np.zeros(0, int), gt_counts, gt_rows)
    conf = np.concatenate(conf)
    order = np.argsort(-conf, kind="stable")
    return Scored(conf[order], np.concatenate(cls)[order].astype(int), np.concatenate(tp)[order],
                  np.concatenate(first)[order], np.concatenate(image)[order], gt_counts, gt_rows)


def at_threshold(s: Scored, tau: float) -> dict:
    """Per-class and macro metrics at a confidence threshold, using the IoU 0.5 matches."""
    keep = s.conf >= tau
    tp = np.bincount(s.cls[keep][s.tp[keep, 0]], minlength=len(s.gt_counts))
    found = np.bincount(s.cls[keep][s.first[keep, 0]], minlength=len(s.gt_counts))
    pred = np.bincount(s.cls[keep], minlength=len(s.gt_counts))
    recall = found / np.maximum(s.gt_counts, 1)
    precision = np.divide(tp, pred, out=np.zeros(len(tp)), where=pred > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(len(tp)), where=(precision + recall) > 0)
    return {
        "recall": recall, "precision": precision, "f1": f1, "tp": found, "pred": pred,
        "macro_recall": float(recall.mean()), "macro_precision": float(precision.mean()),
        "macro_f1": float(f1.mean()), "worst_recall": float(recall.min()),
        "micro_precision": float(tp.sum() / max(pred.sum(), 1)),
    }


def select_tau(val: Scored) -> float:
    """Validation macro-F1 maximiser. Ties resolve to the lowest threshold (favours recall)."""
    scores = [at_threshold(val, t)["macro_f1"] for t in TAU_GRID]
    return float(TAU_GRID[int(np.argmax(scores))])


def average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    """COCO 101-point AP: best precision at or beyond each recall level."""
    envelope = np.append(np.flip(np.maximum.accumulate(np.flip(precision))), 0.0)
    return float(envelope[np.searchsorted(recall, np.linspace(0, 1, 101), side="left")].mean())


def map_scores(s: Scored) -> dict:
    """Class-averaged AP at IoU 0.5 and over IoU 0.5:0.95."""
    ap = np.zeros((len(s.gt_counts), len(IOU_LADDER)))
    for c in range(len(s.gt_counts)):
        mask = s.cls == c
        if s.gt_counts[c] == 0 or not mask.any():
            continue
        tp, first = s.tp[mask], s.first[mask]
        for j in range(len(IOU_LADDER)):
            counted = first[:, j] | ~tp[:, j]  # a repeat view of a found object is neither
            tps = np.cumsum(first[counted, j])
            fps = np.cumsum(~tp[counted, j])
            ap[c, j] = average_precision(tps / s.gt_counts[c], tps / (tps + fps + 1e-12))
    return {"map50": float(ap[:, 0].mean()), "map50_95": float(ap.mean()), "ap50": ap[:, 0]}
