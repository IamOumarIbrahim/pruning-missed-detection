"""Uncertainty for a small, temporally correlated test set.

Frames of one recording are not independent, so the bootstrap resamples whole recordings
(clips) rather than frames. Seeds are kept together in every resample, which makes the
interval cover both data and training variance, and lets two configurations be compared
pairwise on the same resampled clips.
"""
from __future__ import annotations

import numpy as np

from .data import clip_of, split_stems
from .metrics import Scored


def wilson_interval(hits: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Frame-level binomial interval. Optimistic for correlated frames; kept as a lower bound."""
    if total == 0:
        return 0.0, 1.0
    p = hits / total
    denom = 1 + z ** 2 / total
    centre = (p + z ** 2 / (2 * total)) / denom
    half = z * np.sqrt(p * (1 - p) / total + z ** 2 / (4 * total ** 2)) / denom
    return float(centre - half), float(centre + half)


def clip_counts(scored: Scored, tau: float, split: str = "test") -> tuple[np.ndarray, np.ndarray]:
    """True positives and ground truth per (clip, class) at a threshold."""
    clips = sorted({clip_of(s) for s in split_stems(split)})
    index = {c: i for i, c in enumerate(clips)}
    image_clip = np.array([index[clip_of(s)] for s in split_stems(split)])
    n_classes = len(scored.gt_counts)
    tp = np.zeros((len(clips), n_classes))
    gt = np.zeros((len(clips), n_classes))
    keep = (scored.conf >= tau) & scored.first[:, 0]
    np.add.at(tp, (image_clip[scored.image[keep]], scored.cls[keep]), 1)
    np.add.at(gt, (image_clip[scored.gt_image_cls[:, 0]], scored.gt_image_cls[:, 1]), 1)
    return tp, gt


def _resamples(gt: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Clip resamples in which every class keeps at least one ground-truth box."""
    draws = []
    while len(draws) < n:
        pick = rng.integers(0, gt.shape[0], gt.shape[0])
        if (gt[pick].sum(axis=0) > 0).all():
            draws.append(pick)
    return np.array(draws)


def macro_recall_ci(runs: list[tuple[Scored, float]], n: int = 10000, seed: int = 0,
                    split: str = "test") -> dict:
    """Clip-bootstrap 95% interval of macro-recall and of each class recall over seeds."""
    counts = [clip_counts(s, t, split) for s, t in runs]
    gt = counts[0][1]
    picks = _resamples(gt, n, np.random.default_rng(seed))
    tp = np.mean([c[0] for c in counts], axis=0)
    per_class = tp[picks].sum(axis=1) / gt[picks].sum(axis=1)
    macro = per_class.mean(axis=1)
    return {"macro_low": float(np.percentile(macro, 2.5)),
            "macro_high": float(np.percentile(macro, 97.5)),
            "class_low": np.percentile(per_class, 2.5, axis=0),
            "class_high": np.percentile(per_class, 97.5, axis=0)}


def paired_delta(candidate: list[tuple[Scored, float]], reference: list[tuple[Scored, float]],
                 margin: float, n: int = 10000, seed: int = 0, split: str = "test") -> dict:
    """Clip-bootstrap interval of macro-recall(candidate) - macro-recall(reference)."""
    cand = [clip_counts(s, t, split) for s, t in candidate]
    ref = [clip_counts(s, t, split) for s, t in reference]
    gt = cand[0][1]
    picks = _resamples(gt, n, np.random.default_rng(seed))
    tp_c = np.mean([c[0] for c in cand], axis=0)
    tp_r = np.mean([c[0] for c in ref], axis=0)
    denom = gt[picks].sum(axis=1)
    delta = (tp_c[picks].sum(axis=1) / denom).mean(axis=1) - (tp_r[picks].sum(axis=1) / denom).mean(axis=1)
    return {"delta": float((tp_c.sum(axis=0) / gt.sum(axis=0)).mean() - (tp_r.sum(axis=0) / gt.sum(axis=0)).mean()),
            "low": float(np.percentile(delta, 2.5)), "high": float(np.percentile(delta, 97.5)),
            "prob_within_margin": float((delta >= -margin).mean())}
