"""Confidence threshold optimization under safety-recall constraint.

Implements the per-seed threshold tuning from the Objective section:
    tau* = argmax_tau Precision(tau) s.t. min_c Recall_c(tau) >= R_floor
"""

import numpy as np
from eval.metrics import compute_detection_metrics, CLASS_NAMES


def optimize_threshold(matches, gt_counts, r_floor, class_names=None,
                       num_steps=1000):
    """Find tau* maximizing precision subject to min-class recall >= r_floor.

    Args:
        matches: list of (confidence, is_tp, class_id).
        gt_counts: dict class_id -> GT count.
        r_floor: minimum required recall for every safety class.
        class_names: list of class names.
        num_steps: resolution of the threshold sweep.

    Returns:
        dict with tau_star, feasible (bool), metrics (at tau*).
    """
    if class_names is None:
        class_names = CLASS_NAMES

    if not matches:
        return {
            'tau_star': 0.0,
            'feasible': False,
            'metrics': compute_detection_metrics([], gt_counts, 0.0,
                                                 class_names),
        }

    confs = sorted({c for c, _, _ in matches})
    lo, hi = min(confs), max(confs)
    linspace = np.linspace(lo, hi, num_steps).tolist()
    all_thresholds = sorted(set(confs + linspace))

    best_tau = None
    best_precision = -1.0
    best_metrics = None
    feasible = False

    infeasible_tau = None
    infeasible_min_recall = -1.0
    infeasible_metrics = None

    for tau in all_thresholds:
        m = compute_detection_metrics(matches, gt_counts, tau, class_names)
        mr = m['min_safety_recall']
        pr = m['precision']

        if mr >= r_floor:
            feasible = True
            if pr > best_precision:
                best_precision = pr
                best_tau = tau
                best_metrics = m
        if mr > infeasible_min_recall:
            infeasible_min_recall = mr
            infeasible_tau = tau
            infeasible_metrics = m

    if feasible:
        return {'tau_star': float(best_tau), 'feasible': True,
                'metrics': best_metrics}
    return {
        'tau_star': float(infeasible_tau) if infeasible_tau is not None else 0.0,
        'feasible': False,
        'metrics': (infeasible_metrics if infeasible_metrics
                    else compute_detection_metrics(matches, gt_counts, 0.0,
                                                   class_names)),
    }


def compute_r_base(matches_per_seed, gt_counts_per_seed,
                   class_names=None):
    """Compute R_base: mean across seeds of max achievable min-class recall.

    Args:
        matches_per_seed: list of match-lists, one per seed.
        gt_counts_per_seed: list of gt_counts dicts, one per seed.
        class_names: list of class names.

    Returns:
        r_base (float), list of per-seed R_base values.
    """
    if class_names is None:
        class_names = CLASS_NAMES

    per_seed = []
    for matches, gt_counts in zip(matches_per_seed, gt_counts_per_seed):
        confs = sorted({c for c, _, _ in matches}, reverse=True)
        best = 0.0
        for tau in confs:
            m = compute_detection_metrics(matches, gt_counts, tau,
                                          class_names)
            if m['min_safety_recall'] > best:
                best = m['min_safety_recall']
        per_seed.append(best)
    return float(np.mean(per_seed)), per_seed
