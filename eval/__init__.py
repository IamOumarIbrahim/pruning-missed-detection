"""Evaluation utilities: threshold optimization, metrics, and FPS benchmarking."""

from eval.threshold import optimize_threshold
from eval.metrics import compute_detection_metrics, evaluate_model
from eval.benchmark import benchmark_fps
