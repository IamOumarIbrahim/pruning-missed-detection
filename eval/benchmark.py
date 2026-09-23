"""FPS benchmarking with CUDA synchronization."""

import time
import numpy as np
import torch
from ultralytics import YOLO


def benchmark_fps(model_path, imgsz=640, warmup=50, timed_runs=200,
                  device=0):
    """Measure inference FPS.

    Warm-up iterations are excluded. CUDA synchronization is applied
    before each timing boundary.

    Args:
        model_path: Path to .pt or .engine model.
        imgsz: Input image size.
        warmup: Number of warm-up iterations (excluded).
        timed_runs: Number of timed iterations.
        device: CUDA device index.

    Returns:
        dict with fps, mean_latency_ms, std_latency_ms.
    """
    model = YOLO(str(model_path))
    dummy = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

    for _ in range(warmup):
        model.predict(source=dummy, imgsz=imgsz, verbose=False, device=device)

    if torch.cuda.is_available():
        torch.cuda.synchronize(device)

    latencies = []
    for _ in range(timed_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        model.predict(source=dummy, imgsz=imgsz, verbose=False, device=device)
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        latencies.append(time.perf_counter() - t0)

    lat = np.array(latencies)
    mean_lat = float(np.mean(lat))
    return {
        'fps': 1.0 / mean_lat if mean_lat > 0 else 0.0,
        'mean_latency_ms': float(mean_lat * 1000),
        'std_latency_ms': float(np.std(lat) * 1000),
    }
