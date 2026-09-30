"""Quantization-Aware Training (QAT) for YOLO models.

Strategy: fine-tune the model at a lower learning rate to adapt weights for
INT8 quantization, then export via TensorRT INT8 PTQ. The fine-tuning
produces a weight distribution better suited for quantization than the
original training. Reported as a separate INT8 (QAT) row in Table 3.
"""

from pathlib import Path
from ultralytics import YOLO
from quant.ptq import export_ptq
from prune.pruner import PrunedDetectionTrainer


def run_qat(model_path, data_yaml, output_dir, epochs=100, imgsz=640,
            batch=16, seed=0, device=0):
    """QAT fine-tune then export to INT8.

    Args:
        model_path: Path to .pt checkpoint (pruned or baseline).
        data_yaml: Dataset YAML.
        output_dir: Output directory.
        epochs: Fine-tuning epochs.
        imgsz: Input size.
        batch: Batch size.
        seed: Training seed.
        device: CUDA device.

    Returns:
        dict with model_path (.pt) and engine_path (.engine).
    """
    candidate_weights = [
        Path(output_dir) / 'train' / 'qat' / 'weights' / 'best.pt',
        Path('runs') / 'detect' / output_dir / 'train' / 'qat' / 'weights' / 'best.pt',
    ]
    best_weights = None
    for p in candidate_weights:
        if p.exists() and p.stat().st_size > 0:
            best_weights = p
            break

    if best_weights is None:
        model = YOLO(str(model_path))
        model.train(
            trainer=PrunedDetectionTrainer,
            data=str(data_yaml),
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            patience=0,
            amp=True,
            lr0=0.001,
            lrf=0.01,
            seed=seed,
            project=str(Path(output_dir) / 'train'),
            name='qat',
            exist_ok=True,
            device=device,
        )
        for p in candidate_weights:
            if p.exists() and p.stat().st_size > 0:
                best_weights = p
                break

    if best_weights is None or not best_weights.exists():
        raise FileNotFoundError(f"Could not locate best.pt in any candidate paths: {candidate_weights}")

    engine_path = export_ptq(
        str(best_weights), 'int8', str(output_dir),
        data_yaml=data_yaml, imgsz=imgsz, device=device,
    )

    return {
        'model_path': str(best_weights),
        'engine_path': str(engine_path),
    }

