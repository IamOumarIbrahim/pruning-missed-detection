"""L1 structured channel pruning for Ultralytics YOLO models.

Uses torch_pruning (DepGraph) to resolve cross-layer dependencies
automatically. The detection head is excluded from pruning.
"""

import copy
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch_pruning as tp
from ultralytics import YOLO


def prune_model(model_path, pruning_ratio, output_path, imgsz=640):
    """Apply global L1-norm structured channel pruning.

    Channels are ranked by L1 norm of their Conv2d filters. The bottom
    ``pruning_ratio`` fraction is removed. The detection head and layers
    with fewer than 8 output channels are excluded.

    Args:
        model_path: Path to trained Ultralytics checkpoint (.pt).
        pruning_ratio: Fraction of channels to remove (0.0-1.0).
        output_path: Where to save the pruned checkpoint.
        imgsz: Input size for the dependency graph.

    Returns:
        str: Path to the saved pruned model.
    """
    import shutil

    if pruning_ratio <= 0:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(model_path, output_path)
        return str(output_path)

    model = YOLO(str(model_path))
    nn_model = model.model
    device = next(nn_model.parameters()).device

    example_inputs = torch.randn(1, 3, imgsz, imgsz).to(device)

    # Layers to exclude: detection head and very small conv layers.
    ignored_layers = []
    for m in nn_model.modules():
        if m.__class__.__name__ in (
            'Detect', 'Segment', 'Pose', 'OBB', 'Classify',
            'v10Detect', 'WorldDetect',
        ):
            ignored_layers.append(m)
        elif isinstance(m, nn.Conv2d) and m.out_channels < 8:
            ignored_layers.append(m)

    importance = tp.importance.MagnitudeImportance(p=1,
                                                   group_reduction='mean')
    pruner = tp.pruner.MetaPruner(
        model=nn_model,
        example_inputs=example_inputs,
        importance=importance,
        pruning_ratio=pruning_ratio,
        ignored_layers=ignored_layers,
    )
    pruner.step()

    # Save in Ultralytics checkpoint format.
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        'model': nn_model,
        'optimizer': None,
        'train_args': model.overrides,
        'date': datetime.now().isoformat(),
        'epoch': -1,
    }
    torch.save(ckpt, output_path)
    return str(output_path)


def get_model_info(model_path):
    """Model size (MB), parameter count, and FLOPs.

    Returns dict with size_mb, num_params, flops_g.
    """
    model = YOLO(str(model_path))
    size_mb = Path(model_path).stat().st_size / (1024 * 1024)
    num_params = sum(p.numel() for p in model.model.parameters())
    try:
        info = model.info(verbose=False)
        flops = info[1] if isinstance(info, (list, tuple)) and len(info) > 1 else None
    except Exception:
        flops = None
    return {
        'size_mb': float(size_mb),
        'num_params': int(num_params),
        'flops_g': float(flops) if flops is not None else None,
    }
