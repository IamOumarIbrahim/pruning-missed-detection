"""L1 structured channel pruning for Ultralytics YOLO models.

Uses torch_pruning (DepGraph) to resolve cross-layer dependencies
automatically. Supports YOLO models with C3k2, C2f, and C2PSA blocks.
Detection heads, spatial attention, and layers with < 8 channels are preserved.
"""

import copy
from datetime import datetime
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch_pruning as tp
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.modules.block import Conv, Bottleneck, C3k2, C3k, C2f
import ultralytics.nn.modules as um
import ultralytics.nn.modules.block as ub


class C3k2_v2(nn.Module):
    """Pruning-friendly CSP Bottleneck with 2 separate input convolutions instead of chunk(2, 1)."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv0 = Conv(c1, self.c, 1, 1)
        self.cv1 = Conv(c1, self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)

    def forward(self, x):
        y = [self.cv0(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# Register C3k2_v2 into ultralytics namespace so torch_safe_load and pickle can always find it
setattr(ub, 'C3k2_v2', C3k2_v2)
setattr(um, 'C3k2_v2', C3k2_v2)
sys.modules['__main__'].C3k2_v2 = C3k2_v2


def transfer_c3k2_weights(orig, new_m):
    """Transfer weights from C3k2 to C3k2_v2 with exact numerical equivalence."""
    new_m.cv2 = orig.cv2
    new_m.m = orig.m

    # Split cv1 weights and biases across cv0 and cv1
    old_w = orig.cv1.conv.weight.data
    half = old_w.shape[0] // 2
    new_m.cv0.conv.weight.data.copy_(old_w[:half])
    new_m.cv1.conv.weight.data.copy_(old_w[half:])

    old_bn_w = orig.cv1.bn.weight.data
    new_m.cv0.bn.weight.data.copy_(old_bn_w[:half])
    new_m.cv1.bn.weight.data.copy_(old_bn_w[half:])

    old_bn_b = orig.cv1.bn.bias.data
    new_m.cv0.bn.bias.data.copy_(old_bn_b[:half])
    new_m.cv1.bn.bias.data.copy_(old_bn_b[half:])

    old_bn_rm = orig.cv1.bn.running_mean.data
    new_m.cv0.bn.running_mean.data.copy_(old_bn_rm[:half])
    new_m.cv1.bn.running_mean.data.copy_(old_bn_rm[half:])

    old_bn_rv = orig.cv1.bn.running_var.data
    new_m.cv0.bn.running_var.data.copy_(old_bn_rv[:half])
    new_m.cv1.bn.running_var.data.copy_(old_bn_rv[half:])

    for attr in ['i', 'f', 'type', 'np']:
        if hasattr(orig, attr):
            setattr(new_m, attr, getattr(orig, attr))


def replace_c3k2_recursive(module):
    """Recursively replace all C3k2 modules with C3k2_v2."""
    for name, child in module.named_children():
        if isinstance(child, C3k2):
            shortcut = getattr(child.m[0], 'add', True) if len(child.m) > 0 else True
            new_m = C3k2_v2(
                c1=child.cv1.conv.in_channels,
                c2=child.cv2.conv.out_channels,
                n=len(child.m),
                e=child.c / child.cv2.conv.out_channels,
                shortcut=shortcut,
            )
            transfer_c3k2_weights(child, new_m)
            setattr(module, name, new_m)
        else:
            replace_c3k2_recursive(child)


class PrunedDetectionTrainer(DetectionTrainer):
    """Ultralytics DetectionTrainer that fine-tunes pruned architectures directly.
    
    Prevents Ultralytics from re-instantiating the unpruned baseline model
    from YAML configs during model.train().
    """

    def get_model(self, cfg=None, weights=None, verbose=True):
        if isinstance(weights, nn.Module):
            return weights
        return super().get_model(cfg, weights, verbose)


def prune_model(model_path, pruning_ratio, output_path, imgsz=640):
    """Apply global L1-norm structured channel pruning.

    Channels are ranked by L1 norm of their Conv2d filters. The bottom
    ``pruning_ratio`` fraction is removed. Detection heads, spatial attention,
    and layers with fewer than 8 output channels are excluded.

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
    nn_model = copy.deepcopy(model.model)
    device = next(nn_model.parameters()).device

    # Convert C3k2 to C3k2_v2 for channel independence
    replace_c3k2_recursive(nn_model)

    # Enable gradients so torch_pruning autograd tracer can trace full graph
    for p in nn_model.parameters():
        p.requires_grad = True

    example_inputs = torch.randn(1, 3, imgsz, imgsz, requires_grad=True).to(device)

    # Layers to exclude from pruning
    ignored_layers = []
    for m in nn_model.modules():
        if m.__class__.__name__ in (
            'Detect', 'Segment', 'Pose', 'OBB', 'Classify',
            'v10Detect', 'WorldDetect',
            'C2PSA', 'PSABlock', 'Attention',
        ):
            ignored_layers.append(m)
        elif isinstance(m, nn.Conv2d) and m.out_channels < 8:
            ignored_layers.append(m)

    pruner = tp.pruner.GroupNormPruner(
        model=nn_model,
        example_inputs=example_inputs,
        importance=tp.importance.GroupMagnitudeImportance(p=1),  # L1-norm pruning
        iterative_steps=1,
        pruning_ratio=pruning_ratio,
        ignored_layers=ignored_layers,
    )
    pruner.step()

    # Reset requires_grad and test dummy forward
    nn_model.eval()
    for p in nn_model.parameters():
        p.requires_grad = False

    with torch.no_grad():
        _ = nn_model(torch.randn(1, 3, imgsz, imgsz).to(device))

    # Save in Ultralytics checkpoint format
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
