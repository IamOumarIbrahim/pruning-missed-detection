"""Structured L1 channel pruning for Ultralytics YOLO detectors.

Procedure (one shot, then fine-tuning):
  1. Replace every C3k2 block by C3k2_v2, which splits the shared `chunk(2, 1)` into two
     independent convolutions without changing the function the block computes.
  2. Build the dependency graph with torch-pruning, so coupled channels (concatenations,
     residual links) are removed together as one group.
  3. Rank each group by the L1 norm of its filters and remove the weakest `ratio` of the
     channels of that group. Ranking is local to the group, not global across the network.
  4. Fine-tune the pruned network with `PrunedDetectionTrainer`.

Excluded from pruning: the detection head, C2PSA spatial attention, and any convolution
with fewer than `pruning.min_channels` output channels.
"""
from __future__ import annotations

import copy
import shutil
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch_pruning as tp
import ultralytics.nn.modules as um
import ultralytics.nn.modules.block as ub
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.modules.block import C3k2, Conv

from .paths import CFG

EXCLUDED_MODULES = ("Detect", "v10Detect", "WorldDetect", "C2PSA", "PSABlock", "Attention")


class C3k2_v2(nn.Module):
    """C3k2 with two separate input convolutions, so its branches can be pruned independently."""

    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv0 = Conv(c1, self.c, 1, 1)
        self.cv1 = Conv(c1, self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)

    def forward(self, x):
        y = [self.cv0(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# Checkpoints pickle the class by module path. Keep the earlier path `prune.pruner` loadable.
setattr(ub, "C3k2_v2", C3k2_v2)
setattr(um, "C3k2_v2", C3k2_v2)
sys.modules.setdefault("prune", sys.modules[__name__])
sys.modules.setdefault("prune.pruner", sys.modules[__name__])
sys.modules["__main__"].C3k2_v2 = C3k2_v2


def _split_c3k2(old: C3k2) -> C3k2_v2:
    new = C3k2_v2(old.cv1.conv.in_channels, old.cv2.conv.out_channels, len(old.m),
                  old.c / old.cv2.conv.out_channels)
    new.cv2, new.m = old.cv2, old.m
    half = old.cv1.conv.weight.shape[0] // 2
    for attr, part in (("cv0", slice(None, half)), ("cv1", slice(half, None))):
        target = getattr(new, attr)
        target.conv.weight.data.copy_(old.cv1.conv.weight.data[part])
        for name in ("weight", "bias", "running_mean", "running_var"):
            getattr(target.bn, name).data.copy_(getattr(old.cv1.bn, name).data[part])
    for attr in ("i", "f", "type", "np"):
        if hasattr(old, attr):
            setattr(new, attr, getattr(old, attr))
    return new


def _replace_c3k2(module: nn.Module) -> None:
    for name, child in module.named_children():
        if isinstance(child, C3k2):
            setattr(module, name, _split_c3k2(child))
        else:
            _replace_c3k2(child)


class PrunedDetectionTrainer(DetectionTrainer):
    """Fine-tune the pruned module as is, instead of rebuilding the network from its YAML."""

    def get_model(self, cfg=None, weights=None, verbose=True):
        return weights if isinstance(weights, nn.Module) else super().get_model(cfg, weights, verbose)


def prune_checkpoint(source: Path, ratio: float, target: Path, imgsz: int = 640) -> Path:
    """Prune a trained checkpoint and save it in Ultralytics format."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if ratio <= 0:
        shutil.copy2(source, target)
        return target

    model = YOLO(str(source))
    net = copy.deepcopy(model.model)
    _replace_c3k2(net)
    for p in net.parameters():
        p.requires_grad = True

    min_channels = CFG["pruning"]["min_channels"]
    ignored = [m for m in net.modules()
               if type(m).__name__ in EXCLUDED_MODULES
               or (isinstance(m, nn.Conv2d) and m.out_channels < min_channels)]
    device = next(net.parameters()).device
    pruner = tp.pruner.GroupNormPruner(
        model=net,
        example_inputs=torch.randn(1, 3, imgsz, imgsz, requires_grad=True).to(device),
        importance=tp.importance.GroupMagnitudeImportance(p=1),
        iterative_steps=1,
        pruning_ratio=ratio,
        global_pruning=False,
        ignored_layers=ignored,
    )
    pruner.step()

    net.eval()
    for p in net.parameters():
        p.requires_grad = False
    with torch.no_grad():
        net(torch.randn(1, 3, imgsz, imgsz).to(device))
    torch.save({"model": net, "optimizer": None, "train_args": model.overrides,
                "date": datetime.now().isoformat(), "epoch": -1}, target)
    return target
