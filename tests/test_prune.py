"""The block rewrite before pruning must not change what the network computes."""
import copy

import torch
from ultralytics.nn.tasks import DetectionModel

from dms import prune


def test_c3k2_split_preserves_the_output():
    """Near-exact, not bitwise: the new BatchNorm layers keep PyTorch's default epsilon."""
    model = DetectionModel("yolo11n.yaml", ch=3, nc=4, verbose=False).eval()
    split = copy.deepcopy(model)
    prune._replace_c3k2(split)
    split.eval()
    assert any(type(m).__name__ == "C3k2_v2" for m in split.modules())

    x = torch.rand(1, 3, 320, 320)
    with torch.no_grad():
        original, rewritten = model(x), split(x)
    assert (original[0] - rewritten[0]).abs().max() < 0.05
