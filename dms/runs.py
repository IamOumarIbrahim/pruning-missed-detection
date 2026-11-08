"""Registry of every model in the paper: what it is and where its weights live."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .paths import CFG, MODELS

ARCHS = ("yolo11n", "yolo26n")
METHODS = ("baseline", "cropped", "collage")
SEEDS = tuple(CFG["seeds"])
RATIOS = tuple(int(round(r * 100)) for r in CFG["pruning"]["ratios"])


@dataclass(frozen=True)
class Run:
    id: str
    phase: int
    arch: str
    sparsity: int
    quant: str  # fp32 | ptq | qat
    method: str
    resolution: int
    seed: int
    weights: Path

    @property
    def imgsz(self) -> int:
        """Network input size. Inputs are multiples of the maximum stride, so 80 becomes 96."""
        return 96 if self.resolution == 80 else self.resolution


def checkpoint(arch: str, sparsity: int, seed: int) -> Path:
    """FP32 checkpoint of Phase I."""
    leaf = "baseline" if sparsity == 0 else f"pruning_fp32/{sparsity}pct"
    return MODELS / arch / leaf / f"seed_{seed}" / "weights" / "best.pt"


def quantized_dir(arch: str, quant: str, sparsity: int, seed: int) -> Path:
    return MODELS / arch / quant / f"{sparsity}pct" / f"seed_{seed}"


def codesign_dir(method: str, resolution: int, sparsity: int, seed: int) -> Path:
    return MODELS / "codesign" / f"{method}{resolution}_s{sparsity}" / f"seed_{seed}"


def phase1() -> list[Run]:
    runs = []
    for arch in ARCHS:
        for s in (0, *RATIOS):
            for seed in SEEDS:
                runs.append(Run(f"p1/{arch}/s{s}/fp32/seed{seed}", 1, arch, s, "fp32", "baseline",
                                640, seed, checkpoint(arch, s, seed)))
    for s in (0, *RATIOS):
        for seed in SEEDS:
            for quant in ("ptq", "qat"):
                weights = quantized_dir("yolo11n", quant, s, seed) / "model_int8.onnx"
                runs.append(Run(f"p1/yolo11n/s{s}/{quant}/seed{seed}", 1, "yolo11n", s, quant,
                                "baseline", 640, seed, weights))
    return runs


def phase2() -> list[Run]:
    runs = []
    for method in METHODS:
        for res in CFG["input"]["resolutions"]:
            for seed in SEEDS:
                weights = MODELS / "input" / f"{method}_{res}x{res}" / f"seed_{seed}" / "weights" / "best.pt"
                runs.append(Run(f"p2/{method}/{res}/seed{seed}", 2, "yolo11n", 0, "fp32", method,
                                res, seed, weights))
    return runs


def phase3() -> list[Run]:
    runs = []
    root = MODELS / "codesign"
    if not root.exists():
        return runs
    for folder in sorted(root.iterdir()):
        match = re.fullmatch(r"(baseline|cropped|collage)(\d+)_s(\d+)", folder.name)
        if not match:
            continue
        method, res, sparsity = match[1], int(match[2]), int(match[3])
        for seed in SEEDS:
            base = folder / f"seed_{seed}"
            for quant, weights in (("fp32", base / "weights" / "best.pt"),
                                   ("ptq", base / "ptq" / "model_int8.onnx"),
                                   ("qat", base / "qat" / "model_int8.onnx")):
                if weights.exists():
                    runs.append(Run(f"p3/{folder.name}/{quant}/seed{seed}", 3, "yolo11n",
                                    sparsity, quant, method, res, seed, weights))
    return runs
