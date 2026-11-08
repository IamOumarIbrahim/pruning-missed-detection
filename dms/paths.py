"""Repository layout and the experiment configuration (single source of truth)."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RGB = DATA / "processed" / "RGB"
PARTITIONS = DATA / "partitions"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
DETECTIONS = RESULTS / "detections"  # raw low-confidence detections, git-ignored
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
CFG = yaml.safe_load((ROOT / "configs" / "experiment.yaml").read_text(encoding="utf-8"))


def enter_root() -> None:
    """Run from the repo root: dataset YAMLs use paths relative to the working directory."""
    os.chdir(ROOT)
