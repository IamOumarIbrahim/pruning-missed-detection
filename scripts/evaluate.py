"""Run inference for every available model and cache its detections.

  python scripts/evaluate.py                 # all phases
  python scripts/evaluate.py --only fp32     # GPU models only
  python scripts/evaluate.py --only ptq qat  # ONNX models, on the CPU

Results are cached per run, so an interrupted evaluation resumes, and two processes can
work on different subsets at once. `scripts/score.py` turns the cache into metrics.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dms import evaluate, runs  # noqa: E402
from dms.paths import DETECTIONS, enter_root  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=["fp32", "ptq", "qat"])
    parser.add_argument("--phase", nargs="*", type=int, default=[1, 2, 3])
    args = parser.parse_args()
    enter_root()
    registry = runs.phase1() + runs.phase2() + runs.phase3()
    todo = [r for r in registry if r.quant in args.only and r.phase in args.phase and r.weights.exists()]
    print(f"{len(todo)} runs", flush=True)
    for run in todo:
        for split in ("val", "test"):
            if not (DETECTIONS / run.id / f"{split}.npz").exists():
                evaluate.detections(run.id, run.weights, run.method, run.resolution, run.imgsz, split)


if __name__ == "__main__":
    main()
