"""Train every model of the paper. Finished runs are skipped, so a command can be re-issued.

  python scripts/train.py baseline               # Phase I: YOLO11n and YOLO26n
  python scripts/train.py prune                  # Phase I: pruned and fine-tuned, 10-90%
  python scripts/train.py input                  # Phase II: 3 input methods x 4 resolutions
  python scripts/train.py ptq                    # Phase I: INT8 PTQ of every FP32 model
  python scripts/train.py qat                    # Phase I: INT8 QAT of every FP32 model
  python scripts/train.py codesign --method cropped --resolution 160 --sparsity 30
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dms import quant, runs  # noqa: E402
from dms.data import dataset_yaml  # noqa: E402
from dms.paths import CFG, MODELS, enter_root  # noqa: E402
from dms.prune import PrunedDetectionTrainer, prune_checkpoint  # noqa: E402
import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402

enter_root()
TRAIN = CFG["training"]
QUANT = CFG["quantization"]
RGB_YAML = "configs/dmd_rgb.yaml"


def is_complete(best):
    """Ultralytics strips the optimizer only when a run ends, so a crash leaves it in place."""
    if not best.exists():
        return False
    return torch.load(best, map_location="cpu", weights_only=False).get("optimizer") is None


def fit(weights, out_dir, data, imgsz, seed, pruned, **overrides):
    """Train one run into out_dir and return its best checkpoint."""
    best = Path(out_dir) / "weights" / "best.pt"
    if is_complete(best):
        return best
    args = dict(data=str(data), imgsz=imgsz, seed=seed, device=0, amp=True, exist_ok=True, plots=False,
                project=str(Path(out_dir).parent.resolve()), name=Path(out_dir).name, **TRAIN)
    args.update(overrides)
    if pruned:
        args["trainer"] = PrunedDetectionTrainer
    YOLO(str(weights)).train(**args)
    return best


def baseline():
    for arch in runs.ARCHS:
        for seed in runs.SEEDS:
            fit(MODELS / "pretrained" / f"{arch}.pt", MODELS / arch / "baseline" / f"seed_{seed}",
                RGB_YAML, 640, seed, pruned=False)


def prune():
    for arch in runs.ARCHS:
        for ratio in runs.RATIOS:
            for seed in runs.SEEDS:
                init = prune_checkpoint(runs.checkpoint(arch, 0, seed), ratio / 100,
                                        MODELS / arch / "pruned" / f"{ratio}pct" / f"seed_{seed}_pruned.pt")
                fit(init, MODELS / arch / "pruning_fp32" / f"{ratio}pct" / f"seed_{seed}",
                    RGB_YAML, 640, seed, pruned=True)


def input_sweep():
    for method in runs.METHODS:
        for res in CFG["input"]["resolutions"]:
            for seed in runs.SEEDS:
                fit(MODELS / "pretrained" / "yolo11n.pt",
                    MODELS / "input" / f"{method}_{res}x{res}" / f"seed_{seed}",
                    dataset_yaml(method, res), res, seed, pruned=False)


def export_ptq(checkpoint, target, method, resolution, imgsz):
    """FP32 ONNX next to the INT8 graph: both are measured for the size tables."""
    fp32 = Path(target).with_name("model_fp32.onnx")
    if not fp32.exists():
        quant.export_onnx(checkpoint, fp32, imgsz)
    if not Path(target).exists():
        quant.quantize(fp32, target, method, resolution, imgsz)


def ptq():
    for sparsity in (0, *runs.RATIOS):
        for seed in runs.SEEDS:
            export_ptq(runs.checkpoint("yolo11n", sparsity, seed),
                       runs.quantized_dir("yolo11n", "ptq", sparsity, seed) / "model_int8.onnx",
                       "baseline", 640, 640)


def train_qat(checkpoint, out_dir, data, imgsz, seed):
    """True QAT: fake-quantized forward and backward passes, then Q/DQ export of the learned ranges."""
    best = fit(checkpoint, Path(out_dir) / "train", data, imgsz, seed, pruned=True, quantize=8, amp=False,
               epochs=QUANT["qat_epochs"], optimizer="SGD", lr0=QUANT["qat_lr0"], momentum=0.9,
               close_mosaic=0, warmup_epochs=0.0, workers=4)
    target = Path(out_dir) / "model_int8.onnx"
    if not target.exists():
        with quant.scratch_copy(best) as local:
            exported = YOLO(str(local)).export(format="onnx", imgsz=imgsz, opset=17, simplify=False,
                                               dynamic=False, device="cpu", verbose=False)
            quant.fold_weight_quantizers(exported, target)
    return target


def qat():
    for sparsity in (0, *runs.RATIOS):
        for seed in runs.SEEDS:
            train_qat(runs.checkpoint("yolo11n", sparsity, seed),
                      runs.quantized_dir("yolo11n", "qat", sparsity, seed), RGB_YAML, 640, seed)


def codesign(method, resolution, sparsity, use_qat):
    """Start from the model already trained on this input, prune it, fine-tune it, quantize it."""
    data = dataset_yaml(method, resolution)
    for seed in runs.SEEDS:
        base = runs.codesign_dir(method, resolution, sparsity, seed)
        source = MODELS / "input" / f"{method}_{resolution}x{resolution}" / f"seed_{seed}" / "weights" / "best.pt"
        if sparsity == 0:
            best = base / "weights" / "best.pt"
            best.parent.mkdir(parents=True, exist_ok=True)
            if not best.exists():
                shutil.copy2(source, best)
        else:
            init = prune_checkpoint(source, sparsity / 100, base / "init_pruned.pt", imgsz=resolution)
            best = fit(init, base, data, resolution, seed, pruned=True, **CFG["codesign"])
        export_ptq(best, base / "ptq" / "model_int8.onnx", method, resolution, resolution)
        if use_qat:
            train_qat(best, base / "qat", data, resolution, seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("stage", choices=["baseline", "prune", "input", "ptq", "qat", "codesign"])
    parser.add_argument("--method", choices=["baseline", "cropped", "collage"], default="collage")
    parser.add_argument("--resolution", type=int, default=320)
    parser.add_argument("--sparsity", type=int, default=70)
    parser.add_argument("--qat", action="store_true", help="codesign: also train a QAT variant")
    args = parser.parse_args()
    if args.stage == "codesign":
        codesign(args.method, args.resolution, args.sparsity, args.qat)
    else:
        {"baseline": baseline, "prune": prune, "input": input_sweep, "ptq": ptq, "qat": qat}[args.stage]()


if __name__ == "__main__":
    main()
