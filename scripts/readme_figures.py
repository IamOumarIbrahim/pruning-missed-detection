"""Figures for the README, drawn from results/tables (run scripts/report.py first)."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dms.paths import FIGURES, ROOT, TABLES  # noqa: E402

BLUE, ORANGE, AQUA, INK, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#52514e", "#e4e3df"
METHODS = {"baseline": ("Resize", BLUE, "o"), "cropped": ("Crop", ORANGE, "s"), "collage": ("Collage", AQUA, "^")}

plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK,
                     "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "figure.dpi": 150})


def tradeoff(p1: pd.DataFrame, p3: pd.DataFrame, floor: float) -> None:
    """Every INT8 co-design candidate: test recall against compute, with the safety floor."""
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    int8 = p3[p3.quant != "fp32"]
    for method, (label, colour, marker) in METHODS.items():
        f = int8[int8.method == method]
        ax.errorbar(f.gflops, 100 * f.test_macro_recall, yerr=100 * f.test_macro_recall_sd, fmt=marker,
                    color=colour, ms=7, mec="white", mew=1, elinewidth=1, capsize=2, label=label)
    base = p1[(p1.arch == "yolo11n") & (p1.sparsity == 0) & (p1.quant == "fp32")].iloc[0]
    ax.plot(base.gflops, 100 * base.test_macro_recall, "D", color=INK, ms=7, label="Baseline (FP32, 640 px)")
    ax.axhline(100 * floor, color=INK, ls="--", lw=1)
    ax.text(0.105, 100 * floor + 1, f"safety floor {100 * floor:.1f}%", color=INK, fontsize=9)
    best = int8[(int8.method == "collage") & (int8.resolution == 320) & (int8.sparsity == 0)].iloc[0]
    ax.annotate("collage 320 px, INT8\n92.9%, 4x less compute", (best.gflops, 100 * best.test_macro_recall),
                xytext=(0.62, 96), fontsize=9, color=INK, arrowprops=dict(arrowstyle="-", color=INK, lw=0.8))
    ax.set_xscale("log")
    ax.set(xlim=(0.1, 10), ylim=(40, 100), xlabel="GFLOPs (log scale)", ylabel="Test macro-recall (%)")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "tradeoff.png")
    plt.close(fig)


def calibration(p1: pd.DataFrame, legacy: pd.DataFrame) -> None:
    """PTQ seed spread before and after moving-average calibration, against FP32."""
    fp32 = p1[(p1.arch == "yolo11n") & (p1.quant == "fp32")].sort_values("sparsity")
    ptq = p1[(p1.arch == "yolo11n") & (p1.quant == "ptq")].sort_values("sparsity")
    old = legacy[(legacy.phase == 1) & (legacy.quant == "ptq")]
    old = old.groupby("sparsity").test_macro_recall.agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.plot(fp32.sparsity, 100 * fp32.test_macro_recall, "-o", color=BLUE, ms=5, lw=2, label="FP32")
    ax.errorbar(old.sparsity - 1.2, 100 * old["mean"], yerr=100 * old["std"], fmt="s", color=ORANGE, ms=5,
                elinewidth=1.2, capsize=3, label="INT8 PTQ, MinMax ranges")
    ax.errorbar(ptq.sparsity + 1.2, 100 * ptq.test_macro_recall, yerr=100 * ptq.test_macro_recall_sd, fmt="^",
                color=AQUA, ms=6, elinewidth=1.2, capsize=3, label="INT8 PTQ, moving-average ranges")
    ax.set(xlim=(-4, 94), ylim=(0, 100), xticks=range(0, 100, 10), xlabel="Target sparsity (%)",
           ylabel="Test macro-recall (%)")
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "calibration.png")
    plt.close(fig)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    import json
    floor = json.loads((TABLES / "summary.json").read_text())["r_floor"]
    p1, p3 = pd.read_csv(TABLES / "phase1.csv"), pd.read_csv(TABLES / "phase3.csv")
    tradeoff(p1, p3, floor)
    legacy = pd.read_csv(ROOT / "docs" / "archive" / "legacy_results" / "runs_ptq_minmax_calibration.csv")
    calibration(p1, legacy)
    print("saved", *sorted(p.name for p in FIGURES.glob("*.png")))


if __name__ == "__main__":
    main()
