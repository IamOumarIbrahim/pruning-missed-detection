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


def tradeoff(p1: pd.DataFrame, p3: pd.DataFrame, floor: float, recommended: dict) -> None:
    """Every INT8 candidate against compute: the safe zone, and the cheapest point inside it."""
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.axhspan(100 * floor, 100, color=AQUA, alpha=0.08, lw=0)
    ax.axhline(100 * floor, color=INK, ls="--", lw=1)
    int8 = p3[p3.quant != "fp32"]
    for method, (label, colour, marker) in METHODS.items():
        f = int8[int8.method == method]
        safe, unsafe = f[f.safe], f[~f.safe]
        ax.plot(safe.gflops, 100 * safe.test_macro_recall, marker, color=colour, ms=8, mec="white", mew=1,
                ls="none", label=label)
        ax.plot(unsafe.gflops, 100 * unsafe.test_macro_recall, marker, color=colour, ms=7, mfc="white",
                mew=1.5, ls="none")
    base = p1[(p1.arch == "yolo11n") & (p1.sparsity == 0) & (p1.quant == "fp32")].iloc[0]
    ax.plot(base.gflops, 100 * base.test_macro_recall, "D", color=INK, ms=7, ls="none",
            label="Baseline")
    rec = int8.merge(pd.DataFrame([recommended]), on=["method", "resolution", "sparsity", "quant"]).iloc[0]
    ax.plot(rec.gflops, 100 * rec.test_macro_recall, "o", ms=17, mfc="none", mec=INK, mew=1.4, ls="none",
            label="Recommended")
    ax.set_xscale("log")
    ax.set(xlim=(0.1, 12), ylim=(40, 100), xlabel="Compute (GFLOPs, log scale)", ylabel="Test macro-recall (%)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=5, frameon=False, fontsize=9,
              handletextpad=0.3, columnspacing=1.2)
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
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=9,
              handletextpad=0.3, columnspacing=1.2)
    fig.tight_layout()
    fig.savefig(FIGURES / "calibration.png")
    plt.close(fig)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    import json
    summary = json.loads((TABLES / "summary.json").read_text())
    p1, p3 = pd.read_csv(TABLES / "phase1.csv"), pd.read_csv(TABLES / "phase3.csv")
    tradeoff(p1, p3, summary["r_floor"], summary["recommended"])
    legacy = pd.read_csv(ROOT / "docs" / "archive" / "legacy_results" / "runs_ptq_minmax_calibration.csv")
    calibration(p1, legacy)
    print("saved", *sorted(p.name for p in FIGURES.glob("*.png")))


if __name__ == "__main__":
    main()
