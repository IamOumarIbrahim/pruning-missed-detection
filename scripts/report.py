"""Aggregate results/tables/runs.csv into the paper's summary tables and uncertainty table."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dms import evaluate, runs, stats  # noqa: E402
from dms.paths import CFG, TABLES, enter_root  # noqa: E402

CLASSES = CFG["classes"]
DELTA = CFG["safety_margin"]
METRICS = ["test_macro_recall", "test_macro_precision", "test_macro_f1", "test_worst_recall",
           "val_macro_recall", "tuned_macro_recall", "tuned_macro_precision", "tuned_macro_f1",
           "strict_macro_recall", "map50", "map50_95"] + [f"test_recall_{c}" for c in CLASSES]


def summarize(frame: pd.DataFrame, keys: list) -> pd.DataFrame:
    """Mean and sample SD (ddof=1) of every metric over seeds."""
    grouped = frame.groupby(keys, sort=False)
    out = grouped[METRICS].mean()
    out = out.join(grouped[METRICS].std(ddof=1).add_suffix("_sd"))
    out["tuned_tau"] = grouped["tuned_tau"].first()
    out["seeds"] = grouped.size()
    return out.reset_index()


def cost_columns(table: pd.DataFrame, profiles: pd.DataFrame, keys: list) -> pd.DataFrame:
    """Attach cost, and pick the file size that matches each row's quantization."""
    out = table.merge(profiles.drop_duplicates(keys), on=keys, how="left")
    out["file_mb"] = np.where(out.quant == "ptq", out.ptq_mb,
                              np.where(out.quant == "qat", out.qat_mb, out.fp32_mb))
    return out


def phase_runs(group: pd.DataFrame, split: str = "test") -> list:
    """(scored split, threshold) of each seed, for the clip bootstrap."""
    registry = {r.id: r for r in runs.phase1() + runs.phase2() + runs.phase3()}
    return [(evaluate.scored_run(registry[i])[split], evaluate.TAU) for i in group["id"]]


def main():
    enter_root()
    all_runs = pd.read_csv(TABLES / "runs.csv")
    profiles = pd.read_csv(TABLES / "profiles.csv")

    p1 = summarize(all_runs[all_runs.phase == 1], ["arch", "sparsity", "quant"])
    p1 = cost_columns(p1, profiles[profiles.phase == 1], ["arch", "sparsity"])
    reference = {a: p1[(p1.arch == a) & (p1.sparsity == 0) & (p1.quant == "fp32")].iloc[0]
                 for a in runs.ARCHS}
    p1["floor"] = [reference[a].test_macro_recall - DELTA for a in p1.arch]
    p1["safe"] = p1.test_macro_recall >= p1.floor
    p1["eff_flops"] = 100 * p1.test_macro_recall / p1.gflops
    p1["eff_mb"] = 100 * p1.test_macro_recall / p1.file_mb
    p1.to_csv(TABLES / "phase1.csv", index=False)

    floor = reference["yolo11n"].test_macro_recall - DELTA
    val_floor = reference["yolo11n"].val_macro_recall - DELTA
    p2 = summarize(all_runs[all_runs.phase == 2], ["method", "resolution", "quant"])
    p2 = cost_columns(p2, profiles[profiles.phase == 2], ["method", "resolution"])
    p2["safe"] = p2.test_macro_recall >= floor
    p2["recall_per_gflop"] = p2.test_macro_recall / p2.gflops
    p2.to_csv(TABLES / "phase2.csv", index=False)

    p3 = summarize(all_runs[all_runs.phase == 3], ["method", "resolution", "sparsity", "quant"])
    p3 = cost_columns(p3, profiles[profiles.phase == 3], ["method", "resolution", "sparsity"])
    p3["safe"] = p3.test_macro_recall >= floor
    p3["val_passes"] = p3.val_macro_recall >= val_floor
    p3.to_csv(TABLES / "phase3.csv", index=False)

    # Uncertainty against the FP32 YOLO11n baseline, on identical resampled clips.
    base = all_runs[(all_runs.phase == 1) & (all_runs.arch == "yolo11n") & (all_runs.sparsity == 0)
                    & (all_runs.quant == "fp32")]
    base_runs = phase_runs(base)
    rows = []
    for keys, group in all_runs[all_runs.phase == 3].groupby(["method", "resolution", "sparsity", "quant"]):
        cand = phase_runs(group)
        ci, delta = stats.macro_recall_ci(cand), stats.paired_delta(cand, base_runs, DELTA)
        rows.append(dict(zip(["method", "resolution", "sparsity", "quant"], keys))
                    | {"ci_low": ci["macro_low"], "ci_high": ci["macro_high"],
                       "delta": delta["delta"], "delta_low": delta["low"], "delta_high": delta["high"],
                       "prob_within_margin": delta["prob_within_margin"]}
                    | {f"{c}_low": float(ci["class_low"][i]) for i, c in enumerate(CLASSES)}
                    | {f"{c}_high": float(ci["class_high"][i]) for i, c in enumerate(CLASSES)})
    base_ci = stats.macro_recall_ci(base_runs)
    rows.append({"method": "reference", "resolution": 640, "sparsity": 0, "quant": "fp32",
                 "ci_low": base_ci["macro_low"], "ci_high": base_ci["macro_high"]}
                | {f"{c}_low": float(base_ci["class_low"][i]) for i, c in enumerate(CLASSES)}
                | {f"{c}_high": float(base_ci["class_high"][i]) for i, c in enumerate(CLASSES)})
    pd.DataFrame(rows).to_csv(TABLES / "uncertainty.csv", index=False)

    # Paired non-inferiority on validation: candidate minus baseline on the same resampled clips.
    base_val = phase_runs(base, "val")
    deltas = []
    for keys, group in all_runs[all_runs.phase == 3].groupby(["method", "resolution", "sparsity", "quant"]):
        d = stats.paired_delta(phase_runs(group, "val"), base_val, DELTA, split="val")
        deltas.append(dict(zip(["method", "resolution", "sparsity", "quant"], keys))
                      | {"val_delta": d["delta"], "val_delta_low": d["low"]})
    p3 = p3.merge(pd.DataFrame(deltas), on=["method", "resolution", "sparsity", "quant"], how="left")
    p3.to_csv(TABLES / "phase3.csv", index=False)

    # The selection rule (validation only): cheapest INT8 candidate with mean validation recall
    # above the validation floor. The recommendation is reported separately and uses test data:
    # the cheapest INT8 candidate whose paired test difference stays within the margin at 95%.
    keys = ["method", "resolution", "sparsity", "quant"]
    int8 = p3[p3.quant != "fp32"].merge(pd.DataFrame(rows)[keys + ["delta_low"]], on=keys, how="left")
    int8 = int8.sort_values(["gflops", "file_mb"])
    pick = lambda frame: None if frame.empty else {
        k: (v.item() if hasattr(v, "item") else v) for k, v in frame.iloc[0][keys].items()}
    selected = int8[int8.val_passes]
    recommended = int8[int8.delta_low >= -DELTA]

    summary = {"r_base": float(reference["yolo11n"].test_macro_recall), "r_floor": float(floor),
               "val_r_base": float(reference["yolo11n"].val_macro_recall), "val_r_floor": float(val_floor),
               "seed_sd_fixed": float(np.mean(p1.test_macro_recall_sd)),
               "seed_sd_tuned": float(np.mean(p1.tuned_macro_recall_sd)),
               "selected": pick(selected), "recommended": pick(recommended)}
    (TABLES / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
