"""Generate every table and figure of the paper from results/tables into paper/generated."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dms.data import frame_ground_truth  # noqa: E402
from dms.paths import CFG, ROOT, TABLES, enter_root  # noqa: E402

OUT = ROOT / "paper" / "generated"
CLASSES = CFG["classes"]
NAMES = {"yawning": "Yawning", "hand_over_mouth": "Hand-over-mouth", "drinking": "Drinking",
         "phone_use": "Phone use"}
METHOD = {"baseline": "Resize", "cropped": "Crop", "collage": "Collage"}
COLORS = {"blue": "2a78d6", "orange": "eb6834", "aqua": "1baf7a", "ink": "52514e"}
TICK, CROSS = r"\checkmark", r"$\times$"


def pct(mean, sd=None, digits=1):
    text = f"{100 * mean:.{digits}f}"
    return f"${text}\\pm{100 * sd:.{digits}f}$" if sd is not None else text


def write(name: str, text: str) -> None:
    (OUT / name).write_text(text, encoding="utf-8")


def visible_fraction(boxes: np.ndarray, window) -> np.ndarray:
    x1, y1, x2, y2 = window
    w = np.clip(np.minimum(boxes[:, 2], x2) - np.maximum(boxes[:, 0], x1), 0, None)
    h = np.clip(np.minimum(boxes[:, 3], y2) - np.maximum(boxes[:, 1], y1), 0, None)
    return w * h / ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))


def table_coverage() -> None:
    crop = CFG["input"]["crop_box"]
    tiles = [t["src"] for t in CFG["input"]["collage_tiles"].values()]
    rows = []
    for name in CLASSES:
        cells = []
        for split in ("train", "test"):
            truth = frame_ground_truth(split)
            classes = np.concatenate([g[0] for g in truth])
            boxes = np.concatenate([g[1] for g in truth])[classes == CLASSES.index(name)]
            vis_crop = visible_fraction(boxes, crop)
            vis_tile = np.max([visible_fraction(boxes, t) for t in tiles], axis=0)
            cells += [100 * (vis_crop >= 0.999).mean(), 100 * (vis_crop >= 0.5).mean(),
                      100 * (vis_tile >= 0.999).mean(), 100 * (vis_tile >= 0.5).mean()]
        rows.append(f"{NAMES[name]} & " + " & ".join(f"{c:.0f}" for c in cells) + r" \\")
    body = "\n".join(rows)
    write("table_coverage.tex", rf"""\begin{{table}}[t]
\caption{{Share of boxes (\%) kept by each input method}}
\label{{tab:coverage}}
\centering
\footnotesize
\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{@{{}}lrrrrrrrr@{{}}}}
\toprule
 & \multicolumn{{4}}{{c}}{{\textbf{{Train}}}} & \multicolumn{{4}}{{c}}{{\textbf{{Test}}}} \\
\cmidrule(lr){{2-5}}\cmidrule(lr){{6-9}}
 & \multicolumn{{2}}{{c}}{{Crop}} & \multicolumn{{2}}{{c}}{{Collage}} & \multicolumn{{2}}{{c}}{{Crop}} & \multicolumn{{2}}{{c}}{{Collage}} \\
\textbf{{Class}} & Full & $\geq$50\% & Full & $\geq$50\% & Full & $\geq$50\% & Full & $\geq$50\% \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
""")


def table_pruning(p1: pd.DataFrame) -> None:
    fp32 = p1[p1.quant == "fp32"]
    lines = []
    for arch, label in (("yolo11n", "YOLO11n"), ("yolo26n", "YOLO26n")):
        lines.append(rf"\multicolumn{{11}}{{l}}{{\textit{{{label}}}}} \\")
        for r in fp32[fp32.arch == arch].sort_values("sparsity").itertuples():
            sp = "Base" if r.sparsity == 0 else f"{r.sparsity}\\%"
            per_class = " & ".join(pct(getattr(r, f"test_recall_{c}"), getattr(r, f"test_recall_{c}_sd"))
                                   for c in CLASSES)
            lines.append(f"{sp} & {r.params / 1e6:.2f} & {r.gflops:.2f} & {r.fp32_mb:.2f} & "
                         f"{pct(r.test_macro_precision, r.test_macro_precision_sd)} & {per_class} & "
                         f"{pct(r.test_macro_recall, r.test_macro_recall_sd)} & {r.eff_flops:.1f} & "
                         f"{TICK if r.safe else CROSS} \\\\")
        if arch == "yolo11n":
            lines.append(r"\midrule")
    body = "\n".join(lines)
    write("table_pruning.tex", rf"""\begin{{table*}}[!t]
\renewcommand{{\arraystretch}}{{0.85}}
\caption{{FP32 pruning results on the test set at $\tau=0.5$}}
\label{{tab:pruning_fixed_tau}}
\centering
\scriptsize
\setlength{{\tabcolsep}}{{2.6pt}}
\begin{{tabular}}{{@{{}}lrrrccccccrc@{{}}}}
\toprule
 & \multicolumn{{3}}{{c}}{{Cost}} & & \multicolumn{{4}}{{c}}{{Class recall (\%)}} & & & \\
\cmidrule(lr){{2-4}}\cmidrule(lr){{6-9}}
Sparsity & Params (M) & GFLOPs & Size (MB) & $P$ (\%) & Yawning & Hand-mouth & Drinking & Phone use & $R$ (\%) & $\text{{Eff}}_{{\text{{FLOPs}}}}$ & Safe \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table*}}
""")


def table_quant(p1: pd.DataFrame) -> None:
    fp32 = p1[(p1.arch == "yolo11n") & (p1.quant == "fp32")].set_index("sparsity")
    ptq = p1[(p1.arch == "yolo11n") & (p1.quant == "ptq")].set_index("sparsity")
    qat = p1[(p1.arch == "yolo11n") & (p1.quant == "qat")].set_index("sparsity")
    base = fp32.loc[0]
    lines = []
    for s in fp32.index:
        row = [f"{s}", f"{100 * (1 - fp32.loc[s].params / base.params):.1f}", f"{fp32.loc[s].gflops:.2f}",
               pct(fp32.loc[s].test_macro_recall, fp32.loc[s].test_macro_recall_sd)]
        for frame in (ptq, qat):
            if s in frame.index:
                r = frame.loc[s]
                row += [f"{r.file_mb:.2f}", pct(r.test_macro_precision, r.test_macro_precision_sd),
                        pct(r.test_macro_recall, r.test_macro_recall_sd), f"{r.eff_mb:.1f}",
                        TICK if r.safe else CROSS]
            else:
                row += ["--"] * 5
        lines.append(" & ".join(row) + r" \\")
    body = "\n".join(lines)
    write("table_quant.tex", rf"""\begin{{table*}}[!t]
\renewcommand{{\arraystretch}}{{0.85}}
\caption{{YOLO11n FP32, INT8 PTQ, and INT8 QAT over three seeds at $\tau=0.5$}}
\label{{tab:quantization}}
\centering
\scriptsize
\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{@{{}}rrrc rccrc rccrc@{{}}}}
\toprule
 & & & & \multicolumn{{5}}{{c}}{{INT8 PTQ}} & \multicolumn{{5}}{{c}}{{INT8 QAT}} \\
\cmidrule(lr){{5-9}}\cmidrule(lr){{10-14}}
Sparsity (\%) & Param.\ red.\ (\%) & GFLOPs & FP32 $R$ (\%) & MB & $P$ (\%) & $R$ (\%) & $\text{{Eff}}_{{\text{{MB}}}}$ & Safe & MB & $P$ (\%) & $R$ (\%) & $\text{{Eff}}_{{\text{{MB}}}}$ & Safe \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table*}}
""")


def table_input(p2: pd.DataFrame) -> None:
    lines = []
    for method in ("baseline", "cropped", "collage"):
        lines.append(rf"\multicolumn{{9}}{{l}}{{\textit{{{METHOD[method]}}}}} \\")
        for r in p2[p2.method == method].sort_values("resolution", ascending=False).itertuples():
            lines.append(f"{r.resolution} & {r.gflops:.2f} & {pct(r.test_macro_precision, r.test_macro_precision_sd)} & "
                         f"{pct(r.test_macro_recall, r.test_macro_recall_sd)} & "
                         f"{pct(r.test_macro_f1, r.test_macro_f1_sd)} & {100 * r.map50:.1f} & {100 * r.map50_95:.1f} & "
                         f"{pct(r.strict_macro_recall)} & {TICK if r.safe else CROSS} \\\\")
        if method != "collage":
            lines.append(r"\midrule")
    body = "\n".join(lines)
    write("table_input.tex", rf"""\begin{{table*}}[!t]
\renewcommand{{\arraystretch}}{{0.9}}
\caption{{FP32 YOLO11n by input method and resolution at $\tau=0.5$}}
\label{{tab:accuracy}}
\centering
\footnotesize
\setlength{{\tabcolsep}}{{5pt}}
\begin{{tabular}}{{@{{}}rrccc rrrc@{{}}}}
\toprule
Input (px) & GFLOPs & $P$ (\%) & $R$ (\%) & $F1$ (\%) & mAP@0.5 & mAP@0.5:0.95 & $R$ strict\textsuperscript{{a}} (\%) & Safe \\
\midrule
{body}
\bottomrule
\multicolumn{{9}}{{l}}{{\textsuperscript{{a}}A box counts only if at least 50\% of it stays visible.}} \\
\end{{tabular}}
\end{{table*}}
""")


def same(row, choice) -> bool:
    return bool(choice) and (row.method, row.resolution, row.sparsity, row.quant) == (
        choice["method"], choice["resolution"], choice["sparsity"], choice["quant"])


def table_codesign(p3: pd.DataFrame, p1: pd.DataFrame, uncertainty: pd.DataFrame, summary: dict) -> None:
    """INT8 recall on the full grid: input method x resolution x sparsity, cost once per column."""
    int8 = p3[p3.quant != "fp32"]
    sparsities = sorted(int8.sparsity.unique())
    base = p1[(p1.arch == "yolo11n") & (p1.sparsity == 0) & (p1.quant == "fp32")].iloc[0]
    lines = []
    for res in sorted(int8.resolution.unique(), reverse=True):
        block = int8[int8.resolution == res]
        cost = []
        for sp in sparsities:
            c = block[block.sparsity == sp]
            cost.append(f"{c.gflops.iloc[0]:.2f} / {c.file_mb.iloc[0]:.2f}" if len(c) else "--")
        lines.append(rf"\multicolumn{{2}}{{@{{}}l}}{{\textit{{{res} px}}: GFLOPs / MB}} & " + " & ".join(cost) + r" \\")
        lines.append(r"\cmidrule(l){3-" + str(2 + len(sparsities)) + "}")
        for method in ("baseline", "cropped", "collage"):
            cells = []
            for sp in sparsities:
                c = block[(block.method == method) & (block.sparsity == sp)]
                if c.empty:
                    cells.append("--")
                    continue
                r = c.iloc[0]
                text = f"{100 * r.test_macro_recall:.1f} $\\pm$ {100 * r.test_macro_recall_sd:.1f} {TICK if r.safe else CROSS}"
                if same(r, summary["selected"]):
                    text += r"\textsuperscript{S}"
                if same(r, summary["recommended"]):
                    text = r"\cellcolor{dmsaqua!22}\textbf{" + text + "}"
                cells.append(text)
            lines.append(f" & {METHOD[method]} & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    lines.append(rf"\multicolumn{{2}}{{@{{}}l}}{{Baseline, 640 px, FP32}} & \multicolumn{{{len(sparsities)}}}{{l}}"
                 rf"{{{100 * base.test_macro_recall:.1f} $\pm$ {100 * base.test_macro_recall_sd:.1f} at "
                 rf"{base.gflops:.2f} GFLOPs / {base.fp32_mb:.2f} MB}} \\")
    body = "\n".join(lines)
    write("table_codesign.tex", colour_defs() + rf"""
\begin{{table*}}[!t]
\renewcommand{{\arraystretch}}{{0.95}}
\caption{{Test macro-recall (\%) of INT8 PTQ models across input, resolution, and sparsity}}
\label{{tab:codesign_study}}
\centering
\footnotesize
\setlength{{\tabcolsep}}{{6pt}}
\begin{{tabular}}{{@{{}}ll{"c" * len(sparsities)}@{{}}}}
\toprule
 & & \multicolumn{{{len(sparsities)}}}{{c}}{{Sparsity}} \\
\cmidrule(l){{3-{2 + len(sparsities)}}}
 & & {" & ".join(f"{sp}\\%" for sp in sparsities)} \\
\midrule
{body}
\bottomrule
\end{{tabular}}

\smallskip
{{\scriptsize Mean $\pm$ SD over three seeds. {TICK}: holds the {100 * summary["r_floor"]:.1f}\% floor. Shaded: the cheapest configuration that holds it. S: selected on validation by (\ref{{eq:selection}}).}}
\end{{table*}}
""")


def table_uncertainty(uncertainty: pd.DataFrame, p1: pd.DataFrame, p3: pd.DataFrame, summary: dict) -> None:
    keys = ["method", "resolution", "sparsity", "quant"]
    ref = uncertainty[uncertainty.method == "reference"].iloc[0]
    base = p1[(p1.arch == "yolo11n") & (p1.sparsity == 0) & (p1.quant == "fp32")].iloc[0]
    columns = [(base, ref)]
    for choice in (summary["selected"], summary["recommended"]):
        frame = pd.DataFrame([choice])
        columns.append((p3.merge(frame, on=keys).iloc[0], uncertainty.merge(frame, on=keys).iloc[0]))
    rows = []
    for c in CLASSES:
        cells = [f"{100 * m[f'test_recall_{c}']:.1f} ({100 * u[f'{c}_low']:.0f}--{100 * u[f'{c}_high']:.0f})"
                 for m, u in columns]
        rows.append(f"{NAMES[c]} & " + " & ".join(cells) + r" \\")
    rows.append(r"\midrule")
    rows.append("Macro & " + " & ".join(f"{100 * m.test_macro_recall:.1f} ({100 * u.ci_low:.0f}--{100 * u.ci_high:.0f})"
                                        for m, u in columns) + r" \\")
    body = "\n".join(rows)
    write("table_uncertainty.tex", rf"""\begin{{table}}[t]
\caption{{Test recall (\%) with clip-bootstrap 95\% intervals}}
\label{{tab:uncertainty}}
\centering
\footnotesize
\setlength{{\tabcolsep}}{{2.5pt}}
\begin{{tabular}}{{@{{}}lccc@{{}}}}
\toprule
\textbf{{Class}} & \textbf{{Baseline}} & \textbf{{Selected (S)}} & \textbf{{Recommended (R)}} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
""")


def figure_codesign(p3: pd.DataFrame, p1: pd.DataFrame, floor: float, summary: dict) -> None:
    """Recall against compute: one line per input and resolution, from 0% to 70% sparsity."""
    int8 = p3[p3.quant != "fp32"]
    base = p1[(p1.arch == "yolo11n") & (p1.sparsity == 0) & (p1.quant == "fp32")].iloc[0]
    rec = int8.merge(pd.DataFrame([summary["recommended"]]), on=["method", "resolution", "sparsity", "quant"]).iloc[0]
    plots = []
    for method, colour, mark in (("baseline", "dmsblue", "*"), ("cropped", "dmsorange", "square*"),
                                 ("collage", "dmsaqua", "triangle*")):
        for res, style in ((320, "solid"), (160, "densely dashed")):
            f = int8[(int8.method == method) & (int8.resolution == res)].sort_values("gflops")
            if f.empty:
                continue
            plots.append(rf"\addplot[color={colour}, {style}, line width=0.8pt, mark={mark}, mark size=2.2pt"
                         + ("" if res == 320 else ", forget plot") + rf"] coordinates {{{coords(f.gflops, 100 * f.test_macro_recall)}}};")
            if res == 320:
                plots.append(rf"\addlegendentry{{{METHOD[method]}}}")
            unsafe = f[~f.safe]
            if len(unsafe):
                plots.append(rf"\addplot[only marks, color={colour}, mark={mark}, mark size=2.2pt, mark options={{fill=white}}, forget plot] coordinates {{{coords(unsafe.gflops, 100 * unsafe.test_macro_recall)}}};")
    plots.append(rf"\addplot[only marks, color=black, mark=diamond*, mark size=3pt] coordinates {{({base.gflops:.3f},{100 * base.test_macro_recall:.2f})}};")
    plots.append(r"\addlegendentry{Baseline}")
    write("fig_codesign.tex", colour_defs() + rf"""
\begin{{tikzpicture}}
\begin{{axis}}[width=\columnwidth, height=6.0cm, xmode=log, log basis x=10, grid=major,
  grid style={{line width=.2pt, draw=gray!25}}, tick label style={{font=\scriptsize}}, label style={{font=\small}},
  xlabel={{Compute (GFLOPs, log scale)}}, ylabel={{Test macro-recall (\%)}}, xmin=0.1, xmax=12, ymin=40, ymax=100,
  legend style={{font=\scriptsize, draw=none, fill=none, at={{(0.5,1.02)}}, anchor=south, legend columns=-1,
  /tikz/every even column/.append style={{column sep=6pt}}}}, clip=false]
\fill[dmsaqua!10] (axis cs:0.1,{100 * floor:.2f}) rectangle (axis cs:12,100);
\draw[dmsink, dashed, line width=0.8pt] (axis cs:0.1,{100 * floor:.2f}) -- (axis cs:12,{100 * floor:.2f});
{chr(10).join(plots)}
\draw[black, line width=1pt] (axis cs:{rec.gflops:.3f},{100 * rec.test_macro_recall:.2f}) circle (5pt);
\end{{axis}}
\end{{tikzpicture}}
""")


def coords(xs, ys) -> str:
    return " ".join(f"({x:g},{y:.2f})" for x, y in zip(xs, ys))


def colour_defs() -> str:
    return "\n".join(rf"\definecolor{{dms{k}}}{{HTML}}{{{v.upper()}}}" for k, v in COLORS.items())


def figure_efficiency(p1: pd.DataFrame) -> None:
    plots = []
    for arch, colour, mark, label in (("yolo11n", "dmsblue", "*", "YOLO11n"),
                                      ("yolo26n", "dmsorange", "triangle*", "YOLO26n")):
        f = p1[(p1.arch == arch) & (p1.quant == "fp32")].sort_values("sparsity")
        plots.append(rf"\addplot[color={colour}, mark={mark}, mark size=1.5pt, line width=0.9pt] coordinates {{{coords(f.sparsity, f.eff_flops)}}};")
        plots.append(rf"\addlegendentry{{{label}}}")
        limit = 100 * f.floor / f.gflops
        plots.append(rf"\addplot[color={colour}, dashed, line width=0.8pt, mark=none] coordinates {{{coords(f.sparsity, limit)}}};")
        plots.append(rf"\addlegendentry{{{label} limit}}")
    write("fig_efficiency.tex", colour_defs() + rf"""
\begin{{tikzpicture}}
\begin{{axis}}[width=0.95\columnwidth, height=5.0cm, grid=major, grid style={{line width=.2pt, draw=gray!30}},
  tick label style={{font=\scriptsize}}, label style={{font=\small}},
  xlabel={{Target sparsity (\%)}}, ylabel={{$\text{{Eff}}_{{\text{{FLOPs}}}}$}},
  xmin=-3, xmax=93, ymin=0, xtick={{0,10,...,90}},
  legend style={{font=\scriptsize, draw=none, fill=none, at={{(0.5,1.02)}}, anchor=south, legend columns=2, /tikz/every even column/.append style={{column sep=6pt}}}}]
{chr(10).join(plots)}
\end{{axis}}
\end{{tikzpicture}}
""")


def figure_quant(p1: pd.DataFrame) -> None:
    plots = []
    for quant, colour, mark, label in (("fp32", "dmsblue", "*", "FP32"), ("ptq", "dmsorange", "square*", "INT8 PTQ"),
                                       ("qat", "dmsaqua", "triangle*", "INT8 QAT")):
        f = p1[(p1.arch == "yolo11n") & (p1.quant == quant)].sort_values("sparsity")
        if f.empty:
            continue
        pts = " ".join(f"({r.sparsity},{100 * r.test_macro_recall:.2f}) +- (0,{100 * r.test_macro_recall_sd:.2f})"
                       for r in f.itertuples())
        plots.append(rf"\addplot[color={colour}, mark={mark}, mark size=1.5pt, line width=0.9pt, error bars/.cd, y dir=both, y explicit] coordinates {{{pts}}};")
        plots.append(rf"\addlegendentry{{{label}}}")
    write("fig_quant.tex", colour_defs() + rf"""
\begin{{tikzpicture}}
\begin{{axis}}[width=0.95\columnwidth, height=5.0cm, grid=major, grid style={{line width=.2pt, draw=gray!30}},
  tick label style={{font=\scriptsize}}, label style={{font=\small}},
  xlabel={{Target sparsity (\%)}}, ylabel={{Macro-recall (\%)}},
  xmin=-3, xmax=93, ymin=0, ymax=100, xtick={{0,10,...,90}},
  legend style={{font=\scriptsize, draw=none, fill=none, at={{(0.5,1.02)}}, anchor=south, legend columns=-1, /tikz/every even column/.append style={{column sep=6pt}}}},
  error bars/error bar style={{line width=0.5pt, solid}},
  error bars/error mark options={{line width=0.5pt, mark size=1.2pt, rotate=90}}]
{chr(10).join(plots)}
\end{{axis}}
\end{{tikzpicture}}
""")


def figure_input(p2: pd.DataFrame, floor: float) -> None:
    plots = []
    for method, colour, mark in (("baseline", "dmsblue", "*"), ("cropped", "dmsorange", "square*"),
                                 ("collage", "dmsaqua", "triangle*")):
        f = p2[p2.method == method].sort_values("gflops")
        plots.append(rf"\addplot[only marks, color={colour}, mark={mark}, mark size=2.2pt] coordinates {{{coords(f.gflops, 100 * f.test_macro_recall)}}};")
        plots.append(rf"\addlegendentry{{{METHOD[method]}}}")
    plots.append(rf"\addplot[color=dmsink, dashed, line width=0.8pt, forget plot] coordinates {{(0.1,{100 * floor:.2f}) (10,{100 * floor:.2f})}};")
    write("fig_input.tex", colour_defs() + rf"""
\begin{{tikzpicture}}
\begin{{axis}}[width=0.95\columnwidth, height=5.0cm, xmode=log, log basis x=10, grid=major,
  grid style={{line width=.2pt, draw=gray!30}}, tick label style={{font=\scriptsize}}, label style={{font=\small}},
  xlabel={{GFLOPs}}, ylabel={{Macro-recall (\%)}}, xmin=0.1, xmax=10, ymin=60, ymax=100,
  legend style={{font=\scriptsize, draw=none, fill=none, at={{(0.5,1.02)}}, anchor=south, legend columns=-1, /tikz/every even column/.append style={{column sep=6pt}}}}]
{chr(10).join(plots)}
\end{{axis}}
\end{{tikzpicture}}
""")


def main():
    enter_root()
    OUT.mkdir(parents=True, exist_ok=True)
    table_coverage()
    if not (TABLES / "summary.json").exists():
        return
    summary = json.loads((TABLES / "summary.json").read_text(encoding="utf-8"))
    p1 = pd.read_csv(TABLES / "phase1.csv")
    p2 = pd.read_csv(TABLES / "phase2.csv")
    p3 = pd.read_csv(TABLES / "phase3.csv")
    uncertainty = pd.read_csv(TABLES / "uncertainty.csv")
    table_pruning(p1)
    table_quant(p1)
    table_input(p2)
    table_codesign(p3, p1, uncertainty, summary)
    table_uncertainty(uncertainty, p1, p3, summary)
    figure_efficiency(p1)
    figure_quant(p1)
    figure_input(p2, summary["r_floor"])
    figure_codesign(p3, p1, summary["r_floor"], summary)


if __name__ == "__main__":
    main()
