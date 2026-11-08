"""Build the clean and the highlighted manuscript into paper/build.

  python scripts/build_paper.py

Clean: paper/main.tex with every generated table and figure inlined (one file for submission).
Highlighted: the same text with every change from paper/original/main_submitted.tex in blue.
Deleted text is not shown. Needs pdflatex, latexpand and perl (MiKTeX ships all three).
"""
import re
import shutil
import subprocess
from pathlib import Path

PAPER = Path(__file__).resolve().parent.parent / "paper"
BUILD = PAPER / "build"
LATEXDIFF = Path(shutil.which("latexdiff") or "").parents[3] / "scripts" / "latexdiff" / "latexdiff-so"
FLOAT = re.compile(r"\\begin\{(table|figure)(\*?)\}.*?\\end\{\1\2\}", re.DOTALL)

# Added text in blue; deleted text hidden, as the revised version is what gets reviewed.
HIGHLIGHT = "\n".join([
    r"\renewcommand{\DIFadd}[1]{{\protect\color{blue}#1}}",
    r"\renewcommand{\DIFdel}[1]{}",
    "",
])


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def strip_deleted(text: str) -> str:
    """Remove \\DIFdel{...} groups (balanced braces), DIF markers and DIF comments."""
    out, i = [], 0
    while i < len(text):
        if text.startswith(r"\DIFdel{", i):
            depth, i = 1, i + len(r"\DIFdel{")
            while i < len(text) and depth:
                depth += {"{": 1, "}": -1}.get(text[i], 0)
                i += 1
            continue
        out.append(text[i])
        i += 1
    rest = re.sub(r"%[^\n]*", "", "".join(out))
    return re.sub(r"\\DIF(add|del)(begin|end)(FL)?|\\DIFaddbegin|[{}\s]", "", rest)


def drop_empty_items(diff: str) -> str:
    """A list item that latexdiff deleted entirely still leaves its \\item, an empty bullet."""
    pieces = re.split(r"(?=\\item\b)", diff)
    kept = [pieces[0]]
    for piece in pieces[1:]:
        body = re.split(r"\\end\{itemize\}", piece[len(r"\item"):], maxsplit=1)[0]
        kept.append(piece.replace(r"\item", "", 1) if strip_deleted(body) == "" else piece)
    return "".join(kept)


def compile_pdf(name: str) -> None:
    for _ in range(2):  # the second pass resolves references
        subprocess.run(["pdflatex", "-interaction=nonstopmode", f"{name}.tex"], cwd=BUILD,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not (BUILD / f"{name}.pdf").exists():
        raise RuntimeError(f"{name}.pdf was not produced, see {BUILD / (name + '.log')}")


def main():
    BUILD.mkdir(exist_ok=True)
    shutil.copy2(PAPER / "IEEEtran.cls", BUILD / "IEEEtran.cls")
    shutil.copytree(PAPER / "figures", BUILD / "figures", dirs_exist_ok=True)

    flat = subprocess.run(["latexpand", "main.tex"], cwd=PAPER, check=True, capture_output=True,
                          text=True, encoding="utf-8").stdout
    (BUILD / "main_clean.tex").write_text(flat, encoding="utf-8")
    compile_pdf("main_clean")

    # latexdiff cannot mark changes inside tables and plots, so each float is compared as one
    # unit: an unchanged float is kept as is, a new or changed one is set entirely in blue.
    original = (PAPER / "original" / "main_submitted.tex").read_text(encoding="utf-8")
    old_floats = {normalise(m.group(0)) for m in FLOAT.finditer(original)}
    floats = []

    def hold(match):
        floats.append(match.group(0))
        return f"\n\\DIFFLOAT{chr(ord('A') + len(floats) - 1)}\n"

    (BUILD / "old_text.tex").write_text(FLOAT.sub("", original), encoding="utf-8")
    (BUILD / "new_text.tex").write_text(FLOAT.sub(hold, flat), encoding="utf-8")
    diff = subprocess.run(
        ["perl", str(LATEXDIFF), "--type=CFONT", "--subtype=SAFE", "--math-markup=whole",
         "old_text.tex", "new_text.tex"],
        cwd=BUILD, check=True, capture_output=True, text=True, encoding="utf-8").stdout
    for i, body in enumerate(floats):
        if normalise(body) not in old_floats:
            body = re.sub(r"(\\begin\{(?:table|figure)\*?\}(?:\[[^\]]*\])?)", r"\1\\color{blue}", body, count=1)
        diff = diff.replace(f"\\DIFFLOAT{chr(ord('A') + i)}", body, 1)
    diff = drop_empty_items(diff)
    # Hidden deletions must not change spacing: a control word eats the space after it, and a
    # blank line inside deleted text would still end the paragraph.
    diff = re.sub(r"(\\DIF(?:add|del)(?:begin|end)(?:FL)?) +(?=[A-Za-z0-9($])", r"\1{} ", diff)
    diff = re.sub(r"\\DIFdelbegin.*?\\DIFdelend",
                  lambda m: re.sub(r"\n[ \t]*\n", "\n", m.group(0)), diff, flags=re.DOTALL)
    diff = diff.replace(r"\begin{document}", HIGHLIGHT + r"\begin{document}", 1)
    (BUILD / "main_highlighted.tex").write_text(diff, encoding="utf-8")
    for leftover in ("old_text.tex", "new_text.tex"):
        (BUILD / leftover).unlink(missing_ok=True)
    compile_pdf("main_highlighted")
    print(f"built {BUILD / 'main_clean.pdf'} and {BUILD / 'main_highlighted.pdf'}")


if __name__ == "__main__":
    main()
