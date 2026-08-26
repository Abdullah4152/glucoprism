"""Structural check on the paper, standing in for a compiler we do not have.

No TeX engine is installed on this machine, so the failure modes that would
otherwise surface at build time are checked directly:

  1. every \\macro the paper uses is actually defined (canonical.tex, the
     preamble, or a known LaTeX/package builtin);
  2. \\begin/\\end environments balance and nest;
  3. every \\ref has a \\label and every \\input file exists;
  4. \\includegraphics targets resolve on disk;
  5. the third model is not mentioned -- it is an ablation elsewhere, and must
     not appear in the paper or its generated tables;
  6. no bare digit-groups sit where a canonical macro exists, which is how the
     25 cross-document contradictions got in.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

PAPER = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    Path(r"D:\overleaf\glucoprismm\glucoprism_v2\main.tex")
TEX = PAPER.parent

# Control words that come from LaTeX itself or the packages the preamble loads.
BUILTIN = set("""
documentclass usepackage newcommand renewcommand input include begin end item
section subsection paragraph appendix title author date maketitle label ref
caption centering small footnotesize large textbf textit emph texttt underline
toprule midrule bottomrule cmidrule addlinespace multicolumn
includegraphics linewidth textwidth vspace hspace linespread setlength parskip
itemsep quad qquad noindent bfseries mathrm mathbf frac sum prod int left right
times approx pm leq geq neq cdot ldots dots sigma lambda beta delta Delta
mathbb mathcal max min log exp sqrt hline par newpage clearpage
captionsetup num si SI hypersetup nobreakdash footnote cite
top rule text sim to rightarrow leftarrow in subset cup cap forall exists
alpha gamma theta mu nu phi psi omega Omega Phi Psi Sigma Lambda Gamma
S P ell mid parallel dagger ast star circ prime infty
Big big Bigg bigg left right hat bar tilde vec dot ddot
langle rangle lVert rVert lvert rvert tfrac dfrac gtrsim lesssim
tau epsilon varepsilon equiv odot oplus otimes propto
citep citet cite bibliography bibliographystyle
setcounter topfraction bottomfraction textfraction floatfraction
floatpagefraction thesection thesubsection titleformat titlespacing
normalsize scriptsize tiny huge Huge LARGE Large
subsubsection subsection section paragraph subparagraph
resizebox textwidth linewidth iclrfinalcopy
cellcolor columncolor definecolor rowcolor arrayrulecolor
dim textheight bfseries underline
""".split())

# Literals that legitimately coincide with a canonical value but mean something
# else. Each needs a reason, so the exception can be audited rather than grown.
ALLOWED_LITERALS = {
    "1.08": "seed sd in the 5x-scale ablation; collides with \\deltasubject, "
            "which is the 6-seed subject-level delta -- unrelated quantity",
}

problems: list[str] = []
notes: list[str] = []

src = PAPER.read_text(encoding="utf-8")
canon = (TEX / "canonical.tex").read_text(encoding="utf-8")

# ---------------------------------------------------------------- 1. macros
defined = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", canon))
defined |= set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", src))
body = re.sub(r"(?m)^\s*%.*$", "", src)          # strip comment lines
# `\\` is a line break, not the start of a control word: without removing it
# first, "\\Paper under review" reports an undefined macro \Paper.
used = set(re.findall(r"\\([A-Za-z]+)", body.replace("\\\\", " ")))
unknown = sorted(used - defined - BUILTIN)
if unknown:
    problems.append("undefined control words: " + ", ".join(unknown))

unused = sorted(set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", canon)) - used)
if unused:
    notes.append(f"{len(unused)} canonical macros defined but unused "
                 f"(fine -- they are the shared pool): {', '.join(unused[:8])}...")

# ------------------------------------------------------------ 2. environments
stack: list[tuple[str, int]] = []
for n, line in enumerate(body.splitlines(), 1):
    for kind, name in re.findall(r"\\(begin|end)\{([^}]+)\}", line):
        if kind == "begin":
            stack.append((name, n))
        elif not stack:
            problems.append(rf"line {n}: \end{{{name}}} with nothing open")
        elif stack[-1][0] != name:
            problems.append(f"line {n}: \\end{{{name}}} closes "
                            f"\\begin{{{stack[-1][0]}}} from line {stack[-1][1]}")
            stack.pop()
        else:
            stack.pop()
for name, n in stack:
    problems.append(rf"line {n}: \begin{{{name}}} never closed")

# ------------------------------------------------------- 3. refs, labels, input
labels = set(re.findall(r"\\label\{([^}]+)\}", body))
for f in re.findall(r"\\input\{([^}]+)\}", body):
    p = (TEX / f) if f.endswith(".tex") else (TEX / (f + ".tex"))
    if not p.exists():
        problems.append(rf"\input{{{f}}} -> missing {p}")
    else:
        labels |= set(re.findall(r"\\label\{([^}]+)\}",
                                 p.read_text(encoding="utf-8")))
for r in sorted(set(re.findall(r"\\ref\{([^}]+)\}", body))):
    if r not in labels:
        problems.append(rf"\ref{{{r}}} has no \label")

# ------------------------------------------------------------- 4. graphics
for g in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", body):
    if not (TEX / g).resolve().exists():
        problems.append(rf"\includegraphics{{{g}}} -> missing "
                        f"{(TEX / g).resolve()}")

# --------------------------------------------------- 5. the third model is out
BANNED = ["GP-orig", "GlucoPRISM proposal", "glucoprism-original",
          "GlucoPRISM-original"]
for f in sorted(TEX.glob("*.tex")):
    if f.name in {"final_final.tex", "findings.tex"}:
        continue                      # the companion document may discuss it
    t = f.read_text(encoding="utf-8")
    for b in BANNED:
        if b in t:
            problems.append(f"{f.name}: mentions the third model ({b!r})")

# ------------------------------------------------ 6. hand-typed canonical values
vals = {m: v for m, v in re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{([^}]+)\}",
                                    canon)}
prose = re.sub(r"\\(?:label|ref|includegraphics|input)\{[^}]*\}", "", body)
for macro, val in vals.items():
    if not re.fullmatch(r"-?\d+\.\d+", val) or abs(float(val)) < 1.0:
        continue                      # ints and small deltas collide too easily
    if val in ALLOWED_LITERALS:
        continue
    if re.search(rf"(?<![\d.]){re.escape(val)}(?![\d.])", prose):
        problems.append(f"literal {val} typed in prose where \\{macro} exists")

# ------------------------------------------------------------------- report
for n in notes:
    print(f"  note: {n}")
if problems:
    print(f"\n{len(problems)} PROBLEM(S):")
    for p in problems:
        print(f"  - {p}")
    sys.exit(1)
print("\npaper.tex: structurally clean")
