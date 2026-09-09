#!/usr/bin/env python3
"""Regenerate docs/research/jss/draft.md from the authoritative LaTeX sources.

``latex/`` is the manuscript of record; ``draft.md`` is a Markdown rendering of
it kept for review and diffing. Historically the two drifted, because revisions
landed in LaTeX first and the Markdown was re-synced by hand or not at all --- at
one point draft.md was two revision rounds behind and contradicted the paper it
was supposed to mirror. This script removes the hand step.

Numbering is not re-derived. Section, table, figure and citation numbers are read
out of the compiled ``manuscript.aux`` and ``manuscript.bbl``, so the Markdown
carries exactly the numbers the submitted PDF carries. **Build the manuscript
first**; a stale .aux yields stale numbering here.

What is preserved that LaTeX does not have: the two ASCII schematics (the
pipeline overview and the HGT layer stack) and the header note recording the
review model. Both are lifted from the existing draft.md, so they survive
regeneration.

Usage
-----
    cd docs/research/jss/latex && make          # refresh .aux/.bbl first
    python reproduce/render_draft_md.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JSS = ROOT / "docs/research/jss"
LATEX = JSS / "latex"
SEC = LATEX / "sections"
DRAFT = JSS / "draft.md"

SECTIONS = ["sec1_introduction", "sec2_related_work", "sec3_sag_model",
            "sec4_failure_impact_prediction", "sec5_explanation_layer",
            "sec6_experimental_setup", "sec7_results", "sec8_discussion"]


def load_numbering() -> tuple[dict, dict]:
    aux = (LATEX / "manuscript.aux").read_text(encoding="utf8")
    labels: dict[str, str] = {}
    for m in re.finditer(r"\\newlabel\{([^}]+)\}\{\{([^}]*)\}", aux):
        labels.setdefault(m.group(1), re.sub(r"\\[a-zA-Z]+\s*", "", m.group(2)).strip())
    bbl = (LATEX / "manuscript.bbl").read_text(encoding="utf8")
    cites = {k: i + 1 for i, k in enumerate(re.findall(r"\\bibitem\{([^}]+)\}", bbl))}
    return labels, cites


def preprocess(tex: str, labels: dict, cites: dict) -> str:
    # resizebox wrappers hide whole tables from pandoc; unwrap them.
    tex = re.sub(r"\\resizebox\{[^}]*\}\{[^}]*\}\{%?\n", "", tex)
    tex = re.sub(r"\n\}%?\n(\\end\{table\})", r"\n\1", tex)

    def head(m):
        cmd, title, lbl = m.group(1), m.group(2), m.group(3)
        num = labels.get(lbl, "")
        return f"\\{cmd}{{{num + '. ' if num else ''}{title}}}"

    tex = re.sub(r"\\(section|subsection|subsubsection)\{([^}]*)\}\s*\n\\label\{([^}]+)\}",
                 head, tex)
    tex = re.sub(r"\\(?:ref|eqref)\{([^}]+)\}", lambda m: labels.get(m.group(1), "??"), tex)
    tex = re.sub(r"\\cite\{([^}]+)\}",
                 lambda m: "[" + ", ".join(str(cites.get(k.strip(), "?"))
                                           for k in m.group(1).split(",")) + "]", tex)
    for a, b in (("Sections~", "§§"), ("Section~", "§"), ("Table~", "Table "),
                 ("Figure~", "Figure "), ("Equation~", "Equation "), ("Eq.~", "Eq. ")):
        tex = tex.replace(a, b)
    return tex.replace("~", " ")


def to_markdown(tex: str, name: str) -> str:
    r = subprocess.run(["pandoc", "-f", "latex", "-t", "gfm+tex_math_dollars", "--wrap=none"],
                       input=tex, capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"pandoc failed on {name}:\n{r.stderr[:800]}")
    md = r.stdout.replace(r"\[", "[").replace(r"\]", "]")
    return md.strip()


def render_references(cites: dict) -> str:
    bbl = (LATEX / "manuscript.bbl").read_text(encoding="utf8")
    body = bbl.split(r"\begin{thebibliography}", 1)[1].split(r"\end{thebibliography}")[0]
    out = []
    for i, e in enumerate(re.split(r"\\bibitem\{[^}]*\}", body)[1:], 1):
        t = e.strip()
        t = re.sub(r"\\newblock\s*", "", t)
        t = re.sub(r"\\urlprefix\s*", "URL ", t)
        t = re.sub(r"\\href\{([^}]*)\}\{([^}]*)\}", r"[\2](\1)", t)
        t = re.sub(r"\\url\{([^}]*)\}", r"<\1>", t)
        t = re.sub(r"\\(?:path|texttt)\{([^}]*)\}", r"`\1`", t)
        t = re.sub(r"\\emph\{([^}]*)\}", r"*\1*", t)
        t = re.sub(r"\\[a-zA-Z]+\s*", "", t)
        t = t.replace("~", " ").replace("{", "").replace("}", "")
        t = re.sub(r"\\s+", " ", t).strip()
        out.append(f"[{i}] {t}")
    return "\n\n".join(out)


def fenced_block(text: str, needle: str) -> str:
    i = text.index(needle)
    start = text.rindex("```", 0, i)
    end = text.index("```", i) + 3          # the needle sits inside the block
    return text[start:end]


def main() -> int:
    if not (LATEX / "manuscript.aux").exists():
        sys.exit("manuscript.aux missing — run `make` in docs/research/jss/latex first")
    labels, cites = load_numbering()
    old = DRAFT.read_text(encoding="utf8")

    body = "\n\n".join(
        to_markdown(preprocess((SEC / f"{n}.tex").read_text(encoding="utf8"), labels, cites), n)
        for n in SECTIONS)

    body = re.sub(r"^(#{2,}) (\d+\.\d[\d.]*)\. ", r"\1 \2 ", body, flags=re.M)

    def fig(m):
        block = m.group(0)
        src = re.search(r'src="([^"]+)"', block).group(1)
        fid = re.search(r'id="([^"]+)"', block)
        num = labels.get(fid.group(1), "?") if fid else "?"
        cap = re.search(r"<figcaption[^>]*>(.*?)</figcaption>", block, re.S)
        cap = re.sub(r"<[^>]+>", "", cap.group(1)) if cap else ""
        cap = re.sub(r"\\s+", " ", cap).strip()
        return f"![Figure {num}](latex/{src}.png)\n\n*Figure {num}. {cap}*"

    body = re.sub(r"<figure>.*?</figure>", fig, body, flags=re.S)
    body = body.replace("<!-- -->", "").replace("$-$", "-")

    def table_block(m):
        tid, inner = m.group(1), m.group(2).strip()
        lines = inner.split("\n")
        rows = [l for l in lines if l.lstrip().startswith("|")]
        cap = " ".join(l.strip() for l in lines if not l.lstrip().startswith("|")).strip()
        return f"**Table {labels.get(tid, '?')}.** {cap}\n\n" + "\n".join(rows)

    body = re.sub(r'<div id="(tab:[^"]+)">\n(.*?)\n</div>', table_block, body, flags=re.S)

    # Markdown-only extras, carried over from the previous draft.
    pipeline = fenced_block(old, "Software-as-a-Graph (SaG)                              |")
    hgtstack = fenced_block(old, "Heterogeneous Graph Transformer (HGT) Architecture")
    anchor = "![Figure 1](latex/figures/Figure_1.png)"
    body = body.replace(anchor, pipeline + "\n\n" + anchor, 1)

    m = re.search(r"^## 4\.1 .*$", body, flags=re.M)
    nxt = body.index("\n\n", body.index("\n\n", m.end()) + 2)
    body = (body[:nxt] + "\n\n" + hgtstack +
            "\n\n*Figure M1 (this document only). Layered architecture of the Heterogeneous "
            "Graph Transformer predictor. The LaTeX sources carry no counterpart; the equations "
            "it summarises are those of §4.1.2 and §4.2.*" + body[nxt:])

    note = ("> **Figure numbering.** Figures 1–4 are numbered as in the LaTeX submission sources: "
            "Figure 1 pipeline (`Figure_1`), Figure 2 running example (`Figure_2`), Figure 3 "
            "results at a glance (`Figure_5`), Figure 4 HGT attention (`Figure_3`). The ASCII "
            "schematics and Figure M1 are specific to this document. Supplementary Sections S1–S7 "
            "live in `latex/supplementary.tex` and are not reproduced here.")
    body = body.replace(anchor, anchor + "\n\n" + note, 1)

    header = old[:old.index("# Abstract")]
    abstract = (SEC / "abstract.tex").read_text(encoding="utf8").strip()
    keywords = re.search(r"\\begin\{keyword\}(.*?)\\end\{keyword\}",
                         (LATEX / "manuscript.tex").read_text(encoding="utf8"), re.S).group(1)
    keywords = "; ".join(k.strip() for k in keywords.split(r"\sep") if k.strip())
    keywords = re.sub(r"\s+", " ", keywords).replace("--", "\u2013")
    keywords = "**Keywords:** " + keywords + "."
    decl = old[old.index("# Declarations"):].strip()

    DRAFT.write_text(header + "# Abstract\n\n" + abstract + "\n\n" + keywords +
                     "\n\n---\n\n" + body.strip() + "\n\n---\n\n# References\n\n" +
                     render_references(cites) + "\n\n---\n\n" + decl + "\n", encoding="utf8")
    print(f"draft.md regenerated: {len(labels)} labels, {len(cites)} citations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
