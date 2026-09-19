#!/usr/bin/env python3
"""Regenerate docs/research/jss/manuscript.md from the authoritative LaTeX sources.

``latex/`` is the manuscript of record; ``manuscript.md`` is a faithful Markdown
rendering of ``manuscript.tex`` kept for review and diffing.

Numbering is not re-derived. Section, table, figure and citation numbers are read
out of the compiled ``manuscript.aux`` and ``manuscript.bbl``, so the Markdown
carries exactly the numbers the submitted PDF carries. **Build the manuscript
first**; a stale .aux yields stale numbering here.

Usage
-----
    cd docs/research/jss/latex && make          # refresh .aux/.bbl first
    python reproduce/render_manuscript_md.py
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JSS = ROOT / "docs/research/jss"
LATEX = JSS / "latex"
SEC = LATEX / "sections"
MANUSCRIPT_MD = JSS / "manuscript.md"

MD_SECTIONS = JSS / "sections"

SECTIONS = [
    "sec1_introduction",
    "sec2_related_work",
    "sec3_sag_model",
    "sec4_failure_impact_prediction",
    "sec5_explanation_layer",
    "sec6_experimental_setup",
    "sec7_results",
    "sec8_discussion",
    "sec9_conclusion",
]


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
    tex = re.sub(r"\\cmidrule(\([^)]*\))?\{[^}]*\}", "", tex)

    def head(m):
        cmd, title, lbl = m.group(1), m.group(2), m.group(3)
        num = labels.get(lbl, "")
        prefix = f"{num}. " if num else ""
        return f"\\{cmd}{{{prefix}{title}}}"

    tex = re.sub(r"\\(section|subsection|subsubsection)\{([^}]*)\}\s*\n\\label\{([^}]+)\}",
                 head, tex)

    def eq_label(m):
        lbl = m.group(1)
        if lbl.startswith("eq:"):
            num = labels.get(lbl, "")
            return f"\\tag{{{num}}}" if num else ""
        return m.group(0)

    tex = re.sub(r"\\label\{([^}]+)\}", eq_label, tex)
    tex = re.sub(r"\\(?:ref|eqref)\{([^}]+)\}", lambda m: labels.get(m.group(1), "??"), tex)
    tex = re.sub(r"\\cite\{([^}]+)\}",
                 lambda m: "[" + ", ".join(str(cites.get(k.strip(), "?"))
                                           for k in m.group(1).split(",")) + "]", tex)
    for a, b in (("Sections~", "§§"), ("Section~", "§"), ("Table~", "Table "),
                 ("Figure~", "Figure "), ("Equation~", "Equation "), ("Eq.~", "Eq. ")):
        tex = tex.replace(a, b)
    tex = re.sub(r"\\path\{([^}]+)\}", r"\\texttt{\1}", tex)
    return tex.replace("~", " ")


def to_markdown(tex: str, name: str = "") -> str:
    r = subprocess.run(["pandoc", "-f", "latex", "-t", "gfm+tex_math_dollars", "--wrap=none"],
                       input=tex, capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"pandoc failed on {name}:\n{r.stderr[:800]}")
    md = r.stdout.replace(r"\[", "[").replace(r"\]", "]")
    return md.strip()


def render_declarations(cites: dict) -> str:
    tex = (SEC / "declarations.tex").read_text(encoding="utf8")
    tex = re.sub(r"\\section\*\{([^}]+)\}", r"\\section{\1}", tex)
    tex = re.sub(r"\\(?:smallskip|noindent)\s*", "", tex)
    tex = re.sub(r"\\cite\{([^}]+)\}",
                 lambda m: "[" + ", ".join(str(cites.get(k.strip(), "?"))
                                           for k in m.group(1).split(",")) + "]", tex)
    tex = tex.replace("---", "—").replace("--", "–").replace("~", " ")
    md = to_markdown(tex, "declarations")
    return md.strip()


def render_references(cites: dict) -> str:
    bbl = (LATEX / "manuscript.bbl").read_text(encoding="utf8")
    body = bbl.split(r"\begin{thebibliography}", 1)[1].split(r"\end{thebibliography}")[0]
    out = []
    for i, e in enumerate(re.split(r"\\bibitem\{[^}]*\}", body)[1:], 1):
        t = e.strip()
        t = re.sub(r"\\newblock\s*", "", t)
        t = re.sub(r"\\urlprefix\s*", "URL ", t)
        t = re.sub(r"\\href\s*\{\s*([^}]*?)\s*\}\s*\{\\path\{\s*([^}]*?)\s*\}\}", r"[\2](\1)", t)
        t = re.sub(r"\\href\s*\{\s*([^}]*?)\s*\}\s*\{\s*([^}]*?)\s*\}", r"[\2](\1)", t)
        t = re.sub(r"\\url\{([^}]*)\}", r"<\1>", t)
        t = re.sub(r"\\(?:path|texttt)\{([^}]*)\}", r"`\1`", t)
        t = re.sub(r"\\emph\{([^}]*)\}", r"*\1*", t)
        t = re.sub(r"\\[a-zA-Z]+\s*", "", t)
        t = t.replace("~", " ").replace("{", "").replace("}", "")
        t = re.sub(r"\s+", " ", t).strip()
        out.append(f"[{i}] {t}")
    return "\n\n".join(out)


def render_frontmatter() -> tuple[str, str, str]:
    tex = (LATEX / "manuscript.tex").read_text(encoding="utf8")

    m_title = re.search(r"\\title\{([^}]+)\}", tex, re.S)
    title = re.sub(r"\s+", " ", m_title.group(1)).strip() if m_title else "Software-as-a-Graph"

    header = (
        f"# {title}\n\n"
        f"**Authors.** Ibrahim Onuralp Yigit, Feza Buzluca\n\n"
        f"**Affiliation.** Department of Computer Engineering, Istanbul Technical University, "
        f"34469 Istanbul, Turkey\n\n"
        f"**Corresponding author.** Ibrahim Onuralp Yigit — yigiti@itu.edu.tr\n"
    )

    abstract_tex = (SEC / "abstract.tex").read_text(encoding="utf8")
    abstract_tex = re.sub(r"\\(?:emph|textit)\{([^}]+)\}", r"*\1*", abstract_tex)
    abstract_tex = abstract_tex.replace("---", "—").replace("--", "–").replace("~", " ")
    abstract_md = to_markdown(abstract_tex, "abstract")

    m_kw = re.search(r"\\begin\{keyword\}(.*?)\\end\{keyword\}", tex, re.S)
    if m_kw:
        kw_list = [k.strip() for k in m_kw.group(1).split(r"\sep") if k.strip()]
        keywords = "; ".join(kw_list)
        keywords = re.sub(r"\s+", " ", keywords).replace("--", "–")
    else:
        keywords = ""
    keywords_md = f"**Keywords:** {keywords}."

    return header, abstract_md, keywords_md


def postprocess_markdown(body: str, labels: dict) -> str:
    body = re.sub(r"^(#{2,}) (\d+\.\d[\d.]*)\. ", r"\1 \2 ", body, flags=re.M)

    def fig(m):
        block = m.group(0)
        src = re.search(r'src="([^"]+)"', block).group(1)
        fid = re.search(r'id="([^"]+)"', block)
        num = labels.get(fid.group(1), "?") if fid else "?"
        cap = re.search(r"<figcaption[^>]*>(.*?)</figcaption>", block, re.S)
        cap = re.sub(r"<[^>]+>", "", cap.group(1)) if cap else ""
        cap = re.sub(r"\s+", " ", cap).strip()
        return f"![Figure {num}](latex/{src}.png)\n\n*Figure {num}. {cap}*"

    body = re.sub(r"<figure>.*?</figure>", fig, body, flags=re.S)
    body = body.replace("<!-- -->", "").replace("$-$", "-")

    def table_block(m):
        tid, inner = m.group(1), m.group(2).strip()
        lines = inner.split("\n")
        rows = [l for l in lines if l.lstrip().startswith("|")]
        cap = " ".join(l.strip() for l in lines if not l.lstrip().startswith("|")).strip()
        tab_num = labels.get(tid, "?")
        return f"**Table {tab_num}.** {cap}\n\n" + "\n".join(rows)

    body = re.sub(r'<div id="(tab:[^"]+)">\n(.*?)\n</div>', table_block, body, flags=re.S)
    return body.strip()


def render_sections(labels: dict, cites: dict) -> dict[str, str]:
    header, abstract_md, keywords_md = render_frontmatter()

    sections: dict[str, str] = {}
    sections["frontmatter"] = header.rstrip()
    sections["abstract"] = f"# Abstract\n\n{abstract_md}\n\n{keywords_md}"

    for n in SECTIONS:
        raw = to_markdown(preprocess((SEC / f"{n}.tex").read_text(encoding="utf8"), labels, cites), n)
        sections[n] = postprocess_markdown(raw, labels)

    sections["declarations"] = render_declarations(cites)
    sections["references"] = f"# References\n\n{render_references(cites)}"
    return sections


def load_sections_from_markdown() -> dict[str, str]:
    sections: dict[str, str] = {}
    sections["frontmatter"] = (MD_SECTIONS / "frontmatter.md").read_text(encoding="utf8").strip()
    sections["abstract"] = (MD_SECTIONS / "abstract.md").read_text(encoding="utf8").strip()
    for n in SECTIONS:
        sections[n] = (MD_SECTIONS / f"{n}.md").read_text(encoding="utf8").strip()
    sections["declarations"] = (MD_SECTIONS / "declarations.md").read_text(encoding="utf8").strip()
    sections["references"] = (MD_SECTIONS / "references.md").read_text(encoding="utf8").strip()
    return sections


def build_manuscript(sections: dict[str, str] | None = None) -> tuple[str, dict[str, str]]:
    if sections is None:
        if not (LATEX / "manuscript.aux").exists():
            sys.exit("manuscript.aux missing — run `make` in docs/research/jss/latex first")
        labels, cites = load_numbering()
        sections = render_sections(labels, cites)

    body = "\n\n".join(sections[n] for n in SECTIONS)
    fm = sections["frontmatter"]
    ab = sections["abstract"]
    dec = sections["declarations"]
    ref = sections["references"]

    rendered = (
        f"{fm}\n\n"
        f"---\n\n"
        f"{ab}\n\n"
        f"---\n\n"
        f"{body}\n\n"
        f"---\n\n"
        f"{dec}\n\n"
        f"---\n\n"
        f"{ref}\n"
    )
    return rendered, sections


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--check", action="store_true",
        help="do not write; exit 1 if regenerating would change manuscript.md or sections/*.md")
    ap.add_argument(
        "--from-sections", action="store_true",
        help="assemble manuscript.md from docs/research/jss/sections/*.md without touching LaTeX")
    args = ap.parse_args()

    if args.from_sections:
        sections = load_sections_from_markdown()
        rendered, _ = build_manuscript(sections)
        if args.check:
            if not MANUSCRIPT_MD.exists():
                print("manuscript.md does not exist.")
                return 1
            old = MANUSCRIPT_MD.read_text(encoding="utf8")
            if rendered != old:
                import difflib
                diff = list(difflib.unified_diff(
                    old.splitlines(), rendered.splitlines(),
                    fromfile="manuscript.md (on disk)", tofile="manuscript.md (from sections)", lineterm="", n=1))
                print(f"manuscript.md is STALE against sections/*.md — {len(diff)} diff line(s).")
                return 1
            print("manuscript.md is up to date with sections/*.md.")
            return 0

        MANUSCRIPT_MD.write_text(rendered, encoding="utf8")
        print(f"manuscript.md generated successfully from sections at {MANUSCRIPT_MD}")
        return 0

    rendered, sections = build_manuscript()

    if args.check:
        stale = False
        if not MANUSCRIPT_MD.exists():
            print("manuscript.md does not exist.")
            stale = True
        else:
            old = MANUSCRIPT_MD.read_text(encoding="utf8")
            if rendered != old:
                import difflib
                diff = list(difflib.unified_diff(
                    old.splitlines(), rendered.splitlines(),
                    fromfile="manuscript.md (on disk)", tofile="manuscript.md (regenerated)", lineterm="", n=1))
                print(f"manuscript.md is STALE against the LaTeX — {len(diff)} diff line(s).")
                stale = True

        for name, content in sections.items():
            sec_file = MD_SECTIONS / f"{name}.md"
            if not sec_file.exists():
                print(f"sections/{name}.md does not exist.")
                stale = True
            else:
                old_sec = sec_file.read_text(encoding="utf8")
                expected_sec = content + "\n"
                if old_sec != expected_sec:
                    import difflib
                    diff = list(difflib.unified_diff(
                        old_sec.splitlines(), expected_sec.splitlines(),
                        fromfile=f"sections/{name}.md (on disk)", tofile=f"sections/{name}.md (regenerated)", lineterm="", n=1))
                    print(f"sections/{name}.md is STALE against the LaTeX — {len(diff)} diff line(s).")
                    stale = True

        if stale:
            return 1
        print("manuscript.md and sections/*.md are up to date.")
        return 0

    MD_SECTIONS.mkdir(parents=True, exist_ok=True)
    for name, content in sections.items():
        sec_file = MD_SECTIONS / f"{name}.md"
        sec_file.write_text(content + "\n", encoding="utf8")
    print(f"Rendered {len(sections)} section files to {MD_SECTIONS}")

    MANUSCRIPT_MD.write_text(rendered, encoding="utf8")
    print(f"manuscript.md generated successfully at {MANUSCRIPT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
