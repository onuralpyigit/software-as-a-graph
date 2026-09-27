#!/usr/bin/env python3
"""
scripts/generate_merged_papers.py

Generates consolidated (flattened/merged) LaTeX and Markdown files for both:
1. The Current Version (HEAD, bac0ce60, 29 pages)
2. The Advisor Review Base Version (c2d2f792, 22 pages, reviewed by Prof. Feza Buzluca)

Outputs:
  Current (HEAD):
    - docs/research/jss/latex/manuscript_flat.tex
    - docs/research/jss/latex/supplementary_flat.tex
    - docs/research/jss/manuscript.md
    - docs/research/jss/supplementary.md

  Advisor Review Draft (v4 Base, c2d2f792):
    - docs/research/jss/drafts/manuscript_v4_advisor_flat.tex
    - docs/research/jss/drafts/manuscript_v4_advisor.md
    - docs/research/jss/drafts/supplementary_v4_advisor_flat.tex
"""

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JSS = ROOT / "docs/research/jss"
LATEX = JSS / "latex"
DRAFTS = JSS / "drafts"
BASE_COMMIT = "c2d2f7927e89e524dfbd3a1267834e4c85ad6a53"


def flatten_latex(source_path: Path, base_dir: Path, commit: str | None = None) -> str:
    """Recursively expand \\input{} directives in a LaTeX file."""
    if commit:
        cmd = ["git", "show", f"{commit}:{source_path.relative_to(ROOT)}"]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        content = res.stdout
    else:
        content = source_path.read_text(encoding="utf8")

    def repl(m: re.Match) -> str:
        rel_target = m.group(1).strip()
        target_path = base_dir / rel_target
        if not target_path.suffix:
            target_path = target_path.with_suffix(".tex")
        if commit:
            try:
                sub_cmd = ["git", "show", f"{commit}:{target_path.relative_to(ROOT)}"]
                sub_res = subprocess.run(sub_cmd, capture_output=True, text=True, check=True)
                return sub_res.stdout
            except subprocess.CalledProcessError:
                return m.group(0)
        else:
            if target_path.exists():
                return flatten_latex(target_path, base_dir)
            return m.group(0)

    # Match \input{filename} with optional whitespace
    pattern = re.compile(r"\\input\{([^}]+)\}")
    flattened = pattern.sub(repl, content)
    return flattened


def render_supplementary_markdown(flat_tex_path: Path, out_md_path: Path) -> None:
    """Render flattened supplementary LaTeX into clean GitHub-flavored Markdown."""
    tex = flat_tex_path.read_text(encoding="utf8")

    # Extract body between \begin{document} and \end{document}
    m = re.search(r"\\begin\{document\}(.*?)\\end\{document\}", tex, re.DOTALL)
    body = m.group(1) if m else tex

    # Preprocessing to help pandoc parse clean Markdown tables and sections
    body = re.sub(r"\\resizebox\{[^}]*\}\{[^}]*\}\{%?\n?", "", body)
    body = re.sub(r"\n\}%?\n*(\\end\{table\})", r"\n\1", body)
    body = re.sub(r"\n\}%?\n*(\\end\{table\*\})", r"\n\1", body)
    body = re.sub(r"\\cmidrule(\([^)]*\))?\{[^}]*\}", "", body)
    body = re.sub(r"\\shortstack\{((?:[^{}]|\{[^{}]*\})+)\}", lambda sm: " ".join(sm.group(1).split(r"\\")), body)
    body = re.sub(r"\\path\{([^}]+)\}", r"\\texttt{\1}", body)
    body = body.replace("---", "—").replace("--", "–").replace("~", " ")

    res = subprocess.run(
        ["pandoc", "-f", "latex", "-t", "gfm+tex_math_dollars", "--wrap=none"],
        input=body,
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        print(f"Warning: pandoc warning/error on supplementary: {res.stderr[:300]}")

    md = res.stdout.replace(r"\[", "[").replace(r"\]", "]").strip()
    out_md_path.write_text(md + "\n", encoding="utf8")
    print(f"✓ Rendered: {out_md_path} ({len(md)} bytes)")


def main():
    print("=" * 72)
    print("Generating Merged LaTeX and Markdown Files for JSS Paper")
    print("=" * 72)

    DRAFTS.mkdir(parents=True, exist_ok=True)

    # 1. CURRENT VERSION (HEAD)
    print("\n[1/2] Processing Current Version (HEAD)...")

    # 1a. Flatten manuscript.tex
    manuscript_flat = flatten_latex(LATEX / "manuscript.tex", LATEX)
    (LATEX / "manuscript_flat.tex").write_text(manuscript_flat, encoding="utf8")
    print(f"✓ Generated: {LATEX / 'manuscript_flat.tex'} ({len(manuscript_flat)} bytes)")

    # 1b. Flatten supplementary.tex
    supp_flat = flatten_latex(LATEX / "supplementary.tex", LATEX)
    (LATEX / "supplementary_flat.tex").write_text(supp_flat, encoding="utf8")
    print(f"✓ Generated: {LATEX / 'supplementary_flat.tex'} ({len(supp_flat)} bytes)")

    # 1c. Render manuscript.md using official renderer
    subprocess.run([sys.executable, str(ROOT / "reproduce/render_manuscript_md.py")], check=True)

    # 1d. Render supplementary.md
    render_supplementary_markdown(LATEX / "supplementary_flat.tex", JSS / "supplementary.md")

    # 2. ADVISOR REVIEW BASE VERSION (c2d2f792)
    print("\n[2/2] Processing Advisor Review Base Version (c2d2f792)...")

    # 2a. Flatten v4 manuscript.tex
    v4_manuscript_flat = flatten_latex(LATEX / "manuscript.tex", LATEX, commit=BASE_COMMIT)
    (DRAFTS / "manuscript_v4_advisor_flat.tex").write_text(v4_manuscript_flat, encoding="utf8")
    print(f"✓ Generated: {DRAFTS / 'manuscript_v4_advisor_flat.tex'} ({len(v4_manuscript_flat)} bytes)")

    # 2b. Extract v4 manuscript.md from c2d2f792
    cmd = ["git", "show", f"{BASE_COMMIT}:docs/research/jss/manuscript.md"]
    v4_md = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    (DRAFTS / "manuscript_v4_advisor.md").write_text(v4_md, encoding="utf8")
    print(f"✓ Generated: {DRAFTS / 'manuscript_v4_advisor.md'} ({len(v4_md)} bytes)")

    # 2c. Flatten v4 supplementary.tex from c2d2f792
    v4_supp_flat = flatten_latex(LATEX / "supplementary.tex", LATEX, commit=BASE_COMMIT)
    (DRAFTS / "supplementary_v4_advisor_flat.tex").write_text(v4_supp_flat, encoding="utf8")
    print(f"✓ Generated: {DRAFTS / 'supplementary_v4_advisor_flat.tex'} ({len(v4_supp_flat)} bytes)")

    print("\n" + "=" * 72)
    print("All merged LaTeX and Markdown files generated successfully!")
    print("=" * 72)


if __name__ == "__main__":
    main()
