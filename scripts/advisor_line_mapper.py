#!/usr/bin/env python3
"""
scripts/advisor_line_mapper.py

Utility for mapping line numbers from the 22-page advisor review manuscript
(base commit c2d2f792, tagged 'v4-advisor-review-submitted') to:
  1. The specific LaTeX source file and section in the base draft.
  2. The corresponding section and context in current HEAD (bac0ce60).
  3. Git diff of that section between c2d2f792 and HEAD.

Usage:
  python3 scripts/advisor_line_mapper.py --line 18
  python3 scripts/advisor_line_mapper.py --line 266 --diff
  python3 scripts/advisor_line_mapper.py --search "Tukey fence"
  python3 scripts/advisor_line_mapper.py --section "7.1"
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

BASE_COMMIT = "c2d2f7927e89e524dfbd3a1267834e4c85ad6a53"
LATEX_DIR = Path("docs/research/jss/latex")

# Ordered list of anchors: (start_line, end_line, section_title, v4_file, head_file_or_location)
ANCHORS = [
    (1, 10, "1.1 Motivation and Problem (Opening)", "sec1_introduction.tex", "sec1_introduction.tex"),
    (11, 24, "1.1 Sequential Cascades & Architecture-Code Gap", "sec1_introduction.tex", "sec1_introduction.tex"),
    (25, 36, "1.2 The Software-as-a-Graph (SaG) Approach", "sec1_introduction.tex", "sec1_introduction.tex"),
    (37, 49, "1.3 Research Questions (RQ1-RQ4)", "sec1_introduction.tex", "sec1_introduction.tex"),
    (50, 82, "1.4 Contributions (1-5)", "sec1_introduction.tex", "sec1_introduction.tex"),
    (83, 101, "2.1 Dependability Analysis of Distributed Systems", "sec2_related_work.tex", "sec2_related_work.tex"),
    (102, 109, "2.2 Static Code Analysis & Static System Analysis", "sec2_related_work.tex", "sec2_related_work.tex"),
    (110, 117, "2.3 Quality Models & Multi-Criteria Evaluation", "sec2_related_work.tex", "sec2_related_work.tex"),
    (118, 129, "2.4 Graph Learning & Explainability", "sec2_related_work.tex", "sec2_related_work.tex"),
    (130, 144, "3.1 Formal Multigraph Definition & Weights", "sec3_sag_model.tex", "sec3_sag_model.tex"),
    (145, 159, "3.2 QoS-Aware Weights (AHP & Log-normalization)", "sec3_sag_model.tex", "sec3_sag_model.tex"),
    (160, 172, "3.2.1 Logical Dependency Projection (Rules 1-6)", "sec3_sag_model.tex", "sec3_sag_model.tex"),
    (173, 177, "3.3 Dual Graph Views (Structural vs Analysis)", "sec3_sag_model.tex", "sec3_sag_model.tex"),
    (178, 194, "3.4 Typed Node Feature Encoding (18-D base + typed)", "sec3_sag_model.tex", "sec3_sag_model.tex"),
    (195, 202, "4. Ranking Engines Overview", "sec4_failure_impact_prediction.tex", "sec4_failure_impact_prediction.tex"),
    (203, 211, "4.1 Heterogeneous Graph Transformer (HGT-QoS)", "sec4_failure_impact_prediction.tex", "sec4_failure_impact_prediction.tex"),
    (212, 219, "4.1.1 QoS Edge Encoding (16-D)", "sec4_failure_impact_prediction.tex", "sec4_failure_impact_prediction.tex"),
    (220, 232, "4.2 Prediction Head & Composite Ranking Loss", "sec4_failure_impact_prediction.tex", "sec4_failure_impact_prediction.tex"),
    (233, 257, "4.3 Ground-Truth Simulation Oracles (I*, Icomp, Idyn)", "sec4_failure_impact_prediction.tex", "sec4_failure_impact_prediction.tex"),
    (258, 265, "4.4 Input-Label Independence Guarantee", "sec4_failure_impact_prediction.tex", "sec4_failure_impact_prediction.tex"),
    (266, 272, "5. The Explanation Layer: Criticality Attribution", "sec5_explanation_layer.tex", "supplementary.tex (moved to S.35 on HEAD)"),
    (273, 278, "5.1 Grounding in ISO/IEC Standards (FT, A, M)", "sec5_explanation_layer.tex", "supplementary.tex (moved to S.35 on HEAD)"),
    (279, 296, "5.2 Composite Quality Score (Q score & Tukey fence)", "sec5_explanation_layer.tex", "supplementary.tex (moved to S.35 on HEAD)"),
    (297, 301, "5.3 Counterfactual Verification of Remediation", "sec5_explanation_layer.tex", "supplementary.tex (moved to S.35 on HEAD)"),
    (302, 321, "6.1 Corpus (17 Architectures) & Replication Package", "sec6_experimental_setup.tex", "sec6_experimental_setup.tex"),
    (322, 339, "6.2 Predictors (Topo, Topo-QoS, GAT, HGT, Hybrids)", "sec6_experimental_setup.tex", "sec6_experimental_setup.tex"),
    (340, 364, "6.3 Metrics, Protocols & Registered Plan", "sec6_experimental_setup.tex", "sec6_experimental_setup.tex"),
    (365, 392, "7.1 RQ1 SaG Engines vs Structural Baselines (Table 7)", "sec7_results.tex", "sec7_results.tex"),
    (393, 407, "7.1 Why Hybrids Work & Label Noise", "sec7_results.tex", "sec7_results.tex"),
    (408, 414, "7.1.1 How the Hybrids Are Built", "sec7_results.tex", "sec7_results.tex"),
    (415, 440, "7.2 RQ2 What Learned Engines Need (Table 8 2x2)", "sec7_results.tex", "sec7_results.tex"),
    (441, 467, "7.3 RQ3 Zero-Shot Transfer to Open-Source Systems (Table 9)", "sec7_results.tex", "sec7_results.tex"),
    (468, 488, "7.4 RQ4 Analysis Cost & Dominant Stage (Table 10)", "sec7_results.tex", "sec7_results.tex"),
    (489, 505, "8.1 Practical Consequences & Choosing an Engine (Table 11)", "sec8_discussion.tex", "sec8_discussion.tex"),
    (506, 515, "8.1 Where Learned Engines Help & Sustainability", "sec8_discussion.tex", "sec8_discussion.tex"),
    (516, 540, "8.2 Threats to Validity (Construct, Internal, External)", "sec8_discussion.tex", "sec8_discussion.tex"),
    (541, 549, "8.3 Limitations & Future Work", "sec8_discussion.tex", "sec8_discussion.tex"),
    (550, 573, "9. Conclusion", "sec9_conclusion.tex", "sec9_conclusion.tex"),
    (574, 598, "Declarations (CRediT, Data Availability, AI Statement)", "declarations.tex", "declarations.tex"),
    (599, 780, "References [1]-[90]", "refs.bib", "refs.bib"),
]

def get_git_file_content(commit: str, filepath: str) -> str:
    try:
        cmd = ["git", "show", f"{commit}:{filepath}"]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return res.stdout
    except subprocess.CalledProcessError:
        return ""

def show_line_info(line_num: int, show_diff: bool = False):
    found = None
    for start, end, title, v4_file, head_loc in ANCHORS:
        if start <= line_num <= end:
            found = (start, end, title, v4_file, head_loc)
            break

    if not found:
        print(f"Error: Line {line_num} out of bounds (valid range 1-780).")
        return

    start, end, title, v4_file, head_loc = found
    rel_pos = (line_num - start) / max(1, (end - start))

    print("=" * 72)
    print(f"  Advisor Review Reference: Line {line_num} (Range: {start}–{end})")
    print(f"  Section: {title}")
    print(f"  v4 Base Source File (c2d2f792): docs/research/jss/latex/sections/{v4_file}")
    print(f"  Current HEAD Location (bac0ce60): {head_loc}")
    print("=" * 72)

    # Show file diff between c2d2f792 and HEAD if requested
    if show_diff:
        target_path = LATEX_DIR / "sections" / v4_file
        if not target_path.exists():
            target_path = LATEX_DIR / v4_file
        print(f"\n--- Git Diff between c2d2f792 and HEAD for {v4_file} ---")
        cmd = ["git", "diff", BASE_COMMIT, "HEAD", "--", str(target_path)]
        diff_res = subprocess.run(cmd, capture_output=True, text=True)
        if diff_res.stdout:
            lines = diff_res.stdout.splitlines()
            for l in lines[:40]:  # Print first 40 lines
                print(l)
            if len(lines) > 40:
                print(f"... ({len(lines) - 40} more diff lines omitted)")
        else:
            print("No differences found in this file between c2d2f792 and HEAD.")

def search_text(query: str):
    print(f"Searching for '{query}' across v4 base and current HEAD...")
    print("\n[Matches in v4 Base (c2d2f792)]:")
    cmd = ["git", "grep", "-n", query, BASE_COMMIT, "--", "docs/research/jss/"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.stdout:
        for line in res.stdout.splitlines()[:10]:
            print(" ", line)
    else:
        print("  None found.")

    print("\n[Matches in current HEAD]:")
    cmd = ["git", "grep", "-n", query, "HEAD", "--", "docs/research/jss/"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.stdout:
        for line in res.stdout.splitlines()[:10]:
            print(" ", line)
    else:
        print("  None found.")

def main():
    parser = argparse.ArgumentParser(description="Map advisor review lines to LaTeX sources and HEAD diffs.")
    parser.add_argument("--line", type=int, help="Line number from 22-page manuscript (1-780)")
    parser.add_argument("--diff", action="store_true", help="Display git diff against HEAD for the mapped section")
    parser.add_argument("--search", type=str, help="Search text across v4 and HEAD")
    args = parser.parse_args()

    if args.line:
        show_line_info(args.line, show_diff=args.diff)
    elif args.search:
        search_text(args.search)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
