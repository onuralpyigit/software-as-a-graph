# Software-as-a-Graph (SaG) — SoSyM Tool Paper Package

This directory contains the manuscript, documentation, and LaTeX sources for submitting **Software-as-a-Graph (SaG)** to **Software and Systems Modeling (SoSyM)** (Springer) under the **Tool Paper Call**.

---

## Target Call Summary

* **Venue:** *Software and Systems Modeling* (SoSyM) — Springer
* **Track / Collection:** **Tool Paper**
* **Manuscript Title:** *Software-as-a-Graph: A Model-Driven Framework for Graph-Based Architectural Modeling, Diagnostic Analysis, and Counterfactual Refactoring of Publish–Subscribe Systems*
* **Submission Status:** Ready for submission.

---

## Mapping to SoSyM Tool Paper Criteria

| SoSyM Tool Paper Criterion | Where & How Addressed in This Manuscript |
|:---|:---|
| **High Technical Quality** | Formal typed multigraph metamodel ($G = (V, E, \tau_V, \tau_E, w_V, w_E)$), deterministic ISO 25010/25019 quality models, 19 formal anti-pattern specifications, closed-loop simulation verification engine with seed-noise rejection margins. |
| **Effective Presentation** | Structured 10-section narrative spanning domain motivation, metamodel formalisms, hexagonal architecture, multi-modal usability, realistic scenario evaluations, community lessons, and open science. Rich TikZ diagrams and comprehensive tables. |
| **Importance for Community** | **Section 8.1** details 4 major contributions: bridging MDE's asynchronous pub-sub blind spot, providing an open multi-domain benchmark testbed, publishing a reusable anti-pattern catalog, and operationalizing ISO standards in graph models. |
| **Lessons Learnt** | **Section 8.2** details 5 transferable lessons across: (1) Designing (metamodel substrate adequacy); (2) Building (in-memory sandboxing for model transformations); (3) Testing (the verify-before-transform discipline); (4) Evaluating (attribution vs. centrality saturation); (5) Disseminating (multi-modal ergonomics). |
| **Advance over State of the Art** | **Section 2.5 (Table 1)** provides a detailed feature-by-feature comparison against Palladio/PerOpteryx, AADL, Arcan/Designite, and SonarQube, highlighting SaG's unique pub-sub semantics, closed-loop verification, and dual-repository architecture. |
| **Tool Availability** | **Section 9.2** details open-source availability: Apache 2.0 license, public GitHub repository, permanent Zenodo archive with DOI, pip package, Docker Compose deployment, and reproduction scripts. |
| **Applied in Relevant Scenarios** | **Section 7** evaluates SaG across 8 synthetic benchmark scenarios (29–520 vertices across 6 domains) and 3 transcribed real-world open-source systems (Autoware.Universe / ROS 2, Apollo Cyber RT / DDS, and Cloud IoT Mesh / MQTT). |

---

## Directory Structure

```
docs/research/sosym/
├── README.md               # This document: venue mapping, criteria checklist, build instructions
├── outline.md              # Section-by-section outline and reading map
├── draft.md                # Consolidated Markdown manuscript draft
└── latex/                  # Authoritative LaTeX submission sources
    ├── Makefile            # Build automation for pdflatex and bibtex
    ├── manuscript.tex      # Master LaTeX document
    ├── manuscript.pdf      # Compiled 31-page submission PDF
    ├── refs.bib            # BibTeX bibliography with complete citations
    └── sections/           # Modular section files
        ├── abstract.tex
        ├── sec1_introduction.tex
        ├── sec2_related_work.tex
        ├── sec3_graph_metamodel.tex
        ├── sec4_antipattern_catalog.tex
        ├── sec5_tool_architecture.tex
        ├── sec6_tool_usability.tex
        ├── sec7_empirical_application.tex
        ├── sec8_community_and_lessons_learnt.tex
        ├── sec9_discussion_availability.tex
        ├── sec10_conclusion.tex
        └── declarations.tex
```

---

## Building the Manuscript PDF

To build the submission PDF locally:

```bash
cd docs/research/sosym/latex
make clean
make
```

This compiles `manuscript.pdf` with zero LaTeX warnings, zero missing citations, and zero undefined cross-references.

### Switching to the Official Springer Template

Prior to final upload on Springer's Editorial Manager:
1. Download the Springer Nature LaTeX template bundle (`sn-jnl.cls` or `svjour3.cls`).
2. Replace `\documentclass[11pt,a4paper]{article}` in `manuscript.tex` with `\documentclass[sn-basic]{sn-jnl}` (or `\documentclass[smallextended]{svjour3}`).
3. Replace `\bibliographystyle{plainnat}` with `\bibliographystyle{sn-basic}` (or `spbasic.bst`).
