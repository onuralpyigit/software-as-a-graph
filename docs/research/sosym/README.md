# Software-as-a-Graph (SaG) — SoSyM Tool Paper Package

Manuscript, figures and reproduction scripts for submitting **Software-as-a-Graph** to *Software and Systems Modeling* (Springer), collection **Tool Paper**.

* **Title:** *Software-as-a-Graph: An Open-Source Tool for Multi-Layer Graph Modeling, Quality Diagnosis, and Simulation-Verified Refactoring of Publish–Subscribe Architectures*
* **Target Venue:** *Software and Systems Modeling* (SoSyM, Springer) — **Tool Paper** collection
* **License:** GNU GPL v3.0
* **Status:** Revised for the tool-paper call (see [REVISION_NOTES.md](REVISION_NOTES.md)). For a section-by-section map, see [outline.md](outline.md); for a readable Markdown rendering of the manuscript, see [draft.md](draft.md).

---

## Executive Summary

Publish–subscribe middleware such as **ROS 2/DDS, Apache Kafka, and MQTT** decouples components through topics, brokers, and Quality-of-Service (QoS) contracts. While this decoupling makes distributed systems easy to extend, it hides the dependency chains along which failures propagate—and mainstream architecture modeling and smell-detection tools represent topics, brokers, and QoS only indirectly or not at all.

This tool paper presents **Software-as-a-Graph (SaG)**, an open-source modeling and analysis tool purpose-built for publish–subscribe architectures. From a declarative JSON description of a system, SaG builds a typed, attributed multigraph, derives implied component dependencies with seven explicit rules across four layer projections, and runs an end-to-end analysis pipeline:

1. **Explainable quality scoring:** A Reliability–Maintainability (RM) model grounded in ISO/IEC 25010 that decomposes risk into named structural terms rather than opaque scores.
2. **Pub–sub anti-pattern detection:** Nineteen declarative patterns (including pub–sub-specific smells such as `QOS_MISMATCH`, `CHATTY_PAIR`, and `BROKER_OVERLOAD`).
3. **Delta-aware CI quality gating:** Merge-base diffing (`--baseline`) with a fence margin and an expiring waiver register so teams block only newly introduced structural risks rather than existing debt.
4. **Failure-cascade simulation:** A structural-only simulation oracle (`FailureSimulator`) kept strictly independent of analytical scores by automated tests.
5. **Simulation-verified refactoring (`Prescribe`):** Three graph-mutation operators (*topic split*, *anti-affinity reallocation*, *QoS hardening*) where each candidate edit is tested on an isolated in-memory model copy and admitted only if its simulated benefit exceeds seed-to-seed noise across every propagation threshold.

SaG is engineered around a hexagonal Python core (~65k LOC) with a Neo4j adapter for persistent exploration, an in-memory adapter for thousands of fast counterfactual simulations, ~1,600 tests enforcing architectural invariants, and five interfaces (CLI, Python API, FastAPI REST service, Next.js/React web dashboard, and self-contained HTML reports).

### Who This Paper Is For

* **Software & systems architects** reviewing pub–sub designs before deployment for cross-layer bottlenecks and structural risks.
* **Reliability engineers** estimating component and host blast radii without running fault injection on production systems.
* **DevOps teams** integrating pre-merge architecture checks into CI without adopting a heavy modeling IDE.
* **Modeling researchers** seeking an open metamodel, reproducible scenario corpus, simulation oracle, and baselines for evaluating new metrics, detectors, or model transformations.

### Key Highlights & Headline Results

* **Fast enough for CI:** Diagnosis takes **0.05–0.11 s** on five open-source architecture models (60–90 components) and **0.04–5.53 s** on synthetic models up to 326 components (**78.93 s** at 520 components; detectors take $\le 0.17\text{ s}$).
* **Focused, actionable findings:** On five open-source architecture models (*Autoware.Universe*, *Train-Ticket*, *Cloud e-commerce mesh*, *EdgeX Foundry*, *Home Assistant*), **14–22%** of components are directly flagged, with stronger oracle agreement than on denser synthetic models ($\kappa = +0.26$ vs. $-0.03$).
* **Explainable scoring vs. centralities:** On the `app` layer, the RM composite $Q(v)$ achieves mean Spearman $\rho = 0.62$ against simulated cascade impact, outperforming degree centrality ($\rho = 0.49$) in **8 of 8** scenarios (Wilcoxon $p = 0.008$), while remaining weak on hosts/brokers at the `system` layer.
* **Why refactoring rules must be verified:** Across 1,510 rule-generated edits, **32% fail counterfactual verification** (1,028 admitted, 68.1%). Even when every edit is admitted individually, combining them improves simulated impact in **4 of 7** scenarios and regresses in **3 of 7**.
* **Practical CI gating:** Across 90 random single-subscription edits, an absolute gate blocks **100%** of changes on existing debt, whereas the delta-aware gate blocks **26%**, warns on **47%**, and passes **28%**.
* **Eight transferable lessons learnt:** Grounded in concrete defects encountered while building SaG—covering view-specific metric definitions, making unmeasured values unrepresentable as `0.0`, repurposing a test double as a simulation sandbox, verifying both individual rules and their composition, provenance-stamping every reported number, and preserving multigraph semantics end to end.

---

## Mapping to the SoSyM Tool Paper criteria

| Criterion | Where it is addressed |
|:---|:---|
| Technical quality | §3 metamodel, derivation rules and layers; §4 analyses (RM model, 19-pattern catalog, delta-aware gate, simulation, verified refactoring); §5 architecture and the invariants enforced by tests; Appendix A formulas |
| Effective presentation | Running example (mini-ATM) carried through §3–§7; metamodel class diagram; tool screenshots (Fig. 5); real CLI and API listings |
| Importance for the community | §8.1 |
| Lessons learnt | §8.2: eight lessons across designing, building, testing, evaluating and disseminating, each traced to a concrete defect or safeguard |
| Advance over the state of the art | §2 and Table 1 (capability comparison, not a quality comparison) |
| Availability | §9.3: GPL-3.0, GitHub, Zenodo archive (DOI to mint), `pip install -e .`, Docker |
| Applied in relevant scenarios | §7: running example end to end, five open-source architecture models, eight synthetic scenarios (cost, agreement, refactoring), CI scenario |

---

## Layout

```
docs/research/sosym/
├── README.md             # this file
├── REVISION_NOTES.md     # what changed in the revision and why, with evidence
├── outline.md            # section outline
├── draft.md              # Markdown rendering of the LaTeX (generated with pandoc)
├── scripts/
│   ├── make_numbers.py       # results/sosym/*.json  ->  latex/numbers.tex
│   ├── ci_scenario.py        # §7.5 CI measurements
│   └── prescribe_example.py  # §7.2 worked refactoring example
└── latex/                # authoritative sources
    ├── manuscript.tex, refs.bib, numbers.tex (generated), Makefile
    ├── figures/          # screenshots
    └── sections/         # one file per section + appendix
```

## Regenerating the numbers

Every number in §7 (and the few quoted elsewhere) is a macro in `latex/numbers.tex`, generated from provenance-stamped artifacts in `results/sosym/` (git-ignored; archive them with the release):

```bash
PYTHONPATH=. python reproduce/detection_validation.py --layer app    --output results/sosym/detection_validation_app.json
PYTHONPATH=. python reproduce/detection_validation.py --layer system --output results/sosym/detection_validation_system.json
PYTHONPATH=. python reproduce/detection_seed_sweep.py --output-json results/sosym/detection_seed_sweep.json --output-md results/sosym/detection_tables.md
PYTHONPATH=. python reproduce/run_prescribe_all.py --kappa 1.0 --output results/sosym/prescribe_all.json   # hours; see per-scenario timing loop in REVISION_NOTES.md
PYTHONPATH=. python docs/research/sosym/scripts/prescribe_example.py
PYTHONPATH=. python docs/research/sosym/scripts/ci_scenario.py
PYTHONPATH=. python docs/research/sosym/scripts/make_numbers.py
```

## Building the PDF

```bash
cd docs/research/sosym/latex && make clean && make
```

The Makefile stops at the first LaTeX error. A missing screenshot builds as a framed placeholder.

### Switching to the Springer template

1. Download the Springer Nature LaTeX bundle (`sn-jnl.cls`, `sn-basic.bst`).
2. Replace `\documentclass[11pt,a4paper]{article}` with `\documentclass[sn-basic]{sn-jnl}` and move title/authors/abstract into the class's macros.
3. Replace `\bibliographystyle{plainnat}` with `\bibliographystyle{sn-basic}`.

## Open items before upload

* Mint a Zenodo DOI for the tool release and replace the two `[TODO]` DOI placeholders (§9.3, Declarations).
* Fill in affiliations, ORCIDs, e-mail and grant number (`manuscript.tex`, `declarations.tex`).
* [RESOLVED] AuSE overlap statement resolved: AuSE was not submitted; held in reserve as an unpublished companion study (§Declarations).
* Switch to the Springer template.
