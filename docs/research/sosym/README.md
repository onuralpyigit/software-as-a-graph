# Software-as-a-Graph (SaG) — SoSyM Tool Paper Package

Manuscript, figures and reproduction scripts for submitting **Software-as-a-Graph** to *Software and Systems Modeling* (Springer), collection **Tool Paper**.

* **Title:** *Software-as-a-Graph: An Open-Source Tool for Multi-Layer Graph Modeling, Quality Diagnosis, and Simulation-Verified Refactoring of Publish–Subscribe Architectures*
* **Status:** revised for the tool-paper call (see [REVISION_NOTES.md](REVISION_NOTES.md)). Open items before upload are listed at the end of this file.

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
* Decide on the AuSE overlap statement (Declarations) once the AuSE status is known; see REVISION_NOTES.md.
* Switch to the Springer template.
