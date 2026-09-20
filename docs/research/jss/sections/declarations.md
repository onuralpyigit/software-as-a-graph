# Declarations

**CRediT Authorship Contribution Statement.** **Ibrahim Onuralp Yigit:** Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Writing – original draft, Visualization. **Feza Buzluca:** Conceptualization, Methodology, Validation, Writing – review and editing, Supervision.

**Declaration of Competing Interest.** The authors declare no competing financial interests or personal relationships that could have influenced this work.

**Funding.** This research received no grant from public, commercial, or not-for-profit funding agencies.

**Data Availability.** The replication package (datasets, harnesses, checkpoints, scripts) is available on Zenodo under DOI [10.5281/zenodo.14922108](https://doi.org/10.5281/zenodo.14922108) [93] with `uv`/`pip` environments. Synthetic datasets regenerate byte-identically. The deposit ships the artifacts backing every reported table as a single dated bundle (`SaG_JSS_Results_<stamp>/`), assembled by `reproduce/cut_results_bundle.py`: sixteen JSON artifacts plus the tables and figures rendered from them, and a `MANIFEST.json` recording, per file, a SHA-256 digest, the commit it was produced at, whether that working tree was modified, and the corpus digest the run describes. Five artifacts (`atm_scale_sweep_v3`, `loso_significance_v5`, `qos_label_ablation`, `threshold_sensitivity_v3`, `topic_weight_sensitivity_v3`) carry no provenance block, so for those the correspondence to this corpus is asserted by the bundle rather than proved by the artifact. The verification script (`reproduce/reconcile_manuscript.py`) runs standalone on a clean machine against the Zenodo deposit, verifying table quantities against JSON artifacts with git provenance. Prose definitions, equations, and sensitivity tables are verified manually.

**Declaration of Generative AI.** The authors used Anthropic’s Claude for typesetting assistance, taking full responsibility for all content. No generative AI was used to design experiments, analyze data, or generate results.
