# Length justification

*For the "Comments to the Editor" field at submission, per the JSS Guide for Authors: "It is
encouraged that authors submit full-length papers of less than 36 pages single-column… If your
manuscript is longer, please include an explanation in your submission as to why the length is
justified."*

---

The manuscript runs to **24 single-column pages** in the `elsarticle` preprint class, including all
declarations and the reference list (94 entries), well within the recommended limit of 36 pages. No special
length justification is required.

The body maintains each headline result with the empirical evidence needed to inspect it (10 tables, 5 figures).
Secondary analyses, derivations, and protocol details are organized into:
- **Supplementary Material (S1–S43):** sensitivity sweeps, AHP pairwise comparison matrices and consistency proofs,
  corpus parameters and composition, the feature schema, anti-pattern and attention analyses, convergent validity,
  in-distribution results, active-stratum and bootstrap tables, the registered GPU LOSO sweep, gate-vs-oracle
  cost breakdowns, the full predictor taxonomy, per-fold hybrid results, the full control-arm tables (S42),
  and the registration amendment log with the omnibus Holm correction.
- **Public experiment pages** (`docs/research/jss/experiments/` at the submission tag): protocols,
  hyperparameters, reproduction commands, and artifact names for every experiment.

Every table value across both documents is mechanically reconciled against the artifact that produced it.

---

### Production Note: Figure File Mapping

The manuscript body contains 5 figures and the supplementary material contains 3 figures. For Elsevier production typesetters, the mapping between logical document numbers and graphics files is as follows:
- **Body Figure 1** (Overview of SaG): `figures/Figure_1.pdf`
- **Body Figure 2** (Running example: structural and derived graphs): `figures/Figure_2.pdf`
- **Body Figure 3** (Ranking methods and evaluation): `figures/Figure_3.pdf`
- **Body Figure 4** (Main Results under LOSO): `figures/Figure_4.pdf`
- **Body Figure 5** (Critical Set Recall): `figures/Figure_5.pdf`
- **Supplementary Figure S1** (RM composite vs. AHP shrinkage): `figures/Figure_S1.pdf`
- **Supplementary Figure S2** (Relational attention on the ATM case study): `figures/Figure_S2.pdf`
- **Supplementary Figure S3** (Proposed explanation layer): `figures/Figure_S3.pdf`
