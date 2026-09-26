# Length justification

*For the "Comments to the Editor" field at submission, per the JSS Guide for Authors: "It is
encouraged that authors submit full-length papers of less than 36 pages single-column… If your
manuscript is longer, please include an explanation in your submission as to why the length is
justified."*

---

The manuscript runs to **27 single-column pages** in the `elsarticle` preprint class, including all
declarations and the reference list (100 entries), well within the recommended limit. No explanation
is required.

The body keeps each headline result with the evidence needed to check it (11 tables, 5 figures).
Detail was moved out in two directions:
- **Supplementary Material (S1–S33, 30 pages):** sensitivity sweeps, AHP matrices, corpus parameters
  and composition, the feature schema, anti-pattern and attention analyses, convergent validity,
  in-distribution results, the unmatched 2×2, active-stratum and bootstrap tables, the registered
  GPU LOSO sweep, the gate-vs-oracle cost table, the full predictor taxonomy, per-fold hybrid
  results, the amendment log with the omnibus Holm correction, the Amendment 7 dependency counts and
  QoS-attribution controls, the Amendment 8 attribution, directionality and capacity controls, and
  the HGT message-passing equations.
- **Public experiment pages** (`docs/research/jss/experiments/` at the submission tag): protocols,
  hyperparameters, reproduction commands and artifact names for every experiment.

Every table value in both documents is mechanically reconciled against the artifact that produced it.
