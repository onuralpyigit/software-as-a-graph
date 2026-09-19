# Length justification

*For the "Comments to the Editor" field at submission, per the JSS Guide for Authors: "It is
encouraged that authors submit full-length papers of less than 36 pages single-column… If your
manuscript is longer, please include an explanation in your submission as to why the length is
justified."*

---

The manuscript runs to **35 single-column pages** in the `elsarticle` preprint class, of which
**2.8 pages are the reference list** (93 entries). The main text and declarations occupy 32.2 pages.
The manuscript strictly conforms to the journal's recommended limit of less than 36 single-column pages.

We have already moved substantial material out of the body. Fourteen supplementary sections now
carry the parameter-sensitivity sweeps (OFAT and Morris screening), the AHP matrices and their
consistency diagnostics, the generative corpus parameters, the per-scenario corpus composition, the
typed node feature schema, the anti-pattern detection benchmark, the explanation layer's real-world
evaluation, the HGT attention distributions, the cross-oracle convergent-validity analysis, the
in-distribution significance tests, the corpus-subset map, and the running-example figure. What
remains in the body is what a referee needs in order to check a claim without leaving the page.

**1. The paper's contribution is a set of negative and boundary results, and those cost pages to
state precisely.** JSS explicitly welcomes "studies with negative results". This study reports that
its proposed model does *not* significantly outperform an unparameterized QoS-weighted centrality
baseline; that its two architectural mechanisms — relation typing and QoS edge encoding — are
*substitutes rather than complements*, each carrying a main effect (+0.134 and +0.187, Holm-corrected
p = 0.0015) but interacting sub-additively (−0.199, negative on all twelve folds, p = 0.0005); that zero-shot transfer to real systems is not
established once tied labels are excluded, and inverts on two of five architectures; that a
label-free confidence signal previously reported does not replicate; that the explanation layer's
elicited AHP weights are anti-predictive and three of its five AHP matrices are rank-one by
construction; and that the static gate is roughly eleven times more expensive than the simulation it
was intended to displace. Each of these is a claim *against* our own framework, and each required the
ablation, the corrected baseline, or the sensitivity analysis that establishes it. A paper reporting
a clean positive result would be shorter; it would also be less useful.

**2. The empirical program is broad, and each result is scoped to a stated population.** Seven
predictor configurations are evaluated across twelve synthetic architectures under inductive
leave-one-scenario-out cross-validation, twelve more in-distribution, and five authentic open-source
systems zero-shot — 2,812 components against four simulation oracles. Every research question is
answered on an explicitly bounded corpus subset, because pooling entity types triggers a
demonstrated Simpson's paradox. Compressing further would mean dropping an evaluation condition or
leaving the reader unable to reconstruct which comparison rests on which data.

**3. Reproducibility claims are load-bearing and stated in the text.** The corpus regenerates
byte-identically from committed configurations; 403 reported table values are mechanically
reconciled against the JSON artifacts that produced them by a committed script that refuses a clean
run when an artifact is absent or was produced from a modified working tree; the input–label
independence guarantee is asserted in continuous integration; and the primary out-of-distribution
comparison was pre-registered before any result existed, with three dated amendments recording every
subsequent protocol change. Supporting these claims requires stating protocols — evaluation
populations, oracle assignment, substrate parity, model-selection rules — that a shorter paper would
leave implicit and a referee could not check.

**In alignment with the 36-page limit**, the manuscript incorporated the planned trims: the
Reliability–Maintainability decomposition table (Section~5.1 $\to$ Table~S3) and the formal heterogeneous message-passing
equations (Section~4.1.2 $\to$ Section~S1.1) now reside in the online supplementary material, alongside
concise inline syntheses of repetitive experimental protocols, bringing the complete manuscript
comfortably to **35 single-column pages** (including all declarations and 93 references).
