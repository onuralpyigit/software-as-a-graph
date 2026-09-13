# Length justification

*For the "Comments to the Editor" field at submission, per the JSS Guide for Authors: "It is
encouraged that authors submit full-length papers of less than 36 pages single-column… If your
manuscript is longer, please include an explanation in your submission as to why the length is
justified."*

---

The manuscript runs to **38 single-column pages** in the `elsarticle` preprint class, of which
**three are the reference list** (92 entries). The main text and declarations occupy 35 pages. We are
two pages over the encouraged limit and offer this explanation as the Guide asks.

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
*substitutes rather than complements*, each worth a large, robustly significant gain alone
(+0.234 and +0.287, Holm-corrected p = 0.002 and 0.003) and almost nothing once the other is present
(+0.035 and +0.087, both non-significant); that zero-shot transfer to real systems is not
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
byte-identically from committed configurations; 271 reported table values are mechanically
reconciled against the JSON artifacts that produced them by a committed script that refuses a clean
run when an artifact is absent or was produced from a modified working tree; the input–label
independence guarantee is asserted in continuous integration; and the primary out-of-distribution
comparison was pre-registered before any result existed, with three dated amendments recording every
subsequent protocol change. Supporting these claims requires stating protocols — evaluation
populations, oracle assignment, substrate parity, model-selection rules — that a shorter paper would
leave implicit and a referee could not check.

**If the editors prefer the manuscript at 36 pages**, the two further cuts we would make are the
Reliability–Maintainability decomposition table (Section 5.1) and the heterogeneous message-passing
equations (Section 4.1.2), both to the supplement. We have not made them pre-emptively because both
are the definitions the rest of the paper reasons from, and we would rather the editors make that
call than present a body that cannot be read without the supplement.
