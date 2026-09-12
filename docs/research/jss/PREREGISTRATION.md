# Pre-registration — HGL vs Topo-QoS under LOSO

Written **before** any number from the revised harness exists. Registered
2026-09-06. Anything not listed as primary or secondary below is exploratory and
must be reported as such.

## Motivation

`results/loso_all_variants.json` (the pre-revision artifact) reported HGT-QoS
ρ = 0.608 against the training-free Topo-QoS at 0.571 — +0.037 on 5/8 folds,
two-sided Wilcoxon p = 0.64. That artifact was subsequently shown to reproduce
from no commit in the repository and has been withdrawn; it is recorded here
only because it is what motivated this pass. The question it raised stands:
can a heterogeneous model beat Topo-QoS under LOSO by a margin that survives a
signed-rank test?

**Outcome (recorded after the fact, 2026-09-09).** No. On the twelve-fold
corpus the measured margin is +0.127 (9/12, W = 16.0, p = 0.077), and it rests
almost entirely on the ATM fold where Topo-QoS fails outright; excluding that
fold it falls to +0.078 (8/11, p = 0.148). Reported as registered, in
Section 7.1 of the manuscript. The secondary contrast and the typing
comparisons are reported in the same section and in Section 7.2.

## Primary comparison

- **Contrast:** `hgl_qos` (HGT-QoS) vs `topo_qos` (Topo-QoS).
- **Protocol:** leave-one-scenario-out, `cli/loso_evaluate.py`.
- **Population:** `--eval-population application` (Application stratum only).
- **Statistic:** per-fold mean Spearman ρ over seeds `42,123,456,789,2024`;
  two-sided Wilcoxon signed-rank over folds; α = 0.05.
- **Unit of analysis:** the fold. Not the seed, and not the node. Pairing at
  (fold × seed) would be pseudo-replication and is forbidden.

## Secondary comparison

- `hgl` (HGT) vs `topo_qos`, identical protocol.
- Holm correction across the primary and secondary tests.

## Attainable-p floor

The signed-rank statistic is discrete. Report the floor alongside every p:

| n folds | best attainable two-sided p | 1 small loss | 2 small losses | 3 small losses |
|---:|---:|---:|---:|---:|
| 8 | 0.0078 | 0.0156 | 0.0391 | 0.1094 |
| 12 | 0.0005 | 0.0010 | 0.0024 | 0.0068 |

"Small loss" means the losing fold is among the smallest |Δρ| in the set. A
large loss costs far more rank mass than the table suggests.

## Selection rule

Every hyperparameter and representation switch is selected by **inner-LOSO mean
ρ over the N−1 training scenarios of the outer fold**. The outer holdout takes
no part in selection — not in early stopping, not in checkpoint choice, not in
config choice. `reproduce/nested_loso_search.py` asserts this.

Two classes of change are admissible outside the nested search, and only these:

1. **Harness defects** with a mechanism argued independently of the outer
   metric — the per-fold depth inconsistency, validation on the training
   distribution, and the GAT/HGT training-set asymmetry.
2. **Representation switches motivated by a diagnostic that predates any outer
   result** — rank normalization, motivated by the drift measured in
   `results/feature_shift_diagnostic.md` (mpci 115.7×, qos_weight_in 15.5×,
   betweenness 12.3×).

Both classes are still passed through the nested search where they have a free
parameter; the point is that their *motivation* is not the outer number.

## Reporting commitment

The result is reported as measured. If the margin does not reach significance,
the finding is that a training-free QoS-weighted structural score is competitive
with heterogeneous graph learning for cross-domain criticality ranking. That is
a publishable result, not a failed pass, and it will not be re-run with a
different configuration in search of a different answer.

---

## Amendment 1 — inner selection budget (2026-09-06, before any outer result)

**Status when written:** the 12-fold regeneration sweep was still running and no
outer ρ under the revised harness existed yet. The original artifact
(`results/loso_all_variants.json`) had already been shown not to reproduce from
any commit, so there was no outer number to tune toward even accidentally.

**Original rule:** configurations selected by inner-**LOSO** mean ρ over the N−1
training scenarios of each outer fold.

**Amended rule:** configurations selected by inner **holdout** — each
configuration is scored on `k = 2` inner folds (deterministically chosen, evenly
spaced by size, via `_inner_indices`) rather than all 11.

**Reason:** measured cost. The sweep paces at ~40 s per model fit, so full
inner-LOSO is 12 × 8 × 11 = 1,056 fits ≈ 12 hours for one variant's search. The
inner loop's job is to *rank* configurations, not to estimate their performance;
a 2-fold inner sample is a coarser ranking signal but a sufficient one, at 192
fits ≈ 2 hours.

**What is unchanged:** the outer holdout still takes no part in selection — it is
not in the inner set at all, which `reproduce/nested_loso_search.py` asserts per
outer fold. Inner runs use the same 300-epoch budget as the outer folds, so the
configurations are ranked under the budget they will be reported under.

**Cost of the amendment:** selection is noisier, so `c*(k)` may vary more across
outer folds than it would under inner-LOSO. That variation is reported per fold
rather than hidden, and it is itself a finding: a configuration that cannot be
picked stably from the training set is not one to recommend.

---

## Amendment 2 — RQ2 confound controls (2026-09-12, before any control result)

**Status when written:** the four control arms below have been implemented and
smoke-tested for capacity only (3 epochs, 3 folds, CPU — enough to confirm each
arm builds and trains at its intended parameter count, far too little to produce
a ρ worth reading). No control arm has been run at the reported budget on any
fold, so no outcome was known when this amendment was registered. The decision
rule in the last section is therefore a commitment, not a description.

**Why this amendment exists.** Section 7.2 attributes the typed-vs-untyped LOSO
margin (+0.114, 11/12 folds, p = 0.0122) to "relational typing rather than to
substrate, training set, depth, or selection rule". An internal audit found that
list incomplete. Three factors the comparison did not hold constant:

| Factor | As published |
|:---|:---|
| Parameter budget | HGT 434,620 vs GAT-N-QoS 28,168 — a 15.4× gap |
| Message-passing directionality | HGT carries a reverse HGTConv (103,725 params, 24% of the model); the homogeneous GAT propagates along native edge direction only, and `I*(v)` is a downstream-reachability functional |
| Edge-channel width | HGT-QoS reads all 16 edge-feature dimensions; GAT-N-QoS reads dimension 0 alone |

A fourth concerns the label rather than the model: `I*`'s QoS ladder keys on
reliability and transport priority, which are edge-feature dimensions 9 and 11 —
given only to the `-QoS` arms. The QoS ablation therefore compares a predictor
that can see the oracle's severity multipliers against one that cannot.

**These are post-hoc.** They were not part of the original pre-registration and
must be reported as exploratory throughout. Their Wilcoxon contrasts are
Holm-corrected within their own family, never pooled with the primary contrast.

**The arms** (`saag/evaluation/variant_registry.py`, family `control`):

| Arm | Label | Isolates | Parameters |
|:---|:---|:---|---:|
| `gl_full_qos_cap` | GAT-N-QoS-C | capacity | 439,272 (1.011×) |
| `gl_full_cap` | GAT-N-C | capacity, QoS-off replication | 437,496 (1.007×) |
| `gl_full_qos16_cap` | GAT-N-QoS16-C | edge-channel width | 429,992 (0.989×) |
| `hgl_qos_uni` | HGT-QoS-U | directionality | 330,895 |

Plus a label-side arm: the full LOSO sweep repeated against a cache built with
`QOS_FACTOR=none`, so the oracle's QoS ladder is disabled.

The edge-channel arm deliberately receives the 7-dimensional relation one-hot
(dims 2–8) along with the QoS block. That hands the untyped model relation type
*as a feature*, which is the strict form of the control: it separates typing as
an input signal from typing as relation-specific parameters. That distinction
must be stated wherever the arm is reported.

### Decision rule, committed before the results exist

| Outcome | What we will report |
|:---|:---|
| Margin survives all four (shrinks < 0.02, ≥ 10/12 folds, p < 0.05) | Section 7.2 is strengthened: the attribution sentence is rewritten to name capacity, directionality, edge-channel width and label overlap explicitly, with the controls table. |
| Margin shrinks but holds significance | The **controlled** margin becomes the headline in the abstract, Section 1, Section 7.2, Section 8 and Section 9. The uncontrolled +0.114 is reported alongside it, labelled as the naive comparison, with the parameter ratio that accounts for the difference. |
| Margin does not survive | Section 7.2's claim becomes "typed message passing does not outperform a capacity-matched untyped GAT under distribution shift". The abstract's +0.114 is withdrawn. The contribution shifts to per-relation edge criticalities and attention over heterogeneous schemas, which have no untyped counterpart — the argument Section 8.1 already makes as a secondary one. |
| Mixed (e.g. capacity survives, directionality does not) | Reported per confound in the controls table, with the claim narrowed to the factors actually controlled. Results are **not** aggregated into a single verdict. |

**Reporting commitment.** Every arm that is run is reported, whichever way it
comes out, with its parameter count in the table. An arm may be dropped only for
a stated technical failure, never for its result. Parameter counts belong in the
table rather than the prose: their absence is what allowed this confound to
survive internal review in the first place.

**Corpus and device.** The controls are run against the corpus committed on
2026-09-10 and are compared only against arms re-baselined on the same corpus
and the same device. Rows measured on CPU and rows measured on GPU are never
placed in the same comparison; `config.device` in each artifact records which.
