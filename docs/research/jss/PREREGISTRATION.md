# Pre-registration — HGL vs Topo-QoS under LOSO

Written **before** any number from the revised harness exists. Registered
2026-09-06. Anything not listed as primary or secondary below is exploratory and
must be reported as such.

## Motivation

`results/loso_all_variants.json` (the pre-revision artifact) reports HGT-QoS
ρ = 0.608 against the training-free Topo-QoS at 0.571 — +0.037 on 5/8 folds,
two-sided Wilcoxon p = 0.64. The unweighted HGT is significantly *worse*
(−0.132, 1/8, p = 0.039). This pass asks whether a heterogeneous model can beat
Topo-QoS under LOSO by a margin that survives a signed-rank test.

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
