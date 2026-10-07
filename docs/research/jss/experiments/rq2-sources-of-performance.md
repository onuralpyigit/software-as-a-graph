# RQ2 — Sources of predictive performance

**Paper:** §6.2 (`sec:rq2`), Table 7 (`tab:controls`, a digest of the control families); Figure 4C;
discussion in §7.1 (`sec:representation`).
**Supplement:** §S29 (`supp:matched-2x2`, Table S25 `tab:contrasts_matched`), §S30
(`supp:naive-2x2`), §S18 (`supp:fisherz`), §S23 (`supp:seed-robust`), §S38 (`supp:amendment9`),
§S40 (`supp:round8`), §S42 (`supp:controls`: Tables S69–S71, `tab:a14`, `tab:a16`, `tab:a17`).
**Status:** the matched 2×2 is registered secondary (Amendment 2). The control families F1–F13 are
registered secondary (Amendments 14, 16, 17, each written before its arms ran). The attribution
controls (Amendment 8) are exploratory.
**Registration:** [`../PREREGISTRATION.md`](../PREREGISTRATION.md), Amendments 2, 8, 9, 14, 16, 17
and 17b.

## Question

Where does learned accuracy come from? The candidate sources are:
- the graph representation (raw multigraph vs derived dependency graph);
- edge direction;
- degree features;
- the aggregator;
- relation typing;
- the "QoS" inputs;
- the model family.

## The controls, by source

Every learned arm runs under LOSO over the twelve folds with five seeds. Each family runs in one CPU
invocation with its comparators re-run to an exact match (gate G0). The rule outcomes are on the
amendment pages.

| Source | Control | Headline | Page |
|---|---|---|---|
| Relation typing × "QoS" inputs | capacity- and channel-matched 2×2 (`GAT`, `HGT`, `GAT-QoS`, `HGT-QoS`) | typing −0.014 (Holm 0.940); "QoS" +0.073 (Holm 0.127); interaction +0.001 | below |
| "QoS" inputs, with $w_\text{in}$ held | the same 2×2 with `qos_weight_in` held in both arms (F4) | +0.030 (9/12, Holm 0.330): more than half of the "QoS" effect was a QoS-weighted in-degree | [A14](amendments/a14-round8.md) |
| QoS node columns vs edge channel | `GAT-QoS-nf`, `GBM-Feat`, `GBM-Feat-QoS` | node columns +0.095 (Holm 0.024); edge channel −0.023 | [A8](amendments/a08-attribution-controls.md) |
| Message passing on the raw multigraph | `GBM-Feat` (no graph); `HGT-QoS-U` (no reverse pass) | `GBM-Feat` 0.642 ≈ `GAT-QoS` 0.635; `HGT-QoS-U` −0.010 (p = 0.91) | [A8](amendments/a08-attribution-controls.md) |
| Representation | the same learners on the `DEPENDS_ON` graph | `GAT-P-QoS` 0.748 vs `GAT-QoS` 0.635 (+0.113, 12/12) | [A9](amendments/a09-dependency-graph-learning.md) |
| Edge direction | `GAT-QoS-R` (every raw edge also reversed, shared weights) (F8) | direction alone +0.041 (Holm 0.064); `GAT-P-QoS` beats it by +0.072 (Holm 0.014) | [A16](amendments/a16-direction-control.md) |
| Degree features | `GAT-P-QoS−deg` (`in_degree`, $w_\text{in}$ zeroed) (F1) | −0.136 (1/12, Holm 0.0049) | [A14](amendments/a14-round8.md) |
| Aggregator | `GIN-P-QoS−deg` (sum aggregation) (F2) | +0.108 over attention without degree (Holm 0.157) | [A14](amendments/a14-round8.md) |
| Oracle-aligned features | `-min` arms with every oracle-aligned column zeroed (F11) | `GAT-P-QoS-min` beats `GAT-QoS-R-min` by +0.231 (Holm 0.0068); featureless `GIN-P-QoS-const` 0.719 | [A17](amendments/a17-round12.md) |
| Node order | `GAT-P-QoS-perm` (F13) and two more seeds (17b) | 0.712 / 0.740 / 0.734 against 0.748; spread ≈ 0.044 per fold | [A17](amendments/a17-round12.md) |
| Selection rule | the plan's nested selection, stage-1 grid (arm N) | `HGT-QoS` +0.055, `GAT-P-QoS` −0.054, neither significant | [A14](amendments/a14-round8.md) |
| Capacity, directionality | `GAT-w` and `HGT-QoS-U` (Amendment 2's last arms) | −0.011 (p = 0.68) and −0.010 (p = 0.91) against `HGT-QoS` | below |

## The capacity- and channel-matched 2×2 (Amendment 2)

The four learned arms of the registered GPU sweep (§S35) cross relation typing (T) with the QoS
channel (Q), but they are unmatched in two ways. The small untyped GATs have 28,168 parameters
against HGT's 434,620 (15.4×). And `GAT-S-w` reads a scalar edge weight, while `HGT-QoS` reads the
16-D vector. The unmatched 2×2 therefore credited typing with +0.234 when QoS was absent, plus a
strongly negative interaction (§S30). Amendment 2 registered controls that remove both differences
before any control result existed.

| Cell | Arm (variant id) | Parameters | Edge channel |
|---|---|---|---|
| ¬T ¬Q | `GAT` (`gl_full_cap`) | 437,496 | none |
| T ¬Q | `HGT` (`hgl`) | 434,620 | relation one-hot, $w(e)=1$ |
| ¬T Q | `GAT-QoS` (`gl_full_qos16_cap`) | 429,992 | full 16-D vector, including the relation one-hot |
| T Q | `HGT-QoS` (`hgl_qos`) | 434,620 | full 16-D vector |

`GAT-QoS` receives each edge's relation type as an input, so the precise conclusion is that
relation-typed *parameters* add nothing beyond relation-typed *inputs*.

**Two reinterpretations since registration.** Neither changes a number.
- **Per-node models (Amendment 8).** On the raw multigraph no relation targets an Application, so the
  untyped GATs score each Application from its own features. The 2×2 therefore compares typed message
  passing with per-component learning and says nothing about relational typing.
- **What the Q factor switches.** It switches the 16-D edge channel *and* three QoS node columns
  (`qos_weight`, `qos_weight_in`, `qos_weight_out`) together. The gain sits in the node columns, and
  more than half of it is the QoS-weighted in-degree $w_\text{in}$ (F4).

**What "QoS-off" means.** QoS-off arms set every edge weight to 1 and zero the three QoS node
columns. Four node centralities (PageRank, reverse PageRank, betweenness, eigenvector) are still
computed on the QoS-weighted projection. So "QoS-off" means "no explicit QoS channel", not "no QoS
information".

**Seed stability.** Median within-fold seed SD drops from 0.083 (`GAT`) to 0.010 (`GAT-QoS`). It
follows the node columns, not the edge channel (Amendment 8: 0.136 for `GAT-QoS-nf`).

**Late controls.** Every registered model arm has run.
- `GAT-w` is an untyped GAT at HGT's budget that reads a scalar QoS edge weight
  (`make -f reproduce/Makefile rq-capacity`). It reaches 0.633, −0.011 against `HGT-QoS`
  (5/12, p = 0.68).
- `HGT-QoS-U` is HGT without its reverse-direction parameters
  (`make -f reproduce/Makefile rq-directionality`). It reaches 0.632, −0.010 (6/12, p = 0.91), and
  transfers better zero-shot (0.804 vs 0.760).

With these two, the omnibus family is thirteen contrasts. Amendment 2's label-side arm (a LOSO sweep
with the oracle's QoS ladder disabled) was not run as a sweep. §4.3 bounds the label's QoS content
instead ([oracles-and-sensitivity.md](oracles-and-sensitivity.md)).

## Reproduce

```bash
make -f reproduce/Makefile rq2-matched RQ2_DEVICE=cpu   # or cuda; all four cells on ONE device
make -f reproduce/Makefile rq-capacity                  # GAT-w
make -f reproduce/Makefile rq-directionality            # HGT-QoS-U
```

The matched sweep writes `loso_rq2_matched.json` and `loso_significance_rq2_matched.json`. All four
cells must run in one invocation on one device, because the tests pair them by fold. The commands
for the other controls are on their amendment pages.

## Headline result

Representation, not model complexity, carries the learned gain on these oracles:
- Moving the same GAT from the raw multigraph to the derived dependency graph adds +0.113. Of that,
  +0.072 survives a reverse-edge control.
- Without any oracle-aligned feature, the dependency graph still beats the reverse-edge raw graph by
  +0.231.
- On the raw multigraph, relation typing adds nothing, message passing adds nothing (gradient
  boosting on the same per-node features matches), and the "QoS" gain is mostly a QoS-weighted
  in-degree.
- Learned-ranker differences below the node-order spread (about 0.04 per fold) or the configuration
  swing of nested selection (±0.055) should not be read as differences between models.
