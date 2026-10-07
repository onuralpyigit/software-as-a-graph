# Amendment 8: attribution controls on the raw multigraph

**Paper:** §3.4 (`sec:3.4`, dual graph views: what each ranker can see), §6.2 (`sec:rq2`, "Message passing, node order and the
selection rule"), §7.1.
**Supplement:** §S28 (`supp:amendments`, amendment log row A8).
**Status:** exploratory. Post hoc: written 2026-09-26, after every run below existed. It registers
no contrast and changes no registered conclusion. `GBM-Feat` itself was declared post hoc in
Amendment 3 and first run here.
**Registration:** [`../../PREREGISTRATION.md`](../../PREREGISTRATION.md), Amendment 8. It was first
committed as "Amendment 7" on a parallel branch and renumbered at merge, with its text unchanged.

## Question

On the raw multigraph, every relation points away from Applications: Application → Topic, Node or
Library. `GATConv` aggregates from source to target only. As a result, `GAT`, `GAT-QoS` and
Hybrid-GAT score each Application from its own feature vector: on their trained checkpoints,
deleting every edge leaves every Application prediction unchanged. HGT reaches Applications through
its reverse pass, which covers 35–59% of the graph.

The registered 2×2 ([RQ2](../rq2-sources-of-performance.md)) therefore compared typed message
passing with per-component learning. Its Q factor also switched two inputs at once:
- the 16-D edge channel;
- three QoS node columns: `qos_weight`, `qos_weight_in` and `qos_weight_out`.

This amendment separates those ingredients.

## Arms

| Arm (variant id) | Node features | Edge channel | Graph model |
|---|---|---|---|
| `GAT` (`gl_full_cap`) | QoS-off | none | per-node at Applications |
| `GAT-QoS-nf` (`gl_full_qos16_nfmask`) | QoS-off | 16-D | per-node at Applications |
| `GAT-QoS` (`gl_full_qos16_cap`) | QoS-on | 16-D | per-node at Applications |
| `GBM-Feat` (`tab_gbm`) | QoS-off | — | none (gradient boosting per entity type) |
| `GBM-Feat-QoS` (`tab_gbm_qos`) | QoS-on | — | none |

Each input of `GAT-QoS-nf` is bit-identical to one parent arm (`_graft_qos_edge_attr` in
[`cli/loso_evaluate.py`](../../../../../cli/loso_evaluate.py); pinned by
[`tests/test_attribution_controls.py`](../../../../../tests/test_attribution_controls.py)). The
QoS-weighted centralities are present in every arm. Before its first run, `GBM-Feat` was given the
neural arms' per-graph label transform (`normalize_labels_robust`); the earlier code fed it raw
labels.

## Reproduce

```bash
make -f reproduce/Makefile rq-attribution
```

This runs one CPU LOSO sweep of six arms, the contrasts, zero-shot for the five learned arms, and
the receptive-field probe (`reproduce/receptive_field_probe.py`). `GAT`, `GAT-QoS` and `Topo-QoS`
reproduce their earlier rows bit for bit.

## Artifacts

- `results/loso_attribution_cpu.json`
- `results/attribution_contrasts.json`
- `results/realworld_zeroshot_<arm>_attribution.json`
- `results/receptive_field_probe.json`

## Outcome

Five exploratory contrasts, Holm within the family:

| Contrast | Δρ | Won | p_Holm |
|---|---:|:---:|---:|
| QoS node columns (`GAT-QoS` vs `GAT-QoS-nf`) | +0.095 | 11/12 | 0.024 |
| QoS edge channel (`GAT-QoS-nf` vs `GAT`) | −0.023 | 3/12 | 0.192 |
| QoS node columns, no graph model (`GBM-Feat-QoS` vs `GBM-Feat`) | −0.010 | 5/12 | 1.000 |
| Neural vs trees, QoS-off features (`GAT` vs `GBM-Feat`) | −0.079 | 1/12 | 0.037 |
| Neural vs trees, QoS-on features (`GAT-QoS` vs `GBM-Feat-QoS`) | +0.003 | 7/12 | 1.000 |

- **Gradient boosting matches the learned rankers.** `GBM-Feat` reaches 0.642, on par with `GAT-QoS`
  (0.635) and `HGT-QoS` (0.622). It does not significantly beat `Topo-QoS` (+0.089, p = 0.176).
- **The QoS gain of the untyped model comes from the node columns.** The edge channel carries none
  of it. The neural model needs the columns to reach what the trees get from the QoS-weighted
  centralities. Amendment 14 (F4) later showed that most of it is the QoS-weighted in-degree.
- **Seed stability follows the node columns.** Median within-fold seed SD is 0.083 for `GAT`, 0.136
  for `GAT-QoS-nf` and 0.010 for `GAT-QoS`.
- **Zero-shot** ρ is 0.831 for `GAT`, 0.821 for `GAT-QoS-nf`, 0.805 for `GAT-QoS`, 0.757 for
  `GBM-Feat` and 0.759 for `GBM-Feat-QoS`.
- **Receptive field does not explain the per-fold pattern.** HGT's share of the graph in reach does
  not correlate with its per-fold ρ (Spearman 0.12), its gain over `Topo-QoS` (−0.06) or the hybrid
  gain (0.20).

## What it led to

Message passing could be tested only on a substrate where messages reach every scored component.
Amendment 9 ([page](a09-dependency-graph-learning.md)) ran the same learners on the dependency
graph. Amendment 16 ([page](a16-direction-control.md)) added the reverse-edge raw-graph control.
