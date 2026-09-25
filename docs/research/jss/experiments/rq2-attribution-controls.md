# RQ2 — Where learned accuracy comes from: attribution controls

**Paper:** §6.2 (what each engine can see), §7.2, Table 9; §7.3, Table 10 (the `GAT` and `GBM-Feat` columns).
**Registration:** [`../PREREGISTRATION.md`](../PREREGISTRATION.md), Amendment 7. The amendment is post hoc, written after these results existed, and every contrast here is exploratory. `GBM-Feat` itself was declared post hoc in Amendment 3 and first run here.
**Extended results:** Supplement S31.

## Why these controls

On the native multigraph, every relation points away from Applications: Application → Topic, Node or Library. The untyped GATs aggregate from source to target only. As a result, `GAT`, `GAT-QoS` and Hybrid-GAT score each Application from its own feature vector. On their trained checkpoints, deleting every edge leaves every Application prediction unchanged. HGT reaches Applications through its reverse pass, which covers 35–59% of the graph.

The registered 2×2 ([rq2-matched-control.md](rq2-matched-control.md)) therefore compared typed message passing with per-component learning. Its Q factor also switched two inputs at once:
- the 16-D edge channel;
- three QoS node columns: `qos_weight`, `qos_weight_in` and `qos_weight_out`.

The controls below separate those ingredients.

| Arm (code name) | Node features | Edge channel | Graph model |
|---|---|---|---|
| `GAT` (`gl_full_cap`) | QoS-off | none | per-node at Applications |
| `GAT-QoS-nf` (`gl_full_qos16_nfmask`) | QoS-off | 16-D | per-node at Applications |
| `GAT-QoS` (`gl_full_qos16_cap`) | QoS-on | 16-D | per-node at Applications |
| `GBM-Feat` (`tab_gbm`) | QoS-off | — | none (gradient boosting per entity type) |
| `GBM-Feat-QoS` (`tab_gbm_qos`) | QoS-on | — | none |

Each input of `GAT-QoS-nf` is bit-identical to one parent arm (`_graft_qos_edge_attr` in [`cli/loso_evaluate.py`](../../../../cli/loso_evaluate.py); pinned by [`tests/test_attribution_controls.py`](../../../../tests/test_attribution_controls.py)). The QoS-weighted centralities are present in every arm.

## Reproduce

```bash
make -f reproduce/Makefile rq-attribution
```

This runs one CPU LOSO sweep of six arms, the contrasts, zero-shot for the five learned arms, and the receptive-field probe. It writes:
- `loso_attribution_cpu.json`
- `attribution_contrasts.json`
- `realworld_zeroshot_<arm>_attribution.json`
- `receptive_field_probe.json`

`GAT`, `GAT-QoS` and `Topo-QoS` reproduce their earlier rows bit for bit.

## Headline result

| Contrast | Δρ | Won | p_Holm |
|---|---:|:---:|---:|
| QoS node columns (GAT-QoS vs GAT-QoS-nf) | +0.095 | 11/12 | 0.024 |
| QoS edge channel (GAT-QoS-nf vs GAT) | −0.023 | 3/12 | 0.192 |
| QoS node columns, no graph model (GBM-Feat-QoS vs GBM-Feat) | −0.010 | 5/12 | 1.000 |
| Neural vs trees, QoS-off features (GAT vs GBM-Feat) | −0.079 | 1/12 | 0.037 |
| Neural vs trees, QoS-on features (GAT-QoS vs GBM-Feat-QoS) | +0.003 | 7/12 | 1.000 |

What the contrasts show:
- **Gradient boosting matches the learned engines.** `GBM-Feat` reaches ρ = 0.642, on par with `GAT-QoS` (0.635) and `HGT-QoS` (0.622). It does not significantly beat `Topo-QoS` (+0.089, p = 0.176).
- **The QoS gain of the untyped engine comes from the node columns.** The three columns carry it; the edge channel does not. The neural model needs them to reach what the trees get from the QoS-weighted centralities.
- **Transfer follows the same pattern.** Zero-shot ρ is 0.831 for `GAT`, 0.821 for `GAT-QoS-nf`, 0.805 for `GAT-QoS`, 0.757 for `GBM-Feat` and 0.759 for `GBM-Feat-QoS`.

## Notes cut from the paper

- **Seed stability follows the node columns.** Median within-fold seed SD is 0.083 for `GAT`, 0.136 for `GAT-QoS-nf` and 0.010 for `GAT-QoS`. The stabilisation previously credited to the QoS channel is the node columns'.
- **Receptive field does not explain the per-fold pattern.** HGT's share of the graph in reach does not correlate with its per-fold ρ (Spearman 0.12), its gain over `Topo-QoS` (−0.06) or the hybrid gain (0.20). `GAT-QoS` and `HGT-QoS` per-fold ρ correlate at 0.71, and every learned model falls short on Enterprise (0.407–0.533 against 0.795).
- **What is still open.** The registered directionality control (`HGT-QoS-U`) would make HGT per-node too, so it now tests whether HGT's message passing contributes at all. A substrate on which messages reach every scored component is needed before message passing can be tested as a mechanism: reverse edges for the untyped engines, or the Application–Library projection.
