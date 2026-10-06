# Amendment 19: round-14 referee controls

**Paper:** §4.2 (tie order), §6.1 (rate-fed queue-flow GNNs), §6.2 (aggregator control, tie-aware loss, learning curve), §7.2, §7.4, §7.5.
**Registered:** [`PREREGISTRATION.md`](../PREREGISTRATION.md), Amendment 19, committed with its code before any arm ran.
**Review:** [review_2026-10-06_round14.md](../reviews/review_2026-10-06_round14.md).

## Reproduce

```bash
make -f reproduce/Makefile rq-amendment19            # 21 LOSO arms (9 re-run comparators) + 3 zero-shot runs
make -f reproduce/Makefile rq-amendment19-lc         # learning curve: 3 learners x K in {1,2,4,8} x 3 draws
make -f reproduce/Makefile rq-amendment19-analysis   # gate, families, rules, descriptive analyses
```

Run the LOSO sweeps from the main checkout, from a clean tree: the artifacts record `dirty`, and the reconciler refuses dirty artifacts.

## Arms

| Family | Arm | What changes |
|---|---|---|
| F14 | `GIN-QoS-R`, `GIN-QoS-R-min`, `GIN-QoS-R-const` | sum aggregation (GINE) on the raw multigraph with every edge also reversed, with full, oracle-aligned-free and constant features |
| F15 | `GAT-P-QoS→dyn+rate`, `GAT-P-QoS→dyn+rate-e`, `GIN-P-QoS→dyn+rate-e` | the `I_dyn`-trained GNN given each node's summed declared rate, and each Rule-1 edge's share of Eq. 7 |
| F16 | `GAT-P-QoS-tie`, `GAT-QoS-R-tie`, `GAT-P-QoS-tie-perm17/18/19` | a tie-aware listwise loss (tied labels as groups), with three node-order permutations |
| small | `GAT-S-P-QoS` | GAT-P-QoS at width 64 |
| LC | `GAT-P-QoS`, `GIN-P-QoS`, `GAT-QoS` | trained on K ∈ {1, 2, 4, 8} of the fold's eleven training scenarios (three nested draws) |

## Results

Pending.
