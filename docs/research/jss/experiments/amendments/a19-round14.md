# Amendment 19: round-14 referee controls

> **Not yet on `main`.** This amendment, its code, its artifacts and the manuscript revision that
> reports it are on branch `jss-revision-round14` (`b3be1c4a`). The paper references below use that
> branch's numbering, and the `make` targets exist only there. This page was copied from the branch
> so that the amendment log is complete. When the branch merges, keep this file and drop the branch's
> `experiments/amendment19-round14.md`.

**Paper (round-14 branch):** §4.2 (tie order), §6.1 (rate-fed queue-flow GNNs), §6.2 (aggregator
control, tie-aware loss, learning curve), §7.2, §7.4, §7.5.
**Supplement (round-14 branch):** Tables `tab:a19` and `tab:a19lc`; Figure 6.
**Status:** registered secondary. Committed with its code before any arm ran.
**Registration:** [`PREREGISTRATION.md` on the round-14 branch](https://github.com/onuralpyigit/software-as-a-graph/blob/jss-revision-round14/docs/research/jss/PREREGISTRATION.md), Amendment 19.
**Review:** [review_2026-10-06_round14.md (round-14 branch)](https://github.com/onuralpyigit/software-as-a-graph/blob/jss-revision-round14/docs/research/jss/reviews/review_2026-10-06_round14.md).

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

## Outcome

Artifacts: `data/benchmarks/referee_round14_{amendment19,lc,descriptive}.json` (commit `48c196fb`, clean); Supplementary Tables `tab:a19` and `tab:a19lc`; Figure 6. All nine re-run comparators reproduced exactly (G0).

| Family | Result | Rule |
|---|---|---|
| F14 | Under sum aggregation the dependency graph still beats the raw multigraph with reverse edges: +0.239 without oracle-aligned features (12/12, Holm 0.0015), +0.064 with them, +0.289 with no node features | F14a |
| F15 | Rate-fed GNNs improve on the rate-blind one (best 0.665, +0.067) but stay below GBM (0.799) and Eq. 7 (0.830; −0.165, Holm 0.0015) | F15b (second condition) |
| F16 | Tie-aware loss leaves GAT-P-QoS unchanged (0.747) and the direction-controlled gain intact (+0.069, Holm 0.019); permutation spread 0.047 | F16 holds; S2 |
| LC | GAT-P-QoS 0.605 → 0.670 → 0.748 for K = 1, 4, 11; gap to InDeg 0.160 → 0.017; GAT-QoS flat from K = 4 | LC-c overall |
| small | GAT-S-P-QoS 0.638 (−0.110) | — |

The main-sweep artifact was re-stamped from a clean worktree after a concurrent edit dirtied the checkout (see the results log in `PREREGISTRATION.md`).
