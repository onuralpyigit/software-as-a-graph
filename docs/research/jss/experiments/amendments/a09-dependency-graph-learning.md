# Amendment 9: graph learning on the dependency graph

**Paper:** §3.4 (`sec:3.4`, dual graph views), §6.1 (`sec:rq1`, Table 5: `GAT-P-QoS` and `HGT-P-QoS`
rows), §6.2 (`sec:rq2`), §6.3 (Table 8: `GAT-P-QoS`), §7.1 (`sec:representation`), Figure 4.
**Supplement:** §S38 (`supp:amendment9`: per-fold LOSO, zero-shot and contrast tables), §S27 (row A9).
**Status:** registered secondary. Written 2026-09-26, before any learned arm's number existed. The
comparators (`InDeg`, `Reach`) had already been published by Amendment 7.
**Registration:** [`../../PREREGISTRATION.md`](../../PREREGISTRATION.md), Amendment 9 (commit
`0b3f7ca0`). Run with `make -f reproduce/Makefile rq-dependency-graph` at commit `a5846056`, clean
tree, CPU.

## Question

Amendment 8 ([page](a08-attribution-controls.md)) showed that every learned ranker in the paper read
the raw multigraph. On that graph no relation targets an Application, so the untyped GATs are
per-node models and HGT reaches Applications only through its reverse pass. Meanwhile, counting
dependents on the derived Application–Library `DEPENDS_ON` graph ([A7](a07-training-free.md)) gave
`InDeg` 0.764 and `Reach` 0.732.

The question: do the same learners, run on the dependency graph itself, match or beat those counts?

## Substrate and receptive field

The substrate is `derive_depends_on_edges` of each committed topology, the same edge set `InDeg` and
`Reach` read:
- Application and Library nodes;
- one relation, `DEPENDS_ON`, with edges pointing from dependent to dependency;
- $w(e)$ as the edge scalar.

Node features and labels are bit-identical to the native build
(`tests/test_dependency_graph_substrate.py`); only the edges change. Because `GATConv` aggregates
source → target, every component receives messages from its dependents. The receptive-field probe
(`results/receptive_field_probe_dependency_graph.json`) confirms it:
- **Untrained 3-layer GAT.** The receptive field is the Application plus its dependents within three
  hops, for 100% of Applications. On the raw multigraph it is one node.
- **Trained checkpoints.** Deleting every edge moves Application predictions by up to 0.24–0.61, so
  the trained models use the graph. The raw-multigraph GAT checkpoints move by exactly 0.

## Arms

| Label | Variant id | Edge channel | Params | Raw-multigraph counterpart |
|---|---|---|---:|---|
| `GAT-P` | `gl_proj_cap` | none | 437,496 | `GAT` |
| `GAT-P-QoS` | `gl_proj_qos16_cap` | 16-D QoS | 429,992 | `GAT-QoS` |
| `GAT-P+InDeg` | `gl_proj_qos16_indeg_prior` | 16-D QoS + rank-normalized `InDeg` prior | 431,433 | Hybrid-GAT |
| `HGT-P-QoS` | `hgl_proj_qos` | 16-D QoS, bidirectional, width 100 | 430,680 | `HGT-QoS` |

The protocol is 12 LOSO folds × 5 seeds, 300 epochs and 3 layers, plus zero-shot on the five system
models. `GAT-P+InDeg` was registered as "Hybrid-GAT-P". Amendment 13 moved it to the supplement,
because its prior is a reference.

## Reproduce

```bash
make -f reproduce/Makefile rq-dependency-graph
```

## Artifacts

All of these are tracked in git under `results/`:
- `loso_dependency_graph_cpu.json`
- `realworld_zeroshot_{gl_proj_cap,gl_proj_qos16_cap,gl_proj_qos16_indeg_prior,hgl_proj_qos}_dependency_graph.json`
- `receptive_field_probe_dependency_graph.json`
- `dependency_graph_contrasts.json`: per-fold rows, contrasts, zero-shot, decisions and checks.

Checks recorded in `dependency_graph_contrasts.json`:
- `InDeg` and `Reach`, recomputed on the LOSO cache's labels, match `tf_baselines.json` exactly.
- `topo_qos` and `gl_full_qos16_cap`, re-run in the same invocation, reproduce their earlier
  artifacts bit for bit.

## Outcome

LOSO means: `GAT-P` 0.653, `GAT-P-QoS` 0.748, `GAT-P+InDeg` 0.758, `HGT-P-QoS` 0.514. Zero-shot
means: 0.830, 0.806, 0.792, 0.746. Per-fold and per-system values are in §S38.

The registered contrasts (Holm across 12) that the decision rules read:

| Contrast | Δρ | 95% CI | Won | p_Holm |
|:---|---:|:---:|:---:|---:|
| `GAT-P` vs `InDeg` | −0.111 | [−0.163, −0.059] | 1/12 | 0.017 |
| `GAT-P` vs `GAT` | +0.090 | [+0.042, +0.148] | 11/12 | 0.012 |
| `GAT-P-QoS` vs `InDeg` | −0.017 | [−0.072, +0.051] | 4/12 | 1.000 |
| `GAT-P-QoS` vs `GAT-QoS` | +0.113 | [+0.077, +0.162] | 12/12 | 0.006 |
| `GAT-P+InDeg` vs `InDeg` | −0.006 | [−0.015, +0.002] | 3/12 | 0.881 |
| `GAT-P+InDeg` vs Hybrid-GAT | +0.075 | [+0.039, +0.113] | 11/12 | 0.021 |
| `HGT-P-QoS` vs `InDeg` | −0.250 | [−0.377, −0.143] | 0/12 | 0.006 |
| `HGT-P-QoS` vs `HGT-QoS` | −0.107 | [−0.238, +0.014] | 4/12 | 0.881 |

The other four registered contrasts (each arm vs `Reach`) are in §S38.

| Rule | Outcome |
|---|---|
| D1: some arm beats `InDeg` | **Not triggered.** |
| D2: no arm differs from `InDeg` | **Not triggered.** `GAT-P` and `HGT-P-QoS` are significantly worse. |
| D3: every arm's mean Δ vs `InDeg` is below 0 | **Triggered.** |
| M: a dependency-graph arm beats its raw-multigraph counterpart | **Triggered** for all three GAT arms (+0.075 to +0.113). The missing receptive field was a real cause of the raw-multigraph deficit. |
| Z: best zero-shot arm < `Reach` (0.938) | **Triggered.** |

## How the paper reads it now

- **Representation.** Moving the same learner to the dependency graph is worth +0.08 to +0.11.
  `GAT-P-QoS` is the paper's best learned ranker. Amendment 16 ([page](a16-direction-control.md))
  showed that +0.072 of the gain survives a reverse-edge control.
- **References, not predictors.** This page first recommended `InDeg` and `Reach` as the deployment
  rankers. Amendment 13 ([page](a13-reference-demotion.md)) withdrew that: both restate $I^*$'s
  propagation rule, so the paper reports them as references and says that learners *approach* them.
  The paper also says `GAT-P-QoS` is "not significantly different" from `InDeg`. "Matches" was
  dropped, because equivalence within ±0.05 is not established ([A14](a14-round8.md), F3b).
- **Active stratum.** The learners rank components that do cause impact better than `Reach` and
  `Topo-QoS` do: ρ>0 0.440 for `GAT-P-QoS`, against 0.286 and 0.280.

## Limitations

- **Folds are not independent.** The twelve folds share ten of their eleven training graphs, so
  p-values and CIs understate dispersion.
- **`HGT-P-QoS` did not train stably.** Its median within-fold seed SD is 0.208, against 0.026 for
  `GAT-P-QoS`. Some seeds invert the ranking on ATM, SCADA and IoT, so its low mean reflects a
  training failure and says nothing about the substrate. It was run as registered, and nothing was
  tuned afterwards.
- **Training population.** Broker, Topic and Node labels are absent from the projection, so the arms
  train on Applications and Libraries only.
- **Smoke run.** A 2-epoch, 1-seed wiring check ran after the amendment was committed. Nothing was
  changed on the basis of it.
