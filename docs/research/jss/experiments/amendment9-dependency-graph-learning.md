# Amendment 9: graph learning on the dependency graph

**Paper:** not in the manuscript. This is an exploratory evaluation for the JSS paper; the v4
manuscript is unchanged.
**Registered:** [`PREREGISTRATION.md`](../PREREGISTRATION.md), Amendment 9 (commit `0b3f7ca0`),
before any learned arm's number existed. The closed-form comparators (`InDeg`, `Reach`) were
already published by Amendment 7.
**Run:** `make -f reproduce/Makefile rq-dependency-graph` at commit `a5846056`, clean tree, CPU.

## The question

Should the paper predict cascading-failure impact directly from the dependency graph, or train
graph-learning models on it? Two earlier results frame the question:

- **Counting.** Amendment 7 ([page](amendment7-training-free.md)) counted each component's
  dependents on the Application–Library `DEPENDS_ON` projection (Rules 1 and 5, edges point from
  dependent to dependency).
  - `InDeg` (direct dependents) reached LOSO ρ = 0.764.
  - `Reach` (transitive dependents) reached 0.732.
  - Both beat every learned engine in the paper.
- **The learned engines never saw that graph.** Amendment 8
  ([page](rq2-attribution-controls.md)) showed they all read the native multigraph. There, no
  relation targets an Application, so the untyped GATs score each Application from its own
  features. HGT reaches Applications only through its reverse pass.

No learned model had been run under LOSO or zero-shot on the dependency graph itself. This
experiment runs the paper's learners there.

## Substrate and receptive field

The substrate is `derive_depends_on_edges` of each committed topology, the same edge set `InDeg`
and `Reach` read:
- Application and Library nodes, in native order;
- one relation, `DEPENDS_ON`: four (App|Lib → App|Lib) triples on the synthetic corpus, two on
  the real system models;
- `w(e)` as the edge scalar.

Node features and labels are bit-identical to the native build
(`tests/test_dependency_graph_substrate.py`); only the edges change. Scoring is on the
Application population against the native labels, exactly as for every published arm.

`GATConv` aggregates from source to target, so on this graph every component receives messages
from its **dependents**. The receptive-field probe
(`results/receptive_field_probe_dependency_graph.json`) confirms this:

| Probe | Result |
|---|---|
| 3-layer GAT, random weights, 12 folds | Receptive field = the Application plus its dependents within 3 hops, for 100% of Applications; 16–207 nodes on average per fold. On the native graph it is 1 node. |
| HGT (bidirectional) | Covers 51–100% of the graph. |
| Trained checkpoints (seed 42) | Deleting every edge moves Application predictions by up to 0.24–0.61, so the trained models use the graph. The native GAT checkpoints move by exactly 0. |

## Arms

| Label | id | Edge channel | Params | Native counterpart |
|---|---|---|---:|---|
| GAT-P | `gl_proj_cap` | none | 437,496 | GAT |
| GAT-P-QoS | `gl_proj_qos16_cap` | 16-D QoS | 429,992 | GAT-QoS |
| Hybrid-GAT-P | `gl_proj_qos16_indeg_prior` | 16-D QoS + rank-normalised `InDeg` prior | 431,433 | Hybrid-GAT (Topo-QoS prior) |
| HGT-P-QoS | `hgl_proj_qos` | 16-D QoS, bidirectional, width 100 | 430,680 | HGT-QoS |

Protocol: 12 LOSO folds × 5 seeds, 300 epochs, 3 layers; zero-shot on the five system models.

## LOSO results

Spearman ρ on Applications, mean over 5 seeds. The native arms' values come from their own clean
artifacts (`loso_attribution_cpu.json`, `loso_hybrid_gat_cpu.json`,
`loso_directionality_cpu.json`).

| Fold | InDeg | Reach | Topo-QoS | GAT | GAT-P | GAT-QoS | GAT-P-QoS | Hybrid-GAT | Hybrid-GAT-P | HGT-QoS | HGT-P-QoS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ATM | 0.495 | 0.670 | 0.311 | 0.398 | 0.406 | 0.506 | 0.603 | 0.447 | 0.501 | 0.523 | 0.181 |
| AV System | 0.868 | 0.835 | 0.753 | 0.638 | 0.736 | 0.732 | 0.764 | 0.793 | 0.832 | 0.704 | 0.679 |
| Enterprise | 0.891 | 0.826 | 0.795 | 0.511 | 0.777 | 0.407 | 0.738 | 0.768 | 0.872 | 0.426 | 0.758 |
| Financial Trading | 0.874 | 0.803 | 0.586 | 0.721 | 0.767 | 0.713 | 0.800 | 0.797 | 0.848 | 0.695 | 0.734 |
| Healthcare | 0.831 | 0.819 | 0.369 | 0.716 | 0.753 | 0.798 | 0.863 | 0.686 | 0.823 | 0.730 | 0.783 |
| Enterprise Integration (ESB) | 0.529 | 0.612 | 0.430 | 0.500 | 0.519 | 0.630 | 0.758 | 0.568 | 0.525 | 0.548 | 0.464 |
| Industrial SCADA | 0.868 | 0.776 | 0.650 | 0.557 | 0.698 | 0.721 | 0.834 | 0.768 | 0.863 | 0.684 | 0.254 |
| IoT Smart City | 0.898 | 0.655 | 0.351 | 0.609 | 0.590 | 0.720 | 0.747 | 0.654 | 0.891 | 0.688 | 0.209 |
| Logistics Fleet | 0.836 | 0.759 | 0.741 | 0.575 | 0.622 | 0.654 | 0.774 | 0.806 | 0.827 | 0.771 | 0.546 |
| Microservices | 0.548 | 0.656 | 0.265 | 0.326 | 0.611 | 0.479 | 0.649 | 0.429 | 0.544 | 0.475 | 0.547 |
| Real-Time Gaming | 0.836 | 0.834 | 0.810 | 0.683 | 0.797 | 0.685 | 0.795 | 0.825 | 0.851 | 0.789 | 0.601 |
| Telecom RAN | 0.698 | 0.535 | 0.576 | 0.520 | 0.556 | 0.574 | 0.647 | 0.656 | 0.718 | 0.427 | 0.416 |
| **Mean** | **0.764** | **0.732** | **0.553** | **0.563** | **0.653** | **0.635** | **0.748** | **0.683** | **0.758** | **0.622** | **0.514** |
| Mean ρ, active stratum | 0.516 | 0.286 | 0.280 | 0.256 | 0.336 | 0.338 | 0.440 | 0.398 | 0.537 | 0.335 | 0.237 |

## Registered contrasts

Two-sided Wilcoxon over the 12 folds, fold-bootstrap 95% CI, Holm across the 12.

| Contrast | Δρ | 95% CI | Won | p | p_Holm |
|:---|---:|:---:|:---:|---:|---:|
| GAT-P vs InDeg | -0.111 | [-0.163, -0.059] | 1/12 | 0.0024 | 0.017 |
| GAT-P vs Reach | -0.079 | [-0.122, -0.046] | 1/12 | 0.0010 | 0.009 |
| GAT-P vs GAT | +0.090 | [+0.042, +0.148] | 11/12 | 0.0015 | 0.012 |
| GAT-P-QoS vs InDeg | -0.017 | [-0.072, +0.051] | 4/12 | 0.4697 | 1.000 |
| GAT-P-QoS vs Reach | +0.016 | [-0.024, +0.058] | 6/12 | 0.5693 | 1.000 |
| GAT-P-QoS vs GAT-QoS | +0.113 | [+0.077, +0.162] | 12/12 | 0.0005 | 0.006 |
| Hybrid-GAT-P vs InDeg | -0.006 | [-0.015, +0.002] | 3/12 | 0.2036 | 0.881 |
| Hybrid-GAT-P vs Reach | +0.026 | [-0.035, +0.086] | 8/12 | 0.3804 | 1.000 |
| Hybrid-GAT-P vs Hybrid-GAT | +0.075 | [+0.039, +0.113] | 11/12 | 0.0034 | 0.021 |
| HGT-P-QoS vs InDeg | -0.250 | [-0.377, -0.143] | 0/12 | 0.0005 | 0.006 |
| HGT-P-QoS vs Reach | -0.217 | [-0.317, -0.131] | 0/12 | 0.0005 | 0.006 |
| HGT-P-QoS vs HGT-QoS | -0.107 | [-0.238, +0.014] | 4/12 | 0.1763 | 0.881 |

Descriptive only, no correction:

| Comparison | Δρ | Won | p |
|:---|---:|:---:|---:|
| GAT-P vs Topo-QoS | +0.100 | 7/12 | 0.1099 |
| GAT-P-QoS vs Topo-QoS | +0.195 | 10/12 | 0.0068 |
| Hybrid-GAT-P vs Topo-QoS | +0.205 | 12/12 | 0.0005 |
| HGT-P-QoS vs Topo-QoS | -0.039 | 4/12 | 0.4697 |
| GAT-P vs GBM-Feat | +0.011 | 6/12 | 1.0000 |
| GAT-P-QoS vs GBM-Feat | +0.106 | 10/12 | 0.0093 |
| Hybrid-GAT-P vs GBM-Feat | +0.116 | 11/12 | 0.0015 |
| HGT-P-QoS vs GBM-Feat | -0.128 | 3/12 | 0.0923 |

## Zero-shot (five system models, descriptive)

| System | InDeg | Reach | GAT | GAT-P | GAT-QoS | GAT-P-QoS | Hybrid-GAT | Hybrid-GAT-P | HGT-QoS | HGT-P-QoS |
|---|---|---|---|---|---|---|---|---|---|---|
| Autoware.universe (ROS 2) | 0.620 | 0.836 | 0.778 | 0.821 | 0.758 | 0.794 | 0.576 | 0.682 | 0.716 | 0.657 |
| EdgeX Foundry | 0.896 | 0.997 | 0.853 | 0.846 | 0.815 | 0.841 | 0.748 | 0.813 | 0.793 | 0.786 |
| Home Assistant | 0.943 | 0.891 | 0.927 | 0.838 | 0.925 | 0.832 | 0.748 | 0.905 | 0.864 | 0.805 |
| Online Boutique (pub-sub model) | 0.988 | 0.998 | 0.810 | 0.829 | 0.750 | 0.790 | 0.595 | 0.811 | 0.710 | 0.758 |
| Train-Ticket | 0.867 | 0.966 | 0.786 | 0.815 | 0.777 | 0.774 | 0.642 | 0.752 | 0.717 | 0.724 |
| **Mean** | **0.863** | **0.938** | **0.831** | **0.830** | **0.805** | **0.806** | **0.662** | **0.792** | **0.760** | **0.746** |

The native arms' values come from their published zero-shot artifacts (`realworld_zeroshot_*_{attribution,cpu,directionality}.json`, corpus `3afa81f0`). `InDeg` and `Reach` are recomputed on the real-world cache labels and match `tf_baselines.json`.

## Decision rules, as registered

| Rule | Outcome |
|---|---|
| D1: some arm beats `InDeg` (Holm p < .05) | **Not triggered.** No arm beats it. |
| D2: no arm differs from `InDeg` | **Not triggered.** GAT-P and HGT-P-QoS are significantly *worse*. GAT-P-QoS (−0.017) and Hybrid-GAT-P (−0.006) are statistically indistinguishable from it. |
| D3: every arm's mean Δ vs `InDeg` is below 0 | **Triggered.** Learning on the dependency graph does not beat counting dependents. |
| M: a projection arm beats its native counterpart | **Triggered** for all three GAT arms: GAT-P +0.090 (11/12), GAT-P-QoS +0.113 (12/12), Hybrid-GAT-P +0.075 (11/12). The missing receptive field was a real cause of the native engines' deficit. |
| Z: best zero-shot arm < `Reach` (0.938) | **Triggered.** The best is GAT-P at 0.830. The deployment recommendation for unseen systems stays `Reach`. |

## What this means for the paper

1. **Learned engines should read the dependency graph.** Moving the same learners from the native
   multigraph to the `DEPENDS_ON` projection is worth +0.08 to +0.11 LOSO ρ. GAT-P-QoS (0.748)
   and Hybrid-GAT-P (0.758) are the best learned engines this project has produced, above
   Hybrid-GAT (0.683) and HGT-QoS (0.622).
   - Unlike on the native graph (Amendment 8, where GAT trailed GBM-Feat), message passing now
     adds something: GAT-P-QoS beats GBM-Feat by +0.106 (10/12, descriptive).
2. **They reach the counts but do not pass them.** The best learned arm on the dependency graph
   matches `InDeg` within noise (Δ −0.006, CI [−0.015, +0.002]) and never exceeds it.
   - Where the learners do add something is the active stratum, ranking components that do
     cause impact: Hybrid-GAT-P 0.537 and `InDeg` 0.516, against `Reach` 0.286 and Topo-QoS 0.280.
3. **Off the training distribution, counting wins.** On the five real system models `Reach`
   (0.938) and `InDeg` (0.863) beat every learned arm (≤ 0.830). The projection did not improve
   the GATs' zero-shot means: 0.830 vs 0.831 and 0.806 vs 0.805. This is a coincidence of means;
   the per-system values differ.
4. **Recommendation.** For the JSS claim "predicting cascading-failure impact before
   deployment", the dependency graph is the predictor, and the simplest reader of it is enough:
   - `InDeg` for LOSO-style ranking;
   - `Reach` for unseen real systems.

   If the paper keeps a learned engine, it should be GAT-P-QoS or Hybrid-GAT-P, not the native
   engines, and it should be presented as matching dependency counting rather than improving on
   it. Either way the paper must report `InDeg`/`Reach` (see the R3 flag below).

## Limitations

- **Folds are not independent.** The twelve folds share ten of their eleven training graphs, so
  p-values and CIs understate dispersion. This is the same caveat as every LOSO table in the
  project.
- **HGT-P-QoS did not train stably.** Its median within-fold seed SD is 0.208, against 0.026 for
  GAT-P-QoS and 0.056 for native HGT-QoS. On ATM, SCADA and IoT some seeds invert the ranking,
  so its low mean (0.514) reflects a training failure and says nothing about the substrate. It
  was run as registered (width 100, 3 layers, bidirectional); nothing was tuned afterwards.
- **The training population changed.** Broker, Topic and Node labels are absent from the
  projection, so the arms train on Applications and Libraries only.
- **The oracle shares the predictors' dependency semantics** (referee M2 of 2026-09-25).
  `I*(v)` is a damped reverse reachability over the same subscriber → publisher relation. That
  close fit is why counting dependents is hard to beat, and it limits what "beats the oracle's
  own counting" can mean.
- **A smoke run preceded the registered run.** It was a 2-epoch, 1-seed wiring check, run after
  Amendment 9 was committed. Nothing was changed on the basis of it.

## Open obligation (Amendment 7, R3)

Amendment 7's rule R3 says every one of its arms is reported in the manuscript or supplement. R1
was triggered: `InDeg` ≥ 0.622. The current manuscript (v4, restored 2026-09-26) reports neither
`InDeg` nor `Reach`. This page flags that obligation; it does not discharge it.

## Artifacts

All are committed in `results/`, like Amendment 7's:
- `loso_dependency_graph_cpu.json`
- `realworld_zeroshot_{gl_proj_cap,gl_proj_qos16_cap,gl_proj_qos16_indeg_prior,hgl_proj_qos}_dependency_graph.json`
- `receptive_field_probe_dependency_graph.json`
- `dependency_graph_contrasts.json`, which holds the per-fold rows, contrasts, zero-shot results,
  decisions and checks.

Checks recorded in `dependency_graph_contrasts.json`:
- `InDeg` and `Reach` recomputed on the LOSO cache's labels match `tf_baselines.json` exactly
  (max |Δ| = 0).
- `topo_qos` and `gl_full_qos16_cap`, re-run in the same invocation, reproduce their earlier
  artifacts bit for bit. The harness change did not touch the native path.
