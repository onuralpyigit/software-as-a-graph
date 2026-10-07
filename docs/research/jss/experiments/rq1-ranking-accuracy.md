# RQ1 — Ranking accuracy

**Paper:** §6.1 (`sec:rq1`); Table 5 (`tab:hybrid`), Table 6 (`tab:independent_oracles`), Figure 4
(`fig:results`), Figure 5 (`fig:recall`). Rankers: §5.2, Table 4 (`tab:predictor_taxonomy`).
**Supplement:** §S25 (`supp:hybrid-folds`, per-fold hybrids), §S28 (`supp:loso-active`, active
stratum), §S24 (`supp:ap-sensitivity`, articulation term restored), §S35 (`supp:loso-gpu`, the
registered GPU sweep), §S36 (`supp:regimes`, engine regimes), §S43 (`supp:baselines`, further
training-free baselines).
**Status:** the plan's two co-primary contrasts are confirmatory (both null). The hybrids
(Amendments 5–6) are registered secondary. Every dependency-graph learner, reference and
queue-flow result is registered secondary or exploratory, as marked below.
**Registration:** [`../PREREGISTRATION.md`](../PREREGISTRATION.md): the plan, Amendments 5 and 6;
the rows contributed by later amendments are listed under [Where each row comes from](#where-each-row-comes-from).

## Question

How accurately do analytical, hybrid and learned rankers order Applications by simulated cascade
impact on architectures they have never seen? The primary oracle is $I^*$. Table 6 adds the
queue-flow oracle $I_\text{dyn}$ and the multi-criteria oracle $I_\text{comp}$
([oracles-and-sensitivity.md](oracles-and-sensitivity.md)).

## Protocol

- **Folds.** Twelve synthetic scenarios, leave-one-scenario-out (LOSO). Each fold trains on eleven
  and scores the twelfth zero-shot.
- **Evaluation set.** Applications only, 26–300 per fold (1,321 in all), with
  $K = \mathrm{round}(0.2\,|V_\text{app}|)$, so $K$ runs from 5 to 60.
- **Seeds.** $\{42, 123, 456, 789, 2024\}$. The fold score is the mean of per-seed ρ. Table 6's GNN
  rows instead score the mean of the five seeds' predictions (a seed ensemble), so they read higher
  than Table 5 (`GAT-P-QoS` 0.772 vs 0.748).
- **Learned arms.** One fixed configuration, not tuned on any evaluation split: 3 layers, inner-split
  early stopping (patience 30, up to 300 epochs), AdamW (lr $3\times10^{-4}$, weight decay
  $10^{-4}$), cosine warm restarts ($T_0 = 75$, $T_\text{mult} = 2$, $\eta_\text{min} =
  3\times10^{-6}$), loss MSE + 0.5·aux MSE + 0.3·ListMLE + 0.1·pairwise margin ($\gamma = 0.05$).
  HGT uses $D = 64$, 4 heads, dropout 0.10. The plan's nested selection rule was not applied to any
  published learned result (deviation record; applied later as Amendment 14, arm N).
- **Training-free baseline.** `Topo-QoS` $= 0.6\cdot\text{BT}_w + 0.4\cdot\text{AP}$ on the
  Application–Library `DEPENDS_ON` graph (Rules 1 and 5). The AP term reads zero because of an
  implementation defect (the scorer looks up `ap_c_score`, which the cached metrics lack), so every
  table reports QoS-weighted betweenness alone. Restoring it lowers the baseline from 0.553 to 0.533.
  The registered contrasts stay against the registered value.
- **Statistics.** Paired two-sided Wilcoxon over folds and a fold-bootstrap 95% CI ($B = 2{,}000$).
  The registered confirmatory family is `HGT-QoS` vs `Topo-QoS` and `HGT` vs `Topo-QoS`,
  Holm-corrected across the two. Folds share 10 of 11 training scenarios, so every $p$ is nominal;
  the Nadeau–Bengio corrected $t$ is reported beside the hybrid contrasts.

## Hybrids (Amendments 5 and 6)

A hybrid takes a learned ranker and adds one input feature per Application and Library: the
`Topo-QoS` score, rank-normalized within the graph. It also adds a residual output,
$\hat I^*(v) = \sigma(z(v) + \alpha\,\mathrm{logit}(p(v)))$, with one learnable $\alpha$.
Everything else matches the base learner.

| Hybrid (variant id) | Base learner | Extra parameters | Registered |
|---|---|---|---|
| Hybrid-HGT (`hgl_qos_prior`) | `HGT-QoS` (`hgl_qos`) | 321 | Amendment 5, before any hybrid result |
| Hybrid-GAT (`gl_qos16_prior`) | `GAT-QoS` (`gl_full_qos16_cap`) | 1,441 | Amendment 6, before its result but after Amendment 2's |

Each registration fixes two contrasts: against `Topo-QoS` and against its own base learner. Amendment
6 adds a transfer criterion: the GAT hybrid would replace Hybrid-HGT as the recommendation only if it
also transferred at least as well. It does not (0.662 < 0.695).

## Reproduce

```bash
make -f reproduce/Makefile cache
make -f reproduce/Makefile rq-hybrid             # Topo-QoS, HGT-QoS, Hybrid-HGT (one CPU sweep)
make -f reproduce/Makefile rq-hybrid-gat         # GAT-QoS, Hybrid-GAT (one CPU sweep)
make -f reproduce/Makefile rq-dependency-graph   # GAT-P-QoS, HGT-P-QoS (Amendment 9)
PYTHONPATH=. python reproduce/training_free_suite.py all   # InDeg, Reach (Amendment 7)
make -f reproduce/Makefile rq-referee-round8     # Table 6: full-population I_dyn, partial ρ
make -f reproduce/Makefile rq-oracle-robust      # Table 6: GBM-P-QoS→dyn (Amendment 11; ~13 CPU-h of labels)
make -f reproduce/Makefile rq-rate-expansion     # Table 6: rate-weighted reference (Amendment 15)
make -f reproduce/Makefile omnibus               # all thirteen registered contrasts under one Holm
make -f reproduce/Makefile table4                # the registered GPU sweep (Supplement §S35)
```

All main-text learned rows come from CPU sweeps (`--device cpu --torch-threads 1`) with their
comparators in the same invocation. They are never mixed with the GPU rows of §S35: `HGT-QoS` is
0.622 on CPU and 0.638 on GPU. See [repeatability-and-amendments.md](repeatability-and-amendments.md).

## Where each row comes from

`reproduce/reconcile_manuscript.py` checks each table against these artifacts.

| Table rows | Artifact | Amendment page |
|---|---|---|
| Table 5: `Topo-QoS`, `HGT-QoS`, Hybrid-HGT | `loso_hybrid_cpu.json`, `loso_significance_hybrid_cpu.json` | this page |
| Table 5: `GAT-QoS`, Hybrid-GAT | `loso_hybrid_gat_cpu.json`, `loso_significance_hybrid_gat_cpu.json` | this page |
| Table 5: `GAT-P-QoS`, `HGT-P-QoS` | `loso_dependency_graph_cpu.json` | [A9](amendments/a09-dependency-graph-learning.md) |
| Table 5: `InDeg`, `Reach` (references) | `tf_baselines.json` | [A7](amendments/a07-training-free.md), [A13](amendments/a13-reference-demotion.md) |
| Table 5: `Topo-QoS` corrected | `data/benchmarks/topo_ap_sensitivity.json` | [A16](amendments/a16-direction-control.md), [A17](amendments/a17-round12.md) |
| Table 6: every ranker on `I*`, `I_dyn`, `I_comp`, partial ρ | `data/benchmarks/referee_round8_table7.json` | [A12](amendments/a12-referee-round7.md), [A14](amendments/a14-round8.md) |
| Table 6: `GBM-P-QoS→dyn` | `data/benchmarks/oracle_robust_ltr.json` | [A11](amendments/a11-oracle-robust.md) |
| Table 6: `GAT-P-QoS→dyn` | `data/benchmarks/referee_round8_amendment14.json` | [A14](amendments/a14-round8.md) |
| Table 6: rate-weighted reference (Eq. 7) | `data/benchmarks/idyn_rate_expansion.json` | [A15](amendments/a15-rate-expansion.md) |
| Figure 4 | rendered by `reproduce/render_headline_figure.py` from the Table 5 artifacts, the zero-shot artifacts and `loso_rq2_matched.json` | — |
| Figure 5 (recall) | `data/benchmarks/referee_round7_recall.json` ($I^*$), `referee_round10_recall_idyn_full.json` ($I_\text{dyn}$); `reproduce/render_recall_figure.py` | [A12](amendments/a12-referee-round7.md) |

## Headline result

- **Registered primary: null.** `HGT-QoS` vs `Topo-QoS` is +0.085 (9/12 folds, Holm p = 0.303) in
  the registered sweep (§S35), and `HGT` vs `Topo-QoS` is null as well.
- **Hybrids beat the baseline, not their base learners.** Hybrid-HGT +0.103 and Hybrid-GAT +0.130
  over `Topo-QoS` (11/12 folds; family Holm p = 0.0068 and 0.0029; omnibus p = 0.041 and 0.019).
  Neither differs from its base learner (+0.035, p = 0.73; +0.048, p = 0.30), and the same holds with
  the corrected prior (A16, F9). The hybrid gain belongs to the comparator.
- **The dependency graph helps learners.** `GAT-P-QoS` reaches 0.748, against 0.635 for the same
  model on the raw multigraph (A9) and 0.676 for a reverse-edge raw-graph control (A16, +0.072).
- **No learner exceeds the references.** `InDeg` 0.764, `Reach` 0.732 and the first-order expansion
  (Analytic $I^*$) 0.808 restate $I^*$'s rule, so they carry no contrast. `GAT-P-QoS` is not
  significantly different from `InDeg`, and equivalence within ±0.05 is not established (TOST
  p = 0.17).
- **Queue-flow oracle.** The rate-weighted reference (Eq. 7) reaches 0.830 on $I_\text{dyn}$ without
  training. That exceeds the learned approximation `GBM-P-QoS→dyn` (0.799; +0.031, 10/12, A15).
  Learners started from Eq. 7 do not improve on it (A17, F12).
- **No ranker wins every oracle.** On $I_\text{comp}$, raw total degree (0.719) and `Topo-QoS`
  (0.702) rank highest.

## Notes not in the paper

- **Label-noise ceiling of $I^*$.** Re-running the oracle across seeds gives a test–retest ρ of
  0.811–1.000. Top-K sets are noisier (cross-seed Jaccard down to 0.370 on Logistics Fleet), which is
  why Overlap@K margins are less stable than ρ margins. Artifact: `label_stability.json`
  (`reproduce/label_stability_check.py`).
- **In-distribution results** (60/20/20 node splits) are in §S19 (`supp:indist-table`) and §S12.
  They are not compared across model families, because the small GATs read the dependency graph
  while HGT reads the raw multigraph.
- **Engine regimes.** `make -f reproduce/Makefile jss-regimes` (`reproduce/engine_regimes.py`,
  artifact `engine_regimes.json`) correlates per-fold gains with fold descriptors. It is post hoc
  and trains nothing. Across 96 descriptor–gain correlations, none survives Benjamini–Hochberg
  correction (smallest q = 0.32), so each row of §S36 is a hypothesis, not a finding. The full table
  is `tab:regimes` (Table S34).
